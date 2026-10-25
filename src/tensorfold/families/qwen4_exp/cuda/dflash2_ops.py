"""DFlash2 FP16 operators with FP32 accumulation, without Triton/BF16 dispatch.

The convolution and selector equations follow SGLang 3c560089. Native target
operators and target positional transforms are not changed by these functions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def linear(x, weight, *, fp32=False):
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("DFlash2 linear needs matching [rows,input]/[output,input]")
    if x.dtype != torch.float16 or weight.dtype != torch.float16 or x.device != weight.device:
        raise ValueError("DFlash2 linear operands must be colocated FP16")
    value = (torch.mm(x.contiguous(), weight.T, out_dtype=torch.float32) if x.is_cuda
             else x.float() @ weight.float().T)
    return value if fp32 else value.half()


def norm(x, weight, eps):
    value = x.float()
    return (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps) * weight.float()).half()


def rope(x, positions, theta):
    """Bounded absolute-position NeoX RoPE; no full-context trigonometric table."""
    if x.shape[0] != positions.numel() or x.shape[-1] % 2:
        raise ValueError("DFlash2 RoPE rows/positions or rotary width differ")
    half = x.shape[-1] // 2
    inv = theta ** (-torch.arange(half, dtype=torch.float32, device=x.device) * (2.0 / x.shape[-1]))
    phase = positions.float()[:, None] * inv[None]
    shape = (len(positions), *([1] * (x.ndim - 2)), half)
    cos, sin = phase.cos().view(shape), phase.sin().view(shape)
    first, last = x[..., :half].float(), x[..., half:].float()
    return torch.cat((first * cos - last * sin, last * cos + first * sin), -1).half()


def grouped_conv(x, coefficients, base, *, side, block, group_size):
    """Two block-local taps. The previous row never crosses a block boundary."""
    rows, hidden = x.shape
    groups = hidden // group_size
    if rows % block or coefficients.shape != (rows, 2, groups) or base.shape != (2, 2, hidden):
        raise ValueError("DFlash2 grouped convolution does not match its complete block geometry")
    blocks = x.float().view(rows, groups, group_size)
    kernels = base[side].float().view(1, 2, groups, group_size) + coefficients.float().unsqueeze(-1)
    previous = torch.zeros_like(blocks)
    previous[1:].copy_(blocks[:-1])
    previous.view(-1, block, groups, group_size)[:, 0].zero_()
    return (kernels[:, 0] * blocks + kernels[:, 1] * previous).flatten(1).half()


def conv_prepare(x, projection, base, config):
    coeff = linear(x, projection).reshape(len(x), 2, 2, config.hidden // config.group_size)
    return grouped_conv(x, coeff[:, 0], base, side=0, block=config.block, group_size=config.group_size), coeff[:, 1]


def tp_sum(partial, comm):
    """FP32 partial sums in logical-rank order, one exchange for the batch."""
    if comm is None:
        return partial.half()
    partial = partial.contiguous()
    joined = torch.empty((comm.world, *partial.shape), dtype=torch.float32, device=partial.device)
    comm.all_gather(partial, joined.flatten())
    result = joined[0].clone()
    for rank in range(1, comm.world):
        result.add_(joined[rank])
    return result.half()


def window_mask(query_positions, context_positions, *, window):
    # SGLang converts HF's window including current row to window_left=window-1.
    # The complete block is non-causal; the window bound applies to its past.
    past = context_positions[None] >= query_positions[:, None] - (window - 1)
    block = torch.ones((len(query_positions), len(query_positions)), dtype=torch.bool, device=query_positions.device)
    return torch.cat((past, block), -1)


def attention(q, k, v, past_k, past_v, positions, past_positions, *, window):
    if q.ndim != 3 or k.shape != v.shape or past_k.shape != past_v.shape:
        raise ValueError("DFlash2 attention requires [rows,heads,dim] KV pairs")
    heads, kv_heads, dim = q.shape[1], k.shape[1], q.shape[2]
    keys = torch.cat((past_k, k), 0).float()
    values = torch.cat((past_v, v), 0).float()
    queries = q.float().reshape(len(q), kv_heads, heads // kv_heads, dim)
    scores = torch.einsum("qkgd,tkd->kgqt", queries, keys) * (dim ** -0.5)
    allowed = window_mask(positions, past_positions, window=window)
    scores.masked_fill_(~allowed[None, None], -float("inf"))
    probs = torch.softmax(scores, -1)
    return torch.einsum("kgqt,tkd->qkgd", probs, values).reshape(len(q), heads * dim).half()


def mlp(x, gate, up, down, comm):
    gated = (F.silu(linear(x, gate).float()) * linear(x, up).float()).half()
    return tp_sum(linear(gated, down, fp32=True), comm)


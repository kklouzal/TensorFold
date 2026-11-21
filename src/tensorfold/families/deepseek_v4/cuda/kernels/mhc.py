# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
#   vllm/model_executor/kernels/mhc/deepseek_v41.py (blob 913f2334e78c41d345b4d368a3369e11730152f7)
# TensorFold changes (arithmetic verbatim):
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - vllm.triton_utils -> triton_utils; vllm.envs VLLM_V41_COL_PROJ/COL_B/COL_BLOCK/COL_W -> production
#   default VLLM_V41_COL_PROJ=0, so the column-projection kernels (_col_projection, _col_projection2) and
#   their dispatch are unreachable and removed; the bf16 widening condition reduces to ``normalized``.
# - Triton/torch-fallback selection uses on_triton_device() (== tensor.is_cuda in compiled runs; also CPU
#   tensors under TRITON_INTERPRET=1 so interpreter tests exercise the kernels).
# - Triton 3.2 rewrites: _row_projection reads BF16 coefficients as int16 bit patterns widened by a 16-bit
#   shift (BF16 -> FP32 is exactly that); Triton 3.2 emits cvt.f32.bf16, which ptxas rejects below sm_80.
"""V4.1 delayed pre-mix primitives, with FP32 HC residual storage.

The next sublayer consumes the returned pre-mix. It must not be folded into
the current sublayer as in V4. Matrix coefficients, reductions and routing
remain FP32, including the HC stream's wide range; linear sublayers use FP16.
"""

import torch
import torch.nn.functional as F

from .triton_utils import jit, on_triton_device, tl, triton


@jit
def _row_projection(
    x,
    weight,
    out,
    K: tl.constexpr,
    N: tl.constexpr,
    NORMALIZE: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
    BF16_BITS: tl.constexpr = False,
):
    row, col = tl.program_id(0), tl.program_id(1)
    index = tl.arange(0, BLOCK)
    dot, squares = tl.full((), 0, tl.float32), tl.full((), 0, tl.float32)
    # Each row has identical FP32 reduction boundaries for M1 and M6, so
    # speculative verification cannot change a route just by changing M.
    for first in range(triton.cdiv(K, BLOCK)):
        k = first * BLOCK + index
        a = tl.load(x + row * K + k, k < K, 0).to(tl.float32)
        # BF16 storage of BF16-exact coefficients widens to the same FP32.
        if BF16_BITS:
            b = (tl.load(weight + col * K + k, k < K, 0).to(tl.int32) << 16).to(tl.float32, bitcast=True)
        else:
            b = tl.load(weight + col * K + k, k < K, 0).to(tl.float32)
        dot += tl.sum(a * b, 0)
        if NORMALIZE:
            squares += tl.sum(a * a, 0)
    if NORMALIZE:
        dot *= tl.rsqrt(squares / K + EPS)
    tl.store(out + row * N + col, dot)


@jit
def _hc_projection_parts(
    x, weight, dots, squares, M: tl.constexpr, SPLITS: tl.constexpr
):
    row, col, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    index = tl.arange(0, 1024)
    per_split: tl.constexpr = triton.cdiv(20, SPLITS)
    for item in range(per_split):
        group = split * per_split + item
        at = group * 1024 + index
        a = tl.load(x + row * 20480 + at, at < 20480, 0).to(tl.float32)
        b = tl.load(weight + col * 20480 + at, at < 20480, 0)
        if group < 20:
            offset = (group * M + row) * 24 + col
            tl.store(dots + offset, tl.sum(a * b, 0))
            tl.store(squares + offset, tl.sum(a * a, 0))


@jit
def _hc_projection_finish(
    dots, squares, out, ELEMENTS: tl.constexpr, EPS: tl.constexpr
):
    at = tl.program_id(0) * 128 + tl.arange(0, 128)
    dot, square = tl.full((128,), 0, tl.float32), tl.full((128,), 0, tl.float32)
    # Keep all twenty original 1024-element reduction results separately.
    # Combining them per split would change FP32 addition order and routes.
    for group in range(20):
        dot += tl.load(dots + group * ELEMENTS + at, at < ELEMENTS, 0)
        square += tl.load(squares + group * ELEMENTS + at, at < ELEMENTS, 0)
    dot *= tl.rsqrt(square / 20480 + EPS)
    tl.store(out + at, dot, at < ELEMENTS)


def fp32_projection(x, weight, *, normalized=False, eps=1e-20):
    """FP32 gate/mHC projection; fixed per-row reductions for M1..6."""
    if x.ndim != 2 or weight.ndim != 2 or x.shape[1] != weight.shape[1]:
        raise ValueError("V4.1 FP32 projection shapes do not match")
    if (
        weight.dtype not in (torch.float32, torch.bfloat16)
        or not x.is_contiguous()
        or not weight.is_contiguous()
    ):
        raise ValueError("V4.1 coefficients must be contiguous FP32")
    if x.device != weight.device:
        raise ValueError("V4.1 activations and coefficients must share a device")
    if on_triton_device(x) and x.shape[0] <= 6:
        result = torch.empty(
            x.shape[0], weight.shape[0], dtype=torch.float32, device=x.device
        )
        if weight.dtype == torch.bfloat16 and normalized:
            weight = weight.float()
        if normalized and weight.shape == (24, 20480) and x.shape[0] > 0:
            dots = torch.empty(20, x.shape[0], 24, dtype=torch.float32, device=x.device)
            squares = torch.empty_like(dots)
            _hc_projection_parts[(x.shape[0], 24, 8)](
                x,
                weight,
                dots,
                squares,
                x.shape[0],
                8,
                num_warps=4,
                enable_fp_fusion=False,
            )
            _hc_projection_finish[(triton.cdiv(result.numel(), 128),)](
                dots,
                squares,
                result,
                result.numel(),
                eps,
                num_warps=4,
                enable_fp_fusion=False,
            )
            return result
        bits = weight.dtype == torch.bfloat16
        _row_projection[(x.shape[0], weight.shape[0])](
            x,
            weight.view(torch.int16) if bits else weight,
            result,
            K=x.shape[1],
            N=weight.shape[0],
            NORMALIZE=normalized,
            EPS=eps,
            BLOCK=1024,
            BF16_BITS=bits,
            enable_fp_fusion=False,
        )
        return result
    result = F.linear(x.float(), weight.float())
    if normalized:
        # Row windows keep the FP32 square temporary bounded for long chunks.
        for a in range(0, x.shape[0], 2048):
            rows = x[a : a + 2048].float()
            result[a : a + 2048] *= torch.rsqrt(
                rows.square().mean(-1, keepdim=True) + eps
            )
    return result


@jit
def _split_sinkhorn(
    mixes,
    scales,
    base,
    pre,
    post,
    comb,
    EPS: tl.constexpr,
    ITERS: tl.constexpr,
):
    row = tl.program_id(0)
    i = tl.arange(0, 4)
    p = tl.load(mixes + row * 24 + i) * tl.load(scales) + tl.load(base + i)
    q = tl.load(mixes + row * 24 + 4 + i) * tl.load(scales + 1) + tl.load(base + 4 + i)
    tl.store(pre + row * 4 + i, tl.sigmoid(p) + EPS)
    tl.store(post + row * 4 + i, 2.0 * tl.sigmoid(q))
    location = i[:, None] * 4 + i[None, :]
    value = tl.load(mixes + row * 24 + 8 + location) * tl.load(scales + 2) + tl.load(
        base + 8 + location
    )
    value = tl.exp(value - tl.max(value, 1)[:, None])
    value = value / tl.sum(value, 1)[:, None] + EPS
    value = value / (tl.sum(value, 0)[None, :] + EPS)
    for _ in range(ITERS - 1):
        value = value / (tl.sum(value, 1)[:, None] + EPS)
        value = value / (tl.sum(value, 0)[None, :] + EPS)
    tl.store(comb + row * 16 + location, value)


def hc_mixes(residual, fn, scale, base, *, norm_eps=1e-20, hc_eps=1e-6, iters=20):
    if (
        residual.ndim != 3
        or residual.shape[1] != 4
        or residual.dtype not in (torch.float16, torch.float32)
        or not residual.is_contiguous()
        or fn.shape != (24, 4 * residual.shape[2])
        or scale.shape != (3,)
        or base.shape != (24,)
        or scale.dtype != torch.float32
        or base.dtype != torch.float32
        or iters < 1
    ):
        raise ValueError(
            "V4.1 mHC requires four floating streams and FP32 coefficients"
        )
    mixes = fp32_projection(residual.flatten(1), fn, normalized=True, eps=norm_eps)
    pre = torch.empty(residual.shape[0], 4, device=residual.device, dtype=torch.float32)
    post, comb = (
        torch.empty_like(pre),
        torch.empty(
            residual.shape[0], 4, 4, device=residual.device, dtype=torch.float32
        ),
    )
    if on_triton_device(residual):
        _split_sinkhorn[(residual.shape[0],)](
            mixes,
            scale,
            base,
            pre,
            post,
            comb,
            EPS=hc_eps,
            ITERS=iters,
            enable_fp_fusion=False,
        )
    else:
        pre = torch.sigmoid(mixes[:, :4] * scale[0] + base[:4]) + hc_eps
        post = 2 * torch.sigmoid(mixes[:, 4:8] * scale[1] + base[4:8])
        comb = (mixes[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4).softmax(-1) + hc_eps
        comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
        for _ in range(iters - 1):
            comb = comb / (comb.sum(-1, keepdim=True) + hc_eps)
            comb = comb / (comb.sum(-2, keepdim=True) + hc_eps)
    return pre, post, comb


@jit
def _collapse(
    residual,
    pre,
    norm,
    out,
    H: tl.constexpr,
    NORMALIZE: tl.constexpr,
    HALF_BOUNDARY: tl.constexpr,
    EPS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for stream in tl.static_range(4):
        x = tl.load(residual + (row * 4 + stream) * H + col, col < H, 0).to(tl.float32)
        value += x * tl.load(pre + row * 4 + stream)
    # Preserve a caller's FP16 boundary. The model carries HC in FP32 because
    # valid tokens can exceed FP16 range before the following RMSNorm.
    if HALF_BOUNDARY:
        value = value.to(tl.float16).to(tl.float32)
    if NORMALIZE:
        variance = tl.sum(tl.where(col < H, value * value, 0), 0) / H
        value = value * tl.rsqrt(variance + EPS) * tl.load(norm + col, col < H, 0)
    tl.store(out + row * H + col, value, col < H)


def collapse_hc(residual, pre, norm_weight=None, eps=1e-20):
    tokens, hc, hidden = residual.shape
    if hc != 4 or pre.shape != (tokens, 4) or pre.dtype != torch.float32:
        raise ValueError("V4.1 collapse requires four FP32 pre-mix values")
    if (
        residual.dtype not in (torch.float16, torch.float32)
        or not residual.is_contiguous()
    ):
        raise ValueError("V4.1 collapse requires contiguous floating residuals")
    if norm_weight is not None and (
        norm_weight.shape != (hidden,) or norm_weight.dtype != torch.float32
    ):
        raise ValueError("V4.1 normalization weights must retain FP32")
    if not on_triton_device(residual):
        value = (residual.float() * pre[..., None]).sum(1).to(residual.dtype)
        if norm_weight is not None:
            f = value.float()
            value = (
                f * torch.rsqrt(f.square().mean(-1, keepdim=True) + eps) * norm_weight
            ).half()
        return value
    result = torch.empty(
        tokens,
        hidden,
        device=residual.device,
        dtype=torch.float16 if norm_weight is not None else residual.dtype,
    )
    _collapse[(tokens,)](
        residual,
        pre,
        norm_weight if norm_weight is not None else pre,
        result,
        H=hidden,
        NORMALIZE=norm_weight is not None,
        HALF_BOUNDARY=residual.dtype == torch.float16,
        EPS=eps,
        BLOCK=triton.next_power_of_2(hidden),
        num_warps=8,
        enable_fp_fusion=False,
    )
    return result


@jit
def _expand(x, residual, post, comb, out, H: tl.constexpr, BLOCK: tl.constexpr):
    row, output = tl.program_id(0), tl.program_id(1)
    col = tl.program_id(2) * BLOCK + tl.arange(0, BLOCK)
    value = tl.full((BLOCK,), 0, tl.float32)
    for source in tl.static_range(4):
        value += tl.load(comb + row * 16 + source * 4 + output) * tl.load(
            residual + (row * 4 + source) * H + col, col < H, 0
        ).to(tl.float32)
    value += tl.load(post + row * 4 + output) * tl.load(
        x + row * H + col, col < H, 0
    ).to(tl.float32)
    tl.store(out + (row * 4 + output) * H + col, value, col < H)


@jit
def _expand_stream(r0, r1, r2, r3, xv, comb, post, row, output):
    # Same accumulation order as _expand: sources 0..3, then post * x.
    value = tl.load(comb + row * 16 + output) * r0
    value += tl.load(comb + row * 16 + 4 + output) * r1
    value += tl.load(comb + row * 16 + 8 + output) * r2
    value += tl.load(comb + row * 16 + 12 + output) * r3
    value += tl.load(post + row * 4 + output) * xv
    return value


@jit
def _expand_inplace(x, residual, post, comb, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = col < H
    base = residual + row * 4 * H + col
    r0 = tl.load(base, mask, 0).to(tl.float32)
    r1 = tl.load(base + H, mask, 0).to(tl.float32)
    r2 = tl.load(base + 2 * H, mask, 0).to(tl.float32)
    r3 = tl.load(base + 3 * H, mask, 0).to(tl.float32)
    xv = tl.load(x + row * H + col, mask, 0).to(tl.float32)
    for output in tl.static_range(4):
        value = _expand_stream(r0, r1, r2, r3, xv, comb, post, row, output)
        tl.store(base + output * H, value, mask)


def expand_hc(x, residual, post, comb, *, inplace=False):
    tokens, hc, hidden = residual.shape
    if (
        hc != 4
        or x.shape != (tokens, hidden)
        or x.dtype not in (torch.float16, torch.float32)
        or residual.dtype not in (torch.float16, torch.float32)
        or post.shape != (tokens, 4)
        or comb.shape != (tokens, 4, 4)
        or post.dtype != torch.float32
        or comb.dtype != torch.float32
    ):
        raise ValueError("V4.1 post-mix shapes or dtypes do not match")
    if not on_triton_device(x):
        return (
            torch.einsum("tij,tih->tjh", comb, residual.float())
            + post[..., None] * x[:, None].float()
        ).to(residual.dtype)
    if inplace and residual.is_contiguous():
        _expand_inplace[(tokens, triton.cdiv(hidden, 256))](
            x, residual, post, comb, H=hidden, BLOCK=256, enable_fp_fusion=False
        )
        return residual
    result = torch.empty_like(residual)
    _expand[(tokens, 4, triton.cdiv(hidden, 256))](
        x,
        residual,
        post,
        comb,
        result,
        H=hidden,
        BLOCK=256,
        enable_fp_fusion=False,
    )
    return result


def route_experts(x, weight, bias, *, top_k=6, temperature=1.0, scale=1.5):
    if bias.dtype != torch.float32 or bias.shape != (weight.shape[0],):
        raise ValueError("V4.1 routing bias must retain FP32")
    scores = F.softplus(fp32_projection(x, weight) / temperature).sqrt()
    indices = (scores + bias).topk(top_k, dim=-1).indices
    probabilities = scores.gather(1, indices)
    probabilities = probabilities / (probabilities.sum(-1, keepdim=True) + 1e-20)
    return (probabilities * scale).contiguous(), indices

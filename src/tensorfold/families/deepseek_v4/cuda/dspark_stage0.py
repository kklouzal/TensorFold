"""FP32 arithmetic for the first DSpark stage, with the original FP16 cuts.

Each attention matrix multiply retains the five-token request shape. Widen
one projection bank at a time and share that temporary across the requests.
The head owns the tensor-parallel reduction and the native MoE.
"""

from __future__ import annotations

import torch

EPS = 1e-20


def _rms(x, weight):
    values = x.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + EPS) * weight).to(x.dtype)


def _linears(rows, weight, *, output_dtype=torch.float16):
    matrix = weight.float().t()
    return [(row.float() @ matrix).to(output_dtype) for row in rows]


def frequencies(device):
    return 1.0 / (10000.0 ** (torch.arange(0, 64, 2, dtype=torch.float32, device=device) / 64))


def _rope(x, cos, sin):
    shape = (x.shape[0],) + (1,) * (x.ndim - 2) + (32,)
    cos, sin = cos.view(shape), sin.view(shape)
    a, b = x[..., -64::2].float(), x[..., -63::2].float()
    out = x.clone()
    out[..., -64::2] = (a * cos - b * sin).half()
    out[..., -63::2] = (a * sin + b * cos).half()
    return out


def _qdq(x):
    values = x.float().view(*x.shape[:-1], -1, 128)
    amax = values.abs().amax(-1, keepdim=True).clamp_min(1e-4)
    mantissa, exponent = torch.frexp(amax / 448.0)
    scale = torch.ldexp(torch.ones_like(mantissa), torch.where(mantissa == 0.5, exponent - 1, exponent))
    codes = (values / scale).clamp(-448, 448).to(torch.float8_e4m3fn)
    return (codes.float() * scale).view(x.shape).half()


def mixes(residual, fn, scale, base, *, iters=20, eps=1e-6):
    flat = residual.flatten(1)
    mixed = (flat @ fn.t()) * torch.rsqrt(flat.square().mean(-1, keepdim=True) + EPS)
    pre = torch.sigmoid(mixed[:, :4] * scale[0] + base[:4]) + eps
    post = 2 * torch.sigmoid(mixed[:, 4:8] * scale[1] + base[4:8])
    comb = torch.softmax((mixed[:, 8:] * scale[2] + base[8:]).view(-1, 4, 4), -1)
    comb = comb + eps
    comb = comb / (comb.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(-1, keepdim=True) + eps)
        comb = comb / (comb.sum(-2, keepdim=True) + eps)
    return pre, post, comb


def collapse(residual, pre, weight):
    return _rms((residual * pre[..., None]).sum(1), weight).half()


def expand(x, residual, post, comb):
    return torch.einsum("tij,tih->tjh", comb, residual) + post[..., None] * x[:, None].float()


def _pad_axis(x, dim, size, offset):
    shape = list(x.shape)
    shape[dim] = size
    padded = x.new_zeros(shape)
    padded.narrow(dim, offset, x.shape[dim]).copy_(x)
    return padded


def _attention_rows(q, kv, window, live, sink, *, rank=None):
    # K and V are the same FP16 reconstruction. The sink participates in the
    # denominator and supplies no value row.
    keys = torch.cat((torch.where(live[:, None], window, torch.zeros_like(window)), kv)).float()
    valid = torch.cat((live, torch.ones(kv.shape[0], dtype=torch.bool, device=kv.device)))
    if rank is not None:
        # Stage-zero TP8 accuracy candidate: zero filler only; the independent
        # probe needed full-width QK, 16-head PV, and two wo_a groups.
        heads = q.shape[1]
        offset = rank * heads
        q = _pad_axis(q, 1, 64, offset)
    scores = (q.float() @ keys.t()) * 512 ** -0.5
    if rank is not None:
        scores = scores[:, offset:offset + heads].contiguous()
    scores = scores.masked_fill(~valid, float("-inf"))
    sink = sink.view(1, -1, 1)
    maximum = torch.maximum(scores.amax(-1, keepdim=True), sink)
    numerator = torch.exp(scores - maximum)
    probability = (numerator / (numerator.sum(-1, keepdim=True) + torch.exp(sink - maximum))).half()
    if rank is None:
        return (probability.float() @ keys).half()
    offset %= 16
    probability = _pad_axis(probability, 1, 16, offset)
    return (probability.float() @ keys).half()[:, offset:offset + heads].contiguous()


def _project_attention(values, weight, rank):
    if rank is not None:
        offset = rank % 2
        weight = _pad_axis(weight, 0, 2, offset)
    wo_a = weight.float()
    projected = []
    for out in values:
        if rank is not None:
            out = _pad_axis(out, 1, 2, offset)
        value = torch.einsum("ngk,gok->ngo", out.float(), wo_a).half()
        if rank is not None:
            value = value[:, offset:offset + 1]
        projected.append(value.reshape(out.shape[0], -1))
    return projected


def attention_partials(stage, inputs, positions, windows, live, freqs, *, rank=None):
    """Requestwise attention with FP32 output partials, optionally using the stage-zero TP8 candidate.

    A rank selects the tested TP8 stage-zero shapes, trading 8x QK, 2x PV and 2x wo_a matmul work
    for matching the standalone reference on the archived FT fixture. Later stages and world one
    keep their original shapes. Collectives and weight ownership are unchanged.
    """
    query_low = [_rms(q, stage.q_norm) for q in _linears(inputs, stage.wq_a)]
    query = _linears(query_low, stage.wq_b)
    kv = _linears(inputs, stage.wkv)
    groups = stage.wo_a.shape[0]
    values = []
    for q, k, pos, window, valid in zip(query, kv, positions, windows, live, strict=True):
        angles = pos.float()[:, None] * freqs[None]
        cos, sin = torch.cos(angles), torch.sin(angles)
        q = _rope(q.view(q.shape[0], stage.sink.numel(), 512), cos, sin)
        k = _qdq(_rope(_rms(k, stage.kv_norm), cos, sin))
        out = (_attention_rows(q, k, window, valid, stage.sink) if rank is None else
               _attention_rows(q, k, window, valid, stage.sink, rank=rank))
        values.append(_qdq(_rope(out, cos, -sin).reshape(out.shape[0], groups, -1)))

    projected = _project_attention(values, stage.wo_a, rank)
    return torch.cat(_linears(projected, stage.wo_b, output_dtype=torch.float32))

# SPDX-License-Identifier: Apache-2.0
"""TensorFold-owned DeepSeek-V4.1 decode-band kernels (SM70 Triton 3.2): the MoE step's small pieces in one launch
each, and the FP32 gate projection for 7..64 rows.

``route_local``: the vendored router epilogue (``router._route``, arithmetic copied verbatim: sqrt(softplus) scores,
biased top-k, renormalized) plus ``Target.moe``'s EP2 mapping in the same program: global expert id -> this rank's
local id (``id - first``), rows routed elsewhere -> local id 0 with weight 0.0. It replaces route_topk and nine
elementwise launches (int64 -> int32, subtract, two compares, and, two selects, contiguous, FP32 copy) with the
same bits: the probabilities are the same FP32 expression, the mapping is integer compare/select.

``add_halves``: FP32 ``a.float() + b.float()`` of two FP16 [rows, cols] tensors (the routed + shared expert sum
before the cross-rank sum): one launch for two converting copies and an add, the same FP32 add.

``project_rows``: the FP32 gate/mHC projection ``x @ w.T`` for decode bands (patch 89's band, up to
``BAND_ROWS`` rows): every output element is ``_row_projection``'s per-row reduction (1024-element FP32 blocks
summed in K order), so a row's result does not depend on the batch it is in (selection: tests/cuda/deepseek_v41
band bench).
"""

import torch

from .triton_utils import jit, tl, triton

BAND_ROWS = 64          # patch 89: row-wise decode kernels up to this many rows (VLLM_LOCAL_DS41_MU_INVARIANT_ROWS)


@jit
def _route_local(logits, bias, local, weights, first, count, E: tl.constexpr, K: tl.constexpr,
                 SCALE: tl.constexpr, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    e = tl.arange(0, BLOCK)
    valid = e < E
    g = tl.load(logits + t * E + e, valid, 0.0)
    # router._route verbatim: softplus as in torch (threshold 20), then sqrt.
    soft = tl.where(g > 20.0, g, tl.log(1.0 + tl.exp(tl.minimum(g, 20.0))))
    score = tl.sqrt(soft)
    key = tl.where(valid, score + tl.load(bias + e, valid, 0.0), float("-inf"))
    k = tl.arange(0, 8)
    chosen = tl.zeros((8,), tl.int32)
    values = tl.zeros((8,), tl.float32)
    for j in tl.static_range(K):
        best = tl.max(key, 0)
        index = tl.min(tl.where(key == best, e, BLOCK), 0)
        value = tl.sum(tl.where(e == index, score, 0.0), 0)
        chosen = tl.where(k == j, index, chosen)
        values = tl.where(k == j, value, values)
        key = tl.where(e == index, float("-inf"), key)
    total = tl.sum(tl.where(k < K, values, 0.0), 0)
    out = values / (total + 1e-20) * SCALE
    # Target.moe's EP mapping (torch ops before): local id, or 0 with weight 0 for another group's expert
    shifted = chosen - first
    mine = (shifted >= 0) & (shifted < count)
    tl.store(local + t * K + k, tl.where(mine, shifted, 0), k < K)
    tl.store(weights + t * K + k, tl.where(mine, out, 0.0), k < K)


def route_local(logits, bias, top_k, scale, first, count):
    """FP32 logits [T, E] -> (int32 local ids [T, K], FP32 weights [T, K]) of this rank's experts [first, +count)."""
    tokens, experts = logits.shape
    if logits.dtype != torch.float32 or bias.dtype != torch.float32 or top_k > 8 or not logits.is_contiguous():
        raise ValueError("V4.1 fused routing requires contiguous FP32 logits/bias, top_k<=8")
    local = torch.empty(tokens, top_k, dtype=torch.int32, device=logits.device)
    weights = torch.empty(tokens, top_k, dtype=torch.float32, device=logits.device)
    if tokens:
        _route_local[(tokens,)](logits, bias, local, weights, first, count, E=experts, K=top_k, SCALE=scale,
                                BLOCK=triton.next_power_of_2(experts), num_warps=4)
    return local, weights


@jit
def _add_halves(a, b, out, n, BLOCK: tl.constexpr):
    at = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = at < n
    x = tl.load(a + at, mask, 0.0).to(tl.float32)
    y = tl.load(b + at, mask, 0.0).to(tl.float32)
    tl.store(out + at, x + y, mask)


def add_halves(a, b):
    """FP32 a.float() + b.float() of two contiguous FP16 tensors of one shape."""
    if a.shape != b.shape or a.dtype != torch.float16 or b.dtype != torch.float16 or not (
            a.is_contiguous() and b.is_contiguous()):
        raise ValueError("add_halves requires two contiguous FP16 tensors of one shape")
    out = torch.empty(a.shape, dtype=torch.float32, device=a.device)
    n = a.numel()
    _add_halves[(triton.cdiv(n, 1024),)](a, b, out, n, BLOCK=1024, num_warps=4)
    return out


@jit
def _project_rows(x, weight, out, M, K: tl.constexpr, N: tl.constexpr, ROWS: tl.constexpr, BLOCK: tl.constexpr,
                  BF16_BITS: tl.constexpr):
    # One program: one output column for ROWS rows. The weight block is read once per row group (L1 serves the
    # repeats); each (row, column) keeps _row_projection's reduction: 1024-element blocks, tl.sum each, K order.
    group, col = tl.program_id(0), tl.program_id(1)
    index = tl.arange(0, BLOCK)
    for i in tl.static_range(ROWS):
        row = group * ROWS + i
        if row < M:
            dot = tl.full((), 0, tl.float32)
            for first in range(triton.cdiv(K, BLOCK)):
                k = first * BLOCK + index
                a = tl.load(x + row * K + k, k < K, 0).to(tl.float32)
                if BF16_BITS:
                    b = (tl.load(weight + col * K + k, k < K, 0).to(tl.int32) << 16).to(tl.float32, bitcast=True)
                else:
                    b = tl.load(weight + col * K + k, k < K, 0).to(tl.float32)
                dot += tl.sum(a * b, 0)
            tl.store(out + row * N + col, dot)


def project_rows(x, weight, rows_per_program=8):
    """FP32 [M, N] = x [M, K] (FP16/FP32) @ weight [N, K] (FP32 or BF16-exact) .T, M <= BAND_ROWS, per-row
    arithmetic of ``mhc._row_projection`` (unnormalized)."""
    m = x.shape[0]
    result = torch.empty(m, weight.shape[0], dtype=torch.float32, device=x.device)
    bits = weight.dtype == torch.bfloat16
    _project_rows[(triton.cdiv(m, rows_per_program), weight.shape[0])](
        x, weight.view(torch.int16) if bits else weight, result, m, K=x.shape[1], N=weight.shape[0],
        ROWS=rows_per_program, BLOCK=1024, BF16_BITS=bits, num_warps=4, enable_fp_fusion=False)
    return result

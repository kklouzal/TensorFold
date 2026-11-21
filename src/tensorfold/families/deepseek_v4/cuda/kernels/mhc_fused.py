# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
#   vllm/model_executor/kernels/mhc/deepseek_v41_fused.py (blob c38b06bf47b77240a20f65bcda1c5588e9631017)
# TensorFold changes (arithmetic verbatim):
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - vllm.triton_utils -> triton_utils. vllm.envs -> production defaults: VLLM_V41_HC_V3=1,
#   VLLM_V41_HC_V3_GROUPS=8, VLLM_V41_HC_WINDOW=2048, VLLM_V41_HC_V2=0, VLLM_V41_HC_CUDA=0,
#   VLLM_V41_HC_TOKEN_BLOCKED=0 (HC_COLS/PW/FW only configured V2).
# - Removed paths unreachable at those values: V2 (_transition_parts2/_transition_finish2/_hc_transition2),
#   the vLLM CUDA-extension path (_cuda_ready/_hc_transition_cuda) and the token-blocked tl.dot path
#   (_transition_parts_tok/_hc_transition_tok). hc_transition dispatch: V3 for 1..8 tokens, otherwise the
#   windowed per-token kernels, as upstream at these values; V3 up to band.BAND_ROWS (64) tokens (the friend's
#   patch 89 band: the V3 grid is per token, so a token's result does not depend on M).
# - Triton 3.2 rewrites: none.
"""Fused V4.1 HC transition: expand (in place) + next mixes + delayed collapse.

One sublayer boundary of DeepSeek-V4.1 is

    residual = expand(y, residual, post, comb)
    pre', post', comb' = sinkhorn(norm_proj(residual, fn_next))
    inputs = rms_norm(collapse(residual, pre_delayed)) * norm_weight

which previously took five launches and a fresh [T, 4, H] FP32 residual. Here
K1 updates the residual in place and emits fixed-order partial reductions; K2
finishes the mixes, Sinkhorn and the normalized FP16 collapse. Reduction order
is fixed per row, so M1..M6 graphs produce identical routes for a token.
"""

import torch

from .band import BAND_ROWS
from .triton_utils import jit, tl

H = 5120
COLS = 256
CHUNKS = H // COLS  # 20


@jit
def _transition_parts(
    residual,
    y,
    post,
    comb,
    fn,
    cpre,
    cbuf,
    dots,
    squares,
    csquares,
    MIX: tl.constexpr,
    H: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
):
    t, chunk = tl.program_id(0), tl.program_id(1)
    col = chunk * COLS + tl.arange(0, COLS)
    base = residual + t * 4 * H
    r0 = tl.load(base + 0 * H + col)
    r1 = tl.load(base + 1 * H + col)
    r2 = tl.load(base + 2 * H + col)
    r3 = tl.load(base + 3 * H + col)
    yv = tl.load(y + t * H + col).to(tl.float32)
    cb = comb + t * 16
    pb = post + t * 4
    # comb[source * 4 + output], same as the reference expand.
    n0 = (
        tl.load(cb + 0) * r0
        + tl.load(cb + 4) * r1
        + tl.load(cb + 8) * r2
        + tl.load(cb + 12) * r3
        + tl.load(pb + 0) * yv
    )
    n1 = (
        tl.load(cb + 1) * r0
        + tl.load(cb + 5) * r1
        + tl.load(cb + 9) * r2
        + tl.load(cb + 13) * r3
        + tl.load(pb + 1) * yv
    )
    n2 = (
        tl.load(cb + 2) * r0
        + tl.load(cb + 6) * r1
        + tl.load(cb + 10) * r2
        + tl.load(cb + 14) * r3
        + tl.load(pb + 2) * yv
    )
    n3 = (
        tl.load(cb + 3) * r0
        + tl.load(cb + 7) * r1
        + tl.load(cb + 11) * r2
        + tl.load(cb + 15) * r3
        + tl.load(pb + 3) * yv
    )
    tl.store(base + 0 * H + col, n0)
    tl.store(base + 1 * H + col, n1)
    tl.store(base + 2 * H + col, n2)
    tl.store(base + 3 * H + col, n3)
    pc = cpre + t * 4
    coll = (
        n0 * tl.load(pc + 0)
        + n1 * tl.load(pc + 1)
        + n2 * tl.load(pc + 2)
        + n3 * tl.load(pc + 3)
    )
    tl.store(cbuf + t * H + col, coll)
    tl.store(csquares + t * CHUNKS + chunk, tl.sum(coll * coll, 0))
    if MIX:
        tl.store(
            squares + t * CHUNKS + chunk,
            tl.sum(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3, 0),
        )
        for r in tl.static_range(24):
            w = fn + r * 4 * H
            d = (
                tl.load(w + 0 * H + col) * n0
                + tl.load(w + 1 * H + col) * n1
                + tl.load(w + 2 * H + col) * n2
                + tl.load(w + 3 * H + col) * n3
            )
            tl.store(dots + (t * CHUNKS + chunk) * 24 + r, tl.sum(d, 0))


@jit
def _transition_finish(
    dots,
    squares,
    csquares,
    cbuf,
    norm,
    out,
    scale,
    base,
    pre,
    post,
    comb,
    MIX: tl.constexpr,
    H: tl.constexpr,
    CHUNKS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    ITERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    total = tl.full((), 0, tl.float32)
    for c in range(CHUNKS):
        total += tl.load(csquares + t * CHUNKS + c)
    inv = tl.rsqrt(total / H + NORM_EPS)
    for first in range(0, H, BLOCK):
        col = first + tl.arange(0, BLOCK)
        v = tl.load(cbuf + t * H + col) * inv * tl.load(norm + col)
        tl.store(out + t * H + col, v.to(tl.float16))
    if MIX:
        i = tl.arange(0, 4)
        sq = tl.full((), 0, tl.float32)
        for c in range(CHUNKS):
            sq += tl.load(squares + t * CHUNKS + c)
        rs = tl.rsqrt(sq / (4 * H) + NORM_EPS)
        p = tl.zeros((4,), tl.float32)
        q = tl.zeros((4,), tl.float32)
        location = i[:, None] * 4 + i[None, :]
        m = tl.zeros((4, 4), tl.float32)
        for c in range(CHUNKS):
            row = dots + (t * CHUNKS + c) * 24
            p += tl.load(row + i)
            q += tl.load(row + 4 + i)
            m += tl.load(row + 8 + location)
        p = p * rs * tl.load(scale) + tl.load(base + i)
        q = q * rs * tl.load(scale + 1) + tl.load(base + 4 + i)
        tl.store(pre + t * 4 + i, tl.sigmoid(p) + HC_EPS)
        tl.store(post + t * 4 + i, 2.0 * tl.sigmoid(q))
        value = m * rs * tl.load(scale + 2) + tl.load(base + 8 + location)
        value = tl.exp(value - tl.max(value, 1)[:, None])
        value = value / tl.sum(value, 1)[:, None] + HC_EPS
        value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
        for _ in range(ITERS - 1):
            value = value / (tl.sum(value, 1)[:, None] + HC_EPS)
            value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
        tl.store(comb + t * 16 + location, value)


_WINDOW = 2048


def hc_transition(
    residual,
    y,
    post,
    comb,
    collapse_pre,
    norm_weight,
    fn=None,
    scale=None,
    base=None,
    *,
    norm_eps=1e-20,
    hc_eps=1e-6,
    iters=20,
):
    """In-place expand; returns (inputs_fp16, (pre, post, comb) or None)."""
    tokens = residual.shape[0]
    if (
        residual.shape != (tokens, 4, H)
        or residual.dtype != torch.float32
        or not residual.is_contiguous()
        or y.shape != (tokens, H)
        or not y.is_contiguous()
    ):
        raise ValueError("V4.1 fused HC transition requires FP32 [T, 4, 5120]")
    mix = fn is not None
    device = residual.device
    if 0 < tokens <= BAND_ROWS:
        return _hc_transition3(
            residual,
            y,
            post,
            comb,
            collapse_pre,
            norm_weight,
            fn,
            scale,
            base,
            norm_eps,
            hc_eps,
            iters,
        )
    # Per-token kernels: process bounded windows so the FP32 collapse buffer
    # does not scale with the prefill chunk.
    window = min(tokens, _WINDOW)
    cbuf = torch.empty(window, H, dtype=torch.float32, device=device)
    csq = torch.empty(window, CHUNKS, dtype=torch.float32, device=device)
    if mix:
        dots = torch.empty(window, CHUNKS, 24, dtype=torch.float32, device=device)
        sq = torch.empty(window, CHUNKS, dtype=torch.float32, device=device)
        pre = torch.empty(tokens, 4, dtype=torch.float32, device=device)
        post_n = torch.empty_like(pre)
        comb_n = torch.empty(tokens, 4, 4, dtype=torch.float32, device=device)
    else:
        dots = sq = csq
    out = torch.empty(tokens, H, dtype=torch.float16, device=device)
    for a in range(0, tokens, window):
        b = min(a + window, tokens)
        n = b - a
        if mix:
            pre_w, post_w, comb_w = pre[a:b], post_n[a:b], comb_n[a:b]
        else:
            pre_w = post_w = comb_w = csq
        _transition_parts[(n, CHUNKS)](
            residual[a:b],
            y[a:b],
            post[a:b],
            comb[a:b],
            fn if mix else post,
            collapse_pre[a:b],
            cbuf,
            dots,
            sq,
            csq,
            MIX=mix,
            H=H,
            COLS=COLS,
            CHUNKS=CHUNKS,
            num_warps=4,
            enable_fp_fusion=False,
        )
        _transition_finish[(n,)](
            dots,
            sq,
            csq,
            cbuf,
            norm_weight,
            out[a:b],
            scale if mix else csq,
            base if mix else csq,
            pre_w,
            post_w,
            comb_w,
            MIX=mix,
            H=H,
            CHUNKS=CHUNKS,
            NORM_EPS=norm_eps,
            HC_EPS=hc_eps,
            ITERS=iters,
            BLOCK=1024,
            num_warps=4,
            enable_fp_fusion=False,
        )
    return out, ((pre, post_n, comb_n) if mix else None)


_V3_GROUPS = 8


@jit
def _v3_new_residual(residual, y, post, comb, t, col, H: tl.constexpr):
    base = residual + t * 4 * H
    r0 = tl.load(base + 0 * H + col)
    r1 = tl.load(base + 1 * H + col)
    r2 = tl.load(base + 2 * H + col)
    r3 = tl.load(base + 3 * H + col)
    yv = tl.load(y + t * H + col).to(tl.float32)
    cb = comb + t * 16
    pb = post + t * 4
    n0 = (
        tl.load(cb + 0) * r0
        + tl.load(cb + 4) * r1
        + tl.load(cb + 8) * r2
        + tl.load(cb + 12) * r3
        + tl.load(pb + 0) * yv
    )
    n1 = (
        tl.load(cb + 1) * r0
        + tl.load(cb + 5) * r1
        + tl.load(cb + 9) * r2
        + tl.load(cb + 13) * r3
        + tl.load(pb + 1) * yv
    )
    n2 = (
        tl.load(cb + 2) * r0
        + tl.load(cb + 6) * r1
        + tl.load(cb + 10) * r2
        + tl.load(cb + 14) * r3
        + tl.load(pb + 2) * yv
    )
    n3 = (
        tl.load(cb + 3) * r0
        + tl.load(cb + 7) * r1
        + tl.load(cb + 11) * r2
        + tl.load(cb + 15) * r3
        + tl.load(pb + 3) * yv
    )
    return n0, n1, n2, n3


@jit
def _transition_parts3(
    residual,
    y,
    post,
    comb,
    fn,
    cpre,
    cbuf,
    dots,
    squares,
    csquares,
    MIX: tl.constexpr,
    H: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    ROWS: tl.constexpr,
):
    # v1 arithmetic, but the 24 fn rows are split over program_id(2) and the
    # residual is left untouched here (finish3 stores the identical update).
    t, chunk, grp = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    col = chunk * COLS + tl.arange(0, COLS)
    n0, n1, n2, n3 = _v3_new_residual(residual, y, post, comb, t, col, H)
    if grp == 0:
        pc = cpre + t * 4
        coll = (
            n0 * tl.load(pc + 0)
            + n1 * tl.load(pc + 1)
            + n2 * tl.load(pc + 2)
            + n3 * tl.load(pc + 3)
        )
        tl.store(cbuf + t * H + col, coll)
        tl.store(csquares + t * CHUNKS + chunk, tl.sum(coll * coll, 0))
        if MIX:
            tl.store(
                squares + t * CHUNKS + chunk,
                tl.sum(n0 * n0 + n1 * n1 + n2 * n2 + n3 * n3, 0),
            )
    if MIX:
        for i in tl.static_range(ROWS):
            r = grp * ROWS + i
            w = fn + r * 4 * H
            d = (
                tl.load(w + 0 * H + col) * n0
                + tl.load(w + 1 * H + col) * n1
                + tl.load(w + 2 * H + col) * n2
                + tl.load(w + 3 * H + col) * n3
            )
            tl.store(dots + (t * CHUNKS + chunk) * 24 + r, tl.sum(d, 0))


@jit
def _transition_finish3(
    residual,
    y,
    post_in,
    comb_in,
    dots,
    squares,
    csquares,
    cbuf,
    norm,
    out,
    scale,
    base,
    pre,
    post,
    comb,
    MIX: tl.constexpr,
    H: tl.constexpr,
    COLS: tl.constexpr,
    CHUNKS: tl.constexpr,
    NORM_EPS: tl.constexpr,
    HC_EPS: tl.constexpr,
    ITERS: tl.constexpr,
):
    t, chunk = tl.program_id(0), tl.program_id(1)
    col = chunk * COLS + tl.arange(0, COLS)
    n0, n1, n2, n3 = _v3_new_residual(residual, y, post_in, comb_in, t, col, H)
    rb = residual + t * 4 * H
    total = tl.full((), 0, tl.float32)
    for c in tl.static_range(CHUNKS):
        total += tl.load(csquares + t * CHUNKS + c)
    inv = tl.rsqrt(total / H + NORM_EPS)
    v = tl.load(cbuf + t * H + col) * inv * tl.load(norm + col)
    tl.store(out + t * H + col, v.to(tl.float16))
    if MIX:  # noqa: SIM102
        if chunk == 0:
            i = tl.arange(0, 4)
            sq = tl.full((), 0, tl.float32)
            for c in tl.static_range(CHUNKS):
                sq += tl.load(squares + t * CHUNKS + c)
            rs = tl.rsqrt(sq / (4 * H) + NORM_EPS)
            p = tl.zeros((4,), tl.float32)
            q = tl.zeros((4,), tl.float32)
            location = i[:, None] * 4 + i[None, :]
            m = tl.zeros((4, 4), tl.float32)
            for c in tl.static_range(CHUNKS):
                row = dots + (t * CHUNKS + c) * 24
                p += tl.load(row + i)
                q += tl.load(row + 4 + i)
                m += tl.load(row + 8 + location)
            p = p * rs * tl.load(scale) + tl.load(base + i)
            q = q * rs * tl.load(scale + 1) + tl.load(base + 4 + i)
            tl.store(pre + t * 4 + i, tl.sigmoid(p) + HC_EPS)
            tl.store(post + t * 4 + i, 2.0 * tl.sigmoid(q))
            value = m * rs * tl.load(scale + 2) + tl.load(base + 8 + location)
            value = tl.exp(value - tl.max(value, 1)[:, None])
            value = value / tl.sum(value, 1)[:, None] + HC_EPS
            value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
            for _ in range(ITERS - 1):
                value = value / (tl.sum(value, 1)[:, None] + HC_EPS)
                value = value / (tl.sum(value, 0)[None, :] + HC_EPS)
            tl.store(comb + t * 16 + location, value)
    # Every program has read its residual chunk (n above) before storing it.
    tl.store(rb + 0 * H + col, n0)
    tl.store(rb + 1 * H + col, n1)
    tl.store(rb + 2 * H + col, n2)
    tl.store(rb + 3 * H + col, n3)


def _hc_transition3(
    residual,
    y,
    post,
    comb,
    collapse_pre,
    norm_weight,
    fn,
    scale,
    base,
    norm_eps,
    hc_eps,
    iters,
):
    tokens = residual.shape[0]
    mix = fn is not None
    device = residual.device
    cbuf = torch.empty(tokens, H, dtype=torch.float32, device=device)
    csq = torch.empty(tokens, CHUNKS, dtype=torch.float32, device=device)
    if mix:
        dots = torch.empty(tokens, CHUNKS, 24, dtype=torch.float32, device=device)
        sq = torch.empty(tokens, CHUNKS, dtype=torch.float32, device=device)
        pre = torch.empty(tokens, 4, dtype=torch.float32, device=device)
        post_n = torch.empty_like(pre)
        comb_n = torch.empty(tokens, 4, 4, dtype=torch.float32, device=device)
    else:
        dots = sq = pre = post_n = comb_n = csq
    out = torch.empty(tokens, H, dtype=torch.float16, device=device)
    groups = _V3_GROUPS if mix else 1
    _transition_parts3[(tokens, CHUNKS, groups)](
        residual,
        y,
        post,
        comb,
        fn if mix else post,
        collapse_pre,
        cbuf,
        dots,
        sq,
        csq,
        MIX=mix,
        H=H,
        COLS=COLS,
        CHUNKS=CHUNKS,
        ROWS=24 // groups,
        num_warps=4,
        enable_fp_fusion=False,
    )
    _transition_finish3[(tokens, CHUNKS)](
        residual,
        y,
        post,
        comb,
        dots,
        sq,
        csq,
        cbuf,
        norm_weight,
        out,
        scale if mix else csq,
        base if mix else csq,
        pre,
        post_n,
        comb_n,
        MIX=mix,
        H=H,
        COLS=COLS,
        CHUNKS=CHUNKS,
        NORM_EPS=norm_eps,
        HC_EPS=hc_eps,
        ITERS=iters,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return out, ((pre, post_n, comb_n) if mix else None)

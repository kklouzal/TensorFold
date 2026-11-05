"""Fused Gemma 4 layer glue: the norm/residual chains between projections, one program per row.

Each kernel reproduces the unfused operations and their BF16 storage boundaries exactly: a learned RMS norm is the
FP32 normalization (BF16-rounded first under MLX numerics) times the weight with one BF16 rounding; tensor adds and
products round to BF16 as Torch's BF16 elementwise operations do. Only the FP32 sum-of-squares reduction order
differs from the standalone norm kernel; it is fixed per row and independent of the row count.

attention tail:  t = norm(o, w_attn); h = bf16(h + t); dense_in = norm(h, w_mlp); expert_in = norm(h, w_expert)
MoE tail:        s = bf16(norm(dense, w_dense) + norm(routed, w_routed)); h = bf16(bf16(residual + norm(s, w_out)) *
                 scalar); next = norm(h, w_next)   (the next layer's input norm, or the final norm)
GeGLU:           bf16(bf16(gelu_tanh(gate)) * up)
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice


@triton.jit
def _ordered_combine(Y, IDS, WEIGHTS, row, n, N: tl.constexpr, R: tl.constexpr, RS: tl.constexpr):
    """FP32 weights, BF16 products and each ascending-expert BF16 addition."""
    slots = tl.arange(0, RS)
    ids = tl.load(IDS + row * R + slots, mask=slots < R, other=2147483647)
    total = tl.full(n.shape, 0, tl.float32)
    for _ in tl.static_range(R):
        expert = tl.min(ids, 0)
        slot = tl.min(tl.where(ids == expert, slots, 2147483647), 0)
        ids = tl.where(slots == slot, 2147483647, ids)
        y = tl.load(Y + (row * R + slot) * N + n, mask=n < N, other=0).to(tl.float32)
        weight = tl.load(WEIGHTS + row * R + slot)
        weighted = (y * weight).to(tl.bfloat16).to(tl.float32)
        total = (total + weighted).to(tl.bfloat16).to(tl.float32)
    return total


@triton.jit
def _norm(x, mask, w, D: tl.constexpr, EPS: tl.constexpr, ROUND: tl.constexpr):
    """Learned RMS norm of an FP32 row (masked lanes zero) -> FP32 values of the BF16 result."""
    inv = tl.rsqrt(tl.sum(x * x, 0) / D + EPS)
    y = x * inv
    if ROUND:
        y = y.to(tl.bfloat16).to(tl.float32)
    return tl.where(mask, (y * w).to(tl.bfloat16).to(tl.float32), 0.0)


@triton.jit
def _attention_tail(H, ATT, WA, WM, WE, HO, DI, EI, D: tl.constexpr, B: tl.constexpr, EPS: tl.constexpr,
                    ROUND: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, B)
    mask = d < D
    o = tl.load(ATT + row * D + d, mask=mask, other=0.0).to(tl.float32)
    t = _norm(o, mask, tl.load(WA + d, mask=mask, other=0.0).to(tl.float32), D, EPS, ROUND)
    h = (tl.load(H + row * D + d, mask=mask, other=0.0).to(tl.float32) + t).to(tl.bfloat16).to(tl.float32)
    tl.store(HO + row * D + d, h.to(tl.bfloat16), mask=mask)
    tl.store(DI + row * D + d, _norm(h, mask, tl.load(WM + d, mask=mask, other=0.0).to(tl.float32), D, EPS,
                                     ROUND).to(tl.bfloat16), mask=mask)
    tl.store(EI + row * D + d, _norm(h, mask, tl.load(WE + d, mask=mask, other=0.0).to(tl.float32), D, EPS,
                                     ROUND).to(tl.bfloat16), mask=mask)


@triton.jit
def _moe_tail(R, DN, RT, WD, WR, WO, SC, WN, HO, NO, D: tl.constexpr, B: tl.constexpr, EPS: tl.constexpr,
              ROUND: tl.constexpr, IDS, WEIGHTS, TOPK: tl.constexpr, ROUTE_SLOTS: tl.constexpr,
              FUSED_COMBINE: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, B)
    mask = d < D
    dense = _norm(tl.load(DN + row * D + d, mask=mask, other=0.0).to(tl.float32), mask,
                  tl.load(WD + d, mask=mask, other=0.0).to(tl.float32), D, EPS, ROUND)
    if FUSED_COMBINE:
        routed_row = _ordered_combine(RT, IDS, WEIGHTS, row, d, D, TOPK, ROUTE_SLOTS)
        # Match the old BF16 store/masked reload: padding is zero even for nonfinite routing weights.
        routed_row = tl.where(mask, routed_row, 0.0)
    else:
        routed_row = tl.load(RT + row * D + d, mask=mask, other=0.0).to(tl.float32)
    routed = _norm(routed_row, mask, tl.load(WR + d, mask=mask, other=0.0).to(tl.float32), D, EPS, ROUND)
    s = (dense + routed).to(tl.bfloat16).to(tl.float32)
    u = _norm(s, mask, tl.load(WO + d, mask=mask, other=0.0).to(tl.float32), D, EPS, ROUND)
    h = (tl.load(R + row * D + d, mask=mask, other=0.0).to(tl.float32) + u).to(tl.bfloat16).to(tl.float32)
    h = (h * tl.load(SC).to(tl.float32)).to(tl.bfloat16).to(tl.float32)
    tl.store(HO + row * D + d, h.to(tl.bfloat16), mask=mask)
    tl.store(NO + row * D + d, _norm(h, mask, tl.load(WN + d, mask=mask, other=0.0).to(tl.float32), D, EPS,
                                     ROUND).to(tl.bfloat16), mask=mask)


@triton.jit
def _geglu(G, U, OUT, N, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    mask = i < N
    gate = tl.load(G + i, mask=mask, other=0.0).to(tl.float32)
    up = tl.load(U + i, mask=mask, other=0.0).to(tl.float32)
    cube = gate * gate * gate                                  # Torch: kBeta * (x + kKappa * (x * x * x))
    inner = 0.7978845608028654 * (gate + 0.044715 * cube)
    gelu = (0.5 * gate * (1.0 + libdevice.tanh(inner))).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + i, (gelu * up).to(tl.bfloat16), mask=mask)


def attention_tail(h, o, w_attn, w_mlp, w_expert, eps: float, round_normalized: bool):
    """(new residual h, dense MLP input, expert input), each BF16 [R, D]."""
    rows, d = h.shape
    out = torch.empty((3, rows, d), dtype=torch.bfloat16, device=h.device)
    _attention_tail[(rows,)](h.contiguous(), o.contiguous(), w_attn, w_mlp, w_expert, out[0], out[1], out[2],
                             D=d, B=triton.next_power_of_2(d), EPS=eps, ROUND=round_normalized, num_warps=8,
                             enable_fp_fusion=False)
    return out[0], out[1], out[2]


def moe_tail(residual, dense, routed, w_dense, w_routed, w_out, scalar, w_next, eps: float, round_normalized: bool):
    """(layer output h, its next norm with ``w_next``), each BF16 [R, D]."""
    rows, d = residual.shape
    out = torch.empty((2, rows, d), dtype=torch.bfloat16, device=residual.device)
    _moe_tail[(rows,)](residual.contiguous(), dense.contiguous(), routed.contiguous(), w_dense, w_routed, w_out,
                       scalar, w_next, out[0], out[1], D=d, B=triton.next_power_of_2(d), EPS=eps,
                       ROUND=round_normalized, IDS=None, WEIGHTS=None, TOPK=0, ROUTE_SLOTS=0,
                       FUSED_COMBINE=False, num_warps=8, enable_fp_fusion=False)
    return out[0], out[1]


def moe_tail_slots(residual, dense, slots, picks, routing, w_dense, w_routed, w_out, scalar, w_next,
                   eps: float):
    """Consume validated borrowed CT slots immediately on their producer's lane.

    The owner keeps slot/pick storage stable. The caller must enqueue this tail
    before another expert bank can overwrite that shared workspace; no borrowed
    tensor escapes as an output. Norm geometry matches moe_tail exactly.
    """
    rows, d = residual.shape
    out = torch.empty((2, rows, d), dtype=torch.bfloat16, device=residual.device)
    _moe_tail[(rows,)](residual.contiguous(), dense.contiguous(), slots, w_dense, w_routed, w_out,
        scalar, w_next, out[0], out[1], D=d, B=triton.next_power_of_2(d), EPS=eps, ROUND=False,
        IDS=picks, WEIGHTS=routing, TOPK=8, ROUTE_SLOTS=8, FUSED_COMBINE=True,
        num_warps=8, enable_fp_fusion=False)
    return out[0], out[1]


def geglu(gate, up):
    """HF/CT GeGLU: BF16 tanh-GELU storage, then a separate BF16 product."""
    gate, up = gate.contiguous(), up.contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    _geglu[(triton.cdiv(n, 1024),)](gate, up, out, n, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return out

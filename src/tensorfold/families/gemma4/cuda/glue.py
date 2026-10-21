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
              ROUND: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    d = tl.arange(0, B)
    mask = d < D
    dense = _norm(tl.load(DN + row * D + d, mask=mask, other=0.0).to(tl.float32), mask,
                  tl.load(WD + d, mask=mask, other=0.0).to(tl.float32), D, EPS, ROUND)
    routed = _norm(tl.load(RT + row * D + d, mask=mask, other=0.0).to(tl.float32), mask,
                   tl.load(WR + d, mask=mask, other=0.0).to(tl.float32), D, EPS, ROUND)
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


@triton.jit
def _softcap30(X, OUT, N, BLOCK: tl.constexpr):
    i = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(X + i, mask=i < N, other=0.0).to(tl.float32)
    # Torch's CUDA scalar division uses float32(double(1) / scalar).
    # Its BF16 TensorIterator outputs round after division and tanh, before
    # the separate multiplication. Keep both boundaries in this one kernel.
    divided = (x * (1.0 / 30.0)).to(tl.bfloat16).to(tl.float32)
    activated = libdevice.tanh(divided).to(tl.bfloat16).to(tl.float32)
    tl.store(OUT + i, (activated * 30.0).to(tl.bfloat16), mask=i < N)


def softcap30(logits: torch.Tensor) -> torch.Tensor:
    """BF16 ``tanh(logits / 30) * 30`` with every Torch storage boundary.

    The CT target owns this operation; BF16 DFlash borrows project_logits.
    The private FP8 companion's FP32 head retains its separate operation.
    Inputs are the target's fresh contiguous BF16 head outputs. The output
    owns separate storage, including during CUDA graph capture and replay.
    """
    out = torch.empty_like(logits)
    n = logits.numel()
    # Torch preserves subnormal tanh results; libdevice's default FTZ would not.
    _softcap30[(triton.cdiv(n, 1024),)](logits, out, n, BLOCK=1024, num_warps=4,
                                      enable_fp_fusion=False, enable_reflect_ftz=False)
    return out


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
                       ROUND=round_normalized, num_warps=8, enable_fp_fusion=False)
    return out[0], out[1]


def geglu(gate, up):
    """HF/CT GeGLU: BF16 tanh-GELU storage, then a separate BF16 product."""
    gate, up = gate.contiguous(), up.contiguous()
    out = torch.empty_like(gate)
    n = gate.numel()
    _geglu[(triton.cdiv(n, 1024),)](gate, up, out, n, BLOCK=1024, num_warps=4, enable_fp_fusion=False)
    return out

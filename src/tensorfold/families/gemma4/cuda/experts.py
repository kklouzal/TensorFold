"""Native MLX mixed-bit experts with Gemma's BF16 GeGLU and slot boundaries."""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

from .linear import Linear, _affine_dot, _codes


@triton.jit
def _selected(X, IDS, W, S, B, OUT, E: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
              BITS: tl.constexpr, R: tl.constexpr, SLOTTED: tl.constexpr, BN: tl.constexpr):
    slot = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    expert = tl.load(IDS + slot).to(tl.int64)
    valid = (expert >= 0) & (expert < E)
    tl.device_assert(valid, "Gemma routed expert outside expert bank")
    xrow = slot if SLOTTED else slot // R
    y = _affine_dot(X, W, S, B, xrow, expert * N + n, valid & (n < N), K, BITS, BN)
    tl.store(OUT + slot * N + n, y.to(tl.bfloat16), mask=n < N)


@triton.jit
def _gate_up(X, IDS, GW, GS, GB, UW, US, UB, ACT, E: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
             GBITS: tl.constexpr, UBITS: tl.constexpr, R: tl.constexpr, BN: tl.constexpr):
    slot = tl.program_id(0)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    expert = tl.load(IDS + slot).to(tl.int64)
    valid = (expert >= 0) & (expert < E)
    tl.device_assert(valid, "Gemma routed expert outside expert bank")
    wr = expert * N + n
    # Share each input load and group sum across the two projections. Their
    # independent accumulation chains retain the individual linear contract.
    rk = tl.arange(0, 64)
    ga = tl.full((BN,), 0, tl.float32)
    ua = tl.full((BN,), 0, tl.float32)
    for group in range(K // 64):
        k = group * 64 + rk
        x = tl.load(X + (slot // R) * K + k).to(tl.float32)
        gq = _codes(GW, wr[:, None], k[None, :], valid & (n[:, None] < N), K * GBITS // 32, GBITS)
        uq = _codes(UW, wr[:, None], k[None, :], valid & (n[:, None] < N), K * UBITS // 32, UBITS)
        gp = tl.sum(gq.to(tl.float32) * x[None, :], 1)
        up = tl.sum(uq.to(tl.float32) * x[None, :], 1)
        xs = tl.sum(x, 0)
        at = wr * (K // 64) + group
        gs = tl.load(GS + at, mask=valid & (n < N), other=0).to(tl.float32)
        gb = tl.load(GB + at, mask=valid & (n < N), other=0).to(tl.float32)
        us = tl.load(US + at, mask=valid & (n < N), other=0).to(tl.float32)
        ub = tl.load(UB + at, mask=valid & (n < N), other=0).to(tl.float32)
        ga = tl.fma(xs, gb, tl.fma(gp, gs, ga))
        ua = tl.fma(xs, ub, tl.fma(up, us, ua))
    gate = ga.to(tl.bfloat16).to(tl.float32)
    up = ua.to(tl.bfloat16).to(tl.float32)
    # The checkpoint declares gelu_pytorch_tanh. Both linear outputs round to
    # BF16 first; the activation/product has one BF16 storage boundary.
    inner = 0.7978845608028654 * (gate + 0.044715 * gate * gate * gate)
    gelu = 0.5 * gate * (1.0 + libdevice.tanh(inner))
    tl.store(ACT + slot * N + n, (gelu * up).to(tl.bfloat16), mask=n < N)


@triton.jit
def _weighted_sum(Y, WEIGHTS, OUT, N: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    n = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    acc = tl.full((BLOCK,), 0, tl.float32)
    # The vendor contract rounds each weighted slot to BF16, then adds FP32
    # slots in selected order. Tree summation would change that contract.
    for slot in tl.static_range(R):
        y = tl.load(Y + (row * R + slot) * N + n, mask=n < N, other=0).to(tl.float32)
        weight = tl.load(WEIGHTS + row * R + slot).to(tl.float32)
        term = (y * weight).to(tl.bfloat16).to(tl.float32)
        acc = acc + term
    tl.store(OUT + row * N + n, acc.to(tl.bfloat16), mask=n < N)


class ExpertLinear(Linear):
    """Packed [expert, output, packed-input] payload; only selected rows are read."""

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, bits: int, group: int = 64):
        super().__init__(weight, scales, biases, bits, group, _rank=3)
        self.e = weight.shape[0]

    def embedding(self, ids: torch.Tensor) -> torch.Tensor:
        raise ValueError("Gemma CUDA expert banks require explicit expert picks, not embedding lookup")

    def _picks(self, picks: torch.Tensor, rows: int) -> torch.Tensor:
        if (not isinstance(picks, torch.Tensor) or not picks.is_cuda or picks.device != self.weight.device
                or picks.dtype not in (torch.int32, torch.int64) or picks.ndim != 2 or picks.shape[0] != rows
                or not 1 <= picks.shape[1] <= self.e):
            raise ValueError("Gemma CUDA expert picks must be int32/int64 [rows, slots] on the weight device")
        return picks.contiguous()

    def __call__(self, x: torch.Tensor, picks: torch.Tensor) -> torch.Tensor:
        rows = self._input(x)
        if x.ndim not in (2, 3):
            raise ValueError("Gemma CUDA expert input must have shape [rows,K] or [rows,slots,K]")
        ids = self._picks(picks, x.shape[0])
        if x.ndim == 3 and x.shape[1] != ids.shape[1]:
            raise ValueError("Gemma CUDA expert input and picks must have matching slot counts")
        count, slots = ids.shape
        out = torch.empty((count, slots, self.n), dtype=torch.bfloat16, device=x.device)
        _selected[(count * slots, triton.cdiv(self.n, 4))](
            rows, ids, self.weight, self.scales, self.biases, out, E=self.e, N=self.n, K=self.k,
            BITS=self.bits, R=slots, SLOTTED=x.ndim == 3, BN=4, num_warps=4, enable_fp_fusion=False, debug=True)
        return out


class Experts:
    """Immutable banks share CUDA device/shape; scratch belongs to each forward."""

    def __init__(self, gate: ExpertLinear, up: ExpertLinear, down: ExpertLinear):
        if not all(isinstance(v, ExpertLinear) for v in (gate, up, down)):
            raise ValueError("Gemma CUDA Experts requires three ExpertLinear banks")
        if ((gate.e, gate.n, gate.k) != (up.e, up.n, up.k)
                or (down.e, down.n, down.k) != (gate.e, gate.k, gate.n)
                or any(v.weight.device != gate.weight.device for v in (up, down))):
            raise ValueError("Gemma CUDA gate/up/down expert banks have incompatible shapes or devices")
        self.gate, self.up, self.down = gate, up, down

    def nbytes(self) -> int:
        return sum(v.nbytes() for v in (self.gate, self.up, self.down))

    def forward(self, x: torch.Tensor, picks: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        g, u, d = self.gate, self.up, self.down
        rows = g._input(x)
        if x.ndim != 2:
            raise ValueError("Gemma CUDA expert MLP input must have shape [rows,hidden]")
        ids = g._picks(picks, rows.shape[0])
        if (not isinstance(weights, torch.Tensor) or weights.dtype != torch.bfloat16 or not weights.is_cuda
                or weights.device != x.device or tuple(weights.shape) != tuple(ids.shape)):
            raise ValueError("Gemma CUDA routed weights must be BF16 with the same shape/device as expert picks")
        count, slots = ids.shape
        act = torch.empty((count, slots, g.n), dtype=torch.bfloat16, device=x.device)
        _gate_up[(count * slots, triton.cdiv(g.n, 4))](
            rows, ids, g.weight, g.scales, g.biases, u.weight, u.scales, u.biases, act,
            E=g.e, N=g.n, K=g.k, GBITS=g.bits, UBITS=u.bits, R=slots, BN=4, num_warps=4,
            enable_fp_fusion=False, debug=True)
        projected = d(act, ids)
        out = torch.empty((count, d.n), dtype=torch.bfloat16, device=x.device)
        _weighted_sum[(count, triton.cdiv(d.n, 256))](
            projected, weights.contiguous(), out, N=d.n, R=slots, BLOCK=256, num_warps=4, enable_fp_fusion=False)
        return out

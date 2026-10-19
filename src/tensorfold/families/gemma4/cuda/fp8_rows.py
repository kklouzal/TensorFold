"""FP8 E4M3 weights with one FP32 scale per output row, for Gemma DFlash proposals only.

Proposals never change output (the target verifies every token), so the companion may run its projections and its
borrowed vocabulary head on a private FP8 copy: half the bytes of BF16. ``scale_i = max|W_i| / 448`` and
``W8_i = e4m3(W_i / scale_i)``; a row's product is one FP32 tensor-core K loop over BF16-converted codes, scaled once.
The target's own BF16 head is untouched.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl
from triton.language.extra.cuda import libdevice

BN = 64          # output rows a program carries
BK = 128         # K tile
CONVERT_ROWS = 16384   # weight rows quantized per FP32 staging slice at load


@triton.jit
def _rows(X, W8, S, OUT, M, N, K: tl.constexpr, BKC: tl.constexpr, BNC: tl.constexpr, CAP: tl.constexpr,
          SOFTCAP: tl.constexpr):
    n = tl.program_id(0) * BNC + tl.arange(0, BNC)
    nok = n < N
    m = tl.program_id(1) * 16 + tl.arange(0, 16)
    mok = m < M
    acc = tl.zeros((16, BNC), tl.float32)
    for k0 in range(0, K, BKC):
        kk = k0 + tl.arange(0, BKC)
        x = tl.load(X + m.to(tl.int64)[:, None] * K + kk[None, :], mask=mok[:, None], other=0.0)
        w = tl.load(W8 + n.to(tl.int64)[:, None] * K + kk[None, :], mask=nok[:, None], other=0.0)
        acc = tl.dot(x, tl.trans(w.to(tl.bfloat16)), acc)
    y = acc * tl.load(S + n, mask=nok, other=0.0)[None, :]
    mask = mok[:, None] & nok[None, :]
    if SOFTCAP:
        tl.store(OUT + m.to(tl.int64)[:, None] * N + n[None, :], libdevice.tanh(y / CAP) * CAP, mask=mask)
    else:
        tl.store(OUT + m.to(tl.int64)[:, None] * N + n[None, :], y.to(tl.bfloat16), mask=mask)


class Fp8Rows:
    """A private FP8 copy of a BF16 [N, K] weight (K a multiple of 128); call with BF16 rows [M, K]."""

    def __init__(self, weight: torch.Tensor, *, softcap: float | None = None):
        if (not isinstance(weight, torch.Tensor) or weight.dtype != torch.bfloat16 or weight.ndim != 2
                or not weight.is_cuda or weight.shape[1] % BK):
            raise ValueError("FP8 proposal rows need a CUDA BF16 [N, K] weight with K a multiple of 128")
        n, k = weight.shape
        self.n, self.k, self.softcap = n, k, softcap
        self.codes = torch.empty((n, k), dtype=torch.float8_e4m3fn, device=weight.device)
        self.scale = torch.empty((n,), dtype=torch.float32, device=weight.device)
        for i in range(0, n, CONVERT_ROWS):
            w = weight[i:i + CONVERT_ROWS].float()
            top = w.abs().amax(1)
            scale = torch.where(top > 0, top / 448.0, torch.ones_like(top))
            self.codes[i:i + CONVERT_ROWS] = (w / scale[:, None]).to(torch.float8_e4m3fn)
            self.scale[i:i + CONVERT_ROWS] = scale

    def prefix(self, count: int) -> "Fp8Rows":
        """The first ``count`` output rows, sharing storage."""
        view = object.__new__(Fp8Rows)
        view.n, view.k, view.softcap = count, self.k, self.softcap
        view.codes, view.scale = self.codes[:count], self.scale[:count]
        return view

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """[M, N] BF16 rows (or FP32 softcapped logits for a head)."""
        if x.ndim != 2 or x.shape[1] != self.k or x.dtype != torch.bfloat16 or x.device != self.codes.device:
            raise ValueError(f"FP8 proposal rows take BF16 [M, {self.k}] on the weight's device")
        m = x.shape[0]
        out = torch.empty((m, self.n), dtype=torch.float32 if self.softcap else torch.bfloat16, device=x.device)
        _rows[(triton.cdiv(self.n, BN), triton.cdiv(m, 16))](
            x.contiguous(), self.codes, self.scale, out, m, self.n, K=self.k, BKC=BK, BNC=BN,
            CAP=float(self.softcap or 1.0), SOFTCAP=bool(self.softcap), num_warps=4, num_stages=3,
            enable_fp_fusion=False)
        return out

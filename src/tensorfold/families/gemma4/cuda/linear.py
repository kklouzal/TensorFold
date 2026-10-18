"""Gemma's original MLX affine codes, with row-invariant CUDA arithmetic.

The uint32 payload is a little-endian bit stream, including the 6-bit codes
which cross word boundaries. No weight or activation is requantized. Each
64-input group accumulates in FP32, followed by explicit scale and bias FMAs;
the group order and final BF16 rounding are independent of the row count.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _codes(W, row, k, valid, WORDS: tl.constexpr, BITS: tl.constexpr):
    bit = k * BITS
    word = bit // 32
    shift = bit % 32
    low = tl.load(W + row * WORDS + word, mask=valid, other=0).to(tl.uint32)
    if BITS == 6:
        # Only straddling codes read the next word. The last code never does;
        # masking also prevents a speculative read across a row allocation.
        high = tl.load(W + row * WORDS + word + 1, mask=valid & (shift > 32 - BITS), other=0).to(tl.uint32)
        low = (low >> shift) | (high << ((32 - shift) % 32))
    else:
        low = low >> shift
    return low & ((1 << BITS) - 1)


@triton.jit
def _affine_dot(X, W, S, B, xrow, wrow, valid, K: tl.constexpr, BITS: tl.constexpr, BN: tl.constexpr):
    KG: tl.constexpr = K // 64
    WORDS: tl.constexpr = K * BITS // 32
    rk = tl.arange(0, 64)
    acc = tl.full((BN,), 0, tl.float32)
    for g in range(KG):
        x = tl.load(X + xrow * K + g * 64 + rk).to(tl.float32)
        q = _codes(W, wrow[:, None], g * 64 + rk[None, :], valid[:, None], WORDS, BITS).to(tl.float32)
        dot = tl.sum(q * x[None, :], 1)
        sx = tl.sum(x, 0)
        scale = tl.load(S + wrow * KG + g, mask=valid, other=0).to(tl.float32)
        bias = tl.load(B + wrow * KG + g, mask=valid, other=0).to(tl.float32)
        acc = tl.fma(sx, bias, tl.fma(dot, scale, acc))
    return acc


@triton.jit
def _linear(X, W, S, B, OUT, N: tl.constexpr, K: tl.constexpr, BITS: tl.constexpr, BN: tl.constexpr):
    n = tl.program_id(0) * BN + tl.arange(0, BN)
    row = tl.program_id(1)
    y = _affine_dot(X, W, S, B, row, n, n < N, K, BITS, BN)
    tl.store(OUT + row * N + n, y.to(tl.bfloat16), mask=n < N)


@triton.jit
def _embedding(IDS, W, S, B, OUT, N: tl.constexpr, K: tl.constexpr, BITS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    k = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    token = tl.load(IDS + row).to(tl.int64)
    # IDs are validated by the model's token boundary. Keep the kernel safe if
    # an internal caller violates that contract: assert and mask the access.
    valid_id = (token >= 0) & (token < N)
    tl.device_assert(valid_id, "Gemma embedding token outside vocabulary")
    valid = (k < K) & valid_id
    q = _codes(W, token, k, valid, K * BITS // 32, BITS).to(tl.float32)
    scale = tl.load(S + token * (K // 64) + k // 64, mask=valid, other=0).to(tl.float32)
    bias = tl.load(B + token * (K // 64) + k // 64, mask=valid, other=0).to(tl.float32)
    y = tl.fma(q, scale, bias)
    tl.store(OUT + row * K + k, y.to(tl.bfloat16), mask=k < K)


def _geometry(shape: tuple[int, ...], scale_shape: tuple[int, ...], bits: int, group: int,
              rank: int) -> tuple[int, int]:
    """Validate the checkpoint's fixed affine ABI before allocating or launching."""

    if type(bits) is not int or bits not in (4, 6) or type(group) is not int or group != 64:
        raise ValueError("Gemma CUDA affine weights require 4 or 6 bits and group=64")
    if len(shape) != rank or len(scale_shape) != rank or any(type(d) is not int or d <= 0 for d in shape):
        raise ValueError(f"Gemma CUDA affine weights require a positive rank-{rank} packed tensor")
    if shape[:-1] != scale_shape[:-1] or scale_shape[-1] <= 0:
        raise ValueError("Gemma CUDA affine scales must match every packed weight row")
    k = scale_shape[-1] * group
    if k * bits // 32 != shape[-1]:
        raise ValueError("Gemma CUDA affine packed width disagrees with scales, group and bits")
    return shape[-2], k


class Linear:
    """Own the original packed tensors; callers must not mutate them after loading.

    Inputs and outputs are BF16. Scales/biases retain their declared BF16
    representation. CUDA graph users own input/output lifetimes and stream
    ordering; no host synchronization occurs in the projection hot path.
    """

    def __init__(self, weight: torch.Tensor, scales: torch.Tensor, biases: torch.Tensor, bits: int, group: int = 64,
                 *, _rank: int = 2):
        values = (weight, scales, biases)
        if any(not isinstance(t, torch.Tensor) for t in values):
            raise ValueError("Gemma CUDA affine weight, scales and biases must be tensors")
        self.n, self.k = _geometry(tuple(weight.shape), tuple(scales.shape), bits, group, _rank)
        if weight.dtype not in (torch.int32, torch.uint32):
            raise ValueError("Gemma CUDA affine weights must contain packed 32-bit words")
        if any(t.dtype != torch.bfloat16 or tuple(t.shape) != tuple(scales.shape) for t in (scales, biases)):
            raise ValueError("Gemma CUDA affine scales and biases must have matching BF16 shapes")
        if any(not t.is_cuda or t.device != weight.device or not t.is_contiguous() for t in values):
            raise ValueError("Gemma CUDA affine tensors must be contiguous on one CUDA device")
        self.weight, self.scales, self.biases = values
        self.bits, self.group = bits, group
        self._q4 = None
        if _rank == 2 and bits == 4:
            from tensorfold.cuda.kernels.qmm import pack

            self._q4 = pack(weight, scales, biases, group)
            self.weight, self.scales, self.biases = self._q4.weight, self._q4.scales, self._q4.biases

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.weight, self.scales, self.biases))

    def _input(self, x: torch.Tensor) -> torch.Tensor:
        if (not isinstance(x, torch.Tensor) or not x.is_cuda or x.device != self.weight.device
                or x.dtype != torch.bfloat16 or x.ndim < 1 or x.shape[-1] != self.k or x.numel() == 0):
            raise ValueError(f"Gemma CUDA affine input must be nonempty BF16 rows of width {self.k} on its device")
        return x.reshape(-1, self.k).contiguous()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows = self._input(x)
        if self._q4 is not None:
            from tensorfold.cuda.kernels.qmm import matmul

            return matmul(rows, self._q4).reshape(*x.shape[:-1], self.n)
        out = torch.empty((rows.shape[0], self.n), dtype=torch.bfloat16, device=rows.device)
        _linear[(triton.cdiv(self.n, 4), rows.shape[0])](
            rows, self.weight, self.scales, self.biases, out, N=self.n, K=self.k, BITS=self.bits, BN=4,
            num_warps=4, enable_fp_fusion=False)
        return out.reshape(*x.shape[:-1], self.n)

    def embedding(self, ids: torch.Tensor) -> torch.Tensor:
        """Look up original affine rows. IDs have already passed the model token boundary."""

        if self.bits != 6:
            raise ValueError("Gemma CUDA embedding lookup requires the checkpoint's 6-bit tied embedding")
        if (not isinstance(ids, torch.Tensor) or not ids.is_cuda or ids.device != self.weight.device
                or ids.dtype not in (torch.int32, torch.int64) or ids.numel() == 0):
            raise ValueError("Gemma CUDA embedding IDs must be nonempty int32/int64 on the weight device")
        tokens = ids.contiguous()
        out = torch.empty((*tokens.shape, self.k), dtype=torch.bfloat16, device=tokens.device)
        _embedding[(math.prod(tokens.shape), triton.cdiv(self.k, 256))](
            tokens, self.weight, self.scales, self.biases, out, N=self.n, K=self.k, BITS=self.bits, BLOCK=256,
            num_warps=4, enable_fp_fusion=False, debug=True)
        return out

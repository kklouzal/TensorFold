"""Private unconnected BF16 Tensor Core consumer for exact PackedBF16 operands.

Reuses the owned kernels/affine_kernels and moe._router dataflow: fixed16x32
output tile and64-wide K chain, BF16 tl.dot operands, FP32 accumulation. No TF32,
quantization, split-K, whole decoded weight owner, autotuning, or row dispatch.
Required metadata failures use side-effect PTX trap, independent of TRITON_DEBUG.
Invalid raw accesses are also masked before loads; masks do not hide the failure.
Private TARGET is the one current GB10 CUDA device selected by the launcher.
Caller retains immutable tensor storage/stream dependencies until readers retire.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .b16_lossless import PackedBF16, _validate


@triton.jit
def _fatal_if(bad):
    # Triton's is_pure=False emits a side-effecting inline-assembly operation.
    # No debug/assert flag controls it. GPU negative controls must verify this
    # with TRITON_DEBUG=0 and inspect actual compiled PTX before timing.
    tl.inline_asm_elementwise(
        "{ .reg .pred p; setp.ne.u32 p, $1, 0; @p trap; mov.u32 $0, 0; }",
        constraints="=r,r", args=[bad.to(tl.int32)], dtype=tl.int32,
        is_pure=False, pack=1)


@triton.jit
def _weight_tile(LOW, DELTA, META, RAW, cols, group, valid, RAW_BLOCKS,
                 KB: tl.constexpr):
    # Every caller consumes exactly one aligned64-value codec block per column.
    # Validate its common metadata once; exponent eligibility covers ALL64nibbles.
    offset = tl.arange(0, 64)
    block = cols.to(tl.int64) * KB + group
    info = tl.load(META + block, mask=valid, other=0xffffff00).to(tl.uint32)
    index = info >> 8
    compressed = index == 0xffffff
    raw_ok = index < RAW_BLOCKS
    _fatal_if(valid & ~compressed & ~raw_ok)
    raw = tl.load(RAW + index[:, None].to(tl.int64) * 64 + offset[None, :],
                  mask=valid[:, None] & ~compressed[:, None] & raw_ok[:, None], other=0).to(tl.uint16)
    low = tl.load(LOW + block[:, None] * 64 + offset[None, :],
                  mask=valid[:, None] & compressed[:, None], other=0).to(tl.uint32)
    packed = tl.load(DELTA + block[:, None] * 32 + offset[None, :] // 2,
                     mask=valid[:, None] & compressed[:, None], other=0).to(tl.uint32)
    difference = (packed >> ((offset[None, :] & 1) * 4)) & 15
    maximum = info & 255
    bad_exponent = tl.max(difference, axis=1) > maximum
    _fatal_if(valid & compressed & bad_exponent)
    exponent = maximum[:, None] - difference  # unsigned, never signed overflow.
    reconstructed = (((low & 128) << 8) | (exponent << 7) | (low & 127)).to(tl.uint16)
    words = tl.where(compressed[:, None], reconstructed, raw).to(tl.uint16)
    return words.to(tl.bfloat16, bitcast=True)


@triton.jit(do_not_specialize=['M', 'RAW_BLOCKS'])
def _matmul(X, LOW, DELTA, META, RAW, OUT, M, RAW_BLOCKS,
            N: tl.constexpr, K: tl.constexpr):
    rows = tl.arange(0, 16)
    cols = tl.program_id(0) * 32 + tl.arange(0, 32)
    offsets = tl.arange(0, 64)
    accumulator = tl.zeros((16, 32), tl.float32)
    for group in range(K // 64):
        kk = group * 64 + offsets
        x = tl.load(X + rows[:, None].to(tl.int64) * K + kk[None, :], mask=rows[:, None] < M, other=0)
        weight = _weight_tile(LOW, DELTA, META, RAW, cols, group, cols < N,
                              RAW_BLOCKS, K // 64)
        accumulator = tl.dot(x, tl.trans(weight), accumulator,
                             input_precision='ieee', out_dtype=tl.float32)
    tl.store(OUT + rows[:, None].to(tl.int64) * N + cols[None, :], accumulator.to(tl.bfloat16, fp_downcast_rounding="rtne"),
             mask=(rows[:, None] < M) & (cols[None, :] < N))


@triton.jit(do_not_specialize=['RAW_BLOCKS'])
def _decode(LOW, DELTA, META, RAW, OUT, RAW_BLOCKS,
            N: tl.constexpr, K: tl.constexpr):
    group = tl.program_id(0) % (K // 64)
    cols = (tl.program_id(0) // (K // 64)) * 32 + tl.arange(0, 32)
    kk = group * 64 + tl.arange(0, 64)
    value = _weight_tile(LOW, DELTA, META, RAW, cols, group, cols < N,
                         RAW_BLOCKS, K // 64)
    tl.store(OUT + cols[:, None].to(tl.int64) * K + kk[None, :], value, mask=cols[:, None] < N)


def matmul_rows(x: torch.Tensor, weight: PackedBF16):
    """M1..8 always use physicalM16; input dtype/stride never silently converted."""
    _validate(weight, cuda=True)
    n, k = weight.shape
    if (not isinstance(x, torch.Tensor) or x.device != weight.low.device or x.dtype != torch.bfloat16
            or x.ndim != 2 or not x.is_contiguous() or not 1 <= x.shape[0] <= 8 or x.shape[1] != k):
        raise ValueError('contiguous same-device BF16 [M1..8,K] input required')
    out = torch.empty((x.shape[0], n), device=x.device, dtype=torch.bfloat16)
    _matmul[(triton.cdiv(n, 32),)](
        x, weight.low, weight.delta, weight.metadata, weight.raw, out,
        x.shape[0], len(weight.raw), n, k, num_warps=4, num_stages=2,
        enable_fp_fusion=False, debug=False)
    return out


def unpack(weight: PackedBF16):
    """Diagnostic full decodedBF16 matrix; never retained by fused projection."""
    _validate(weight, cuda=True)
    n, k = weight.shape
    blocks = triton.cdiv(n, 32) * (k // 64)
    if blocks > (1 << 31) - 1:
        raise ValueError("decoder grid exceeds CUDA x-axis limit")
    out = torch.empty((n, k), device=weight.low.device, dtype=torch.bfloat16)
    _decode[(blocks,)](
        weight.low, weight.delta, weight.metadata, weight.raw, out, len(weight.raw), n, k,
        num_warps=4, enable_fp_fusion=False, debug=False)
    return out

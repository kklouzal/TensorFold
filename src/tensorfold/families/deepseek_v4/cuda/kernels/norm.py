# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
#   vllm/models/deepseek_v41/norm.py (blob 9f32f42bdfa6c0e107cc681ab1597a17fa1c48e4)
# TensorFold changes (kernel arithmetic verbatim):
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - vllm.triton_utils -> triton_utils.
# - rms_norm uses on_triton_device(x) (identical in compiled runs; CPU only under TRITON_INTERPRET).
# - The wide path follows the pinned ATen four-value mean load and reduction order.
"""One-launch FP32 RMS arithmetic for V4.1's FP16 rows."""

import torch

from .triton_utils import jit, on_triton_device, tl, triton

RMS_WIDTHS = (128, 512, 1280, 5120)


@jit
def _short_rms(x, weight, out, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    at = tl.arange(0, BLOCK)
    value = tl.load(x + row * H + at, at < H, 0).to(tl.float32)
    variance = tl.sum(value * value, 0) / H
    result = value * tl.rsqrt(variance + EPS) * tl.load(weight + at, at < H, 0)
    tl.store(out + row * H + at, result, at < H)


@jit
def _prefill_rms(x, weight, out, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    source = x + row * H
    four = tl.arange(0, 32) * 4
    sum0 = tl.full((32,), 0, tl.float32)
    sum1 = tl.full((32,), 0, tl.float32)
    sum2 = tl.full((32,), 0, tl.float32)
    sum3 = tl.full((32,), 0, tl.float32)
    # ATen CUDA Reduce.cuh, pinned 05adb434: one warp for an aligned row,
    # with four interleaved FP32 accumulators per lane and a descending
    # warp reduction. Form its squared FP32 inputs before each addition.
    for first in range(H // 128):
        at = first * 128 + four
        a = tl.load(source + at).to(tl.float32)
        b = tl.load(source + at + 1).to(tl.float32)
        c = tl.load(source + at + 2).to(tl.float32)
        d = tl.load(source + at + 3).to(tl.float32)
        sum0 += a * a
        sum1 += b * b
        sum2 += c * c
        sum3 += d * d
    variance = tl.sum(((sum0 + sum1) + sum2) + sum3, 0) * (1.0 / H)
    scale = tl.rsqrt(variance + EPS)
    at = tl.arange(0, BLOCK)
    value = tl.load(source + at, at < H, 0).to(tl.float32)
    result = value * scale * tl.load(weight + at, at < H, 0)
    tl.store(out + row * H + at, result, at < H)


def rms_norm(x, weight, eps):
    rows, hidden = x.shape if x.ndim == 2 else (0, 0)
    if (
        1 <= rows <= 8192
        and hidden in RMS_WIDTHS
        and x.dtype == torch.float16
        and x.is_contiguous()
        and weight.shape == (hidden,)
        and weight.dtype == torch.float32
        and weight.device == x.device
        and weight.is_contiguous()
        and on_triton_device(x)
    ):
        output = torch.empty_like(x)
        kernel, warps = (_short_rms, 4) if rows <= 64 else (_prefill_rms, 1)
        kernel[(rows,)](
            x,
            weight,
            output,
            H=hidden,
            EPS=eps,
            BLOCK=triton.next_power_of_2(hidden),
            num_warps=warps,
            enable_fp_fusion=False,
        )
        return output
    values = x.float()
    return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + eps) * weight).to(x.dtype)

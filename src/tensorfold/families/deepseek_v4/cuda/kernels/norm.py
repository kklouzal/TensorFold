# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
#   vllm/models/deepseek_v41/norm.py (blob 9f32f42bdfa6c0e107cc681ab1597a17fa1c48e4)
# TensorFold changes (kernel arithmetic verbatim):
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - vllm.triton_utils -> triton_utils.
# - short_rms_norm: x.is_cuda -> on_triton_device(x) (identical in compiled runs; CPU only under TRITON_INTERPRET).
"""One-launch FP32 RMS arithmetic for V4.1's short FP16 rows."""

import torch

from .triton_utils import jit, on_triton_device, tl, triton


@jit
def _short_rms(x, weight, out, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    at = tl.arange(0, BLOCK)
    value = tl.load(x + row * H + at, at < H, 0).to(tl.float32)
    variance = tl.sum(value * value, 0) / H
    result = value * tl.rsqrt(variance + EPS) * tl.load(weight + at, at < H, 0)
    tl.store(out + row * H + at, result, at < H)


def short_rms_norm(x, weight, eps):
    if (
        x.ndim != 2
        or not on_triton_device(x)
        or x.dtype != torch.float16
        or not 1 <= x.shape[0] <= 6
        or x.shape[1] not in (128, 512, 1280, 5120)
        or not x.is_contiguous()
        or weight.shape != (x.shape[1],)
        or weight.dtype != torch.float32
        or weight.device != x.device
        or not weight.is_contiguous()
    ):
        raise ValueError("V4.1 short RMS requires FP16 M1..6 and FP32 weights")
    output = torch.empty_like(x)
    _short_rms[(x.shape[0],)](
        x,
        weight,
        output,
        H=x.shape[1],
        EPS=eps,
        BLOCK=triton.next_power_of_2(x.shape[1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return output

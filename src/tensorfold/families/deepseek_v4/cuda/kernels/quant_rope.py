# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
#   vllm/models/deepseek_v41/quant_rope.py (blob aa848699e3a115f773e1c44efe81244cc7dc7210)
# TensorFold changes (kernel arithmetic verbatim):
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - vllm.triton_utils / fp8_software imports -> package-relative.
# - attention_qdq / apply_rope: x.is_cuda -> on_triton_device(x) (identical in compiled runs; CPU only under TRITON_INTERPRET).
"""SM70 software attention QDQ and V4.1 interleaved rotary embeddings."""

import math

import torch

from .fp8_software import fp8_e4m3fn_bits_to_fp32_bitcast, fp32_to_fp8_e4m3fn_bits
from .triton_utils import jit, on_triton_device, tl, triton


# Official DeepSeek-V4.1-Flash-1017: one UE8M0 scale per 32 window or attention-output channels.
ATTENTION_FP8_GROUP = 32


@jit
def _ceil_power_two(x):
    bits = x.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23).to(tl.int32) - 127
    exponent += ((bits & 0x7FFFFF) != 0).to(tl.int32)
    return ((exponent + 127) << 23).to(tl.float32, bitcast=True)


@jit
def _fp4_value(x):
    a = tl.abs(x)
    # E2M1 round-to-nearest-even at all seven midpoints.
    value = tl.where(a > 0.25, 0.5, 0.0)
    value = tl.where(a >= 0.75, 1.0, value)
    value = tl.where(a > 1.25, 1.5, value)
    value = tl.where(a >= 1.75, 2.0, value)
    value = tl.where(a > 2.5, 3.0, value)
    value = tl.where(a >= 3.5, 4.0, value)
    value = tl.where(a > 5.0, 6.0, value)
    return (
        value.to(tl.uint32, bitcast=True) | (x.to(tl.uint32, bitcast=True) & 0x80000000)
    ).to(tl.float32, bitcast=True)


@jit
def _qdq(x, out, GROUP: tl.constexpr, FP4: tl.constexpr, E4M3_SCALE: tl.constexpr):
    group = tl.program_id(0)
    i = tl.arange(0, GROUP)
    value = tl.load(x + group * GROUP + i).to(tl.float32)
    amax = tl.max(tl.abs(value), 0)
    if FP4:
        if E4M3_SCALE:
            # Keep an all-zero group's scale nonzero, exactly as training.
            amax = tl.maximum(amax, 6.0 * (2.0**-9))
            scale = fp8_e4m3fn_bits_to_fp32_bitcast(
                fp32_to_fp8_e4m3fn_bits(tl.div_rn(amax, 6.0))
            )
        else:
            amax = tl.maximum(amax, 6.0 * (2.0**-126))
            scale = _ceil_power_two(amax * (1.0 / 6.0))
        quant = _fp4_value(tl.div_rn(value, scale))
    else:
        amax = tl.maximum(amax, 1e-4)
        scale = _ceil_power_two(amax * (1.0 / 448.0))
        quant = fp8_e4m3fn_bits_to_fp32_bitcast(
            fp32_to_fp8_e4m3fn_bits(tl.div_rn(value, scale))
        )
    tl.store(out + group * GROUP + i, quant * scale)


def attention_qdq(x, *, fp4=False, group=128, e4m3_scale=False, inplace=False):
    """Carry the official quantize/dequantize values in FP16 cache storage."""
    if (
        not on_triton_device(x)
        or x.dtype != torch.float16
        or not x.is_contiguous()
        or x.ndim < 2
        or x.shape[-1] % group
        or (fp4 and (group, e4m3_scale) not in ((16, True), (32, False)))
        or (not fp4 and (group not in (ATTENTION_FP8_GROUP, 128) or e4m3_scale))
    ):
        raise ValueError("Unsupported V4.1 FP16 attention QDQ geometry")
    result = x if inplace else torch.empty_like(x)
    _qdq[(x.numel() // group,)](
        x,
        result,
        GROUP=group,
        FP4=fp4,
        E4M3_SCALE=e4m3_scale,
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result


def rope_table(max_length=16384, *, compressed=False, device="cuda", rope=None):
    """Checkpoint YaRN applies to compressed layers even below 16K tokens."""
    if rope is None:
        from ..weights import Geometry

        rope = Geometry().rope_parameters()
    dim, base = 64, rope.compressed_theta if compressed else rope.window_theta
    frequencies = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if compressed:
        original, factor = rope.original_context, rope.factor

        def corrected(rotations):
            return (
                dim
                * math.log(original / (rotations * 2 * math.pi))
                / (2 * math.log(base))
            )

        low, high = (
            max(math.floor(corrected(rope.beta_fast)), 0),
            min(math.ceil(corrected(rope.beta_slow)), dim - 1),
        )
        ramp = (
            (torch.arange(dim // 2, dtype=torch.float32) - low) / max(high - low, 1e-3)
        ).clamp(0, 1)
        smooth = 1 - ramp
        frequencies = frequencies / factor * (1 - smooth) + frequencies * smooth
    angles = torch.outer(torch.arange(max_length), frequencies)
    phase = torch.polar(torch.ones_like(angles), angles)
    table = torch.cat((phase.real, phase.imag), dim=-1)
    # Final row: the frequencies. Positions beyond the table rotate on the fly
    # (same FP32 position * frequency product as torch.outer).
    tail = torch.zeros(1, 64, dtype=torch.float32)
    tail[0, :32] = frequencies
    return torch.cat((table, tail)).to(device)


@jit
def _rotate(
    x,
    positions,
    table,
    out,
    LENGTH,
    HEADS: tl.constexpr,
    D: tl.constexpr,
    INVERSE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    token, head = tl.program_id(0), tl.program_id(1)
    col = tl.arange(0, BLOCK)
    base = (token * HEADS + head) * D
    nonrotary = col < D - 64
    tl.store(out + base + col, tl.load(x + base + col, nonrotary, 0), nonrotary)
    i = tl.arange(0, 32)
    position = tl.load(positions + token).to(tl.int64)
    if position < LENGTH:
        c, s = (
            tl.load(table + position * 64 + i),
            tl.load(table + position * 64 + 32 + i),
        )
    else:
        angle = position.to(tl.float32) * tl.load(table + LENGTH * 64 + i)
        c, s = tl.cos(angle), tl.sin(angle)
    if INVERSE:
        s = -s
    a = tl.load(x + base + D - 64 + 2 * i).to(tl.float32)
    b = tl.load(x + base + D - 64 + 2 * i + 1).to(tl.float32)
    tl.store(out + base + D - 64 + 2 * i, a * c - b * s)
    tl.store(out + base + D - 64 + 2 * i + 1, a * s + b * c)


def apply_rope(x, positions, table, *, inverse=False, inplace=False):
    original_shape = x.shape
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if (
        x.ndim != 3
        or x.dtype != torch.float16
        or not on_triton_device(x)
        or x.shape[-1] not in (128, 512)
        or not x.is_contiguous()
        or positions.shape != (x.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or positions.device != x.device
        or table.device != x.device
        or table.ndim != 2
        or table.shape[1] != 64
        or table.dtype != torch.float32
    ):
        raise ValueError("V4.1 RoPE requires FP16 values and the FP32 64-dim table")
    result = x if inplace else torch.empty_like(x)
    _rotate[(x.shape[0], x.shape[1])](
        x,
        positions,
        table,
        result,
        table.shape[0] - 1,
        HEADS=x.shape[1],
        D=x.shape[-1],
        INVERSE=inverse,
        BLOCK=triton.next_power_of_2(x.shape[-1]),
        num_warps=4,
        enable_fp_fusion=False,
    )
    return result.view(original_shape)

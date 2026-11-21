# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
#   vllm/models/deepseek_v4/common/ops/fp8_software.py (blob e4c4e348af3bf577dc610d25f115e727e143c062)
# TensorFold changes:
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - vllm.triton_utils -> triton_utils.
# - Triton-interpreter (CPU test) form of _round_nearest_even_i32 only: inline PTX is unsupported there, so
#   (x + 1.5 * 2**23) - 1.5 * 2**23 in FP32 rounds to nearest even, exact for |x| < 2**22. Every caller passes
#   |x| <= 448 * 512 or a mantissa * 8 <= 16. Compiled runs keep the verbatim cvt.rni PTX.

"""Software E4M3FN conversion helpers for GPUs without FP8 instructions."""

from .triton_utils import INTERPRET, jit, tl

if INTERPRET:

    @jit
    def _round_nearest_even_i32(x):
        return ((x + 12582912.0) - 12582912.0).to(tl.int32)

else:

    @jit
    def _round_nearest_even_i32(x):
        return tl.inline_asm_elementwise(
            "cvt.rni.s32.f32 $0, $1;",
            constraints="=r,f",
            args=[x],
            dtype=tl.int32,
            is_pure=True,
            pack=1,
        )


@jit
def fp8_e4m3fn_bits_to_fp32(bits):
    """Decode E4M3FN bytes using scalar arithmetic available on SM70."""
    value_bits = bits.to(tl.int32)
    sign = tl.where((value_bits & 0x80) != 0, -1.0, 1.0)
    exponent = (value_bits >> 3) & 0x0F
    mantissa = value_bits & 0x07
    subnormal = mantissa.to(tl.float32) * (2.0**-9)
    normal = (1.0 + mantissa.to(tl.float32) * 0.125) * tl.exp2(
        exponent.to(tl.float32) - 7.0
    )
    decoded = tl.where(exponent == 0, subnormal, normal) * sign
    return tl.where((exponent == 15) & (mantissa == 7), float("nan"), decoded)


@jit
def fp8_e4m3fn_bits_to_fp32_bitcast(bits):
    """Decode E4M3FN bytes by constructing exact IEEE FP32 normal values."""
    value_bits = bits.to(tl.int32)
    exponent = (value_bits >> 3) & 0x0F
    mantissa = value_bits & 0x07
    sign_bits = (value_bits & 0x80) << 24
    normal_bits = sign_bits | ((exponent + 120) << 23) | (mantissa << 20)
    normal = normal_bits.to(tl.float32, bitcast=True)

    sign = tl.where((value_bits & 0x80) != 0, -1.0, 1.0)
    subnormal = mantissa.to(tl.float32) * (2.0**-9) * sign
    decoded = tl.where(exponent == 0, subnormal, normal)
    return tl.where((exponent == 15) & (mantissa == 7), float("nan"), decoded)


@jit
def fp32_to_fp8_e4m3fn_bits(x):
    """Encode finite FP32 values to saturated E4M3FN round-to-nearest-even."""
    sign_bit = (x.to(tl.int32, bitcast=True) >> 24) & 0x80
    value = tl.minimum(tl.abs(x), 448.0)

    subnormal_mantissa = _round_nearest_even_i32(value * 512.0)
    # A rounded value of eight is the minimum normal encoding (0x08).
    subnormal_mantissa = tl.minimum(tl.maximum(subnormal_mantissa, 0), 8)

    safe_value = tl.maximum(value, 2.0**-126)
    exponent_unbiased = tl.floor(tl.log2(safe_value)).to(tl.int32)
    exponent_scale = tl.exp2(exponent_unbiased.to(tl.float32))
    mantissa = _round_nearest_even_i32((safe_value / exponent_scale - 1.0) * 8.0)
    carry = mantissa >= 8
    exponent = exponent_unbiased + 7 + carry.to(tl.int32)
    mantissa = tl.where(carry, 0, mantissa)

    overflow = (exponent > 15) | ((exponent == 15) & (mantissa > 6))
    exponent = tl.minimum(tl.maximum(exponent, 1), 15)
    mantissa = tl.minimum(tl.maximum(mantissa, 0), 7)
    normal_bits = (exponent << 3) | mantissa
    normal_bits = tl.where(overflow, 0x7E, normal_bits)

    magnitude_bits = tl.where(value < (2.0**-6), subnormal_mantissa, normal_bits)
    magnitude_bits = tl.where(value == 0.0, 0, magnitude_bits)
    return (sign_bit | magnitude_bits).to(tl.uint8)

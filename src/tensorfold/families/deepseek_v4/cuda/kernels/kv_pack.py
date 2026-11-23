# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
#   vllm/models/deepseek_v41/kv_pack.py (blob 6c5dca12573be09893aa90358835d47fcabab537)
# TensorFold changes (kernel arithmetic verbatim):
# - @triton.jit -> triton_utils.jit: the same JITFunction; launches add the SM70 Triton 3.2 profile options
#   (ptx_version=83, needed with CUDA 12.9 ptxas). Kernel arithmetic unchanged.
# - vllm.triton_utils / fp8_software imports -> package-relative.
"""Packed V4.1 KV rows.

The checkpoint stores QDQ values in the caches. Compressed latents use E2M1
groups of 16 with an E4M3 scale; indexer keys use E2M1 groups of 32 with a
power-of-two scale. Keeping the codes and scales reproduces the FP16 cache
values bit for bit at a fraction of the bytes:

  compressed row (512): 256 B nibbles + 32 B E4M3 scales  = 288 B
  index row      (128):  64 B nibbles +  4 B scale exponents = 68 B

The optional SM70 RotorQuant window and compressed latents share codec 11's
layout (256 B nibbles + eight FP16 scales), in its rotated basis.
"""

import torch

from tensorfold.families.qwen4_exp.cuda.rotorquant_ref import CENTROIDS_4

from .fp8_software import fp8_e4m3fn_bits_to_fp32_bitcast, fp32_to_fp8_e4m3fn_bits
from .triton_utils import jit, tl

C_DIM, C_GROUP, C_ROW = 512, 16, 288
I_DIM, I_GROUP, I_ROW = 128, 32, 68
ROTOR_ROW = C_DIM // 2 + (C_DIM // 64) * 2
ROTOR_CENTROIDS = tl.constexpr(CENTROIDS_4)


@jit
def _ceil_power_two_bits(x):
    """Biased exponent of the smallest power of two >= x (x > 0, normal)."""
    bits = x.to(tl.uint32, bitcast=True)
    exponent = (bits >> 23).to(tl.int32)
    return exponent + ((bits & 0x7FFFFF) != 0).to(tl.int32)


@jit
def _fp4_code(x):
    """E2M1 code (sign << 3 | magnitude index) with the _fp4_value rounding."""
    a = tl.abs(x)
    m = tl.where(a > 0.25, 1, 0)
    m = tl.where(a >= 0.75, 2, m)
    m = tl.where(a > 1.25, 3, m)
    m = tl.where(a >= 1.75, 4, m)
    m = tl.where(a > 2.5, 5, m)
    m = tl.where(a >= 3.5, 6, m)
    m = tl.where(a > 5.0, 7, m)
    sign = ((x.to(tl.uint32, bitcast=True) >> 31) & 1).to(tl.int32)
    return (sign << 3) | m


@jit
def _fp4_decode(code):
    m = code & 7
    value = tl.where(m == 1, 0.5, 0.0)
    value = tl.where(m == 2, 1.0, value)
    value = tl.where(m == 3, 1.5, value)
    value = tl.where(m == 4, 2.0, value)
    value = tl.where(m == 5, 3.0, value)
    value = tl.where(m == 6, 4.0, value)
    value = tl.where(m == 7, 6.0, value)
    sign = ((code >> 3) & 1).to(tl.uint32) << 31  # keeps -0.0 like _fp4_value
    return (value.to(tl.uint32, bitcast=True) | sign).to(tl.float32, bitcast=True)


@jit
def pack_c_group(x_ptr, row_ptr, group):
    """Pack group `group` (16 values from x_ptr) into a 288-byte row."""
    i = tl.arange(0, 16)
    value = tl.load(x_ptr + group * 16 + i).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), 0), 6.0 * (2.0**-9))
    scale_bits = fp32_to_fp8_e4m3fn_bits(tl.div_rn(amax, 6.0))
    scale = fp8_e4m3fn_bits_to_fp32_bitcast(scale_bits)
    code = _fp4_code(tl.div_rn(value, scale))
    pair = tl.reshape(code, (8, 2))
    lo, hi = tl.split(pair)
    tl.store(row_ptr + group * 8 + tl.arange(0, 8), (lo | (hi << 4)).to(tl.uint8))
    tl.store(row_ptr + 256 + group, scale_bits)


@jit
def unpack_c_row(row_ptr):
    """[512] FP16 values of a packed compressed row (fp32 product -> fp16)."""
    j = tl.arange(0, 512)
    byte = tl.load(row_ptr + j // 2).to(tl.int32)
    code = tl.where(j % 2 == 0, byte & 15, (byte >> 4) & 15)
    scale = fp8_e4m3fn_bits_to_fp32_bitcast(tl.load(row_ptr + 256 + j // 16))
    return (_fp4_decode(code) * scale).to(tl.float16)


@jit
def unpack_rotor_row(row_ptr, valid):
    """Codec 11 decoded to FP16 in its rotated basis (tf_rq4::half_group_scale)."""
    j = tl.arange(0, C_DIM)
    byte = tl.load(row_ptr + j // 2, valid, 0).to(tl.int32)
    code = (byte >> (4 * (j % 2))) & 15
    half_ptr = row_ptr + C_DIM // 2 + 2 * (j // 64)
    lo = tl.load(half_ptr, valid, 0).to(tl.uint16)
    hi = tl.load(half_ptr + 1, valid, 0).to(tl.uint16)
    scale = (lo | (hi << 8)).to(tl.uint16).to(tl.float16, bitcast=True).to(tl.float32)
    # The pinned V100 Triton hashes every branch of a JIT dependency. Select
    # only from the Q4 table here, without the newer DSL's eight-bit gather.
    centroid = tl.full((C_DIM,), ROTOR_CENTROIDS[15], tl.float32)
    for index in tl.static_range(15):
        centroid = tl.where(code == index, ROTOR_CENTROIDS[index], centroid)
    value = (centroid * scale).to(tl.float16)
    return tl.where(valid, value, tl.full((C_DIM,), 0, tl.float16))


@jit
def pack_i_group(x_ptr, row_ptr, group):
    i = tl.arange(0, 32)
    value = tl.load(x_ptr + group * 32 + i).to(tl.float32)
    amax = tl.maximum(tl.max(tl.abs(value), 0), 6.0 * (2.0**-126))
    exponent = _ceil_power_two_bits(amax * (1.0 / 6.0))
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    code = _fp4_code(tl.div_rn(value, scale))
    pair = tl.reshape(code, (16, 2))
    lo, hi = tl.split(pair)
    tl.store(row_ptr + group * 16 + tl.arange(0, 16), (lo | (hi << 4)).to(tl.uint8))
    tl.store(row_ptr + 64 + group, exponent.to(tl.uint8))


@jit
def unpack_i_row(row_ptr):
    j = tl.arange(0, 128)
    byte = tl.load(row_ptr + j // 2).to(tl.int32)
    code = tl.where(j % 2 == 0, byte & 15, (byte >> 4) & 15)
    exponent = tl.load(row_ptr + 64 + j // 32).to(tl.int32)
    scale = (exponent << 23).to(tl.float32, bitcast=True)
    return (_fp4_decode(code) * scale).to(tl.float16)


@jit
def _pack_rows(
    x,
    out,
    ROW: tl.constexpr,
    DIM: tl.constexpr,
    GROUP: tl.constexpr,
    COMPRESSED: tl.constexpr,
):
    row, group = tl.program_id(0).to(tl.int64), tl.program_id(1)
    if COMPRESSED:
        pack_c_group(x + row * DIM, out + row * ROW, group)
    else:
        pack_i_group(x + row * DIM, out + row * ROW, group)


@jit
def _unpack_rows(
    rows, out, ROW: tl.constexpr, DIM: tl.constexpr, COMPRESSED: tl.constexpr
):
    row = tl.program_id(0).to(tl.int64)
    if COMPRESSED:
        tl.store(out + row * DIM + tl.arange(0, DIM), unpack_c_row(rows + row * ROW))
    else:
        tl.store(out + row * DIM + tl.arange(0, DIM), unpack_i_row(rows + row * ROW))


def pack_rows(x, compressed):
    """Reference packer for [N, 512] / [N, 128] pre-QDQ FP16 rows (tests)."""
    dim, group, size = (
        (C_DIM, C_GROUP, C_ROW) if compressed else (I_DIM, I_GROUP, I_ROW)
    )
    assert x.dtype == torch.float16 and x.is_contiguous() and x.shape[1] == dim
    out = torch.empty(x.shape[0], size, dtype=torch.uint8, device=x.device)
    _pack_rows[(x.shape[0], dim // group)](
        x, out, ROW=size, DIM=dim, GROUP=group, COMPRESSED=compressed
    )
    return out


def unpack_rows(rows, compressed):
    dim, size = (C_DIM, C_ROW) if compressed else (I_DIM, I_ROW)
    out = torch.empty(rows.shape[0], dim, dtype=torch.float16, device=rows.device)
    _unpack_rows[(rows.shape[0],)](rows, out, ROW=size, DIM=dim, COMPRESSED=compressed)
    return out

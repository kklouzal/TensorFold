"""Independent cache sides, selected once by scalar shader traits.

Each side owns its physical byte stride and metadata precision. Decoding keeps
the stored basis; the key basis belongs to Q and the value basis to the merged
output. Failed-frame sentinels require the owning State's status check before
publication. Native FP16 underflow remains an accepted legacy approximation.
"""

from __future__ import annotations

import triton
import triton.language as tl
import torch

from . import kvquant, rotorquant_kernel as rotor


def resolve_pair(bits, codec, k_bits, k_codec, v_bits, v_codec, head_dim):
    """Validate shader widths (BF16 is zero) and return the ordered scalar plan."""

    fields = (k_bits, k_codec, v_bits, v_codec)
    if all(field is None for field in fields):
        fields = (bits, codec, bits, codec)
    elif any(field is None for field in fields):
        raise ValueError("independent KV shaders require all four key/value bit and codec fields")
    if any(type(field) is not int for field in fields):
        raise ValueError("KV shader bits and codecs must be integers")
    if type(head_dim) is not int or head_dim <= 0 or head_dim & (head_dim - 1):
        raise ValueError("KV shaders require a positive power-of-two head dimension")
    for width, family in (fields[:2], fields[2:]):
        if family == 0:
            valid = width == 0 or (width in (4, 8) and head_dim % 32 == 0)
        else:
            valid = (head_dim % 128 == 0 and (
                (width == 3 and family in (1, 2)) or (width == 4 and family in (1, 2, 3, 4, 5))
                or (width == 6 and family in (4, 6, 7)) or (width == 7 and family == 4)
                or (width == 8 and family in (4, 8))))
        if not valid:
            raise ValueError(f"unsupported KV shader bits={width}, codec={family}")
    return fields


def validate_storage(data, scale, capacity, heads, head_dim, bits, codec, device):
    """Check new pair tensor ABI without reading GPU values or allocating memory."""

    width = head_dim if bits == 0 else head_dim * bits // 8
    dtype = torch.bfloat16 if bits == 0 else torch.uint8 if codec or bits != 8 else torch.int8
    if (data.shape != (capacity, heads, width) or data.dtype != dtype or data.device != device
            or not data.is_contiguous()):
        raise ValueError("KV side payload differs from its contiguous per-side shader layout")
    if bits:
        group = 128 if codec else 32
        dtype = torch.float32 if codec else torch.float16
        if (scale is None or scale.shape != (capacity, heads, head_dim // group) or scale.dtype != dtype
                or scale.device != device or not scale.is_contiguous()):
            raise ValueError("KV side metadata differs from its own group, precision or contiguous layout")


@triton.jit
def canonical_bf16_operand(x):
    """Preserve every BF16 bit and make its native tensor-core layout boundary explicit."""

    bits = x.to(tl.uint16, bitcast=True)
    bits = tl.inline_asm_elementwise("mov.b16 $0, $1;", "=h,h", [bits], dtype=tl.uint16,
                                    is_pure=False, pack=1)
    return bits.to(tl.bfloat16, bitcast=True)


@triton.jit
def source_guard(x, STATUS, error, LAYER: tl.constexpr):
    valid = (x == x) & (tl.abs(x) <= 2**30)
    if tl.sum((~valid).to(tl.int32), axis=0) != 0:
        tl.atomic_or(STATUS, error)
        tl.atomic_cas(STATUS + 1, -2, LAYER)
    return tl.where(valid, x, 0.0)


@triton.jit
def scale_guard(scale, x, STATUS, LAYER: tl.constexpr, CODEC: tl.constexpr):
    """Check the actual stored precision; keep failed operations finite and poisoned."""

    if CODEC:
        limit: tl.constexpr = (128 if CODEC == 8 else 32 if CODEC == 6 or CODEC == 7 else 8) * 2**30
        nonzero = tl.max(tl.abs(x), axis=1) > 0.0
        bad = (scale != scale) | (scale < 0.0) | (scale > limit) | ((scale == 0.0) & nonzero)
    else:
        # scale has already been rounded to FP16 by the native encoder.
        bad = (scale != scale) | (scale < 0.0) | (scale > 65504.0)
    if tl.sum(bad.to(tl.int32), axis=0) != 0:
        tl.atomic_or(STATUS, 8)
        tl.atomic_cas(STATUS + 1, -2, LAYER)
    return tl.where(bad, 0.0, scale)


@triton.jit
def transform(x, M: tl.constexpr, D: tl.constexpr, BITS: tl.constexpr, CODEC: tl.constexpr,
              INVERSE: tl.constexpr = False):
    if CODEC:
        return rotor.rotate(x, M=M, W=D, CODEC=CODEC, INVERSE=INVERSE)
    elif BITS:
        return tl.reshape(kvquant.h32(tl.reshape(x, (M * D // 32, 32)), M=M * D // 32), (M, D))
    return x


@triton.jit
def store_side(x, DATA, SCALE, slot, STATUS, D: tl.constexpr, BITS: tl.constexpr, CODEC: tl.constexpr,
               VALIDATE: tl.constexpr, LAYER: tl.constexpr):
    """Store one BF16-source head using only this side's packed layout and metadata."""

    if CODEC:
        M: tl.constexpr = D // 128
        groups = tl.arange(0, M)
        block = tl.reshape(x.to(tl.float32), (M, 128))
        if BITS == 3:
            low, high, scale = rotor.quant_groups_3(block, M=M, CODEC=CODEC)
            base = slot * (D * 3 // 8) + groups[:, None] * 48
            tl.store(DATA + base + tl.arange(0, 32)[None, :], low)
            tl.store(DATA + base + 32 + tl.arange(0, 16)[None, :], high)
        elif BITS == 6:
            low, high, scale = rotor.quant_groups_6(block, M=M, CODEC=CODEC)
            base = slot * (D * 3 // 4) + groups[:, None] * 96
            tl.store(DATA + base + tl.arange(0, 64)[None, :], low)
            tl.store(DATA + base + 64 + tl.arange(0, 32)[None, :], high)
        elif BITS == 7:
            low, middle, high, scale = rotor.quant_groups_7(block, M=M, CODEC=CODEC)
            base = slot * (D * 7 // 8) + groups[:, None] * 112
            tl.store(DATA + base + tl.arange(0, 64)[None, :], low)
            tl.store(DATA + base + 64 + tl.arange(0, 32)[None, :], middle)
            tl.store(DATA + base + 96 + tl.arange(0, 16)[None, :], high)
        elif BITS == 8:
            code, scale = rotor.quant_groups_8(block, M=M, CODEC=CODEC)
            base = slot * D + groups[:, None] * 128
            tl.store(DATA + base + tl.arange(0, 128)[None, :], code)
        else:
            tl.static_assert(BITS == 4)
            code, scale = rotor.quant_groups_4(block, M=M, CODEC=CODEC)
            base = slot * (D // 2) + groups[:, None] * 64
            tl.store(DATA + base + tl.arange(0, 64)[None, :], code)
        if VALIDATE:
            scale = scale_guard(scale, block, STATUS, LAYER, CODEC)
        tl.store(SCALE + slot * M + groups, scale)
    elif BITS:
        M: tl.constexpr = D // 32
        groups = tl.arange(0, M)
        block = tl.reshape(x.to(tl.float32), (M, 32))
        if BITS == 4:
            code, scale = kvquant.quant_groups_4(block, M=M)
            tl.store(DATA + slot * (D // 2) + groups[:, None] * 16 + tl.arange(0, 16)[None, :], code)
        else:
            tl.static_assert(BITS == 8)
            code, scale = kvquant.quant_groups_8(block, M=M)
            tl.store(DATA + slot * D + groups[:, None] * 32 + tl.arange(0, 32)[None, :], code)
        if VALIDATE:
            scale = scale_guard(scale, block, STATUS, LAYER, CODEC)
        tl.store(SCALE + slot * M + groups, scale)
    else:
        tl.store(DATA + slot * D + tl.arange(0, D), x.to(tl.bfloat16))


@triton.jit
def load_side(DATA, SCALE, ki, valid, hk, HK: tl.constexpr, D: tl.constexpr, M: tl.constexpr,
              BITS: tl.constexpr, CODEC: tl.constexpr):
    """Decode [M,D] in the stored basis; invalid rows return zero without out-of-range loads."""

    row = ki.to(tl.int64) * HK + hk
    if CODEC:
        group = tl.arange(0, D // 128)
        scale = tl.load(SCALE + row[:, None] * (D // 128) + group[None, :],
                        mask=valid[:, None], other=0.0)
        if BITS == 3:
            low = tl.arange(0, D // 4)
            high = tl.arange(0, D // 8)
            base = row[:, None] * (D * 3 // 8)
            lo = tl.load(DATA + base + (low[None, :] // 32) * 48 + low[None, :] % 32,
                         mask=valid[:, None], other=0)
            hi = tl.load(DATA + base + (high[None, :] // 16) * 48 + 32 + high[None, :] % 16,
                         mask=valid[:, None], other=0)
            out = rotor.dequant_group_3(lo, hi, scale, M=M, W=D)
        elif BITS == 6:
            low = tl.arange(0, D // 2)
            high = tl.arange(0, D // 4)
            base = row[:, None] * (D * 3 // 4)
            lo = tl.load(DATA + base + (low[None, :] // 64) * 96 + low[None, :] % 64,
                         mask=valid[:, None], other=0)
            hi = tl.load(DATA + base + (high[None, :] // 32) * 96 + 64 + high[None, :] % 32,
                         mask=valid[:, None], other=0)
            out = rotor.dequant_group_6(lo, hi, scale, M=M, W=D)
        elif BITS == 7:
            low = tl.arange(0, D // 2)
            middle = tl.arange(0, D // 4)
            high = tl.arange(0, D // 8)
            base = row[:, None] * (D * 7 // 8)
            lo = tl.load(DATA + base + (low[None, :] // 64) * 112 + low[None, :] % 64,
                         mask=valid[:, None], other=0)
            mid = tl.load(DATA + base + (middle[None, :] // 32) * 112 + 64 + middle[None, :] % 32,
                          mask=valid[:, None], other=0)
            hi = tl.load(DATA + base + (high[None, :] // 16) * 112 + 96 + high[None, :] % 16,
                         mask=valid[:, None], other=0)
            out = rotor.dequant_group_7(lo, mid, hi, scale, M=M, W=D)
        elif BITS == 8:
            code = tl.load(DATA + row[:, None] * D + tl.arange(0, D)[None, :],
                           mask=valid[:, None], other=0)
            out = rotor.dequant_group_8(code, scale, M=M, W=D)
        else:
            tl.static_assert(BITS == 4)
            code = tl.load(DATA + row[:, None] * (D // 2) + tl.arange(0, D // 2)[None, :],
                           mask=valid[:, None], other=0)
            out = rotor.dequant_group_4(code, scale, M=M, W=D)
    elif BITS:
        group = tl.arange(0, D // 32)
        scale = tl.load(SCALE + row[:, None] * (D // 32) + group[None, :],
                        mask=valid[:, None], other=0.0)
        if BITS == 4:
            code = tl.load(DATA + row[:, None] * (D // 2) + tl.arange(0, D // 2)[None, :],
                           mask=valid[:, None], other=0)
            out = kvquant.dequant_group_4(code, scale, M=M, W=D)
        else:
            tl.static_assert(BITS == 8)
            code = tl.load(DATA + row[:, None] * D + tl.arange(0, D)[None, :],
                           mask=valid[:, None], other=0)
            out = kvquant.dequant_group_8(code, scale, M=M, W=D)
    else:
        out = tl.load(DATA + row[:, None] * D + tl.arange(0, D)[None, :],
                      mask=valid[:, None], other=0.0)
    return canonical_bf16_operand(out)

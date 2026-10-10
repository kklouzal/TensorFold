"""Packed RotorQuant-family KV primitives for Flash Next's CUDA attention.

This is a TensorFold-owned implementation of fixed orthogonal block rotations
and Gaussian Lloyd-Max scalar quantization, not an import of the upstream
prototype.  The immutable tables and the independent reference live in
``rotorquant_ref``.  CODEC 1 is the periodic four-angle Planar transform;
CODEC 2 is left multiplication by the unit quaternion (1/2, 1/2, 1/2, 1/2).
CODECs 3/5 mix within 64 coordinates with three quaternion stages and preserve
the reconstructed 128-coordinate group's norm. CODEC 4 adds a final Givens stage
to mix all 128 coordinates and stores the original RMS. CODECs 4/5 apply the
pinned pair-sign preconditioner before rotation and after inverse rotation.
CODECs 6/7 share CODEC 4's exact basis and use six-bit norm-corrected scales;
CODEC 7 also contracts coding coordinates to the outer centroid when needed.
CODEC 8 uses the same signed Iso128 basis and norm-corrected eight-bit metadata.

Quantization normalizes each 128-coordinate group by its original FP32 RMS.
Three-bit groups store 32 low-two-bit bytes followed by 16 high-bit bytes;
four-bit groups store 64 bytes, even coordinate in the low nibble; six-bit groups
store 64 low-nibble bytes followed by 32 high-two-bit bytes. RMS is
stored separately as FP32; CODECs 3/5/6/7 multiply it by the reciprocal RMS of the
selected centroid vector. Exact zero groups have scale zero and the lower
central centroid index.  Attention keeps decoded K/V in the rotated frame;
only the final weighted value sum is inverse-rotated.

Callers preselect CODEC and width and validate shapes.  Explicit CUDA rounding
intrinsics preserve staged FP32 codec arithmetic independently of a caller's
compiler fusion policy; preexisting model/softmax arithmetic can stay unchanged.
No allocation, host synchronization, device selection or mutable state occurs
inside these helpers.  Inputs are bounded finite model tensors; metadata and
payload ownership, stream ordering and graph lifetimes belong to KVCache.
"""

from __future__ import annotations

import triton
import triton.language as tl

from .rotorquant_ref import (
    CENTROIDS_3,
    CENTROIDS_4,
    CENTROIDS_6,
    CENTROIDS_7,
    CENTROIDS_8,
    ISO_QUATERNION,
    PLANAR_COS,
    PLANAR_SIN,
    SIGN_PAIR_NEGATIVE_MASK,
    THRESHOLDS_3,
    THRESHOLDS_4,
    THRESHOLDS_6,
    THRESHOLDS_7,
    THRESHOLDS_8,
    WIDE_GIVENS_COS,
    WIDE_GIVENS_SIN,
)

# Capture the canonical actual FP32 values as compile-time constants.  A seed
# alone would not establish the same basis/codebook across independent paths.
_C3 = tl.constexpr(CENTROIDS_3)
_C4 = tl.constexpr(CENTROIDS_4)
_C6 = tl.constexpr(CENTROIDS_6)
_C7 = tl.constexpr(CENTROIDS_7)
_C8 = tl.constexpr(CENTROIDS_8)
_T3 = tl.constexpr(THRESHOLDS_3)
_T4 = tl.constexpr(THRESHOLDS_4)
_T6 = tl.constexpr(THRESHOLDS_6)
_BINARY_T6 = tl.constexpr(tuple(tuple(THRESHOLDS_6[2 * step * cell + step - 1]
                                      for cell in range(64 // (2 * step)))
                               for step in (32, 16, 8, 4, 2, 1)))
_BINARY_T7 = tl.constexpr(tuple(tuple(THRESHOLDS_7[2 * step * cell + step - 1]
                                      for cell in range(128 // (2 * step)))
                               for step in (64, 32, 16, 8, 4, 2, 1)))
_BINARY_T8 = tl.constexpr(tuple(tuple(THRESHOLDS_8[2 * step * cell + step - 1]
                                      for cell in range(256 // (2 * step)))
                               for step in (128, 64, 32, 16, 8, 4, 2, 1)))
_COS = tl.constexpr(PLANAR_COS)
_SIN = tl.constexpr(PLANAR_SIN)
_QUAT = tl.constexpr(ISO_QUATERNION)
_SIGN_MASK = tl.constexpr(SIGN_PAIR_NEGATIVE_MASK)
_WIDE_COS = tl.constexpr(WIDE_GIVENS_COS)
_WIDE_SIN = tl.constexpr(WIDE_GIVENS_SIN)


@triton.jit
def _mul_rn(a, b):
    # Explicit .rn prevents fusion; omitting .ftz preserves finite subnormals.
    # Triton's default libdevice reflect-ftz policy would flush these operands.
    return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _add_rn(a, b):
    return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sub_rn(a, b):
    return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", constraints="=f,f,f", args=[a, b],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _sqrt_rn(a):
    return tl.inline_asm_elementwise("sqrt.rn.f32 $0, $1;", constraints="=f,f", args=[a],
                                     dtype=tl.float32, is_pure=True, pack=1)


@triton.jit
def _quaternion_stage(x, M: tl.constexpr, W: tl.constexpr,
                      LOW: tl.constexpr, INVERSE: tl.constexpr):
    """Disjoint true unit-quaternion rotations along one coordinate bitpair."""

    tl.static_assert(LOW == 1 or LOW == 4 or LOW == 16)
    tl.static_assert(W % (4 * LOW) == 0)
    tl.static_assert(_QUAT[0] == 0.5 and _QUAT[1] == 0.5 and _QUAT[2] == 0.5 and _QUAT[3] == 0.5)
    ordered = tl.trans(tl.reshape(x, (M, W // (4 * LOW), 4, LOW)), 0, 1, 3, 2)
    ac, bd = tl.split(tl.reshape(ordered, (M * W // 4, 2, 2)))
    a, c = tl.split(ac)
    b, d = tl.split(bd)
    if INVERSE:
        first = _add_rn(_add_rn(_add_rn(a, b), c), d)
        second = _sub_rn(_add_rn(_add_rn(-a, b), c), d)
        third = _add_rn(_add_rn(_sub_rn(-a, b), c), d)
        fourth = _add_rn(_sub_rn(_add_rn(-a, b), c), d)
    else:
        first = _sub_rn(_sub_rn(_sub_rn(a, b), c), d)
        second = _add_rn(_sub_rn(_add_rn(a, b), c), d)
        third = _sub_rn(_add_rn(_add_rn(a, b), c), d)
        fourth = _add_rn(_add_rn(_sub_rn(a, b), c), d)
    # Joining (0,2) with (1,3) yields contiguous (0,1,2,3).
    out = _mul_rn(tl.join(tl.join(first, third), tl.join(second, fourth)), _QUAT[0])
    restored = tl.trans(tl.reshape(out, (M, W // (4 * LOW), LOW, 4)), 0, 1, 3, 2)
    return tl.reshape(restored, (M, W))


@triton.jit
def _pair_signs(x, W: tl.constexpr):
    """Pinned signs repeat every 128 coordinates; no runtime RNG/table state."""

    pair = ((tl.arange(0, W) // 2) % 64).to(tl.uint64)
    negative = (tl.full((W,), _SIGN_MASK, tl.uint64) >> pair) & 1
    return tl.where(negative[None, :] != 0, -x, x)


@triton.jit
def _wide_givens(x, M: tl.constexpr, W: tl.constexpr, INVERSE: tl.constexpr):
    """Final pi/4 rotation across bit6, independently per 128-coordinate group."""

    a, b = tl.split(tl.trans(tl.reshape(x, (M, W // 128, 2, 64)), 0, 1, 3, 2))
    s: tl.constexpr = -_WIDE_SIN if INVERSE else _WIDE_SIN
    first = _sub_rn(_mul_rn(a, _WIDE_COS), _mul_rn(b, s))
    second = _add_rn(_mul_rn(a, s), _mul_rn(b, _WIDE_COS))
    return tl.reshape(tl.trans(tl.join(first, second), 0, 1, 3, 2), (M, W))


@triton.jit
def rotate(x, M: tl.constexpr, W: tl.constexpr, CODEC: tl.constexpr, INVERSE: tl.constexpr = False):
    """FP32 ``[M, W]`` fixed transform; W is a multiple of 128."""

    tl.static_assert(W % 128 == 0)
    tl.static_assert(1 <= CODEC <= 8)
    # Six-bit scale policies share exactly the existing signed Iso128 basis.
    BASIS: tl.constexpr = 4 if CODEC == 6 or CODEC == 7 or CODEC == 8 else CODEC
    x = x.to(tl.float32)
    if BASIS == 1:
        a, b = tl.split(tl.reshape(x, (M, W // 2, 2)))
        pair = tl.arange(0, W // 2) % 4
        c = tl.full((W // 2,), _COS[3], tl.float32)
        s = tl.full((W // 2,), _SIN[3], tl.float32)
        for i in tl.static_range(3):
            c = tl.where(pair == i, _COS[i], c)
            s = tl.where(pair == i, _SIN[i], s)
        if INVERSE:
            s = -s
        first = _sub_rn(_mul_rn(a, c[None, :]), _mul_rn(b, s[None, :]))
        second = _add_rn(_mul_rn(a, s[None, :]), _mul_rn(b, c[None, :]))
        return tl.reshape(tl.join(first, second), (M, W))
    elif BASIS == 2:
        return _quaternion_stage(x, M, W, 1, INVERSE)
    else:
        if INVERSE:
            if BASIS == 4:
                x = _wide_givens(x, M, W, True)
            x = _quaternion_stage(x, M, W, 16, True)
            x = _quaternion_stage(x, M, W, 4, True)
            x = _quaternion_stage(x, M, W, 1, True)
            if BASIS == 4 or BASIS == 5:
                x = _pair_signs(x, W)
        else:
            if BASIS == 4 or BASIS == 5:
                x = _pair_signs(x, W)
            x = _quaternion_stage(x, M, W, 1, False)
            x = _quaternion_stage(x, M, W, 4, False)
            x = _quaternion_stage(x, M, W, 16, False)
            if BASIS == 4:
                x = _wide_givens(x, M, W, False)
        return x


@triton.jit
def _normalized(x, M: tl.constexpr):
    """FP32 ``[M,128]`` -> unit-RMS coordinates and original FP32 RMS.

    The absmax scale prevents squaring large finite inputs from overflowing.
    Substituting one for both zero denominators avoids invalid intermediates;
    the zero group's normalized coordinates and stored RMS are exactly zero.
    """

    a = tl.max(tl.abs(x), axis=1)
    z = tl.math.div_rn(x, tl.where(a > 0.0, a, 1.0)[:, None])
    t = _sqrt_rn(_mul_rn(tl.sum(_mul_rn(z, z), axis=1), 0.0078125))
    unit = tl.math.div_rn(z, tl.where(t > 0.0, t, 1.0)[:, None])
    return unit, _mul_rn(a, t)


@triton.jit
def _index_step_6(x, q, LEVEL: tl.constexpr):
    """One fixed binary-lifting decision; its threshold row is compile-time."""

    step: tl.constexpr = 32 >> LEVEL
    row: tl.constexpr = _BINARY_T6[LEVEL]
    boundary = tl.full(x.shape, row[0], tl.float32)
    for cell in tl.static_range(1, len(row)):
        boundary = tl.where(q >= 2 * step * cell, row[cell], boundary)
    return q + (x > boundary).to(tl.int32) * step


@triton.jit
def _index_step_high(x, q, LEVEL: tl.constexpr, BITS: tl.constexpr):
    """A fresh scoped seven/eight-bit decision with a canonical threshold row."""

    tl.static_assert(BITS == 7 or BITS == 8)
    step: tl.constexpr = (1 << (BITS - 1)) >> LEVEL
    row: tl.constexpr = _BINARY_T7[LEVEL] if BITS == 7 else _BINARY_T8[LEVEL]
    boundary = tl.full(x.shape, row[0], tl.float32)
    for cell in tl.static_range(1, len(row)):
        boundary = tl.where(q >= 2 * step * cell, row[cell], boundary)
    return q + (x > boundary).to(tl.int32) * step


@triton.jit
def _indices(x, BITS: tl.constexpr):
    """Nearest canonical centroid indices; exact midpoint chooses lower."""

    q = tl.full(x.shape, 0, tl.int32)
    if BITS == 3:
        for i in tl.static_range(7):
            q += (x > _T3[i]).to(tl.int32)
    elif BITS == 4:
        for i in tl.static_range(15):
            q += (x > _T4[i]).to(tl.int32)
    elif BITS == 6:
        # Binary lifting over63 boundaries: q is a multiple of2*step before
        # each decision. Exact threshold still chooses its lower centroid.
        q = _index_step_6(x, q, 0)
        q = _index_step_6(x, q, 1)
        q = _index_step_6(x, q, 2)
        q = _index_step_6(x, q, 3)
        q = _index_step_6(x, q, 4)
        q = _index_step_6(x, q, 5)
    else:
        tl.static_assert(BITS == 7 or BITS == 8)
        q = _index_step_high(x, q, 0, BITS)
        q = _index_step_high(x, q, 1, BITS)
        q = _index_step_high(x, q, 2, BITS)
        q = _index_step_high(x, q, 3, BITS)
        q = _index_step_high(x, q, 4, BITS)
        q = _index_step_high(x, q, 5, BITS)
        q = _index_step_high(x, q, 6, BITS)
        if BITS == 8:
            q = _index_step_high(x, q, 7, BITS)
    return q


@triton.jit
def _centroids(q, BITS: tl.constexpr):
    """Unsigned centroid indices -> canonical FP32 coordinate values."""

    if BITS == 3:
        out = tl.full(q.shape, _C3[7], tl.float32)
        for i in tl.static_range(7):
            out = tl.where(q == i, _C3[i], out)
    elif BITS == 4:
        out = tl.full(q.shape, _C4[15], tl.float32)
        for i in tl.static_range(15):
            out = tl.where(q == i, _C4[i], out)
    elif BITS == 6:
        out = tl.full(q.shape, _C6[63], tl.float32)
        for i in tl.static_range(63):
            out = tl.where(q == i, _C6[i], out)
    elif BITS == 7:
        out = tl.full(q.shape, _C7[127], tl.float32)
        for i in tl.static_range(127):
            out = tl.where(q == i, _C7[i], out)
    else:
        tl.static_assert(BITS == 8)
        # The canonical constant table is shared by all rows. Register gather
        # avoids expanding 255 selections for every decoded tile coordinate.
        coordinate = tl.arange(0, 256)
        table = tl.full((256,), _C8[255], tl.float32)
        for i in tl.static_range(255):
            table = tl.where(coordinate == i, _C8[i], table)
        out = tl.gather(tl.broadcast_to(table[None, :], (q.shape[0], 256)), q, 1)
    return out


@triton.jit
def quant_groups_3(x, M: tl.constexpr, CODEC: tl.constexpr):
    """``[M,128]`` -> low plane ``[M,32]``, high plane ``[M,16]``, RMS ``[M]``."""

    tl.static_assert(CODEC == 1 or CODEC == 2)
    unit, rms = _normalized(x.to(tl.float32), M)
    q = _indices(rotate(unit, M, 128, CODEC), 3)
    low = tl.sum(tl.reshape(q & 3, (M, 32, 4)) << (tl.arange(0, 4) * 2)[None, None, :], axis=2)
    high = tl.sum(tl.reshape(q >> 2, (M, 16, 8)) << tl.arange(0, 8)[None, None, :], axis=2)
    return low.to(tl.uint8), high.to(tl.uint8), rms


@triton.jit
def quant_groups_4(x, M: tl.constexpr, CODEC: tl.constexpr):
    """``[M,128]`` -> nibble bytes ``[M,64]`` and FP32 scale ``[M]``."""

    unit, rms = _normalized(x.to(tl.float32), M)
    q = _indices(rotate(unit, M, 128, CODEC), 4)
    if CODEC == 3 or CODEC == 5:
        c = _centroids(q, 4)
        energy = tl.sum(_mul_rn(c, c), axis=1)
        # Every centroid is nonzero, so even a zero source has positive energy.
        correction = _sqrt_rn(tl.math.div_rn(128.0, energy))
        rms = _mul_rn(rms, correction)
    low, high = tl.split(tl.reshape(q, (M, 64, 2)))
    return (low | (high << 4)).to(tl.uint8), rms


@triton.jit
def quant_groups_6(x, M: tl.constexpr, CODEC: tl.constexpr):
    """``[M,128]`` -> low4/high2 planes and original or norm-corrected RMS.

    CODEC7 scales rotated coding coordinates to the outer centroid before
    choosing indices. CODECs6/7 preserve the source RMS in reconstruction;
    alpha is deliberately absent from their final reconstruction scale.
    """

    tl.static_assert(CODEC == 4 or CODEC == 6 or CODEC == 7)
    unit, rms = _normalized(x.to(tl.float32), M)
    rotated = rotate(unit, M, 128, CODEC)
    if CODEC == 7:
        alpha = tl.maximum(1.0, tl.math.div_rn(tl.max(tl.abs(rotated), axis=1), _C6[63]))
        rotated = tl.math.div_rn(rotated, alpha[:, None])
    q = _indices(rotated, 6)
    if CODEC == 6 or CODEC == 7:
        c = _centroids(q, 6)
        energy = tl.sum(_mul_rn(c, c), axis=1)
        correction = _sqrt_rn(tl.math.div_rn(128.0, energy))
        rms = _mul_rn(rms, correction)
    low_a, low_b = tl.split(tl.reshape(q & 15, (M, 64, 2)))
    high = tl.sum(tl.reshape(q >> 4, (M, 32, 4)) << (tl.arange(0, 4) * 2)[None, None, :], axis=2)
    return (low_a | (low_b << 4)).to(tl.uint8), high.to(tl.uint8), rms


@triton.jit
def quant_groups_7(x, M: tl.constexpr, CODEC: tl.constexpr):
    """``[M,128]`` -> low4/mid2/high1 planes and original FP32 RMS."""

    tl.static_assert(CODEC == 4)
    unit, rms = _normalized(x.to(tl.float32), M)
    q = _indices(rotate(unit, M, 128, CODEC), 7)
    a, b = tl.split(tl.reshape(q & 15, (M, 64, 2)))
    middle = tl.sum(tl.reshape((q >> 4) & 3, (M, 32, 4)) << (tl.arange(0, 4) * 2)[None, None, :], axis=2)
    high = tl.sum(tl.reshape(q >> 6, (M, 16, 8)) << tl.arange(0, 8)[None, None, :], axis=2)
    return (a | (b << 4)).to(tl.uint8), middle.to(tl.uint8), high.to(tl.uint8), rms


@triton.jit
def correct_scale8(code, original_rms, M: tl.constexpr):
    """The norm8 metadata policy: canonical code squares, positive FP32 sum, RN div/sqrt/mul.

    Codes remain original-RMS Gaussian indices. Every pinned centroid is
    nonzero; even zero-source groups have positive energy and +0 metadata.
    Callers contain nonfinite or erased nonzero RMS before publishing State.
    """

    centroid = _centroids(code.to(tl.int32), 8)
    energy = tl.sum(_mul_rn(centroid, centroid), axis=1)
    correction = _sqrt_rn(tl.math.div_rn(128.0, energy))
    return _mul_rn(original_rms, correction)


@triton.jit
def quant_groups_8(x, M: tl.constexpr, CODEC: tl.constexpr):
    """``[M,128]`` -> unsigned indices and original or reconstruction-norm RMS."""

    tl.static_assert(CODEC == 4 or CODEC == 8)
    unit, rms = _normalized(x.to(tl.float32), M)
    code = _indices(rotate(unit, M, 128, CODEC), 8).to(tl.uint8)
    if CODEC == 8:
        rms = correct_scale8(code, rms, M)
    return code, rms


@triton.jit
def dequant_group_3(low, high, scale, M: tl.constexpr, W: tl.constexpr):
    """Planes ``[M,W/4]``, ``[M,W/8]`` and RMS ``[M,W/128]`` -> rotated BF16."""

    tl.static_assert(W % 128 == 0)
    lo = tl.reshape((low.to(tl.int32)[:, :, None] >> (tl.arange(0, 4) * 2)[None, None, :]) & 3, (M, W))
    hi = tl.reshape((high.to(tl.int32)[:, :, None] >> tl.arange(0, 8)[None, None, :]) & 1, (M, W))
    q = lo | (hi << 2)
    c = tl.reshape(_centroids(q, 3), (M, W // 128, 128))
    s = tl.reshape(scale.to(tl.float32), (M, W // 128, 1))
    return tl.reshape(c * s, (M, W)).to(tl.bfloat16)


@triton.jit
def dequant_group_4(code, scale, M: tl.constexpr, W: tl.constexpr):
    """Nibble bytes ``[M,W/2]`` and RMS ``[M,W/128]`` -> rotated BF16."""

    tl.static_assert(W % 128 == 0)
    code = code.to(tl.int32)
    q = tl.reshape(tl.join(code & 15, code >> 4), (M, W))
    c = tl.reshape(_centroids(q, 4), (M, W // 128, 128))
    s = tl.reshape(scale.to(tl.float32), (M, W // 128, 1))
    return tl.reshape(c * s, (M, W)).to(tl.bfloat16)


@triton.jit
def dequant_group_6(low, high, scale, M: tl.constexpr, W: tl.constexpr):
    """Planes ``[M,W/2]``, ``[M,W/4]``, scale ``[M,W/128]`` -> rotated BF16.

    Each128-coordinate group's64 low-nibble bytes precede its32 high-two-bit
    bytes. The caller gathers separate power-of-two axes from that96-byte block.
    """

    tl.static_assert(W % 128 == 0)
    low = low.to(tl.int32)
    lo = tl.reshape(tl.join(low & 15, low >> 4), (M, W))
    hi = tl.reshape((high.to(tl.int32)[:, :, None] >> (tl.arange(0, 4) * 2)[None, None, :]) & 3, (M, W))
    q = lo | (hi << 4)
    c = tl.reshape(_centroids(q, 6), (M, W // 128, 128))
    s = tl.reshape(scale.to(tl.float32), (M, W // 128, 1))
    return tl.reshape(_mul_rn(c, s), (M, W)).to(tl.bfloat16)


@triton.jit
def dequant_group_7(low, middle, high, scale, M: tl.constexpr, W: tl.constexpr):
    """Low4/mid2/high1 planes and FP32 RMS -> rotated BF16 ``[M,W]``.

    Physical groups are112 bytes:64 low-nibble,32 middle-two-bit,16 high-bit.
    The caller gathers each plane independently across complete128 groups.
    """

    tl.static_assert(W % 128 == 0)
    low = low.to(tl.int32)
    lo = tl.reshape(tl.join(low & 15, low >> 4), (M, W))
    mid = tl.reshape((middle.to(tl.int32)[:, :, None] >> (tl.arange(0, 4) * 2)[None, None, :]) & 3, (M, W))
    hi = tl.reshape((high.to(tl.int32)[:, :, None] >> tl.arange(0, 8)[None, None, :]) & 1, (M, W))
    c = tl.reshape(_centroids(lo | (mid << 4) | (hi << 6), 7), (M, W // 128, 128))
    s = tl.reshape(scale.to(tl.float32), (M, W // 128, 1))
    return tl.reshape(_mul_rn(c, s), (M, W)).to(tl.bfloat16)


@triton.jit
def dequant_group_8(code, scale, M: tl.constexpr, W: tl.constexpr):
    """Unsigned byte indices and FP32 RMS -> rotated BF16 ``[M,W]``."""

    tl.static_assert(W % 128 == 0)
    c = tl.reshape(_centroids(code.to(tl.int32), 8), (M, W // 128, 128))
    s = tl.reshape(scale.to(tl.float32), (M, W // 128, 1))
    return tl.reshape(_mul_rn(c, s), (M, W)).to(tl.bfloat16)


@triton.jit
def _register_centroids(values: tl.constexpr, N: tl.constexpr, M: tl.constexpr):
    coordinate = tl.arange(0, N)
    table = tl.full((N,), values[N - 1], tl.float32)
    for i in tl.static_range(N - 1):
        table = tl.where(coordinate == i, values[i], table)
    return tl.broadcast_to(table[None, :], (M, N))


@triton.jit
def _centroids_lookup(q, BITS: tl.constexpr):
    tl.static_assert(BITS == 7)
    table = _register_centroids(_C7, 128, q.shape[0])
    return tl.gather(table, q, 1)


@triton.jit
def dequant_group_7_lookup(low, middle, high, scale, M: tl.constexpr, W: tl.constexpr):
    """Low4/mid2/high1 planes and FP32 RMS -> rotated BF16 ``[M,W]``.

    Physical groups are112 bytes:64 low-nibble,32 middle-two-bit,16 high-bit.
    The caller gathers each plane independently across complete128 groups.
    """

    tl.static_assert(W % 128 == 0)
    low = low.to(tl.int32)
    lo = tl.reshape(tl.join(low & 15, low >> 4), (M, W))
    mid = tl.reshape((middle.to(tl.int32)[:, :, None] >> (tl.arange(0, 4) * 2)[None, None, :]) & 3, (M, W))
    hi = tl.reshape((high.to(tl.int32)[:, :, None] >> tl.arange(0, 8)[None, None, :]) & 1, (M, W))
    c = tl.reshape(_centroids_lookup(lo | (mid << 4) | (hi << 6), 7), (M, W // 128, 128))
    s = tl.reshape(scale.to(tl.float32), (M, W // 128, 1))
    return tl.reshape(_mul_rn(c, s), (M, W)).to(tl.bfloat16)

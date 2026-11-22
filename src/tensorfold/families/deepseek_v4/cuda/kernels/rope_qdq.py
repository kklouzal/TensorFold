# SPDX-License-Identifier: Apache-2.0
"""TensorFold-owned fusion of the vendored V4.1 RoPE and attention QDQ (``quant_rope``): one launch for
``attention_qdq(apply_rope(x, positions, table, inverse=...), fp4=..., group=...)``.

Bitwise the composition: a program holds one (token, head) row of D (128 / 512) FP16 values; the last 64 are
rotated with ``_rotate``'s FP32 expressions (cos/sin from the table, or position x frequency past it; ``a*c - b*s``
and ``a*s + b*c``, no FMA contraction), rounded to FP16 as ``_rotate``'s store, then every GROUP-value group is
quantized and dequantized with ``_qdq``'s arithmetic (the group maximum is exact in any order). Group boundaries
never cross a row (D % GROUP == 0), so the row grid equals ``_qdq``'s flattened group grid.
"""

import torch

from .quant_rope import _ceil_power_two, _fp4_value
from .fp8_software import fp8_e4m3fn_bits_to_fp32_bitcast, fp32_to_fp8_e4m3fn_bits
from .triton_utils import jit, on_triton_device, tl


@jit
def _rope_qdq(x, positions, table, out, ring, req_slot, valid, weight, LENGTH, HEADS: tl.constexpr, D: tl.constexpr,
              INVERSE: tl.constexpr, GROUP: tl.constexpr, FP4: tl.constexpr, E4M3_SCALE: tl.constexpr,
              RING_ROWS: tl.constexpr, RING_SLOTS: tl.constexpr, NORMALIZE: tl.constexpr, EPS: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)
    base = (token * HEADS + head) * D
    col = tl.arange(0, D)
    value = tl.load(x + base + col)
    if NORMALIZE:
        own = value.to(tl.float32)
        variance = tl.sum(own * own, 0) / D
        # Preserve the separate short-RMS FP16 store before exchanging the rotary pairs.
        value = (own * tl.rsqrt(variance + EPS) * tl.load(weight + col)).to(tl.float16)
    rotary = col >= D - 64
    pair = tl.where(rotary, (col - (D - 64)) // 2, 0)
    position = tl.load(positions + token).to(tl.int64)
    if position < LENGTH:
        c = tl.load(table + position * 64 + pair)
        s = tl.load(table + position * 64 + 32 + pair)
    else:
        angle = position.to(tl.float32) * tl.load(table + LENGTH * 64 + pair)
        c, s = tl.cos(angle), tl.sin(angle)
    if INVERSE:
        s = -s
    odd = (col % 2) == 1
    if NORMALIZE:
        even_value, odd_value = tl.split(tl.reshape(value, (D // 2, 2)))
        partner = tl.reshape(tl.join(odd_value, even_value), (D,)).to(tl.float32)
    else:
        partner = tl.load(x + base + (col ^ 1)).to(tl.float32)
    own = value.to(tl.float32)
    a = tl.where(odd, partner, own)
    b = tl.where(odd, own, partner)
    turned = tl.where(odd, a * s + b * c, a * c - b * s)
    rope = tl.where(rotary, turned.to(tl.float16), value).to(tl.float32)
    groups = tl.reshape(rope, (D // GROUP, GROUP))
    amax = tl.max(tl.abs(groups), 1)
    if FP4:
        if E4M3_SCALE:
            amax = tl.maximum(amax, 6.0 * (2.0**-9))
            scale = fp8_e4m3fn_bits_to_fp32_bitcast(fp32_to_fp8_e4m3fn_bits(tl.div_rn(amax, 6.0)))
        else:
            amax = tl.maximum(amax, 6.0 * (2.0**-126))
            scale = _ceil_power_two(amax * (1.0 / 6.0))
        quant = _fp4_value(tl.div_rn(groups, scale[:, None]))
    else:
        amax = tl.maximum(amax, 1e-4)
        scale = _ceil_power_two(amax * (1.0 / 448.0))
        quant = fp8_e4m3fn_bits_to_fp32_bitcast(fp32_to_fp8_e4m3fn_bits(tl.div_rn(groups, scale[:, None])))
    result = tl.reshape(quant * scale[:, None], (D,)).to(tl.float16)
    tl.store(out + base + col, result)
    if RING_ROWS:
        slot = tl.load(req_slot + token).to(tl.int64)
        live = (tl.load(valid + token) != 0) & (slot >= 0) & (slot < RING_SLOTS) & (position >= 0)
        tl.store(ring + (slot * RING_ROWS + position % RING_ROWS) * D + col, result, live)


def rope_qdq(x, positions, table, *, inverse=False, fp4=False, group=128, e4m3_scale=False,
             decode_ring=None, decode_norm=None):
    """attention_qdq(apply_rope(x, positions, table, inverse=inverse), fp4=fp4, group=group, e4m3_scale=...)
    for FP16 x [T, D] or [T, H, D] (D 128 / 512), in one launch; returns x's shape.

    ``decode_ring`` carries (ring, request slots, valid rows) for a window projection with one row per live
    slot. The caller supplies nonnegative positions, orders these writes before attention, and guarantees
    that each live slot has one writer.

    For a short decode window, ``decode_norm`` carries (FP32 RMS weight, epsilon) for the unnormalized
    FP16 projection. Rotation consumes the rounded FP16 normalized values.
    """
    shape = x.shape
    if x.ndim == 2:
        x = x.unsqueeze(1)
    if (
        x.ndim != 3
        or x.dtype != torch.float16
        or not on_triton_device(x)
        or x.shape[-1] not in (128, 512)
        or not x.is_contiguous()
        or x.shape[-1] % group
        or positions.shape != (x.shape[0],)
        or positions.dtype not in (torch.int32, torch.int64)
        or table.ndim != 2 or table.shape[1] != 64 or table.dtype != torch.float32
        or (fp4 and (group, e4m3_scale) not in ((16, True), (32, False)))
        or (not fp4 and (group != 128 or e4m3_scale))
    ):
        raise ValueError("Unsupported V4.1 fused RoPE + QDQ geometry")
    ring = req_slot = valid = None
    if decode_ring is not None:
        from .mu_packed import RING

        ring, req_slot, valid = decode_ring
        if (x.shape[1:] != (1, 512) or fp4 or inverse or ring.ndim != 3 or ring.shape[1:] != (RING, 512)
                or ring.shape[0] < 2 or ring.dtype != torch.float16 or req_slot.ndim != 1
                or req_slot.numel() < x.shape[0] or req_slot.dtype != torch.int32
                or valid.shape != (x.shape[0],) or valid.dtype != torch.bool
                or table.shape[0] < 2
                or any(t.device != x.device or not t.is_contiguous() for t in (ring, req_slot, valid, positions, table))):
            raise ValueError("V4.1 decode ring QDQ requires one FP16 window row per live slot")
    weight, eps = None, 0.0
    if decode_norm is not None:
        weight, eps = decode_norm
        if (ring is None or len(shape) != 2 or not 1 <= x.shape[0] <= 64
                or weight.shape != (512,) or weight.dtype != torch.float32
                or weight.device != x.device or not weight.is_contiguous()):
            raise ValueError("V4.1 fused decode window RMS requires short FP16 rows and a local FP32 weight")
    out = torch.empty_like(x)
    if x.shape[0]:
        _rope_qdq[(x.shape[0], x.shape[1])](
            x, positions, table, out, ring, req_slot, valid, weight, table.shape[0] - 1,
            HEADS=x.shape[1], D=x.shape[-1], INVERSE=inverse, GROUP=group, FP4=fp4, E4M3_SCALE=e4m3_scale,
            RING_ROWS=0 if ring is None else ring.shape[1], RING_SLOTS=0 if ring is None else ring.shape[0] - 1,
            NORMALIZE=weight is not None, EPS=eps,
            num_warps=4, enable_fp_fusion=False)
    return out.view(shape)

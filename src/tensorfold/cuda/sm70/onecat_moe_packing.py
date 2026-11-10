# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Adapted from 1CatAI/1Cat-vLLM 1b57260a834d40faa32bfc6ac18f050414a8f042
# gguf_moe_planes.py (a799ab3425af40527ed289d5535bb01da09c9ff4)
# gguf_dmv_formats.py (fd0580f8ffc1311ad43640836efc0f7a577732de).
# Modifications: owned E256/lane160 boundaries, original project codebooks,
# IQ4_XS planes, explicit output ownership and single-expert bounded staging.
# See onecat_LICENSE.txt and onecat_NOTICE.txt for license and source notices.
"""CPU packing into caller-funded pinned1Cat GU planes; no GPU/runtime import.

Caller retains a stable original GGUF array and exclusively owns outputs.
This is a representation producer, not coefficient-equivalence certification.
"""
import hashlib

import numpy as np

IQ2S, IQ3S, IQ3X = 7, 5, 6
ROWMAP = np.array([((lane >> 2) & 3) * 8 + (lane & 3) + (4 if lane & 16 else 0)
                   for lane in range(32)])
BLOCK_BYTES = {18: 98, 21: 110, 22: 82, 23: 136}
def _pack_iq2s(raw):
    N = raw.shape[0]
    b = raw.reshape(N, -1, 82)
    nb = b.shape[1]
    G, T = nb * 2, N // 32
    assert N % 32 == 0
    d = (
        b[..., 0:2].copy().view("<u2")[..., 0].astype(np.uint32)
    )  # (N, nb) fp16 bits
    lo = b[..., 2:34].reshape(N, nb, 8, 4)  # 8 steps x 4 index bytes
    sg = b[..., 34:66].reshape(
        N, nb, 8, 4
    )  # 8 steps x 4 sign bytes (one per 8 weights)
    qh = b[..., 66:74].reshape(N, nb, 8).astype(np.uint32)
    sc = b[..., 74:82].reshape(N, nb, 8).astype(np.uint32)
    lo_w = np.ascontiguousarray(lo).view("<u4").reshape(N, G, 4)  # word per step
    sg_w = np.ascontiguousarray(sg).view("<u4").reshape(N, G, 4)
    qh = qh.reshape(N, G, 4)
    hiw = np.zeros((N, G), np.uint32)
    for s in range(4):
        hiw |= (qh[..., s] & 0xFF) << (8 * s)  # 4 indices x 2 bits per step
    scw = np.zeros((N, G), np.uint32)
    scs = sc.reshape(N, G, 4)
    for s in range(4):
        scw |= (scs[..., s] & 0xFF) << (8 * s)
    d16 = np.repeat(d, 2, axis=1)  # (N, G)
    def lane(a):
        return a.reshape((T, 32) + a.shape[1:])[:, ROWMAP]
    codes = np.empty((N, G, 2, 4), np.uint32)
    codes[:, :, 0] = lo_w
    codes[:, :, 1] = sg_w
    codes = lane(codes).transpose(0, 2, 3, 1, 4)  # (T, G, 2, L, 4)
    meta = np.stack([hiw, d16, scw, np.zeros_like(hiw)], -1)
    meta = lane(meta).transpose(0, 2, 1, 3)  # (T, G, L, 4)
    def f(a):
        return np.ascontiguousarray(a).astype("<u4").view(np.uint8).reshape(-1)
    return IQ2S, f(codes), f(meta)


def _split_signs(w):
    """32 sign bits in weight order -> even weights in bits 0-15, odd in 16-31."""
    out = np.zeros_like(w)
    for p in range(16):
        out |= ((w >> (2 * p)) & 1) << p
        out |= ((w >> (2 * p + 1)) & 1) << (16 + p)
    return out


def _pack_iq3(raw, gtype):
    N = raw.shape[0]
    if gtype == 21:
        b = raw.reshape(N, -1, 110)
        nb = b.shape[1]
        d = b[..., 0:2].copy().view("<u2")[..., 0].astype(np.uint32)
        idx = b[..., 2:66].reshape(N, nb, 8, 8)
        hi = b[..., 66:74].reshape(N, nb, 8)
        signs = b[..., 74:106].reshape(N, nb, 8, 4)
        sc = b[..., 106:110]
        nib = np.stack([sc & 15, sc >> 4], -1).reshape(N, nb, 8)
        fmt = IQ3S
    elif gtype == 18:
        b = raw.reshape(N, -1, 98)
        nb = b.shape[1]
        d = b[..., 0:2].copy().view("<u2")[..., 0].astype(np.uint32)
        idx = b[..., 2:66].reshape(N, nb, 8, 8)
        aux = b[..., 66:98].copy().view("<u4").reshape(N, nb, 8)
        from tensorfold.families.qwen4_exp.gguf_tables import KSIGNS_IQ2XS
        ks = np.asarray(KSIGNS_IQ2XS, dtype=np.uint8)
        s7 = (aux[..., None] >> np.array([0, 7, 14, 21], np.uint32)) & 127
        signs = ks[s7]
        nib = (aux >> 28).astype(np.uint8)
        hi = np.zeros((N, nb, 8), np.uint8)
        fmt = IQ3X
    else:
        raise ValueError(gtype)
    S, G, T = nb * 8, nb * 2, N // 32
    assert N % 32 == 0
    idxw = (
        np.ascontiguousarray(idx).reshape(N, S, 8).view("<u4").reshape(N, G, 4, 2)
    )
    sgw = np.ascontiguousarray(signs).reshape(N, S, 4).view("<u4").reshape(N, G, 4)
    d16 = np.repeat(
        d, 2, axis=1
    )  # (N, G)
    nibg = nib.reshape(N, G, 4).astype(np.uint32)
    dsw = d16 | (
        (
            nibg[..., 0]
            | (nibg[..., 1] << 4)
            | (nibg[..., 2] << 8)
            | (nibg[..., 3] << 12)
        )
        << 16
    )
    def lane(a):
        return a.reshape((T, 32) + a.shape[1:])[:, ROWMAP]
    codes = np.empty((N, G, 3, 4), np.uint32)
    codes[:, :, 0] = idxw[:, :, 0:2].reshape(N, G, 4)
    codes[:, :, 1] = idxw[:, :, 2:4].reshape(N, G, 4)
    codes[:, :, 2] = _split_signs(sgw)
    codes = lane(codes).transpose(0, 2, 3, 1, 4)  # (T, G, 3, L, 4)
    if fmt == IQ3S:
        hig = hi.reshape(N, G, 4).astype(np.uint32)
        hiw = (
            hig[..., 0] | (hig[..., 1] << 8) | (hig[..., 2] << 16) | (hig[..., 3] << 24)
        )
        meta = lane(np.stack([hiw, dsw], -1)).transpose(0, 2, 1, 3)  # (T, G, L, 2)
    else:
        meta = lane(dsw).transpose(0, 2, 1)  # (T, G, L)
    def f(a):
        return np.ascontiguousarray(a).astype("<u4").view(np.uint8).reshape(-1)
    return fmt, f(codes), f(meta)


def _pack_iq4xs(raw):
    """Owned LUT4 plane producer; preserve original codes, no requantization."""
    n = raw.shape[0]
    blocks = raw.reshape(n, 10, 136)
    d = blocks[..., :2].copy().view("<f2").astype(np.float32)
    high = blocks[..., 2:4].copy().view("<u2")[..., 0]
    group = np.arange(8)
    low = (blocks[..., 4:8, None] >> np.array([0, 4], np.uint8)) & 15
    scale = low.reshape(n, 10, 8).astype(np.int16)
    scale |= (((high[..., None] >> (2 * group)) & 3) << 4).astype(np.int16)
    scale -= 32
    # This Half scale is an intentional PINNED decoder boundary. Original
    # GGUF F32 decode->Half may differ; coefficient qualification is separate.
    half_scale = (d * scale.astype(np.float32)).astype("<f2")
    scales = half_scale.view("<u2").reshape(n, 20, 4)
    packed = blocks[..., 8:].reshape(n, 10, 8, 16)
    q = np.concatenate([packed & 15, packed >> 4], axis=-1).reshape(n, 20, 4, 4, 8)
    words = np.bitwise_or.reduce(q.astype(np.uint32) << (4 * np.arange(8, dtype=np.uint32)), axis=-1)
    codes = words.reshape(5, 32, 20, 4, 4)[:, ROWMAP].transpose(0, 2, 3, 1, 4)
    meta = scales.reshape(5, 32, 20, 4)[:, ROWMAP].transpose(0, 2, 1, 3)
    return 3, np.ascontiguousarray(codes, dtype="<u4").view(np.uint8).reshape(-1), np.ascontiguousarray(meta, dtype="<u2").view(np.uint8).reshape(-1)


def requirements(ggml_type):
    """Payload extents only; runtime-managed NumPy heap/RSS is not certified."""
    if type(ggml_type) is not int or ggml_type not in BLOCK_BYTES:
        raise ValueError("SC117 GU IQ3XXS/IQ3S/IQ2S/IQ4XS type required")
    codes = 204800 if ggml_type == 23 else 102400 if ggml_type == 22 else 153600
    scales = 51200 if ggml_type == 22 else 12800 if ggml_type == 18 else 25600
    return {"codes_shape": (256, codes), "scales_shape": (256, scales),
            "source_expert_staging_bytes": 160 * 10 * BLOCK_BYTES[ggml_type],
            "output_bytes": 256 * (codes + scales),
            "max_experts_in_flight": 1, "rows_per_expert": 160,
            "columns": 2560, "managed_heap_qualification": "UNRUN"}


def table(ggml_type):
    """Use current original GGML codebook authority; no gguf-py dependency."""
    requirements(ggml_type)
    from tensorfold.families.qwen4_exp.gguf_tables import IQ2S_GRID, IQ3S_GRID, IQ3XXS_GRID

    if ggml_type == 23:
        return np.empty(0, np.uint8)
    if ggml_type == 22:
        return np.asarray(IQ2S_GRID, dtype="<u8").view(np.uint8).copy()
    return np.concatenate([np.asarray(IQ3S_GRID, dtype="<u4"),
                           np.asarray(IQ3XXS_GRID, dtype="<u4")]).view(np.uint8)


def _pack_expert(raw, ggml_type):
    # Only the validated bank adapter supplies fixed160x10-block source here.
    if ggml_type == 23:
        return _pack_iq4xs(raw)
    if ggml_type == 22:
        return _pack_iq2s(raw)
    return _pack_iq3(raw, ggml_type)


def pack_owned_into(raw, codes, scales, *, first, lane, ggml_type, column_offset=0):
    """Fill one E256 bank into caller-owned output planes, one expert at a time.

    Source is the stable original512x640 expert bank (K may include a selected
    whole-block input offset). Lane selects original output rows lane*160.
    Positive row strides and mmap inputs are accepted. An expert stride of0
    intentionally permits stable repeated/broadcast source views: all source
    access is read-only, and caller stability/lifetime still applies. Nonzero
    expert strides must keep expert spans disjoint. Arbitrary subclass dispatch,
    overlapping outputs, partial quant blocks and implicit casts are refused
    before writes. Caller funds/retains raw, outputs and NumPy staging;
    no positive publication/consumer use until this function returns.
    On failure outputs are incomplete and must not be consumed.
    """
    spec = requirements(ggml_type)
    if type(first) is not int or first not in (0, 256):
        raise ValueError("owned E256 first must be0 or256")
    if type(lane) is not int or lane not in (0, 1, 2, 3):
        raise ValueError("ETP4 width lane required")
    if type(column_offset) is not int or column_offset < 0 or column_offset % 256:
        raise ValueError("GU input offset must retain original256-value blocks")
    if type(raw) not in (np.ndarray, np.memmap) or raw.dtype != np.uint8 or raw.ndim != 3:
        raise TypeError("original UInt8 GGUF expert-bank array required")
    block = BLOCK_BYTES[ggml_type]
    start = column_offset // 256 * block
    width = 10 * block
    if raw.shape[:2] != (512, 640) or raw.shape[2] < start + width or raw.shape[2] % block:
        raise ValueError("original512x640 SC117 bank extent differs")
    if raw.strides[2] != 1 or raw.strides[1] < raw.shape[2] or raw.strides[0] < 0:
        raise ValueError("source requires contiguous original payload bytes and nonoverlapping positive rows")
    if raw.strides[0] and raw.strides[0] < (raw.shape[1] - 1) * raw.strides[1] + raw.shape[2]:
        raise ValueError("source expert spans overlap")
    for output, shape in ((codes, spec["codes_shape"]), (scales, spec["scales_shape"])):
        if (type(output) is not np.ndarray or output.dtype != np.uint8 or output.shape != shape
                or not output.flags.c_contiguous or not output.flags.writeable):
            raise ValueError("caller-owned writable exact contiguous plane extent required")
        if np.may_share_memory(raw, output):
            raise ValueError("plane output may alias original source")
    if np.may_share_memory(codes, scales):
        raise ValueError("plane outputs overlap")
    digest = hashlib.sha256()
    fmt = None
    for expert in range(256):
        staged = raw[first + expert, lane * 160:(lane + 1) * 160, start:start + width]
        staged = np.array(staged, dtype=np.uint8, copy=True, order="C", subok=False)
        digest.update(memoryview(staged))
        fmt, packed_codes, packed_scales = _pack_expert(staged, ggml_type)
        # Producer/type disagreement is fatal; never pad/truncate into the ABI.
        if packed_codes.shape != codes[expert].shape or packed_scales.shape != scales[expert].shape:
            raise RuntimeError("pinned packing extent disagrees with owned GU ABI")
        codes[expert] = packed_codes
        scales[expert] = packed_scales
        del staged, packed_codes, packed_scales
    return {"schema": "tensorfold.onecat-owned-gu-packing-v1", "first": first, "lane": lane,
            "ggml_type": ggml_type, "column_offset": column_offset, "columns": 2560,
            "completed_experts": 256, "format": fmt, "source_payload_sha256": digest.hexdigest(),
            "codes_sha256": hashlib.sha256(memoryview(codes)).hexdigest(),
            "scales_sha256": hashlib.sha256(memoryview(scales)).hexdigest(),
            "coefficient_contract": "pinned-half2-reconstruction-not-original-equivalence",
            "resources": spec}

"""Bounded row decoders for the pinned GGML mixed IQ/K block layouts.

Integer layout/codebook authority: llama.cpp b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9.
Arithmetic preserves the reference's FP32 scale/product boundaries. This host
path is an independent import/control implementation, not a weight conversion.
"""

import numpy as np

from .gguf_tables import IQ2S_GRID, IQ3S_GRID, IQ3XXS_GRID, KSIGNS_IQ2XS, KVALUES_IQ4NL


def decode(raw, kind, width, block_bytes):
    blocks = raw.reshape(-1, block_bytes)
    count = 64 if kind == 42 else 256
    i = np.arange(count)
    d_at = 208 if kind == 14 else 0
    d = blocks[:, d_at:d_at + 2].copy().view("<f2").astype(np.float32)
    if kind == 42:
        q = ((blocks[:, 2 + i // 4] >> (2 * (i % 4))) & 3).astype(np.int16) - 1
        result = d * q.astype(np.float32)
    elif kind == 14:
        half, quarter, lane = i // 128, (i % 128) // 32, i % 32
        low = (blocks[:, half * 64 + (quarter % 2) * 32 + lane] >> (4 * (quarter // 2))) & 15
        high = (blocks[:, 128 + half * 32 + lane] >> (2 * quarter)) & 3
        q = (low | (high << 4)).astype(np.int16) - 32
        scales = blocks[:, 192:208].view(np.int8)[:, half * 8 + quarter * 2 + lane // 16]
        result = (d * scales.astype(np.float32)) * q.astype(np.float32)
    elif kind == 23:
        group, element = i // 32, i % 32
        high = blocks[:, 2:4].copy().view("<u2")
        scales = ((blocks[:, 4 + group // 2] >> (4 * (group % 2))) & 15).astype(np.int16)
        scales |= (((high >> (2 * group)) & 3) << 4).astype(np.int16)
        scales -= 32
        codes = (blocks[:, 8 + group * 16 + element % 16] >> (4 * (element // 16))) & 15
        values = np.asarray(KVALUES_IQ4NL, dtype=np.float32)[codes]
        result = (d * scales.astype(np.float32)) * values
    elif kind == 22:
        group, sub, element = i // 32, (i % 32) // 8, i % 8
        codes = blocks[:, 2 + group * 4 + sub].astype(np.uint16)
        codes = codes | ((blocks[:, 66 + group].astype(np.uint16) << (8 - 2 * sub)) & 0x300)
        grid = np.asarray(IQ2S_GRID, dtype=np.uint64)[codes]
        values = ((grid >> (8 * element).astype(np.uint64)) & 255).astype(np.float32)
        signs = blocks[:, 34 + group * 4 + sub]
        nibble = (blocks[:, 74 + group] >> (4 * (sub // 2))) & 15
        scale = (d * (np.float32(.5) + nibble.astype(np.float32))) * np.float32(.25)
        result = (scale * values) * np.where((signs >> element) & 1, np.float32(-1), np.float32(1))
    elif kind == 18:
        group, sub, element = i // 32, (i % 32) // 8, i % 8
        auxiliary = blocks[:, 66:98].copy().view("<u4")[:, group]
        signs = np.asarray(KSIGNS_IQ2XS, dtype=np.uint8)[(auxiliary >> (7 * sub)) & 127]
        codes = blocks[:, 2 + group * 8 + sub * 2 + element // 4]
        grid = np.asarray(IQ3XXS_GRID, dtype=np.uint32)[codes]
        values = ((grid >> (8 * (element % 4))) & 255).astype(np.float32)
        scale = (d * (np.float32(.5) + (auxiliary >> 28).astype(np.float32))) * np.float32(.5)
        result = (scale * values) * np.where((signs >> element) & 1, np.float32(-1), np.float32(1))
    elif kind == 21:
        group, sub, element = i // 32, (i % 32) // 8, i % 8
        codes = blocks[:, 2 + group * 8 + sub * 2 + element // 4].astype(np.uint16)
        codes = codes | ((blocks[:, 66 + group].astype(np.uint16) << (8 - 2 * sub - element // 4)) & 256)
        grid = np.asarray(IQ3S_GRID, dtype=np.uint32)[codes]
        values = ((grid >> (8 * (element % 4))) & 255).astype(np.float32)
        signs = blocks[:, 74 + group * 4 + sub]
        scales = (blocks[:, 106 + group // 2] >> (4 * (group % 2))) & 15
        scale = d * (1 + 2 * scales.astype(np.float32))
        result = (scale * values) * np.where((signs >> element) & 1, np.float32(-1), np.float32(1))
    else:
        raise ValueError("this decoder requires a declared mixed GGML block type")
    return result.reshape(raw.shape[0], width)

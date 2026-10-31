"""Independent byte-stream oracle for Gemma's native MLX affine checkpoint ABI."""

from __future__ import annotations

import math
import struct


def pack_codes(rows, bits):
    """Encode codes through individual bits/bytes, independently of CUDA word shifts."""

    import numpy as np

    rows = np.asarray(rows)
    if bits not in (4, 6) or rows.shape[-1] % 64 or np.any(rows < 0) or np.any(rows >= 1 << bits):
        raise ValueError("oracle requires unsigned 4/6-bit codes in whole 64-input groups")
    flat = rows.reshape(-1, rows.shape[-1])
    encoded = np.empty((len(flat), rows.shape[-1] * bits // 32), dtype='<u4')
    # Bound scratch independently of model size. NumPy's bit-to-byte encoder
    # supplies an oracle independent of the kernel's word-straddle shifts.
    for start in range(0, len(flat), 16):
        block = flat[start:start + 16]
        stream = ((block[..., None] >> np.arange(bits)) & 1).astype(np.uint8).reshape(len(block), -1)
        encoded[start:start + len(block)] = np.packbits(stream, axis=-1, bitorder='little').view('<u4')
    return encoded.reshape(*rows.shape[:-1], rows.shape[-1] * bits // 32)



def decode_bytes(words, bits, k):
    """Decode little-endian bytes one bit at a time, including every word straddle."""

    import numpy as np

    data = np.asarray(words, dtype='<u4').tobytes()
    return [sum(((data[(i * bits + b) // 8] >> ((i * bits + b) % 8)) & 1) << b for b in range(bits))
            for i in range(k)]


def bf16(value):
    """Round finite FP32 to BF16 nearest-even using a scalar representation oracle."""

    raw = struct.unpack('<I', struct.pack('<f', value))[0]
    rounded = (raw + 0x7FFF + ((raw >> 16) & 1)) & 0xFFFF0000
    return struct.unpack('<f', struct.pack('<I', rounded))[0]


def linear64(x, codes, scales, biases):
    """FP64 mathematical affine reference; never a serving implementation."""

    return [math.fsum(float(v) * (int(q) * float(scales[n][i // 64]) + float(biases[n][i // 64]))
                      for i, (v, q) in enumerate(zip(x, row))) for n, row in enumerate(codes)]

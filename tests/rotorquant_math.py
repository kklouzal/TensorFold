"""Scalar helpers shared by the RotorQuant codebook tests' independent references."""

import math
import struct


def f32(value):
    """``value`` rounded to the nearest float32."""
    return struct.unpack("<f", struct.pack("<f", value))[0]


def phi(value):
    """The standard normal density."""
    return 0.0 if math.isinf(value) else math.exp(-value * value / 2) / math.sqrt(2 * math.pi)

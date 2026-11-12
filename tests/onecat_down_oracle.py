"""Independent exact-dyadic Q8_1/error oracle; no candidate imports/GPU math.

Only the canonical binary16/32 rounding primitives are reused. Invalid/nonfinite
input is outside this finite oracle and must be represented as explicit poison.
"""

from fractions import Fraction
import struct

from tests.sm70_native_rn_oracle import bits_to_fraction, round_fraction
from tensorfold.families.qwen4_exp.gguf_tables import KVALUES_IQ4NL


def q8_block(half_bits):
    if len(half_bits) != 32:
        raise ValueError("one32-value block required")
    values = [bits_to_fraction(int(v), 16) for v in half_bits]
    maximum = max(map(abs, values))
    sums = [round_fraction(v, negative_zero=int(bits) == 0x8000) for v, bits in zip(values, half_bits)]
    for offset in (16, 8, 4, 2, 1):
        sums = [
            round_fraction(
                bits_to_fraction(v) + bits_to_fraction(sums[lane ^ offset]),
                negative_zero=v == sums[lane ^ offset] == 0x80000000,
            )
            for lane, v in enumerate(sums)
        ]
    d_bits = round_fraction(maximum / 127)
    d = bits_to_fraction(d_bits)
    codes = []
    for value in values:
        ratio = bits_to_fraction(round_fraction(value / d)) if maximum else Fraction()
        magnitude = abs(ratio)
        integer = (2 * magnitude.numerator + magnitude.denominator) // (2 * magnitude.denominator)
        q = -integer if ratio < 0 else integer
        if not -127 <= q <= 127:
            raise ArithmeticError("defined quantizer range not established")
        codes.append(q)
    scale = round_fraction(d, 16)
    total = round_fraction(bits_to_fraction(sums[0]), 16, negative_zero=sums[0] == 0x80000000)
    return struct.pack("<HH", scale, total) + bytes(q & 255 for q in codes)


def route_bits(raw90, q180, output_bits=16):
    if output_bits not in (16, 32):
        raise ValueError("explicit Half/F32 output rounding required")
    if len(raw90) != 90 or len(q180) != 180:
        raise ValueError("originalK160/Q8_1 extents required")
    dot = 0
    for group in range(5):
        block = raw90[group * 18 : (group + 1) * 18]
        dw = bits_to_fraction(int.from_bytes(block[:2], "little"), 16)
        q = q180[group * 36 : (group + 1) * 36]
        dx = bits_to_fraction(int.from_bytes(q[:2], "little"), 16)
        integer = 0
        for k in range(32):
            code = block[2 + k % 16] >> (4 * (k // 16)) & 15
            activation = q[4 + k] if q[4 + k] < 128 else q[4 + k] - 256
            integer += KVALUES_IQ4NL[code] * activation
        pair = bits_to_fraction(round_fraction(dw * dx))
        term = bits_to_fraction(round_fraction(integer * pair))
        dot = round_fraction(bits_to_fraction(dot) + term)
    return round_fraction(bits_to_fraction(dot), output_bits)


def quantization_bound(half_bits, qbytes):
    """A priori bound; actual errors are observations, not fitted margins."""
    values = [bits_to_fraction(int(v), 16) for v in half_bits]
    maximum = max(map(abs, values))
    if not maximum:
        return Fraction()
    u = Fraction(1, 2**24)
    d = bits_to_fraction(round_fraction(maximum / 127))
    dh = bits_to_fraction(int.from_bytes(qbytes[:2], "little"), 16)
    # q=round-away(RN32(x/d)); |RN32(x/d)-x/d| <=u*|x/d|.
    return d * Fraction(1, 2) + u * maximum + 127 * abs(dh - d)

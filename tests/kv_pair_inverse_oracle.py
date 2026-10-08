"""Independent fixed-transform RN/BF16 envelopes used only by CUDA tests."""
from __future__ import annotations
import math
import struct
from fractions import Fraction
from tensorfold.families.qwen4_exp.cuda import rotorquant_ref as ref

def inverse_rounding_envelope(values, variant):
    """Bound fixed rotation with absolute input terms and FP32 RN operation depth.

    The signed quaternion has 3additions/stage, Givens2mult+1add, plus scale.
    Three quaternion stages => conservative path depth16 (inclGivens); a
    signed64 inverse depth12. Planar depth3; IsoFast depth4. Coefficients have
    magnitude<=1; absolute matrix bound follows the fixed independent oracle.
    """
    codec = ref.variant_id(variant)
    depth = 3 if codec == 1 else 4 if codec == 2 else 12 if codec in (3, 5) else 16
    ideal = ref.rotate_oracle(values, variant, inverse=True)
    absolute = absolute_inverse_matrix_product(values, variant)
    u = 2**-24
    gamma = depth * u / (1 - depth * u)
    # Absolute stage amplification of each underflow RN term is bounded by
    # 16: three quaternion L1 stage gains2 and one Givens gain below2.
    # The staged FP64 factorization has fewer than16 RN operations per path.
    # Gamma64 conservatively covers center and absolute-bound evaluation;
    # signed128 centers can depend on all128 inputs, not merely64.
    u64 = 2**-53
    gamma64 = 64 * u64 / (1 - 64 * u64)
    error = [(gamma + gamma64) * item + depth * 16 * 2**-150 for item in absolute]
    return ideal, error

def absolute_inverse_matrix_product(values, variant):
    """Absolute stage matrices retain cancellation-free error contributions.

    Multiplying absolute matrices (rather than abs of the composite) safely
    bounds each rounded intermediate in the fixed matrix factorization.
    """
    codec = ref.variant_id(variant)
    output = list(map(abs, values))
    if codec == 1:
        for start in range(0, len(output), 2):
            c = abs(ref.PLANAR_COS[(start // 2) % 4])
            s = abs(ref.PLANAR_SIN[(start // 2) % 4])
            a, b = output[start:start + 2]
            output[start], output[start + 1] = c * a + s * b, s * a + c * b
        return output
    if codec in (4, 6, 7, 8):
        c, s = abs(ref.WIDE_GIVENS_COS), abs(ref.WIDE_GIVENS_SIN)
        for start in range(0, len(output), 128):
            for i in range(64):
                a, b = output[start + i], output[start + i + 64]
                output[start + i], output[start + i + 64] = c * a + s * b, s * a + c * b
    width = 4 if codec == 2 else 128 if codec in (4, 6, 7, 8) else 64
    for stride in ((1,) if codec == 2 else (16, 4, 1)):
        before = output.copy()
        for start in range(0, len(output), width):
            for high in range(0, width, 4 * stride):
                for low in range(stride):
                    indices = [start + high + low + k * stride for k in range(4)]
                    result = math.fsum(before[i] for i in indices) / 2
                    for i in indices:
                        output[i] = result
    return output


def fp32_bits(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("operand must be a finite exact FP32 scalar")
    try:
        raw = struct.unpack("<I", struct.pack("<f", value))[0]
        result = struct.unpack("<f", struct.pack("<I", raw))[0]
    except (OverflowError, struct.error) as error:
        raise ValueError("operand exceeds finite FP32") from error
    if not math.isfinite(result) or result != value:
        raise ValueError("operand must already be exactly FP32")
    return raw

def fp32_from_bits(raw):
    if type(raw) is not int or not 0 <= raw < 1 << 32 or raw >> 23 & 255 == 255:
        raise ValueError("finite FP32 bits required")
    return struct.unpack("<f", struct.pack("<I", raw))[0]

def _power(exponent):
    return Fraction(1 << exponent) if exponent >= 0 else Fraction(1, 1 << -exponent)

def round_binary_bits(value: Fraction, precision: int, *, negative_zero=False):
    """Exact integer RN-even rounding to finite F32/BF16, with subnormals."""
    if not isinstance(value, Fraction) or precision not in (8, 24):
        raise ValueError("exact Fraction and precision8/24 required")
    sign = value < 0 or (value == 0 and negative_zero)
    value = abs(value)
    sign_bit = int(sign) << (precision + 7)
    if not value:
        return sign_bit
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < _power(exponent):
        exponent -= 1
    quantum = max(exponent, -126) - (precision - 1)
    scaled = value / _power(quantum)
    integer, remainder = divmod(scaled.numerator, scaled.denominator)
    twice = remainder * 2
    integer += twice > scaled.denominator or (twice == scaled.denominator and integer & 1)
    if not integer:
        return sign_bit
    if integer == 1 << precision:
        integer >>= 1
        quantum += 1
    if integer < 1 << (precision - 1):
        return sign_bit | integer
    rounded_exponent = quantum + precision - 1
    if rounded_exponent > 127:
        raise ValueError("rounding exceeds finite output domain")
    return sign_bit | ((rounded_exponent + 127) << (precision - 1)) | (integer - (1 << (precision - 1)))

def bf16_order(raw):
    if type(raw) is not int or not 0 <= raw < 1 << 16 or raw >> 7 & 255 == 255:
        raise ValueError("finite BF16 bits required")
    return (~raw & 0xFFFF) if raw & 0x8000 else raw | 0x8000

def bf16_interval(lower: Fraction, upper: Fraction):
    """Monotone RN-even endpoint range; signed zeros both denote exact zero."""
    if lower > upper:
        raise ValueError("inverse interval endpoints are reversed")
    low_bits, high_bits = (round_binary_bits(x, 8) for x in (lower, upper))
    low, high = bf16_order(low_bits), bf16_order(high_bits)
    if lower <= 0 <= upper:
        low, high = min(low, bf16_order(0x8000)), max(high, bf16_order(0))
    return {"lower_order": low, "upper_order": high,
            "lower_bits": low_bits, "upper_bits": high_bits,
            "nonzero_singleton": low == high,
            "numeric_zero_only": low >= bf16_order(0x8000) and high <= bf16_order(0)}


def native_inverse_rounding_envelope(values):
    """Physical FP32 H32 coefficient,5 RN adds and1 RN multiply.

    The center uses the exact stored dyadic coefficient, rather than imposing
    real-coefficient involution. Gamma6 bounds each contributing input path.
    Subnormal addition of binary32 operands is exact; final multiply may
    underflow, contributing one half-smallest-subnormal absolute term.
    """
    if len(values) % 32 or not values:
        raise ValueError("native inverse requires complete nonempty32-coordinate groups")
    coefficient = struct.unpack("<f", struct.pack("<f", 1 / math.sqrt(32)))[0]
    u = 2**-24
    gamma = 6 * u / (1 - 6 * u)
    u64 = 2**-53
    gamma64 = 64 * u64 / (1 - 64 * u64)
    centers, errors = [], []
    for start in range(0, len(values), 32):
        block = values[start:start + 32]
        for value in block:
            fp32_bits(value)
        magnitude = math.fsum(map(abs, block)) * coefficient
        for i in range(32):
            center = math.fsum((-value if (i & j).bit_count() & 1 else value) for j,value in enumerate(block)) * coefficient
            centers.append(center)
            errors.append(math.nextafter((gamma + gamma64) * magnitude + 2**-150, math.inf))
    return centers, errors

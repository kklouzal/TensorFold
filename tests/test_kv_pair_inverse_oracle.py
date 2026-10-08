"""Source/stdlib proof checks; never import Torch or create GPU state."""
from __future__ import annotations

from fractions import Fraction
import math
from pathlib import Path
import random
import struct
import unittest

from tests.kv_pair_inverse_oracle import (bf16_interval, bf16_order, fp32_from_bits,
    inverse_rounding_envelope as bound, native_inverse_rounding_envelope, round_binary_bits, ref)


def rn(value):
    bits = round_binary_bits(value, 24)
    return Fraction.from_float(fp32_from_bits(bits))


def staged(values, variant):
    """Exact dyadic F32 RN, matching only declared operation sequence."""
    codec = ref.variant_id(variant)
    output = [Fraction.from_float(value) for value in values]
    if codec == 1:
        for i in range(0, len(output), 2):
            c = Fraction.from_float(ref.PLANAR_COS[(i // 2) % 4])
            s = Fraction.from_float(ref.PLANAR_SIN[(i // 2) % 4])
            a, b = output[i:i + 2]
            output[i], output[i + 1] = rn(rn(c * a) + rn(s * b)), rn(rn(-s * a) + rn(c * b))
        return output
    if codec in (4, 6, 7, 8):
        c = Fraction.from_float(ref.WIDE_GIVENS_COS)
        s = Fraction.from_float(ref.WIDE_GIVENS_SIN)
        for begin in range(0, len(output), 128):
            for i in range(64):
                a, b = output[begin + i], output[begin + i + 64]
                output[begin + i], output[begin + i + 64] = rn(rn(c * a) + rn(s * b)), rn(rn(-s * a) + rn(c * b))
    width = 4 if codec == 2 else 128 if codec in (4, 6, 7, 8) else 64
    for stride in ((1,) if codec == 2 else (16, 4, 1)):
        before = output.copy()
        for begin in range(0, len(output), width):
            for high in range(0, width, 4 * stride):
                for low in range(stride):
                    index = [begin + high + low + k * stride for k in range(4)]
                    a, b, c, d = [before[i] for i in index]
                    components = (rn(rn(rn(a + b) + c) + d), rn(rn(rn(-a + b) + c) - d),
                                  rn(rn(rn(-a - b) + c) + d), rn(rn(rn(-a + b) - c) + d))
                    for i, value in zip(index, components, strict=True):
                        output[i] = rn(value / 2)
    if codec in (4, 5, 6, 7, 8):
        output = [-value if ref.SIGN_PAIR_NEGATIVE_MASK >> ((i // 2) % 64) & 1 else value
                  for i, value in enumerate(output)]
    return output


VARIANTS = ("planar", "isofast", "iso64-norm", "signed-iso128", "signed-iso64-norm",
            "signed-iso128-norm", "signed-iso128-outlier-norm", "signed-iso128-norm8")


class Bounds(unittest.TestCase):
    def check_fixture(self, values):
        for variant in VARIANTS:
            with self.subTest(variant=variant):
                center, error = bound(values, variant)
                result = staged(values, variant)
                self.assertEqual(len(center), len(values))
                self.assertTrue(all(math.isfinite(x) and x >= 0 for x in error))
                for i, (c, e, actual) in enumerate(zip(center, error, result, strict=True)):
                    self.assertLessEqual(abs(actual - Fraction.from_float(c)), Fraction.from_float(e),
                                         f"coordinate {i}")
                    lower = Fraction.from_float(c) - Fraction.from_float(e)
                    upper = Fraction.from_float(c) + Fraction.from_float(e)
                    interval = bf16_interval(lower, upper)
                    bits = round_binary_bits(actual, 8)
                    order = bf16_order(bits)
                    self.assertLessEqual(interval["lower_order"], order)
                    self.assertGreaterEqual(interval["upper_order"], order)

    def test_zero_and_signed_cancellation(self):
        self.check_fixture([0.] * 256)
        self.check_fixture([(-1.) ** i for i in range(256)])

    def test_random_normal_multiple_groups(self):
        rng = random.Random(99031)
        self.check_fixture([struct.unpack("<f", struct.pack("<f", rng.uniform(-2, 2)))[0] for _ in range(256)])

    def test_large_finite_norm8_domain(self):
        self.check_fixture([float((i % 7 - 3) * 2**37) for i in range(256)])

    def test_binary32_smallest_subnormal_and_small_normal(self):
        self.check_fixture([float((i % 7 - 3) * 2**-149) for i in range(256)])
        self.check_fixture([float((i % 7 - 3) * 2**-130) for i in range(256)])


    def test_native_physical_coefficient_and_subnormals_obey_bound(self):
        coefficient = Fraction.from_float(struct.unpack("<f", struct.pack("<f",1/math.sqrt(32)))[0])
        for exponent in (0,37,-130,-149):
            values=[float((i%7-3)*2.**exponent) for i in range(256)]
            center,error=native_inverse_rounding_envelope(values)
            actual=[]
            for start in range(0,len(values),32):
                block=[Fraction.from_float(value) for value in values[start:start+32]]
                for stride in (1,2,4,8,16):
                    before=block.copy()
                    for first in range(0,32,2*stride):
                        for index in range(stride):
                            a,b=before[first+index],before[first+index+stride]
                            block[first+index],block[first+index+stride]=rn(a+b),rn(a-b)
                actual.extend(rn(value*coefficient) for value in block)
            for c,e,a in zip(center,error,actual,strict=True):
                self.assertLessEqual(abs(a-Fraction.from_float(c)),Fraction.from_float(e))

    def test_fixed_source_contains_no_unknown_oracle_symbol_or_skip_filter(self):
        source = (Path(__file__).resolve().parent / "cuda/test_flashnext_kv_pairs.py").read_text()
        self.assertNotIn("rotate_oracle_absolute", source)
        self.assertNotIn("xfail", source)
        self.assertNotIn("allclose", source)
        self.assertNotIn("atol=", source)
        self.assertIn("itertools.product(FORMATS, repeat=2)", source)
        self.assertNotIn("1.05", source)


if __name__ == "__main__":
    unittest.main()

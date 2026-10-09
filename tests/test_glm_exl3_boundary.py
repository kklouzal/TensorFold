"""Source-executable GLM native integer boundary oracle; no SDK/native calls."""
from __future__ import annotations

import itertools
from pathlib import Path
import re
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/glm5_next/cuda/exl3.cpp'
MAXIMUM = 2**63 - 1


def conditions():
    text = SOURCE.read_text()
    grouped = text[text.index('void grouped('):text.index('\nvoid rot_in(')]
    divider = re.search(r'TORCH_CHECK\((SK > 0.*?),\s*"split and tile divisors', grouped, re.S).group(1)
    capacity = re.search(r'TORCH_CHECK\((mats > 0.*?),\s*"Z too small"', grouped, re.S).group(1)
    def translate(value):
        return ' '.join(value.replace('&&', ' and ').replace('Z.numel()', 'size').replace('/', '//').split())
    return grouped, compile(translate(divider), 'actual-GLM-divider-condition', 'eval'), \
        compile(translate(capacity), 'actual-GLM-capacity-condition', 'eval')


class Boundary(unittest.TestCase):
    def test_actual_short_circuit_divisor_condition_matches_independent_big_integer_oracle(self):
        _, condition, _ = conditions()
        values = (-2**63, -1, 0, 1, 2, 4, 8, 16, 2**31, 2**59, 2**60, MAXIMUM)
        count = 0
        for split, warps, tile in itertools.product(values, repeat=3):
            got = bool(eval(condition, {'SK': split, 'warps': warps, 'nt': tile, 'maximum': MAXIMUM}))
            wanted = split > 0 and warps > 0 and tile > 0 and 16 * split * warps <= MAXIMUM and 16 * tile <= MAXIMUM
            self.assertEqual(got, wanted)
            if got:
                self.assertTrue(0 < 16 * split * warps <= MAXIMUM)
                self.assertTrue(0 < 16 * tile <= MAXIMUM)
            count += 1
        self.assertEqual(count, 1728)

    def test_actual_capacity_condition_uses_safe_division_and_preserves_all_positive_counts(self):
        _, _, condition = conditions()
        count = 0
        values = (-1, 0, 1, 2, 7, MAXIMUM)
        for matrices, pairs, outputs, split, size in itertools.product(values, repeat=5):
            got = bool(eval(condition, {'mats': matrices, 'P': pairs, 'N': outputs, 'SK': split, 'size': size}))
            # SK positivity is the preceding mandatory condition, not this
            # output bound. Compare the reachable positive-split region only.
            if split <= 0:
                continue
            wanted = matrices > 0 and pairs > 0 and outputs > 0 and size >= matrices * split * pairs * outputs
            self.assertEqual(got, wanted)
            count += 1
        self.assertEqual(count, 5184)

    def test_guards_precede_native_entry_and_no_output_product_can_overflow(self):
        grouped, _, _ = conditions()
        self.assertLess(grouped.index('SK > 0'), grouped.index('exl3_grouped_cuda('))
        self.assertLess(grouped.index('mats > 0'), grouped.index('exl3_grouped_cuda('))
        self.assertNotIn('mats * SK * P * N', grouped)
        self.assertIn('SK <= Z.numel() / mats / P / N', grouped)


if __name__ == '__main__':
    unittest.main()

"""Finite address/ownership proofs for the private native Gemma-up choices."""
import unittest
from collections import Counter


def reference_pair_members(picks):
    """Independent plan coordinates: ascending expert, then original flat pair."""
    experts = [expert for row in picks for expert in row]
    return sorted(range(len(experts)), key=lambda member: (experts[member], member))


class GemmaUpLayout(unittest.TestCase):
    def test_reference_pairs_use_grouped_scratch_coordinates(self):
        # Regression: the original [row, selected slot] oracle produced
        # [2,0,1], but native down scratch owns [expert0,expert1,expert2].
        self.assertEqual(reference_pair_members(((2, 0, 1),)), [1, 2, 0])

    def test_reference_pairs_retain_rows_slots_and_duplicate_experts(self):
        picks = ((2, 0, 1), (0, 2, 0), (1, 1, 2))
        members = reference_pair_members(picks)
        self.assertEqual(members, [1, 3, 5, 2, 6, 7, 0, 4, 8])
        self.assertEqual([divmod(member, 3) for member in members],
                         [(0, 1), (1, 0), (1, 2), (0, 2), (2, 0), (2, 1), (0, 0), (1, 1), (2, 2)])
        self.assertEqual(sorted(members), list(range(9)))

    def test_copy_partition_covers_each_16byte_chunk_once(self):
        for warps in (2, 4):
            with self.subTest(warps=warps):
                owners = Counter(chunk for thread in range(warps * 32)
                                 for chunk in range(thread, 2304 // 16, warps * 32))
                self.assertEqual(owners, Counter(range(144)))
                self.assertTrue(all(0 <= chunk * 16 <= 2304 - 16 for chunk in owners))

    def test_warp_partition_keeps_whole_16column_quantizers_and_disjoint_outputs(self):
        for warps in (2, 4):
            span = 8 // warps
            self.assertEqual(span % 2, 0)
            for count in range(1, 17):
                codes, scales = Counter(), Counter()
                for warp in range(warps):
                    for lane in range(32):
                        row0, thread = lane // 4, lane % 4
                        for pair in range(0, span, 2):
                            column = (warp * span + pair) * 8
                            self.assertEqual(column % 16, 0)
                            for high in (0, 8):
                                row = row0 + high
                                if row >= count:
                                    continue
                                codes[(row, column // 2 + thread)] += 1
                                codes[(row, column // 2 + 4 + thread)] += 1
                                if thread == 0:
                                    scales[(row, column // 16)] += 1
                self.assertEqual(codes, Counter((row, byte) for row in range(count) for byte in range(32)))
                self.assertEqual(scales, Counter((row, block) for row in range(count) for block in range(4)))

    def test_ring_reuses_only_previously_consumed_buffers(self):
        # This proves our ring indices, not CUDA's documented wait/barrier
        # implementation. The consumer's existing CTA barrier remains required.
        for stages in (2, 3, 4):
            for groups in range(1, 130):
                loaded = {group: group for group in range(min(groups, stages - 1))}
                for group in range(groups):
                    self.assertEqual(loaded[group % stages], group)
                    future = group + stages - 1
                    if future < groups:
                        buffer = future % stages
                        self.assertNotEqual(buffer, group % stages)
                        self.assertLess(loaded.get(buffer, -1), group)
                        loaded[buffer] = future


if __name__ == '__main__':
    unittest.main()

"""Independent SM70 prompt passes preserve fairness, bounds and modern behavior.

These controls invoke the installed MultiDecoder methods without allocating any
model, CUDA object or decoder buffers. No source rewriting or private fixtures.
"""
from types import SimpleNamespace
import unittest

from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder


def make(*, provider="sm70", profile="optimized", lengths=(8, 10, 12), rows=64):
    result = object.__new__(MultiDecoder)
    result.w = SimpleNamespace(meta={"execution_provider": provider, "sm70_profile": profile})
    result.filling = [SimpleNamespace(sid=index, prompt=list(range(length)), background=False, points=[])
                      for index, length in enumerate(lengths)]
    result.fills = {stream.sid: [None, False, 0] for stream in result.filling}
    result.passed = {}
    result._pass_rows = lambda: rows
    result._points = lambda stream, start, count: [p for p in stream.points if start < p <= start + count]
    return result


class TestPromptPass(unittest.TestCase):
    def test_scope_and_modern_reference_fallback(self):
        for provider, profile, count in (("sm70", "optimized", 1), ("sm70", "reference", 3),
                                         ("cuda", "optimized", 3), ("cuda", "reference", 3),
                                         ("sm70", None, 3), (None, "optimized", 3)):
            with self.subTest(provider=provider, profile=profile):
                self.assertEqual(len(make(provider=provider, profile=profile)._pieces()), count)

    def test_empty_fill_and_zero_row_pass(self):
        self.assertEqual(make(lengths=())._pieces(), [])
        self.assertEqual(make()._pieces(0), [])

    def test_explicit_and_computed_row_limits(self):
        value = make(lengths=(100, 200), rows=17)
        self.assertEqual(value._pieces(), [(value.filling[0], 0, 17)])
        self.assertEqual(value._pieces(9), [(value.filling[0], 0, 9)])

    def test_singleton_piece_matches_unchanged_reference_policy(self):
        for start in (0, 3, 9):
            for rows in (1, 4, 40):
                a, b = make(profile="reference", lengths=(11,)), make(lengths=(11,))
                a.fills[0][2] = b.fills[0][2] = start
                old, new = a._pieces(rows), b._pieces(rows)
                self.assertEqual([(s.sid, at, n) for s, at, n in old],
                                 [(s.sid, at, n) for s, at, n in new])

    def test_stable_arrival_tie_is_preserved(self):
        value = make(lengths=(10, 10, 10))
        self.assertEqual(value._pieces()[0][0].sid, 0)

    def test_foreground_then_shortest_remaining_is_preserved(self):
        value = make(lengths=(10, 8, 6))
        value.filling[2].background = True
        self.assertEqual(value._pieces()[0][0].sid, 1)
        value.fills[0][2] = 9
        self.assertEqual(value._pieces()[0][0].sid, 0)

    def test_overdue_background_takes_precedence(self):
        value = make(lengths=(10, 500, 6))
        value.filling[1].background = True
        value.passed = {1: 8}
        self.assertEqual(value._pieces()[0][0].sid, 1)

    def test_fill_guard_bounds_repeated_foreground_preemption(self):
        value = make(lengths=(1000,))
        background = value.filling[0]
        background.background = True
        selected = None
        for tick in range(9):
            short = SimpleNamespace(sid=100 + tick, prompt=[1], background=False, points=[])
            value.filling = [background, short]
            value.fills[short.sid] = [None, False, 0]
            pieces = value._pieces()
            value._note_passed(pieces)
            if pieces[0][0] is background:
                selected = tick
                break
        self.assertEqual(selected, 8)

    def test_selected_prompt_age_resets_and_stale_ages_retire(self):
        value = make()
        value.passed = {0: 8, 1: 3, 2: 2, 99: 12}
        value._note_passed(value._pieces())
        self.assertEqual(value.passed, {0: 0, 1: 4, 2: 3})

    def test_existing_cut_budget_is_preserved(self):
        value = make(lengths=(20, 40))
        value.filling[0].points = [2, 4, 6, 8, 10, 12]
        self.assertEqual(value._pieces()[0][1:], (0, 8))
        self.assertEqual(len([p for p in value.filling[0].points if p < 8]), 3)

    def test_reference_retains_multiple_end_rows_up_to_existing_bound(self):
        value = make(profile="reference", lengths=(1,) * 20)
        self.assertEqual(len(value._pieces()), 16)
        self.assertEqual(len(make(lengths=(1,) * 20)._pieces()), 1)

    def test_every_prompt_advances_once_under_unchanged_row_bound(self):
        value = make(lengths=(30, 35, 41), rows=7)
        took = {s.sid: [] for s in value.filling}
        while value.filling:
            pieces = value._pieces()
            self.assertEqual(len(pieces), 1)
            value._note_passed(pieces)
            stream, start, count = pieces[0]
            self.assertGreater(count, 0)
            self.assertLessEqual(count, 7)
            took[stream.sid] += list(range(start, start + count))
            value.fills[stream.sid][2] += count
            if value.fills[stream.sid][2] == len(stream.prompt):
                value.filling.remove(stream)
        self.assertEqual(took, {0: list(range(30)), 1: list(range(35)), 2: list(range(41))})


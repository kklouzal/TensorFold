"""Actual dense receiver/frame source execution; stdlib only, no SDK math."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
import random
from types import SimpleNamespace
import unittest

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/tensorfold/families/qwen3_5/cuda/multi.py'
TREE = ast.parse(SOURCE.read_bytes())


def fixture():
    nodes = [n for n in TREE.body if isinstance(n, ast.FunctionDef)
             and n.name in {'_unflatten', '_received_windows', '_received_paths'}]
    owner = next(n for n in TREE.body if isinstance(n, ast.ClassDef) and n.name == 'MultiDecoder')
    methods = []
    for n in owner.body:
        if isinstance(n, ast.FunctionDef) and n.name in {'_verify', 'follow'}:
            method = copy.deepcopy(n)
            method.decorator_list = []  # Opaque no_grad provider is outside this host-only scope.
            methods.append(method)
    driver = ast.ClassDef(name='Actual', bases=[], keywords=[], body=methods, decorator_list=[])
    namespace = {'TREE': 1, 'ROUND': 2, 'ADMIT': 1, 'DONE': 3, 'FILL': 4, 'FILLS': 5}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *nodes, driver], type_ignores=[])),
                 '<actual-dense-frame-receiver>', 'exec'), namespace)
    return namespace


class Frames(unittest.TestCase):
    def test_original_legal_slices_match_independent_consuming_encoder_oracle(self):
        api = fixture()
        rng = random.Random(7549)
        for pairs in (False, True):
            for repetition in range(2000):
                expected, flat = [], []
                for group in range(rng.randrange(9)):
                    width = rng.randrange(17)
                    tokens = [rng.randrange(10000) for _ in range(width)]
                    parents = [rng.randrange(-1, row) for row in range(width)]
                    expected.append((tokens, parents) if pairs else tokens)
                    flat.extend([width, *tokens, *(parents if pairs else [])])
                self.assertEqual(api['_unflatten'](flat, pairs, groups=len(expected)), expected)
                self.assertEqual(api['_unflatten'](flat, pairs), expected)

    def test_negative_bool_fractional_truncated_group_count_and_widths_refuse(self):
        api = fixture()
        for flat, pairs, kwargs in (([-1], False, {}), ([-1], True, {}), ([True, 1], False, {}),
                                    ([1.0, 1], False, {}), ([2, 1], False, {}), ([2, 1, 2, -1], True, {}),
                                    ([0], False, {'groups': 0}), ([], True, {'groups': 1}),
                                    ([0], True, {'groups': 1, 'max_rows': 16}),
                                    ([17, *range(17)], False, {'max_rows': 16}),
                                    ([], False, {'groups': True}), ([], False, {'max_rows': 0})):
            with self.subTest(flat=flat, pairs=pairs, kwargs=kwargs), self.assertRaises(ValueError):
                api['_unflatten'](flat, pairs, **kwargs)

    def test_window_integer_token_parent_geometry_and_path_tree_contract(self):
        api = fixture()
        good = [([2, 3, 4, 5], [-1, 0, 0, 2]), ([0], [-1])]
        api['_received_windows'](good, 16, 6)
        api['_received_paths'](good, [[0, 2, 3], [0]])
        for wins in ([([False], [-1])], [([2, True], [-1, 0])], [([2, -1], [-1, 0])],
                     [([6], [-1])], [([2, 3], [-1, True])], [([2, 3], [-1, 1])],
                     [([2, 3], [-1, -1])], [([2], [0])], [([2, 3], [-1])]):
            with self.subTest(wins=wins), self.assertRaises(ValueError):
                api['_received_windows'](wins, 16, 6)
        for paths in ([[]], [[1], [0]], [[0, 3], [0]], [[0, 2, 1], [0]],
                      [[0, True], [0]], [[0, 4], [0]], [[0, -1], [0]]):
            with self.subTest(paths=paths), self.assertRaises(ValueError):
                api['_received_paths'](good, paths)

    def test_actual_verify_bad_received_frame_precedes_masks_deepest_and_target_work(self):
        for flat in ([-1], [1, 2, -1], [1, 2, -1, 1, 3, 0],
                     [1, -1, -1, 1, 3, -1], [1, 2, -1, 2, 3, 4, -1, 2]):
            api = fixture()
            calls = []
            owner = api['Actual']()
            owner.drafts, owner.depth, owner.block, owner.max_rows = False, True, 16, 16
            owner.rank, owner.split, owner.device = 1, True, object()
            owner.w = SimpleNamespace(config=SimpleNamespace(vocab=6))
            owner._masks = lambda *args: self.fail('bad frame reached grammar mutation')
            owner._deepest = lambda *args: self.fail('bad frame reached planning')
            api['_share'] = lambda *args: calls.append('received') or flat
            api['multi_tree_forward'] = lambda *args, **kwargs: self.fail('bad frame reached target work')
            with self.subTest(flat=flat), self.assertRaises(ValueError):
                owner._verify([(100, 2, 2, 0), (101, 2, 3, 0)])
            self.assertEqual(calls, ['received'])

    def test_actual_follow_bad_received_paths_precedes_any_commit_or_constraint_update(self):
        wins = [([2, 3], [-1, 0]), ([4], [-1])]
        for flat in ([1, 0], [-1], [2, 0, 2, 1, 0], [2, 0, 0, 1, 0]):
            api = fixture()
            messages = iter(([2, 2, 100, 2, 2, 0, 101, 2, 4, 0], flat))
            api['_share'] = lambda *args: next(messages)
            owner = api['Actual']()
            owner.device, owner.max_rows = object(), 16
            owner._verify = lambda plan: (wins, object(), object(), [0, 2], [None, None])
            owner._commit = lambda *args: self.fail('bad received path reached target-state mutation')
            with self.subTest(flat=flat), self.assertRaises(ValueError):
                owner.follow()

    def test_actual_follow_valid_received_paths_commit_once_then_clean_empty_stop(self):
        api = fixture()
        messages = iter(([2, 2, 100, 2, 2, 0, 101, 2, 4, 0], [2, 0, 1, 1, 0], []))
        api['_share'] = lambda *args: next(messages)
        owner, calls = api['Actual'](), []
        owner.device, owner.max_rows = object(), 16
        wins = [([2, 3], [-1, 0]), ([4], [-1])]
        owner._verify = lambda plan: (wins, object(), object(), [0, 2], [None, None])
        owner._commit = lambda plan, wins, record, taps, starts, paths: calls.append(paths)
        owner.streams = {100: SimpleNamespace(constraint=None), 101: SimpleNamespace(constraint=None)}
        owner.follow()
        self.assertEqual(calls, [[[0, 1], [0]]])


if __name__ == '__main__':
    unittest.main()

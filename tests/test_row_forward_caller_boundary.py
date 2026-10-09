"""Actual row caller metadata, cache coverage and pre-mutation controls.

Opaque arrays and extracted source methods exercise host boundaries only. Original
shader/arithmetic equivalence is checked separately in the archived source proof;
these controls make no native, numerical, Apple or performance claim.
"""
from __future__ import annotations

import ast
from contextlib import contextmanager
from itertools import islice
from operator import index
from pathlib import Path
import random
import sys
from types import ModuleType, SimpleNamespace
from typing import Sequence
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/tensorfold/kernels/qwen/dense/v1/row_forward.py'


class Array:
    made = []

    def __init__(self, contents=(), dtype=None, *, size=None, shape=None):
        self.contents, self.dtype = tuple(contents), dtype
        self.size = len(self.contents) if size is None else size
        self.shape = (1, 8, self.size, 64) if shape is None else shape
        self.ndim = len(self.shape)
        self.writes = []
        type(self).made.append((self.contents, dtype))

    def __setitem__(self, item, value):
        self.writes.append((item, value))

    def __getitem__(self, item):
        shape = (item.stop - item.start, *self.shape[1:])
        return Array(dtype=self.dtype, shape=shape)


def methods(tree, names):
    selected = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                and node.name in names]
    for node in selected:
        if isinstance(node, ast.FunctionDef):
            node.decorator_list = []
    return selected


@contextmanager
def fixture(*, row_attention=True):
    tree = ast.parse(SOURCE.read_bytes())
    namespace = {'mx': SimpleNamespace(array=Array, int32='I32'), 'Sequence': Sequence, 'Any': object,
                 'integer_index': index, 'islice': islice, 'ROW_ATTENTION': row_attention,
                 'row_matmul': SimpleNamespace(BACKEND=SimpleNamespace(max_rows=128, name='opaque')),
                 '_CHAINS': {width: tuple(range(-1, width - 1)) for width in range(1, 257)},
                 '_chain': lambda parents: list(parents) == list(range(-1, len(parents) - 1))}
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    selected = methods(tree, {'Record', '_Rows', '_stream_inputs', '_rows_forward', '_attend', 'hidden_rows',
                              '_row_integer', '_array_shape', '_commit_inputs', '_commit_validated', 'commit', 'keep_rows',
                              '_implicit_stream_inputs'})
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *selected], type_ignores=[])),
                 '<actual-row-caller-boundaries>', 'exec'), namespace)
    lane_tree = ModuleType('lane_tree')
    lane_tree_tree = ast.parse((SOURCE.parent / 'lane_tree.py').read_bytes())
    tree_namespace = {}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *methods(lane_tree_tree, {'tree_paths'})],
                                                    type_ignores=[])), '<original-tree-paths>', 'exec'), tree_namespace)
    lane_tree.tree_paths = tree_namespace['tree_paths']
    module = ModuleType('tensorfold.kernels.qwen.dense.v1')
    module.lane_tree = lane_tree
    module.row_attention = SimpleNamespace(row_sdpa=lambda *args: ('raw', args))
    module.exact_attention = SimpleNamespace(exact_sdpa=lambda *args: ('exact', args))
    Array.made = []
    with patch.dict(sys.modules, {'tensorfold.kernels.qwen.dense.v1': module}):
        yield namespace


class RowCallers(unittest.TestCase):
    def commit_fixture(self, api, *, n_keep=3):
        window, start = 3, 5
        kv = SimpleNamespace(keys=Array(dtype='BF16', shape=(1, 2, 16, 32)),
                             values=Array(dtype='BF16', shape=(1, 2, 16, 32)), offset=8)
        mutation = []
        def trim(n):
            mutation.append(('trim', n))
            kv.offset -= n
        kv.trim = trim
        class Recurrent(list):
            def advance(self, n):
                mutation.append(('advance', n))
        recurrence = Recurrent([None, None])
        q = Array(dtype='BF16', shape=(1, window, 2, 32))
        v = Array(dtype='BF16', shape=(1, window, 4, 16))
        g, beta = (Array(dtype=dt, shape=(1, window, 4)) for dt in ('F32', 'BF16'))
        state = Array(dtype='F32', shape=(1, 4, 16, 32))
        state_out = Array(dtype='F32', shape=state.shape)
        tails = Array(dtype='BF16', shape=(window, n_keep, 64))
        extra = (Array(dtype='BF16', shape=(1, n_keep, 64)),
                 Array(dtype='BF16', shape=(1, window, 128)), 64, state_out, tails, True)
        cache = [kv, recurrence]
        record = api['Record']()
        record.width, record.start, record.parents = window, start, (-1, 0, 1)
        record.borrowed_cache = tuple(cache)
        record.extend([('kv', Array(dtype='BF16', shape=(1, 2, window, 32)),
                       Array(dtype='BF16', shape=(1, 2, window, 32))),
                      ('gdn', q, q, v, g, beta, state, extra, n_keep)])
        api['ints'] = lambda path: Array(path, 'I32')
        api['mx'].take = lambda array, taken, axis: Array(dtype=array.dtype,
            shape=tuple(len(taken.contents) if at == axis else d for at, d in enumerate(array.shape)))
        module = sys.modules['tensorfold.kernels.qwen.dense.v1']
        def replay(*args):
            mutation.append(('replay', args[-2].contents))
            return state_out
        module.lane_tree.replay_path = replay
        Array.made.clear()
        return cache, record, mutation, state_out

    def test_complete_commit_preflight_refuses_late_layer_before_any_write(self):
        for bad in ('count', 'kind', 'tails', 'heads', 'owner', 'offset'):
            with self.subTest(bad=bad), fixture() as api:
                cache, record, mutations, _ = self.commit_fixture(api)
                if bad == 'count':
                    record.pop()
                elif bad == 'kind':
                    record[1] = ('kv',)
                elif bad == 'tails':
                    extra = (*record[1][7][:4], Array(dtype='BF16', shape=(3, 2, 64)), True)
                    record[1] = (*record[1][:7], extra, 3)
                elif bad == 'heads':
                    record[1][3].shape = (1, 3, 3, 16)
                elif bad == 'owner':
                    record.borrowed_cache = (object(), cache[1])
                else:
                    cache[0].offset = 9
                Array.made.clear()
                with self.assertRaises(ValueError):
                    api['commit'](cache, record, [0, 1], 3, 5)
                self.assertEqual(mutations, [])
                self.assertEqual(cache[0].keys.writes, [])
                self.assertEqual(Array.made, [])
                self.assertFalse(record.failed)

    def test_commit_path_cardinality_types_and_ancestry_are_checked_before_device_work(self):
        for path in ([], [True], [.5], [-1], [3], [0, 0], [0, 2], [1], [0, 1, 2, 2]):
            with self.subTest(path=path), fixture() as api:
                cache, record, mutations, _ = self.commit_fixture(api)
                with self.assertRaises(ValueError):
                    api['commit'](cache, record, path, 3, 5)
                self.assertEqual((mutations, Array.made), ([], []))
        class Oversized(Sequence):
            def __init__(self): self.reads = 0
            def __len__(self): return 1
            def __getitem__(self, at):
                self.reads += 1
                if self.reads > 2:
                    raise AssertionError('unbounded commit path producer')
                return 0
        with fixture() as api:
            cache, record, mutations, _ = self.commit_fixture(api)
            path = Oversized()
            with self.assertRaises(ValueError):
                api['commit'](cache, record, path, 3, 5)
            self.assertEqual(path.reads, 2)
            self.assertEqual((mutations, Array.made), ([], []))

    def test_original_whole_and_partial_commit_operation_order_and_zero_history(self):
        for path, expected in (([0, 1, 2], [('trim', 0), ('advance', 3)]),
                               ([0, 1], [('trim', 1), ('replay', (0, 1)), ('advance', 2)])):
            for history in (0, 3):
                with self.subTest(path=path, history=history), fixture() as api:
                    cache, record, mutations, state_out = self.commit_fixture(api, n_keep=history)
                    api['commit'](cache, record, path, 3, 5)
                    self.assertEqual(mutations, expected)
                    self.assertIs(cache[1][1], state_out)
                    self.assertEqual(cache[1][0].shape, (1, history, 64))
                    self.assertEqual(cache[0].offset, 5 + len(path))

    def test_failed_commit_poison_refuses_retry_without_replacing_primary(self):
        with fixture() as api:
            cache, record, mutations, _ = self.commit_fixture(api)
            primary = KeyboardInterrupt()
            def fail(n): raise primary
            cache[0].trim = fail
            with self.assertRaises(KeyboardInterrupt) as actual:
                api['commit'](cache, record, [0, 1], 3, 5)
            self.assertIs(actual.exception, primary)
            self.assertTrue(record.failed)
            self.assertEqual(mutations, [])
            with self.assertRaises(ValueError):
                api['commit'](cache, record, [0, 1], 3, 5)
            self.assertEqual(mutations, [])

    def test_replay_head_width_covers_whole_simd_lanes_before_any_cache_change(self):
        for width in (16, 33, 63):
            with self.subTest(width=width), fixture() as api:
                cache, record, mutations, _ = self.commit_fixture(api)
                record[1][1].shape = (1, 3, 2, width)
                record[1][6].shape = (1, 4, 16, width)
                record[1][7][3].shape = (1, 4, 16, width)
                with self.assertRaises(ValueError):
                    api['commit'](cache, record, [0, 1], 3, 5)
                self.assertEqual((mutations, Array.made), ([], []))
                self.assertEqual(cache[0].offset, 8)

    def test_commit_requires_untrimmed_window_end_but_whole_then_partial_remains_valid(self):
        with fixture() as api:
            cache, record, mutations, _ = self.commit_fixture(api)
            cache[0].offset = 7
            with self.assertRaises(ValueError):
                api['commit'](cache, record, [0, 1], 3, 5)
            self.assertEqual((mutations, Array.made), ([], []))
        with fixture() as api:
            cache, record, mutations, _ = self.commit_fixture(api)
            api['commit'](cache, record, [0, 1, 2], 3, 5)
            api['commit'](cache, record, [0, 1], 3, 5)
            self.assertEqual(cache[0].offset, 7)
            before = list(mutations)
            with self.assertRaises(ValueError):
                api['commit'](cache, record, [0, 1], 3, 5)
            self.assertEqual(mutations, before)

    def test_prefix_commit_preserves_delegated_packed_and_rotating_storage(self):
        for storage in ('packed', 'rotating'):
            with self.subTest(storage=storage), fixture() as api:
                cache, record, mutations, state_out = self.commit_fixture(api)
                if storage == 'packed':
                    cache[0].bits = 4
                    cache[0].keys = (object(), object(), object())
                    cache[0].values = (object(), object(), object())
                else:
                    cache[0].keys = Array(dtype='F16', shape=(1, 2, 4, 32))
                    cache[0].values = Array(dtype='F16', shape=(1, 2, 4, 32))
                resident = cache[0].keys, cache[0].values
                Array.made.clear()
                api['commit'](cache, record, [0, 1], 3, 5)
                self.assertEqual(mutations, [('trim', 1), ('replay', (0, 1)), ('advance', 2)])
                self.assertEqual(cache[0].offset, 7)
                self.assertIs(cache[0].keys, resident[0])
                self.assertIs(cache[0].values, resident[1])
                self.assertIs(cache[1][1], state_out)

    def test_relocation_still_requires_addressable_full_resident_buffers_before_write(self):
        for storage in ('packed', 'rotating'):
            with self.subTest(storage=storage), fixture() as api:
                cache, record, mutations, _ = self.commit_fixture(api)
                record.parents = (-1, -1, 1)
                if storage == 'packed':
                    cache[0].bits = 4
                    cache[0].keys = (object(), object(), object())
                    cache[0].values = (object(), object(), object())
                else:
                    cache[0].keys = Array(dtype='BF16', shape=(1, 2, 4, 32))
                    cache[0].values = Array(dtype='BF16', shape=(1, 2, 4, 32))
                Array.made.clear()
                with self.assertRaises(ValueError):
                    api['commit'](cache, record, [1, 2], 3, 5)
                self.assertEqual((mutations, Array.made), ([], []))
                self.assertEqual(cache[0].offset, 8)
                self.assertFalse(record.failed)

    def test_original_replay_and_whole_commit_keep_their_different_state_dtype_contracts(self):
        for dtype in ('F16', 'BF16'):
            for path in ([0, 1, 2], [0, 1]):
                with self.subTest(dtype=dtype, path=path), fixture() as api:
                    cache, record, mutations, state_out = self.commit_fixture(api)
                    record[1][6].dtype = dtype
                    cache[1][1] = record[1][6]
                    replayed = Array(dtype=dtype, shape=record[1][6].shape)
                    def replay(*args):
                        self.assertIs(args[5], record[1][6])
                        mutations.append(('replay', args[-2].contents))
                        return replayed
                    sys.modules['tensorfold.kernels.qwen.dense.v1'].lane_tree.replay_path = replay
                    api['commit'](cache, record, path, 3, 5)
                    self.assertIs(cache[1][1], state_out if len(path) == 3 else replayed)
                    self.assertEqual(cache[1][1].dtype, 'F32' if len(path) == 3 else dtype)

    def test_unsigned_state_span_and_sdk_tail_storage_are_not_given_signed_array_floors(self):
        with fixture() as api:
            cache, record, mutations, _ = self.commit_fixture(api)
            heads = 1 << 22
            entry = list(record[1])
            entry[3].shape = (1, 3, heads, 16)
            entry[4].shape = entry[5].shape = (1, 3, heads)
            entry[6].shape = (1, heads, 16, 32)
            extra = list(entry[7])
            extra[3].shape = entry[6].shape
            extra[2] = (1 << 30)
            extra[0].shape = (1, 3, extra[2])
            extra[4].shape = (3, 3, extra[2])
            entry[7] = tuple(extra)
            record[1] = tuple(entry)
            Array.made.clear()
            api['commit'](cache, record, [0, 1, 2], 3, 5)
            self.assertEqual(mutations, [('trim', 0), ('advance', 3)])

    def test_final_signed_position_keeps_python_sdk_window_end_without_narrowing(self):
        with fixture() as api:
            start = (1 << 31) - 1
            calls = []
            item = SimpleNamespace(keys=(object(),), values=(object(),), offset=start + 1,
                                   trim=lambda count: calls.append(count))
            record = api['Record']()
            record.start, record.width, record.parents = start, 1, (-1,)
            record.borrowed_cache = (item,)
            record.append(('kv', Array(shape=(1, 1, 1, 32)), Array(shape=(1, 1, 1, 32))))
            api['commit']([item], record, [0], 1, start)
            self.assertEqual(calls, [0])

    def test_ordered_forests_absolute_positions_and_records_independent_oracle(self):
        rng = random.Random(411050)
        with fixture() as api:
            for _ in range(1000):
                widths = [rng.randrange(1, 17) for _ in range(rng.randrange(1, 5))]
                parents = [[rng.randrange(-4, row) for row in range(width)] for width in widths]
                starts = [rng.randrange(0, 500) for _ in widths]
                expected = []
                for stream, start in zip(parents, starts):
                    for row in range(len(stream)):
                        depth, parent = 0, stream[row]
                        while parent >= 0:
                            depth += 1
                            parent = stream[parent]
                        expected.append(start + depth)
                rows = api['_Rows'](parents, starts)
                self.assertEqual(rows.positions.contents, tuple(expected))
                self.assertEqual(rows.widths, widths)
                self.assertEqual(rows.offsets, [sum(widths[:at]) for at in range(len(widths))])
                self.assertEqual([record.parents for record in rows.records], [tuple(p) for p in parents])
                self.assertEqual([record.start for record in rows.records], starts)

    def test_parent_start_position_and_empty_refusals_precede_device_array(self):
        cases = [([], []), ([[]], [0]), ([[-1]], []), ([[-1]], [True]), ([[-1]], [.5]),
                 ([[-1]], [-1]), ([[-1]], [1 << 31]), ([[-1, 0]], [(1 << 31) - 1]),
                 ([[-1, True]], [0]), ([[-1, .5]], [0]), ([[-1, 1]], [0]),
                 ([[-1, 2]], [0]), ([[-1] * 129], [0]), ([[-(1 << 31) - 1]], [0])]
        for parents, starts in cases:
            with self.subTest(parents=parents, starts=starts), fixture() as api:
                with self.assertRaises(ValueError):
                    api['_Rows'](parents, starts)
                self.assertEqual(Array.made, [])
        with fixture() as api:
            class Integral:
                def __index__(self):
                    return 7
            self.assertEqual(api['_Rows']([[-1]], [Integral()]).starts, [7])
            self.assertEqual(api['_Rows']([[-1]], [(1 << 31) - 1]).positions.contents, ((1 << 31) - 1,))

    def test_declared_parent_start_and_outer_producers_are_bounded(self):
        class Oversized(Sequence):
            def __init__(self, value):
                self.value, self.reads = value, 0
            def __len__(self):
                return 2
            def __getitem__(self, at):
                self.reads += 1
                if self.reads > 3:
                    raise AssertionError('producer over-consumed beyond admitted count plus one')
                return self.value
        for which in ('parents', 'starts', 'outer'):
            with self.subTest(which=which), fixture() as api:
                producer = Oversized(-1)
                with self.assertRaises(ValueError):
                    if which == 'parents':
                        api['_Rows']([producer], [0])
                    elif which == 'starts':
                        api['_Rows']([[-1], [-1]], producer)
                    else:
                        api['_stream_inputs'](producer, 2)
                self.assertEqual(producer.reads, 3)
                self.assertEqual(Array.made, [])

    def test_actual_forward_bounds_outer_producers_before_model_or_device_work(self):
        class Oversized(Sequence):
            def __init__(self):
                self.reads = 0
            def __len__(self):
                return 1
            def __getitem__(self, at):
                self.reads += 1
                if self.reads > 2:
                    raise AssertionError('outer stream producer over-consumed')
                return [1]
        with fixture() as api:
            producer = Oversized()
            core = SimpleNamespace(layers=[0], embed_tokens=lambda _: self.fail('oversized producer reached model'))
            with self.assertRaises(ValueError):
                api['_rows_forward'](core, producer, [[-1]], [[0]], [0])
            self.assertEqual(producer.reads, 2)
            self.assertEqual(Array.made, [])

    def test_actual_forward_refuses_incomplete_extra_and_malformed_before_embedding(self):
        cases = [([[1]], [[-1]], [[]], [0]), ([[1]], [[-1]], [[0, 1, 2]], [0]),
                 ([[1]], [[-1]], [[0, 1]], [-1]), ([[1]], [[True]], [[0, 1]], [0]),
                 ([], [], [], []), ([[1, 2]], [[-1, 0]], [[0, 1]], [(1 << 31) - 1]),
                 ([[1]], [[-1], [-1]], [[0, 1]], [0])]
        for args in cases:
            with self.subTest(args=args), fixture() as api:
                core = SimpleNamespace(layers=[0, 1], embed_tokens=lambda _: self.fail('bad caller reached embedding'))
                api['_token_ids'] = lambda _: self.fail('bad caller narrowed tokens')
                with self.assertRaises(ValueError):
                    api['_rows_forward'](core, *args)

    def test_nonchain_refusal_precedes_any_attention_mutation_and_target_projection(self):
        with fixture(row_attention=False) as api:
            core = SimpleNamespace(layers=[0], embed_tokens=lambda _: self.fail('tree reached target projection'))
            api['_token_ids'] = lambda _: self.fail('tree narrowed tokens')
            with self.assertRaises(NotImplementedError):
                api['_rows_forward'](core, [[1, 2, 3]], [[-1, 0, 0]], [[0]], [4])
            record = []
            cache = SimpleNamespace(update_and_fetch=lambda *args: self.fail('unsupported tree changed cache'))
            with self.assertRaises(NotImplementedError):
                api['_attend'](SimpleNamespace(scale=1), None, None, None, cache, (-1, 0, 0), False, record)
            self.assertEqual(record, [])

    def test_valid_actual_forward_visits_every_layer_and_preserves_lazy_window_identity(self):
        with fixture(row_attention=False) as api:
            calls, lazy = [], Array(size=2)
            norm = SimpleNamespace(weight='weight', eps='eps')
            layers = [SimpleNamespace(input_layernorm=norm, post_attention_layernorm=norm,
                                      self_attn=SimpleNamespace(o_proj='out'),
                                      mlp=SimpleNamespace(down_proj='down')) for _ in range(2)]
            core = SimpleNamespace(layers=layers, norm=norm,
                                   embed_tokens=lambda tokens: calls.append(('embed', tokens)) or 'hidden')
            # This control owns the unchanged per-layer loop/lazy token path.
            # Complete layer/cache and token admission has independent actual
            # source controls in test_row_forward_forward_boundary.py.
            api['_forward_cache_inputs'] = lambda layers, caches, rows: caches
            api['_forward_windows'] = lambda core, windows, widths: windows
            api['_token_ids'] = lambda windows: calls.append(('tokens', windows[0])) or 'ids'
            api['add_norm'] = lambda hidden, pending, *args: (hidden, 'normed')
            api['_attention'] = lambda attn, x, items, rows: calls.append(('attention', tuple(items))) or 'attention'
            api['project'] = lambda module, x: x
            api['mlp_act'] = lambda x: x
            api['_gate_up'] = lambda mlp, x: x
            api['mx'].async_eval = lambda *args: calls.append(('async', args))
            result, rows = api['_rows_forward'](core, [lazy], [[-1, 0]], [['a', 'b']], [5])
            self.assertEqual(result, 'normed')
            self.assertEqual(calls[:2], [('tokens', lazy), ('embed', 'ids')])
            self.assertEqual([call for call in calls if call[0] == 'attention'],
                             [('attention', ('a',)), ('attention', ('b',))])
            self.assertEqual(rows.positions.contents, (5, 6))

    def test_implicit_start_preserves_value_for_the_owned_integer_admission(self):
        with fixture() as api:
            api['_rows_forward'] = lambda core, windows, parents, caches, starts, **kw: (None, SimpleNamespace(records=starts))
            cache = SimpleNamespace(keys=None, offset=.75)
            # The facade must not turn an invalid fraction into a valid zero.
            self.assertEqual(api['hidden_rows'](SimpleNamespace(layers=[0]), [[1]], [[cache]])[1], [.75])


if __name__ == '__main__':
    unittest.main()

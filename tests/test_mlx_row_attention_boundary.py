"""Actual host/parent/offset controls; no MLX/native/numerical execution."""
from __future__ import annotations

import ast
import math
from itertools import islice
from operator import index
from pathlib import Path
import random
from types import SimpleNamespace
from typing import Sequence
import unittest

ROOT = Path(__file__).resolve().parents[1]
TREE = ast.parse((ROOT / 'src/tensorfold/kernels/qwen/dense/v1/row_attention.py').read_bytes())


class Array:
    allocations = []

    def __init__(self, contents=None, dtype='BF16', *, shape=None):
        self.contents, self.dtype = contents, dtype
        self.shape = tuple(shape) if shape is not None else (len(contents),)
        self.ndim = len(self.shape)
        if contents is not None:
            self.allocations.append((tuple(contents), dtype))


def fixture():
    launches = []
    Array.allocations = []

    def native(name, *parameters):
        def launch(**arguments):
            launches.append((name, parameters, arguments))
            return [Array(dtype=dtype, shape=shape) for shape, dtype in
                    zip(arguments['output_shapes'], arguments['output_dtypes'])]
        return launch

    namespace = {'mx': SimpleNamespace(array=Array, bfloat16='BF16', int32='I32', float32='F32', contiguous=lambda x: x),
                 'index': index, 'math': math, 'islice': islice, 'CK': 128, 'SPLIT': 4, 'BLK': 4,
                 'ints': lambda contents: Array(contents, 'I32'),
                 '_const': lambda key, make: make(), '_kernel': native}
    methods = [n for n in TREE.body if isinstance(n, ast.FunctionDef)
               and n.name in {'_parents', '_paths', 'paths_of', 'row_sdpa'}]
    future = ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0)
    exec(compile(ast.fix_missing_locations(ast.Module(body=[future, *methods], type_ignores=[])),
                 '<actual-row-attention-boundary>', 'exec'), namespace)
    return namespace, launches


class Boundary(unittest.TestCase):
    def test_original_ordered_forest_scalar_oracle_and_all_negative_roots(self):
        api, _ = fixture()
        rng = random.Random(84013)
        for count in range(0, 129):
            for repetition in range(8):
                parents = [rng.randrange(-3, row) for row in range(count)]
                expected = []
                for row in range(count):
                    reverse, parent = [row], parents[row]
                    while parent >= 0:
                        reverse.append(parent)
                        parent = parents[parent]
                    expected.append(reverse[::-1])
                depths, paths = api['paths_of'](parents)
                self.assertEqual(paths, expected)
                self.assertEqual(depths, [len(path) - 1 for path in expected])

    def test_parent_references_and_mutation_refuse_before_expansion(self):
        api, _ = fixture()
        original = api['_paths']
        api['_paths'] = lambda parents: self.fail('invalid parents must not expand')
        for parents in ((0,), (-1, 1), (-1, 2), (-1, True), (-1, 0.0), (-1, .5)):
            with self.subTest(parents=parents), self.assertRaises(ValueError):
                api['paths_of'](parents)
        class Changed(Sequence):
            def __len__(self):
                return 2
            def __getitem__(self, at):
                if at < 3:
                    return -1
                raise IndexError
        with self.assertRaises(ValueError):
            api['paths_of'](Changed())
        api['_paths'] = original

    def test_oversized_declared_sequence_is_consumed_only_through_count_plus_one(self):
        class Oversized(Sequence):
            def __init__(self):
                self.reads = 0
            def __len__(self):
                return 2
            def __getitem__(self, at):
                self.reads += 1
                if self.reads > 3:
                    raise AssertionError('parent iterator exceeded admitted cardinality plus one')
                return -1
        for raw in (False, True):
            api, launches = fixture()
            api['_paths'] = lambda parents: self.fail('oversized parent sequence must not expand paths')
            parents = Oversized()
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                if raw:
                    api['row_sdpa'](Array(shape=(1, 8, 2, 64)), Array(shape=(1, 2, 130, 64)),
                                    Array(shape=(1, 2, 130, 64)), .0625, 128, parents)
                else:
                    api['paths_of'](parents)
            self.assertEqual(parents.reads, 3)
            self.assertEqual(Array.allocations, [])
            self.assertEqual(launches, [])

    def test_valid_query_dtypes_negative_or_zero_finite_scale_and_exact_metadata_launch_order(self):
        api, launches = fixture()
        for query_type in ('BF16', 'F16', 'F32', 'I32', 'U32'):
            for scale in (0.0625, -0.0625, -0.0):
                queries = Array(dtype=query_type, shape=(1, 8, 3, 64))
                keys = Array(shape=(1, 2, 260, 64))
                values = Array(shape=keys.shape)
                output = api['row_sdpa'](queries, keys, values, scale, 129, (-100, 0, 0))
                self.assertEqual((output.shape, output.dtype), (queries.shape, query_type))
                partial, merge = launches[-2:]
                self.assertEqual((partial[0], merge[0]), ('partial', 'merge'))
                self.assertEqual(partial[2]['grid'], (512, 2, 6))
                self.assertEqual(partial[2]['inputs'][-1].contents, [129, 3, 260, 2, 2])
                self.assertEqual(partial[2]['inputs'][3].contents, [0, 1, 1])
                self.assertEqual(partial[2]['inputs'][4].contents, [0, 0, 0, 1, 0, 2])
                self.assertEqual(partial[2]['inputs'][5].contents, [scale])

    def test_metadata_rank_head_dtype_capacity_position_scale_refuse_before_allocation(self):
        variants = [
            ((1, 0, 3, 64), (1, 2, 260, 64), None, 129, (-1, 0, 1), .0625),
            ((1, 8, 0, 64), (1, 2, 260, 64), None, 129, (), .0625),
            ((1, 8, 3, 64), (1, 0, 260, 64), None, 129, (-1, 0, 1), .0625),
            ((2, 8, 3, 64), (1, 2, 260, 64), None, 129, (-1, 0, 1), .0625),
            ((1, 8, 3, 63), (1, 2, 260, 63), None, 129, (-1, 0, 1), .0625),
            ((1, 8, 3, 64), (1, 2, 260, 128), None, 129, (-1, 0, 1), .0625),
            ((1, 8, 3, 64), (1, 2, 130, 64), None, 129, (-1, 0, 1), .0625),
            ((1, 8, 3, 64), (1, 2, 260, 64), 'F32', 129, (-1, 0, 1), .0625),
            ((1, 8, 3, 64), (1, 2, 260, 64), None, -1, (-1, 0, 1), .0625),
            ((1, 8, 3, 64), (1, 2, 260, 64), None, True, (-1, 0, 1), .0625),
            ((1, 8, 3, 64), (1, 2, 260, 64), None, 129.0, (-1, 0, 1), .0625),
            ((1, 8, 3, 64), (1, 2, 260, 64), None, 129, (-1, 0, 1), math.nan),
            ((1, 8, 3, 64), (1, 2, 260, 64), None, 129, (-1, 0, 1), math.inf),
            ((1, 8, 3, 64), (1, 2, 260, 64), None, 129, (-1, 0, 1), 1e100),
        ]
        for qshape, kshape, kd, start, parents, scale in variants:
            api, launches = fixture()
            queries = Array(shape=qshape)
            keys = Array(dtype=kd or 'BF16', shape=kshape)
            values = Array(shape=kshape)
            with self.subTest(qshape=qshape, kshape=kshape, start=start, scale=scale), self.assertRaises(ValueError):
                api['row_sdpa'](queries, keys, values, scale, start, parents)
            self.assertEqual(Array.allocations, [])
            self.assertEqual(launches, [])

    def test_padded_signed32_and_static_shared_limits_refuse_before_paths_or_native_work(self):
        variants = [((1, 9, 1, 64), (1, 1, 129, 64), 128),
                    ((1, 8, 1, 1024), (1, 1, 129, 1024), 128),
                    ((1, 8, 1, 64), (1, 1, (1 << 31) - 1, 64), (1 << 31) - 2),
                    ((1, 8, 1, 64), (1, 1, 1 << 31, 64), 0),
                    ((1, 8, 1 << 24, 64), (1, 1, 1 << 25, 64), 0)]
        for qshape, kshape, start in variants:
            api, launches = fixture()
            api['_paths'] = lambda parents: self.fail('bad padded span must not expand paths')
            with self.subTest(qshape=qshape, kshape=kshape), self.assertRaises(ValueError):
                api['row_sdpa'](Array(shape=qshape), Array(shape=kshape), Array(shape=kshape), .0625, start, (-1,))
            self.assertEqual(launches, [])
            self.assertEqual(Array.allocations, [])


if __name__ == '__main__':
    unittest.main()

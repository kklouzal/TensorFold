"""Self-contained boundary controls; numerical CUDA contracts live in tests/cuda."""
from __future__ import annotations

import ast
import builtins
import copy
import math
import operator
from pathlib import Path
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'src/tensorfold/families/nemotron_h/cuda/sampler.py'


def functions(path):
    return {n.name: n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef)}


class Metadata:
    def __init__(self, shape, dtype='float32', device='cuda:0', *, contiguous=True, stride=1):
        self.shape, self.dtype, self.device = (shape,) if isinstance(shape, int) else shape, dtype, device
        self.ndim, self.is_cuda = len(self.shape), device.startswith('cuda')
        self.contiguous_flag, self.step = contiguous, stride

    def numel(self):
        return math.prod(self.shape)

    def is_contiguous(self):
        return self.contiguous_flag

    def stride(self, axis):
        return self.step

    def contiguous(self):
        return self

    def __getitem__(self, index):
        if index is None:
            shape = (1, *self.shape)
        elif isinstance(index, int):
            shape = self.shape[1:]
        elif isinstance(index, slice):
            shape = (len(range(self.shape[0])[index]), *self.shape[1:])
        else:
            raise AssertionError('metadata-only index')
        return Metadata(shape, self.dtype, self.device)

    def __add__(self, offset):
        return self

    def view(self, dtype):
        size = {'float32': 4, 'int32': 4, 'int64': 8}
        shape = (*self.shape[:-1], self.shape[-1] * size[self.dtype] // size[dtype])
        return Metadata(shape, dtype, self.device)


class SourceControls(unittest.TestCase):
    def setUp(self):
        original = builtins.__import__
        def guarded(name, *a, **kw):
            if name.split('.')[0] in {'torch', 'numpy', 'triton', 'mlx', 'cuda', 'cupy', 'ctypes', 'tensorfold'}:
                raise AssertionError('numerical/native imports forbidden: ' + name)
            return original(name, *a, **kw)
        owner = patch.object(builtins, '__import__', guarded)
        owner.start()
        self.addCleanup(owner.stop)
        self.tree = functions(SOURCE)
        torch = SimpleNamespace(Tensor=Metadata, **{name: name for name in
                                ('float16', 'bfloat16', 'float32', 'float64', 'int32', 'int64')})
        self.scope = {'torch': torch, 'operator': operator, 'math': math, 'MARGIN': 8}
        names = ('token_list', '_outputs', '_buffers', 'candidate_count')
        module = ast.Module([copy.deepcopy(self.tree[n]) for n in names], [])
        exec(compile(ast.fix_missing_locations(module), str(SOURCE), 'exec'), self.scope)
        self.logits, self.out = Metadata((2, 64)), Metadata((2,), 'int32')
        self.meta = Metadata((4,), 'int32')
        self.params = SimpleNamespace(seed=Metadata((1,), 'int64'), fp=Metadata((3,), 'float64'))

    def invoke(self, **changed):
        args = dict(logits=self.logits, meta=self.meta, params=self.params, out=self.out,
                    offset=0, prob=None, id_map=None, cuda=True)
        args.update(changed)
        return self.scope['_buffers'](**args)

    def test_valid_metadata_and_legal_strided_outputs(self):
        self.assertEqual(self.invoke(), (2, 64))
        self.assertEqual(self.invoke(out=Metadata((2,), 'int64', contiguous=False, stride=2),
                                    prob=Metadata((2,), contiguous=False, stride=3)), (2, 64))
        self.assertEqual(self.invoke(id_map=Metadata((64,), 'int64', contiguous=False)), (2, 64))

    def test_malformed_score_output_and_parameter_metadata_refuse(self):
        for changed in ({'logits': Metadata((2, 64), 'int32')}, {'logits': Metadata((64,))},
                        {'logits': Metadata((2, 0))}, {'out': Metadata((1,), 'int32')},
                        {'out': Metadata((2,), 'float32')}, {'out': Metadata((2,), 'int32', 'cuda:1')},
                        {'out': Metadata((2,), 'int32', stride=0)}, {'meta': Metadata((0,), 'int32')},
                        {'meta': Metadata((4,), 'float32')}, {'meta': Metadata((4,), 'int32', contiguous=False)},
                        {'offset': True}, {'offset': 0.5}, {'prob': Metadata((2,), 'int32')},
                        {'prob': Metadata((3,))}, {'prob': Metadata((2,), device='cuda:1')},
                        {'params': SimpleNamespace(seed=Metadata((2,), 'int64'), fp=self.params.fp)},
                        {'params': SimpleNamespace(seed=self.params.seed, fp=Metadata((3,), 'float32'))}):
            with self.subTest(changed=changed), self.assertRaises(ValueError):
                self.invoke(**changed)

    def test_map_shape_dtype_device_and_row_candidate_geometry(self):
        for table in (Metadata((63,), 'int64'), Metadata((64,), 'float32'),
                      Metadata((64,), 'int64', 'cuda:1'), Metadata((2, 64), 'int64')):
            with self.assertRaises(ValueError):
                self.invoke(id_map=table)
        self.assertEqual(self.invoke(id_map=Metadata((2, 64), 'int32'), row_ids=True), (2, 64))

    def test_empty_rows_do_not_inspect_unused_position_policy_or_map(self):
        self.assertEqual(self.invoke(logits=Metadata((0, 64)), out=Metadata((0,), 'int32'),
                                    meta=None, params=None, offset=None, id_map=None), (0, 64))

    def test_model_map_startup_bounds_before_allocation(self):
        check = self.scope['token_list']
        self.assertEqual(check(reversed(range(64)), 128, 64), list(reversed(range(64))))
        for values in ([], [0] * 64, list(range(63)), [True] + list(range(1, 64)),
                       [0.0] + list(range(1, 64)), [-1] + list(range(1, 64)),
                       [128] + list(range(1, 64)), [(1 << 31)] + list(range(1, 64))):
            with self.assertRaises(ValueError):
                check(values, 128, 64)
        self.assertEqual(check(range(128), 256, 128), list(range(128)))

    def test_duplicate_and_overlong_iterators_stop_before_unbounded_growth(self):
        consumed = []
        def duplicates():
            for _ in range(10000):
                consumed.append(0)
                yield 0
        with self.assertRaises(ValueError):
            self.scope['token_list'](duplicates(), 64, 64)
        self.assertEqual(len(consumed), 2)
        consumed.clear()
        def excess():
            for token in range(10000):
                consumed.append(token)
                yield token
        with self.assertRaises(ValueError):
            self.scope['token_list'](excess(), 64, 64)
        self.assertEqual(len(consumed), 65)

    def test_candidate_counts_cover_declared_split_modes(self):
        count = self.scope['candidate_count']
        def policy(k):
            return SimpleNamespace(top_k=k, temperature=0.8)
        self.assertEqual(count(128, None, minimum=28), 28)
        self.assertEqual(count(128, policy(40), minimum=28), 48)
        self.assertEqual(count(128, policy(0), minimum=28), 128)
        self.assertEqual(count(64, policy(1000), minimum=28), 64)
        with self.assertRaises(ValueError):
            count(1024, policy(249), minimum=28)
        for width, minimum in ((0, 0), (True, 0), (64, -1), (64, False)):
            with self.assertRaises(ValueError):
                count(width, None, minimum=minimum)


    def test_boundary_repair_integer_membership_oracle(self):
        # Combinatorial candidate membership only; no emulated floating/GPU execution.
        rng = random.Random(137)
        for width in (2, 7, 16, 64):
            for count in range(1, min(width, 28) + 1):
                values = [rng.randrange(-2, 3) for _ in range(width)]
                ids = rng.sample(range(1000), width)
                original = sorted(range(width), key=lambda col: (-values[col], -col))[:count]
                rng.shuffle(original)
                slots = [values[col] for col in original]
                edge = min(slots)
                exact = sorted(range(width), key=lambda col: (-values[col], ids[col]))[:count]
                above = sum(v > edge for v in slots)
                seen = 0
                repaired = []
                for col, value in zip(original, slots):
                    if value == edge:
                        repaired.append(exact[above + seen])
                        seen += 1
                    else:
                        repaired.append(col)
                self.assertEqual(set(repaired), set(exact))
                self.assertEqual([values[col] for col in repaired], slots)


if __name__ == '__main__':
    unittest.main()

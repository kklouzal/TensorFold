"""Check accepted-row metadata before CUDA indexing without importing an SDK."""
from __future__ import annotations

import ast
import copy
from numbers import Integral
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'src/tensorfold/cuda/logprobs.py'


class Tensor:
    def __init__(self, shape=(3, 10), *, cuda=True, inner_stride=1):
        self.shape, self.is_cuda = shape, cuda
        self.ndim, self.device, self.inner_stride = len(shape), 'cuda:0', inner_stride

    def stride(self, axis):
        return self.inner_stride if axis == 1 else self.shape[1]


def collector(top=0):
    return SimpleNamespace(top=top, rows={}, add=lambda *args: (_ for _ in ()).throw(
        AssertionError('collector mutated before metadata proof')))


def capture_without_device_work():
    calls = []

    def forbidden(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError('device work started')

    torch = SimpleNamespace(Tensor=Tensor, tensor=forbidden, empty=forbidden,
                            long='long', float32='float32')
    namespace = {'torch': torch, 'Integral': Integral,
                 'tr': SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b)}
    tree = ast.parse(SOURCE.read_bytes())
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('capture', '_validate_capture', '_wide_offsets')]
    for node in nodes:
        node.decorator_list = []
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[])),
                 str(SOURCE), 'exec'), namespace)
    return namespace, calls


class Boundary(unittest.TestCase):
    def test_malformed_metadata_rejected_before_device_or_collector_work(self):
        cases = []

        def valid():
            return [Tensor(), [1, 2, 3], [4, 5, 6], collector(), None]

        for index, value in [(0, None), (0, Tensor((30,))), (0, Tensor((1, 3, 10))),
                             (0, Tensor((3, 0))), (0, Tensor(cuda=False)),
                             (0, Tensor(inner_stride=2)), (0, Tensor((2, 10))),
                             (1, [-1, 2, 3]), (1, [1, 2, 10]), (1, [1, True, 3]),
                             (1, [1, 2.0, 3]), (1, [1, 2, '3']),
                             (2, [4, 5]), (2, [4, 5, 6, 7]), (2, [4, -1, 6]),
                             (2, [4, True, 6]), (2, [4, 5.0, 6]),
                             (3, collector(-1)), (3, collector(True)), (3, collector(1.0)),
                             (4, [0, 1]), (4, [0, 1, 2, 2]), (4, [0, -1, 2]),
                             (4, [0, 1, 3]), (4, [0, True, 2]), (4, [0, 1.0, 2])]:
            args = valid()
            args[index] = value
            cases.append(args)
        # Full-batch proof must precede chunking: invalid tail cannot leave
        # already-published collector rows or poison the device context.
        cases.append([Tensor((3, 3_000_000)), [1, 2, 3_000_000], [4, 5, 6], collector(), None])
        cases.append([Tensor((3, 3_000_000)), [1, 2, 3], [4, 5, 6], collector(), [0, 1, 3]])
        for args in cases:
            with self.subTest(args=args):
                namespace, calls = capture_without_device_work()
                with self.assertRaises(ValueError):
                    namespace['capture'](*args)
                self.assertEqual(calls, [])
                self.assertEqual(args[3].rows, {})
        self.assertEqual(len(cases), 28)

    def test_valid_metadata_and_noncontiguous_outer_rows_are_preserved(self):
        namespace, calls = capture_without_device_work()
        validate = namespace['_validate_capture']
        for top in (0, 3, 100):
            validate(Tensor(), [0, 9, 3], [0, 5, 10**30], collector(top), None)
            validate(Tensor((20, 10)), [0, 9, 3], [4, 4, 6], collector(top), [19, 0, 19])
            validate(Tensor((0, 10)), [], [], collector(top), [])
        self.assertEqual(calls, [])

    def test_disabled_and_empty_collection_do_not_inspect_logits(self):
        namespace, calls = capture_without_device_work()
        namespace['capture'](None, [1], None, None)
        namespace['capture'](None, [], None, collector())
        self.assertEqual(calls, [])

    def test_valid_collection_reaches_original_device_allocation(self):
        namespace, calls = capture_without_device_work()
        with self.assertRaisesRegex(AssertionError, 'device work started'):
            namespace['capture'](Tensor(), [0, 9, 3], [4, 5, 6], collector())
        self.assertEqual(calls, [(((3, 1, 2),), {'dtype': 'float32', 'device': 'cuda:0'})])

    def test_pointer_width_includes_masked_lanes_and_scratch_offsets(self):
        namespace, calls = capture_without_device_work()
        wide = namespace['_wide_offsets']
        # Express the independent oracle by traversing each boundary, not
        # by reusing the implementation's combined maximum.
        for rows in (1, 2, 3, 2048, 2**20):
            for stride in (0, 1, 248_128, 2**30, 2**31 - 1, 2**31):
                for tiles in (1, 3, 243, 2**20, 2**21):
                    padded = 1 << (tiles - 1).bit_length()
                    expected = any(v > 2**31 - 1 for v in (
                        (rows - 1) * stride + 1024 * (tiles - 1) + 1023,
                        ((rows - 1) * tiles + tiles - 1) * 2 + 1,
                        ((rows - 1) * tiles + padded - 1) * 2 + 1))
                    self.assertEqual(wide(rows, stride, tiles), expected)
        self.assertFalse(wide(2048, 248_128, 243))
        self.assertTrue(wide(2, 2**31, 1))
        self.assertFalse(wide(1, 0, 2**21))
        self.assertTrue(wide(1, 0, 2**21 + 1))
        self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()

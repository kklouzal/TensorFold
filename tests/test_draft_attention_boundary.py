"""DFlash pointer-table metadata boundaries without an accelerator SDK."""
from __future__ import annotations

import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest


SOURCE = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/qwen3_5/cuda/draft_attention.py'


class Tensor:
    def __init__(self, shape, dtype='bf16', device='cuda:0', contiguous=True, ptr=16):
        self.shape, self.dtype, self.device = tuple(shape), dtype, device
        self.ndim, self.is_cuda, self.contiguous, self.ptr = len(shape), device.startswith('cuda'), contiguous, ptr

    def is_contiguous(self):
        return self.contiguous

    def data_ptr(self):
        return self.ptr


class Table(Tensor):
    def __init__(self, values, dtype, device='cpu'):
        super().__init__((len(values),), dtype, device)
        self.values = list(values)

    def pin_memory(self):
        return self

    def to(self, destination, **kwargs):
        self.device = destination if destination.startswith('cuda') else self.device
        self.dtype = destination if not destination.startswith('cuda') else self.dtype
        return self

    def __getitem__(self, key):
        return Table(self.values[key], self.dtype, self.device)


def wrappers(path, *, allow_allocations=True):
    calls = []

    def empty(shape, *, dtype, device):
        if not allow_allocations:
            raise AssertionError('invalid metadata crossed allocation boundary')
        calls.append(('empty', tuple(shape), dtype, device))
        return Tensor(shape, dtype, device, ptr=16 * len(calls))

    def table(values, *, dtype):
        if not allow_allocations:
            raise AssertionError('invalid metadata crossed pointer-table construction')
        calls.append(('table', tuple(values), dtype))
        return Table(values, dtype)

    class Kernel:
        def __init__(self, name):
            self.name = name

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                if not allow_allocations:
                    raise AssertionError('invalid metadata crossed kernel launch')
                def describe(value):
                    if isinstance(value, Table):
                        return ('table', value.values, value.dtype)
                    if isinstance(value, Tensor):
                        return ('tensor', value.shape, value.dtype, value.device, value.ptr)
                    return value
                calls.append(('kernel', self.name, grid, tuple(describe(v) for v in args), kwargs))
            return launch

    namespace = {'torch': SimpleNamespace(Tensor=Tensor, bfloat16='bf16', int64='int64', int32='int32',
                                          tensor=table, empty=empty),
                 'triton': SimpleNamespace(next_power_of_2=lambda n: 1 << (n - 1).bit_length(),
                                           cdiv=lambda a, b: (a + b - 1) // b),
                 '_block_attention': Kernel('block'), '_append': Kernel('append')}
    tree = ast.parse(path.read_bytes())
    nodes = [copy.deepcopy(n) for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('block_attention', 'append')]
    module = ast.fix_missing_locations(ast.Module(body=nodes, type_ignores=[]))
    exec(compile(module, str(path), 'exec', flags=__import__('__future__').annotations.compiler_flag), namespace)
    return namespace, calls


def valid_block():
    return [Tensor((4, 4, 64)), Tensor((2, 4, 64)), Tensor((2, 4, 64)),
            [Tensor((2, 3, 64)), Tensor((2, 5, 64))],
            [Tensor((2, 3, 64)), Tensor((2, 5, 64))], 2, 32, .125]


class Boundary(unittest.TestCase):
    def test_valid_metadata_reaches_one_identical_declared_launch(self):
        namespace, calls = wrappers(SOURCE)
        out = namespace['block_attention'](*valid_block())
        self.assertEqual(out.shape, (4, 256))
        self.assertEqual(calls[0], ('table', (16, 16, 16, 16, 3, 5), 'int64'))
        self.assertEqual(calls[-1][2], (2, 2))
        self.assertEqual(calls[-1][4]['G'], 2)
        self.assertEqual(calls[-1][4]['LP'], 16)
        namespace, calls = wrappers(SOURCE)
        outs = namespace['append'](Tensor((2, 4, 64)), [None, Tensor((2, 5, 64))], [1, 3], 4)
        self.assertEqual([v.shape for v in outs], [(2, 1, 64), (2, 4, 64)])
        self.assertEqual(calls[2], ('table', (16, 16, 16, 32, 0, 1, 0, 1, 5, 3, 1, 4), 'int64'))
        self.assertEqual(calls[-1][2], (2, 2, 1))

    def test_block_invalid_metadata_stops_before_allocation_and_launch(self):
        cases = []
        for index, value in [(0, None), (0, Tensor((4, 4))), (1, Tensor((2, 4, 64), 'fp16')),
                             (2, Tensor((2, 4, 64), device='cuda:1')), (0, Tensor((4, 4, 64), device='cpu')),
                             (0, Tensor((4, 4, 64), contiguous=False)), (5, 0), (5, True), (5, 1.0),
                             (0, Tensor((0, 4, 64))), (1, Tensor((0, 4, 64))), (3, [])]:
            args = valid_block()
            args[index] = value
            cases.append(args)
        for changed in (Tensor((2, 3, 64), 'fp16'), Tensor((2, 3, 64), device='cuda:1'),
                        Tensor((2, 3, 64), contiguous=False), Tensor((2, 3, 64), ptr=17),
                        Tensor((2, 3, 32)), Tensor((2, 2**31, 64)), None):
            args = valid_block()
            args[3][0] = changed
            cases.append(args)
        for args in cases:
            with self.subTest(args=args):
                namespace, calls = wrappers(SOURCE, allow_allocations=False)
                with self.assertRaises(ValueError):
                    namespace['block_attention'](*args)
                self.assertEqual(calls, [])
        self.assertEqual(len(cases), 19)

    def test_append_invalid_metadata_stops_before_allocation_and_launch(self):
        cases = []
        for new in (None, Tensor((2, 4)), Tensor((2, 4, 64), 'fp16'),
                    Tensor((2, 4, 64), device='cpu'), Tensor((2, 4, 64), contiguous=False),
                    Tensor((0, 4, 64)), Tensor((2, 4, 0))):
            cases.append((new, [None], [4], 32))
        for olds, sizes, window in [([], [4], 32), ([None], [3], 32), ([None], [-4], 32),
                                    ([None], [True], 32), ([None], [4.0], 32), ([None], [4], -1),
                                    ([None], [4], True), ([None], [4], 2.0)]:
            cases.append((Tensor((2, 4, 64)), olds, sizes, window))
        for old in (Tensor((2, 3, 64), 'fp16'), Tensor((2, 3, 64), device='cpu'),
                    Tensor((2, 3, 64), contiguous=False), Tensor((2, 3, 64), ptr=17),
                    Tensor((2, 3, 32)), Tensor((2, 3)), object()):
            cases.append((Tensor((2, 4, 64)), [old], [4], 32))
        for args in cases:
            with self.subTest(args=args):
                namespace, calls = wrappers(SOURCE, allow_allocations=False)
                with self.assertRaises(ValueError):
                    namespace['append'](*args)
                self.assertEqual(calls, [])
        self.assertEqual(len(cases), 22)

    def test_zero_add_window_and_empty_context_are_preserved(self):
        namespace, calls = wrappers(SOURCE)
        outs = namespace['append'](Tensor((2, 3, 64)), [None, Tensor((2, 0, 64))], [0, 3], 0)
        self.assertEqual([t.shape for t in outs], [(2, 0, 64), (2, 0, 64)])
        self.assertEqual(calls[-1][2], (2, 2, 0))


if __name__ == '__main__':
    unittest.main()

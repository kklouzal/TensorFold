"""Lane-matmul metadata guards, without importing a numerical runtime."""
from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / 'src/tensorfold/cuda/kernels/qmm.py'


class Tensor:
    def __init__(self, shape=(3, 128), *, cuda=True, dtype='bf16', inner_stride=1):
        self.shape, self.is_cuda, self.dtype = shape, cuda, dtype
        self.device = 'gpu'
        self.inner_stride = inner_stride

    def dim(self):
        return len(self.shape)

    def stride(self, index):
        return self.inner_stride if index == 1 else self.shape[1] + 16


def namespace():
    calls = []
    nodes = [node for node in ast.parse(SOURCE.read_text()).body if isinstance(node, ast.FunctionDef)
             and node.name in ('group_sums', 'matmul_group')]

    def allocation(shape, **kwargs):
        calls.append(('allocation', shape, kwargs))
        return 'owned-result'

    class Kernel:
        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                calls.append(('launch', grid, args, kwargs))
            return launch

    ns = {'torch': SimpleNamespace(Tensor=Tensor, bfloat16='bf16', float32='fp32', empty=allocation),
          'triton': SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b), '_group_sums': Kernel(),
          'Q4': object, 'grouped': lambda device: False}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), 'exec'), ns)
    return ns, calls


class Boundary(unittest.TestCase):
    def test_malformed_group_geometry_is_rejected_before_allocation_or_launch(self):
        cases = [(None, 64), (Tensor((128,)), 64), (Tensor((1, 2, 128)), 64),
                 (Tensor((0, 128)), 64), (Tensor((3, 0)), 64), (Tensor((3, 127)), 64),
                 (Tensor(cuda=False), 64), (Tensor(dtype='fp32'), 64),
                 (Tensor(inner_stride=2), 64), (Tensor(), 0), (Tensor(), -64),
                 (Tensor(), True), (Tensor(), 64.), (Tensor(), 16)]
        for x, gs in cases:
            with self.subTest(x=x, gs=gs):
                ns, calls = namespace()
                with self.assertRaises(ValueError):
                    ns['group_sums'](x, gs)
                self.assertEqual(calls, [])

    def test_valid_groups_retain_original_shape_stride_and_launch_options(self):
        for gs in (32, 64):
            ns, calls = namespace()
            x = Tensor()
            self.assertEqual(ns['group_sums'](x, gs), 'owned-result')
            self.assertEqual(calls[0], ('allocation', (3, 128 // gs), {'dtype': 'fp32', 'device': 'gpu'}))
            self.assertEqual(calls[1], ('launch', (3, 1), (x, 'owned-result', 144),
                                       {'KG': 128 // gs, 'GS': gs, 'GB': 16, 'num_warps': 2}))

    def test_group_split_count_cannot_silently_drop_or_ignore_weights(self):
        for weights, splits in ((2, [1]), (1, [1, 2]), (0, [1])):
            ns, calls = namespace()
            with self.assertRaises(ValueError):
                ns['matmul_group'](Tensor(), [SimpleNamespace(k=128)] * weights, sks=splits)
            self.assertEqual(calls, [])


if __name__ == '__main__':
    unittest.main()

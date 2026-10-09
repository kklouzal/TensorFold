"""Guard and valid launch metadata oracle; no numerical runtime imported."""

import ast
from pathlib import Path
import types
import unittest

REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "src/tensorfold/families/qwen3_5/cuda/qmm.py"


class Tensor:
    def __init__(self, shape, dtype="bf16", device="cuda:0", contiguous=True):
        self.shape = shape
        self.dtype = dtype
        self.device = device
        self.is_cuda = device.startswith("cuda")
        self._contiguous = contiguous

    def dim(self):
        return len(self.shape)

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return Tensor(self.shape, self.dtype, self.device, True)


class Controls(unittest.TestCase):
    def setUp(self):
        self.calls = []
        self.allocations = []
        owner = self

        class Kernel:
            def __init__(inner, name):
                inner.name = name

            def __getitem__(inner, grid):

                def execute(*args, **kwargs):
                    owner.calls.append((inner.name, grid, args, kwargs))

                return execute

        def empty(shape, dtype, device):
            value = Tensor(shape, dtype, device)
            self.allocations.append(value)
            return value

        torch = types.SimpleNamespace(
            is_tensor=lambda value: isinstance(value, Tensor),
            bfloat16="bf16",
            float16="f16",
            float32="f32",
            float64="f64",
            int32="i32",
            uint32="u32",
            empty=empty,
        )
        self.scope = {
            "torch": torch,
            "triton": types.SimpleNamespace(cdiv=lambda a, b: (a + b - 1) // b),
            "BN": 64,
            "_group_sums": Kernel("sums"),
            "_qmm": Kernel("qmm"),
            "_reduce": Kernel("reduce"),
        }
        tree = ast.parse(SOURCE.read_text())
        functions = [
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in ("bucket", "split_k", "group_sums", "lane_matmul")
        ]
        exec(
            compile(
                ast.Module(body=functions, type_ignores=[]),
                str(SOURCE),
                "exec",
                flags=__import__("__future__").annotations.compiler_flag,
            ),
            self.scope,
        )

    def data(self, m=3, n=80, k=512):
        return [Tensor((m, k)), Tensor((n, k // 8), "i32"), Tensor((n, k // 64)), Tensor((n, k // 64))]

    def reject(self, values, **kwargs):
        self.calls.clear()
        self.allocations.clear()
        with self.assertRaises(ValueError):
            self.scope["lane_matmul"](*values, **kwargs)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.allocations, [])

    def test_invalid_tensor_geometry_dtype_and_device_before_work(self):
        for index, changed in [
            (0, None),
            (0, Tensor((3, 512), "f32")),
            (0, Tensor((0, 512))),
            (0, Tensor((3, 513))),
            (1, Tensor((80,), "i32")),
            (1, Tensor((80, 64), "f32")),
            (1, Tensor((0, 64), "i32")),
            (1, Tensor((80, 64), "i32", "cuda:1")),
            (1, Tensor((80, 64), "i32", contiguous=False)),
            (2, Tensor((80, 7))),
            (2, Tensor((80, 8), "i32")),
            (3, Tensor((80, 8), device="cpu")),
            (3, Tensor((80, 8), contiguous=False)),
        ]:
            with self.subTest(index=index, shape=getattr(changed, "shape", None)):
                values = self.data()
                values[index] = changed
                self.reject(values)

    def test_invalid_supplied_sums_and_launch_configuration_before_work(self):
        for xs in [
            object(),
            Tensor((3, 7), "f32"),
            Tensor((3, 8)),
            Tensor((3, 8), "f32", "cuda:1"),
            Tensor((3, 8), "f32", contiguous=False),
        ]:
            with self.subTest(xs=xs):
                self.reject(self.data(), xs=xs)
        for kwargs in [{"sk": -1}, {"sk": 3}, {"sk": 16}, {"bm": 8}]:
            with self.subTest(kwargs=kwargs):
                self.reject(self.data(), **kwargs)

    def test_group_sums_rejects_malformed_before_allocation(self):
        for value in [
            None,
            Tensor((3,)),
            Tensor((3, 512), "f32"),
            Tensor((3, 512), device="cpu"),
            Tensor((3, 512), contiguous=False),
            Tensor((0, 512)),
            Tensor((3, 511)),
        ]:
            self.calls.clear()
            self.allocations.clear()
            with self.assertRaises(ValueError):
                self.scope["group_sums"](value)
            self.assertEqual(self.calls, [])
            self.assertEqual(self.allocations, [])

    def test_all_standard_floating_scale_bias_dtype_combinations_preserve_launch(self):
        for scale_dtype in ["bf16", "f16", "f32", "f64"]:
            for bias_dtype in ["bf16", "f16", "f32", "f64"]:
                for sk in [1, 2]:
                    data = self.data()
                    data[2].dtype = scale_dtype
                    data[3].dtype = bias_dtype
                    self.calls.clear()
                    self.allocations.clear()
                    self.scope["lane_matmul"](*data, sk=sk)
                    qmm = next((value for value in self.calls if value[0] == "qmm"))
                    self.assertIs(qmm[2][3], data[2])
                    self.assertIs(qmm[2][4], data[3])
                    self.assertEqual(qmm[3]["SK"], sk)

    def test_valid_projection_dispatch_and_storage_contract(self):
        data = self.data(m=3, n=80, k=512)
        result = self.scope["lane_matmul"](*data, sk=2)
        self.assertEqual((result.shape, result.dtype, result.device), ((3, 80), "bf16", "cuda:0"))
        self.assertEqual(
            [(value.shape, value.dtype) for value in self.allocations],
            [((3, 8), "f32"), ((3, 80), "bf16"), ((2, 3, 80), "f32")],
        )
        self.assertEqual(
            [(name, grid) for name, grid, args, kwargs in self.calls],
            [("sums", (3, 1)), ("qmm", (1, 2, 2)), ("reduce", (1,))],
        )
        projection = self.calls[1]
        self.assertEqual(
            projection[3], {"N": 80, "K": 512, "SK": 2, "BM": 16, "BLOCK_N": 64, "num_warps": 4, "num_stages": 3}
        )


if __name__ == "__main__":
    unittest.main()

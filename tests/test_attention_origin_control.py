"""Model-owned attention address contracts; stdlib only, no numerical runtime."""
from __future__ import annotations

import __future__
import ast
from dataclasses import dataclass
from functools import cached_property
import gc
from pathlib import Path
from types import SimpleNamespace
import unittest
import weakref

ROOT = Path(__file__).resolve().parents[1]
ATTENTION = ROOT / "src/tensorfold/cuda/kernels/attention.py"


class Tensor:
    def __init__(self, address, *, dtype="bf16", device="cuda:0", contiguous=True, count=32, owner=None):
        self.address, self.dtype, self.device = address, dtype, device
        self.contiguous, self.count, self.owner = contiguous, count, owner
        self.is_cuda = device.startswith("cuda:")

    def is_contiguous(self):
        return self.contiguous

    def numel(self):
        return self.count

    def data_ptr(self):
        return self.address

    def view(self, dtype):
        return Tensor(self.address, dtype=dtype, device=self.device, count=2 * self.count, owner=self)


def functions():
    tree = ast.parse(ATTENTION.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {"offsets", "address_origin"}]
    namespace = {"torch": SimpleNamespace(float32="f32", bfloat16="bf16")}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(ATTENTION), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return SimpleNamespace(**namespace)


class OriginControls(unittest.TestCase):
    def test_address_only_view_retains_exact_frequency_owner_without_other_runtime_calls(self):
        api = functions()
        frequency = Tensor(4096, dtype="f32", count=16)
        reference = weakref.ref(frequency)
        view = api.address_origin(frequency)
        self.assertIs(view.owner, frequency)
        self.assertEqual((view.dtype, view.address, view.count), ("bf16", 4096, 32))
        del frequency
        gc.collect()
        self.assertIsNotNone(reference())
        del view
        gc.collect()
        self.assertIsNone(reference())

    def test_cache_effective_addresses_match_for_positive_negative_empty_and_native_limits(self):
        api = functions()
        for base in (16, 4096, (1 << 64) - 16):
            origin = Tensor(base)
            for address in (0, 16, 4080, 8192, (1 << 64) - 16):
                cache = Tensor(address, count=0 if address == 0 else 32)
                result = api.offsets([(cache, cache)], origin)
                self.assertEqual([base + 2 * value for value in result], [address, address])
                self.assertTrue(all(-(1 << 63) <= value < (1 << 63) for value in result))

    def test_origin_and_cache_boundary_refusals_precede_tensor_allocation(self):
        api = functions()
        for kw in ({"dtype": "f32"}, {"device": "cpu"}, {"contiguous": False}, {"count": 0}):
            with self.subTest(origin=kw), self.assertRaises(ValueError):
                api.offsets([(Tensor(8192), Tensor(8192))], Tensor(4096, **kw))
        for address in (4097,):
            with self.assertRaises(ValueError):
                api.offsets([(Tensor(8192), Tensor(8192))], Tensor(address))
        for cache in (Tensor(8192, dtype="f32"), Tensor(8192, device="cuda:1"),
                      Tensor(8192, contiguous=False), Tensor(8193), Tensor(1 << 65)):
            with self.subTest(cache=cache.__dict__), self.assertRaises(ValueError):
                api.offsets([(cache, cache)], Tensor(4096))
        for frequency in (Tensor(4096, dtype="bf16"), Tensor(4096, dtype="f32", device="cpu"),
                          Tensor(4096, dtype="f32", count=0), Tensor(4097, dtype="f32")):
            with self.assertRaises(ValueError):
                api.address_origin(frequency)

    def test_model_property_is_bounded_owned_and_all_loaders_resolve_at_startup(self):
        source = ROOT / "src/tensorfold/families/qwen3_5/cuda/weights.py"
        tree = ast.parse(source.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Weights")
        prop = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "attention_origin")
        self.assertEqual([ast.unparse(d) for d in prop.decorator_list], ["cached_property"])
        self.assertEqual(ast.unparse(prop.body[-1]), "return address_origin(self.inv_freq)")
        # The fixed model lifetime owns the cached view; no process cache exists.
        self.assertNotIn("lru_cache", ATTENTION.read_text())
        self.assertNotIn("def base(", ATTENTION.read_text())
        for name in ("weights.py", "exl3_load.py", "nvfp4_load.py"):
            path = source.parent / name
            text = path.read_text()
            first = text.index("w.inv_freq = inv.to(torch.float32).to(device)")
            second = text.index("_ = w.attention_origin", first)
            self.assertLess(first, second)

    def test_all_linear_models_do_not_resolve_or_allocate_an_attention_origin(self):
        class NoAttention:
            layers = [SimpleNamespace(linear=True)]
            @property
            def attention_origin(self):
                raise AssertionError('all-linear model accessed an unused address origin')
        family = ROOT / "src/tensorfold/families/qwen3_5/cuda"
        for name in ("weights.py", "exl3_load.py", "nvfp4_load.py"):
            tree = ast.parse((family / name).read_text())
            guard = next(n for n in ast.walk(tree) if isinstance(n, ast.If)
                         and ast.unparse(n.test) == "any((not layer.linear for layer in w.layers))")
            exec(compile(ast.Module(body=[guard], type_ignores=[]), str(family / name), "exec"),
                 {"w": NoAttention()})
        tree = ast.parse((family / "forward.py").read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_cache_offsets")
        namespace = {}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(family / "forward.py"), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        self.assertEqual(namespace["_cache_offsets"]([], [], None), {})

    def test_staged_forward_retains_origin_and_cache_storage(self):
        path = ROOT / "src/tensorfold/families/qwen3_5/cuda/forward.py"
        tree = ast.parse(path.read_text())
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Staged")
        namespace = {"dataclass": dataclass, "cached_property": cached_property}
        exec(compile(ast.Module(body=[cls], type_ignores=[]), str(path), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        origin, k, v = Tensor(4096), Tensor(8192), Tensor(16384)
        references = [weakref.ref(t) for t in (origin, k, v)]
        staged = namespace["Staged"](1, None, None, None, None, {}, None, None, None, origin, ((k, v),))
        del origin, k, v
        gc.collect()
        self.assertTrue(all(reference() is not None for reference in references))
        del staged
        gc.collect()
        self.assertTrue(all(reference() is None for reference in references))
        mtp = (ROOT / "src/tensorfold/families/qwen3_5_moe/cuda/mtp.py").read_text()
        self.assertIn("self.origin, self.caches = cache.origin, (cache.k, cache.v)", mtp)
        self.assertIn("other.origin = self.origin", mtp)

    def test_all_project_attention_calls_supply_the_live_origin(self):
        paths = ("src/tensorfold/families/qwen3_5/cuda/forward.py",
                 "src/tensorfold/families/qwen3_5_moe/cuda/mtp.py", "tests/cuda/test_attention.py")
        count = 0
        for name in paths:
            tree = ast.parse((ROOT / name).read_text())
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and ast.unparse(node.func) in {"tree_attention.attention", "shared.attention"}:
                    self.assertIn("origin", [keyword.arg for keyword in node.keywords])
                    count += 1
        self.assertEqual(count, 7)


if __name__ == "__main__":
    unittest.main()

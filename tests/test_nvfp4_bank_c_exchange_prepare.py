"""Stdlib-only source preparation/dispatch boundaries; never imports Torch."""

import ast
import hashlib
import sys
import tempfile
import unittest
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

SOURCE = Path(__file__).parents[1] / "src/tensorfold/cuda/nvfp4/bank_cpu.py"


def owned_function(name, **bindings):
    node = next(item for item in ast.parse(SOURCE.read_bytes()).body
                if isinstance(item, ast.FunctionDef) and item.name == name)
    scope = {"lru_cache": lru_cache, **bindings}
    exec(compile(ast.Module([node], []), str(SOURCE), "exec"), scope)  # noqa: S102 - one authenticated owned function, no original module body.
    return scope[name]


class Preparation(unittest.TestCase):
    def test_absent_provider_and_advertised_invalid_value_are_distinct(self):
        class Tensor:
            pass

        increment = object()
        torch = SimpleNamespace(Tensor=Tensor, autograd=SimpleNamespace(graph=SimpleNamespace(increment_version=increment)))
        producer = owned_function("_producer", torch=torch)
        self.assertIsNone(producer())
        producer.cache_clear()
        Tensor.__dlpack_c_exchange_api__ = None
        self.assertEqual(producer(), (Tensor, None, increment))

    def test_static_producer_is_bound_once_and_keeps_its_callback(self):
        class Tensor:
            __dlpack_c_exchange_api__ = object()

        first, second = object(), object()
        graph = SimpleNamespace(increment_version=first)
        producer = owned_function("_producer", torch=SimpleNamespace(Tensor=Tensor, autograd=SimpleNamespace(graph=graph)))
        selected = producer()
        graph.increment_version = second
        self.assertIs(producer(), selected)
        self.assertIs(selected[2], first)

    def test_native_provider_interface_returns_only_the_already_retained_context_module(self):
        owner = ModuleType("actual_extension")
        resolve = owned_function("_native_module")
        for context in ((object(), object(), object(), owner), SimpleNamespace(module_owner=owner)):
            self.assertIs(resolve(SimpleNamespace(__self__=context)), owner)

    def test_plain_direct_inputs_never_create_storage_views_or_numpy_aliases(self):
        class Tensor:
            pass

        inputs = tuple(Tensor() for _ in range(5))
        calls = []

        def prepare():
            return lambda *args: calls.append(args)

        def forbidden(*_args):
            raise AssertionError("direct input must not evaluate NumPy/storage-view work")

        pack = owned_function("pack", _producer=lambda: (Tensor, object(), object()), _ext=prepare,
                              _numpy_ext=forbidden, _array=forbidden)
        pack(*inputs, 64, 128)
        self.assertEqual(calls, [(*inputs, 64, 128)])

    def test_subclass_selects_required_route_before_direct_compilation(self):
        class Tensor:
            pass

        class Subclass(Tensor):
            pass

        primary = OSError("required provider selected")

        def legacy():
            raise primary

        def forbidden():
            raise AssertionError("subclass must not compile direct module")

        pack = owned_function("pack", _producer=lambda: (Tensor, object(), object()),
                              _ext=forbidden, _numpy_ext=legacy)
        with self.assertRaises(OSError) as result:
            pack(Subclass(), *(Tensor() for _ in range(4)), 64, 64)
        self.assertIs(result.exception, primary)

    def test_int_like_geometry_selects_required_route_without_conversion_or_direct_compilation(self):
        class Tensor:
            pass

        class Integer(int):
            pass

        class Index:
            def __index__(self):
                raise AssertionError("route selection must not convert geometry")

        primary = OSError("required geometry-conversion provider selected")

        def legacy():
            raise primary

        def forbidden():
            raise AssertionError("int-like geometry must not compile direct module")

        pack = owned_function("pack", _producer=lambda: (Tensor, object(), object()),
                              _ext=forbidden, _numpy_ext=legacy)
        inputs = tuple(Tensor() for _ in range(5))
        for geometry in (Integer(64), Index(), True):
            for width, dims in ((geometry, 64), (64, geometry)):
                with self.subTest(width=type(width), dims=type(dims)), self.assertRaises(OSError) as result:
                    pack(*inputs, width, dims)
                self.assertIs(result.exception, primary)

    def test_transitive_owned_header_changes_native_identity_after_every_owner_closes(self):
        module = ModuleType("torch.utils.cpp_extension")
        events, requests = [], []

        def load(**kwargs):
            events.append("build")
            requests.append(kwargs)
            return kwargs

        module.load = load
        content = {"bank_exchange.cpp": b"native source", "core.h": b"core1", "kernel.h": b"kernel"}

        @contextmanager
        def stream(path):
            events.append(("open", path.name))
            try:
                yield SimpleNamespace(read=lambda: content[path.name])
            finally:
                events.append(("close", path.name))

        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, {"torch.utils.cpp_extension": module}):
            prepare = owned_function("_load", __file__=str(Path(directory) / "bank_cpu.py"),
                                     Path=Path, hashlib=hashlib, _regular_stream=stream)
            first = prepare("bank_exchange.cpp", ("core.h", "kernel.h"))
            self.assertEqual(events[-2:], [("close", "kernel.h"), "build"])
            content["core.h"] = b"core2"
            second = prepare("bank_exchange.cpp", ("core.h", "kernel.h"))
        self.assertNotEqual(first["name"], second["name"])
        self.assertFalse(first["with_cuda"])
        self.assertEqual(first["extra_cflags"], ["-O3", "-std=c++17", "-Wall", "-Wextra", "-Werror"])


if __name__ == "__main__":
    unittest.main()

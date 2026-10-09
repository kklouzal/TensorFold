"""Actual constructor/close AST with labeled ctypes/SDK substitutes, no native."""
from __future__ import annotations

import ast
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold.cleanup import finish, rollback
from tensorfold.cuda.engine_lifetime import EngineLifetime

ROOT = Path(__file__).resolve().parents[1]


def actual_class(relative, name, namespace):
    tree = ast.parse((ROOT / relative).read_bytes())
    node = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == name)
    exec(compile(ast.fix_missing_locations(ast.Module([ast.ImportFrom("__future__", [ast.alias("annotations")], 0), node], [])), relative, "exec"), namespace)
    return namespace[name]


def communicator(*, init_error=None, release_error=None, fence_error=None):
    calls = []
    class Pointer:
        def __init__(self):
            self.value = None
    class Function:
        def __init__(self, name):
            self.name = name
        def __call__(self, *args):
            calls.append(self.name)
            if self.name == "ncclCommInitRank":
                args[0].value = 1234
                if init_error is not None:
                    raise init_error
            if self.name in ("ncclCommDestroy", "ncclCommAbort") and release_error is not None:
                raise release_error
            return b"failure" if self.name == "ncclGetErrorString" else 0
    lib = SimpleNamespace(**{name: Function(name) for name in
                          ("ncclGetErrorString", "ncclGetUniqueId", "ncclCommInitRank", "ncclAllGather",
                           "ncclCommDestroy", "ncclCommAbort")})
    class Store:
        def __init__(self, *args, **kwargs):
            calls.append("store")
        def set(self, *args):
            calls.append("set")
        def get(self, *args):
            return b"x" * 128
    def fence(device):
        calls.append("fence")
        if fence_error is not None:
            raise fence_error
    class Tensor:
        dtype, device = "f32", "cuda:0"
        def numel(self):
            return 2
        def is_contiguous(self):
            return True
        def data_ptr(self):
            return 100
        def element_size(self):
            return 4
    runtime = SimpleNamespace(device=lambda *args: "cuda:0", Tensor=Tensor,
                              cuda=SimpleNamespace(current_device=lambda: 0, synchronize=fence,
                                                   current_stream=lambda device: SimpleNamespace(cuda_stream=123)))
    ctypes = SimpleNamespace(c_void_p=Pointer, c_int=int, c_char_p=bytes, c_size_t=int,
                             POINTER=lambda value: value, byref=lambda value: value, sizeof=lambda value: 128,
                             memmove=lambda *args: None, addressof=lambda value: 0)
    namespace = {"threading": threading, "ctypes": ctypes, "torch": runtime,
                 "rollback": rollback, "_library": lambda: lib,
                 "_UniqueId": lambda: SimpleNamespace(internal=b"x" * 128), "_DTYPES": {"f32": 7}}
    cls = actual_class("src/tensorfold/cuda/comm.py", "NCCL", namespace)
    distributed = ModuleType("torch.distributed")
    distributed.TCPStore = Store
    return cls, calls, distributed


class ConstructorLifecycleTests(unittest.TestCase):
    def test_empty_slot_publication_and_reopen_refusal_preserve_live_owner(self):
        cls, calls, distributed = communicator()
        owner = cls()
        self.assertEqual(calls, [])
        with patch.dict(sys.modules, {"torch.distributed": distributed}):
            owner.open(0, 1, "127.0.0.1", 0)
            with self.assertRaises(RuntimeError):
                owner.open(0, 1, "127.0.0.1", 0)
        self.assertFalse(owner.closed)
        self.assertEqual(owner.comm.value, 1234)
        owner.close()

    def test_close_waits_accepted_collective_post_before_fencing_and_destroy(self):
        cls, calls, distributed = communicator()
        owner = cls()
        with patch.dict(sys.modules, {"torch.distributed": distributed}):
            owner.open(0, 1, "127.0.0.1", 0)
        entered, release, posted, retired = (threading.Event() for _ in range(4))
        failures = []
        def collective(*args):
            entered.set()
            if not release.wait(5):
                raise AssertionError("fixture collective deadline")
            posted.set()
            return 0
        owner.lib.ncclAllGather = collective
        runtime = owner.all_gather.__globals__["torch"]
        def gather():
            try:
                owner.all_gather(runtime.Tensor(), runtime.Tensor())
            except BaseException as error:
                failures.append(error)
        def close():
            try:
                owner.close()
            except BaseException as error:
                failures.append(error)
            finally:
                retired.set()
        worker, closer = threading.Thread(target=gather), threading.Thread(target=close)
        worker.start()
        try:
            self.assertTrue(entered.wait(5))
            closer.start()
            self.assertNotIn("fence", calls)
            release.set()
            self.assertTrue(retired.wait(5))
            self.assertTrue(posted.is_set())
        finally:
            release.set()
            worker.join(5)
            if closer.ident is not None:
                closer.join(5)
        self.assertEqual(failures, [])
        self.assertLess(calls.index("fence"), calls.index("ncclCommDestroy"))

    def test_successful_native_close_fences_before_destroy_and_is_one_shot(self):
        cls, calls, distributed = communicator()
        with patch.dict(sys.modules, {"torch.distributed": distributed}):
            owner = cls()
            owner.open(0, 1, "127.0.0.1", 0)
        owner.close()
        self.assertTrue(owner.closed and owner.retired)
        self.assertLess(calls.index("fence"), calls.index("ncclCommDestroy"))
        owner.close()
        self.assertEqual(calls.count("ncclCommDestroy"), 1)
        self.assertIsNone(owner.store)
        for operation in (lambda: owner.all_gather(None, None), lambda: owner.ready("loading"), owner.barrier):
            with self.assertRaises(RuntimeError):
                operation()

    def test_partial_native_init_aborts_before_fence_preserving_exact_primary(self):
        primary = KeyboardInterrupt()
        cls, calls, distributed = communicator(init_error=primary)
        with patch.dict(sys.modules, {"torch.distributed": distributed}):
            with self.assertRaises(KeyboardInterrupt) as caught:
                owner = cls()
                owner.open(0, 1, "127.0.0.1", 0)
        self.assertIs(caught.exception, primary)
        self.assertLess(calls.index("ncclCommAbort"), calls.index("fence"))

    def test_ambiguous_native_release_never_retries_or_reclaims_owner(self):
        primary, close = KeyboardInterrupt(), OSError()
        cls, calls, distributed = communicator(init_error=primary, release_error=close)
        with patch.dict(sys.modules, {"torch.distributed": distributed}):
            with self.assertRaises(KeyboardInterrupt) as caught:
                owner = cls()
                owner.open(0, 1, "127.0.0.1", 0)
        self.assertIs(caught.exception, primary)
        owner = primary.__dict__["_tensorfold_retained_owners"][0]
        self.assertEqual(owner.comm.value, 1234)
        self.assertFalse(owner.retired)
        with self.assertRaises(RuntimeError):
            owner.close(abort=True)
        self.assertEqual(calls.count("ncclCommAbort"), 1)
        self.assertIs(owner._release_error, close)

    def test_bad_external_ranges_do_not_acquire_native_or_store(self):
        for args in ((True, 1, "host", 0), (0, 0, "host", 0), (1, 1, "host", 0),
                     (0, 1, "host", -1), (0, 1, "host", 65536), (0, 1, None, 0)):
            cls, calls, distributed = communicator()
            with patch.dict(sys.modules, {"torch.distributed": distributed}):
                with self.assertRaises(ValueError):
                    owner = cls()
                    owner.open(*args)
            self.assertEqual(calls, [])

    def test_device_fence_retry_does_not_repeat_successful_native_abort(self):
        fence = OSError()
        cls, calls, distributed = communicator(fence_error=fence)
        with patch.dict(sys.modules, {"torch.distributed": distributed}):
            owner = cls()
            owner.open(0, 1, "127.0.0.1", 0)
        with self.assertRaises(OSError):
            owner.close(abort=True)
        self.assertTrue(owner._native_retired)
        self.assertFalse(owner.retired)
        with self.assertRaises(OSError):
            owner.close(abort=True)
        self.assertEqual(calls.count("ncclCommAbort"), 1)

    def test_family_actual_constructor_rollback_retains_owned_resources_on_failed_fence(self):
        for path, name, kwargs in (
            ("src/tensorfold/families/nemotron_h/cuda/app.py", "NemotronEngine", {}),
            ("src/tensorfold/families/glm5_next/cuda/engine.py", "GlmEngine", {"rank": 0, "master": "host", "port": 1}),
        ):
            primary, fence, resource = KeyboardInterrupt(), OSError(), object()
            namespace = {"DRAFTS": 4, "CONFIDENCE": .5, "DEFAULT_POLICY": "auto", "EngineLifetime": EngineLifetime,
                         "rollback": rollback, "finish": finish}
            cls = actual_class(path, name, namespace)
            calls = []
            def initialize(owner, *args, **kwargs):
                owner.w = resource
                owner.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: (_ for _ in ()).throw(fence)))
                owner.comm = SimpleNamespace(close=lambda **kwargs: calls.append(kwargs["abort"]))
                raise primary
            cls._initialize = initialize
            with self.assertRaises(KeyboardInterrupt) as caught:
                cls(Path("model"), **kwargs)
            self.assertIs(caught.exception, primary)
            owner = primary.__dict__["_tensorfold_retained_owners"][0]
            self.assertIs(owner.w, resource)
            self.assertEqual(calls, [True])
            self.assertFalse(owner._lifetime.retired)
            owner.torch.cuda.synchronize = lambda device: None
            owner.close(abort=True)
            self.assertIsNone(owner.w)

    def test_glm_borrowed_communicator_not_aborted_during_constructor_failure(self):
        primary, calls = KeyboardInterrupt(), []
        namespace = {"DEFAULT_POLICY": "auto", "EngineLifetime": EngineLifetime, "rollback": rollback, "finish": finish}
        cls = actual_class("src/tensorfold/families/glm5_next/cuda/engine.py", "GlmEngine", namespace)
        borrowed = SimpleNamespace(close=lambda **kwargs: self.fail("borrowed communicator retired"))
        def initialize(owner, *args, **kwargs):
            owner.comm = kwargs["comm"]
            owner.torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda device: calls.append("fence")))
            raise primary
        cls._initialize = initialize
        with self.assertRaises(KeyboardInterrupt) as caught:
            cls(Path("model"), rank=0, master="host", port=1, comm=borrowed)
        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, ["fence"])
        self.assertNotIn("_tensorfold_retained_owners", primary.__dict__)

    def test_low_engine_actual_constructor_failure_fences_partial_scratch(self):
        calls, primary = [], KeyboardInterrupt()
        namespace = {"ROWS": 16, "PREFILL_ROWS": 2048, "rollback": rollback,
                     "torch": SimpleNamespace(no_grad=lambda: lambda function: function,
                                              cuda=SimpleNamespace(synchronize=lambda device: calls.append(device)))}
        cls = actual_class("src/tensorfold/families/nemotron_h/cuda/engine.py", "Engine", namespace)
        def initialize(owner, *args, **kwargs):
            owner.device, owner.scratch = "cuda:0", object()
            raise primary
        cls._initialize = initialize
        with self.assertRaises(KeyboardInterrupt) as caught:
            cls(object())
        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, ["cuda:0"])

    def test_canonical_callers_publish_empty_communicator_before_open(self):
        for relative in ("src/tensorfold/families/nemotron_h/cuda/app.py",
                         "src/tensorfold/families/glm5_next/cuda/engine.py",
                         "src/tensorfold/families/qwen4_exp/cuda/engine.py"):
            tree = ast.parse((ROOT / relative).read_bytes())
            calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "NCCL"]
            self.assertEqual(len(calls), 1)
            self.assertEqual((calls[0].args, calls[0].keywords), ([], []))
            opens = [node for node in ast.walk(tree) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                     and node.func.attr == "open" and ast.unparse(node.func.value) == "self.comm"]
            self.assertEqual(len(opens), 1)
            self.assertGreater(opens[0].lineno, calls[0].lineno)


if __name__ == "__main__":
    unittest.main()

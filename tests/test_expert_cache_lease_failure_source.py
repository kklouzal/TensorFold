"""Actual lease exception transport with real exceptions, callbacks and threads.

No SDK/native import or GPU/event/kernel behavior is claimed. Metadata/device
fixtures exercise owned source control flow and native exception identities.
"""

from __future__ import annotations

import ast
import builtins
from contextlib import contextmanager, nullcontext
from pathlib import Path
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load():
    ordinary = builtins.__import__

    def guard(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "numpy", "triton", "mlx", "cuda", "cupy", "ctypes"}:
            raise AssertionError("numerical/native import forbidden")
        return ordinary(name, *args, **kwargs)

    with patch.object(builtins, "__import__", guard):
        cleanup = {}
        exec(compile((ROOT / "src/tensorfold/cleanup.py").read_text(), "actual_cleanup", "exec"), cleanup)
        ns = {"contextmanager": contextmanager, "raise_failures": cleanup["raise_failures"]}
        source = ast.parse((ROOT / "src/tensorfold/cuda/expert_cache.py").read_text())
        host = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "HostExpertCache")
        functions = [n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "_integer"]
        host.body = [n for n in host.body if isinstance(n, ast.FunctionDef) and n.name in {"lease", "_usable"}]
        source = ast.parse((ROOT / "src/tensorfold/cuda/exl3/host_experts.py").read_text())
        exl = next(n for n in source.body if isinstance(n, ast.ClassDef) and n.name == "Exl3HostExpertCache")
        exl.body = [n for n in exl.body if isinstance(n, ast.FunctionDef) and n.name == "lease"]
        module = ast.Module(
            body=[
                ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
                *functions,
                host,
                exl,
            ],
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), "actual_expert_cache_lease_source", "exec"), ns)
        return ns


class Controls(unittest.TestCase):
    def fixture(self, variable, record, *, copy_error=None):
        ns = load()
        stream = SimpleNamespace(wait_event=lambda event: None)
        ns["torch"] = SimpleNamespace(
            cuda=SimpleNamespace(
                device=lambda device: nullcontext(),
                current_stream=lambda device: stream,
                is_current_stream_capturing=lambda: False,
            ),
            is_inference=lambda tensor: False,
        )
        cls = ns["Exl3HostExpertCache"] if variable else ns["HostExpertCache"]
        cache = object.__new__(cls)
        cache._lock = threading.RLock()
        cache._closed = cache._active = False
        cache._failure = cache._last_stream = None
        cache.device = "metadata only"
        cache.capacity = 4
        cache._last_use = SimpleNamespace(record=record)
        cache._size_layout = SimpleNamespace(kind="variable")
        cache._shared_cells = {}
        cache._shared_protected = frozenset()
        cache._layers = {
            7: SimpleNamespace(count=4, source=(SimpleNamespace(_version=0),), versions=(0,), views=("original",))
        }
        resident = {(7, 1): 0}
        keys = [(7, 1), None, None, None]
        policy = SimpleNamespace(
            resident=resident, keys=keys, recency=[0] * 4, tick=0, _logical=None, _batch_scan_limit=None
        )

        def touch(keys):
            policy.tick += 1

        def remove(slot):
            previous = keys[slot]
            if previous is not None:
                del resident[previous]
            keys[slot] = None

        def install(slot, key):
            keys[slot] = key
            resident[key] = slot

        policy.touch = touch
        policy.remove = remove
        policy.install = install
        policy.victim = lambda protected: next(slot for slot in range(4) if slot not in protected)
        cache._policy = policy
        cache.hits = cache.misses = cache.evictions = cache.copied_bytes = 0

        def copy(*args):
            if copy_error is not None:
                raise copy_error
            return 16

        cache._copy = copy
        return cache, stream

    @staticmethod
    def leaves(error):
        if isinstance(error, BaseExceptionGroup):
            result = []
            for item in error.exceptions:
                result.extend(Controls.leaves(item))
            return result
        return [error]

    def test_success_payload_mapping_counters_and_retirement_unchanged(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                records = []
                cache, stream = self.fixture(variable, records.append)
                with cache.lease(7, [1]) as result:
                    self.assertEqual(result, (("original",), {1: 0}))
                    self.assertTrue(cache._active)
                self.assertEqual(records, [stream])
                self.assertEqual((cache.hits, cache.misses, cache.copied_bytes), (1, 0, 0))
                self.assertFalse(cache._active)
                self.assertIs(cache._last_stream, stream)
                self.assertIsNone(cache._failure)

    def test_consumer_and_record_failure_keep_all_prior_native_roots(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                primary = OSError("consumer")
                cause, context, secondary = (
                    LookupError("prior cause"),
                    ArithmeticError("prior context"),
                    RuntimeError("record"),
                )
                primary.__cause__ = cause
                primary.__context__ = context

                def record(stream):
                    primary.__cause__ = primary.__context__ = None
                    raise secondary

                cache, stream = self.fixture(variable, record)
                with self.assertRaises(OSError) as caught:
                    with cache.lease(7, [1]):
                        raise primary
                self.assertIs(caught.exception, primary)
                leaves = self.leaves(primary.__cause__)
                self.assertIn(cause, leaves)
                self.assertIn(context, leaves)
                self.assertIn(secondary, leaves)
                self.assertFalse(cache._active)
                self.assertIs(cache._last_stream, stream)
                with self.assertRaises(RuntimeError):
                    cache._usable()

    def test_successful_record_callback_cannot_clear_consumer_cause_context_or_suppression(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                primary = OSError("consumer")
                cause, context = LookupError("cause"), ArithmeticError("context")
                primary.__cause__ = cause
                primary.__context__ = context
                primary.__suppress_context__ = False

                def record(stream):
                    primary.__cause__ = primary.__context__ = None
                    primary.__suppress_context__ = True

                cache, _ = self.fixture(variable, record)
                with self.assertRaises(OSError) as caught:
                    with cache.lease(7, [1]):
                        raise primary
                self.assertIs(caught.exception, primary)
                self.assertIs(primary.__cause__, cause)
                self.assertIs(primary.__context__, context)
                self.assertFalse(primary.__suppress_context__)
                self.assertFalse(cache._active)
                self.assertIsNone(cache._failure)

    def test_record_only_failure_retains_identity_and_last_stream_owner(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                primary, cause = RuntimeError("record"), LookupError("cause")
                primary.__cause__ = cause
                cache, stream = self.fixture(variable, lambda stream: (_ for _ in ()).throw(primary))
                with self.assertRaises(RuntimeError) as caught:
                    with cache.lease(7, [1]):
                        pass
                self.assertIs(caught.exception, primary)
                self.assertIs(primary.__cause__, cause)
                self.assertFalse(cache._active)
                self.assertIs(cache._last_stream, stream)
                self.assertIsNotNone(cache._failure)

    def test_fill_failure_and_event_failure_preserve_ownership_without_yielding(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                primary, cause, secondary = OSError("fill"), LookupError("fill cause"), RuntimeError("record")
                primary.__cause__ = cause

                def record(stream):
                    primary.__cause__ = None
                    raise secondary

                cache, stream = self.fixture(variable, record, copy_error=primary)
                with self.assertRaises(OSError) as caught:
                    with cache.lease(7, [2]):
                        self.fail("failed fill yielded payload")
                self.assertIs(caught.exception, primary)
                self.assertEqual(set(map(id, self.leaves(primary.__cause__))), {id(cause), id(secondary)})
                self.assertFalse(cache._active)
                self.assertIs(cache._last_stream, stream)
                self.assertIsNotNone(cache._failure)

    def test_baseexception_consumer_and_malformed_notes_do_not_replace_primary(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                primary = GeneratorExit("cancelled consumer")
                cause, secondary = KeyboardInterrupt("native prior"), OSError("record")
                primary.__cause__ = cause
                primary.__notes__ = None
                cache, _ = self.fixture(variable, lambda stream: (_ for _ in ()).throw(secondary))
                with self.assertRaises(GeneratorExit) as caught:
                    with cache.lease(7, [1]):
                        raise primary
                self.assertIs(caught.exception, primary)
                leaves = self.leaves(primary.__cause__)
                self.assertIn(cause, leaves)
                self.assertIn(secondary, leaves)
                self.assertFalse(cache._active)

    def test_real_thread_keeps_lease_lock_until_event_callback_finishes(self):
        for variable in (False, True):
            with self.subTest(variable=variable):
                began, release = threading.Event(), threading.Event()
                primary, cause, secondary = OSError("consumer"), LookupError("cause"), RuntimeError("record")
                primary.__cause__ = cause

                def record(stream):
                    began.set()
                    if not release.wait(2):
                        raise TimeoutError("fixture callback timed out")
                    primary.__cause__ = None
                    raise secondary

                cache, _ = self.fixture(variable, record)
                failures = []

                def consume():
                    try:
                        with cache.lease(7, [1]):
                            raise primary
                    except BaseException as error:
                        failures.append(error)

                worker = threading.Thread(target=consume)
                worker.start()
                try:
                    self.assertTrue(began.wait(2))
                    self.assertTrue(cache._active)
                    self.assertFalse(cache._lock.acquire(timeout=0.03))
                finally:
                    release.set()
                    worker.join(2)
                self.assertFalse(worker.is_alive())
                self.assertEqual(failures, [primary])
                self.assertIn(cause, self.leaves(primary.__cause__))
                self.assertIn(secondary, self.leaves(primary.__cause__))
                self.assertFalse(cache._active)


if __name__ == "__main__":
    unittest.main()

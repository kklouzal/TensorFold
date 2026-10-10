"""PLE loader/engine journals with real Futures/threads and stdlib source controls."""
from __future__ import annotations

import __future__
import ast
import copy
from concurrent.futures import Future, ThreadPoolExecutor
import importlib.util
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
FAMILY = ROOT / "src/tensorfold/families/qwen4_exp"
spec = importlib.util.spec_from_file_location("ple_table_owner_control", FAMILY / "ple_lifetime.py")
API = importlib.util.module_from_spec(spec)
cleanup_spec = importlib.util.spec_from_file_location("ple_cleanup_source", ROOT / "src/tensorfold/cleanup.py")
CLEANUP = importlib.util.module_from_spec(cleanup_spec)
cleanup_spec.loader.exec_module(CLEANUP)
with patch.dict(sys.modules, {"tensorfold.cleanup": CLEANUP}):
    spec.loader.exec_module(API)


def failures(error):
    """Independent identity oracle for native cause groups, without foreign hooks."""
    cause = BaseException.__cause__.__get__(error)
    if type(cause) in (BaseExceptionGroup, ExceptionGroup):
        return list(cause.exceptions)
    return [] if cause is None else [cause]


class Table:
    def __init__(self, events, name="table"):
        self.events, self.name = events, name
        self.closes = 0
        self.rows, self.width = 3, 32
        self.failure = None

    def close(self):
        self.events.append(self.name)
        self.closes += 1
        if self.failure is not None:
            raise self.failure


class PLETableControl(unittest.TestCase):
    def test_acquisition_precedes_failure_and_close_consumes_success_once(self):
        events = []
        owner = API.PLETables()
        first, second = Table(events, "first"), Table(events, "second")
        self.assertIs(owner.acquire(first), first)
        owner.acquire(second)
        owner.close(lambda: events.append("fence"))
        self.assertEqual(events, ["fence", "first", "second"])
        self.assertTrue(owner.closed)
        owner.close(lambda: self.fail("closed owner fenced again"))
        self.assertEqual(first.closes, 1)

    def test_real_pending_prefetch_drains_before_fence_and_table_close(self):
        events = []
        owner = API.PLETables()
        owner.acquire(Table(events))
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        errors = []
        def prefetch():
            entered.set()
            if not release.wait(3):
                raise TimeoutError("controlled prefetch release")
            events.append("read-done")
        def close():
            try:
                owner.close(lambda: events.append("fence"))
            except BaseException as error:
                errors.append(error)
            finally:
                closed.set()
        with ThreadPoolExecutor(1) as pool:
            owner.reads.append(pool.submit(prefetch))
            self.assertTrue(entered.wait(1))
            thread = threading.Thread(target=close)
            thread.start()
            self.assertFalse(closed.wait(.03))
            self.assertEqual(events, [])
            release.set()
            thread.join(2)
            self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(events, ["read-done", "fence", "table"])

    def test_incomplete_future_retains_tables_and_retry_observes_original_result(self):
        events = []
        owner = API.PLETables()
        table = owner.acquire(Table(events))
        error = KeyboardInterrupt("interrupted read wait")
        actual = Future()
        class Interrupted:
            first = True
            def cancel(self):
                return False
            def result(self):
                if self.first:
                    self.first = False
                    raise error
                return actual.result()
            def done(self):
                return actual.done()
            def cancelled(self):
                return False
            def exception(self, timeout):
                return actual.exception(timeout=timeout)
        future = Interrupted()
        owner.reads.append(future)
        with self.assertRaises(KeyboardInterrupt) as caught:
            owner.close(lambda: self.fail("incomplete read fenced"))
        self.assertIs(caught.exception, error)
        self.assertEqual(owner.tables, [table])
        self.assertEqual(owner.reads, [future])
        self.assertEqual(events, [])
        actual.set_result(0.1)
        owner.close(lambda: events.append("fence"))
        self.assertEqual(events, ["fence", "table"])

    def test_completed_prefetch_error_surfaces_after_cleanup(self):
        events = []
        owner = API.PLETables()
        owner.acquire(Table(events))
        primary = OSError("prefetch read failed")
        future = Future()
        future.set_exception(primary)
        owner.reads.append(future)
        with self.assertRaises(OSError) as caught:
            owner.close(lambda: events.append("fence"))
        self.assertIs(caught.exception, primary)
        self.assertTrue(owner.closed)
        self.assertEqual(events, ["fence", "table"])

    def test_fence_failure_retains_every_table_then_retry_closes(self):
        events = []
        owner = API.PLETables()
        table = owner.acquire(Table(events))
        failure = OSError("consumer fence failed")
        def fence():
            raise failure
        with self.assertRaises(OSError) as caught:
            owner.close(fence)
        self.assertIs(caught.exception, failure)
        self.assertEqual(owner.tables, [table])
        self.assertFalse(owner.closed or events)
        owner.close(lambda: events.append("fence"))
        self.assertEqual(events, ["fence", "table"])

    def test_failed_close_retains_only_failed_owner_and_all_other_closes_attempt(self):
        events = []
        owner = API.PLETables()
        first = owner.acquire(Table(events, "first"))
        second = owner.acquire(Table(events, "second"))
        first.failure = OSError("first close failed")
        with self.assertRaises(OSError):
            owner.close(lambda: events.append("fence"))
        self.assertEqual(events, ["fence", "first", "second"])
        self.assertEqual(owner.tables, [first])
        self.assertEqual(second.closes, 1)
        first.failure = None
        owner.close(lambda: events.append("fence"))
        self.assertTrue(owner.closed)
        self.assertEqual(first.closes, 2)

    def test_retained_loader_journal_bypasses_foreign_attribute_setters(self):
        calls = []
        class Opaque(RuntimeError):
            def __setattr__(self, name, value):
                calls.append(name)
                raise LookupError("foreign setter")
        primary = Opaque("failed load")
        owner, cleanup = API.PLETables(), OSError("failed close")
        API._retain_failure(primary, "ple_tables", owner)
        API._retain_failure(primary, "ple_cleanup_error", cleanup)
        dictionary = BaseException.__dict__["__dict__"].__get__(primary)
        self.assertIs(dictionary["ple_tables"], owner)
        self.assertIs(dictionary["ple_cleanup_error"], cleanup)
        self.assertFalse(calls)

    def test_actual_loader_rollback_attempts_every_close_before_opaque_diagnostics(self):
        tree = ast.parse((FAMILY / "cuda/weights.py").read_text())
        load = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load")
        handler = next(node.handlers[0] for node in load.body if isinstance(node, ast.Try)
                       and any(isinstance(part, ast.Name) and part.id == "ple_tables"
                               for part in ast.walk(node.handlers[0])))
        rollback = ast.FunctionDef(name="rollback", args=ast.arguments(posonlyargs=[], args=[],
                    kwonlyargs=[], kw_defaults=[], defaults=[]), body=[ast.Try(
                    body=[ast.Raise(exc=ast.Name(id="failure", ctx=ast.Load()))],
                    handlers=[copy.deepcopy(handler)], orelse=[], finalbody=[])], decorator_list=[])
        for notes in ("malformed", "opaque"):
            events, setters = [], []
            class Opaque(RuntimeError):
                def __setattr__(self, name, value):
                    setters.append(name)
                    raise LookupError("foreign setter")
                def __getattribute__(self, name):
                    if name == "__notes__" and notes == "opaque":
                        raise LookupError("opaque note provider")
                    return super().__getattribute__(name)
            primary = Opaque("load failed")
            dictionary = BaseException.__dict__["__dict__"].__get__(primary)
            if notes == "malformed":
                dict.__setitem__(dictionary, "__notes__", "malformed provider notes")
            owner = API.PLETables()
            table = owner.acquire(Table(events))
            table.failure = OSError("table close failed")
            reader = Table(events, "reader")
            reader.failure = OSError("reader close failed")
            expert = Table(events, "expert")
            expert.failure = OSError("expert close failed")
            namespace = {"failure": primary, "ple_tables": owner, "rd": reader,
                         "expert_cache": expert, "device": "controlled-device",
                         "torch": SimpleNamespace(cuda=SimpleNamespace(
                             synchronize=lambda _: events.append("fence"))),
                         "_retain_failure": API._retain_failure, "_note": API._note}
            exec(compile(ast.fix_missing_locations(ast.Module(body=[rollback], type_ignores=[])),
                         str(FAMILY / "cuda/weights.py"), "exec"), namespace)
            with self.subTest(notes=notes), self.assertRaises(Opaque) as caught:
                namespace["rollback"]()
            self.assertIs(caught.exception, primary)
            self.assertIsInstance(primary.__cause__, TypeError if notes == "malformed" else LookupError)
            self.assertEqual(events, ["fence", "table", "reader", "expert"])
            self.assertIs(dictionary["ple_tables"], owner)
            self.assertIs(dictionary["ple_cleanup_error"], table.failure)
            self.assertEqual([error for _, error in dictionary["load_cleanup_errors"]],
                             [table.failure, reader.failure, expert.failure])
            self.assertFalse(setters)
            table.failure = None
            owner.close(lambda: None)
            self.assertTrue(owner.closed)

    def test_malformed_note_cannot_replace_first_close_failure(self):
        events = []
        owner = API.PLETables()
        first = owner.acquire(Table(events, "first"))
        second = owner.acquire(Table(events, "second"))
        primary = first.failure = OSError("first close failed")
        primary.__notes__ = "malformed provider notes"
        second.failure = LookupError("second close failed")
        with self.assertRaises(OSError) as caught:
            owner.close(lambda: events.append("fence"))
        self.assertIs(caught.exception, primary)
        causes = failures(caught.exception)
        self.assertIn(second.failure, causes)
        self.assertTrue(any(type(error) is TypeError for error in causes))
        self.assertEqual(owner.tables, [first, second])
        self.assertEqual(events, ["fence", "first", "second"])
        first.failure = second.failure = None
        owner.close(lambda: None)
        self.assertTrue(owner.closed)

    def test_opaque_note_provider_cannot_replace_acquisition_primary(self):
        events = []
        owner = API.PLETables()
        class Opaque(MemoryError):
            def __getattribute__(self, name):
                if name == "__notes__":
                    raise LookupError("opaque provider note access")
                return super().__getattribute__(name)
        primary = Opaque("registry publication failed")
        class Broken(list):
            def append(self, _):
                raise primary
        owner.tables = Broken()
        table = Table(events)
        table.failure = OSError("rollback close failed")
        with self.assertRaises(Opaque) as caught:
            owner.acquire(table)
        self.assertIs(caught.exception, primary)
        causes = failures(caught.exception)
        self.assertIn(table.failure, causes)
        self.assertTrue(any(type(error) is LookupError for error in causes))
        self.assertIs(owner.unpublished, table)
        table.failure = None
        owner.close(lambda: None)
        self.assertTrue(owner.closed)

    def test_opaque_note_provider_cannot_replace_first_close_primary(self):
        events = []
        owner = API.PLETables()
        first = owner.acquire(Table(events, "first"))
        second = owner.acquire(Table(events, "second"))
        class Opaque(OSError):
            def __getattribute__(self, name):
                if name == "__notes__":
                    raise LookupError("opaque notes")
                return super().__getattribute__(name)
        primary = first.failure = Opaque("first close failed")
        second.failure = OSError("second close failed")
        with self.assertRaises(Opaque) as caught:
            owner.close(lambda: events.append("fence"))
        self.assertIs(caught.exception, primary)
        causes = failures(caught.exception)
        self.assertIn(second.failure, causes)
        self.assertTrue(any(type(error) is LookupError for error in causes))
        self.assertEqual(events, ["fence", "first", "second"])
        self.assertEqual(owner.tables, [first, second])
        first.failure = second.failure = None
        owner.close(lambda: None)

    def test_malformed_note_cannot_replace_acquisition_primary(self):
        events = []
        owner = API.PLETables()
        primary = MemoryError("registry publication failed")
        primary.__notes__ = 7
        class Broken(list):
            def append(self, _):
                raise primary
        owner.tables = Broken()
        table = Table(events)
        table.failure = OSError("rollback close failed")
        with self.assertRaises(MemoryError) as caught:
            owner.acquire(table)
        self.assertIs(caught.exception, primary)
        causes = failures(caught.exception)
        self.assertIn(table.failure, causes)
        self.assertTrue(any(type(error) is TypeError for error in causes))
        self.assertIs(owner.unpublished, table)
        table.failure = None
        owner.close(lambda: None)

    def test_lost_acquisition_publication_retains_failed_cleanup_authority(self):
        events = []
        owner = API.PLETables()
        original = MemoryError("table registry append failed")
        class Broken(list):
            def append(self, _):
                raise original
        owner.tables = Broken()
        table = Table(events)
        cleanup = table.failure = OSError("table close failed")
        with self.assertRaises(MemoryError) as caught:
            owner.acquire(table)
        self.assertIs(caught.exception, original)
        self.assertIs(caught.exception.__cause__, cleanup)
        self.assertIs(owner.unpublished, table)
        table.failure = None
        owner.close(lambda: events.append("fence"))
        self.assertTrue(owner.closed)
        self.assertIsNone(owner.unpublished)

    def test_all_close_failures_and_first_native_roots_survive_foreign_mutation(self):
        events = []
        owner = API.PLETables()
        first, second = owner.acquire(Table(events, "first")), owner.acquire(Table(events, "second"))
        primary, cleanup = OSError("first close"), LookupError("second close")
        cause, context = ValueError("prior cause"), RuntimeError("prior context")
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        first.failure = primary

        def mutate():
            events.append("second")
            BaseException.__cause__.__set__(primary, None)
            BaseException.__context__.__set__(primary, None)
            BaseException.__suppress_context__.__set__(primary, False)
            raise cleanup

        second.close = mutate
        with self.assertRaises(OSError) as caught:
            owner.close(lambda: events.append("fence"))
        self.assertIs(caught.exception, primary)
        self.assertEqual(failures(primary), [cause, context, cleanup])
        self.assertEqual(events, ["fence", "first", "second"])
        self.assertEqual(owner.tables, [first, second])
        self.assertFalse(owner.closed)
        first.failure = None
        second.close = lambda: events.append("second retry")
        owner.close(lambda: events.append("fence retry"))
        self.assertTrue(owner.closed)

    def test_completed_future_failure_and_fence_error_keep_prior_roots_and_tables(self):
        owner = API.PLETables()
        table = owner.acquire(Table([]))
        primary, cleanup = OSError("completed read"), LookupError("fence refused")
        cause = ValueError("read prior cause")
        BaseException.__cause__.__set__(primary, cause)
        future = Future()
        future.set_exception(primary)
        owner.reads.append(future)

        def fence():
            BaseException.__cause__.__set__(primary, None)
            raise cleanup

        with self.assertRaises(OSError) as caught:
            owner.close(fence)
        self.assertIs(caught.exception, primary)
        self.assertEqual(failures(primary), [cause, cleanup])
        self.assertEqual(owner.tables, [table])
        self.assertFalse(owner.closed)
        self.assertEqual(owner.reads, [])
        owner.close(lambda: None)
        self.assertTrue(owner.closed)

    def test_acquisition_cleanup_cannot_mutate_publication_native_roots(self):
        owner = API.PLETables()
        primary, cleanup = MemoryError("registry refused"), OSError("table close refused")
        cause, context = ValueError("prior cause"), RuntimeError("prior context")
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)

        class Broken(list):
            def append(self, _):
                raise primary

        owner.tables = Broken()
        table = Table([])

        def close():
            BaseException.__cause__.__set__(primary, None)
            BaseException.__context__.__set__(primary, None)
            raise cleanup

        table.close = close
        with self.assertRaises(MemoryError) as caught:
            owner.acquire(table)
        self.assertIs(caught.exception, primary)
        self.assertEqual(failures(primary), [cause, context, cleanup])
        self.assertIs(owner.unpublished, table)
        table.close = lambda: None
        owner.close(lambda: None)
        self.assertTrue(owner.closed)

    def test_group_allocation_failure_preserves_primary_and_all_status_references(self):
        owner = API.PLETables()
        first, second = owner.acquire(Table([])), owner.acquire(Table([]))
        primary, cleanup, cause = OSError("first"), LookupError("second"), ValueError("prior")
        allocation = MemoryError("controlled group allocation")
        first.failure, second.failure = primary, cleanup
        BaseException.__cause__.__set__(primary, cause)

        def fail(*args):
            raise allocation

        with patch.dict(CLEANUP.__dict__, {"BaseExceptionGroup": fail}), self.assertRaises(OSError) as caught:
            owner.close(lambda: None)
        self.assertIs(caught.exception, primary)
        self.assertIs(BaseException.__cause__.__get__(primary), allocation)
        dictionary = BaseException.__dict__["__dict__"].__get__(primary)
        self.assertEqual(dictionary["_tensorfold_retained_failures"], [cause, cleanup])
        self.assertEqual(owner.tables, [first, second])
        first.failure = second.failure = None
        owner.close(lambda: None)
        self.assertTrue(owner.closed)

    def test_successful_acquisition_cleanup_restores_original_native_roots(self):
        owner = API.PLETables()
        primary, cause, context = MemoryError("registry refused"), ValueError("cause"), RuntimeError("context")
        BaseException.__cause__.__set__(primary, cause)
        BaseException.__context__.__set__(primary, context)
        BaseException.__suppress_context__.__set__(primary, True)

        class Broken(list):
            def append(self, _):
                raise primary

        owner.tables = Broken()
        table = Table([])

        def close():
            BaseException.__cause__.__set__(primary, None)
            BaseException.__context__.__set__(primary, None)
            BaseException.__suppress_context__.__set__(primary, False)

        table.close = close
        with self.assertRaises(MemoryError) as caught:
            owner.acquire(table)
        self.assertIs(caught.exception, primary)
        self.assertIs(BaseException.__cause__.__get__(primary), cause)
        self.assertIs(BaseException.__context__.__get__(primary), context)
        self.assertTrue(BaseException.__suppress_context__.__get__(primary))
        self.assertIsNone(owner.unpublished)
        owner.close(lambda: None)
        self.assertTrue(owner.closed)

    def test_both_real_loader_ple_functions_publish_before_row_and_projection_failures(self):
        tree = ast.parse((FAMILY / "cuda/weights.py").read_text())
        load = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "load")
        for name in ("ple_layer", "ple_nvfp4"):
            function = next(node for node in load.body if isinstance(node, ast.FunctionDef) and node.name == name)
            acquisition = next(node for node in function.body if isinstance(node, ast.Assign)
                               and any(isinstance(target, ast.Name) and target.id == "table" for target in node.targets))
            self.assertIsInstance(acquisition.value, ast.Call)
            self.assertEqual(ast.unparse(acquisition.value.func), "ple_tables.acquire")
            owner = API.PLETables()
            table = Table([])
            table.rows = 4  # actual production rows check fails immediately after acquisition
            value = SimpleNamespace(cpu=lambda: SimpleNamespace(numpy=lambda: ()))
            ngram = SimpleNamespace(rows=3, dims=32, check=lambda *args: None)
            namespace = {"ple_tables": owner, "open_table": lambda *args, **kwargs: table,
                         "cfg": SimpleNamespace(ngram=lambda _: ngram, ngram_shards=1),
                         "raw": lambda _: value, "shard_keys": lambda *args: ["t"], "prefix": "",
                         "rd": SimpleNamespace(where={"t.weight": "table"}), "model_dir": Path("model"),
                         "table_scale": lambda *args: 1.0, "ple_on_ssd": False}
            exec(compile(ast.Module(body=[function], type_ignores=[]), str(FAMILY / "cuda/weights.py"), "exec",
                         flags=__future__.annotations.compiler_flag), namespace)
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "hold 4 rows"):
                namespace[name]("layer.ple", 0)
            self.assertEqual(owner.tables, [table])
            owner.close(lambda: None)
            self.assertEqual(table.closes, 1)

    def test_actual_engine_resource_phase_fences_before_ple_and_keeps_retry_owner(self):
        tree = ast.parse((FAMILY / "cuda/engine.py").read_text())
        engine = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "FlashNextEngine")
        method = next(node for node in engine.body if isinstance(node, ast.FunctionDef) and node.name == "_close_resources")
        namespace = {}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(FAMILY / "cuda/engine.py"), "exec",
                     flags=__future__.annotations.compiler_flag), namespace)
        events = []
        owner = API.PLETables()
        owner.acquire(Table(events))
        engine = SimpleNamespace(w=SimpleNamespace(device="owned-device", meta={"ple_tables": owner}))
        fake = ModuleType("torch")
        fake.cuda = SimpleNamespace(synchronize=lambda _: events.append("fence"))
        with patch.dict(sys.modules, {"torch": fake}):
            namespace["_close_resources"](engine)
        self.assertEqual(events, ["fence", "fence", "table"])
        self.assertTrue(owner.closed)


if __name__ == "__main__":
    unittest.main()

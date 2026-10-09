"""Native table operation journals with real stdlib scheduling, no numeric imports."""

import ast
import __future__
import dis
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import threading
import struct
import stat
from types import SimpleNamespace
import unittest
import tempfile
import weakref
from snapshot_fd_transport import TransportOwner

SOURCE = Path(__file__).resolve().parents[1] / "src/tensorfold/families/qwen4_exp/table_lifetime.py"
spec = importlib.util.spec_from_file_location("ssd_lifetime_source_control", SOURCE)
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


def owner():
    node = next(
        n for n in ast.parse(SOURCE.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "TableLifetime"
    )
    namespace = {"threading": threading, "_note": api._note, "_cause": api._cause}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE), "exec"), namespace)
    calls = []
    fds = [7]
    pool = SimpleNamespace(close=lambda: calls.append("pool"))

    def release(values):
        calls.append("fds")
        values.clear()

    return namespace["TableLifetime"](pool, fds, release), calls


def file_helpers():
    path = SOURCE.parent / "ssd_table.py"
    tree = ast.parse(path.read_text())
    names = {"_retain_file", "_release_table_files", "_span"}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    namespace = {"release_files": api.release_files}
    exec(
        compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag),
        namespace,
    )
    return namespace


def layout_api():
    path = SOURCE.parent / "ssd_table.py"
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "SSDTable")
    method = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "_layout")
    header_path = SOURCE.parents[2] / "cuda/tensor_file.py"
    header_spec = importlib.util.spec_from_file_location("ssd_header_control", header_path)
    header = importlib.util.module_from_spec(header_spec)
    header_spec.loader.exec_module(header)
    core_path = SOURCE.parents[2] / "file_io.py"
    core_spec = importlib.util.spec_from_file_location("ssd_file_core_control", core_path)
    core = importlib.util.module_from_spec(core_spec)
    core_spec.loader.exec_module(core)
    core._owned_slot = TransportOwner  # labeled Python substitute; native atomicity remains ROOT-only
    view_path = SOURCE.parent / "table_file.py"
    view = next(n for n in ast.parse(view_path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == "TableFile")
    view_namespace = {"FileStreams": core.FileStreams, "os": os, "io": io, "stat": stat}
    exec(compile(ast.Module(body=[view], type_ignores=[]), str(view_path), "exec",
                 flags=__future__.annotations.compiler_flag), view_namespace)

    class Array(list):
        flags = SimpleNamespace(writeable=True)

    namespace = dict(
        file_helpers(),
        Path=Path,
        os=os,
        stat=stat,
        _note=api._note,
        _cause=api._cause,
        _no_cache=lambda fd: None,
        read_header_stream=header.read_header_stream,
        SIZES={"U32": 4, "BF16": 2},
        _KINDS=(("weight", "U32", 4), ("scales", "BF16", 2), ("biases", "BF16", 2)),
        np=SimpleNamespace(array=lambda values, **kwargs: Array(values), int64=object()),
        TableFile=view_namespace["TableFile"],
        _file_core=core,
        _file_view_namespace=view_namespace,
    )
    exec(
        compile(
            ast.Module(body=[method], type_ignores=[]), str(path), "exec", flags=__future__.annotations.compiler_flag
        ),
        namespace,
    )
    return namespace


class NativeLifetime(unittest.TestCase):
    def test_shared_native_open_failure_retains_published_SSD_owner_for_retry(self):
        namespace = layout_api()
        primary = KeyboardInterrupt("after labeled native acquisition")
        statuses = [OSError("close unconsumed first"), OSError("close unconsumed second")]
        created = []

        class InterruptedOwner(TransportOwner):
            def open(self, *args, **kwargs):
                super().open(*args, **kwargs)
                created.append(self)
                raise primary

            def close(self):
                if statuses:
                    raise statuses.pop(0)
                super().close()

        namespace["_file_core"]._owned_slot = InterruptedOwner
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            path.write_bytes(b"payload must not be read after failed acquisition")
            instance = SimpleNamespace(_files=[], _fds=[], _unpublished_file=[None])
            with self.assertRaises(KeyboardInterrupt) as caught:
                namespace["_layout"](instance, [(path, {}, {}, {})], False)
            self.assertIs(caught.exception, primary)
            file = instance._unpublished_file[0]
            self.assertIs(file._scope.live_owners[0], created[0])
            descriptor = created[0].fileno()
            self.assertFalse(file.closed)
            with self.assertRaises(OSError):
                namespace["_release_table_files"](
                    instance._fds, files=instance._files, unpublished=instance._unpublished_file
                )
            self.assertIs(instance._unpublished_file[0], file)
            self.assertFalse(created[0].closed)
            namespace["_release_table_files"](
                instance._fds, files=instance._files, unpublished=instance._unpublished_file
            )
            self.assertIsNone(instance._unpublished_file[0])
            self.assertTrue(created[0].closed)
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_interrupted_borrowed_FileIO_return_keeps_native_descriptor_journal(self):
        namespace = layout_api()
        view = namespace["TableFile"]
        offset = next(part.offset for part in dis.get_instructions(view.open) if part.opname == "STORE_ATTR"
                      and part.argval == "raw")
        primary, observations, unraisable = KeyboardInterrupt("before borrowed raw STORE"), [], []

        def trace(frame, event, arg):
            if frame.f_code is view.open.__code__:
                frame.f_trace_opcodes = True
                if event == "opcode" and frame.f_lasti == offset:
                    record = frame.f_locals["record"]
                    observations.append((frame.f_locals["self"], record, record.owner.fileno()))
                    raise primary
            return trace

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            path.write_bytes(b"header must not be read after interrupted return")
            instance = SimpleNamespace(_files=[], _fds=[], _unpublished_file=[None])
            prior, prior_unraisable = sys.gettrace(), sys.unraisablehook
            current = sys._getframe()
            prior_opcodes = current.f_trace_opcodes
            try:
                # CPython 3.12 enables opcode callbacks from the active-frame
                # flags when settrace is installed, before the callee event.
                current.f_trace_opcodes = True
                sys.unraisablehook = lambda status: unraisable.append(status.exc_value)
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    namespace["_layout"](instance, [(path, {}, {}, {})], False)
            finally:
                sys.settrace(prior)
                sys.unraisablehook = prior_unraisable
                current.f_trace_opcodes = prior_opcodes
            self.assertIs(caught.exception, primary)
            file, record, descriptor = observations[0]
            self.assertIs(instance._unpublished_file[0], file)
            self.assertIsNone(record.raw)
            self.assertIs(file._scope.live_owners[0], record.owner)
            self.assertEqual(os.fstat(descriptor).st_size, path.stat().st_size)
            self.assertEqual(unraisable, [])  # discarded closefd=False view carries no descriptor-close status
            namespace["_release_table_files"](
                instance._fds, files=instance._files, unpublished=instance._unpublished_file
            )
            self.assertIsNone(instance._unpublished_file[0])
            with self.assertRaises(OSError):
                os.fstat(descriptor)

    def test_native_cause_keeps_prior_owned_status_and_deduplicates_cleanup(self):
        primary = KeyboardInterrupt("work interrupted")
        prior, cleanup = LookupError("first repair failed"), OSError("outer close failed")
        BaseException.__cause__.__set__(primary, prior)
        combined = api._cause(primary, cleanup)
        self.assertEqual(combined.exceptions, (cleanup, prior))
        BaseException.__cause__.__set__(primary, combined)
        self.assertIs(api._cause(primary, cleanup), combined)

    def test_layout_uses_the_same_strict_opened_header_and_refuses_forged_descriptors(self):
        namespace = layout_api()
        entries = [
            {"dtype": "U32", "shape": [1, 8], "data_offsets": [0, 32]},
            {"dtype": "BF16", "shape": [1, 2], "data_offsets": [32, 36]},
            {"dtype": "BF16", "shape": [1, 2], "data_offsets": [36, 40]},
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            raw = json.dumps(dict(zip(("w", "s", "b"), entries))).encode()
            path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"x" * 40)
            for forged in (False, True):
                selected = json.loads(json.dumps(entries))
                if forged:
                    selected[0]["data_offsets"] = [4, 36]
                instance = SimpleNamespace(_files=[], _fds=[], _unpublished_file=[None])
                try:
                    if forged:
                        with self.assertRaisesRegex(ValueError, "opened checkpoint header"):
                            namespace["_layout"](instance, [(path, *selected)], False)
                    else:
                        namespace["_layout"](instance, [(path, *selected)], False)
                        self.assertEqual(instance.rows, 1)
                        self.assertEqual(instance.bases[0], (8 + len(raw), 40 + len(raw), 44 + len(raw)))
                finally:
                    namespace["_release_table_files"](
                        instance._fds, files=instance._files, unpublished=instance._unpublished_file
                    )

    def test_trace_before_layout_file_publication_retains_a_real_descriptor_for_close(self):
        namespace = layout_api()
        function = next(
            n
            for n in ast.parse((SOURCE.parent / "ssd_table.py").read_text()).body
            if isinstance(n, ast.ClassDef) and n.name == "SSDTable"
        )
        method = next(n for n in function.body if isinstance(n, ast.FunctionDef) and n.name == "_layout")
        line = next(
            n.lineno
            for n in ast.walk(method)
            if isinstance(n, ast.Assign) and any(ast.unparse(t) == "self._unpublished_file[0]" for t in n.targets)
        )
        primary = KeyboardInterrupt("before first descriptor publication")
        captured = []

        def trace(frame, event, arg):
            if (
                frame.f_code.co_filename == str(SOURCE.parent / "ssd_table.py")
                and event == "line"
                and frame.f_lineno == line
            ):
                captured.append(frame.f_locals["file"])
                raise primary
            return trace

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            path.write_bytes(b"opaque bytes: header is not read before interruption")
            instance = SimpleNamespace(_files=[], _fds=[], _unpublished_file=[None])
            prior_trace = sys.gettrace()
            try:
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    namespace["_layout"](instance, [(path, {}, {}, {})], False)
            finally:
                sys.settrace(prior_trace)
            self.assertIs(caught.exception, primary)
            self.assertIs(instance._unpublished_file[0], captured[0])
            self.assertTrue(captured[0].closed)
            self.assertEqual(captured[0]._scope._records, [])  # no acquisition before slot publication
            namespace["_release_table_files"](
                instance._fds, files=instance._files, unpublished=instance._unpublished_file
            )
            self.assertTrue(captured[0].closed)

    def test_both_constructor_cleanup_sites_preserve_malformed_note_primary(self):
        for filename, classname in (("ssd_table.py", "SSDTable"), ("native_ssd.py", "NativeSSDTable")):
            path = SOURCE.parent / filename
            cls = next(
                n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef) and n.name == classname
            )
            primary, cleanup = KeyboardInterrupt("lifetime acquisition interrupted"), OSError("pool cleanup failed")
            primary.__notes__ = 1

            def failed():
                raise cleanup

            pool = SimpleNamespace(close=failed, shutdown=lambda **kwargs: failed())

            def lifetime(*args):
                raise primary

            namespace = dict(
                file_helpers(),
                SSDTable=object,
                ThreadPoolExecutor=lambda *args, **kwargs: pool,
                load_reader=lambda: SimpleNamespace(Reader=lambda _: pool),
                TableLifetime=lifetime,
                PythonPool=api.PythonPool,
                partial=partial,
                weakref=weakref,
                WORKERS=16,
                _note=api._note,
                _cause=api._cause,
            )
            exec(
                compile(
                    ast.Module(body=[cls], type_ignores=[]),
                    str(path),
                    "exec",
                    flags=__future__.annotations.compiler_flag,
                ),
                namespace,
            )
            instance = namespace[classname].__new__(namespace[classname])
            with self.subTest(classname=classname), self.assertRaises(KeyboardInterrupt) as caught:
                instance.__init__([])
            self.assertIs(caught.exception, primary)
            self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
            self.assertIs(primary.__cause__.exceptions[0], cleanup)
            self.assertEqual(instance._unpublished_file, [None])

    def test_unpublished_file_slot_survives_failed_close_and_consumes_original_owner(self):
        helpers = file_helpers()
        with tempfile.TemporaryFile() as actual:

            class File:
                fail = True

                @property
                def closed(self):
                    return actual.closed

                def close(self):
                    if self.fail:
                        raise OSError("controlled unconsumed close")
                    actual.close()

            file = File()
            slot = [None]
            fds = []
            files = []
            helpers["_retain_file"](slot, file)
            life = api.TableLifetime(
                SimpleNamespace(close=lambda: None),
                fds,
                partial(helpers["_release_table_files"], files=files, unpublished=slot),
            )
            with self.assertRaises(OSError):
                life.close()
            self.assertIs(slot[0], file)
            self.assertFalse(life.closed or actual.closed)
            file.fail = False
            life.close()
            self.assertTrue(life.closed and actual.closed)
            self.assertEqual(slot, [None])

    def test_unknown_callback_retirement_publishes_error_before_quiescence(self):
        life, calls = owner()
        primary = OSError("submit accepted before raising")
        interruption = KeyboardInterrupt("worker retirement interrupted")
        paused, release, done = threading.Event(), threading.Event(), threading.Event()
        failures, caught = [], []
        tree = ast.parse(SOURCE.read_text())
        invoke = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "invoke")
        boundary = min(
            n.lineno
            for n in ast.walk(invoke)
            if isinstance(n, ast.If) and ast.unparse(n.test) == "interrupted is not None"
        )

        class Condition(threading.Condition):
            entries = 0

            def __enter__(self):
                if threading.current_thread().name == "ssd-retirement-worker":
                    self.entries += 1
                    if self.entries == 3:
                        raise interruption
                return super().__enter__()

        life.condition = Condition()

        def trace(frame, event, arg):
            if (
                frame.f_code.co_filename == str(SOURCE)
                and frame.f_code.co_name == "invoke"
                and event == "line"
                and frame.f_lineno == boundary
            ):
                paused.set()
                if not release.wait(3):
                    raise TimeoutError("release retirement boundary")
            return trace

        class Pool:
            def submit(self, function, index, item):
                def run():
                    sys.settrace(trace)
                    try:
                        function(index, item)
                    except BaseException as error:
                        failures.append(error)
                    finally:
                        sys.settrace(None)

                self.thread = threading.Thread(target=run, name="ssd-retirement-worker")
                self.thread.start()
                if not paused.wait(2):
                    raise TimeoutError("worker did not pause")
                raise primary

            def shutdown(self, wait=True):
                return  # interrupted Thread.join bookkeeping is not callback completion

        pool = Pool()
        life.pool = api.PythonPool(pool)

        def borrower():
            try:
                life.call(life.parallel, pool, lambda item: None, [0])
            except BaseException as error:
                caught.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=borrower)
        thread.start()
        try:
            self.assertTrue(paused.wait(2))
            self.assertFalse(done.wait(0.02))
            self.assertFalse(life.condition.acquire(blocking=False))
            self.assertEqual(calls, [])
        finally:
            release.set()
            thread.join(3)
            pool.thread.join(3)
        self.assertFalse(thread.is_alive() or pool.thread.is_alive())
        self.assertEqual(caught, [primary])
        self.assertEqual(failures, [interruption])
        self.assertTrue(any("KeyboardInterrupt" in note for note in primary.__notes__))
        self.assertFalse(life.active or life.workers or life.reads)
        life.close()
        self.assertEqual(calls, ["fds"])

    def test_interrupted_join_cannot_release_active_or_late_callback_file_authority(self):
        life, calls = owner()
        entered, release, done = threading.Event(), threading.Event(), threading.Event()
        failures, work_calls = [], []
        primary = KeyboardInterrupt("unknown accepted callback")

        class IncompleteJoin:
            def submit(self, fn, index, item):
                if index == 0:
                    self.thread = threading.Thread(target=lambda: self.run(fn, index, item))
                    self.thread.start()
                    if not entered.wait(2):
                        raise AssertionError("actual file callback did not enter")
                    return SimpleNamespace(result=lambda: None)
                self.late = (fn, index, item)
                raise primary

            def run(self, fn, index, item):
                try:
                    fn(index, item)
                except BaseException as error:
                    failures.append(error)

            def shutdown(self, wait=True):
                return  # model a consumed Thread.join state lock

        pool = IncompleteJoin()
        life.pool = api.PythonPool(pool)

        def work(item):
            work_calls.append(item)
            entered.set()
            if not release.wait(2):
                raise TimeoutError("release controlled read")
            raise LookupError("actual accepted read failed")

        def borrower():
            try:
                life.call(life.parallel, pool, work, [0, 1])
            except BaseException as error:
                failures.append(error)
            finally:
                done.set()

        thread = threading.Thread(target=borrower)
        thread.start()
        self.assertTrue(entered.wait(1))
        self.assertFalse(done.wait(0.02))
        self.assertEqual(calls, [])
        release.set()
        thread.join(2)
        pool.thread.join(2)
        self.assertFalse(thread.is_alive() or pool.thread.is_alive())
        self.assertIn(primary, failures)
        self.assertTrue(any("LookupError" in note for note in primary.__notes__))
        life.close()
        fn, index, item = pool.late
        fn(index, item)  # a late callback cannot reacquire closed file authority
        self.assertEqual(work_calls, [0])
        self.assertFalse(life.active or life.workers or life.reads)

    def test_pool_close_interruption_remains_primary_if_retry_fails(self):
        life, calls = owner()
        primary, secondary = KeyboardInterrupt("first close"), OSError("retry close")
        pending = [primary, secondary]

        def close():
            raise pending.pop(0)

        life.pool = SimpleNamespace(close=close)
        with self.assertRaises(KeyboardInterrupt) as caught:
            life.close()
        self.assertIs(caught.exception, primary)
        self.assertIs(caught.exception.__cause__, secondary)
        self.assertEqual(life.fds, [7])
        self.assertFalse(life.closed)

    def test_malformed_notes_keep_close_primary_cleanup_and_retry_authority(self):
        life, calls = owner()
        primary, secondary = KeyboardInterrupt("first close"), OSError("retry close")
        primary.__notes__ = 1
        pending = [primary, secondary]

        def close():
            if pending:
                raise pending.pop(0)
            calls.append("pool")

        life.pool = SimpleNamespace(close=close)
        with self.assertRaises(KeyboardInterrupt) as caught:
            life.close()
        self.assertIs(caught.exception, primary)
        self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
        self.assertIs(primary.__cause__.exceptions[0], secondary)
        self.assertIsInstance(primary.__cause__.exceptions[1].exceptions[0], TypeError)
        self.assertEqual(life.fds, [7])
        self.assertFalse(life.closed)
        life.close()
        self.assertTrue(life.closed)
        self.assertEqual(calls, ["pool", "fds"])

    def test_shadowed_cause_hooks_cannot_replace_the_native_close_primary(self):
        life, calls = owner()

        class Primary(KeyboardInterrupt):
            @property
            def __cause__(self):
                raise AssertionError("foreign cause getter")

            @__cause__.setter
            def __cause__(self, value):
                raise AssertionError("foreign cause setter")

            def add_note(self, message):
                raise AssertionError("foreign note hook")

        primary, secondary = Primary(), OSError("close failed")
        primary.__notes__ = object()
        pending = [primary, secondary]

        def close():
            raise pending.pop(0)

        life.pool = SimpleNamespace(close=close)
        with self.assertRaises(Primary) as caught:
            life.close()
        self.assertIs(caught.exception, primary)
        cause = BaseException.__cause__.__get__(primary, type(primary))
        self.assertIsInstance(cause, BaseExceptionGroup)
        self.assertIs(cause.exceptions[0], secondary)
        self.assertIsInstance(cause.exceptions[1].exceptions[0], TypeError)
        self.assertEqual(life.fds, [7])
        self.assertFalse(life.closed)

    def test_malformed_notes_keep_parallel_primary_after_all_real_workers_retire(self):
        life, calls = owner()
        primary, secondary = RuntimeError("first read"), LookupError("second read")
        primary.__notes__ = object()
        completed = []
        started = threading.Barrier(2)

        def work(item):
            completed.append(item)
            started.wait(2)
            raise (primary if item == 0 else secondary)

        with ThreadPoolExecutor(2) as pool:
            life.pool = api.PythonPool(pool)
            with self.assertRaises(RuntimeError) as caught:
                life.call(life.parallel, pool, work, [0, 1])
        self.assertIs(caught.exception, primary)
        self.assertEqual(set(completed), {0, 1})
        self.assertFalse(life.active or life.workers or life.reads)
        self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
        self.assertIsInstance(primary.__cause__.exceptions[0], TypeError)
        self.assertEqual(life.fds, [7])
        life.close()
        self.assertTrue(life.closed)
        self.assertEqual(calls, ["fds"])

    def test_constructor_failure_after_pool_acquisition_closes_original_pool(self):
        root = SOURCE.parent
        for filename, classname, pool_name in (
            ("ssd_table.py", "SSDTable", "_pool"),
            ("native_ssd.py", "NativeSSDTable", "_native"),
        ):
            with self.subTest(classname=classname):
                node = next(
                    n
                    for n in ast.parse((root / filename).read_text()).body
                    if isinstance(n, ast.ClassDef) and n.name == classname
                )
                calls = []
                pool = SimpleNamespace(
                    close=lambda: calls.append("close"), shutdown=lambda wait: calls.append("shutdown")
                )
                primary = OSError("lifetime allocation failed")

                def failed_lifetime(*args):
                    raise primary

                namespace = {
                    **file_helpers(),
                    "SSDTable": object,
                    "ThreadPoolExecutor": lambda *a, **k: pool,
                    "load_reader": lambda: SimpleNamespace(Reader=lambda _: pool),
                    "TableLifetime": failed_lifetime,
                    "PythonPool": api.PythonPool,
                    "release_files": api.release_files,
                    "partial": partial,
                    "_note": api._note,
                    "_cause": api._cause,
                    "weakref": weakref,
                    "WORKERS": 16,
                }
                exec(
                    compile(
                        ast.Module(body=[node], type_ignores=[]),
                        str(root / filename),
                        "exec",
                        flags=__future__.annotations.compiler_flag,
                    ),
                    namespace,
                )
                cls = namespace[classname]
                instance = cls.__new__(cls)
                with self.assertRaises(OSError) as caught:
                    instance.__init__([])
                self.assertIs(caught.exception, primary)
                self.assertIs(getattr(instance, pool_name), pool)
                self.assertEqual(calls, ["close" if classname == "NativeSSDTable" else "shutdown"])
                self.assertEqual(instance._files, [])

    def test_file_owner_consumed_close_interrupt_never_closes_reused_descriptor(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bytes"
            path.write_bytes(b"new descriptor authority")
            original = open(path, "rb", buffering=0)
            old_fd = original.fileno()
            replacements = []

            class Consumed:
                @property
                def closed(self):
                    return original.closed

                def close(self):
                    original.close()
                    replacements.append(open(path, "rb", buffering=0))
                    raise KeyboardInterrupt("after close consumption and descriptor reuse")

            files, fds = [Consumed()], [old_fd]
            life = api.TableLifetime(SimpleNamespace(close=lambda: None), fds, partial(api.release_files, files=files))
            try:
                with self.assertRaises(KeyboardInterrupt):
                    life.close()
                self.assertTrue(life.closed)
                self.assertEqual(files, [])
                self.assertEqual(fds, [])
                self.assertEqual(len(replacements), 1)
                self.assertEqual(replacements[0].fileno(), old_fd)
                self.assertEqual(os.pread(old_fd, 24, 0), b"new descriptor authority")
            finally:
                for file in replacements:
                    file.close()

    def test_unpublished_fd_list_still_releases_its_owned_file(self):
        with tempfile.TemporaryFile() as file:
            files, fds = [file], []
            life = api.TableLifetime(SimpleNamespace(close=lambda: None), fds, partial(api.release_files, files=files))
            life.close()
            self.assertTrue(file.closed)
            self.assertEqual(files, [])

    def test_python_parallel_failure_drains_later_reads_and_observes_errors(self):
        life, calls = owner()
        entered, release, ended = threading.Event(), threading.Event(), []
        primary = OSError("first read failed")
        with ThreadPoolExecutor(2) as pool:
            life.pool = api.PythonPool(pool)

            def work(item):
                if item == 0:
                    entered.set()
                    if not release.wait(2):
                        raise AssertionError("second accepted read did not release first")
                    raise primary
                if not entered.wait(2):
                    raise AssertionError("first accepted read did not start")
                release.set()
                ended.append(item)
                raise LookupError("later read failed")

            with self.assertRaises(OSError) as caught:
                life.call(life.parallel, pool, work, [0, 1])
            self.assertIs(caught.exception, primary)
            self.assertEqual(ended, [1])
            self.assertTrue(any("LookupError" in note for note in primary.__notes__))
            self.assertFalse(life.active or life.workers or life.reads)
            with self.assertRaises(ValueError):
                life.call(lambda: None)
        life.close()

    def test_python_submit_accepts_before_interrupt_terminal_journal_is_drained(self):
        life, calls = owner()
        entered = threading.Event()
        primary = KeyboardInterrupt("after actual pool accepts read")

        class InterruptedPool(ThreadPoolExecutor):
            def submit(self, fn, index, job):
                future = super().submit(fn, index, job)
                if index == 1:
                    if not entered.wait(2):
                        raise AssertionError("accepted unknown read did not start")
                    raise primary
                return future

        def work(item):
            if item == 1:
                entered.set()
                raise LookupError("accepted unknown read failure")

        with InterruptedPool(2) as pool:
            life.pool = api.PythonPool(pool)
            with self.assertRaises(KeyboardInterrupt) as caught:
                life.call(life.parallel, pool, work, [0, 1, 2])
            self.assertIs(caught.exception, primary)
            self.assertTrue(any("LookupError" in note for note in primary.__notes__))
            self.assertFalse(life.active or life.workers or life.reads)
        life.close()

    def test_worker_self_close_refuses_before_pool_join(self):
        life, calls = owner()
        with ThreadPoolExecutor(1) as pool:
            life.pool = api.PythonPool(pool)
            with self.assertRaisesRegex(RuntimeError, "cannot close its own"):
                life.call(life.parallel, pool, lambda _: life.close(), [0])
        life.close()

    def test_publication_interruption_rolls_back_its_exact_token(self):
        life, calls = owner()

        class Interrupted(dict):
            def __setitem__(self, key, value):
                super().__setitem__(key, value)
                raise KeyboardInterrupt("after journal publication")

        life.active = Interrupted()
        with self.assertRaises(KeyboardInterrupt):
            life.call(lambda: self.fail("work must not start"))
        self.assertEqual(life.active, {})
        life.close()
        self.assertEqual(calls, ["pool", "fds"])

    def test_trace_interrupt_before_work_cannot_strand_a_borrower(self):
        life, calls = owner()
        tree = ast.parse(SOURCE.read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "call")
        line = next(n.lineno for n in ast.walk(function) if isinstance(n, ast.Return))

        def trace(frame, event, arg):
            if (
                frame.f_code.co_filename == str(SOURCE)
                and frame.f_code.co_name == "call"
                and event == "line"
                and frame.f_lineno == line
            ):
                raise KeyboardInterrupt("before protected work")
            return trace

        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt):
                life.call(lambda: self.fail("interrupted before execution"))
        finally:
            sys.settrace(previous)
        self.assertEqual(life.active, {})
        life.close()

    def test_close_drains_the_entire_borrow_before_releasing_fds(self):
        life, calls = owner()
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        result = []

        def work():
            entered.set()
            if not release.wait(2):
                raise TimeoutError("controlled release")
            result.append("read complete")

        borrower = threading.Thread(target=lambda: life.call(work))
        borrower.start()
        self.assertTrue(entered.wait(1))
        closer = threading.Thread(target=lambda: (life.close(), closed.set()))
        closer.start()
        self.assertFalse(closed.wait(0.02))
        self.assertEqual(calls, [])
        release.set()
        borrower.join(2)
        closer.join(2)
        self.assertFalse(borrower.is_alive() or closer.is_alive())
        self.assertEqual(result, ["read complete"])
        self.assertEqual(calls, ["pool", "fds"])

    def test_borrower_self_close_fails_before_waiting_on_itself(self):
        life, calls = owner()
        with self.assertRaisesRegex(RuntimeError, "cannot close its own"):
            life.call(life.close)
        self.assertEqual(life.active, {})
        self.assertEqual(calls, [])
        life.close()

    def test_retirement_interrupt_retries_and_preserves_work_primary(self):
        life, calls = owner()

        class InterruptedCondition(threading.Condition):
            entries = 0

            def __enter__(self):
                self.entries += 1
                if self.entries == 2:
                    raise KeyboardInterrupt("retirement acquisition")
                return super().__enter__()

        life.condition = InterruptedCondition()
        primary = OSError("read failed")

        def work():
            raise primary

        with self.assertRaises(OSError) as caught:
            life.call(work)
        self.assertIs(caught.exception, primary)
        self.assertIsInstance(caught.exception.__cause__, KeyboardInterrupt)
        self.assertEqual(life.active, {})
        life.close()

    def test_failed_pool_drain_retains_authority_then_retry_closes(self):
        life, calls = owner()
        original = life.pool
        primary = OSError("drain failed")
        life.pool = SimpleNamespace(close=lambda: (_ for _ in ()).throw(primary))
        with self.assertRaises(OSError) as caught:
            life.close()
        self.assertIs(caught.exception, primary)
        self.assertEqual(life.fds, [7])
        with self.assertRaises(ValueError):
            life.call(lambda: 1)
        life.pool = original
        life.close()
        life.close()
        self.assertTrue(life.closed)
        self.assertEqual(life.fds, [])


if __name__ == "__main__":
    unittest.main()

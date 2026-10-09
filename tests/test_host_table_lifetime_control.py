"""Mapped table ownership/FD/thread controls: stdlib only, no numeric runtime execution."""
from __future__ import annotations

import __future__
import ast
import builtins
import gc
import importlib.util
import io
import json
import mmap
import os
from pathlib import Path
import stat
import struct
import sys
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from functools import partial
from types import SimpleNamespace
import unittest
import weakref
from unittest.mock import patch
from snapshot_fd_transport import TransportOwner, substitute_owners

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "src/tensorfold/families/qwen4_exp/host_table.py"


def stdlib_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


LIFE = stdlib_module("mapped_control_lifetime", "src/tensorfold/families/qwen4_exp/table_lifetime.py")
HEADER = stdlib_module("mapped_control_header", "src/tensorfold/cuda/tensor_file.py")
CORE = stdlib_module("mapped_control_file_core", "src/tensorfold/file_io.py")
CORE._owned_slot = TransportOwner  # explicitly labeled test substitute
KINDS = {"U32": 4, "BF16": 2, "U8": 1, "F8_E4M3": 1}


def definitions(*names, extra=None):
    source = ast.parse(SOURCE.read_text())
    nodes = [node for node in source.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    namespace = {"threading": threading, "weakref": weakref, "contextmanager": contextmanager,
                 "partial": partial, "TableLifetime": LIFE.TableLifetime, "PythonPool": LIFE.PythonPool,
                 "release_files": LIFE.release_files, "ThreadPoolExecutor": ThreadPoolExecutor,
                 "Path": Path, "os": os, "stat": stat, "SIZES": KINDS, "GATHER_THREADS": 2,
                 "PREFETCH_READ": 7, "read_header_stream": HEADER.read_header_stream, "_header": HEADER.read_header,
                 "FileStreams": CORE.FileStreams}
    # Fault adapters receive a borrowed descriptor; production always uses
    # actual unbuffered io.FileIO(closefd=False) under the native scope.
    namespace["io"] = SimpleNamespace(FileIO=lambda fd, mode, closefd: (
        namespace["open"](fd, mode, buffering=0, closefd=closefd) if "open" in namespace
        else io.FileIO(fd, mode, closefd=closefd)))
    shared = ast.parse((SOURCE.parent / "table_file.py").read_text())
    file_class = next(node for node in shared.body if isinstance(node, ast.ClassDef) and node.name == "TableFile")
    exec(compile(ast.Module(body=[file_class], type_ignores=[]), str(SOURCE.parent / "table_file.py"), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    namespace["_HostFile"] = namespace["TableFile"]
    if extra:
        namespace.update(extra)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(SOURCE), "exec",
                 flags=__future__.annotations.compiler_flag), namespace)
    return namespace


def safetensors(path, entry=None, payload=b"abcdefgh", raw=None):
    entry = entry or {"dtype": "BF16", "shape": [2, 2], "data_offsets": [0, len(payload)]}
    text = raw if raw is not None else json.dumps({"t.weight": entry}).encode()
    path.write_bytes(struct.pack("<Q", len(text)) + text + payload)
    return entry, 8 + len(text)


class Array:
    """Real stdlib mmap ownership facade; no arithmetic or ndarray emulation."""
    def __init__(self, file, *, dtype, mode, offset, shape):
        self._mmap = mmap.mmap(file.fileno(), 0, access=mmap.ACCESS_READ)
        self.offset, self.shape = offset, shape
        self.nbytes = shape[0] * shape[1] * dtype
        self.filename = file.name

    def bytes(self):
        return self._mmap[self.offset:self.offset + self.nbytes]


def prefetch_dual_failure(test, factory, patch_open):
    """Real executor/FD: hostile exception setters and notes cannot mask work."""
    for malformed in (False, True):
        setter_calls = []
        class Work(RuntimeError):
            def __setattr__(self, name, value):
                if name == "prefetch_lifetime":
                    setter_calls.append(name)
                    raise LookupError("work exception setter")
                return super().__setattr__(name, value)
        class Cleanup(OSError):
            def __setattr__(self, name, value):
                if name == "prefetch_lifetime":
                    setter_calls.append(name)
                    raise LookupError("cleanup exception setter")
                return super().__setattr__(name, value)
        primary, cleanup = Work("actual read failed"), Cleanup("actual close failed")
        if malformed:
            BaseException.__dict__["__dict__"].__get__(primary)["__notes__"] = "malformed notes"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytes"
            path.write_bytes(b"abcdefgh")
            actual = open(path, "rb", buffering=0)
            class File:
                name = str(path)
                fail = True
                @property
                def closed(self):
                    return actual.closed
                def fileno(self):
                    return actual.fileno()
                def seek(self, offset, whence=0):
                    actual.seek(offset, whence)
                def readinto(self, _):
                    raise primary
                def close(self):
                    if self.fail:
                        raise cleanup
                    actual.close()
            file = File()
            try:
                before = set(threading.enumerate())
                with patch_open(lambda *a, **k: file):
                    with test.assertRaises(Work) as caught:
                        factory([SimpleNamespace(filename=path, offset=0, nbytes=8)], 1)
                test.assertIs(caught.exception, primary)
                test.assertFalse(setter_calls)
                dictionary = BaseException.__dict__["__dict__"].__get__(primary)
                owner = dictionary["prefetch_lifetime"]
                test.assertIs(BaseException.__dict__["__dict__"].__get__(cleanup)["prefetch_lifetime"], owner)
                test.assertFalse(owner.closed or actual.closed)
                test.assertFalse(any(thread.is_alive() for thread in set(threading.enumerate()) - before))
                test.assertIsInstance(primary.__cause__, TypeError if malformed else Cleanup)
                file.fail = False
                owner.close()
                test.assertTrue(owner.closed and actual.closed)
            finally:
                actual.close()


def journaled_return_case(test, run, published, *, worker=False):
    """Real FD + Python-return transport; actual C_RETURN/EIO is ROOT-only."""
    primary, cleanup = KeyboardInterrupt("published owner open return"), OSError("consumed close error")
    owners, replacements, fired = [], [], []
    class Interrupted(TransportOwner):
        def __init__(self):
            super().__init__()
            owners.append(self)
        def close(self):
            if not self.closed:
                descriptor = self.fileno()
                super().close()
                replacement = TransportOwner()
                replacements.append(replacement)  # valid closed slot before open
                replacement.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                test.assertEqual(replacement.fileno(), descriptor)
                raise cleanup
    def profile(frame, event, argument):
        if event == "return" and frame.f_code is TransportOwner.open.__code__ and owners and frame.f_locals.get("self") is owners[0]:
            file = published()
            test.assertIs(file._scope._records[0].owner, owners[0])
            test.assertIsNone(file._scope._records[0].raw)
            test.assertFalse(owners[0].closed)
            fired.append(owners[0].fileno())
            raise primary
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "native-return"
        entry, _ = safetensors(path)
        previous = threading.getprofile() if worker else sys.getprofile()
        try:
            if worker:
                threading.setprofile(profile)
            else:
                sys.setprofile(profile)
            with patch.object(CORE, "_owned_slot", Interrupted):
                with test.assertRaises(KeyboardInterrupt) as caught:
                    run(path, entry)
            test.assertIs(caught.exception, primary)
            test.assertEqual(len(fired), 1)
            test.assertTrue(owners[0].closed)
            test.assertIs(BaseException.__cause__.__get__(primary), cleanup)
            test.assertEqual(len(replacements), 1)
            test.assertEqual(os.read(replacements[0].fileno(), 8), path.read_bytes()[:8])
        finally:
            if worker:
                threading.setprofile(previous)
            else:
                sys.setprofile(previous)
            for replacement in replacements:
                replacement.close()


class HostTableControl(unittest.TestCase):
    def setUp(self):
        substitute_owners(self)

    def test_map_open_return_interruption_observes_published_slot_and_explicit_consumed_close(self):
        table, _ = self.owner()
        observed = []
        def run(path, entry):
            with table._construction():
                observed.append(table._files)
                table._map(path, entry, 2, ("weight", "BF16", 2))
        journaled_return_case(self, run, lambda: observed[0][0])
        self.assertTrue(table._life.closed)
        self.assertFalse(table._files or table._fds)

    def test_prefetch_open_return_interruption_observes_published_slot_and_explicit_consumed_close(self):
        namespace = self.prefetch_namespace()
        original = namespace["_prefetch_file"]
        observed = []
        @contextmanager
        def file_context(path, files, guard, slot):
            observed.append(files)
            with original(path, files, guard, slot) as file:
                yield file
        namespace["_prefetch_file"] = file_context
        def run(path, entry):
            namespace["_prefetch"]([SimpleNamespace(filename=path, offset=0, nbytes=8)], 1)
        before = set(threading.enumerate())
        journaled_return_case(self, run, lambda: observed[0][0], worker=True)
        self.assertFalse(observed[0])
        self.assertFalse(any(thread.is_alive() for thread in set(threading.enumerate()) - before))

    def owner(self, extra=None):
        namespace = definitions("_note", "_release_unpublished_file", "_retire_local_file", "_release_mapped", "_unlock_pins", "_joined_copy", "_MappedTable", "_random_access",
                                extra={"np": SimpleNamespace(memmap=Array), "_span": self.span, **(extra or {})})
        return namespace["_MappedTable"](), namespace

    @staticmethod
    def span(entry, kind, data, size, name):
        tree = ast.parse((SOURCE.parent / "ssd_table.py").read_text())
        node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_span")
        namespace = {}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(SOURCE.parent / "ssd_table.py"), "exec"), namespace)
        return namespace["_span"](entry, kind, data, size, name)

    def test_owned_references_drop_but_saved_real_map_stays_readable(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            entry, _ = safetensors(path)
            table, _ = self.owner()
            with table._construction():
                table.values = table._arrays()
                kept = table._map(path, entry, 2, ("weight", "BF16", 2))
                table.values.append(kept)
                self.assertFalse(table._files[0].closed)
            self.assertFalse(table._files or table._sources or table._fds)
            mapping = weakref.ref(kept)
            table.close()
            self.assertEqual(table.values, [])
            self.assertEqual(kept.bytes(), b"abcdefgh")
            with self.assertRaises(ValueError):
                table._life.call(lambda: None)
            table.close()
            del kept
            gc.collect()
            self.assertIsNone(mapping())

    def test_same_descriptor_maps_header_and_payload_after_path_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            replacement = Path(directory) / "replacement"
            entry, _ = safetensors(path)
            safetensors(replacement, payload=b"ABCDEFGH")
            table, namespace = self.owner()
            original = namespace["read_header_stream"]
            def replace(stream, sizes, *, label):
                result = original(stream, sizes, label=label)
                os.replace(replacement, path)
                return result
            namespace["read_header_stream"] = replace
            with table._construction():
                table.values = table._arrays()
                table.values.append(table._map(path, entry, 2, ("weight", "BF16", 2)))
            self.assertEqual(table.values[0].bytes(), b"abcdefgh")
            table.close()

    def test_trace_before_map_slot_publication_observes_valid_closed_slot(self):
        tree = ast.parse(SOURCE.read_text())
        cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "_MappedTable")
        method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_map")
        line = next(node.lineno for node in ast.walk(method) if isinstance(node, ast.Assign)
                    and any(ast.unparse(target) == "self._unpublished_file[0]" for target in node.targets))
        primary = KeyboardInterrupt("before FileIO slot publication")
        captured = []
        def trace(frame, event, argument):
            if event == "line" and frame.f_code.co_filename == str(SOURCE) and frame.f_code.co_name == "_map" and frame.f_lineno == line:
                captured.append(frame.f_locals["file"])
                raise primary
            return trace
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            entry, _ = safetensors(path)
            table, _ = self.owner()
            prior = sys.gettrace()
            try:
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    with table._construction():
                        table._map(path, entry, 2, ("weight", "BF16", 2))
            finally:
                sys.settrace(prior)
            self.assertIs(caught.exception, primary)
            self.assertEqual(len(captured), 1)
            self.assertTrue(captured[0].closed and table._life.closed)
            self.assertIsNone(table._unpublished_file[0])
            self.assertFalse(table._fds or table._files)

    def test_native_open_interruption_retains_published_owner_for_explicit_retry(self):
        primary, cleanup = KeyboardInterrupt("native open return interrupted"), OSError("unconsumed close")
        refused, owners = [True], []
        class Interrupted(TransportOwner):
            def __init__(self):
                super().__init__()
                owners.append(self)
            def open(self, *args, **kwargs):
                super().open(*args, **kwargs)
                raise primary
            def close(self):
                if refused[0] and not self.closed:
                    raise cleanup
                super().close()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            entry, _ = safetensors(path)
            table, _ = self.owner()
            try:
                with patch.object(CORE, "_owned_slot", Interrupted):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        with table._construction():
                            table._map(path, entry, 2, ("weight", "BF16", 2))
                self.assertIs(caught.exception, primary)
                self.assertFalse(table._life.closed or owners[0].closed)
                kept = table._unpublished_file[0]
                self.assertIs(kept._scope.live_owners[0], owners[0])
                self.assertEqual(os.pread(owners[0].fileno(), 8, 0), path.read_bytes()[:8])
                refused[0] = False
                table.close()
                self.assertTrue(owners[0].closed and table._life.closed)
            finally:
                refused[0] = False
                table.close()

    def test_trace_before_prefetch_slot_publication_observes_valid_closed_slot(self):
        tree = ast.parse(SOURCE.read_text())
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_prefetch_file")
        line = next(node.lineno for node in ast.walk(function) if isinstance(node, ast.Assign)
                    and any(ast.unparse(target) == "slot[0]" for target in node.targets))
        primary = KeyboardInterrupt("before prefetch FileIO slot publication")
        captured = []
        def trace(frame, event, argument):
            if event == "line" and frame.f_code.co_filename == str(SOURCE) and frame.f_code.co_name == "_prefetch_file" and frame.f_lineno == line:
                captured.append(frame.f_locals["file"])
                raise primary
            return trace
        namespace = self.prefetch_namespace()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytes"
            path.write_bytes(b"prefetch bytes")
            files, slot = [], [None]
            prior = sys.gettrace()
            try:
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    with namespace["_prefetch_file"](path, files, threading.Lock(), slot):
                        self.fail("interrupted publication yielded file authority")
            finally:
                sys.settrace(prior)
            self.assertIs(caught.exception, primary)
            self.assertEqual(len(captured), 1)
            self.assertTrue(captured[0].closed)
            self.assertFalse(files)
            self.assertIsNone(slot[0])

    def test_closed_slot_publication_refusal_before_and_after_append_acquires_no_descriptor(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            entry, _ = safetensors(path)
            for after in (False, True):
                primary = KeyboardInterrupt("accepted publication interrupted") if after else MemoryError("publication refused")
                table, _ = self.owner()
                class Refusing(list):
                    def append(self, file):
                        self_test.assertTrue(file.closed)
                        if after:
                            super().append(file)
                        raise primary
                self_test = self
                class Owner(type(table)):
                    def __setattr__(self, name, value):
                        super().__setattr__(name, Refusing() if name == "_files" else value)
                table = Owner()
                with patch.object(CORE, "_owned_slot", side_effect=AssertionError("acquisition before closed-slot journal")):
                    with self.assertRaises(type(primary)) as caught:
                        with table._construction():
                            table._map(path, entry, 2, ("weight", "BF16", 2))
                self.assertIs(caught.exception, primary)
                self.assertTrue(table._life.closed)
                self.assertFalse(table._files or table._fds or table._sources)
                self.assertIsNone(table._unpublished_file[0])

    def test_acquired_owner_close_failure_retains_native_scope_and_retry_authority(self):
        primary, cleanup = MemoryError("native acquisition return interrupted"), OSError("close refused")
        refused, owners = [True], []
        class Interrupted(TransportOwner):
            def __init__(self):
                super().__init__()
                owners.append(self)
            def open(self, *args, **kwargs):
                super().open(*args, **kwargs)
                raise primary
            def close(self):
                if refused[0] and not self.closed:
                    raise cleanup
                super().close()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            entry, _ = safetensors(path)
            table, _ = self.owner()
            try:
                with patch.object(CORE, "_owned_slot", Interrupted):
                    with self.assertRaises(MemoryError) as caught:
                        with table._construction():
                            table._map(path, entry, 2, ("weight", "BF16", 2))
                self.assertIs(caught.exception, primary)
                self.assertFalse(owners[0].closed or table._life.closed)
                owner = table._unpublished_file[0]
                self.assertIs(owner._scope.live_owners[0], owners[0])
                namespace = BaseException.__dict__["__dict__"].__get__(cleanup)
                self.assertIs(dict.__getitem__(namespace, "_tensorfold_host_file_scope"), owner._scope)
                refused[0] = False
                table.close()
                self.assertTrue(owners[0].closed and table._life.closed)
            finally:
                refused[0] = False
                table.close()

    def test_consumed_native_close_interrupt_never_closes_reused_real_descriptor(self):
        primary, interruption = MemoryError("native open return interrupted"), KeyboardInterrupt("close consumed")
        replacements, closes = [], []
        class Interrupted(TransportOwner):
            def open(self, *args, **kwargs):
                super().open(*args, **kwargs)
                raise primary
            def close(self):
                if not self.closed:
                    descriptor = self.fileno()
                    super().close()
                    replacement = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
                    replacements.append(replacement)
                    closes.append(descriptor)
                    self_test.assertEqual(descriptor, replacement)
                    raise interruption
        self_test = self
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            entry, _ = safetensors(path)
            table, _ = self.owner()
            try:
                with patch.object(CORE, "_owned_slot", Interrupted):
                    with self.assertRaises(MemoryError) as caught:
                        with table._construction():
                            table._map(path, entry, 2, ("weight", "BF16", 2))
                self.assertIs(caught.exception, primary)
                self.assertIs(BaseException.__cause__.__get__(primary), interruption)
                self.assertTrue(table._life.closed)
                self.assertEqual(closes, replacements)
                self.assertEqual(len(replacements), 1)
                self.assertEqual(os.read(replacements[0], 8), path.read_bytes()[:8])
            finally:
                for descriptor in replacements:
                    os.close(descriptor)

    def test_in_bounds_supplied_metadata_must_match_the_opened_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            entry, _ = safetensors(path)
            altered = {**entry, "shape": [1, 2], "data_offsets": [2, 6]}
            table, _ = self.owner()
            with self.assertRaisesRegex(ValueError, "disagrees"):
                with table._construction():
                    table._map(path, altered, 2, ("weight", "BF16", 2))
            self.assertTrue(table._life.closed)
            self.assertFalse(table._fds or table._files or table._sources)

    def test_constructor_failure_releases_original_pool_file_and_map(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            entry, _ = safetensors(path)
            table, _ = self.owner()
            failure = LookupError("later shard admission")
            with self.assertRaises(LookupError) as caught:
                with table._construction():
                    table.values = table._arrays()
                    value = table._map(path, entry, 2, ("weight", "BF16", 2))
                    table.values.append(value)
                    file = table._files[0]
                    raise failure
            self.assertIs(caught.exception, failure)
            self.assertTrue(file.closed)
            self.assertFalse(table.values or table._files or table._sources or table._fds)
            self.assertTrue(table._life.closed)
            value._mmap.close()

    def test_failed_lifetime_acquisition_closes_acquired_executor(self):
        calls = []
        pool = SimpleNamespace(shutdown=lambda wait: calls.append(wait))
        failure = MemoryError("lifetime allocation")
        def broken(*args):
            raise failure
        table, _ = self.owner({"ThreadPoolExecutor": lambda *a, **k: pool, "TableLifetime": broken})
        with self.assertRaises(MemoryError) as caught:
            with table._construction():
                self.fail("construction yielded after failed lifetime acquisition")
        self.assertIs(caught.exception, failure)
        self.assertEqual(calls, [True])

    def test_close_waits_for_real_borrower_then_drops_owned_references(self):
        table, _ = self.owner()
        with table._construction():
            values = table._arrays()
            values.append(object())
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        failures = []
        def borrow():
            try:
                table._life.call(lambda: (entered.set(), release.wait(3)))
            except BaseException as error:
                failures.append(error)
        def close():
            try:
                table.close()
            except BaseException as error:
                failures.append(error)
            finally:
                closed.set()
        borrower = threading.Thread(target=borrow)
        closer = threading.Thread(target=close)
        borrower.start()
        self.assertTrue(entered.wait(1))
        closer.start()
        self.assertFalse(closed.wait(.03))
        self.assertEqual(len(values), 1)
        release.set()
        borrower.join(2)
        closer.join(2)
        self.assertFalse(borrower.is_alive() or closer.is_alive())
        self.assertFalse(failures or values)

    def test_observed_copy_failure_drains_other_batch_and_next_copy_works(self):
        table, namespace = self.owner()
        with table._construction():
            pass
        started, release, finished = threading.Event(), threading.Event(), threading.Event()
        original = OSError("copy failed")
        failures = []
        def copy(job):
            if job == 0:
                started.set()
                raise original
            if not release.wait(3):
                raise TimeoutError("controlled release")
            finished.set()
        def call():
            try:
                table._life.call(namespace["_joined_copy"], table._life, table._pool, copy, [0, 1], 2)
            except BaseException as error:
                failures.append(error)
        worker = threading.Thread(target=call)
        worker.start()
        self.assertTrue(started.wait(1))
        self.assertTrue(worker.is_alive())
        release.set()
        worker.join(2)
        self.assertFalse(worker.is_alive())
        self.assertTrue(finished.is_set())
        self.assertEqual(failures, [original])
        self.assertIsNone(table._life.broken)
        out = []
        table._life.call(namespace["_joined_copy"], table._life, table._pool, out.append, list(range(20)), 2)
        self.assertEqual(sorted(out), list(range(20)))
        table.close()

    def test_unknown_accepted_submission_contains_future_before_maps_retire(self):
        table, namespace = self.owner()
        with table._construction():
            pass
        entered, release = threading.Event(), threading.Event()
        primary = KeyboardInterrupt("accepted before Future publication")
        real = table._pool.submit
        def submit(fn, *args):
            real(fn, *args)
            if not entered.wait(1):
                raise AssertionError("accepted callback did not enter")
            release.set()
            raise primary
        table._pool.submit = submit
        def copy(_):
            entered.set()
            if not release.wait(3):
                raise TimeoutError("controlled callback drain")
        with self.assertRaises(KeyboardInterrupt) as caught:
            table._life.call(namespace["_joined_copy"], table._life, table._pool, copy, [0], 1)
        self.assertIs(caught.exception, primary)
        self.assertFalse(table._life.active or table._life.workers or table._life.reads)
        with self.assertRaises(ValueError):
            table._life.call(lambda: None)
        table.close()

    def test_header_reuses_strict_real_fd_protocol(self):
        namespace = definitions("read_header")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "table"
            entry, _ = safetensors(path)
            self.assertEqual(namespace["read_header"](path), {"t.weight": entry})
            for raw in (b'{"x":{},"x":{}}', b'{"x":NaN}', b'[]'):
                safetensors(path, raw=raw)
                with self.assertRaises(ValueError):
                    namespace["read_header"](path)
            path.unlink()
            os.mkfifo(path)
            with self.assertRaises(ValueError):
                namespace["read_header"](path)

    def prefetch_namespace(self, **extra):
        return definitions("_note", "_retain_prefetch_lifetime", "_release_unpublished_file", "_retire_local_file", "_release_prefetch_files", "_joined_copy",
                           "_prefetch_file", "_prefetch", extra=extra)

    def test_prefetch_finishes_real_short_reads_and_drains_owned_files(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytes"
            path.write_bytes(b"0000abcdefghijklmnopqrstuvwxyz")
            read_spans, opened = [], []
            real = open
            class Short:
                def __init__(self, *args, **kwargs):
                    self.file = real(*args, **kwargs)
                    opened.append(self)
                @property
                def closed(self):
                    return self.file.closed
                def fileno(self):
                    return self.file.fileno()
                def seek(self, offset, whence=0):
                    return self.file.seek(offset, whence)
                def readinto(self, out):
                    at = self.file.tell()
                    got = self.file.readinto(out[:2])
                    read_spans.append((at, bytes(out[:got])))
                    return got
                def close(self):
                    self.file.close()
            namespace = self.prefetch_namespace(open=Short)
            array = SimpleNamespace(filename=path, offset=4, nbytes=26)
            self.assertGreaterEqual(namespace["_prefetch"]([array], 3), 0)
            self.assertTrue(all(file.closed for file in opened))
            actual = b"".join(data for _, data in sorted(read_spans))
            self.assertEqual(actual, b"abcdefghijklmnopqrstuvwxyz")
            self.assertEqual(len(actual), 26)

    def test_prefetch_early_eof_fails_and_thread_resources_quiesce(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytes"
            path.write_bytes(b"abcdefgh")
            real = open
            closed = []
            class EOF:
                def __init__(self, *args, **kwargs):
                    self.file = real(*args, **kwargs)
                @property
                def closed(self):
                    return self.file.closed
                def fileno(self):
                    return self.file.fileno()
                def seek(self, at, whence=0):
                    self.file.seek(at, whence)
                def readinto(self, _):
                    return 0
                def close(self):
                    self.file.close()
                    closed.append(self.file.closed)
            namespace = self.prefetch_namespace(open=EOF)
            before = set(threading.enumerate())
            with self.assertRaisesRegex(OSError, "short read"):
                namespace["_prefetch"]([SimpleNamespace(filename=path, offset=0, nbytes=8)], 2)
            self.assertTrue(closed and all(closed))
            self.assertFalse(any(thread.is_alive() for thread in set(threading.enumerate()) - before))

    def test_prefetch_published_native_scope_survives_failed_close_until_explicit_retry(self):
        primary, cleanup = MemoryError("native open return interrupted"), OSError("close refused")
        refused, owners = [True], []
        class Interrupted(TransportOwner):
            def __init__(self):
                super().__init__()
                owners.append(self)
            def open(self, *args, **kwargs):
                super().open(*args, **kwargs)
                raise primary
            def close(self):
                if refused[0] and not self.closed:
                    raise cleanup
                super().close()
        namespace = self.prefetch_namespace()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytes"
            path.write_bytes(b"prefetch")
            files, slot = [], [None]
            try:
                with patch.object(CORE, "_owned_slot", Interrupted):
                    with self.assertRaises(MemoryError) as caught:
                        with namespace["_prefetch_file"](path, files, threading.Lock(), slot):
                            self.fail("interrupted native open yielded")
                self.assertIs(caught.exception, primary)
                self.assertEqual(len(files), 1)
                self.assertFalse(files[0].closed or owners[0].closed)
                self.assertIsNone(slot[0])
                refused[0] = False
                namespace["_release_prefetch_files"]([], files=files, slots=[slot])
                self.assertTrue(owners[0].closed)
                self.assertFalse(files)
            finally:
                refused[0] = False
                namespace["_release_prefetch_files"]([], files=files, slots=[slot])

    def test_read_ahead_wrapper_failure_closes_acquired_real_file_owner(self):
        primary = MemoryError("wrapper construction failed")
        with tempfile.TemporaryFile() as actual:
            table = SimpleNamespace(close=actual.close)
            def failed(_):
                raise primary
            namespace = definitions("_note", "_ssd_table", extra={"SSDTable": lambda _: table,
                                                                    "ReadAhead": failed})
            with patch.dict(os.environ, {"TENSORFOLD_SSD_NATIVE_THREADS": ""}):
                with self.assertRaises(MemoryError) as caught:
                    namespace["_ssd_table"]([])
            self.assertIs(caught.exception, primary)
            self.assertTrue(actual.closed)

    def test_real_prefetch_dual_failure_opaque_setters_and_malformed_notes_keep_primary(self):
        namespace = self.prefetch_namespace()
        @contextmanager
        def replace_open(function):
            namespace["open"] = function
            yield
        prefetch_dual_failure(self, namespace["_prefetch"], replace_open)

    def test_annotation_provider_raising_same_primary_does_not_create_self_cause(self):
        namespace = definitions("_note")
        class Same(RuntimeError):
            def __getattribute__(self, name):
                if name == "__notes__":
                    raise self
                return super().__getattribute__(name)
        primary = Same("primary")
        with self.assertRaises(Same) as caught:
            namespace["_note"](primary, "cleanup note")
        self.assertIs(caught.exception, primary)
        self.assertIsNot(primary.__cause__, primary)

    def test_prefetch_worker_admission_precedes_pool_or_file_acquisition(self):
        namespace = self.prefetch_namespace(ThreadPoolExecutor=lambda *a, **k: self.fail("pool acquired"))
        for workers in (0, -1, True, 1.5, "3"):
            with self.subTest(workers=workers), self.assertRaises(ValueError):
                namespace["_prefetch"]([], workers)

    def test_pin_interruption_keeps_rollback_authority_and_serializes_other_lockers(self):
        entered, release = threading.Event(), threading.Event()
        calls = []
        primary = KeyboardInterrupt("pin syscall returned before interruption")
        class Call:
            def __init__(self, function):
                self.function = function
            def __call__(self, *args):
                return self.function(*args)
        def lock(*args):
            calls.append("lock")
            entered.set()
            if not release.wait(3):
                raise TimeoutError("controlled pin release")
            raise primary
        fake = SimpleNamespace(CDLL=lambda *a, **k: SimpleNamespace(
            mlock=Call(lock), munlock=Call(lambda *args: calls.append("unlock") or 0)),
            c_void_p=object, c_size_t=object)
        original = builtins.__import__
        def importing(name, *args, **kwargs):
            return fake if name == "ctypes" else original(name, *args, **kwargs)
        table, _ = self.owner()
        with table._construction():
            arrays = table._arrays()
            arrays.append(SimpleNamespace(ctypes=SimpleNamespace(data=42), nbytes=8))
        failures = []
        def pin():
            try:
                table._life.call(table._lock_arrays, arrays)
            except BaseException as error:
                failures.append(error)
        with patch("builtins.__import__", side_effect=importing):
            thread = threading.Thread(target=pin)
            thread.start()
            self.assertTrue(entered.wait(1))
            self.assertEqual(len(table._pins), 1)
            self.assertIs(table._pins[0][0], arrays[0])
            closed = threading.Event()
            closer = threading.Thread(target=lambda: (table.close(), closed.set()))
            closer.start()
            self.assertFalse(closed.wait(.03))
            self.assertEqual(calls, ["lock"])
            release.set()
            thread.join(2)
            closer.join(2)
            self.assertFalse(thread.is_alive() or closer.is_alive())
        self.assertEqual(failures, [primary])
        self.assertEqual(calls, ["lock", "unlock"])
        self.assertFalse(table._pins or arrays)

    def test_failed_unpin_retains_arrays_until_explicit_close_retry(self):
        table, _ = self.owner()
        results = [1, 0]
        with table._construction():
            arrays = table._arrays()
            held = object()
            arrays.append(held)
            table._pins.append((held, lambda *args: results.pop(0), 42, 8, False))
        with self.assertRaises(OSError):
            table.close()
        self.assertFalse(table._life.closed)
        self.assertEqual(arrays, [held])
        self.assertIs(table._pins[0][0], held)
        table.close()
        self.assertTrue(table._life.closed)
        self.assertFalse(table._pins or arrays)

    def test_unlock_retains_authority_on_failure_and_consumes_only_success(self):
        namespace = definitions("_unlock_pins")
        results = [1, 0]
        kept = object()
        pins = [(kept, lambda *args: results.pop(0), 42, 8, False)]
        with self.assertRaises(OSError):
            namespace["_unlock_pins"](pins)
        self.assertEqual(pins[0][0], kept)
        namespace["_unlock_pins"](pins)
        self.assertEqual(pins, [])


if __name__ == "__main__":
    unittest.main()

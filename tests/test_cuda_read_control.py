"""Owned CUDA reader control with actual stdlib Futures, no accelerator imports."""
import ast
from concurrent.futures import Future
import errno
import json
import os
from pathlib import Path
import struct
import tempfile
import threading
from types import SimpleNamespace
import unittest

from tensorfold.cuda.capacity import SIZES
from tensorfold.cuda.tensor_file import MAX_HEADER_BYTES, MAX_DIMENSION, byte_range, read_header, tensor_shape
from snapshot_fd_transport import substitute_owners

ROOT = Path(__file__).resolve().parents[1]


def controls():
    path = ROOT / "src/tensorfold/cuda/direct_read.py"
    tree = ast.parse(path.read_bytes())
    selected = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef))
                and node.name in ("ReadAhead", "Reader", "in_background", "wait_all")]
    body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *selected]
    namespace = {"torch": SimpleNamespace(_C=SimpleNamespace()), "threading": threading,
                 "Reader": object, "byte_range": byte_range, "os": os, "errno": errno, "PIECE": 64 << 20}
    exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
    return SimpleNamespace(**{name: namespace[name] for name in ("ReadAhead", "Reader", "in_background", "wait_all")})


class ReadControl(unittest.TestCase):
    def setUp(self):
        self.api = controls()

    def owner(self, failure=None):
        class Transport:
            def read(inner, path, begin, count):
                if failure:
                    raise failure
                return bytes(range(begin, begin + count))
        return self.api.ReadAhead(reader=Transport(), threads=1, run=4, gap=0)

    def test_taken_failures_are_observed_once_but_dropped_failures_surface_at_close(self):
        for consumed in (False, True):
            primary = OSError("owned read failed")
            owner = self.owner(primary)
            owner.queue([("x", "fixture", 0, 4, None)], cut=lambda raw, _: raw)
            if consumed:
                with self.assertRaises(OSError) as caught:
                    owner.take("x")
                self.assertIs(caught.exception, primary)
                owner.close()
            else:
                owner.drop(["x"])
                with self.assertRaises(OSError) as caught:
                    owner.close()
                self.assertIs(caught.exception, primary)
            self.assertIsNone(owner.pool)
            self.assertFalse(owner._owned)
            owner.close()

    def test_close_waits_for_dropped_running_read_and_allows_owned_next_window(self):
        entered, release, closed = threading.Event(), threading.Event(), threading.Event()
        owner = self.owner()

        def blocked(*unused):
            entered.set()
            self.assertTrue(release.wait(2))
            return b"abcd"

        owner.reader.read = blocked
        owner.queue([("x", "fixture", 0, 4, None)], cut=lambda raw, _: raw)
        self.assertTrue(entered.wait(1))
        owner.drop(["x"])
        thread = threading.Thread(target=lambda: (owner.close(), closed.set()))
        thread.start()
        self.assertFalse(closed.wait(.02))
        with self.assertRaises(RuntimeError):
            owner.queue([("y", "fixture", 0, 4, None)])
        release.set()
        thread.join(2)
        self.assertTrue(closed.is_set())
        owner.queue([("y", "fixture", 0, 4, None)], cut=lambda raw, _: raw)
        self.assertEqual(owner.take("y"), b"abcd")
        owner.close()

    def test_worker_self_close_refuses_instead_of_joining_itself(self):
        owner = self.owner()
        owner.queue([("x", "fixture", 0, 4, None)], cut=lambda *args: owner.close())
        with self.assertRaisesRegex(RuntimeError, "cannot close its own"):
            owner.take("x")
        owner.close()

    def test_worker_control_reentry_refuses_before_waiting_for_borrower_lock(self):
        for name, args in (("queue", ([],)), ("take", ("x",)), ("drop", (["x"],))):
            owner = self.owner()
            entered = threading.Event()

            def copy(*unused):
                self.assertTrue(entered.wait(2))
                return getattr(owner, name)(*args)

            owner.queue([("x", "fixture", 0, 4, None)], cut=copy)
            # Hold the actual borrower lock while its Future runs the copier.
            # A worker must reject before trying to acquire this lock.
            with owner._lock:
                entered.set()
                with self.assertRaisesRegex(RuntimeError, "cannot reenter"):
                    owner.take("x")
            owner.close()

    def test_stream_fence_failure_retains_owner_and_retry_finishes(self):
        owner = self.owner()
        primary = RuntimeError("stream fence failed")
        calls = []

        def synchronize():
            calls.append(1)
            if len(calls) == 1:
                raise primary

        stream = SimpleNamespace(synchronize=synchronize)
        owner.stream = stream
        with self.assertRaises(RuntimeError) as caught:
            owner.close()
        self.assertIs(caught.exception, primary)
        self.assertIs(owner.stream, stream)
        with self.assertRaises(RuntimeError):
            owner.queue([])
        owner.close()
        self.assertIsNone(owner.stream)

    def test_malformed_or_duplicate_ranges_refuse_before_any_pool(self):
        for items in ([('x', 'p', -1, 1, None)], [('x', 'p', 0, True, None)],
                      [('x', 'p', 2, 1, None)], [('x', 'p', 0, 1)],
                      [('x', 'p', 0, 1, None), ('x', 'p', 1, 2, None)]):
            owner = self.owner()
            with self.assertRaises(ValueError):
                owner.queue(items)
            self.assertIsNone(owner.pool)

    def test_wait_all_observes_every_result_after_first_failure_or_cancel(self):
        for cancelled in (False, True):
            first, second, third = Future(), Future(), Future()
            primary = RuntimeError("first")
            if cancelled:
                first.cancel()
            else:
                first.set_exception(primary)
            second.set_exception(ValueError("second"))
            third.set_result("ok")
            futures = [first, second, third]
            with self.assertRaises(BaseException) as caught:
                self.api.wait_all(futures)
            if not cancelled:
                self.assertIs(caught.exception, primary)
            self.assertFalse(futures)
            self.assertEqual(len(caught.exception.__notes__), 1)

    def test_background_publication_failure_drains_started_job(self):
        entered, release, completed = threading.Event(), threading.Event(), threading.Event()
        primary = RuntimeError("append failed")

        class Reject:
            def append(self, value):
                self.assertion = entered.wait(1)
                release.set()
                raise primary

        def job():
            entered.set()
            self.assertTrue(release.wait(2))
            completed.set()

        target = Reject()
        with self.assertRaises(RuntimeError) as caught:
            self.api.in_background(job, target)
        self.assertIs(caught.exception, primary)
        self.assertTrue(target.assertion)
        self.assertTrue(completed.is_set())

    def test_buffered_reader_checks_current_descriptor_before_allocating(self):
        reader = self.api.Reader()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bytes"
            path.write_bytes(b"a")
            # The SDK double intentionally has no empty(): any allocation
            # before descriptor validation would fail with AttributeError.
            with self.assertRaises(OSError):
                reader._buffered(path, 0, 2)

    def test_drop_callback_failure_is_observed_without_foreign_formatting(self):
        owner = self.owner()
        primary = RuntimeError("discard failed")

        owner.queue([("x", "fixture", 0, 4, None)], cut=lambda raw, _: raw)
        future = owner.ahead["x"]
        # Force the callback to execute synchronously if its worker finished,
        # covering the callback boundary without relying on scheduling.
        future.result()
        original = owner._retire
        owner._retire = lambda: (_ for _ in ()).throw(primary)
        try:
            owner.drop(["x"])
        except RuntimeError as direct:
            self.assertIs(direct, primary)  # direct control retirement also fails
        owner._retire = original
        with self.assertRaises(RuntimeError) as caught:
            owner.close()
        self.assertIs(caught.exception, primary)
        self.assertIsNone(owner._discard_error)


class FamilyReader(unittest.TestCase):
    def reader(self, directory):
        path = ROOT / "src/tensorfold/families/qwen4_exp/cuda/reader.py"
        tree = ast.parse(path.read_bytes())
        node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "_Reader")
        namespace = {"json": json, "os": os, "Path": Path, "_DT": {"BF16": "BF16"}}
        body = [ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node]
        exec(compile(ast.fix_missing_locations(ast.Module(body=body, type_ignores=[])), str(path), "exec"), namespace)
        owner = namespace["_Reader"].__new__(namespace["_Reader"])
        owner.dir = Path(directory)
        header = {"x": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}
        raw = json.dumps(header).encode()
        (owner.dir / "file").write_bytes(struct.pack("<Q", len(raw)) + raw + b"ab")
        owner.where, owner.headers, owner.touched = {"x": "file"}, {}, set()
        owner._header = lambda shard: (8 + len(raw), header)
        calls = []
        owner.device = "cpu"
        owner.reads = SimpleNamespace(queue=lambda *a: calls.append(a), take=lambda name: None)
        owner.io = SimpleNamespace(read=lambda *a: calls.append(a))
        return owner, header, calls

    def test_queue_and_get_validate_mutable_metadata_before_any_io(self):
        with tempfile.TemporaryDirectory() as directory:
            owner, header, calls = self.reader(directory)
            owner.queue(["x"])
            self.assertEqual(len(calls), 1)
            calls.clear()
            header["x"]["data_offsets"][1] += 2
            for operation in (lambda: owner.queue(["x"]), lambda: owner.get("x")):
                with self.assertRaisesRegex(ValueError, "byte range"):
                    operation()
            self.assertFalse(calls)

    def test_changed_index_target_is_reauthorized_before_a_read(self):
        with tempfile.TemporaryDirectory() as directory:
            owner, _, calls = self.reader(directory)
            owner.where["x"] = "../outside"
            with self.assertRaisesRegex(ValueError, "parent directories"):
                owner.get("x")
            self.assertFalse(calls)

    def test_completed_payload_failure_releases_staging_but_incomplete_owner_retains_it(self):
        with tempfile.TemporaryDirectory() as directory:
            for incomplete in (False, True):
                owner, _, calls = self.reader(directory)
                primary = RuntimeError("read failed")

                def failed_close():
                    raise primary

                owner.reads = SimpleNamespace(close=failed_close, _shutdown_pending=incomplete)
                owner.io.close = lambda: calls.append("staging-close")
                with self.assertRaises(RuntimeError) as caught:
                    owner.close()
                self.assertIs(caught.exception, primary)
                self.assertEqual(calls, [] if incomplete else ["staging-close"])


class TensorFile(unittest.TestCase):
    def setUp(self):
        substitute_owners(self)

    def test_empty_shapes_still_obey_native_dimension_and_checked_multiply_ranges(self):
        for shape in ([2**64, 0], [MAX_DIMENSION + 1, 0], [MAX_DIMENSION, 3, 0]):
            with self.assertRaisesRegex(ValueError, "range|int64|usize"):
                tensor_shape(shape)
        for shape in ([], [0], [0, MAX_DIMENSION], [MAX_DIMENSION, 0]):
            self.assertEqual(tensor_shape(shape), 1 if not shape else 0)
    def file(self, directory, entries, payload=b"\x00\x00", *, raw=None):
        path = Path(directory) / "tensor.safetensors"
        header = json.dumps(entries).encode() if raw is None else raw
        path.write_bytes(struct.pack("<Q", len(header)) + header + payload)
        return path

    def test_scalar_empty_unicode_and_metadata_keep_exact_valid_geometry(self):
        with tempfile.TemporaryDirectory() as directory:
            entries = {"empty": {"dtype": "BF16", "shape": [0, 9], "data_offsets": [0, 0]},
                       "x☃": {"dtype": "BF16", "shape": [], "data_offsets": [0, 2]},
                       "__metadata__": {"note": "valid☃"}}
            path = self.file(directory, entries)
            base, actual = read_header(path, SIZES)
            self.assertEqual(actual, entries)
            self.assertEqual(base + 2, path.stat().st_size)

    def test_malformed_ranges_types_cover_overlap_gaps_truncation_and_bool(self):
        entry = {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}
        cases = [{"x": {**entry, "shape": [True]}}, {"x": {**entry, "data_offsets": [0, True]}},
                 {"x": {**entry, "data_offsets": [-1, 1]}}, {"x": {**entry, "shape": [2]}},
                 {"x": {**entry, "dtype": "UNKNOWN"}}, {"x": entry, "y": entry},
                 {"x": {**entry, "data_offsets": [1, 3]}}, {"__metadata__": {"note": 1}}]
        with tempfile.TemporaryDirectory() as directory:
            for entries in cases:
                with self.assertRaises((ValueError, OSError)):
                    read_header(self.file(directory, entries), SIZES)
            with self.assertRaises(ValueError):
                read_header(self.file(directory, {"x": entry}, payload=b"\x00\x00\x00"), SIZES)
            with self.assertRaises(ValueError):
                read_header(self.file(directory, {}, raw=b'{"x":{},"x":{}}'), SIZES)
            with self.assertRaises(ValueError):
                read_header(self.file(directory, {}, raw=b'{"x":NaN}'), SIZES)
            path = Path(directory) / "huge"
            path.write_bytes(struct.pack("<Q", MAX_HEADER_BYTES + 1))
            with path.open("r+b") as stream:
                stream.truncate(MAX_HEADER_BYTES + 9)
            with self.assertRaises(ValueError):
                read_header(path, SIZES)

    def test_byte_spans_reject_bad_inputs_and_allow_empty_eof(self):
        for offset, count in ((-1, 0), (0, -1), (True, 1), (0, 1.0)):
            with self.assertRaises(ValueError):
                byte_range(offset, count, 2, "fixture")
        for offset, count in ((3, 0), (1, 2), (0, 2**64)):
            with self.assertRaises(OSError):
                byte_range(offset, count, 2, "fixture")
        byte_range(2, 0, 2, "fixture")


if __name__ == "__main__":
    unittest.main()

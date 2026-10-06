"""CPU-only independent byte oracle, boundary, concurrency and lifetime tests."""

import errno
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from tensorfold.families.qwen4_exp.ssd_table import SSDTable
from tensorfold.families.qwen4_exp.native_ssd import NativeSSDTable, NativeReadAhead, load_reader
from tensorfold.families.qwen4_exp import host_table

pytestmark = [
    pytest.mark.torch,
    pytest.mark.skipif(sys.platform != "linux", reason="native SSD reader requires Linux lifetime and pread contracts"),
]


class NativeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.mod = load_reader()
        cls.tmp = tempfile.TemporaryDirectory()
        cls.files, cls.parts, cls.starts = [], [], [0]
        rng = np.random.default_rng(913)
        for i, count in enumerate((17003, 53, 481)):
            parts = [rng.integers(0, 256, (count, width), dtype=np.uint8) for width in (80, 10, 10)]
            entries, at = [], 0
            for part, width, dtype, item in zip(parts, (80, 10, 10), ("U32", "BF16", "BF16"), (4, 2, 2)):
                entries.append(
                    {"shape": [count, width // item], "dtype": dtype, "data_offsets": [at, at + part.nbytes]}
                )
                at += part.nbytes
            header = json.dumps(dict(zip(("w", "s", "b"), entries))).encode()
            path = Path(cls.tmp.name) / f"shard{i}.safetensors"
            with path.open("wb") as f:
                f.write(struct.pack("<Q", len(header)))
                f.write(header)
                for part in parts:
                    f.write(part.tobytes())
            cls.files.append((path, *entries))
            cls.parts.append(parts)
            cls.starts.append(cls.starts[-1] + count)
        cls.oracle = [np.concatenate([p[i] for p in cls.parts]) for i in range(3)]

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def thread_baseline(self):
        """Settle the /proc census after previously joined native/Python threads.

        task-artifacts/native-thread-suite-repro.log records joined thread TIDs
        that remained visible briefly and disappeared by the next 1-ms sample.
        Three unchanged 5-ms samples prevent borrowing a preceding test's
        transient TID as this test's baseline.
        """
        deadline = time.monotonic() + 1.0
        previous = set(os.listdir("/proc/self/task"))
        unchanged = 0
        while time.monotonic() < deadline:
            time.sleep(0.005)
            current = set(os.listdir("/proc/self/task"))
            unchanged = unchanged + 1 if current == previous else 0
            if unchanged == 3:
                return current
            previous = current
        self.fail(f"thread census did not settle: {sorted(previous)}")

    def assert_threads_released(self, before):
        """Joined workers must disappear within a bounded observation window."""
        deadline = time.monotonic() + 1.0
        while True:
            current = set(os.listdir("/proc/self/task"))
            if current == before:
                return
            if time.monotonic() >= deadline:
                self.assertEqual(current, before, "persistent thread identity leak after joins")
            time.sleep(0.001)

    def assert_rows(self, got, ids):
        flat = np.asarray(ids, dtype=np.int64).reshape(-1)
        for result, source, dtype in zip(got, self.oracle, (np.uint32, np.uint16, np.uint16)):
            np.testing.assert_array_equal(result, source[flat].copy().view(dtype))

    def test_properties_and_canonical_runs(self):
        rng = np.random.default_rng(119)
        cases = [
            np.array([], dtype=np.int64),
            np.array([], dtype=float),
            np.arange(17003),
            np.array([0, 17002, 17003, 17055, 17056, 17536, 0, 17003])[::-1],
        ]
        cases += [rng.integers(0, self.starts[-1], size=(rng.integers(1, 400), 3)) for _ in range(40)]
        for workers in (1, 16, 32, 64):
            table = NativeSSDTable(self.files, workers=workers)
            try:
                for ids in cases:
                    self.assert_rows(table.gather(ids), ids)
                unique = np.arange(self.starts[-1], dtype=np.int64)
                shard = np.searchsorted(table.starts, unique, side="right") - 1
                where = table.fidx[shard]
                local = unique - table.starts[shard]
                at = 0
                for part, width in enumerate((table.wrow, table.grow, table.grow)):
                    offsets = table.bases[shard, part] + local * width
                    original = table._reads(where, offsets, width, np.empty((unique.size, width), np.uint8))
                    expected = np.array(
                        [[fd, offset, size, dest + at] for fd, offset, size, _, dest in original], np.int64
                    )
                    np.testing.assert_array_equal(table._runs(where, offsets, width, at), expected)
                    at += unique.size * width
            finally:
                table.close()
            with self.assertRaises(ValueError):
                table.gather([0])
            table.close()

    def test_constructor_failure_drains_and_closes(self):
        before = self.thread_baseline()
        fds_before = len(os.listdir("/proc/self/fd"))
        broken = Path(self.tmp.name) / "truncated.safetensors"
        broken.write_bytes(b"1234")
        wrong = dict(self.files[0][1])
        wrong["dtype"] = "U8"
        cases = [[], [(self.files[0][0], wrong, *self.files[0][2:])], [(broken, *self.files[0][1:])]]
        for workers in (16, 32, 64):
            for files in cases:
                with self.assertRaises(ValueError):
                    NativeSSDTable(files, workers=workers)
                self.assertEqual(len(os.listdir("/proc/self/fd")), fds_before)
                self.assert_threads_released(before)

    def test_canonical_host_table_selector(self):
        key = "TENSORFOLD_SSD_NATIVE_THREADS"
        old = os.environ.pop(key, None)
        try:
            table = host_table._ssd_table(self.files, read_ahead=False)
            try:
                self.assertIs(type(table), SSDTable)
            finally:
                table.close()
            for count in ("16", "32", "64"):
                os.environ[key] = count
                table = host_table._ssd_table(self.files)
                try:
                    self.assertIsInstance(table, NativeReadAhead)
                    self.assert_rows(table.gather([0, 17003]), [0, 17003])
                finally:
                    table.close()
                table = host_table._ssd_table(self.files, read_ahead=False)
                try:
                    self.assertIsInstance(table, NativeSSDTable)
                finally:
                    table.close()
            os.environ[key] = "17"
            with self.assertRaises(ValueError):
                host_table._ssd_table(self.files)
        finally:
            os.environ.pop(key, None)
            if old is not None:
                os.environ[key] = old

    def test_id_errors(self):
        table = NativeSSDTable(self.files)
        try:
            for ids in ([self.starts[-1]], [-1], np.array([2**64 - 1], np.uint64)):
                with self.assertRaises(ValueError):
                    table.gather(ids)
            for ids in ([1.1], ["3"], [True], np.array([2], object)):
                with self.assertRaises(TypeError):
                    table.gather(ids)
        finally:
            table.close()

    def test_boundary_and_io_errors(self):
        reader = self.mod.Reader(16)
        path = Path(self.tmp.name) / "boundary.bin"
        path.write_bytes(bytes(range(256)) * 1024)
        fd = os.open(path, os.O_RDONLY)
        out = np.zeros(128, np.uint8)
        try:
            valid = np.array([[fd, 10, 16, 0], [fd, 100, 32, 16]], np.int64)
            reader.read(valid, out)
            np.testing.assert_array_equal(out[:16], np.arange(10, 26, dtype=np.uint8))
            malformed = [
                valid.astype(np.int32),
                valid[:, ::2],
                valid.reshape(-1),
                np.array([[-1, 0, 1, 0]], np.int64),
                np.array([[2**40, 0, 1, 0]], np.int64),
                np.array([[fd, -1, 1, 0]], np.int64),
                np.array([[fd, 0, 0, 0]], np.int64),
                np.array([[fd, 0, 1, -1]], np.int64),
                np.array([[fd, 0, 129, 0]], np.int64),
                np.array([[fd, 2**63 - 1, 1, 0]], np.int64),
                np.array([[fd, 0, 16, 0], [fd, 32, 8, 8]], np.int64),
            ]
            for plan in malformed:
                with self.assertRaises(ValueError):
                    reader.read(plan, out)
            readonly = out.copy()
            readonly.flags.writeable = False
            for output in (readonly, out[::2], out.astype(np.uint16), out.reshape(2, 64)):
                with self.assertRaises(ValueError):
                    reader.read(valid, output)
            reader.read(np.empty((0, 4), np.int64), np.empty(0, np.uint8))
            with self.assertRaises(OSError) as exc:
                reader.read(np.array([[fd, path.stat().st_size - 1, 2, 0]], np.int64), out)
            self.assertEqual(exc.exception.errno, errno.EIO)
            bad = os.dup(fd)
            os.close(bad)
            with self.assertRaises(OSError) as exc:
                reader.read(np.array([[bad, 0, 1, 0]], np.int64), out)
            self.assertEqual(exc.exception.errno, errno.EBADF)
            # Error dispatch must drain all workers before exposing a partially written output.
            plan = np.array([[fd, 0, 1, i] for i in range(127)] + [[bad, 0, 1, 127]], np.int64)
            for _ in range(80):
                with self.assertRaises(OSError):
                    reader.read(plan, out)
                snapshot = out.copy()
                time.sleep(0.001)
                np.testing.assert_array_equal(out, snapshot)
        finally:
            os.close(fd)
            reader.close()
        with self.assertRaises(ValueError):
            reader.read(np.empty((0, 4), np.int64), np.empty(0, np.uint8))
        reader.close()
        for count in (0, 65):
            with self.assertRaises(ValueError):
                self.mod.Reader(count)

    def test_concurrent_gather_and_close(self):
        before = self.thread_baseline()
        fds_before = len(os.listdir("/proc/self/fd"))
        for workers in (16, 32, 64):
            table = NativeSSDTable(self.files, workers=workers)
            with ThreadPoolExecutor(8) as executor:
                ids = [np.random.default_rng(i).integers(0, self.starts[-1], 400) for i in range(64)]
                results = list(executor.map(table.gather, ids))
                for result, rows in zip(results, ids):
                    self.assert_rows(result, rows)
            real = table._native
            entered = threading.Event()
            proceed = threading.Event()
            closed = threading.Event()

            class Gate:
                def read(self, *args):
                    entered.set()
                    proceed.wait(5)
                    real.read(*args)

                def close(self):
                    real.close()

            table._native = Gate()  # test-only controllable accepted operation; ownership stays with lifetime
            with ThreadPoolExecutor(2) as executor:
                result = executor.submit(table.gather, [0, 17003])
                self.assertTrue(entered.wait(5))
                closer = executor.submit(lambda: (table.close(), closed.set()))
                time.sleep(0.02)
                self.assertFalse(closed.is_set())
                self.assertTrue(all(os.fstat(fd).st_size > 0 for fd in table._fds))
                proceed.set()
                self.assert_rows(result.result(timeout=5), [0, 17003])
                closer.result(timeout=5)
            self.assertEqual(table._fds, [])
        self.assertEqual(len(os.listdir("/proc/self/fd")), fds_before)
        self.assert_threads_released(before)

    def test_interrupted_close_wait_can_retry(self):
        table = NativeSSDTable(self.files, workers=1)
        owner = table._life
        original = owner.condition.wait_for

        def interrupt(predicate):
            if predicate():
                return True
            raise KeyboardInterrupt("controlled wait interruption")

        owner.active = 1
        owner.condition.wait_for = interrupt
        with self.assertRaises(KeyboardInterrupt):
            table.close()
        self.assertFalse(owner.closing)
        owner.condition.wait_for = original
        owner.active = 0
        table.close()
        wrapper = NativeReadAhead(NativeSSDTable(self.files, workers=1))
        original = wrapper._condition.wait_for
        wrapper._active = 1
        wrapper._condition.wait_for = interrupt
        with self.assertRaises(KeyboardInterrupt):
            wrapper.close()
        self.assertFalse(wrapper._closing)
        wrapper._condition.wait_for = original
        wrapper._active = 0
        wrapper.close()

    def test_failed_pool_close_retains_ownership_for_retry(self):
        table = NativeSSDTable(self.files, workers=16)
        real = table._life.pool

        class Flaky:
            failed = False

            def close(self):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("injected before-drain error")
                real.close()

        table._life.pool = Flaky()
        with self.assertRaises(RuntimeError):
            table.close()
        self.assertTrue(table._fds)
        self.assertFalse(table._life.closed)
        with self.assertRaises(ValueError):
            table.gather([0])
        table.close()
        self.assertEqual(table._fds, [])

    def test_native_fd_close_interruption_and_validation(self):
        table = NativeSSDTable(self.files, workers=16)
        owned = list(table._fds)
        release = table._life.release_fds
        calls = []

        def once(fds):
            calls.append(list(fds))
            if len(calls) == 1:
                raise KeyboardInterrupt("injected before native entry")
            release(fds)

        table._life.release_fds = once
        with self.assertRaises(KeyboardInterrupt):
            table.close()
        self.assertEqual(len(calls), 2)
        self.assertTrue(table._life.closed)
        self.assertEqual(table._fds, [])
        for fd in owned:
            with self.assertRaises(OSError):
                os.fstat(fd)
        table.close()
        fd = os.open(self.files[0][0], os.O_RDONLY)
        for owner in ([fd, fd], [-1], [fd, 2**40], [fd, "x"]):
            with self.assertRaises(ValueError):
                self.mod.close_fds(owner)
            self.assertGreater(os.fstat(fd).st_size, 0)
        owner = [fd]
        self.mod.close_fds(owner)
        self.assertEqual(owner, [])
        with self.assertRaises(OSError):
            os.fstat(fd)
        bad = os.open(self.files[0][0], os.O_RDONLY)
        os.close(bad)
        owner = [bad]
        with self.assertRaises(OSError):
            self.mod.close_fds(owner)
        self.assertEqual(owner, [])
        table = NativeSSDTable(self.files, workers=1)
        old = os.close

        def interrupted(fd):
            raise KeyboardInterrupt("before Python os.close syscall")

        os.close = interrupted
        try:
            table.close()
        finally:
            os.close = old
        self.assertEqual(table._fds, [])

    def test_wrapper_cleanup_retry_is_reachable(self):
        table = NativeSSDTable(self.files, workers=1)
        real = table._life.pool

        class Flaky:
            failed = False

            def close(self):
                if not self.failed:
                    self.failed = True
                    raise RuntimeError("before underlying drain")
                real.close()

        table._life.pool = Flaky()
        wrapper = NativeReadAhead(table)
        with self.assertRaises(RuntimeError):
            wrapper.close()
        self.assertTrue(table._fds)
        wrapper.close()
        self.assertEqual(table._fds, [])

    def test_read_ahead_bounded_ownership_and_errors(self):
        table = NativeReadAhead(NativeSSDTable(self.files, workers=16))
        try:
            for i in range(100):
                table.read_ahead(np.array([i, 17003], np.int64))
            self.assertLessEqual(len(table._ahead) + len(table._retired), 3)
            with ThreadPoolExecutor(8) as executor:
                rows = [np.array([i, 17003], np.int64) for i in range(40)]
                results = list(executor.map(table.gather, rows))
                for result, ids in zip(results, rows):
                    self.assert_rows(result, ids)
            for ids in (np.array([2**64 - 1], np.uint64), [-1]):
                with self.assertRaises(ValueError):
                    table.read_ahead(ids)
        finally:
            table.close()
        table.close()
        for operation in (table.gather, table.read_ahead):
            with self.assertRaises(ValueError):
                operation([0])
        entered = threading.Event()
        proceed = threading.Event()

        class Fake:
            rows = 100

            def gather(self, ids):
                entered.set()
                proceed.wait(5)
                return ids.copy()

            def close(self):
                self.closed = True

        fake = Fake()
        ahead = NativeReadAhead(fake)
        ids = np.array([3], np.int64)
        ahead.read_ahead(ids)
        self.assertTrue(entered.wait(5))
        for i in range(2000):
            ahead.read_ahead([i % 90 + 4])
        self.assertLessEqual(ahead._pool._work_queue.qsize(), 2)
        self.assertLessEqual(len(ahead._outstanding), 3)
        ids[0] = 9
        proceed.set()
        np.testing.assert_array_equal(ahead.gather([3]), [3])
        ahead.close()
        self.assertTrue(fake.closed)

        class Failing:
            rows = 100

            def gather(self, ids):
                raise OSError(errno.EIO, "oracle speculative failure")

            def close(self):
                self.closed = True

        fake = Failing()
        ahead = NativeReadAhead(fake)
        ahead.read_ahead([1])
        with self.assertRaises(OSError):
            ahead.close()
        self.assertTrue(fake.closed)

        class CleanupFailing(Failing):
            def close(self):
                self.closed = True
                raise RuntimeError("cleanup failure")

        fake = CleanupFailing()
        ahead = NativeReadAhead(fake)
        ahead.read_ahead([1])
        time.sleep(0.01)
        with self.assertRaises(OSError) as error:
            ahead.close()
        self.assertIsInstance(error.exception.__cause__, RuntimeError)
        self.assertTrue(fake.closed)


if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(NativeTests))
    print(
        json.dumps(
            {
                "ok": result.wasSuccessful(),
                "tests": result.testsRun,
                "failures": len(result.failures),
                "errors": len(result.errors),
            }
        )
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)

"""Owned persisted-file copy contracts; stdlib only, no tensor/native loader."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from snapshot_fd_transport import substitute_owners
from tensorfold.file_io import _buffered

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from tensorfold.engine.snapshot_file import private_snapshot  # noqa: E402


class Controls(unittest.TestCase):
    def setUp(self):
        substitute_owners(self)

    def fixture(self):
        temporary = tempfile.TemporaryDirectory(prefix="snapshot-source-test-")
        self.addCleanup(temporary.cleanup)
        path = Path(temporary.name) / "owned.safetensors"
        data = b"01234567" + bytes(range(251)) * 10000
        path.write_bytes(data)
        return path, data

    def test_complete_private_copy_hash_bound_and_lifetime(self):
        path, data = self.fixture()
        with private_snapshot(path, max_bytes=len(data)) as copy:
            self.assertEqual(copy.path.read_bytes(), data)
            self.assertEqual(copy.sha256, hashlib.sha256(data).hexdigest())
            self.assertEqual(copy.size, len(data))
            self.assertEqual(copy.path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(copy.path.parent.stat().st_mode & 0o777, 0o700)
            private = copy.path
            path.write_bytes(b"new original")
            self.assertEqual(private.read_bytes(), data)
        self.assertFalse(private.exists())
        self.assertFalse(private.parent.exists())

    def test_byte_budget_type_and_limit_refuse_before_temp_acquisition(self):
        path, data = self.fixture()
        for budget in (False, 7, len(data) - 1, 1.0):
            with patch(
                "tensorfold.engine.snapshot_file.tempfile.TemporaryDirectory",
                side_effect=AssertionError("private output allocated before admission"),
            ):
                with self.assertRaises(ValueError):
                    with private_snapshot(path, max_bytes=budget):
                        self.fail("consumer reached")

    def test_fifo_directory_symlink_refused_without_blocking(self):
        path, _ = self.fixture()
        fifo = path.with_name("fifo")
        os.mkfifo(fifo)
        link = path.with_name("link")
        link.symlink_to(path)
        for invalid in (fifo, path.parent, link):
            with self.assertRaises((ValueError, OSError)):
                with private_snapshot(invalid, max_bytes=1 << 24):
                    self.fail("consumer reached")

    def test_consumer_interruption_keeps_primary_and_removes_copy(self):
        path, data = self.fixture()
        primary = KeyboardInterrupt("caller interrupted")
        with self.assertRaises(KeyboardInterrupt) as caught:
            with private_snapshot(path, max_bytes=len(data)) as copy:
                private = copy.path
                raise primary
        self.assertIs(caught.exception, primary)
        self.assertFalse(private.parent.exists())

    def test_malformed_notes_never_replace_primary_and_both_cleanup_failures_remain_visible(self):
        path, data = self.fixture()
        primary, cleanup = KeyboardInterrupt("consumer interrupted"), OSError("private directory cleanup failed")
        primary.__notes__ = 123
        original = tempfile.TemporaryDirectory.cleanup

        def failed_directory(directory):
            original(directory)
            raise cleanup

        with patch("tensorfold.engine.snapshot_file.tempfile.TemporaryDirectory.cleanup", failed_directory):
            with self.assertRaises(KeyboardInterrupt) as caught:
                with private_snapshot(path, max_bytes=len(data)) as copied:
                    private = copied.path
                    raise primary
        self.assertIs(caught.exception, primary)
        self.assertFalse(private.exists())
        self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
        self.assertIs(primary.__cause__.exceptions[0], cleanup)
        self.assertIsInstance(primary.__cause__.exceptions[1], TypeError)

    def test_copy_read_primary_survives_real_close_failure_and_all_owners_drain(self):
        path, data = self.fixture()

        class Primary(KeyboardInterrupt):
            def __str__(self):
                raise AssertionError("foreign primary str called")

            def add_note(self, note):
                raise AssertionError("foreign primary method called")

        primary, secondary = Primary(), OSError("source close failed")
        original = _buffered
        resources = []

        class Stream:
            def __init__(self, stream, reader):
                self.stream, self.reader = stream, reader
                self.closes = 0

            def read(self, count):
                raise primary

            def close(self):
                self.closes += 1
                self.stream.close()
                if self.reader:
                    raise secondary

            def __getattr__(self, name):
                return getattr(self.stream, name)

        def fdopen(fd, mode):
            wrapper = Stream(original(fd, mode), mode == "rb")
            resources.append(wrapper)
            return wrapper

        with patch("tensorfold.file_io._buffered", fdopen):
            with self.assertRaises(Primary) as caught:
                with private_snapshot(path, max_bytes=len(data)):
                    self.fail("consumer reached")
        self.assertIs(caught.exception, primary)
        self.assertIs(primary.__cause__, secondary)
        self.assertTrue(any("OSError" in n for n in primary.__notes__))
        self.assertEqual(len(resources), 2)
        self.assertTrue(all(r.stream.closed for r in resources))
        self.assertTrue(all(r.closes == 1 for r in resources))

    def test_descriptor_transfer_failure_closes_original_owned_fd(self):
        path, data = self.fixture()
        original_open = os.open
        descriptors = []

        def opened(*args):
            fd = original_open(*args)
            descriptors.append(fd)
            return fd

        primary = OSError("fdopen failed before ownership transfer")
        with (
            patch("tensorfold.engine.snapshot_file.os.open", opened),
            patch("tensorfold.file_io._buffered", side_effect=primary),
        ):
            with self.assertRaises(OSError) as caught:
                with private_snapshot(path, max_bytes=len(data)):
                    self.fail("consumer reached")
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])

    def test_source_mutation_during_copy_and_short_write_refuse(self):
        for bad in ("mutate", "short"):
            path, data = self.fixture()
            original = _buffered
            resources = []

            class Stream:
                def __init__(self, stream, reader):
                    self.stream, self.reader = stream, reader
                    self.first = True

                def read(self, count):
                    chunk = self.stream.read(count)
                    if self.reader and self.first and bad == "mutate":
                        self.first = False
                        with path.open("ab") as mutation:
                            mutation.write(b"extra")
                    return chunk

                def write(self, chunk):
                    if bad == "short":
                        return self.stream.write(chunk[:1])
                    return self.stream.write(chunk)

                def __getattr__(self, name):
                    return getattr(self.stream, name)

            def fdopen(fd, mode):
                wrapper = Stream(original(fd, mode), mode == "rb")
                resources.append(wrapper)
                return wrapper

            with patch("tensorfold.file_io._buffered", fdopen):
                with self.assertRaises((ValueError, OSError)):
                    with private_snapshot(path, max_bytes=len(data)):
                        self.fail("consumer reached")
            self.assertTrue(all(r.stream.closed for r in resources))


if __name__ == "__main__":
    unittest.main()

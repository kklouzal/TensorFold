"""Core regular-stream contract with labeled Python ownership transport."""

import os
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snapshot_fd_transport import TransportOwner, substitute_owners
from tensorfold.file_io import FileStreams


class CoreControls(unittest.TestCase):
    def setUp(self):
        substitute_owners(self)

    def test_budget_and_minimum_checks_precede_native_acquisition(self):
        for count in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                FileStreams(max_files=count)
        scope = FileStreams(max_files=1)
        with patch("tensorfold.file_io._owned_slot", side_effect=AssertionError("acquired")):
            for minimum, maximum in ((True, None), (-1, None), (1.5, None), (8, 7), (0, True)):
                with self.assertRaises(ValueError):
                    scope.open("unused", os.O_RDONLY, "rb", min_bytes=minimum, max_bytes=maximum)
        self.assertTrue(scope.retired)

    def test_owner_only_preserves_incremental_reads_without_io_allocations(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "chunks"
            contents = bytes(range(251)) * 100
            path.write_bytes(contents)
            scope = FileStreams(max_files=1)
            with (
                patch("tensorfold.file_io.io.FileIO", side_effect=AssertionError("raw allocated")),
                patch("tensorfold.file_io._buffered", side_effect=AssertionError("buffer allocated")),
            ):
                record, before = scope.open_descriptor(path, os.O_RDONLY, max_bytes=len(contents))
                self.assertIsNone(record.raw)
                self.assertIsNone(record.stream)
                self.assertEqual(before.st_size, len(contents))
                digest, observed = hashlib.sha256(), 0
                try:
                    while block := os.read(record.owner.fileno(), 97):
                        digest.update(block)
                        observed += len(block)
                    self.assertEqual(observed, len(contents))
                    self.assertEqual(digest.hexdigest(), hashlib.sha256(contents).hexdigest())
                finally:
                    self.assertEqual(scope.drain(), [])
            self.assertTrue(scope.retired)

    def test_closed_slot_is_journaled_before_actual_acquisition_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "owned"
            path.write_bytes(b"actual descriptor")
            scope = FileStreams(max_files=1)
            primary, opened = KeyboardInterrupt("after acquired open"), []
            control = self

            class Slot(TransportOwner):
                def open(self, *args, **kwargs):
                    control.assertEqual(len(scope._records), 1)
                    control.assertIs(scope._records[0].owner, self)
                    control.assertTrue(self.closed)
                    super().open(*args, **kwargs)
                    opened.append(self.fileno())
                    raise primary

            with patch("tensorfold.file_io._owned_slot", Slot):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    scope.open_descriptor(path, os.O_RDONLY)
            self.assertIs(caught.exception, primary)
            self.assertEqual(len(scope.live_owners), 1)
            self.assertEqual(os.fstat(opened[0]).st_size, path.stat().st_size)
            self.assertEqual(scope.drain(), [])
            self.assertTrue(scope.retired)
            with self.assertRaises(OSError):
                os.fstat(opened[0])

    def test_owner_only_empty_file_and_failed_admission_remain_bounded(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "empty"
            path.touch()
            scope = FileStreams(max_files=1)
            record, _ = scope.open_descriptor(path, os.O_RDONLY, min_bytes=0, max_bytes=0)
            self.assertEqual(os.read(record.owner.fileno(), 1), b"")
            self.assertEqual(scope.close(record), [])
            with patch("tensorfold.file_io._owned_slot", side_effect=AssertionError("second acquisition")):
                with self.assertRaises(RuntimeError):
                    scope.open_descriptor(path, os.O_RDONLY)
            other = FileStreams(max_files=1)
            try:
                with self.assertRaises(ValueError):
                    other.open_descriptor(path, os.O_RDONLY, min_bytes=1)
                self.assertEqual(len(other.live_owners), 1)
                self.assertIsNone(other._records[0].raw)
                self.assertIsNone(other._records[0].stream)
            finally:
                self.assertEqual(other.drain(), [])

    def test_empty_regular_file_and_borrowed_reader_lifetime(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "empty"
            path.touch()
            scope = FileStreams(max_files=1)
            record, info = scope.open(path, os.O_RDONLY, "rb", max_bytes=0)
            self.assertEqual(info.st_size, 0)
            self.assertEqual(record.stream.read(1), b"")
            self.assertFalse(scope.retired)
            self.assertEqual(scope.live_owners, (record.owner,))
            self.assertEqual(scope.close(record), [])
            self.assertTrue(scope.retired)
            self.assertEqual(scope.live_owners, ())
            self.assertEqual(scope.drain(), [])

    def test_failed_geometry_open_remains_owned_until_caller_drains(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "small"
            path.write_bytes(b"one")
            scope = FileStreams(max_files=1)
            with self.assertRaises(ValueError):
                scope.open(path, os.O_RDONLY, "rb", min_bytes=4)
            self.assertFalse(scope.retired)
            (owner,) = scope.live_owners
            self.assertEqual(os.fstat(owner.fileno()).st_size, 3)
            self.assertEqual(scope.drain(), [])
            self.assertTrue(scope.retired)

    def test_other_scope_record_is_rejected_without_retiring_its_owner(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "owned"
            path.write_bytes(b"data")
            first, other = FileStreams(max_files=1), FileStreams(max_files=1)
            record, _ = first.open(path, os.O_RDONLY, "rb")
            try:
                with self.assertRaises(ValueError):
                    other.close(record)
                self.assertFalse(first.retired)
                self.assertFalse(record.owner.closed)
            finally:
                self.assertEqual(first.drain(), [])

    def test_interrupted_record_retirement_does_not_skip_other_owned_records(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / "owned"
            path.write_bytes(b"data")
            scope = FileStreams(max_files=2)
            first, _ = scope.open(path, os.O_RDONLY, "rb")
            second, _ = scope.open(path, os.O_RDONLY, "rb")
            interrupted = KeyboardInterrupt("retirement control interrupted")
            original = scope._drain

            def fault(record):
                if record is second:
                    raise interrupted
                return original(record)

            try:
                with patch.object(scope, "_drain", side_effect=fault):
                    self.assertEqual(scope.drain(), [interrupted])
                self.assertTrue(first.done)
                self.assertFalse(second.done)
                self.assertFalse(second.owner.closed)
                self.assertEqual(scope.live_owners, (second.owner,))
            finally:
                self.assertEqual(scope.drain(), [])


if __name__ == "__main__":
    unittest.main()

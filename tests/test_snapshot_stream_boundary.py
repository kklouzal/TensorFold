"""Borrowed IO/cleanup journal controls using labeled FD transport substitutes.

These real FileIO/buffer/descriptor schedules do not prove the native owner's
atomic acquisition/consumption. ROOT qualifies that independent extension.
"""

import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from snapshot_fd_transport import TransportOwner, substitute_owners
from tensorfold.engine.snapshot_stream import SnapshotStreams
from tensorfold.file_io import _buffered


class StreamControls(unittest.TestCase):
    def setUp(self):
        substitute_owners(self)
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "owned"
        self.path.write_bytes(b"01234567")

    def test_borrowed_stream_retirement_leaves_native_owner_for_explicit_close(self):
        scope = SnapshotStreams()
        record, _ = scope.open(self.path, os.O_RDONLY, "rb", max_bytes=8)
        descriptor = record.owner.fileno()
        record.stream.close()
        self.assertTrue(record.raw.closed)
        self.assertFalse(record.owner.closed)
        self.assertEqual(os.fstat(descriptor).st_size, 8)
        scope.finish()
        self.assertTrue(record.owner.closed)
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_three_fd_budget_and_input_validation_precede_acquisition(self):
        scope = SnapshotStreams()
        with patch("tensorfold.file_io._owned_slot", side_effect=AssertionError("acquired")):
            for budget in (True, 7, 8.0):
                with self.assertRaises(ValueError):
                    scope.open(self.path, os.O_RDONLY, "rb", max_bytes=budget)
        for _ in range(3):
            scope.open(self.path, os.O_RDONLY, "rb", max_bytes=8)
        try:
            with patch("tensorfold.file_io._owned_slot", side_effect=AssertionError("fourth")):
                with self.assertRaises(RuntimeError):
                    scope.open(self.path, os.O_RDONLY, "rb", max_bytes=8)
        finally:
            scope.finish()

    def test_unentered_close_retries_exact_owner_then_preserves_initial_error(self):
        interrupted = KeyboardInterrupt("before native close")
        calls = []

        class Owner(TransportOwner):
            def close(self):
                calls.append(self)
                if len(calls) == 1:
                    raise interrupted
                super().close()

        scope = SnapshotStreams()
        with patch("tensorfold.file_io._owned_slot", Owner):
            record, _ = scope.open(self.path, os.O_RDONLY, "rb", max_bytes=8)
        with self.assertRaises(KeyboardInterrupt) as caught:
            scope.finish()
        self.assertIs(caught.exception, interrupted)
        self.assertEqual(calls, [record.owner, record.owner])
        self.assertTrue(record.raw.closed)
        self.assertTrue(record.owner.closed)

    def test_consumed_close_failure_never_retries_even_with_fd_number_reuse(self):
        consumed = OSError("consumed close")
        calls, replacements = [], []

        class Owner(TransportOwner):
            def open(self, path, flags, mode=0o600):
                self.path = path
                super().open(path, flags, mode)

            def close(self):
                calls.append(self)
                descriptor = self.fileno()
                super().close()
                replacement = os.open(self.path, os.O_RDONLY)
                replacements.append(replacement)
                self.descriptor_reused = replacement == descriptor
                raise consumed

        # The substitute stores its selected fixture path; the native type has
        # no mutable attributes and is independently tested by ROOT.
        scope = SnapshotStreams()
        with patch("tensorfold.file_io._owned_slot", Owner):
            record, _ = scope.open(self.path, os.O_RDONLY, "rb", max_bytes=8)
        try:
            with self.assertRaises(OSError) as caught:
                scope.finish()
            self.assertIs(caught.exception, consumed)
            self.assertEqual(calls, [record.owner])
            self.assertTrue(record.owner.descriptor_reused)
            self.assertEqual(os.fstat(replacements[0]).st_size, 8)
        finally:
            for descriptor in replacements:
                os.close(descriptor)

    def test_two_unretired_close_failures_keep_journal_and_exact_opaque_primary(self):
        class Primary(KeyboardInterrupt):
            def __getattribute__(self, name):
                if name in ("__dict__", "__cause__", "__notes__"):
                    raise AssertionError("foreign exception hook")
                return super().__getattribute__(name)

        primary = Primary("caller")
        primary.__notes__ = 123
        previous = ValueError("earlier")
        primary.__cause__ = previous
        failures = [OSError("first pre-entry"), OSError("second pre-entry")]
        blocked = True

        class Owner(TransportOwner):
            def close(self):
                if blocked:
                    raise failures.pop(0)
                super().close()

        scope = SnapshotStreams()
        with patch("tensorfold.file_io._owned_slot", Owner):
            record, _ = scope.open(self.path, os.O_RDONLY, "rb", max_bytes=8)
        try:
            with self.assertRaises(Primary) as caught:
                scope.finish(primary)
            self.assertIs(caught.exception, primary)
            storage = BaseException.__dict__["__dict__"].__get__(primary, type(primary))
            self.assertEqual(storage["_tensorfold_snapshot_fd_owners"], (record.owner,))
            self.assertIs(storage["_tensorfold_snapshot_fd_scope"], scope)
            self.assertTrue(record.raw.closed)
            self.assertFalse(record.owner.closed)
            self.assertEqual(os.fstat(record.owner.fileno()).st_size, 8)
            cause = BaseException.__cause__.__get__(primary, type(primary))
            self.assertIs(cause.exceptions[0], previous)
            self.assertEqual(len(cause.exceptions), 4)  # two close errors + annotation failure
        finally:
            blocked = False
            scope.finish()

    def test_foreign_close_cannot_erase_caller_native_cause_or_context(self):
        primary, cause, context, cleanup = KeyboardInterrupt(), ValueError(), LookupError(), OSError()
        primary.__cause__, primary.__context__ = cause, context

        class Slot(TransportOwner):
            def close(self):
                super().close()
                primary.__cause__ = primary.__context__ = None
                raise cleanup

        scope = SnapshotStreams()
        with patch("tensorfold.file_io._owned_slot", Slot):
            record, _ = scope.open(self.path, os.O_RDONLY, "rb", max_bytes=8)
        with self.assertRaises(KeyboardInterrupt) as caught:
            scope.finish(primary)
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__cause__.exceptions, (cause, context, cleanup))
        self.assertTrue(record.owner.closed)
        self.assertTrue(record.done)

    def test_failed_pending_buffer_and_raw_close_retain_owner_until_views_retire(self):
        blocked = True
        original_raw = io.FileIO
        pending, close_errors = [], []

        class Raw(io.RawIOBase):
            def __init__(self, *args, **kwargs):
                self.borrowed = original_raw(*args, **kwargs)

            def writable(self):
                return True

            def write(self, data):
                return self.borrowed.write(data)

            def close(self):
                if blocked:
                    error = KeyboardInterrupt("raw close not entered")
                    close_errors.append(error)
                    raise error
                self.borrowed.close()
                super().close()

        class Buffer:
            def __init__(self, raw, mode):
                self.buffer = _buffered(raw, mode)
                pending.append(self.buffer)

            def write(self, data):
                return self.buffer.write(data)

            def close(self):
                if blocked:
                    raise OSError("buffer flush not entered")
                self.buffer.close()

        scope = SnapshotStreams()
        with (
            patch("tensorfold.file_io.io.FileIO", Raw),
            patch("tensorfold.file_io._buffered", Buffer),
        ):
            record, _ = scope.open(self.path, os.O_WRONLY, "wb")
        record.stream.write(b"changed!")
        try:
            with self.assertRaises(OSError) as caught:
                scope.finish()
            self.assertEqual(len(close_errors), 2)
            self.assertFalse(record.raw.closed)
            self.assertFalse(record.owner.closed)
            self.assertEqual(self.path.read_bytes(), b"01234567")
            self.assertIs(caught.exception.__dict__["_tensorfold_snapshot_fd_scope"], scope)
        finally:
            blocked = False
            scope.finish()
        self.assertTrue(record.raw.closed)
        self.assertTrue(pending[0].closed)
        self.assertTrue(record.owner.closed)
        self.assertEqual(self.path.read_bytes(), b"changed!")


if __name__ == "__main__":
    unittest.main()

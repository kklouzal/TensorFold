"""Owned header-only indexing and borrowed-stream schema checks, no SDK."""

import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from snapshot_fd_transport import substitute_owners
from tensorfold.file_io import _buffered

from tensorfold.cuda.tensor_file import read_header, read_header_stream
from tensorfold.engine.snapshot_file import snapshot_header


class HeaderControls(unittest.TestCase):
    def setUp(self):
        substitute_owners(self)

    def test_raw_fd_same_primary_cleanup_does_not_create_self_cause(self):
        with tempfile.TemporaryDirectory() as root:
            path, _, _ = self.fixture(root)
            primary = KeyboardInterrupt("same interruption")
            original = os.close
            closed = []

            def close(fd):
                original(fd)
                closed.append(fd)
                raise primary

            with (
                patch("tensorfold.file_io._buffered", side_effect=primary),
                patch("tensorfold.engine.snapshot_file.os.close", side_effect=close),
            ):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    snapshot_header(path, max_bytes=2 << 20, sizes={"U8": 1})
            self.assertIs(caught.exception, primary)
            self.assertIsNot(primary.__cause__, primary)
            self.assertEqual(len(closed), 1)
            with self.assertRaises(OSError):
                os.fstat(closed[0])

    def test_shared_owning_header_close_preserves_malformed_note_primary(self):
        with tempfile.TemporaryDirectory() as root:
            path, _, _ = self.fixture(root)
            primary, cleanup = KeyboardInterrupt("owned header interrupted"), OSError("owned stream close failed")
            primary.__notes__ = 123
            streams = []

            class BrokenStream:
                def __init__(self, stream):
                    self.stream = stream

                def fileno(self):
                    return self.stream.fileno()

                def read(self, count):
                    raise primary

                def tell(self):
                    return self.stream.tell()

                def close(self):
                    self.stream.close()
                    raise cleanup

            def broken_buffer(raw, mode):
                stream = _buffered(raw, mode)
                streams.append(stream)
                return BrokenStream(stream)

            with patch("tensorfold.file_io._buffered", side_effect=broken_buffer):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    read_header(path, {"U8": 1})
            self.assertIs(caught.exception, primary)
            self.assertEqual(len(streams), 1)
            self.assertTrue(streams[0].closed)
            self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
            self.assertIs(primary.__cause__.exceptions[0], cleanup)
            self.assertIsInstance(primary.__cause__.exceptions[1], TypeError)

    def fixture(self, root):
        path = Path(root) / "cache.safetensors"
        header = {
            "__metadata__": {"format": "2"},
            "data": {"dtype": "U8", "shape": [1 << 20], "data_offsets": [0, 1 << 20]},
        }
        raw = json.dumps(header).encode()
        with path.open("wb") as stream:
            stream.write(struct.pack("<Q", len(raw)) + raw)
            stream.truncate(8 + len(raw) + (1 << 20))
        return path, header, len(raw)

    def test_borrowed_stream_same_complete_schema_without_reopen_or_close(self):
        with tempfile.TemporaryDirectory() as root:
            path, expected, count = self.fixture(root)
            owned = read_header(path, {"U8": 1})
            with path.open("rb") as stream:
                with patch("tensorfold.cuda.tensor_file._regular_stream", side_effect=AssertionError("reopened")):
                    borrowed = read_header_stream(stream, {"U8": 1}, label=path)
                self.assertFalse(stream.closed)
                self.assertEqual(stream.tell(), 8 + count)
                self.assertEqual(stream.read(1), b"\0")
            self.assertEqual(owned, borrowed)
            self.assertEqual(borrowed[1], expected)

    def test_index_reads_header_only_without_temporary_payload_copy(self):
        with tempfile.TemporaryDirectory() as root:
            path, expected, count = self.fixture(root)
            reads, streams = [], []
            original = _buffered

            class Stream:
                def __init__(self, stream):
                    self.stream = stream

                def read(self, size):
                    reads.append(size)
                    return self.stream.read(size)

                def __getattr__(self, name):
                    return getattr(self.stream, name)

            def fdopen(*args):
                result = Stream(original(*args))
                streams.append(result)
                return result

            with (
                patch("tensorfold.file_io._buffered", fdopen),
                patch(
                    "tensorfold.engine.snapshot_file.tempfile.TemporaryDirectory",
                    side_effect=AssertionError("payload copied"),
                ),
            ):
                header, identity = snapshot_header(path, max_bytes=path.stat().st_size, sizes={"U8": 1})
            self.assertEqual(header, expected)
            self.assertEqual(reads, [8, count])
            self.assertEqual(identity[2], path.stat().st_size)
            self.assertTrue(all(stream.stream.closed for stream in streams))

    def test_nonzero_borrowed_position_refuses_without_closing_or_rewinding(self):
        with tempfile.TemporaryDirectory() as root:
            path, _, _ = self.fixture(root)
            with path.open("rb") as stream:
                stream.seek(1)
                with self.assertRaises(ValueError):
                    read_header_stream(stream, {"U8": 1}, label=path)
                self.assertEqual(stream.tell(), 1)
                self.assertFalse(stream.closed)

    def test_owned_index_budget_symlink_fifo_and_directory_refuse_without_temp(self):
        with tempfile.TemporaryDirectory() as root:
            path, _, _ = self.fixture(root)
            alias, fifo = Path(root) / "alias", Path(root) / "fifo"
            alias.symlink_to(path)
            os.mkfifo(fifo)
            for invalid in (alias, fifo, Path(root)):
                with self.subTest(path=invalid), self.assertRaises((OSError, ValueError)):
                    snapshot_header(invalid, max_bytes=1 << 22, sizes={"U8": 1})
            with self.assertRaises(ValueError):
                snapshot_header(path, max_bytes=path.stat().st_size - 1, sizes={"U8": 1})

    def test_header_read_primary_survives_physically_closed_descriptor_failure(self):
        with tempfile.TemporaryDirectory() as root:
            path, _, _ = self.fixture(root)

            class Primary(KeyboardInterrupt):
                def __str__(self):
                    raise AssertionError("foreign primary formatting")

                def add_note(self, value):
                    raise AssertionError("foreign primary annotation")

            primary, cleanup = Primary(), OSError("close also failed")
            original, streams = _buffered, []

            class Stream:
                def __init__(self, stream):
                    self.stream = stream
                    self.closes = 0

                def read(self, size):
                    raise primary

                def close(self):
                    self.closes += 1
                    self.stream.close()
                    raise cleanup

                def __getattr__(self, name):
                    return getattr(self.stream, name)

            def fdopen(*args):
                value = Stream(original(*args))
                streams.append(value)
                return value

            with patch("tensorfold.file_io._buffered", fdopen), self.assertRaises(Primary) as caught:
                snapshot_header(path, max_bytes=path.stat().st_size, sizes={"U8": 1})
            self.assertIs(caught.exception, primary)
            self.assertIs(primary.__cause__, cleanup)
            self.assertEqual(streams[0].closes, 1)
            self.assertTrue(streams[0].stream.closed)

    def test_malformed_notes_preserve_header_primary_and_close_annotation_statuses(self):
        with tempfile.TemporaryDirectory() as root:
            path, _, _ = self.fixture(root)
            primary, cleanup = KeyboardInterrupt("header interrupted"), OSError("close failed")
            primary.__notes__ = 123
            original = _buffered
            streams = []

            class Stream:
                def __init__(self, stream):
                    self.stream = stream

                def read(self, size):
                    raise primary

                def close(self):
                    self.stream.close()
                    raise cleanup

                def __getattr__(self, name):
                    return getattr(self.stream, name)

            def fdopen(*args):
                result = Stream(original(*args))
                streams.append(result)
                return result

            with (
                patch("tensorfold.file_io._buffered", fdopen),
                self.assertRaises(KeyboardInterrupt) as caught,
            ):
                snapshot_header(path, max_bytes=path.stat().st_size, sizes={"U8": 1})
            self.assertIs(caught.exception, primary)
            self.assertTrue(streams[0].stream.closed)
            self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
            self.assertIs(primary.__cause__.exceptions[0], cleanup)
            self.assertIsInstance(primary.__cause__.exceptions[1], TypeError)


if __name__ == "__main__":
    unittest.main()

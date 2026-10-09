"""Actual bounded file bytes and independent masked SHA oracle, no SDK."""

import hashlib
import json
import os
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from snapshot_fd_transport import substitute_owners
from tensorfold.file_io import _buffered

from tensorfold.engine.snapshot_integrity import (
    FIELD,
    ZERO,
    DigestPlan,
    digest_location,
    seal_snapshot,
    verify_snapshot,
)
from tensorfold.engine.snapshot_file import private_snapshot


class IntegrityControls(unittest.TestCase):
    def setUp(self):
        substitute_owners(self)

    def fixture(self, root, *, payload=b"\0\0\x80\x3f", extra=None, header_override=None, dtype="U8"):
        metadata = {"format": "2", "model": "owned", "tokens": "[1]", "layers": "[]", FIELD: ZERO}
        if extra:
            metadata.update(extra)
        header = {
            "data": {
                "dtype": dtype,
                "shape": [len(payload) // (4 if dtype == "F32" else 1)],
                "data_offsets": [0, len(payload)],
            },
            "__metadata__": metadata,
        }
        raw = json.dumps(header, ensure_ascii=False).encode() if header_override is None else header_override(header)
        path = Path(root) / "snapshot.safetensors"
        path.write_bytes(struct.pack("<Q", len(raw)) + raw + payload)
        return path, header

    def test_seal_exact_mask_oracle_and_single_copy_digest(self):
        with tempfile.TemporaryDirectory() as root:
            path, header = self.fixture(root)
            initial = path.read_bytes()
            raw = initial[8 : 8 + struct.unpack("<Q", initial[:8])[0]]
            _, offset = digest_location(raw, header)
            expected = hashlib.sha256(initial[: offset + 8] + b"0" * 64 + initial[offset + 72 :]).hexdigest()
            identity, digest = seal_snapshot(path, sizes={"U8": 1}, max_bytes=1 << 20)
            self.assertEqual(digest, expected)
            self.assertEqual(verify_snapshot(path, sizes={"U8": 1}, max_bytes=1 << 20), (identity, expected))
            with private_snapshot(path, max_bytes=1 << 20, integrity_sizes={"U8": 1}, hash_copy=False) as copy:
                self.assertIsNone(copy.sha256)
                self.assertEqual(copy.integrity_sha256, expected)
                self.assertEqual(copy.path.read_bytes(), path.read_bytes())
                private = copy.path
            self.assertFalse(private.exists())

    def test_typed_finite_float_word_corruption_is_refused_before_consumer(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.fixture(root, dtype="F32")
            seal_snapshot(path, sizes={"F32": 4}, max_bytes=1 << 20)
            raw = bytearray(path.read_bytes())
            raw[-1] = 0x40  # F32 1.0 -> 4.0, still finite and valid geometry
            self.assertEqual(struct.unpack("<f", path.read_bytes()[-4:])[0], 1.0)
            self.assertEqual(struct.unpack("<f", raw[-4:])[0], 4.0)
            path.write_bytes(raw)
            with self.assertRaisesRegex(ValueError, "digest differs"):
                with private_snapshot(path, max_bytes=1 << 20, integrity_sizes={"F32": 4}, hash_copy=False):
                    self.fail("native consumer reached corrupt typed data")

    def test_metadata_corruption_and_digest_corruption_are_refused(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.fixture(root)
            seal_snapshot(path, sizes={"U8": 1}, max_bytes=1 << 20)
            original = path.read_bytes()
            for changed in (
                original.replace(b"owned", b"other", 1),
                original.replace(b'"tokens": "[1]"', b'"tokens": "[2]"', 1),
            ):
                path.write_bytes(changed)
                with self.assertRaises(ValueError):
                    verify_snapshot(path, sizes={"U8": 1}, max_bytes=1 << 20)
            path.write_bytes(original)
            raw = bytearray(original)
            start = original.index(FIELD.encode()) + len(FIELD) + 4
            raw[start] = ord("a") if raw[start] != ord("a") else ord("b")
            path.write_bytes(raw)
            with self.assertRaises(ValueError):
                verify_snapshot(path, sizes={"U8": 1}, max_bytes=1 << 20)

    def test_mask_scopes_metadata_not_nested_free_text_and_handles_escapes(self):
        with tempfile.TemporaryDirectory() as root:
            path, header = self.fixture(
                root, extra={"opaque": json.dumps({FIELD: ZERO, "x": 'escaped " quote \\ path'})}
            )
            before = path.read_bytes()
            seal_snapshot(path, sizes={"U8": 1}, max_bytes=1 << 20)
            after = path.read_bytes()
            self.assertEqual(
                sum(a != b for a, b in zip(before, after)),
                sum(c != "0" for c in verify_snapshot(path, sizes={"U8": 1}, max_bytes=1 << 20)[1]),
            )
            self.assertIn(json.dumps(header["__metadata__"]["opaque"]).encode(), after)
            verify_snapshot(path, sizes={"U8": 1}, max_bytes=1 << 20)

    def test_mask_chunk_boundaries_match_independent_whole_file_oracle(self):
        raw = bytes(range(251)) * 15
        for offset in (0, 1, 63, 64, 65, 127, 128, len(raw) - 64):
            expected = hashlib.sha256(raw[:offset] + b"0" * 64 + raw[offset + 64 :]).hexdigest()
            for size in (1, 7, 63, 64, 65, 128, 1024):
                plan = DigestPlan(expected, offset)
                digest = hashlib.sha256()
                for position in range(0, len(raw), size):
                    plan.update(digest, raw[position : position + size], position)
                plan.validate(digest)

    def test_budget_nonregular_missing_or_escaped_digest_refuse(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.fixture(root)
            with self.assertRaises(ValueError):
                seal_snapshot(path, sizes={"U8": 1}, max_bytes=8)
            fifo = Path(root) / "fifo"
            os.mkfifo(fifo)
            with self.assertRaises(ValueError):
                verify_snapshot(fifo, sizes={"U8": 1}, max_bytes=1 << 20)
            for escaped in (False, True):
                path, header = self.fixture(root)
                raw = path.read_bytes()
                if escaped:
                    body = raw[8:].replace(FIELD.encode(), b"\\u0074ensorfold_sha256")
                else:
                    header["__metadata__"].pop(FIELD)
                    body = json.dumps(header).encode() + raw[-4:]
                count = len(body) - 4
                path.write_bytes(struct.pack("<Q", count) + body)
                with self.assertRaises(ValueError):
                    verify_snapshot(path, sizes={"U8": 1}, max_bytes=1 << 20)

    def test_same_primary_close_preserved_without_self_cause(self):
        with tempfile.TemporaryDirectory() as root:
            path, _ = self.fixture(root)
            primary = KeyboardInterrupt("read+close")
            original = _buffered

            class Broken:
                def __init__(self, stream):
                    self.stream = stream

                def fileno(self):
                    return self.stream.fileno()

                def tell(self):
                    return self.stream.tell()

                def read(self, count):
                    raise primary

                def close(self):
                    self.stream.close()
                    raise primary

            with patch(
                "tensorfold.file_io._buffered",
                side_effect=lambda fd, mode: Broken(original(fd, mode)),
            ):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    verify_snapshot(path, sizes={"U8": 1}, max_bytes=1 << 20)
            self.assertIs(caught.exception, primary)
            self.assertIsNot(primary.__cause__, primary)


if __name__ == "__main__":
    unittest.main()

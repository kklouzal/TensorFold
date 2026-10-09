"""ROOT-only common file stream controls with the actual protocol2 owner.

No numerical SDK is needed. Source checks never import this module locally;
ROOT supplies the compiled owner and records its exact source/binary receipt.
"""

import errno
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest

from tensorfold._fd_owner import OwnedFD
from tensorfold import file_io


class NativeFileControls(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-file-stream-")
        self.addCleanup(self.temporary.cleanup)
        self.path = Path(self.temporary.name) / "input"
        self.payload = bytes(range(251)) * 4300
        self.path.write_bytes(self.payload)

    def assert_closed(self, descriptor):
        with self.assertRaises(OSError) as caught:
            os.fstat(descriptor)
        self.assertEqual(caught.exception.errno, errno.EBADF)

    def test_owner_only_incremental_hash_and_empty_file(self):
        for payload in (self.payload, b""):
            with self.subTest(bytes=len(payload)):
                self.path.write_bytes(payload)
                scope = file_io.FileStreams(max_files=1)
                record, before = scope.open_descriptor(self.path, os.O_RDONLY, max_bytes=len(payload))
                descriptor = record.owner.fileno()
                try:
                    digest, consumed = hashlib.sha256(), 0
                    while chunk := os.read(descriptor, 65537):
                        digest.update(chunk)
                        consumed += len(chunk)
                    self.assertEqual(before.st_size, consumed)
                    self.assertEqual(digest.hexdigest(), hashlib.sha256(payload).hexdigest())
                    self.assertIsNone(record.raw)
                    self.assertIsNone(record.stream)
                finally:
                    self.assertEqual(scope.drain(), [])
                self.assertTrue(scope.retired)
                self.assert_closed(descriptor)

    def test_buffered_write_retires_views_before_actual_owner(self):
        scope = file_io.FileStreams(max_files=1)
        record, _ = scope.open(self.path, os.O_WRONLY | os.O_TRUNC, "wb")
        descriptor = record.owner.fileno()
        record.stream.write(b"pending borrowed bytes")
        self.assertEqual(self.path.read_bytes(), b"")
        self.assertEqual(scope.drain(), [])
        self.assertTrue(record.stream.closed)
        self.assertTrue(record.raw.closed)
        self.assertTrue(record.owner.closed)
        self.assertEqual(self.path.read_bytes(), b"pending borrowed bytes")
        self.assert_closed(descriptor)

    def test_post_open_native_return_interruption_preserves_journaled_slot(self):
        scope = file_io.FileStreams(max_files=1)
        primary, owners, descriptors = KeyboardInterrupt("after common native open"), [], []
        previous = sys.getprofile()

        def profile(frame, event, function):
            if (
                frame.f_code is file_io.FileStreams.open_descriptor.__code__
                and event == "c_return"
                and getattr(function, "__name__", None) == "open"
                and isinstance(getattr(function, "__self__", None), OwnedFD)
            ):
                owner = function.__self__
                self.assertIs(scope._records[-1].owner, owner)
                owners.append(owner)
                descriptors.append(owner.fileno())
                raise primary

        try:
            sys.setprofile(profile)
            with self.assertRaises(KeyboardInterrupt) as caught:
                scope.open_descriptor(self.path, os.O_RDONLY)
            self.assertIs(caught.exception, primary)
        finally:
            sys.setprofile(previous)
        self.assertEqual(len(owners), 1)
        self.assertIs(scope.live_owners[0], owners[0])
        self.assertEqual(os.fstat(descriptors[0]).st_size, len(self.payload))
        self.assertEqual(scope.drain(), [])
        self.assertTrue(scope.retired)
        self.assert_closed(descriptors[0])

    def test_failed_open_native_exception_keeps_valid_closed_slot(self):
        scope = file_io.FileStreams(max_files=1)
        primary, owners = KeyboardInterrupt("after failed common open"), []
        previous = sys.getprofile()

        def profile(frame, event, function):
            if (
                frame.f_code is file_io.FileStreams.open_descriptor.__code__
                and event == "c_exception"
                and getattr(function, "__name__", None) == "open"
                and isinstance(getattr(function, "__self__", None), OwnedFD)
            ):
                owner = function.__self__
                self.assertIs(scope._records[-1].owner, owner)
                self.assertTrue(owner.closed)
                owners.append(owner)
                raise primary

        try:
            sys.setprofile(profile)
            with self.assertRaises(KeyboardInterrupt) as caught:
                scope.open_descriptor(self.path.with_name("missing"), os.O_RDONLY)
            self.assertIs(caught.exception, primary)
        finally:
            sys.setprofile(previous)
        self.assertEqual(len(owners), 1)
        self.assertEqual(scope.live_owners, (owners[0],))
        self.assertEqual(scope.drain(), [])
        self.assertTrue(scope.retired)

    def test_consumed_close_return_never_retires_reused_descriptor(self):
        scope = file_io.FileStreams(max_files=1)
        record, _ = scope.open_descriptor(self.path, os.O_RDONLY)
        descriptor = record.owner.fileno()
        primary, replacements = KeyboardInterrupt("after consumed common close"), []
        previous = sys.getprofile()

        def profile(frame, event, function):
            if (
                not replacements
                and event == "c_return"
                and getattr(function, "__name__", None) == "close"
                and getattr(function, "__self__", None) is record.owner
            ):
                replacement = OwnedFD()
                replacements.append(replacement)
                replacement.open(self.path, os.O_RDONLY)
                raise primary

        try:
            sys.setprofile(profile)
            errors = scope.drain()
            self.assertEqual(errors, [primary])
        finally:
            sys.setprofile(previous)
        try:
            self.assertEqual(len(replacements), 1)
            self.assertTrue(scope.retired)
            self.assertEqual(replacements[0].fileno(), descriptor)
            self.assertEqual(os.read(replacements[0].fileno(), 16), self.payload[:16])
            self.assertEqual(scope.drain(), [])
        finally:
            for replacement in replacements:
                replacement.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

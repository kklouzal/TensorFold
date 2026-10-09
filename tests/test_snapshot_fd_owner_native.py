"""ROOT-only publisher/actual native owner composition; no numerical SDK.

The ordinary Python transport suite uses a substitute owner. These controls
require the built native module and current canonical publisher source.
"""

from __future__ import annotations

import errno
import inspect
import os
from pathlib import Path
import stat
import sys
import tempfile
import unittest

from tensorfold._fd_owner import OwnedFD
from tensorfold.engine import snapshot_payload as api


class NativePublisherControls(unittest.TestCase):
    key = "b" * 32

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="native-snapshot-publisher-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.directory = self.root / "snapshots"

    def write(self, path):
        path.write_bytes(b"abcdefghpayload")

    def assert_closed(self, descriptor):
        with self.assertRaises(OSError) as caught:
            os.fstat(descriptor)
        self.assertEqual(caught.exception.errno, errno.EBADF)

    def interrupt_line(self, code, line, select):
        primary = KeyboardInterrupt("owned publisher interrupted before close")
        descriptors, fired = [], []
        previous = sys.gettrace()

        def trace(frame, event, argument):
            if not fired and frame.f_code is code and event == "line" and frame.f_lineno == line:
                owner = select(frame.f_locals)
                if owner is not None and not owner.closed:
                    descriptors.append(owner.fileno())
                    fired.append(True)
                    raise primary
            return trace

        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt) as caught:
                api.publish_snapshot(self.directory, self.key, self.write)
            self.assertIs(caught.exception, primary)
        finally:
            sys.settrace(previous)
        self.assertEqual(fired, [True])
        self.assertEqual(len(descriptors), 1)
        self.assert_closed(descriptors[0])
        return primary

    def publish_close_lines(self):
        text, first = inspect.getsourcelines(api.publish_snapshot)
        return [
            first + index
            for index, line in enumerate(text)
            if "_close_owned(reservation_owner)" in line or "_close_owned(file_owner)" in line
        ]

    def test_atomic_publication_and_reuse_observation_with_actual_owners(self):
        target = api.publish_snapshot(self.directory, self.key, self.write)
        self.assertEqual(target.read_bytes(), b"abcdefghpayload")
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        observed = []
        self.assertIsNone(
            api.publish_snapshot(
                self.directory,
                self.key,
                lambda p: self.fail("valid target unexpectedly rewritten"),
                valid_existing=lambda p: True,
                reuse_observed=lambda *args: observed.append(args),
            )
        )
        self.assertEqual(len(observed), 1)
        self.assertEqual(observed[0][0], target)
        self.assertEqual(len(observed[0][1]), 5)
        self.assertEqual(observed[0][1][:3], observed[0][2][:3])

    def test_interrupt_before_reserve_file_close_retires_native_owner(self):
        lines = self.publish_close_lines()
        self.assertEqual(len(lines), 2)
        self.interrupt_line(api.publish_snapshot.__code__, lines[0], lambda values: values["reservation_owner"])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_interrupt_before_completed_file_close_retires_native_owner(self):
        lines = self.publish_close_lines()
        self.interrupt_line(api.publish_snapshot.__code__, lines[1], lambda values: values["file_owner"])
        self.assertEqual(list(self.directory.iterdir()), [])

    def test_interrupt_before_directory_close_preserves_committed_target(self):
        text, first = inspect.getsourcelines(api._close_owned)
        line = next(first + i for i, value in enumerate(text) if value.strip() == "owner.close()")

        def select(values):
            owner = values["owner"]
            return owner if stat.S_ISDIR(os.fstat(owner.fileno()).st_mode) else None

        primary = self.interrupt_line(api._close_owned.__code__, line, select)
        self.assertIn("committed", BaseException.__dict__["__dict__"].__get__(primary)["__notes__"][0])
        self.assertEqual((self.directory / f"{self.key}.safetensors").read_bytes(), b"abcdefghpayload")

    def test_native_close_return_error_never_closes_reused_descriptor(self):
        replacement_path = self.root / "replacement"
        replacement_path.write_bytes(b"replacement-owned-data")
        primary, replacements, consumed = KeyboardInterrupt("after native close"), [], []
        previous = sys.getprofile()

        def profile(frame, event, argument):
            if (
                not consumed
                and event == "c_call"
                and getattr(argument, "__name__", None) == "close"
                and isinstance(getattr(argument, "__self__", None), OwnedFD)
            ):
                consumed.append(argument.__self__.fileno())
            if (
                not replacements
                and event == "c_return"
                and getattr(argument, "__name__", None) == "close"
                and isinstance(getattr(argument, "__self__", None), OwnedFD)
            ):
                replacement = OwnedFD()
                replacements.append(replacement)
                replacement.open(replacement_path, os.O_RDONLY)
                raise primary

        try:
            sys.setprofile(profile)
            with self.assertRaises(KeyboardInterrupt) as caught:
                api.publish_snapshot(self.directory, self.key, self.write)
            self.assertIs(caught.exception, primary)
        finally:
            sys.setprofile(previous)
        try:
            self.assertEqual(len(replacements), 1)
            self.assertEqual(replacements[0].fileno(), consumed[0])
            self.assertEqual(os.read(replacements[0].fileno(), 32), b"replacement-owned-data")
            self.assertEqual(list(self.directory.iterdir()), [])
        finally:
            for replacement in replacements:
                replacement.close()

    def test_owned_close_pre_entry_interrupt_settles_same_live_owner(self):
        path = self.root / "owned"
        path.write_bytes(b"data")
        owner = OwnedFD()
        owner.open(path, os.O_RDONLY)
        descriptor = owner.fileno()
        primary, fired = KeyboardInterrupt("before native close"), []
        previous = sys.getprofile()

        def profile(frame, event, argument):
            if (
                not fired
                and event == "c_call"
                and getattr(argument, "__name__", None) == "close"
                and getattr(argument, "__self__", None) is owner
            ):
                fired.append(True)
                raise primary

        try:
            sys.setprofile(profile)
            with self.assertRaises(KeyboardInterrupt) as caught:
                api._close_owned(owner)
            self.assertIs(caught.exception, primary)
        finally:
            sys.setprofile(previous)
        self.assertEqual(fired, [True])
        self.assertTrue(owner.closed)
        self.assert_closed(descriptor)

    def test_published_slot_before_native_open_return_preserves_every_owner(self):
        for selected, field in enumerate(("reservation_owner", "file_owner", "directory_owner"), 1):
            with self.subTest(selected=selected, field=field):
                primary, descriptors, returned = KeyboardInterrupt("after native open"), [], []
                previous = sys.getprofile()

                def profile(frame, event, argument):
                    if (
                        frame.f_code is api.publish_snapshot.__code__
                        and event == "c_return"
                        and getattr(argument, "__name__", None) == "open"
                        and isinstance(getattr(argument, "__self__", None), OwnedFD)
                    ):
                        returned.append(argument.__self__)
                        if len(returned) == selected:
                            self.assertIs(frame.f_locals[field], argument.__self__)
                            descriptors.append(argument.__self__.fileno())
                            raise primary

                try:
                    sys.setprofile(profile)
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        api.publish_snapshot(self.directory, self.key, self.write)
                    self.assertIs(caught.exception, primary)
                finally:
                    sys.setprofile(previous)
                self.assertEqual(len(returned), selected)
                self.assertTrue(all(owner.closed for owner in returned))
                self.assertEqual(len(descriptors), 1)
                self.assert_closed(descriptors[0])
                self.assertEqual(list(self.directory.iterdir()), [])

    def test_failed_native_open_retains_valid_closed_published_slot(self):
        self.directory.mkdir()
        target = self.directory / (self.key + ".safetensors")
        target.write_bytes(b"validold")
        primary, owners = KeyboardInterrupt("after failed native open"), []

        def validate(path):
            path.unlink()
            return True

        previous = sys.getprofile()

        def profile(frame, event, argument):
            if (
                frame.f_code is api._valid_existing.__code__
                and event == "c_exception"
                and getattr(argument, "__name__", None) == "open"
                and isinstance(getattr(argument, "__self__", None), OwnedFD)
            ):
                owner = argument.__self__
                self.assertIs(frame.f_locals["owner"], owner)
                self.assertTrue(owner.closed)
                owners.append(owner)
                raise primary

        try:
            sys.setprofile(profile)
            with self.assertRaises(KeyboardInterrupt) as caught:
                api.publish_snapshot(self.directory, self.key, self.write, valid_existing=validate)
            self.assertIs(caught.exception, primary)
        finally:
            sys.setprofile(previous)
        self.assertEqual(len(owners), 1)
        self.assertTrue(owners[0].closed)
        self.assertEqual(list(self.directory.iterdir()), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)

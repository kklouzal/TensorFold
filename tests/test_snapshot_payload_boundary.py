"""Schema/filesystem and Python transport controls; no numerical SDK.

The native owner is substituted here; its atomic acquisition/consumption is
qualified separately with the actual extension on ROOT's remote host.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch

from tensorfold.engine.snapshot_payload import capture_snapshot, publish_snapshot
from tensorfold.engine.snapshot_registry import LayerSchema, Registry, TensorSchema


class Tensor:
    def __init__(self, shape, dtype="F32", kind="array"):
        self.shape, self.dtype, self.kind = shape, dtype, kind


class Layer:
    drafted = 0
    transient = ("scratch",)

    def __init__(self):
        self.offset, self.drafted = 0, 0
        self.keys = self.history = None
        self.side = [None, None]


def registry():
    return Registry(
        (
            LayerSchema(
                Layer(),
                {
                    "keys": TensorSchema(("F32",), (1, None, 4), 32),
                    "history": TensorSchema(("I32",), (None,), 8),
                    "side.0": TensorSchema(("F32",), (1, None, 4), 32),
                    "side.1": TensorSchema(("F32",), (1, None, 4), 32),
                },
                {"offset": (0, 8), "drafted": (0, 2)},
                list_lengths={"side": 2},
                numpy_fields=frozenset({"history"}),
                required_tensors=frozenset({"keys", "history"}),
            ),
        ),
        token_limit=8,
        token_id_limit=100,
        tensor_byte_limit=256,
        sizes={"F32": 4, "I32": 4},
    )


def layer():
    item = Layer()
    item.offset = 3
    item.keys = Tensor([1, 3, 4])
    item.history = Tensor([3], "I32", "numpy")
    return item


def capture(item, **kwargs):
    return capture_snapshot(
        registry(),
        "model-content-v1",
        [1, 2, 3],
        [item],
        tensor_kind=lambda value: value.kind if type(value) is Tensor else None,
        describe_tensor=lambda value: {"dtype": value.dtype, "shape": list(value.shape)},
        **kwargs,
    )


class CaptureControls(unittest.TestCase):
    def test_exact_owned_metadata_numpy_refs_and_all_none_list(self):
        item = layer()
        payload = capture(item)
        self.assertEqual(payload.numpy_keys, frozenset({"0.history"}))
        self.assertIs(payload.arrays["0.history"], item.history)
        self.assertIs(payload.arrays["0.keys"], item.keys)
        self.assertEqual(json.loads(payload.metadata["tokens"]), [1, 2, 3])
        self.assertEqual(payload.metadata["format"], "2")
        self.assertEqual(
            json.loads(payload.metadata["layers"])[0],
            {
                "class": f"{Layer.__module__}:Layer",
                "plain": {"offset": 3, "drafted": 0},
                "arrays": ["keys"],
                "numpy": ["history"],
                "lists": {"side": {"length": 2, "slots": []}},
            },
        )
        item.side[0] = Tensor([1, 1, 4])
        self.assertNotIn("0.side.0", payload.arrays)

    def test_sparse_native_list_and_declared_class_defaults(self):
        item = layer()
        del item.drafted
        item.side[1] = Tensor([1, 3, 4])
        item.scratch = object()
        payload = capture(item)
        self.assertIs(payload.arrays["0.side.1"], item.side[1])
        entry = json.loads(payload.metadata["layers"])[0]
        self.assertEqual(entry["plain"]["drafted"], 0)
        self.assertEqual(entry["lists"]["side"], {"length": 2, "slots": [1]})
        self.assertNotIn("scratch", entry["plain"])

    def test_stored_false_auxiliary_is_excluded_without_callbacks(self):
        class Auxiliary:
            stored = False

        item = layer()
        payload = capture_snapshot(
            registry(),
            "model",
            [1],
            [Auxiliary(), item, Auxiliary()],
            tensor_kind=lambda value: value.kind,
            describe_tensor=lambda value: {"dtype": value.dtype, "shape": value.shape},
        )
        self.assertEqual(len(json.loads(payload.metadata["layers"])), 1)

    def test_tokens_and_classes_refuse_before_tensor_inspection(self):
        def forbidden(*args):
            self.fail("tensor callback ran before boundary admission")

        for tokens in ([True], [-1], [100], list(range(9)), [1.0], "1"):
            with self.assertRaises(ValueError):
                capture_snapshot(
                    registry(), "model", tokens, [layer()], tensor_kind=forbidden, describe_tensor=forbidden
                )
        for cache in ([], [object()], [layer(), layer()]):
            with self.assertRaises(ValueError):
                capture_snapshot(registry(), "model", [1], cache, tensor_kind=forbidden, describe_tensor=forbidden)

    def test_oversized_list_refuses_before_list_copy_or_tensor_callback(self):
        item = layer()
        item.side = [None] * 1024

        def forbidden(*args):
            self.fail("list copy or tensor callback ran before capacity admission")

        # A list transient declaration is valid and avoids referring to the
        # replaced tuple symbol in the field-type gate. Only this module's
        # tuple-copy operation is intercepted; real Registry code is unchanged.
        with (
            patch.object(Layer, "transient", ["scratch"]),
            patch("tensorfold.engine.snapshot_payload.tuple", side_effect=forbidden, create=True),
        ):
            with self.assertRaises(ValueError):
                capture_snapshot(registry(), "model", [1], [item], tensor_kind=forbidden, describe_tensor=forbidden)

    def test_field_list_representation_and_schema_bounds_refuse(self):
        cases = []
        item = layer()
        item.extra = 1
        cases.append(item)
        item = layer()
        del item.offset
        cases.append(item)
        item = layer()
        item.offset = True
        cases.append(item)
        item = layer()
        item.side = [None] * 3
        cases.append(item)
        item = layer()
        item.side[0] = item.history
        cases.append(item)
        item = layer()
        item.keys = Tensor([1, 9, 4])
        cases.append(item)
        item = layer()
        item.keys = Tensor([1, 3, 4], "I32")
        cases.append(item)
        item = layer()
        item.keys = None
        cases.append(item)
        item = layer()
        item.drafted = float("nan")
        cases.append(item)
        for item in cases:
            with self.assertRaises(ValueError):
                capture(item)


class PublicationControls(unittest.TestCase):
    key = "a" * 32

    def setUp(self):
        from snapshot_fd_transport import TransportOwner

        patcher = patch("tensorfold.engine.snapshot_payload._owned_slot", TransportOwner)
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_each_publisher_slot_is_published_before_post_acquisition_interruption(self):
        from snapshot_fd_transport import TransportOwner
        import inspect
        from tensorfold.engine import snapshot_payload as api

        for selected in (1, 2, 3):
            with self.subTest(selected=selected):
                directory = self.directory()
                primary, slots, descriptors = KeyboardInterrupt("after open"), [], []
                control = self

                class Slot(TransportOwner):
                    def __init__(self):
                        super().__init__()
                        slots.append(self)

                    def open(self, *args, **kwargs):
                        caller = inspect.currentframe().f_back
                        name = ("reservation_owner", "file_owner", "directory_owner")[len(slots) - 1]
                        control.assertIs(caller.f_locals[name], self)
                        control.assertIs(caller.f_code, api.publish_snapshot.__code__)
                        super().open(*args, **kwargs)
                        descriptors.append(self.fileno())
                        if len(slots) == selected:
                            raise primary

                with patch("tensorfold.engine.snapshot_payload._owned_slot", Slot):
                    with self.assertRaises(KeyboardInterrupt) as caught:
                        publish_snapshot(directory, self.key, lambda p: p.write_bytes(b"validdata"))
                self.assertIs(caught.exception, primary)
                self.assertEqual(len(slots), selected)
                self.assertTrue(all(owner.closed for owner in slots))
                for descriptor in descriptors:
                    with self.assertRaises(OSError):
                        os.fstat(descriptor)
                self.assertEqual(list(directory.iterdir()), [])

    def test_existing_validator_slot_is_published_before_open_failure(self):
        from snapshot_fd_transport import TransportOwner
        import inspect
        from tensorfold.engine import snapshot_payload as api

        directory = self.directory()
        target = directory / f"{self.key}.safetensors"
        target.write_bytes(b"validold")
        primary, slots, descriptors = KeyboardInterrupt("after existing open"), [], []
        control = self

        class Slot(TransportOwner):
            def __init__(self):
                super().__init__()
                slots.append(self)

            def open(self, *args, **kwargs):
                caller = inspect.currentframe().f_back
                control.assertIs(caller.f_locals["owner"], self)
                control.assertIs(caller.f_code, api._valid_existing.__code__)
                super().open(*args, **kwargs)
                descriptors.append(self.fileno())
                raise primary

        with patch("tensorfold.engine.snapshot_payload._owned_slot", Slot):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_snapshot(directory, self.key, lambda p: self.fail("writer ran"), valid_existing=lambda p: True)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(slots), 1)
        self.assertTrue(slots[0].closed)
        with self.assertRaises(OSError):
            os.fstat(descriptors[0])
        self.assertEqual(target.read_bytes(), b"validold")

    def test_publisher_cleanup_cannot_erase_primary_native_cause_or_context(self):
        from snapshot_fd_transport import TransportOwner

        directory = self.directory()
        primary, cause, context, cleanup = KeyboardInterrupt(), ValueError(), LookupError(), OSError()
        primary.__cause__, primary.__context__ = cause, context
        slots = []

        class Slot(TransportOwner):
            def __init__(self):
                super().__init__()
                slots.append(self)

            def close(self):
                was_closed = self.closed
                super().close()
                if self is slots[-1] and len(slots) == 2 and not was_closed:
                    primary.__cause__ = primary.__context__ = None
                    raise cleanup

        with (
            patch("tensorfold.engine.snapshot_payload._owned_slot", Slot),
            patch("tensorfold.engine.snapshot_payload.os.fsync", side_effect=primary),
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_snapshot(directory, self.key, lambda p: p.write_bytes(b"validdata"))
        self.assertIs(caught.exception, primary)
        self.assertEqual(primary.__cause__.exceptions, (cause, context, cleanup))
        self.assertTrue(all(owner.closed for owner in slots))
        self.assertEqual(list(directory.iterdir()), [])

    def test_close_retry_preserves_prior_fields_even_when_same_error_is_rethrown(self):
        from snapshot_fd_transport import TransportOwner
        from tensorfold.engine import snapshot_payload as api

        directory = self.directory()
        path = directory / "owned"
        path.write_bytes(b"data")
        primary, cause, context = KeyboardInterrupt(), ValueError(), LookupError()
        primary.__cause__, primary.__context__ = cause, context
        calls = []

        class Slot(TransportOwner):
            def close(self):
                calls.append(self)
                if len(calls) == 2:
                    super().close()
                    primary.__cause__ = primary.__context__ = None
                raise primary

        owner = Slot()
        owner.open(path, os.O_RDONLY)
        with self.assertRaises(KeyboardInterrupt) as caught:
            api._close_owned(owner)
        self.assertIs(caught.exception, primary)
        self.assertEqual(calls, [owner, owner])
        self.assertIs(primary.__cause__, cause)
        self.assertIs(primary.__context__, context)
        self.assertTrue(owner.closed)

    def test_successful_cleanup_preserves_suppressed_native_context(self):
        from tensorfold.engine import snapshot_payload as api

        for suppressed in (False, True):
            with self.subTest(suppressed=suppressed):
                primary, context = ValueError("ordinary validation"), StopIteration()
                primary.__context__, primary.__suppress_context__ = context, suppressed
                with self.assertRaises(ValueError) as caught:
                    api._finish(primary, [])
                self.assertIs(caught.exception, primary)
                self.assertIsNone(primary.__cause__)
                self.assertIs(primary.__context__, context)
                self.assertIs(primary.__suppress_context__, suppressed)

    def directory(self):
        temporary = tempfile.TemporaryDirectory(prefix="snapshot-publish-control-")
        self.addCleanup(temporary.cleanup)
        return Path(temporary.name)

    def test_actual_atomic_file_private_permissions_and_two_fsyncs(self):
        directory = self.directory()
        observed = []

        def write(path):
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            self.assertEqual(path.parent.stat().st_mode & 0o777, 0o700)
            path.write_bytes(b"abcdefghnew")

        real_fsync = os.fsync

        def sync(fd):
            observed.append(stat.S_IFMT(os.fstat(fd).st_mode))
            real_fsync(fd)

        with patch("tensorfold.engine.snapshot_payload.os.fsync", side_effect=sync):
            target = publish_snapshot(directory, self.key, write)
        self.assertEqual(target.read_bytes(), b"abcdefghnew")
        self.assertEqual(target.stat().st_mode & 0o777, 0o600)
        self.assertEqual(observed, [stat.S_IFREG, stat.S_IFDIR])
        self.assertEqual(list(directory.iterdir()), [target])

    def test_invalid_existing_rebuilt_and_valid_existing_skips_write(self):
        directory = self.directory()
        target = directory / f"{self.key}.safetensors"
        target.write_bytes(b"oldinvalid")
        result = publish_snapshot(
            directory,
            self.key,
            lambda p: p.write_bytes(b"newvalid"),
            valid_existing=lambda p: p.read_bytes() == b"newvalid",
        )
        self.assertEqual(result, target)
        self.assertIsNone(
            publish_snapshot(
                directory,
                self.key,
                lambda p: self.fail("validated file should skip write"),
                valid_existing=lambda p: True,
            )
        )
        self.assertEqual(target.read_bytes(), b"newvalid")

    def test_callback_failure_preserves_old_target_and_cleans_owned_temp(self):
        directory = self.directory()
        target = directory / f"{self.key}.safetensors"
        target.write_bytes(b"oldvalid")
        primary = KeyboardInterrupt("writer interrupted")

        def write(path):
            path.write_bytes(b"partial")
            raise primary

        with self.assertRaises(KeyboardInterrupt) as caught:
            publish_snapshot(directory, self.key, write)
        self.assertIs(caught.exception, primary)
        self.assertEqual(target.read_bytes(), b"oldvalid")
        self.assertEqual(list(directory.iterdir()), [target])

    def test_file_fsync_failure_prevents_commit(self):
        directory = self.directory()
        primary = OSError("file fsync failed")
        with patch("tensorfold.engine.snapshot_payload.os.fsync", side_effect=primary):
            with self.assertRaises(OSError) as caught:
                publish_snapshot(directory, self.key, lambda p: p.write_bytes(b"newvalid"))
        self.assertIs(caught.exception, primary)
        self.assertEqual(list(directory.iterdir()), [])

    def test_fsync_and_real_descriptor_close_failures_preserve_primary(self):
        directory = self.directory()

        class Opaque(KeyboardInterrupt):
            def __str__(self):
                raise AssertionError("foreign primary formatted")

            def add_note(self, text):
                raise AssertionError("foreign primary method called")

        primary = Opaque()
        real_close = os.close
        failed_fd, closes = [], []

        def sync(fd):
            failed_fd.append(fd)
            raise primary

        def close(fd):
            real_close(fd)
            if failed_fd and fd == failed_fd[0]:
                failed_fd.clear()
                closes.append(fd)
                raise OSError("real descriptor closed, then close failure")

        with (
            patch("tensorfold.engine.snapshot_payload.os.fsync", side_effect=sync),
            patch("tensorfold.engine.snapshot_payload.os.close", side_effect=close),
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_snapshot(directory, self.key, lambda p: p.write_bytes(b"newvalid"))
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(closes), 1)
        self.assertEqual(list(directory.iterdir()), [])
        self.assertTrue(primary.__notes__)

    def test_writer_and_real_temporary_cleanup_failures_preserve_primary(self):
        directory = self.directory()
        primary = KeyboardInterrupt("writer interrupted")
        real_cleanup = tempfile.TemporaryDirectory.cleanup
        calls = []

        def cleanup(owner):
            real_cleanup(owner)
            calls.append(owner.name)
            raise OSError("owned private path cleaned, then cleanup failure")

        def write(path):
            raise primary

        with patch(
            "tensorfold.engine.snapshot_payload.tempfile.TemporaryDirectory.cleanup", autospec=True, side_effect=cleanup
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_snapshot(directory, self.key, write)
        self.assertIs(caught.exception, primary)
        self.assertEqual(len(calls), 1)
        self.assertEqual(list(directory.iterdir()), [])

    def test_invalid_exception_notes_do_not_replace_primary(self):
        directory = self.directory()
        primary = KeyboardInterrupt("writer interrupted")
        primary.__notes__ = ()
        real_cleanup = tempfile.TemporaryDirectory.cleanup

        def cleanup(owner):
            real_cleanup(owner)
            raise OSError("owned private path cleaned, then cleanup failure")

        def write(path):
            raise primary

        with patch(
            "tensorfold.engine.snapshot_payload.tempfile.TemporaryDirectory.cleanup", autospec=True, side_effect=cleanup
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_snapshot(directory, self.key, write)
        self.assertIs(caught.exception, primary)
        self.assertIsInstance(primary.__cause__, BaseExceptionGroup)
        self.assertTrue(any(isinstance(error, TypeError) for error in primary.__cause__.exceptions))
        self.assertTrue(any(isinstance(error, OSError) for error in primary.__cause__.exceptions))
        self.assertEqual(list(directory.iterdir()), [])

    def test_existing_touch_observation_follows_successful_same_fd_close(self):
        directory = self.directory()
        target = directory / f"{self.key}.safetensors"
        target.write_bytes(b"validold")
        before = target.stat()
        closes, observations = [], []
        real_close = os.close

        def close(fd):
            real_close(fd)
            closes.append(fd)

        def observe(path, first, last):
            self.assertEqual(path, target)
            self.assertEqual(
                first, (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns)
            )
            current = path.stat()
            self.assertEqual(
                last, (current.st_dev, current.st_ino, current.st_size, current.st_mtime_ns, current.st_ctime_ns)
            )
            self.assertEqual(len(closes), 1)
            with self.assertRaises(OSError):
                os.fstat(closes[0])
            observations.append(last)

        with patch("tensorfold.engine.snapshot_payload.os.close", side_effect=close):
            self.assertIsNone(
                publish_snapshot(
                    directory,
                    self.key,
                    lambda p: self.fail("valid immutable target must be reused"),
                    valid_existing=lambda p: True,
                    reuse_observed=observe,
                )
            )
        self.assertEqual(len(observations), 1)

    def test_existing_failed_close_retains_primary_and_suppresses_observation(self):
        directory = self.directory()
        (directory / f"{self.key}.safetensors").write_bytes(b"validold")
        primary = KeyboardInterrupt("utime interrupted")
        primary.__notes__ = 123
        secondary = OSError("actual descriptor consumed before reported failure")
        real_close = os.close

        def close(fd):
            real_close(fd)
            raise secondary

        with (
            patch("tensorfold.engine.snapshot_payload.os.utime", side_effect=primary),
            patch("tensorfold.engine.snapshot_payload.os.close", side_effect=close),
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_snapshot(
                    directory,
                    self.key,
                    lambda p: self.fail("writer ran"),
                    valid_existing=lambda p: True,
                    reuse_observed=lambda *args: self.fail("failed close published a reuse receipt"),
                )
        self.assertIs(caught.exception, primary)
        cause = BaseException.__cause__.__get__(primary, type(primary))
        self.assertIn(secondary, cause.exceptions)
        self.assertTrue(any(isinstance(error, TypeError) for error in cause.exceptions))

    def test_identical_cleanup_failure_has_no_self_cause(self):
        directory = self.directory()
        primary = KeyboardInterrupt("writer interrupted")
        real_cleanup = tempfile.TemporaryDirectory.cleanup

        def cleanup(owner):
            real_cleanup(owner)
            raise primary

        def write(path):
            raise primary

        with patch(
            "tensorfold.engine.snapshot_payload.tempfile.TemporaryDirectory.cleanup", autospec=True, side_effect=cleanup
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_snapshot(directory, self.key, write)
        self.assertIs(caught.exception, primary)
        self.assertIsNone(BaseException.__cause__.__get__(primary, type(primary)))
        self.assertEqual(list(directory.iterdir()), [])

    def test_all_distinct_cleanup_failures_survive_malformed_primary_notes(self):
        directory = self.directory()
        primary = KeyboardInterrupt("directory sync interrupted")
        primary.__notes__ = 123
        close_error, temp_error = OSError("directory close failed"), OSError("private cleanup failed")
        owned_directory = []
        real_close, real_sync = os.close, os.fsync
        real_cleanup = tempfile.TemporaryDirectory.cleanup

        def sync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                owned_directory.append(fd)
                raise primary
            real_sync(fd)

        def close(fd):
            real_close(fd)
            if owned_directory and fd == owned_directory[0]:
                owned_directory.clear()
                raise close_error

        def cleanup(owner):
            real_cleanup(owner)
            raise temp_error

        with (
            patch("tensorfold.engine.snapshot_payload.os.fsync", side_effect=sync),
            patch("tensorfold.engine.snapshot_payload.os.close", side_effect=close),
            patch(
                "tensorfold.engine.snapshot_payload.tempfile.TemporaryDirectory.cleanup",
                autospec=True,
                side_effect=cleanup,
            ),
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_snapshot(directory, self.key, lambda p: p.write_bytes(b"newvalid"))
        self.assertIs(caught.exception, primary)
        causes = BaseException.__cause__.__get__(primary, type(primary)).exceptions
        self.assertIn(close_error, causes)
        self.assertIn(temp_error, causes)
        self.assertNotIn(primary, causes)
        self.assertTrue(any(isinstance(error, TypeError) for error in causes))
        self.assertEqual((directory / f"{self.key}.safetensors").read_bytes(), b"newvalid")

    def test_shadowed_cause_attribute_does_not_mask_primary(self):
        directory = self.directory()

        class Opaque(KeyboardInterrupt):
            @property
            def __cause__(self):
                raise AssertionError("foreign cause accessor ran")

        primary, secondary = Opaque(), OSError("private cleanup failure")
        real_cleanup = tempfile.TemporaryDirectory.cleanup

        def write(path):
            raise primary

        def cleanup(owner):
            real_cleanup(owner)
            raise secondary

        with patch(
            "tensorfold.engine.snapshot_payload.tempfile.TemporaryDirectory.cleanup", autospec=True, side_effect=cleanup
        ):
            with self.assertRaises(KeyboardInterrupt) as caught:
                publish_snapshot(directory, self.key, write)
        self.assertIs(caught.exception, primary)
        self.assertIn(secondary, BaseException.__cause__.__get__(primary, type(primary)).exceptions)

    def test_retained_publisher_owner_bypasses_foreign_dictionary_setter(self):
        directory = self.directory()
        primary, secondary = KeyboardInterrupt("writer primary"), OSError("owner remains open")

        class OpaqueDict(dict):
            def __setitem__(self, key, value):
                raise LookupError("foreign dictionary setter")

        primary.__dict__ = OpaqueDict()
        owners = []

        class PendingOwner:
            def __init__(self):
                self.fd = None
                self.closed = True
                self.fail_close = False
                owners.append(self)

            def open(self, path, flags, mode=0o600):
                self.fd = os.open(path, flags, mode)
                self.closed = False

            def fileno(self):
                return self.fd

            def close(self):
                if self.fail_close:
                    raise secondary
                if not self.closed:
                    os.close(self.fd)
                    self.closed = True

        def write(path):
            # Reopen the operation-owned file and publish it as the current
            # protected owner through a failed subsequent acquisition.
            raise primary

        # Fail the initial close so it remains live through all cleanup. The
        # established primary is the opaque close exception in this scope.
        secondary.__dict__ = OpaqueDict()

        def acquire():
            owner = PendingOwner()
            owner.fail_close = True
            return owner

        try:
            with patch("tensorfold.engine.snapshot_payload._owned_slot", side_effect=acquire):
                with self.assertRaises(OSError) as caught:
                    publish_snapshot(directory, self.key, write)
            self.assertIs(caught.exception, secondary)
            journal = dict.__getitem__(
                BaseException.__dict__["__dict__"].__get__(secondary), "_tensorfold_snapshot_fd_owners"
            )
            self.assertEqual(journal, tuple(owners))
            os.fstat(owners[0].fd)
        finally:
            for owner in owners:
                owner.fail_close = False
                owner.close()

    def test_retained_existing_owner_preserves_opaque_primary_dictionary(self):
        directory = self.directory()
        target = directory / f"{self.key}.safetensors"
        target.write_bytes(b"validold")
        primary, secondary = KeyboardInterrupt("utime interrupted"), OSError("live owner close interrupted")

        class OpaqueDict(dict):
            def __setitem__(self, key, value):
                raise LookupError("foreign retained-journal setter")

        primary.__dict__ = OpaqueDict(__notes__=123)
        owners = []

        class PendingOwner:
            def __init__(self):
                self.fd = None
                self.closed = True
                owners.append(self)

            def open(self, path, flags, mode=0o600):
                self.fd = os.open(path, flags, mode)
                self.closed = False

            def fileno(self):
                return self.fd

            def close(self):
                raise secondary

        try:
            with (
                patch("tensorfold.engine.snapshot_payload._owned_slot", PendingOwner),
                patch("tensorfold.engine.snapshot_payload.os.utime", side_effect=primary),
            ):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    publish_snapshot(
                        directory, self.key, lambda p: self.fail("writer ran"), valid_existing=lambda p: True
                    )
            self.assertIs(caught.exception, primary)
            self.assertEqual(
                dict.__getitem__(BaseException.__dict__["__dict__"].__get__(primary), "_tensorfold_snapshot_fd_owners"),
                tuple(owners),
            )
            self.assertIn(secondary, BaseException.__cause__.__get__(primary).exceptions)
        finally:
            for owner in owners:
                os.close(owner.fd)

    def test_directory_fsync_failure_reports_committed_target(self):
        directory = self.directory()
        primary = OSError("directory fsync failed")
        real_fsync = os.fsync

        def sync(fd):
            if stat.S_ISDIR(os.fstat(fd).st_mode):
                raise primary
            real_fsync(fd)

        with patch("tensorfold.engine.snapshot_payload.os.fsync", side_effect=sync):
            with self.assertRaises(OSError) as caught:
                publish_snapshot(directory, self.key, lambda p: p.write_bytes(b"newvalid"))
        self.assertIs(caught.exception, primary)
        self.assertIn("committed", primary.__notes__[0])
        self.assertEqual((directory / f"{self.key}.safetensors").read_bytes(), b"newvalid")
        self.assertEqual(len(list(directory.iterdir())), 1)

    def test_concurrent_valid_publish_is_retained(self):
        directory = self.directory()
        target = directory / f"{self.key}.safetensors"

        def write(path):
            path.write_bytes(b"oursvalid")
            target.write_bytes(b"othervalid")

        self.assertIsNone(
            publish_snapshot(directory, self.key, write, valid_existing=lambda p: p.read_bytes() == b"othervalid")
        )
        self.assertEqual(target.read_bytes(), b"othervalid")
        self.assertEqual(list(directory.iterdir()), [target])

    def test_changed_target_identity_cannot_suppress_publication(self):
        directory = self.directory()
        target = directory / f"{self.key}.safetensors"
        target.write_bytes(b"oldvalid")

        def validate(path):
            new = directory / "replacement"
            new.write_bytes(b"invalid!")
            new.replace(path)
            return True

        result = publish_snapshot(directory, self.key, lambda p: p.write_bytes(b"oursvalid"), valid_existing=validate)
        self.assertEqual(result, target)
        self.assertEqual(target.read_bytes(), b"oursvalid")

    def test_key_refuses_before_directory_or_writer_acquisition(self):
        directory = self.directory() / "uncreated"
        for key in ("../escape", "A" * 32, "a" * 31, False):
            with self.assertRaises(ValueError):
                publish_snapshot(directory, key, lambda p: self.fail("writer acquired"))
        self.assertFalse(directory.exists())


if __name__ == "__main__":
    unittest.main()

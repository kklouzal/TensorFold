"""ROOT-only installed native FD controls; no tensor/numeric SDK imports.

Run on ROOT's authorized remote host after building/installing the platform
wheel: python -W error tests/test_fd_owner_native.py. Missing extension fails.
These controls do not qualify Apple native execution or consumed EIO injection.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import dis
import errno
import gc
import hashlib
import os
from pathlib import Path
import pickle
import resource
import sys
import tempfile
import unittest
import weakref

from tensorfold import _fd_owner
from tensorfold._fd_owner import OwnedFD


class NativeFDControls(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tensorfold-native-fd-test-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.path = self.directory / "owned β file"
        self.path.write_bytes(b"01234567")

    def slot(self):
        owner = OwnedFD()
        self.addCleanup(owner.close)  # closed slot published before acquisition
        return owner

    def test_installed_binary_and_maintained_source_supplier_metadata(self):
        binary = Path(_fd_owner.__file__)
        self.assertIn(".abi3.", binary.name)
        self.assertEqual(_fd_owner._ownership_version, 2)
        self.assertEqual(_fd_owner._limited_api, 0x030B0000)
        source = binary.with_name("_fd_owner.c")
        self.assertEqual(_fd_owner._source_sha256, hashlib.sha256(source.read_bytes()).hexdigest())
        self.assertEqual(OwnedFD.__module__, "tensorfold._fd_owner")

    def test_regular_directory_unicode_and_bytes_owners_are_noninheritable(self):
        for path, flags in ((self.path, os.O_RDONLY), (os.fsencode(self.path), os.O_RDONLY),
                            (self.directory, os.O_RDONLY | os.O_DIRECTORY)):
            owner = self.slot()
            try:
                self.assertIsNone(owner.open(path, flags))
                descriptor = owner.fileno()
                self.assertFalse(owner.closed)
                self.assertFalse(os.get_inheritable(descriptor))
                os.fstat(descriptor)
            finally:
                owner.close()
            self.assertTrue(owner.closed)
            owner.close()
            with self.assertRaises(ValueError):
                owner.fileno()
            with self.assertRaises(OSError) as caught:
                os.fstat(descriptor)
            self.assertEqual(caught.exception.errno, errno.EBADF)

    def test_openat_borrows_parent_and_preserves_nofollow_boundary(self):
        parent = self.slot()
        try:
            parent.open(self.directory, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
            child = self.slot()
            try:
                child.open(self.path.name, os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent.fileno())
                self.assertEqual(os.read(child.fileno(), 8), b"01234567")
            finally:
                child.close()
            (self.directory / "escape").symlink_to(self.path)
            refused = self.slot()
            with self.assertRaises(OSError):
                refused.open("escape", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=parent.fileno())
            self.assertTrue(refused.closed)
            self.assertFalse(parent.closed)
            os.fstat(parent.fileno())
        finally:
            parent.close()

    def test_invalid_paths_modes_and_argument_callbacks_fail_before_acquisition(self):
        target = self.directory / "not-created"
        primary = RuntimeError("path callback failure")
        class BadPath:
            def __fspath__(self):
                raise primary
        owner = self.slot()
        with self.assertRaises(RuntimeError) as caught:
            owner.open(BadPath(), os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        self.assertIs(caught.exception, primary)
        for mode in (-1, 0o10000):
            with self.assertRaises(ValueError):
                owner.open(target, os.O_WRONLY | os.O_CREAT | os.O_EXCL, mode)
        with self.assertRaises(ValueError):
            owner.open(str(target) + "\0suffix", os.O_WRONLY | os.O_CREAT)
        with self.assertRaises(OverflowError):
            owner.open(target, os.O_WRONLY | os.O_CREAT, dir_fd=2**100)
        self.assertFalse(target.exists())
        with self.assertRaises(FileNotFoundError):
            owner.open(target, os.O_RDONLY)
        self.assertTrue(owner.closed)
        self.assertIsNone(owner.open(self.path, os.O_RDONLY))

    def test_owner_cannot_be_mutated_subclassed_cycled_or_uninitialized(self):
        owner = self.slot()
        try:
            owner.open(self.path, os.O_RDONLY)
            with self.assertRaises(AttributeError):
                owner.closed = False
            with self.assertRaises(AttributeError):
                owner.other = owner
            with self.assertRaises(TypeError):
                weakref.ref(owner)
            with self.assertRaises(TypeError):
                type("BadOwner", (OwnedFD,), {})
            with self.assertRaises(TypeError):
                object.__new__(OwnedFD)
            with self.assertRaises(TypeError):
                pickle.dumps(owner)
            with self.assertRaises(TypeError):
                OwnedFD.close = lambda _: None
        finally:
            owner.close()

    def test_concurrent_close_results_retire_once_after_all_calls_are_observed(self):
        owner = self.slot()
        owner.open(self.path, os.O_RDONLY)
        descriptor = owner.fileno()
        with ThreadPoolExecutor(8) as executor:
            results = [executor.submit(owner.close) for _ in range(64)]
            self.assertTrue(all(future.result(timeout=5) is None for future in results))
        self.assertTrue(owner.closed)
        with self.assertRaises(OSError):
            os.fstat(descriptor)

    def test_invalid_external_close_reports_error_and_never_retries_number(self):
        # Deliberately violate borrowed-FD ownership to exercise EBADF failure
        # containment; this is not evidence of valid consumed EIO semantics.
        owner = self.slot()
        owner.open(self.path, os.O_RDONLY)
        descriptor = owner.fileno()
        os.close(descriptor)
        with self.assertRaises(OSError) as caught:
            owner.close()
        self.assertEqual(caught.exception.errno, errno.EBADF)
        self.assertTrue(owner.closed)
        replacement = os.open(self.path, os.O_RDONLY)
        try:
            owner.close()
            os.fstat(replacement)
        finally:
            os.close(replacement)

    def test_bounded_descriptor_exhaustion_reports_os_error_and_recovers_after_close(self):
        original = resource.getrlimit(resource.RLIMIT_NOFILE)
        owners = []
        try:
            limit = 64 if original[0] == resource.RLIM_INFINITY else min(original[0], 64)
            resource.setrlimit(resource.RLIMIT_NOFILE, (limit, original[1]))
            for _ in range(128):
                owner = OwnedFD()
                owners.append(owner)  # closed slot journal precedes .open
                try:
                    owner.open(self.path, os.O_RDONLY)
                except OSError as error:
                    self.assertEqual(error.errno, errno.EMFILE)
                    break
            else:
                self.fail("bounded process descriptor admission did not exhaust")
        finally:
            for owner in owners:
                owner.close()
            resource.setrlimit(resource.RLIMIT_NOFILE, original)
        owner = self.slot()
        owner.open(self.path, os.O_RDONLY)
        owner.close()

    def test_interrupt_before_native_close_keeps_exact_owner_for_explicit_cleanup(self):
        owner = self.slot()
        owner.open(self.directory, os.O_RDONLY | os.O_DIRECTORY)
        descriptor = owner.fileno()
        def cleanup():
            owner.close()
        target = cleanup.__code__.co_firstlineno + 1
        primary = KeyboardInterrupt("before native close")
        previous = sys.gettrace()
        def trace(frame, event, argument):
            if frame.f_code is cleanup.__code__ and event == "line" and frame.f_lineno == target:
                raise primary
            return trace
        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt) as caught:
                cleanup()
            self.assertIs(caught.exception, primary)
        finally:
            sys.settrace(previous)
        try:
            self.assertFalse(owner.closed)
            os.fstat(descriptor)
        finally:
            owner.close()
        self.assertTrue(owner.closed)

    def test_native_close_return_interruption_cannot_close_reused_descriptor(self):
        owner = self.slot()
        owner.open(self.path, os.O_RDONLY)
        descriptor = owner.fileno()
        replacement = []
        primary = KeyboardInterrupt("after native consumption")
        previous = sys.getprofile()
        def profile(frame, event, function):
            if event == "c_return" and getattr(function, "__self__", None) is owner and function.__name__ == "close":
                self.assertTrue(owner.closed)
                new = os.open(self.path, os.O_RDONLY)
                if new != descriptor:
                    os.dup2(new, descriptor)
                    os.close(new)
                replacement.append(descriptor)
                raise primary
        try:
            sys.setprofile(profile)
            with self.assertRaises(KeyboardInterrupt) as caught:
                owner.close()
            self.assertIs(caught.exception, primary)
        finally:
            sys.setprofile(previous)
        try:
            self.assertEqual(replacement, [descriptor])
            owner.close()
            os.fstat(descriptor)
        finally:
            if replacement:
                os.close(replacement[0])

    @unittest.skipUnless(sys.platform == "linux", "exact fresh-slot FD enumeration uses Linux procfs")
    def test_closed_constructor_return_before_python_store_acquires_nothing(self):
        def acquire():
            owner = OwnedFD()
            return owner
        store = next(i.offset for i in dis.get_instructions(acquire) if i.opname == "STORE_FAST" and i.argval == "owner")
        identity = self.path.stat().st_dev, self.path.stat().st_ino
        descriptors, fired = [], []
        primary = KeyboardInterrupt("before closed-slot caller store")
        def witness(code, offset):
            if code is acquire.__code__ and offset == store:
                fired.append(offset)
                for path in Path("/proc/self/fd").iterdir():
                    try:
                        info = os.fstat(int(path.name))
                    except OSError:
                        continue
                    if (info.st_dev, info.st_ino) == identity:
                        descriptors.append(int(path.name))
                raise primary
        if hasattr(sys, "monitoring"):
            # CPython 3.12's public local INSTRUCTION event runs before STORE;
            # it does not depend on late f_trace_opcodes activation.
            monitor = sys.monitoring
            tool = next((i for i in range(6) if monitor.get_tool(i) is None), None)
            self.assertIsNotNone(tool, "no free bounded monitoring tool ID")
            monitor.use_tool_id(tool, "tensorfold-native-fd-before-store")
            try:
                monitor.register_callback(tool, monitor.events.INSTRUCTION, witness)
                monitor.set_local_events(tool, acquire.__code__, monitor.events.INSTRUCTION)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    acquire()
                self.assertIs(caught.exception, primary)
            finally:
                monitor.set_local_events(tool, acquire.__code__, 0)
                monitor.register_callback(tool, monitor.events.INSTRUCTION, None)
                monitor.free_tool_id(tool)
        else:
            # The declared CPython 3.11 floor supplies per-frame opcode tracing.
            previous = sys.gettrace()
            current = sys._getframe()
            previous_opcodes = current.f_trace_opcodes
            def trace(frame, event, argument):
                if frame.f_code is acquire.__code__:
                    frame.f_trace_opcodes = True
                    if event == "opcode":
                        witness(frame.f_code, frame.f_lasti)
                return trace
            try:
                current.f_trace_opcodes = True
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    acquire()
                self.assertIs(caught.exception, primary)
            finally:
                sys.settrace(previous)
                current.f_trace_opcodes = previous_opcodes
        self.assertEqual(fired, [store], "the required pre-publication instruction witness did not fire exactly once")
        gc.collect()
        self.assertEqual(descriptors, [], "fresh constructor must never acquire a descriptor")

    def test_fresh_slot_close_and_successful_one_shot_acquisition(self):
        owner = self.slot()
        self.assertTrue(owner.closed)
        self.assertIsNone(owner.close())
        with self.assertRaises(ValueError):
            owner.fileno()
        with self.assertRaises(TypeError):
            OwnedFD(self.path, os.O_RDONLY)
        with self.assertRaises(TypeError):
            OwnedFD(path=self.path, flags=os.O_RDONLY)
        owner.open(self.path, os.O_RDONLY)
        descriptor = owner.fileno()
        for retired in (False, True):
            if retired:
                owner.close()
            with self.assertRaises(RuntimeError):
                owner.open(self.path, os.O_RDONLY)
            self.assertEqual(owner.closed, retired)
            if not retired:
                self.assertEqual(owner.fileno(), descriptor)

    def test_path_callback_reentry_cannot_overwrite_or_close_acquisition(self):
        owner = self.slot()
        calls = []
        testcase = self

        class Reenter:
            def __fspath__(self):
                testcase.assertTrue(owner.closed)
                with testcase.assertRaises(RuntimeError):
                    owner.open(testcase.path, os.O_RDONLY)
                with testcase.assertRaises(RuntimeError):
                    owner.close()
                with testcase.assertRaises(ValueError):
                    owner.fileno()
                calls.append(True)
                return os.fspath(testcase.path)

        owner.open(Reenter(), os.O_RDONLY)
        self.assertEqual(calls, [True])
        self.assertFalse(owner.closed)
        self.assertEqual(os.read(owner.fileno(), 8), b"01234567")

    def test_native_open_return_interruption_retains_journaled_live_owner(self):
        journal = [self.slot()]
        owner = journal[0]
        primary = KeyboardInterrupt("after native open before caller return")
        previous = sys.getprofile()
        descriptors = []

        def profile(frame, event, function):
            if event == "c_return" and getattr(function, "__self__", None) is owner and function.__name__ == "open":
                self.assertIs(journal[0], owner)
                self.assertFalse(owner.closed)
                descriptors.append(owner.fileno())
                raise primary

        try:
            sys.setprofile(profile)
            with self.assertRaises(KeyboardInterrupt) as caught:
                owner.open(self.path, os.O_RDONLY)
            self.assertIs(caught.exception, primary)
        finally:
            sys.setprofile(previous)
        self.assertEqual(len(descriptors), 1)
        self.assertFalse(journal[0].closed)
        self.assertEqual(os.read(journal[0].fileno(), 8), b"01234567")
        journal[0].close()
        self.assertTrue(journal[0].closed)
        with self.assertRaises(OSError) as caught:
            os.fstat(descriptors[0])
        self.assertEqual(caught.exception.errno, errno.EBADF)

    def test_interrupt_before_open_return_result_store_retains_journaled_owner(self):
        journal = [self.slot()]
        owner = journal[0]

        def acquire():
            completed = journal[0].open(self.path, os.O_RDONLY)
            return completed

        store = next(i.offset for i in dis.get_instructions(acquire) if i.opname == "STORE_FAST" and i.argval == "completed")
        fired, descriptors = [], []
        primary = KeyboardInterrupt("after open before result store")

        def witness(code, offset):
            if code is acquire.__code__ and offset == store:
                self.assertIs(journal[0], owner)
                self.assertFalse(owner.closed)
                descriptors.append(owner.fileno())
                fired.append(offset)
                raise primary

        if hasattr(sys, "monitoring"):
            monitor = sys.monitoring
            tool = next((i for i in range(6) if monitor.get_tool(i) is None), None)
            self.assertIsNotNone(tool)
            monitor.use_tool_id(tool, "tensorfold-native-open-before-store")
            try:
                monitor.register_callback(tool, monitor.events.INSTRUCTION, witness)
                monitor.set_local_events(tool, acquire.__code__, monitor.events.INSTRUCTION)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    acquire()
                self.assertIs(caught.exception, primary)
            finally:
                monitor.set_local_events(tool, acquire.__code__, 0)
                monitor.register_callback(tool, monitor.events.INSTRUCTION, None)
                monitor.free_tool_id(tool)
        else:
            previous = sys.gettrace()
            current = sys._getframe()
            previous_opcodes = current.f_trace_opcodes

            def trace(frame, event, argument):
                if frame.f_code is acquire.__code__:
                    frame.f_trace_opcodes = True
                    if event == "opcode":
                        witness(frame.f_code, frame.f_lasti)
                return trace

            try:
                current.f_trace_opcodes = True
                sys.settrace(trace)
                with self.assertRaises(KeyboardInterrupt) as caught:
                    acquire()
                self.assertIs(caught.exception, primary)
            finally:
                sys.settrace(previous)
                current.f_trace_opcodes = previous_opcodes
        self.assertEqual(fired, [store])
        self.assertFalse(owner.closed)
        self.assertEqual(os.read(descriptors[0], 8), b"01234567")
        owner.close()
        with self.assertRaises(OSError) as caught:
            os.fstat(descriptors[0])
        self.assertEqual(caught.exception.errno, errno.EBADF)


if __name__ == "__main__":
    unittest.main(verbosity=2)

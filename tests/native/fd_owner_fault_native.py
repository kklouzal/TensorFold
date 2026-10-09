"""ROOT-only Linux actual consumed-close faults and SIGINT lifetime controls.

Load one explicitly hashed ABI3 provider and project-owned LD_PRELOAD shim.
This worker imports no tensor SDK and never compiles or changes production code.
The launch/build packet supplies the approved paths and resource supervisor.
"""
from __future__ import annotations

import os
import sys

if os.environ.get("TENSORFOLD_ROOT_REMOTE_QUALIFICATION") != "fd-owner-fault-v1":
    raise SystemExit("ROOT remote FD qualification guard required before any native load")
if sys.platform != "linux" or sys.implementation.name != "cpython" or sys.version_info < (3, 11):
    raise SystemExit("Linux CPython >=3.11 qualification required")

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import ctypes
import dis
import errno
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import signal
import tempfile
import threading
import unittest

EXPECTED_SOURCE = "e4f9e18724e1747d8b6ce5a67d8d228fa08fa14f695739acfd0b5bd1629d8db0"
STATS_FIELDS = (
    "version", "phase", "target_fd", "injected_errno", "underlying_calls",
    "underlying_result", "underlying_errno", "consumed_witness", "replacement_fd",
    "replacement_errno", "retry_calls", "passthrough_calls", "identity_mismatches", "setup_error",
)


def error_name(error):
    """Bounded actual class name without invoking exception metaclass hooks."""
    return str.__getitem__(type.__dict__['__name__'].__get__(type(error)), slice(None, 64))


def raise_grouped(primary, causes):
    """Group owned statuses, preserving the primary if allocation fails.

    The fallback list is operation-owned diagnostic state; callers must treat
    it as read-only. Grouping/retention errors become the explicit primary cause.
    No resource work happens here, and callbacks/retirement precede this call.
    """
    try:
        dictionary = BaseException.__dict__['__dict__'].__get__(primary, type(primary))
        dict.__setitem__(dictionary, '_tensorfold_retained_failures', causes)
    except BaseException as retention:
        if retention is primary:
            raise primary
        raise primary from retention
    try:
        grouped = BaseExceptionGroup('owned cleanup and retained native failures', causes)
    except BaseException as allocation:
        if allocation is primary:
            raise primary
        raise primary from allocation
    dict.__delitem__(dictionary, '_tensorfold_retained_failures')
    raise primary from grouped


def _transport(primary, errors):
    if primary is None and errors:
        primary, errors = errors[0], errors[1:]
    if primary is None:
        return
    previous = (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary))
    try:
        dictionary = BaseException.__dict__['__dict__'].__get__(primary, type(primary))
        retained = dict.get(dictionary, '_tensorfold_retained_failures')
        if retained is not None and type(retained) is not list:
            raise ValueError('owned retained failure payload must remain a list')
        retained = () if retained is None else tuple(retained)
    except BaseException as retention:
        if retention is primary:
            raise primary
        raise primary from retention
    others = []
    for error in errors:
        if error is not primary and all(error is not previous for previous in others):
            others.append(error)
    annotations = []
    annotation_failed = False
    messages = ['owned cleanup also failed (' +
                error_name(error) + ')'
                for error in others[:8]]
    if len(others) > 8:
        messages.append('additional owned failure count: ' + str(len(others) - 8))
    for message in messages:
        try:
            BaseException.add_note(primary, message)
        except BaseException as annotation:
            annotation_failed = True
            if annotation is not primary and all(annotation is not error for error in [*others, *annotations]):
                annotations.append(annotation)
            break
    causes = []
    if others or annotations:
        for error in (*retained, *previous):
            if error is not None and error is not primary and all(error is not item for item in causes):
                causes.append(error)
    for error in [*others, *annotations]:
        if all(error is not previous for previous in causes):
            causes.append(error)
    if len(causes) == 1 and not annotation_failed:
        raise primary from causes[0]
    if causes:
        raise_grouped(primary, causes)
    raise primary



def raise_failures(primary, errors):
    """Preserve primary identity even if cold diagnostic preparation exhausts.

    Required work has retired before entry. These original status references
    and native fields remain traceback-frame-owned if labels, annotations,
    retained-payload snapshots or group preparation itself cannot allocate.
    This fallback reports that distinct transport failure as the explicit
    cause; it does not promise allocation succeeds or a complete group forms.
    """
    if primary is None:
        if not errors:
            return
        primary = errors[0]
    native_cause = BaseException.__cause__.__get__(primary)
    native_context = BaseException.__context__.__get__(primary)
    try:
        _transport(primary, errors)
    except BaseException as failure:
        if failure is primary:
            raise
        # Keep the named native/status references in this traceback frame.
        # The guards do not format exceptions or allocate a new status payload.
        if native_cause is primary:
            native_cause = None
        if native_context is primary:
            native_context = None
        raise primary from failure


def finish(primary, operations):
    """Attempt every independent cleanup before propagating native statuses."""
    # Strong references precede any foreign cleanup/monitor callback.
    native_cause = None if primary is None else BaseException.__cause__.__get__(primary)
    native_context = None if primary is None else BaseException.__context__.__get__(primary)
    errors = []
    for operation in operations:
        try:
            operation()
        except BaseException as error:
            errors.append(error)
    try:
        retained = [root for root in (native_cause, native_context)
                    if root is not None and root is not primary]
        retained.extend(errors)
    except BaseException as transport:
        if primary is not None:
            raise primary from transport
        raise
    raise_failures(primary, retained)


class Stats(ctypes.Structure):
    _fields_ = [(name, ctypes.c_int64) for name in STATS_FIELDS]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def descriptors():
    result = {}
    for name in os.listdir("/proc/self/fd"):
        try:
            info = os.fstat(int(name))
        except OSError as error:
            if error.errno != errno.EBADF:
                raise
            continue
        result[int(name)] = (info.st_dev, info.st_ino, info.st_mode)
    return result


def identity(descriptor):
    info = os.fstat(descriptor)
    return info.st_dev, info.st_ino


def load_suppliers(args, provider_module=None):
    provider, shim, source = args.provider.resolve(strict=True), args.shim.resolve(strict=True), args.source.resolve(strict=True)
    if args.source_sha256 != EXPECTED_SOURCE or sha(source) != EXPECTED_SOURCE:
        raise ValueError("exact approved maintained provider C source required")
    if sha(provider) != args.provider_sha256 or sha(shim) != args.shim_sha256:
        raise ValueError("actual provider/shim binary identity mismatch")
    if not provider.name.endswith(".abi3.so"):
        raise ValueError("approved Linux ABI3 extension binary required")
    if os.environ.get("LD_PRELOAD") != str(shim):
        raise ValueError("one exact isolated shim must be the complete LD_PRELOAD value")
    if threading.current_thread() is not threading.main_thread():
        raise ValueError("signal/control owner must be the process main thread")
    if hasattr(sys, "_is_gil_enabled") and not sys._is_gil_enabled():
        raise ValueError("provider contract requires the declared GIL-enabled CPython runtime")
    library = ctypes.CDLL(str(shim), use_errno=True)
    library.tf_fd_fault_arm.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_char_p]
    library.tf_fd_fault_arm.restype = ctypes.c_int
    library.tf_fd_fault_snapshot.argtypes = [ctypes.POINTER(Stats)]
    library.tf_fd_fault_snapshot.restype = ctypes.c_int
    library.tf_fd_fault_reset.argtypes = []
    library.tf_fd_fault_reset.restype = ctypes.c_int
    library.tf_fd_fault_stats_size.argtypes = []
    library.tf_fd_fault_stats_size.restype = ctypes.c_uint
    if library.tf_fd_fault_stats_size() != ctypes.sizeof(Stats):
        raise ValueError("shim stats protocol size mismatch")
    if ctypes.cast(ctypes.CDLL(None).close, ctypes.c_void_p).value != ctypes.cast(library.close, ctypes.c_void_p).value:
        raise ValueError("LD_PRELOAD close symbol is not the approved shim")
    module = provider_module
    if module is None:
        module = sys.modules.get("tensorfold._fd_owner")
    reused = module is not None
    if module is None:
        spec = importlib.util.spec_from_file_location("_fd_owner", provider)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    if module._ownership_version != 2 or module._limited_api != 0x030B0000 or module._source_sha256 != EXPECTED_SOURCE:
        raise ValueError("actual native supplier metadata differs from approved ABI/source")
    if Path(module.__file__).resolve() != provider:
        raise ValueError("native provider origin changed")
    return module, library, {
        "provider": str(provider), "provider_sha256": sha(provider), "source": str(source),
        "source_sha256": sha(source), "shim": str(shim), "shim_sha256": sha(shim),
        "python": sys.version, "executable": sys.executable, "platform": sys.platform,
        "abi_floor": "0x030B0000", "ownership_version": 2,
        "reused_approved_provider_module": reused,
        "loaded_origin_scope": "explicit exact provider; canonical wheel/RECORD authority is a separate Root gate",
    }


class Controls(unittest.TestCase):
    api = None
    shim = None
    witnesses = {}

    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="tensorfold-fd-fault-")
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.original, self.replacement, self.sentinel = [self.directory / name for name in ("original", "replacement", "sentinel")]
        self.original.write_bytes(b"original bytes")
        self.replacement.write_bytes(b"replacement remains owned")
        self.sentinel.write_bytes(b"unrelated descriptor")
        self.journal = []
        self.marker_reset_failed = False
        self.addCleanup(self.drain_journal)
        self.assertEqual(self.shim.tf_fd_fault_reset(), 0)
        self.assertIs(threading.current_thread(), threading.main_thread())

    def snapshot(self):
        stats = Stats()
        self.assertEqual(self.shim.tf_fd_fault_snapshot(ctypes.byref(stats)), 0)
        self.assertEqual(stats.version, 1)
        return {name: getattr(stats, name) for name in STATS_FIELDS}

    def arm(self, descriptor, code):
        self.assertEqual(self.shim.tf_fd_fault_arm(descriptor, code, os.fsencode(self.replacement)), 0)

    def new_slot(self):
        owner = self.api.OwnedFD()
        self.journal.append(owner) # closed slot published before any acquisition
        return owner

    def retire_slot(self, owner):
        self.assertTrue(owner.closed)
        self.journal[:] = [slot for slot in self.journal if slot is not owner]

    def drain_journal(self):
        if self.marker_reset_failed:
            raise RuntimeError("marked qualification reset failed; dependent journal retained")
        finish(None, [owner.close for owner in self.journal])
        self.journal.clear()

    @contextmanager
    def fixture(self, code):
        owner = None
        primary = None
        try:
            owner = self.new_slot()
            self.assertIsNone(owner.open(self.original, os.O_RDONLY))
            descriptor = owner.fileno()
            self.arm(descriptor, code)
            yield owner, descriptor
        except BaseException as error:
            primary = error
        finally:
            # Snapshot before disarming; reset awaits the native sequence. Never
            # close a replacement by number until its inode proves exact owner.
            finish(primary, [lambda: self.cleanup(owner)])

    def cleanup(self, owner):
        stats, errors = None, []
        try:
            stats = self.snapshot()
        except BaseException as error:
            errors.append(error)
        try:
            self.assertEqual(self.shim.tf_fd_fault_reset(), 0)
            self.marker_reset_failed = False
        except BaseException as error:
            # A reset failure leaves dependent marked authority unusable. Keep
            # the exact native owner; unrelated owners still get cleanup.
            self._failed_owner = owner
            self.marker_reset_failed = True
            errors.append(error)
            raise_failures(None, errors)
        def close_owner():
            if owner is not None and not owner.closed:
                owner.close()
            if owner is not None and owner.closed:
                self.retire_slot(owner)
        def close_replacement():
            if stats is not None and stats["replacement_fd"] >= 0:
                replacement = stats["replacement_fd"]
                self.assertEqual(identity(replacement), (self.replacement.stat().st_dev, self.replacement.stat().st_ino))
                os.close(replacement)
        try:
            finish(None, [close_owner, close_replacement])
        except BaseException as error:
            errors.append(error)
        raise_failures(None, errors)

    def consumed(self, owner, descriptor, code):
        self.assertTrue(owner.closed)
        with self.assertRaises(ValueError):
            owner.fileno()
        stats = self.snapshot()
        self.assertEqual((stats["phase"], stats["target_fd"], stats["injected_errno"]), (2, descriptor, code))
        self.assertEqual((stats["underlying_calls"], stats["underlying_result"], stats["underlying_errno"]), (1, 0, 0))
        self.assertEqual((stats["consumed_witness"], stats["replacement_fd"], stats["replacement_errno"]), (1, descriptor, 0))
        self.assertEqual((stats["retry_calls"], stats["identity_mismatches"], stats["setup_error"]), (0, 0, 0))
        self.assertEqual(os.pread(descriptor, 64, 0), b"replacement remains owned")
        self.assertFalse(os.get_inheritable(descriptor))
        return stats

    def fault(self, code):
        sentinel = os.open(self.sentinel, os.O_RDONLY | os.O_CLOEXEC)
        owner = None
        primary = None
        try:
            before = identity(sentinel)
            owner = self.new_slot()
            self.assertIsNone(owner.open(self.original, os.O_RDONLY))
            descriptor = owner.fileno()
            self.arm(descriptor, code)
            with self.assertRaises(OSError) as caught:
                owner.close()
            self.assertEqual(caught.exception.errno, code)
            self.consumed(owner, descriptor, code)
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(owner.close) for _ in range(64)]
                self.assertTrue(all(future.result(timeout=5) is None for future in futures))
            self.assertEqual(identity(sentinel), before)
            self.assertEqual(os.pread(sentinel, 64, 0), b"unrelated descriptor")
            self.consumed(owner, descriptor, code)
            # Drop all returned-owner references while the exact replacement is
            # still live and the marker still observes any repeated native close.
            del futures, caught
            self.retire_slot(owner)
            owner = None
            gc.collect()
            stats = self.snapshot()
            self.assertEqual((stats["underlying_calls"], stats["retry_calls"]), (1, 0))
            self.assertEqual(os.pread(descriptor, 64, 0), b"replacement remains owned")
            self.assertEqual(identity(sentinel), before)
            self.witnesses[self._testMethodName] = dict(stats, destructor_observed_after_reference_retirement=True)
        except BaseException as error:
            primary = error
        finally:
            finish(primary, [lambda: self.cleanup(owner), lambda: os.close(sentinel)])

    def test_actual_consumed_EIO_with_same_number_reuse_and_no_retry(self):
        self.fault(errno.EIO)

    def test_actual_consumed_EINTR_with_same_number_reuse_and_no_retry(self):
        self.fault(errno.EINTR)

    def test_arm_reset_and_unrelated_real_closes_are_scoped(self):
        with self.fixture(errno.EIO) as (owner, descriptor):
            self.assertEqual(self.shim.tf_fd_fault_arm(descriptor, errno.EINTR, os.fsencode(self.replacement)), errno.EBUSY)
            others = [os.open(self.sentinel, os.O_RDONLY | os.O_CLOEXEC) for _ in range(64)]
            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = [executor.submit(os.close, fd) for fd in others]
                self.assertTrue(all(future.result(timeout=5) is None for future in futures))
                refusal = executor.submit(self.shim.tf_fd_fault_reset).result(timeout=5)
            self.assertEqual(refusal, errno.EPERM)
            stats = self.snapshot()
            self.assertEqual((stats["phase"], stats["underlying_calls"]), (1, 0))
            self.assertGreaterEqual(stats["passthrough_calls"], 64)
            self.assertFalse(owner.closed)
            self.assertEqual(os.pread(descriptor, 64, 0), b"original bytes")
            self.witnesses[self._testMethodName] = stats

    @contextmanager
    def actual_sigint(self):
        delivered = []
        previous = signal.getsignal(signal.SIGINT)
        def handler(number, frame):
            delivered.append(number)
            signal.default_int_handler(number, frame)
        signal.signal(signal.SIGINT, handler)
        primary = None
        try:
            yield delivered
        except BaseException as error:
            primary = error
        finally:
            finish(primary, [lambda: signal.signal(signal.SIGINT, previous)])

    def signal_close(self, event):
        with self.fixture(0) as (owner, descriptor), self.actual_sigint() as delivered:
            fired = []
            previous = sys.getprofile()
            def profile(frame, current_event, function):
                if current_event == event and getattr(function, "__self__", None) is owner and function.__name__ == "close":
                    fired.append((current_event, owner.closed))
                    os.kill(os.getpid(), signal.SIGINT)
            primary = None
            try:
                sys.setprofile(profile)
                with self.assertRaises(KeyboardInterrupt):
                    owner.close()
            except BaseException as error:
                primary = error
            finally:
                finish(primary, [lambda: sys.setprofile(previous)])
            self.assertEqual(delivered, [signal.SIGINT])
            self.assertEqual(fired, [(event, event == "c_return")])
            if event == "c_call":
                self.assertFalse(owner.closed)
                self.assertEqual(os.pread(descriptor, 64, 0), b"original bytes")
                stats = self.snapshot()
                self.assertEqual((stats["phase"], stats["underlying_calls"]), (1, 0))
            else:
                owner.close()
                stats = self.consumed(owner, descriptor, 0)
            self.witnesses[self._testMethodName] = dict(stats, signal_delivery=delivered, native_event=fired)

    def test_actual_SIGINT_pre_C_entry_keeps_exact_live_owner(self):
        self.signal_close("c_call")

    def test_actual_SIGINT_post_C_return_preserves_same_number_replacement(self):
        self.signal_close("c_return")

    def test_actual_SIGINT_journaled_open_before_result_store_preserves_explicit_close_status(self):
        for code in (0, errno.EIO, errno.EINTR):
            with self.subTest(cleanup_errno=code):
                self.journaled_open_interrupt(code)

    def journaled_open_interrupt(self, code):
        owner = self.new_slot()
        def acquire():
            result = owner.open(self.original, os.O_RDONLY)
            return result
        store = next(i.offset for i in dis.get_instructions(acquire) if i.opname == "STORE_FAST" and i.argval == "result")
        expected_identity = self.original.stat().st_dev, self.original.stat().st_ino
        fired, acquired = [], []
        def witness(event_code, offset):
            if event_code is acquire.__code__ and offset == store:
                fired.append(offset)
                acquired.extend(fd for fd, values in descriptors().items() if values[:2] == expected_identity)
                self.assertEqual(len(acquired), 1)
                self.arm(acquired[0], code)
                os.kill(os.getpid(), signal.SIGINT)
        with self.actual_sigint() as delivered:
            primary = None
            try:
                if hasattr(sys, "monitoring"):
                    monitor = sys.monitoring
                    tool = next((i for i in range(6) if monitor.get_tool(i) is None), None)
                    self.assertIsNotNone(tool)
                    monitor.use_tool_id(tool, "tensorfold-fd-actual-SIGINT-before-store")
                    monitor_primary = None
                    try:
                        monitor.register_callback(tool, monitor.events.INSTRUCTION, witness)
                        monitor.set_local_events(tool, acquire.__code__, monitor.events.INSTRUCTION)
                        with self.assertRaises(KeyboardInterrupt) as interrupted:
                            acquire()
                    except BaseException as error:
                        monitor_primary = error
                    finally:
                        finish(monitor_primary, [
                            lambda: monitor.set_local_events(tool, acquire.__code__, 0),
                            lambda: monitor.register_callback(tool, monitor.events.INSTRUCTION, None),
                            lambda: monitor.free_tool_id(tool),
                        ])
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
                    trace_primary = None
                    try:
                        current.f_trace_opcodes = True
                        sys.settrace(trace)
                        with self.assertRaises(KeyboardInterrupt) as interrupted:
                            acquire()
                    except BaseException as error:
                        trace_primary = error
                    finally:
                        finish(trace_primary, [lambda: sys.settrace(previous),
                                               lambda: setattr(current, "f_trace_opcodes", previous_opcodes)])
                self.assertEqual(fired, [store])
                self.assertEqual(delivered, [signal.SIGINT])
                self.assertTrue(any(slot is owner for slot in self.journal))
                self.assertFalse(owner.closed)
                self.assertEqual(owner.fileno(), acquired[0])
                self.assertEqual(os.pread(acquired[0], 64, 0), b"original bytes")
                self.assertEqual(self.snapshot()["underlying_calls"], 0)
                if code:
                    with self.assertRaises(OSError) as cleanup:
                        owner.close()
                    self.assertEqual(cleanup.exception.errno, code)
                    with self.assertRaises(KeyboardInterrupt) as transported:
                        raise_failures(interrupted.exception, [cleanup.exception])
                    self.assertIs(transported.exception, interrupted.exception)
                    self.assertIs(BaseException.__cause__.__get__(transported.exception), cleanup.exception)
                else:
                    owner.close()
                stats = self.consumed(owner, acquired[0], code)
                owner.close()
                self.witnesses[self._testMethodName + ":" + str(code)] = dict(
                    stats, signal_delivery=delivered, store_instruction=store,
                    closed_slot_journaled_before_open=True, explicit_operation_close_error=bool(code),
                    unraisable_operation_credit=False)
            except BaseException as error:
                primary = error
            finally:
                finish(primary, [lambda: self.cleanup(owner)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("provider", "source", "shim", "output"):
        parser.add_argument("--" + name, required=True, type=Path)
    for name in ("provider-sha256", "source-sha256", "shim-sha256"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("qualification output must be a fresh Root-owned file")
    Controls.api, Controls.shim, metadata = load_suppliers(args)
    before = descriptors()
    result = unittest.TextTestRunner(verbosity=2).run(unittest.defaultTestLoader.loadTestsFromTestCase(Controls))
    gc.collect()
    after = descriptors()
    report = {
        "schema": 1, "status": "PASS" if result.wasSuccessful() and before == after else "FAIL",
        "supplier": metadata, "cases": result.testsRun, "witnesses": Controls.witnesses,
        "failures": result.failures, "errors": result.errors, "skipped": result.skipped,
        "FDs_before": before, "FDs_after": after, "FD_identity_set_preserved": before == after,
        "scope": "actual Linux close fault/reuse and real process SIGINT at deterministic C/opcode boundaries",
        "unrun_or_excluded": ["macOS/XNU", "other ABI/toolchains/architectures", "foreign pthread cancellation", "random signal stress", "canonical wheel RECORD/model/performance gates"],
    }
    # unittest failures contain TestCase objects; render only their names/logs.
    for key in ("failures", "errors", "skipped"):
        report[key] = [(str(case), message) for case, message in report[key]]
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())

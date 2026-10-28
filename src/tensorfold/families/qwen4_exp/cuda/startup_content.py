"""Opt-in startup-owned content hashing; no persistent hash authority."""
from concurrent.futures import CancelledError
import os
import platform
import sys
import threading
import time

from tensorfold.cleanup import raise_failures

ENV = "TENSORFOLD_STARTUP_CONTENT_OVERLAP"
WORKERS_ENV = "TENSORFOLD_STARTUP_CONTENT_WORKERS"
STACK_LIMIT = 64 << 20
HASH_WORKING_BYTES = 8 << 20
MAX_ARTIFACT_FILES = 64
MAX_INVENTORY_TEXT_BYTES = 64 << 10


def _validate_inventory(files):
    if type(files) is not dict or not 0 < len(files) <= MAX_ARTIFACT_FILES:
        raise ValueError("startup capture requires its bounded original preload inventory")
    text = 0
    for name, path in files.items():
        values = (name, str(path))
        if (len(name.encode("utf-8")) > 4096
                or any(not value.isascii() or any(not 32 <= ord(char) <= 126 for char in value) for value in values)):
            raise ValueError("startup capture cannot prove its8MiB working bound for this inventory text")
        text += sum(map(len, values))
    if text > MAX_INVENTORY_TEXT_BYTES:
        raise ValueError("startup capture inventory exceeds its64KiB text working bound")
    return files


class StartupContentPlan:
    """One process-owned configuration; callers exclude thread-default changes."""
    def __init__(self, files, *, workers=1):
        self.files = _validate_inventory(files)
        if type(workers) is not int or workers not in (1, 3) or workers > len(files):
            raise ValueError("startup content workers require one or three nonempty file partitions")
        self.workers = workers
        from tensorfold._fd_owner import _thread_stack_layout

        if (sys.platform != "linux" or sys.implementation.name != "cpython"
                or sys.version_info[:2] != (3, 12) or platform.libc_ver()[0] != "glibc"):
            raise ValueError("startup content overlap requires the pinned Linux/glibc CPython3.12 process")
        self.python_stack_setting = threading.stack_size()
        stack, guard = _thread_stack_layout(self.python_stack_setting)
        if not 0 < stack <= STACK_LIMIT or not 0 <= guard <= STACK_LIMIT:
            raise ValueError("startup content thread stack cannot fit its64MiB stack bound")
        self.guard_bytes = guard
        # The stack ceiling, actual guard and hash/metadata working bound are
        # charged globally once before starting the sole rank0 worker.
        self.host_bytes = workers * (STACK_LIMIT + guard + HASH_WORKING_BYTES)

    def record(self):
        return {"global_host_bytes": self.host_bytes, "workers_global": self.workers,
                "stack_upper_bytes": STACK_LIMIT, "guard_bytes": self.guard_bytes,
                "hash_and_metadata_working_bytes": HASH_WORKING_BYTES,
                "hash_and_metadata_working_bytes_total": self.workers * HASH_WORKING_BYTES,
                "stack_and_guard_bytes_total": self.workers * (STACK_LIMIT + self.guard_bytes),
                "scope": "startup through manifest binding; included in loading and steady host commitment"}

    def start(self, engine, manifest):
        from .execution_identity import _capture, _opened_stats

        if threading.stack_size() != self.python_stack_setting:
            raise ValueError("Python thread stack configuration changed after startup admission")
        from tensorfold._fd_owner import _thread_stack_layout
        stack, guard = _thread_stack_layout(self.python_stack_setting)
        if not 0 < stack <= STACK_LIMIT or guard > self.guard_bytes:
            raise ValueError("pthread default stack configuration changed after startup admission")
        files = engine._execution_preload.get("capture_files")
        if files is not self.files:
            raise ValueError("startup capture inventory owner changed after admission")
        before = engine._execution_preload["artifacts"]
        if _opened_stats(files) != before:
            raise ValueError("startup artifact closure exceeds its funded metadata bound or changed")
        if self.workers == 1:
            return StartupContentCapture(self, lambda cancelled: _capture(
                files, before, manifest["artifacts"], cancelled=cancelled)[0])
        partitions = partition_files(files, before, self.workers)
        operations = []
        for group in partitions:
            selected_files = {name: files[name] for name in group}
            selected_stats = {name: before[name] for name in group}
            selected_expected = {name: manifest["artifacts"][name] for name in group}
            operations.append(lambda cancelled, f=selected_files, b=selected_stats, e=selected_expected:
                              _capture(f, b, e, cancelled=cancelled)[0])
        return StartupContentGroup(self, operations, partitions)


class StartupContentCapture:
    """One joined non-daemon worker; source guards never consume pending work."""
    def __init__(self, plan, operation):
        self.plan, self.operation = plan, operation
        self.cancel = threading.Event()
        self.operation_retired = threading.Event()
        self.join_returned = False
        self.authority = self.error = None
        self.finish_errors, self.wait_errors = [None], [None]
        self.cancel_observed = self.error_observed = False
        self.started_ns = time.monotonic_ns()
        self.finished_ns = self.wait_started_ns = self.wait_finished_ns = None
        self.start_attempted = self.closed = False
        self.thread = threading.Thread(target=self._run, name="tf-startup-content", daemon=False)

    def start(self):
        if self.closed or self.start_attempted:
            raise RuntimeError("startup content worker is one-shot and cannot start after retirement")
        self.start_attempted = True
        self.thread.start()

    def _run(self):
        try:
            from tensorfold._fd_owner import _current_thread_stack_layout
            stack, guard = _current_thread_stack_layout()
            if not 0 < stack <= STACK_LIMIT or guard > self.plan.guard_bytes:
                raise ValueError("actual content worker exceeds its admitted stack/guard")
            self.authority = self.operation(self.cancel.is_set)
        except BaseException as error:
            self.error = error
            self.cancel_observed = isinstance(error, CancelledError) and self.cancel.is_set()
        finally:
            try:
                self.operation = None
                self.finished_ns = time.monotonic_ns()
            except BaseException as error:
                self.finish_errors[0] = error
            finally:
                # Thread.join/is_alive can lie after signal interruption on
                # the pinned runtime. Publish owned retirement even if timing
                # diagnostics fail, retaining that secondary in its fixed slot.
                self.operation_retired.set()

    def _raise_outcome(self, primary):
        first, second = self.finish_errors[0], self.wait_errors[0]
        try:
            errors = [error for error in (first, second) if error is not None]
        except BaseException as allocation:
            if primary is not None:
                raise primary from allocation
            if first is not None:
                raise first from allocation
            if second is not None:
                raise second from allocation
            raise
        raise_failures(primary, errors)

    def result(self):
        if self.closed:
            raise RuntimeError("startup content capture already retired")
        try:
            self.wait_started_ns = time.monotonic_ns()
        except BaseException as error:
            self.wait_errors[0] = error
        self.thread.join()
        self.join_returned = True
        self.operation_retired.wait()
        try:
            self.wait_finished_ns = time.monotonic_ns()
        except BaseException as error:
            if self.wait_errors[0] is None:
                self.wait_errors[0] = error
        if self.error is not None or self.finish_errors[0] is not None or self.wait_errors[0] is not None:
            self.error_observed = True
            self._raise_outcome(self.error)
        if self.authority is None:
            raise RuntimeError("joined content worker produced no issued authority")
        return self.authority

    def close(self):
        if self.closed:
            return
        self.cancel.set()
        if self.start_attempted:
            # A raised Thread.start does not prove native work was never
            # accepted. Join is the authority; an unjoinable interrupted
            # acquisition remains retained by engine rollback for teardown.
            self.thread.join()
            self.join_returned = True
            self.operation_retired.wait()
        self.closed = True
        if not self.error_observed:
            primary = self.error if not self.cancel_observed else None
            if primary is not None or self.finish_errors[0] is not None or self.wait_errors[0] is not None:
                self.error_observed = True
                self._raise_outcome(primary)

    def authorities(self):
        return (self.result(),)

    def record(self):
        return {"started_monotonic_ns": self.started_ns, "finished_monotonic_ns": self.finished_ns,
                "bytes_hashed": None if self.authority is None else self.authority.bytes_read,
                "registration_wait_started_monotonic_ns": self.wait_started_ns,
                "registration_wait_finished_monotonic_ns": self.wait_finished_ns,
                "registration_wait_ns": (None if self.wait_started_ns is None or self.wait_finished_ns is None
                                         else self.wait_finished_ns - self.wait_started_ns),
                "worker_operation_retired": self.operation_retired.is_set(),
                "thread_join_returned": self.join_returned,
                "native_stack_reclamation": "not inferred from Python join; fixed host commitment retained through engine",
                "scope": "current-process content authority; no disk/stat-only hash cache"}


def partition_files(files, before, workers):
    """Independent whole files only, greedy size balance with stable tie order."""
    bins, sizes = [[] for _ in range(workers)], [0] * workers
    for name in sorted(files, key=lambda name: (-before[name]["stat"][2], name)):
        chosen = min(range(workers), key=lambda index: (sizes[index], len(bins[index]), index))
        bins[chosen].append(name)
        sizes[chosen] += before[name]["stat"][2]
    if any(not group for group in bins):
        raise ValueError("startup content partitions must be nonempty")
    return tuple(tuple(sorted(group)) for group in bins)


def _raise_slots(primary, slots, extra=()):
    """Retain all six fixed originals through diagnostic allocation failure."""
    first = slots[0] if len(slots) > 0 else None
    second = slots[1] if len(slots) > 1 else None
    third = slots[2] if len(slots) > 2 else None
    fourth = extra[0] if len(extra) > 0 else None
    fifth = extra[1] if len(extra) > 1 else None
    sixth = extra[2] if len(extra) > 2 else None
    try:
        errors = [error for error in (first, second, third, fourth, fifth, sixth)
                  if error is not None and error is not primary]
    except BaseException as allocation:
        if primary is not None:
            raise primary from allocation
        if first is not None:
            raise first from allocation
        if second is not None:
            raise second from allocation
        if third is not None:
            raise third from allocation
        if fourth is not None:
            raise fourth from allocation
        if fifth is not None:
            raise fifth from allocation
        if sixth is not None:
            raise sixth from allocation
        raise
    raise_failures(primary, errors)


def _expected_cancel(error, capture):
    return error is capture.error and isinstance(error, CancelledError) and capture.cancel_observed


def _pure_expected_cancel(error, capture):
    return (_expected_cancel(error, capture)
            and capture.finish_errors[0] is None and capture.wait_errors[0] is None
            and BaseException.__cause__.__get__(error) is None
            and BaseException.__context__.__get__(error) is None)


def _primary_group_result(errors, captures):
    first, second, third = errors
    # Preserve a genuine partition/parent failure ahead of sibling cancellation
    # carrying an independently reported diagnostic/transport secondary.
    if first is not None and not _expected_cancel(first, captures[0]):
        return first
    if second is not None and not _expected_cancel(second, captures[1]):
        return second
    if third is not None and not _expected_cancel(third, captures[2]):
        return third
    if first is not None:
        return first
    if second is not None:
        return second
    return third


class StartupContentGroup:
    """Exactly three pre-owned Captures; no executor or chunk-hash combination."""
    def __init__(self, plan, operations, partitions):
        if len(operations) != 3 or len(partitions) != 3:
            raise ValueError("parallel startup capture requires exactly three file partitions")
        self.plan, self.partitions = plan, partitions
        self.start_attempted = self.closed = False
        self.result_errors, self.close_errors, self.cancel_errors = [None]*3, [None]*3, [None]*3
        self.cancel_errors_observed = False
        self.notify_errors = ([None]*3, [None]*3, [None]*3)
        self.values, self.hash_started_ns = [None]*3, [None]*3
        self.captures = tuple(StartupContentCapture(plan, operation) for operation in operations)
        events = tuple(capture.cancel for capture in self.captures)
        # All Capture owners and their journals exist before any Thread.start.
        # A genuine hash failure immediately cancels siblings between reads.
        for index, capture in enumerate(self.captures):
            original, errors = capture.operation, self.notify_errors[index]
            starts = self.hash_started_ns
            def operation(cancelled, original=original, errors=errors, index=index):
                try:
                    starts[index] = time.monotonic_ns()
                    return original(cancelled)
                except BaseException as primary:
                    for at, event in enumerate(events):
                        try:
                            event.set()
                        except BaseException as error:
                            errors[at] = error
                    _raise_slots(primary, errors)
            capture.operation = operation

    def start(self):
        if self.closed or self.start_attempted:
            raise RuntimeError("startup content group is one-shot")
        self.start_attempted = True
        for capture in self.captures:
            capture.start()

    def _cancel_all(self):
        if self.cancel_errors_observed:
            for index in range(3):
                self.cancel_errors[index] = None
            self.cancel_errors_observed = False
        for index, capture in enumerate(self.captures):
            try:
                capture.cancel.set()
            except BaseException as error:
                if self.cancel_errors[index] is None:
                    self.cancel_errors[index] = error

    def authorities(self):
        if self.closed:
            raise RuntimeError("startup content group already retired")
        for index, capture in enumerate(self.captures):
            try:
                self.values[index] = capture.result()
            except BaseException as error:
                # Sibling cancellations are expected only when their own
                # Capture observed its cancellation. Genuine hash/interrupt/
                # finalizer failures remain independently reportable.
                if not _pure_expected_cancel(error, capture):
                    self.result_errors[index] = error
                self._cancel_all()
        primary = _primary_group_result(self.result_errors, self.captures)
        self.cancel_errors_observed = True
        _raise_slots(primary, self.result_errors, self.cancel_errors)
        for value in self.values:
            if value is None:
                raise CancelledError("all startup content partitions were cancelled")
        return tuple(self.values)

    def close(self):
        if self.closed:
            return
        self._cancel_all()
        all_closed = True
        for index, capture in enumerate(self.captures):
            self.close_errors[index] = None
            try:
                capture.close()
            except BaseException as error:
                self.close_errors[index] = error
            all_closed = all_closed and capture.closed
        self.closed = all_closed
        primary = None
        for error in self.close_errors:
            if error is not None:
                primary = error
                break
        self.cancel_errors_observed = True
        _raise_slots(primary, self.close_errors, self.cancel_errors)

    def record(self):
        records = [{**capture.record(), "hash_operation_started_monotonic_ns": self.hash_started_ns[index],
                    "thread_native_id": capture.thread.native_id}
                   for index, capture in enumerate(self.captures)]
        starts = [row["registration_wait_started_monotonic_ns"] for row in records]
        ends = [row["registration_wait_finished_monotonic_ns"] for row in records]
        return {"workers_global": 3, "partitions": [list(group) for group in self.partitions], "workers": records,
                "started_monotonic_ns": min(row["started_monotonic_ns"] for row in records),
                "finished_monotonic_ns": max(row["finished_monotonic_ns"] for row in records),
                "bytes_hashed": sum(row["bytes_hashed"] for row in records),
                "registration_wait_ns": None if any(value is None for value in (*starts, *ends)) else max(ends)-min(starts),
                "worker_operation_retired": all(row["worker_operation_retired"] for row in records),
                "native_stack_reclamation": "not inferred; three fixed worker commitments retained through engine",
                "scope": "independent whole-file SHA256 authorities issued in this process; no persisted hash cache or combined chunk digest"}


def prepare(engine, manifest):
    selected = os.environ.get(ENV)
    if selected is None:
        if os.environ.get(WORKERS_ENV) is not None:
            raise ValueError(WORKERS_ENV + " requires enabled startup content overlap")
        return None
    if selected != "1":
        raise ValueError(ENV + " accepts only1 when enabled; unset retains serial hashing")
    if (manifest is None or getattr(engine, "_profile_build", None) is not None
            or engine.tp != 8 or engine.method_geometry.method != "mtp"
            or getattr(engine, "media_profile", None) is not None
            or engine.sm70_profile not in ("source-owner-pieces-trial", "source-owner-target-greedy-trial",
                                          "source-owner-target-guard-trial", "source-owner-target-prelude-trial", "source-owner-target-dense-trial")):
        raise ValueError("startup content overlap requires an explicit private TP8 owner MTP manifest")
    files = getattr(engine, "_execution_preload", {}).get("capture_files")
    if type(files) is not dict:
        raise ValueError("startup content overlap requires the original manifest preload inventory")
    workers = os.environ.get(WORKERS_ENV, "1")
    if workers not in ("1", "3"):
        raise ValueError(WORKERS_ENV + " accepts only1 or3")
    return StartupContentPlan(files, workers=int(workers))

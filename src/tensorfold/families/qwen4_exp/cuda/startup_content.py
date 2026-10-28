"""Opt-in startup-owned content hashing; no persistent hash authority."""
from concurrent.futures import CancelledError
import os
import platform
import sys
import threading
import time

from tensorfold.cleanup import raise_failures

ENV = "TENSORFOLD_STARTUP_CONTENT_OVERLAP"
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
    def __init__(self, files):
        self.files = _validate_inventory(files)
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
        self.host_bytes = STACK_LIMIT + guard + HASH_WORKING_BYTES

    def record(self):
        return {"global_host_bytes": self.host_bytes, "workers_global": 1,
                "stack_upper_bytes": STACK_LIMIT, "guard_bytes": self.guard_bytes,
                "hash_and_metadata_working_bytes": HASH_WORKING_BYTES,
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
        return StartupContentCapture(self, lambda cancelled: _capture(
            files, before, manifest["artifacts"], cancelled=cancelled)[0])


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


def prepare(engine, manifest):
    selected = os.environ.get(ENV)
    if selected is None:
        return None
    if selected != "1":
        raise ValueError(ENV + " accepts only1 when enabled; unset retains serial hashing")
    if (manifest is None or getattr(engine, "_profile_build", None) is not None
            or engine.tp != 8 or engine.method_geometry.method != "mtp"
            or getattr(engine, "media_profile", None) is not None
            or engine.sm70_profile not in ("source-owner-pieces-trial", "source-owner-target-greedy-trial",
                                          "source-owner-target-guard-trial", "source-owner-target-prelude-trial",
                                          "source-owner-attention-graphs-trial")):
        raise ValueError("startup content overlap requires an explicit private TP8 owner MTP manifest")
    files = getattr(engine, "_execution_preload", {}).get("capture_files")
    if type(files) is not dict:
        raise ValueError("startup content overlap requires the original manifest preload inventory")
    return StartupContentPlan(files)

"""Opt-in startup-owned content hashing; no persistent hash authority."""
from concurrent.futures import CancelledError
import os
import platform
import sys
import threading
import time

ENV = "TENSORFOLD_STARTUP_CONTENT_OVERLAP"
STACK_LIMIT = 64 << 20
HASH_WORKING_BYTES = 8 << 20
MAX_ARTIFACT_FILES = 64


class StartupContentPlan:
    """One process-owned configuration; callers exclude thread-default changes."""
    def __init__(self):
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
        from .execution_identity import _capture, _opened_stats, artifact_files

        if threading.stack_size() != self.python_stack_setting:
            raise ValueError("Python thread stack configuration changed after startup admission")
        from tensorfold._fd_owner import _thread_stack_layout
        stack, guard = _thread_stack_layout(self.python_stack_setting)
        if not 0 < stack <= STACK_LIMIT or guard > self.guard_bytes:
            raise ValueError("pthread default stack configuration changed after startup admission")
        files, _ = artifact_files(engine)
        before = engine._execution_preload["artifacts"]
        if (not 0 < len(files) <= MAX_ARTIFACT_FILES or _opened_stats(files) != before
                or any(len(name.encode("utf-8")) > 4096 for name in files)):
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
        self.cancel_observed = self.error_observed = False
        self.started_ns = time.monotonic_ns()
        self.finished_ns = None
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
            self.operation = None
            self.finished_ns = time.monotonic_ns()
            # Thread.join/is_alive can lie after signal interruption on the
            # pinned CPython runtime. Only this operation-owned final latch
            # proves file consumers and their cleanup have retired.
            self.operation_retired.set()

    def result(self):
        if self.closed:
            raise RuntimeError("startup content capture already retired")
        self.thread.join()
        self.join_returned = True
        self.operation_retired.wait()
        if self.error is not None:
            self.error_observed = True
            raise self.error
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
        if self.error is not None and not self.error_observed and not self.cancel_observed:
            self.error_observed = True
            raise self.error

    def record(self):
        return {"started_monotonic_ns": self.started_ns, "finished_monotonic_ns": self.finished_ns,
                "bytes_hashed": None if self.authority is None else self.authority.bytes_read,
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
            or engine.sm70_profile not in ("source-owner-pieces-trial", "source-owner-target-greedy-trial")):
        raise ValueError("startup content overlap requires an explicit private TP8 owner MTP manifest")
    return StartupContentPlan()

"""Per-rank startup phase timeline of the DeepSeek-V4.1 SM70 engine (one log line per rank).

Each phase records wall seconds, this process's CPU seconds (every thread: ``time.process_time``) and the bytes it
read from storage (``/proc/self/io`` read_bytes: block-layer reads, so page-cache hits count 0). CPU near wall x
threads means CPU-bound work; storage bytes show a cold page cache. Phases may overlap (a background phase runs
beside the foreground ones); ``since`` is the phase's start relative to the process start (interpreter launch).
"""

from __future__ import annotations

import os
import time
from contextlib import contextmanager


def _read_bytes() -> int:
    try:
        with open("/proc/self/io", "rb") as f:
            for line in f:
                if line.startswith(b"read_bytes:"):
                    return int(line.split()[1])
    except OSError:
        pass
    return 0


def _process_age() -> float:
    """Seconds since this process started (``/proc/self/stat`` start time against the boot clock)."""
    try:
        with open("/proc/self/stat", "rb") as f:
            fields = f.read().rsplit(b")", 1)[1].split()
        ticks = int(fields[19])           # field 22 (starttime), counted after the "(comm)" field
        return time.clock_gettime(time.CLOCK_BOOTTIME) - ticks / os.sysconf("SC_CLK_TCK")
    except (OSError, ValueError, IndexError):
        return 0.0


class Timeline:
    def __init__(self, rank: int):
        self.rank = rank
        self.origin = time.perf_counter() - _process_age()
        self.rows: list[tuple[str, float, float, float, int]] = []   # name, since, wall, cpu, storage bytes

    @contextmanager
    def phase(self, name: str):
        """Time the block; it may set ``note[0]`` (a detail string shown after the name)."""
        start, cpu, read = time.perf_counter(), time.process_time(), _read_bytes()
        note = [""]
        try:
            yield note
        finally:
            label = f"{name} ({note[0]})" if note[0] else name
            self.add(label, start, time.perf_counter() - start, time.process_time() - cpu, _read_bytes() - read)

    def add(self, name, start, wall, cpu=0.0, read=0):
        self.rows.append((name, start - self.origin, wall, cpu, read))

    def line(self) -> str:
        parts = []
        for name, since, wall, cpu, read in self.rows:
            extra = f" cpu {cpu:.1f}" if cpu >= 0.05 else ""
            if read >= 1 << 26:
                extra += f" disk {read / 2**30:.1f}G"
            parts.append(f"{name} @{since:.1f} {wall:.1f}s{extra}")
        return (f"[tensorfold] rank {self.rank} startup ({time.perf_counter() - self.origin:.1f} s since process "
                f"start): " + "; ".join(parts))


# The lane's native extensions (torch cpp_extension JIT builds, cached under TORCH_EXTENSIONS_DIR).
EXTENSIONS = ("tensorfold.cuda.small_gather", "tensorfold.cuda.sm70.ds41_dense", "tensorfold.cuda.sm70.onecat_ds41",
              "tensorfold.cuda.sm70.ds41_decode_moe", "tensorfold.cuda.sm70.ds41_lattice_moe",
              "tensorfold.cuda.sm70.ds41_sparse_attention")


def _extension(name):
    import importlib
    started = time.perf_counter()
    importlib.import_module(name).extension()
    return time.perf_counter() - started


class Prebuild:
    """Builds (first start) or loads every extension of the lane at once, one thread each, beside the rest of the
    startup: a first start compiles them in parallel instead of one after another as the load reaches each.
    Every module's ``extension()`` is cached once built, so the caller must ``wait`` for a module before its first
    own use (a concurrent first call would build and import it twice). Worker threads are joined at interpreter exit,
    so a failing startup never leaves a build half-written."""

    def __init__(self, timeline: Timeline, names=EXTENSIONS):
        from concurrent.futures import ThreadPoolExecutor
        self.timeline, self.started = timeline, time.perf_counter()
        self.pool = ThreadPoolExecutor(len(names), thread_name_prefix="ds41-ext")
        self.jobs = {name: self.pool.submit(_extension, name) for name in names}
        self.done = False

    def wait(self, *names):
        """Re-raises a module's build failure."""
        for name in names or self.jobs:
            self.jobs[name].result()
        if not names and not self.done:
            self.done = True
            self.pool.shutdown(wait=True)
            slowest = max(self.jobs, key=lambda n: self.jobs[n].result())
            self.timeline.add(f"extensions (slowest {slowest.rsplit('.', 1)[-1]} {self.jobs[slowest].result():.1f})",
                              self.started, time.perf_counter() - self.started)


class ExpertStartup:
    """Opt-in resource ownership until the DeepSeek engine is usable."""

    def __init__(self):
        self.shards = None
        self.history = None
        self.observer = None
        self.mapper = None
        self.engram_job = None
        self.engram_fd = None
        self.engram = None

    def map_engram(self, fd, construct):
        from concurrent.futures import ThreadPoolExecutor

        # This descriptor is independent of Shards. The worker gives HostEngram a
        # separate descriptor of its own and never borrows Shards' fd table.
        self.engram_fd = fd
        self.mapper = ThreadPoolExecutor(1, thread_name_prefix="ds41-engram")

        def load():
            from tensorfold.cleanup import rollback
            from tensorfold.cuda.cohort import notify_failure

            try:
                return construct(lambda _path: os.dup(fd))
            except BaseException as primary:
                rollback(self, primary, notify_failure)

        self.engram_job = self.mapper.submit(load)
        return self.engram_job

    def retire_mapper(self, primary=None):
        from tensorfold.cleanup import finish

        if self.mapper is not None:
            self.mapper.shutdown(wait=True, cancel_futures=True)
            self.mapper = None

        def observe():
            job, self.engram_job = self.engram_job, None
            if job is not None and not job.cancelled():
                error = job.exception()
                if error is not None and error is not primary:
                    raise error

        def close_lease():
            if self.engram_fd is not None:
                fd, self.engram_fd = self.engram_fd, None
                os.close(fd)

        finish((observe, close_lease))

    def abort(self, engine, primary=None):
        from tensorfold.cleanup import finish
        from tensorfold.cuda.cohort import notify_failure

        self.engine = engine

        def retire_native():
            if self.mapper is not None:
                return  # a failed join retains the possibly live mapper and mapping
            scheduler = getattr(engine, "scheduler", None)
            if scheduler is not None:
                scheduler.close()
            if self.engram is not None or getattr(engine, "graphs", None) is not None:
                import torch

                torch.cuda.synchronize()
                engine.graphs = None
                if self.engram is not None:
                    self.engram.close()

        operations = [notify_failure]
        if self.shards is not None:
            operations.append(self.shards.close)
        if self.observer is not None:
            operations.append(self.observer.abort_startup)
        elif self.history is not None:
            operations.append(self.history.close)
        # The mapper can still be waiting in HostEngram's store exchange.
        # Notify the process supervisor before waiting for that borrower.
        operations.extend((lambda: self.retire_mapper(primary), retire_native))
        finish(operations)

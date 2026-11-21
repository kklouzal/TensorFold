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

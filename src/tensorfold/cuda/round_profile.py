"""Opt-in timing of concurrent decode rounds: ``TENSORFOLD_ROUND_PROFILE=<path>`` appends one JSON line a round.

Off (unset), callers hold ``None`` and each mark costs one ``is not None`` test: no events, allocation or I/O.
On, a mark records the host time and, where the GPU timeline matters, a CUDA event from a preallocated ring.
A round's events are read at the next round's start, after that round's own host waits, so profiling adds no
synchronization. Records leave the engine thread through ``record_writer.JsonLines``'s bounded queue (the oldest
record is dropped, and counted, when the writer falls behind); a daemon thread writes them.

A record's ``phases`` are ``[name, host_ms, gpu_ms]`` from the previous mark: ``gpu_ms`` is the GPU timeline's
progress between consecutive event marks (busy plus idle), ``null`` for host-only marks.
"""

from __future__ import annotations

import os
import time

import torch

from tensorfold.cuda.record_writer import JsonLines

ENV = "TENSORFOLD_ROUND_PROFILE"
EVENTS = 96            # event marks a round may record; later marks in the round keep host time only
IN_FLIGHT = 4          # rounds whose events may still be pending when the next round records its own


class RoundProfile:
    def __init__(self, path: str, *, queue_size: int = 4096) -> None:
        self.ring = [torch.cuda.Event(enable_timing=True) for _ in range(EVENTS * IN_FLIGHT)]
        self.slot = 0                      # the ring segment this round records into
        self.used = 0                      # events this round used in its segment
        self.marks: list[tuple[str, int, int]] = []
        self.pending: list[tuple[dict, list[tuple[str, int, int]]]] = []
        self.unresolved = 0                # rounds written without GPU times (events not ready in time)
        self.writer = JsonLines(path, name="tensorfold-round-profile", stamp=self._stamp, queue_size=queue_size)

    @classmethod
    def from_env(cls) -> RoundProfile | None:
        path = os.environ.get(ENV, "")
        return cls(path) if path and torch.cuda.is_available() else None

    def begin(self) -> None:
        self._resolve(free=False)
        self.marks, self.used = [], 0
        self.mark("begin", gpu=True)

    def mark(self, name: str, gpu: bool = False) -> None:
        index = -1
        if gpu and self.used < EVENTS:
            index = self.slot * EVENTS + self.used
            self.used += 1
            self.ring[index].record()
        self.marks.append((name, time.perf_counter_ns(), index))

    def end(self, **fields) -> None:
        self.mark("end", gpu=True)
        fields["t"] = time.time()
        self.pending.append((fields, self.marks))
        self.marks = []
        self.slot = (self.slot + 1) % IN_FLIGHT
        if len(self.pending) >= IN_FLIGHT:           # the oldest round's ring segment is next: emit it now
            self._resolve(free=True)

    def _resolve(self, free: bool) -> None:
        """Emit finished rounds in order; ``free``: also the oldest unfinished one whose segment is reused next."""

        while self.pending:
            fields, marks = self.pending[0]
            last = next((m[2] for m in reversed(marks) if m[2] >= 0), -1)
            ready = last < 0 or self.ring[last].query()
            if not ready and not (free and len(self.pending) >= IN_FLIGHT):
                return
            self.pending.pop(0)
            phases, prev_ns, prev_ev = [], marks[0][1], marks[0][2]
            for name, ns, ev in marks[1:]:
                gpu = None
                if ev >= 0 and ready:
                    if prev_ev >= 0:
                        gpu = round(self.ring[prev_ev].elapsed_time(self.ring[ev]), 4)
                    prev_ev = ev
                phases.append([name, round((ns - prev_ns) / 1e6, 4), gpu])
                prev_ns = ns
            if not ready:
                self.unresolved += 1
            fields["wall_ms"] = round((marks[-1][1] - marks[0][1]) / 1e6, 4)
            fields["phases"] = phases
            self.writer.put(fields)

    def _stamp(self, record: dict) -> None:
        record["dropped"] = self.writer.dropped
        record["unresolved"] = self.unresolved

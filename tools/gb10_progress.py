"""Existing single-process GB10 heartbeat and hot gate with owned stream IO.

State, clocks, cadence, JSON format and refusal policy match campaign_bench.
Only stream ownership changes: a read/write failure survives an independent
close failure. A multi-failure outcome remains fatal and never admits a prompt.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import ClassVar

from gb10_io import opened_file, raise_failures


class Progress:
    """The heartbeat a supervisor (the GB10 runner's watchdog) reads while a tool talks to a server: one small JSON
    file, replaced atomically, holding the current phase ("prefill" until the first generated text, reasoning or
    tool-call delta of a stream, "decode" after it, "wait" for a non-streamed reply, "idle" between requests), when
    that phase started, the last event's time, the request in flight (``request``, ``prompt_tokens``, and for "wait"
    ``max_tokens``; kept across that request's phases) and its chunk count. Facts only: the supervisor owns the
    stall policy. Writes are rate-limited to one per second except at phase changes. One per process (tools here are
    single-threaded CLIs): ``configure`` sets the path and resets the state, ``configure(None)`` turns it off again;
    unconfigured, every call is a no-op."""

    path: ClassVar[str | None] = None
    state: ClassVar[dict] = {}
    written: ClassVar[float] = 0.0
    CARRIED: ClassVar[tuple[str, ...]] = ("request", "prompt_tokens", "max_tokens")   # one request's identity

    @classmethod
    def configure(cls, path: str | None) -> None:
        cls.path, cls.state, cls.written = path, {}, 0.0
        cls.update("idle", force=True)

    @classmethod
    def update(cls, phase: str, *, force: bool = False, **fields) -> None:
        if cls.path is None:
            return
        now = time.time()
        if phase != cls.state.get("phase"):
            carried = {k: cls.state[k] for k in cls.CARRIED if k in cls.state and phase != "idle"}
            cls.state = {"phase": phase, "phase_since": now, **carried}
            force = True
        cls.state.update(fields, last_event=now, pid=os.getpid())
        if force or now - cls.written >= 1.0:
            temporary = f"{cls.path}.{os.getpid()}.tmp"
            with opened_file(temporary, "w", encoding="utf-8") as f:
                json.dump(cls.state, f)
            os.replace(temporary, cls.path)
            cls.written = now


class HotGate:
    """The campaign's hot-state gate (after the 2026-10-19 GB10 hang #3, a prefill burst from a hot decode state):
    before each new prompt a tool waits while the shared guard's hot-state file exists (gpu_guard: any zone above
    85 C or the GPU above 75 C, or a state it cannot read), at most ``max_s``; a host that stays hot is an error, never
    a prompt. ``configure`` takes the file (default the TFC_HOT_FLAG the guard exports; none: no gate, for tools run
    outside a guarded session); ``waited_s`` sums the waits. One per process (single-threaded CLIs), as Progress."""

    path: ClassVar[Path | None] = None
    max_s: ClassVar[float] = 600.0             # under the runner's 900 s idle allowance
    waited_s: ClassVar[float] = 0.0

    @classmethod
    def configure(cls, path: str | None) -> None:
        path = path or os.environ.get("TFC_HOT_FLAG")
        cls.path, cls.waited_s = (Path(path) if path else None), 0.0

    @classmethod
    def wait(cls) -> float:
        """Block until the host is out of the hot state; the seconds waited (0 without a gate)."""

        if cls.path is None:
            return 0.0
        began = time.monotonic()
        while True:
            try:
                cls.path.stat()
            except FileNotFoundError:
                break
            except OSError as error:
                raise RuntimeError(f"hot-state file unreadable: {type(error).__name__}: no new prompt") from error
            remaining = cls.max_s - (time.monotonic() - began)
            if remaining <= 0:
                opened, reason_error = False, None
                cleanup_errors = [None]
                try:
                    with opened_file(cls.path, "r") as stream:
                        opened = True
                        try:
                            reason = stream.read().strip()
                        except OSError as error:
                            # Apply optional-read recovery only after close succeeds.
                            reason_error = error
                except BaseException as error:
                    if opened:
                        if reason_error is not None:
                            cleanup_errors[0] = error
                            raise_failures(reason_error, cleanup_errors)
                        raise
                    if not isinstance(error, OSError):
                        raise
                    reason_error = error
                if isinstance(reason_error, FileNotFoundError):  # cooled before/during read; owner retired
                    break
                if reason_error is not None:
                    reason = f"hot-state reason unreadable: {type(reason_error).__name__}"
                    raise RuntimeError(f"the host stayed hot for {cls.max_s:.0f} s ({reason}): no new prompt") from reason_error
                raise RuntimeError(f"the host stayed hot for {cls.max_s:.0f} s ({reason}): no new prompt")
            Progress.update("idle")
            time.sleep(min(1.0, remaining))
        waited = time.monotonic() - began
        cls.waited_s += waited
        return waited

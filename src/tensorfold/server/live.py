"""One redrawn live throughput line, cleared by other output, off unless stdout is a terminal."""

from __future__ import annotations

import collections
import os
import shutil
import sys
import threading
import time
from typing import Any, Callable

from tensorfold.thread_work import ThreadWork

WINDOW = 2.0  # seconds a decode rate averages over, and a prefill chunk's rate stays shown
EVERY = 0.5  # seconds between redraws
CLEAR = "\r\x1b[2K"  # back to the line's start and erase it


class Meter:
    """Retain the last ``window`` seconds; sample a nonblocking monotonic clock under the state lock."""

    def __init__(self, window: float = WINDOW, clock: Callable[[], float] = time.monotonic) -> None:
        self.window, self.clock = float(window), clock
        self._events: collections.deque[tuple[float, int]] = collections.deque()
        self._lock = threading.Lock()

    def add(self, tokens: int) -> None:
        if tokens > 0:
            with self._lock:
                now = self.clock()
                self._prune(now)
                self._events.append((now, int(tokens)))

    def _prune(self, now: float) -> None:
        """Drop expired events while holding ``_lock``; an exact boundary stays."""

        while self._events and self._events[0][0] < now - self.window:
            self._events.popleft()

    def rate(self) -> float:
        with self._lock:
            now = self.clock()
            self._prune(now)
            return sum(n for _, n in self._events) / self.window


class ChunkRate:
    """Prefill: the newest prompt chunk's own tokens a second, shown for ``window`` seconds after it ends."""

    def __init__(self, window: float = WINDOW, clock: Callable[[], float] = time.monotonic) -> None:
        self.window, self.clock = float(window), clock
        self._last: tuple[float, float] | None = None  # (when it ended, tokens a second)

    def add(self, tokens: int, seconds: float) -> None:
        if tokens > 0 and seconds > 0:
            self._last = (self.clock(), tokens / seconds)

    def rate(self) -> float:
        last = self._last
        return last[1] if last is not None and self.clock() - last[0] <= self.window else 0.0


def status(scheduler: Any) -> str:
    """``[tensorfold] 3 connections (1 waiting) · decode 142 tok/s · prefill 1,210 tok/s``."""

    waiting = scheduler.waiting
    open_ = scheduler.active + len(scheduler.filling) + waiting
    line = f"[tensorfold] {open_} connection{'' if open_ == 1 else 's'}"
    if waiting:
        line += f" ({waiting} waiting)"
    return line + f" · decode {scheduler.decoded.rate():,.0f} tok/s · prefill {scheduler.prefilled.rate():,.0f} tok/s"


class LiveLine:
    """Redraws ``render()`` on the terminal's last line; writes through ``install``'s proxies clear it first."""

    def __init__(self, render: Callable[[], str], out: Any, every: float = EVERY) -> None:
        self.render, self.out, self.every = render, out, float(every)
        self._lock = threading.RLock()
        self._shown = False
        self._line_start = True  # the newest write ended its line: a redraw can't split a log line
        self._stop = threading.Event()
        self._work = ThreadWork(self._tick)
        self._thread = threading.Thread(target=self._work.run, name="tensorfold-live", daemon=True)
        self._saved: tuple[Any, Any] | None = None

    def write(self, text: str, real: Any) -> int:
        with self._lock:
            if self._shown:
                self.out.write(CLEAR)
                self.out.flush()
                self._shown = False
            written = real.write(text)
            if text:
                self._line_start = text.endswith("\n")
            return written

    def draw(self) -> None:
        with self._lock:
            if self._stop.is_set() or not self._line_start:
                return
            try:
                text = self.render()
            except Exception as exc:  # noqa: BLE001 - a status line must never take the server down
                text = f"[tensorfold] status unavailable: {type(exc).__name__}"
            width = max(20, shutil.get_terminal_size((100, 20)).columns - 1)  # a wrapped line can't be redrawn
            self.out.write(CLEAR + text[:width])
            self.out.flush()
            self._shown = True

    def _tick(self) -> None:
        while not self._stop.wait(self.every):
            self.draw()

    def install(self) -> "LiveLine":
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout, sys.stderr = _Proxy(self, sys.stdout), _Proxy(self, sys.stderr)
        try:
            self._work.launch(self._thread)
        except BaseException as primary:
            try:
                self.stop()  # start may have entered the worker before cancellation
            except BaseException as cleanup:
                raise primary from cleanup
            raise
        return self

    def stop(self) -> None:
        """Stop and join before restoring proxies; a stalled output owner stays retained."""
        if threading.current_thread() is self._thread:
            raise RuntimeError("the live status worker cannot join itself")
        self._stop.set()
        failure = None
        try:
            self._work.drain(self._thread, timeout=5.0)
        except BaseException as error:
            if not self._work.retired.is_set():
                raise
            failure = error
        try:
            with self._lock:
                if self._saved is not None:
                    sys.stdout, sys.stderr = self._saved
                    self._saved = None
                if self._shown:
                    self.out.write(CLEAR)
                    self.out.flush()
                    self._shown = False
        except BaseException as cleanup:
            if failure is not None:
                raise failure from cleanup
            raise
        if failure is not None:
            raise failure


class _Proxy:
    """stdout or stderr with the live line cleared before each write."""

    def __init__(self, live: LiveLine, real: Any) -> None:
        self._live, self._real = live, real

    def write(self, text: str) -> int:
        return self._live.write(text, self._real)

    def flush(self) -> None:
        self._real.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._real, name)


def start(app: Any) -> LiveLine | None:
    """The live line for ``app``'s scheduler in a terminal; None when stdout is redirected or TENSORFOLD_NO_LIVE=1."""

    if os.environ.get("TENSORFOLD_NO_LIVE") == "1" or not sys.stdout.isatty():
        return None
    scheduler = getattr(app, "scheduler", None)
    return None if scheduler is None else LiveLine(lambda: status(scheduler), sys.stdout).install()

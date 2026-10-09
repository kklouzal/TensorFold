"""One-shot callback ownership independent of Python Thread.join state.

CPython 3.12 may report an interrupted join as complete while its target still
executes. The owner cancels admission before entry or waits for this callback's
own final retirement journal. A cancelled pending native thread can later run
only this empty controller, never the retired model/output/file callback.
"""

from __future__ import annotations

import threading
import time
import math


class ThreadWork:
    """Bound one callback to one native thread; lifecycle calls are owner-only."""

    def __init__(self, callback):
        self._lock = threading.Lock()
        self._callback = callback
        self._entered = self._cancelled = self._launched = False
        self.retired = threading.Event()
        self.failure = None

    def run(self):
        callback = None
        try:
            with self._lock:
                if self._cancelled:
                    return
                self._entered = True
                callback = self._callback
            callback()
        except BaseException as error:
            self.failure = error
        finally:
            # Last callback use precedes publication. Its complete status is
            # observable even when Thread.is_alive/join has lost native state.
            with self._lock:
                self._callback = None
            callback = None
            self.retired.set()

    def launch(self, thread):
        with self._lock:
            if self._launched or self._cancelled:
                raise RuntimeError("thread callback may be launched only once")
            self._launched = True
            # Thread.start waits for bootstrap, not for this lock. If it
            # raises after spawning, the callback cannot yet have entered.
            try:
                thread.start()
            except BaseException:
                self._cancelled = True
                self._callback = None
                self.retired.set()
                raise

    def cancel(self):
        """Remove an unentered callback; entered work must retire itself."""
        with self._lock:
            self._cancelled = True
            if not self._entered:
                self._callback = None
                self.retired.set()

    @property
    def launched(self):
        with self._lock:
            return self._launched

    @property
    def cancelled_before_entry(self):
        with self._lock:
            return self._cancelled and not self._entered

    def drain(self, thread, *, timeout, cancel_unentered=True):
        """Wait for resource use to retire, then settle public native join.

        A RuntimeError from join-before-bootstrap cannot prove no scheduled
        thread. It is harmless only after cancellation has mechanically removed
        its callback; that pending runtime-managed controller owns no resources.
        This proves callback quiescence, not OS-thread reaping after an
        interrupted native join. A failed callback remains an explicit error.
        """
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout < 0:
            raise ValueError("thread drain timeout must be a finite nonnegative number")
        if threading.current_thread() is thread:
            raise RuntimeError("a callback worker cannot drain itself")
        deadline = time.monotonic() + timeout
        if cancel_unentered:
            self.cancel()
        if not self.retired.wait(max(0.0, deadline - time.monotonic())):
            raise TimeoutError("thread callback did not retire; its resources remain owned")
        if self.launched:
            try:
                thread.join(timeout=max(0.0, deadline - time.monotonic()))
            except RuntimeError:
                with self._lock:
                    if self._entered or self._callback is not None:
                        raise
        if self.failure is not None:
            raise RuntimeError("owned thread callback failed") from self.failure

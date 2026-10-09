"""One mutable CUDA engine request at a time, with explicit retirement.

The request owns every graph replay and cache mutation until it returns. Close
stops admission before waiting for that request. Its retirement callback must
fence outstanding device work before releasing buffers; failed retirement
retains the caller's engine journal for retry or process containment.
"""
from __future__ import annotations

from contextlib import contextmanager
import threading


class EngineLifetime:
    def __init__(self):
        self.lock = threading.Lock()
        self.closing = threading.Event()
        self.active = None
        self.failed = None
        self.retired = False

    @contextmanager
    def request(self, prepare=None):
        if self.active == threading.get_ident():
            raise RuntimeError("a CUDA engine request cannot reenter its mutable state")
        with self.lock:
            if self.closing.is_set() or self.failed is not None:
                raise RuntimeError("CUDA engine request admission is closed") from self.failed
            prepared = False
            try:
                self.active = threading.get_ident()
                value = prepare() if prepare is not None else None
                prepared = True
                yield value
            except BaseException as error:
                if prepared:
                    self.failed = error
                raise
            finally:
                self.active = None

    def close(self, retire):
        self.closing.set()
        if self.active == threading.get_ident():
            raise RuntimeError("retire CUDA engine after its request callback returns")
        with self.lock:
            if not self.retired:
                retire()
                self.retired = True

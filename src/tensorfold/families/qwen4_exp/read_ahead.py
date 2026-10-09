"""One model-owned bounded Future/ID protocol for host n-gram row readers.

The wrapper owns underlying-table close. Callers must not close or mutate the
borrowed table while this wrapper is live. Accepted gathers/read-aheads copy
IDs once, bound speculation, surface unconsumed failures, and drain every owned
Future before table close. Shutdown failure retains the table and Futures for
explicit retry; new work remains rejected. No worker may join itself.
"""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import threading
import numpy as np


def row_ids(ids, rows, *, copy=False):
    """Validate external row IDs before narrowing; copied asynchronous IDs are owned."""
    value = np.asarray(ids).reshape(-1)
    if value.size and value.dtype.kind not in "iu":
        raise TypeError(f"n-gram row ids must be integers, not {value.dtype}")
    if value.size and (value.min() < 0 or value.max() >= rows):
        raise ValueError(f"n-gram row ids must lie in [0, {rows})")
    value = np.array(value, dtype=np.int64, copy=True, order="C") if copy else value.astype(np.int64, copy=False)
    if copy:
        value.flags.writeable = False
    return value


class ReadAhead:
    """Depth-two cache, at most three outstanding reads; all reads drain before FDs close.

    Accepted gather operations borrow the wrapper. close rejects new operations,
    waits for gathers, cancels queued speculation at terminal close, joins its worker, surfaces any
    unconsumed read failure, then closes the underlying table. IDs are copied for
    async ownership, so caller mutation cannot change a future's requested rows.
    """
    depth = 2

    def __init__(self, table):
        self.table = table
        self._gather = getattr(table, "_gather_owned", table.gather)
        self._reader_thread = None
        self._shutdown_failure = None
        self._pool = ThreadPoolExecutor(1, thread_name_prefix="ngram-read-ahead")
        self._ahead = {}
        self._retired = set()
        self._outstanding = set()
        self._condition = threading.Condition()
        self._active = 0
        self._closing = self._closed = False
        self._cleanup_pending = False

    def __getattr__(self, name):
        if name == "table":
            raise AttributeError(name)
        return getattr(self.table, name)

    def _ids(self, ids):
        value = row_ids(ids, self.table.rows, copy=True)
        return value.tobytes(), value

    def _read(self, ids):
        with self._condition:
            self._reader_thread = threading.current_thread()
        return self._gather(ids)

    def _check(self):
        if self._closing or self._shutdown_failure is not None:
            raise ValueError("the n-gram read-ahead table is closed or has failed shutdown") from self._shutdown_failure
        for future in list(self._retired):
            if future.done():
                self._retired.remove(future)
                future.result()  # never turn a failed speculative read into silent success

    def read_ahead(self, ids):
        key, value = self._ids(ids)
        with self._condition:
            self._check()
            self._outstanding = {f for f in self._outstanding if not f.done()}
            if key not in self._ahead and len(self._outstanding) < self.depth + 1:
                while len(self._ahead) >= self.depth:
                    self._retired.add(self._ahead.pop(next(iter(self._ahead))))
                future = self._pool.submit(self._read, value)
                self._ahead[key] = future
                self._outstanding.add(future)

    def gather(self, ids):
        key, value = self._ids(ids)
        with self._condition:
            self._check()
            future = self._ahead.pop(key, None)
            self._active += 1
        try:
            return future.result() if future is not None else self._gather(value)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def close(self):
        with self._condition:
            if threading.current_thread() is self._reader_thread:
                raise RuntimeError("the n-gram read worker cannot join itself")
            self._condition.wait_for(lambda: not self._closing or self._closed)
            if self._closed:
                if self._cleanup_pending:
                    self.table.close()
                    self._cleanup_pending = False
                return
            self._closing = True
            self._condition.notify_all()
            try:
                self._condition.wait_for(lambda: self._active == 0)
            except BaseException:
                self._closing = False
                self._condition.notify_all()
                raise
            futures = set(self._ahead.values()) | self._retired | self._outstanding
            self._ahead.clear()
            self._retired.clear()
            self._outstanding.clear()
        failure = cleanup_error = None
        joined = False
        try:
            # Interrupted joins must still complete before the table's FDs may close.
            while True:
                try:
                    self._pool.shutdown(wait=True, cancel_futures=True)
                    break
                except KeyboardInterrupt as exc:
                    if cleanup_error is None:
                        cleanup_error = exc
            for future in futures:
                if not future.cancelled():
                    while True:
                        try:
                            future.result()
                            break
                        except KeyboardInterrupt as exc:
                            if future.done():
                                if failure is None:
                                    failure = exc
                                break
                            if cleanup_error is None:
                                cleanup_error = exc
                        except BaseException as exc:
                            if not future.done():
                                raise
                            if failure is None:
                                failure = exc
                            break
            joined = True
            self._shutdown_failure = None
        except BaseException as shutdown_failure:
            with self._condition:
                self._shutdown_failure = shutdown_failure
                self._retired = futures
                self._closing = False
                self._condition.notify_all()
            raise
        finally:
            if joined:
                try:
                    self.table.close()
                except BaseException as exc:
                    self._cleanup_pending = True
                    if cleanup_error is None:
                        cleanup_error = exc
                finally:
                    with self._condition:
                        self._closed = True
                        self._condition.notify_all()
        if failure is not None:
            if cleanup_error is not None:
                raise failure from cleanup_error
            raise failure
        if cleanup_error is not None:
            raise cleanup_error

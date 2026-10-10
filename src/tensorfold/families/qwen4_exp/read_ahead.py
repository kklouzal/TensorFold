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

from tensorfold.cleanup import raise_failures

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
        self._close_statuses = []

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
        marker, futures, observed = object(), None, []
        try:
            with self._condition:
                if threading.current_thread() is self._reader_thread:
                    raise RuntimeError("the n-gram read worker cannot join itself")
                self._condition.wait_for(lambda: not self._closing or self._closed)
                if self._closed:
                    if self._cleanup_pending:
                        self.table.close()
                        self._cleanup_pending = False
                    return
                self._closing = marker
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
            failures = [row for row in self._close_statuses if row[2]]
            cleanup_errors = [row for row in self._close_statuses if not row[2]]
            observed.extend(failures + cleanup_errors)

            def record(error, terminal):
                if all(error is not row[0] for row in failures + cleanup_errors):
                    prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                             BaseException.__suppress_context__.__get__(error))
                    row = (error, prior, terminal)
                    observed.append(row)
                    (failures if terminal else cleanup_errors).append(row)

            def restore():
                for error, prior, _ in failures + cleanup_errors:
                    BaseException.__cause__.__set__(error, prior[0])
                    BaseException.__context__.__set__(error, prior[1])
                    BaseException.__suppress_context__.__set__(error, prior[2])

            joined = False
            try:
                # Interrupted joins must still complete before the table's FDs may close.
                while True:
                    try:
                        self._pool.shutdown(wait=True, cancel_futures=True)
                        break
                    except KeyboardInterrupt as exc:
                        record(exc, False)
                for future in futures:
                    if not future.cancelled():
                        while True:
                            try:
                                future.result()
                                break
                            except KeyboardInterrupt as exc:
                                if future.done():
                                    record(exc, True)
                                    break
                                record(exc, False)
                            except BaseException as exc:
                                if not future.done():
                                    raise
                                record(exc, True)
                                break
                joined = True
                self._shutdown_failure = None
            except BaseException as shutdown_failure:
                record(shutdown_failure, False)
                pending_statuses = failures + cleanup_errors
                with self._condition:
                    restore()
                    self._shutdown_failure = shutdown_failure
                    self._retired = futures
                    self._close_statuses = pending_statuses
                    self._closing = False
                    self._condition.notify_all()
                raise_failures(shutdown_failure, [row[0] for row in pending_statuses])
            finally:
                if joined:
                    try:
                        self.table.close()
                    except BaseException as exc:
                        self._cleanup_pending = True
                        record(exc, False)
                    finally:
                        with self._condition:
                            self._closed = True
                            self._condition.notify_all()
            restore()
            self._close_statuses = []
            errors = [row[0] for row in failures + cleanup_errors]
            if errors:
                raise_failures(errors[0], errors[1:])
        except BaseException as primary:
            prior = (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary),
                     BaseException.__suppress_context__.__get__(primary))
            secondary = []
            for error, roots, _ in observed:
                if error is not primary:
                    BaseException.__cause__.__set__(error, roots[0])
                    BaseException.__context__.__set__(error, roots[1])
                    BaseException.__suppress_context__.__set__(error, roots[2])
                    secondary.append(error)
            try:
                with self._condition:
                    # Only this operation can repair its publication. The full
                    # local journal exists before any authoritative set clears.
                    if self._closing is marker and not self._closed:
                        if futures is not None:
                            self._retired = futures
                            self._shutdown_failure = primary
                        self._closing = False
                        self._condition.notify_all()
            except BaseException as cleanup:
                BaseException.__cause__.__set__(primary, prior[0])
                BaseException.__context__.__set__(primary, prior[1])
                BaseException.__suppress_context__.__set__(primary, prior[2])
                raise_failures(primary, [*secondary, cleanup])
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])
            raise_failures(primary, secondary)

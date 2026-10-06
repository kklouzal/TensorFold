"""Optional Linux native PLE reader: exact on-demand rows, owned pool and explicit teardown.

Selected once at startup. Compilation errors are fatal. No table payload is loaded
until gather. IDs/layout/coalescing match v0.6.1 SSDTable; all native destinations
are independently checked. A table borrows its open FDs to a native batch and
keeps them alive until every accepted gather and read-ahead future is drained.
"""
from __future__ import annotations
import hashlib
from pathlib import Path
import sys
import threading
import weakref
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from tensorfold.families.qwen4_exp.ssd_table import SSDTable, MAX_READ


def load_reader():
    """Compile/cache only CPU C++; versioned source identity prevents stale binary reuse."""
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_name("ssd_read.cpp")
    identity = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return load(name="tf_ssd_read_" + identity, sources=[str(source)],
                extra_cflags=["-O3", "-std=c++17"], with_cuda=False, verbose=False)


class _Lifetime:
    """Own pool/FDs; active gathers borrow both. Closing waits before releasing either."""
    def __init__(self, pool, fds, release_fds):
        self.pool, self.fds, self.release_fds = pool, fds, release_fds
        self.condition = threading.Condition()
        self.active = 0
        self.closing = self.closed = False
        self.broken = None

    def enter(self):
        with self.condition:
            if self.closing or self.broken is not None:
                raise ValueError("the n-gram table is closed or unusable") from self.broken
            self.active += 1

    def leave(self):
        with self.condition:
            self.active -= 1
            self.condition.notify_all()

    def close(self):
        with self.condition:
            self.condition.wait_for(lambda: not self.closing or self.closed)
            if self.closed:
                return
            self.closing = True
            try:
                self.condition.wait_for(lambda: self.active == 0)
            except BaseException:
                self.closing = False
                self.condition.notify_all()
                raise
        error = None
        # Native close is noexcept and has no Python callbacks. An interrupt can
        # arrive after it finishes; retry is idempotent and proves the drain.
        while True:
            try:
                self.pool.close()
                break
            except KeyboardInterrupt as exc:
                if error is None:
                    error = exc
            except BaseException as exc:
                with self.condition:
                    self.broken = exc
                    self.closing = False
                    self.condition.notify_all()
                raise
        # The native GIL-held close marks every FD consumed before an interrupt
        # can reach Python. A pre-entry interrupt leaves the whole owner intact.
        while self.fds:
            try:
                self.release_fds(self.fds)
            except KeyboardInterrupt as exc:
                if error is None:
                    error = exc
            except BaseException as exc:
                if self.fds:
                    with self.condition:
                        self.broken = exc
                        self.closing = False
                        self.condition.notify_all()
                    raise
                if error is None:
                    error = exc
        with self.condition:
            self.closed = True
            self.condition.notify_all()
        if error is not None:
            raise error


class NativeSSDTable(SSDTable):
    """Same table bytes and canonical runs; direct native pread eliminates Python per-read copies."""
    def __init__(self, files, *, workers: int = 16, nocache: bool = True):
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ValueError("native SSD workers must lie in [1, 64]")
        self.workers = workers
        self._fds = []
        module = load_reader()
        self._native = module.Reader(workers)
        self._life = _Lifetime(self._native, self._fds, module.close_fds)
        self._closer = weakref.finalize(self, self._life.close)
        try:
            self._layout(files, nocache)
            # Layout's file-bound checks establish safe int64 row-offset arithmetic.
            for value in (self.starts, self.fidx, self.bases, self._fd_of):
                value.flags.writeable = False
            if self.rows > np.iinfo(np.int64).max or any(x < 0 for x in self.bases.reshape(-1)):
                raise ValueError("n-gram layout does not fit native int64 descriptors")
        except BaseException:
            self.close()
            raise

    def _runs(self, where, offsets, width, at):
        """Canonical SSDTable._reads coalescing, represented as checked native descriptors."""
        n = offsets.size
        if not n:
            return np.empty((0, 4), dtype=np.int64)
        cut = np.ones(n, dtype=bool)
        cut[1:] = (where[1:] != where[:-1]) | (offsets[1:] != offsets[:-1] + width)
        first = np.flatnonzero(cut)
        cut |= (np.arange(n) - first[np.cumsum(cut) - 1]) % max(1, MAX_READ // width) == 0
        first = np.flatnonzero(cut)
        rows = np.diff(first, append=n)
        plan = np.empty((first.size, 4), dtype=np.int64)
        plan[:, 0], plan[:, 1] = self._fd_of[where[first]], offsets[first]
        plan[:, 2], plan[:, 3] = rows * width, at + first * width
        return plan

    def gather(self, ids):
        self._life.enter()
        try:
            flat = np.asarray(ids).reshape(-1)
            if flat.size and flat.dtype.kind not in "iu":
                raise TypeError(f"n-gram row ids must be integers, not {flat.dtype}")
            if flat.size and (flat.min() < 0 or flat.max() >= self.rows):
                raise ValueError(f"n-gram row ids must lie in [0, {self.rows})")
            unique, inverse = np.unique(flat.astype(np.int64), return_inverse=True)
            shard = np.searchsorted(self.starts, unique, side="right") - 1
            local, where = unique - self.starts[shard], self.fidx[shard]
            count = unique.size
            size = int(count) * (self.wrow + 2 * self.grow)
            if size > sys.maxsize:
                raise ValueError("n-gram gather exceeds the native allocation range")
            output = np.empty(size, dtype=np.uint8)
            outs, plans, at = [], [], 0
            for part, width in enumerate((self.wrow, self.grow, self.grow)):
                outs.append(output[at:at + count * width].reshape(count, width))
                plans.append(self._runs(where, self.bases[shard, part] + local * width, width, at))
                at += count * width
            self._native.read(np.concatenate(plans), output)
            words, scales, biases = (out[inverse] for out in outs)
            return words.view(np.uint32), scales.view(np.uint16), biases.view(np.uint16)
        finally:
            self._life.leave()

    def close(self):
        self._life.close()
        self._closer.detach()


class NativeReadAhead:
    """Depth-two cache, at most three outstanding reads; all reads drain before FDs close.

    Accepted gather operations borrow the wrapper. close rejects new operations,
    waits for gathers, cancels queued speculation at terminal close, joins its worker, surfaces any
    unconsumed read failure, then closes the underlying table. IDs are copied for
    async ownership, so caller mutation cannot change a future's requested rows.
    """
    depth = 2

    def __init__(self, table):
        self.table = table
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
        value = np.array(np.asarray(ids).reshape(-1), copy=True, order="C")
        # Keep canonical gather validation, including rejecting noninteger nonempty IDs.
        if value.size and value.dtype.kind not in "iu":
            raise TypeError(f"n-gram row ids must be integers, not {value.dtype}")
        if value.size and (value.min() < 0 or value.max() >= self.table.rows):
            raise ValueError(f"n-gram row ids must lie in [0, {self.table.rows})")
        key = np.ascontiguousarray(value, dtype=np.int64).tobytes()
        return key, value

    def _check(self):
        if self._closing:
            raise ValueError("the n-gram read-ahead table is closed")
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
                future = self._pool.submit(self.table.gather, value)
                self._ahead[key] = future
                self._outstanding.add(future)

    def gather(self, ids):
        key, value = self._ids(ids)
        with self._condition:
            self._check()
            future = self._ahead.pop(key, None)
            self._active += 1
        try:
            return future.result() if future is not None else self.table.gather(value)
        finally:
            with self._condition:
                self._active -= 1
                self._condition.notify_all()

    def close(self):
        with self._condition:
            self._condition.wait_for(lambda: not self._closing or self._closed)
            if self._closed:
                if self._cleanup_pending:
                    self.table.close()
                    self._cleanup_pending = False
                return
            self._closing = True
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
                    try:
                        future.result()
                    except BaseException as exc:
                        if failure is None:
                            failure = exc
        finally:
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

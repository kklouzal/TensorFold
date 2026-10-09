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
from functools import partial
import weakref
import numpy as np
from tensorfold.families.qwen4_exp.ssd_table import SSDTable, MAX_READ, _release_table_files
from tensorfold.families.qwen4_exp.table_lifetime import TableLifetime, _note, _cause
from tensorfold.families.qwen4_exp.read_ahead import ReadAhead as NativeReadAhead, row_ids  # noqa: F401


def load_reader():
    """Compile/cache only CPU C++; versioned source identity prevents stale binary reuse."""
    from torch.utils.cpp_extension import load
    source = Path(__file__).with_name("ssd_read.cpp")
    identity = hashlib.sha256(source.read_bytes()).hexdigest()[:16]
    return load(name="tf_ssd_read_" + identity, sources=[str(source)],
                extra_cflags=["-O3", "-std=c++17"], with_cuda=False, verbose=False)


class NativeSSDTable(SSDTable):
    """Same table bytes and canonical runs; direct native pread eliminates Python per-read copies."""
    def __init__(self, files, *, workers: int = 16, nocache: bool = True):
        if type(workers) is not int or not 1 <= workers <= 64:
            raise ValueError("native SSD workers must lie in [1, 64]")
        self.workers = workers
        self._fds = []
        self._files = []
        self._unpublished_file = [None]
        self._native = self._life = self._closer = None
        try:
            module = load_reader()
            self._native = module.Reader(workers)
            self._life = TableLifetime(self._native, self._fds,
                                       partial(_release_table_files, files=self._files, unpublished=self._unpublished_file))
            self._closer = weakref.finalize(self, self._life.close)
            self._layout(files, nocache)
            # Layout's file-bound checks establish safe int64 row-offset arithmetic.
            for value in (self.starts, self.fidx, self.bases, self._fd_of):
                value.flags.writeable = False
            if self.rows > np.iinfo(np.int64).max or any(x < 0 for x in self.bases.reshape(-1)):
                raise ValueError("n-gram layout does not fit native int64 descriptors")
        except BaseException as primary:
            try:
                if self._life is not None:
                    self.close()
                else:
                    if self._native is not None:
                        self._native.close()
                    _release_table_files(self._fds, files=self._files, unpublished=self._unpublished_file)
            except BaseException as cleanup:
                _note(primary, "native table construction cleanup also failed; owner retained")
                raise primary from _cause(primary, cleanup)
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
        return self._life.call(self._gather_ids, ids)

    def _gather_ids(self, ids):
        return self._gather_flat(row_ids(ids, self.rows))

    def _gather_owned(self, flat):
        """Borrowed immutable int64 IDs already validated by owned ReadAhead."""
        return self._life.call(self._gather_flat, flat)

    def _gather_flat(self, flat):
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

    def close(self):
        self._life.close()
        if self._closer is not None:
            self._closer.detach()

"""Read n-gram (PLE) rows from the checkpoint's files at each lookup, holding no table in memory."""

from __future__ import annotations

import os
import stat
import sys
import weakref
from functools import partial
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

from .read_ahead import row_ids
from .table_lifetime import PythonPool, TableLifetime, release_files, _note, _cause
from .table_file import TableFile
from tensorfold.cuda.capacity import SIZES
from tensorfold.cuda.tensor_file import read_header_stream

WORKERS = 16                # reads in flight at once: os.pread releases the GIL
MAX_READ = 1 << 20          # bytes one read of adjacent rows may cover
_KINDS = (("weight", "U32", 4), ("scales", "BF16", 2), ("biases", "BF16", 2))


def _retain_file(slot, file):
    """Publish exact local ownership even after an interrupted first publication."""
    interrupted = None
    while True:
        try:
            slot[0] = file
            return interrupted
        except KeyboardInterrupt as error:
            if interrupted is None:
                interrupted = error


def _release_table_files(fds, *, files, unpublished):
    file = unpublished[0]
    if file is not None:
        try:
            if not file.closed:
                file.close()
        finally:
            if file.closed:
                unpublished[0] = None
    release_files(fds, files)


def _no_cache(fd: int) -> None:
    """Keep the file's pages out of the cache (macOS F_NOCACHE) or its readahead off (Linux FADV_RANDOM)."""

    if sys.platform == "darwin":
        import fcntl

        fcntl.fcntl(fd, getattr(fcntl, "F_NOCACHE", 48), 1)       # 48: F_NOCACHE in <sys/fcntl.h>
    elif hasattr(os, "posix_fadvise"):
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_RANDOM)


def _span(entry: object, kind: tuple[str, str, int], data: int, size: int, name: str) -> tuple[int, int, int]:
    """(rows, bytes a row, file offset of row 0) of one tensor; refuses a dtype, shape or range it can't read."""

    part, dtype, item = kind
    if not isinstance(entry, dict) or entry.get("dtype") != dtype:
        raise ValueError(f"{name}: the n-gram {part} must be a {dtype} tensor")
    shape, span = entry.get("shape"), entry.get("data_offsets")
    if not (isinstance(shape, list) and len(shape) == 2 and all(type(n) is int and n > 0 for n in shape)):
        raise ValueError(f"{name}: the n-gram {part} shape {shape} is not [rows, columns]")
    if not (isinstance(span, list) and len(span) == 2 and all(type(n) is int for n in span) and span[0] >= 0
            and span[1] - span[0] == shape[0] * shape[1] * item and data + span[1] <= size):
        raise ValueError(f"{name}: the n-gram {part} bytes {span} disagree with its shape or pass the file's end")
    return shape[0], shape[1] * item, data + span[0]


def _fill(reads: list[tuple[int, int, int, memoryview, int]]) -> None:
    """Each (fd, offset, size, out, at) read into ``out[at:]``, finishing short reads; end of file means it changed."""

    for fd, offset, size, out, at in reads:
        done = 0
        while done < size:
            data = os.pread(fd, size - done, offset + done)
            if not data:
                raise OSError(f"short read of the n-gram tables at byte {offset + done}: the checkpoint changed")
            out[at + done:at + done + len(data)] = data
            done += len(data)


class SSDTable:
    """HostTable's rows, byte for byte, read from the checkpoint at each gather: deduplicated, coalesced, parallel."""

    def __init__(self, files: list[tuple[Path, dict, dict, dict]], *, nocache: bool = True) -> None:
        self._fds: list[int] = []
        self._files = []
        self._unpublished_file = [None]
        self._pool = self._life = self._closer = None
        try:
            self._pool = ThreadPoolExecutor(WORKERS, thread_name_prefix="ple-ssd")
            self._life = TableLifetime(PythonPool(self._pool), self._fds,
                                       partial(_release_table_files, files=self._files, unpublished=self._unpublished_file))
            self._closer = weakref.finalize(self, self._life.close)
            self._layout(files, nocache)
        except BaseException as primary:
            try:
                if self._life is not None:
                    self.close()
                else:
                    if self._pool is not None:
                        self._pool.shutdown(wait=True)
                    _release_table_files(self._fds, files=self._files, unpublished=self._unpublished_file)
            except BaseException as cleanup:
                _note(primary, "SSD table construction cleanup also failed; owner retained")
                raise primary from _cause(primary, cleanup)
            raise

    def _layout(self, files: list[tuple[Path, dict, dict, dict]], nocache: bool) -> None:
        if not files:
            raise ValueError("the n-gram table has no shards")
        opened: dict[Path, tuple[int, int, int]] = {}
        starts, fidx, bases, widths = [0], [], [], None
        total = 0
        for path, *entries in files:
            path = Path(path)
            if path not in opened:
                file = None
                try:
                    file = TableFile(path)
                    self._unpublished_file[0] = file
                    file.open()                    # closed owner is published before native acquisition
                    self._files.append(file)
                    self._unpublished_file[0] = None
                    fd = file.fileno()
                    self._fds.append(fd)
                except BaseException as primary:
                    if file is not None:
                        interruption = _retain_file(self._unpublished_file, file)
                        if interruption is not None:
                            _note(primary, "unpublished SSD file ownership repair was interrupted")
                            raise primary from _cause(primary, interruption)
                    raise
                opened_stat = os.fstat(fd)
                if not stat.S_ISREG(opened_stat.st_mode):
                    raise ValueError(f"{path.name}: the n-gram checkpoint must be a regular file")
                if nocache:
                    _no_cache(fd)
                size = opened_stat.st_size
                data, header = read_header_stream(file, SIZES, label=path)
                descriptors = {(info["dtype"], tuple(info["shape"]), tuple(info["data_offsets"]))
                               for name, info in header.items() if name != "__metadata__"}
                opened[path] = (len(self._fds) - 1, data, size, descriptors)
            index, data, size, descriptors = opened[path]
            (rows, wrow, w0), (srows, grow, s0), (brows, brow, b0) = (
                _span(entry, kind, data, size, path.name) for entry, kind in zip(entries, _KINDS))
            if any((entry["dtype"], tuple(entry["shape"]), tuple(entry["data_offsets"])) not in descriptors
                   for entry in entries):
                raise ValueError(f"{path.name}: the n-gram tensor metadata disagrees with its opened checkpoint header")
            if not rows == srows == brows or wrow != 8 * grow or brow != grow:
                raise ValueError(f"{path.name}: an n-gram shard is not 4-bit rows with a scale and bias every 32")
            if widths not in (None, (wrow, grow)):
                raise ValueError(f"{path.name}: the n-gram shards differ in row width")
            widths = (wrow, grow)
            total += rows * (wrow + 2 * grow)
            starts.append(starts[-1] + rows)
            fidx.append(index)
            bases.append((w0, s0, b0))
        self.starts = np.array(starts, dtype=np.int64)
        self.rows = int(self.starts[-1])
        self.fidx = np.array(fidx, dtype=np.int64)
        self.bases = np.array(bases, dtype=np.int64)        # [shard, component]: file offset of row 0
        self.wrow, self.grow = widths
        self.nbytes = total
        self._fd_of = np.array(self._fds, dtype=np.int64)

    def gather(self, ids: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Validate public IDs once, then gather original checkpoint bytes."""
        return self._life.call(self._gather_ids, ids)

    def _gather_ids(self, ids):
        return self._gather_flat(row_ids(ids, self.rows))

    def _gather_owned(self, flat):
        """Internal owned immutable int64 IDs; lifetime may change independently."""
        return self._life.call(self._gather_flat, flat)

    def _gather_flat(self, flat):
        unique, inverse = np.unique(flat.astype(np.int64), return_inverse=True)
        shard = np.searchsorted(self.starts, unique, side="right") - 1
        local, where = unique - self.starts[shard], self.fidx[shard]
        outs, reads = [], []
        for part, width in enumerate((self.wrow, self.grow, self.grow)):
            outs.append(np.empty((unique.size, width), dtype=np.uint8))
            reads += self._reads(where, self.bases[shard, part] + local * width, width, outs[-1])
        batches = min(WORKERS, len(reads))
        if batches > 1:
            self._life.parallel(self._pool, _fill, [reads[i::batches] for i in range(batches)])
        else:
            _fill(reads)
        words, scales, biases = (out[inverse] for out in outs)
        return words.view(np.uint32), scales.view(np.uint16), biases.view(np.uint16)

    def _reads(self, where: np.ndarray, offsets: np.ndarray, width: int, out: np.ndarray) -> list[tuple]:
        """One read per run of rows adjacent in one file, each at most MAX_READ bytes, into ``out``'s rows."""

        n = offsets.size
        if not n:
            return []
        cut = np.ones(n, dtype=bool)
        cut[1:] = (where[1:] != where[:-1]) | (offsets[1:] != offsets[:-1] + width)
        first = np.flatnonzero(cut)
        cut |= (np.arange(n) - first[np.cumsum(cut) - 1]) % max(1, MAX_READ // width) == 0
        first = np.flatnonzero(cut)
        rows = np.diff(first, append=n)
        view = memoryview(out.reshape(-1))
        return list(zip(self._fd_of[where[first]].tolist(), offsets[first].tolist(), (rows * width).tolist(),
                        [view] * first.size, (first * width).tolist()))

    def prefetch(self, workers: int = 8) -> float:
        """Nothing to warm: rows are read at each lookup, so this reads nothing (0 seconds)."""

        return 0.0

    def close(self) -> None:
        """Close the checkpoint's files and the read threads (also done at exit); a later gather raises."""

        self._life.close()
        if self._closer is not None:
            self._closer.detach()

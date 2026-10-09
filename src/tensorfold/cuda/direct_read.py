"""Checkpoint bytes read with O_DIRECT in 64 MiB spans (many requests in flight), to the GPU or host; buffered where refused."""

from __future__ import annotations

import errno
import os
from pathlib import Path
import threading

import torch

from .tensor_file import byte_range

PIECE = 64 << 20         # bytes a direct read fills: the size of each pinned staging piece
ALIGN = 4096             # O_DIRECT's file offset, length and buffer alignment

DTYPES = {"BOOL": torch.bool, "U8": torch.uint8, "I8": torch.int8, "U16": torch.uint16, "I16": torch.int16,
          "U32": torch.uint32, "I32": torch.int32, "U64": torch.uint64, "I64": torch.int64, "F16": torch.float16,
          "BF16": torch.bfloat16, "F32": torch.float32, "F64": torch.float64, "F8_E4M3": torch.float8_e4m3fn,
          "F8_E5M2": torch.float8_e5m2}


def _up(x: int) -> int:
    return -(-x // ALIGN) * ALIGN


class Reader:
    """Byte ranges of files as uint8 tensors on a device; ``direct`` is False once O_DIRECT has been refused."""

    def __init__(self) -> None:
        self.direct = hasattr(os, "O_DIRECT")
        self.staged = os.name == "nt"               # Windows has no O_DIRECT: reads there buffer into pinned staging
        self.staging: list[list] = []         # [pinned piece, event of its last copy]
        self.turn = 0

    def read(self, path: str | Path, offset: int, n: int, device: str | torch.device = "cpu", *,
             pinned: bool = False) -> torch.Tensor:
        """Bytes [offset, offset + n) of ``path`` as a new uint8 tensor on ``device`` (``pinned``: page-locked, direct reads only)."""

        cuda = torch.device(device).type == "cuda"
        if type(offset) is not int or type(n) is not int or offset < 0 or n < 0:
            raise ValueError("checkpoint byte offsets and lengths must be nonnegative integers")
        if n > 0 and self.direct:
            try:
                return self._to_device(path, offset, n, device) if cuda else self._to_host(path, offset, n, pinned)
            except OSError as exc:
                if exc.errno != errno.EINVAL:
                    raise
                self.direct = False               # the file system refuses O_DIRECT, on the open or on a read
        if n > 0 and self.staged and (cuda or pinned):
            # Windows stages here: one page-locked block where that means host RAM, filled by one buffered read
            raw = self._buffered(path, offset, n, pinned=torch.cuda.is_available())
            return raw.to(device) if cuda else raw
        raw = self._buffered(path, offset, n)
        return raw.to(device) if cuda else raw

    def close(self) -> None:
        """Give the pinned staging back to the system, not to the host allocator's cache."""

        if self.staging:
            for _, copied in self.staging:
                if copied is not None:
                    copied.synchronize()
            self.staging.clear()
            getattr(torch._C, "_host_emptyCache", lambda: None)()

    def _buffered(self, path, offset: int, n: int, pinned: bool = False) -> torch.Tensor:
        """One buffered read of ``n`` bytes, page-locked when ``pinned`` (how Windows stages, having no O_DIRECT)."""

        with open(path, "rb", buffering=0) as f:
            byte_range(offset, n, os.fstat(f.fileno()).st_size, path)
            raw = torch.empty((n,), dtype=torch.uint8, pin_memory=pinned, device="cpu")
            view = memoryview(raw.numpy())
            f.seek(offset)
            at = 0
            while at < n:
                got = f.readinto(view[at:at + PIECE])
                if not got:
                    raise IOError(f"short read of {path} at {offset + at}")
                at += got
        return raw

    @staticmethod
    def _fill(fd: int, view: memoryview, lo: int, need: int, path) -> None:
        """Read the aligned span at ``lo`` into ``view`` until ``need`` bytes arrived (the span may run past EOF)."""

        got = 0
        while got < need:
            k = os.preadv(fd, [view[got:]], lo + got)
            if k <= 0:
                raise IOError(f"short read of {path} at {lo + got}")
            got += k

    def _to_host(self, path, offset: int, n: int, pinned: bool = False) -> torch.Tensor:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            size = os.fstat(fd).st_size
            byte_range(offset, n, size, path)
            lo, end = offset // ALIGN * ALIGN, _up(size)
            hi = min(_up(offset + n), end)
            block = torch.empty((hi - lo + ALIGN,), dtype=torch.uint8, pin_memory=pinned)
            lead = -block.data_ptr() % ALIGN      # host blocks need not be page-aligned: align the span here
            view = memoryview(block[lead:lead + hi - lo].numpy())
            skip = offset - lo
            at = 0
            while at < skip + n:                  # one read of at most PIECE bytes at a time
                take = min(PIECE, hi - lo - at)
                self._fill(fd, view[at:at + take], lo + at, min(take, skip + n - at), path)
                at += take
        finally:
            os.close(fd)
        out = block[lead + skip:lead + skip + n]
        if pinned or (lead + skip) % 8 == 0:      # pinned: uploaded as bytes, and a clone would not be pinned
            return out
        return out.clone()                        # views as any dtype need an aligned start

    def _to_device(self, path, offset: int, n: int, device) -> torch.Tensor:
        fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
        try:
            size = os.fstat(fd).st_size
            byte_range(offset, n, size, path)
            if not self.staging:
                for _ in range(2):                # pinned blocks need not be page-aligned: align the pieces here
                    block = torch.empty((PIECE + 3 * ALIGN,), dtype=torch.uint8, pin_memory=True)
                    lead = -block.data_ptr() % ALIGN
                    self.staging.append([block[lead:lead + PIECE + 2 * ALIGN], None])
            end = _up(size)
            out = torch.empty((n,), dtype=torch.uint8, device=device)
            stream = torch.cuda.current_stream(out.device)   # the copies' stream, whichever device is current
            at = 0
            while at < n:
                take = min(PIECE, n - at)
                slot = self.staging[self.turn]
                self.turn ^= 1
                piece, copied = slot
                if copied is not None:
                    copied.synchronize()          # the piece's previous copy has finished
                lo = (offset + at) // ALIGN * ALIGN
                hi = min(_up(offset + at + take), end)
                skip = offset + at - lo
                self._fill(fd, memoryview(piece.numpy())[:hi - lo], lo, skip + take, path)
                out[at:at + take].copy_(piece[skip:skip + take], non_blocking=True)
                slot[1] = torch.cuda.Event()
                slot[1].record(stream)
                at += take
            return out
        finally:
            os.close(fd)


def read_header(path: str | Path) -> tuple[int, dict]:
    """(offset of the data, header) of a safetensors file."""

    from .capacity import SIZES
    from .tensor_file import read_header as validated_header

    return validated_header(path, {dtype: SIZES[dtype] for dtype in DTYPES})


class SafeTensors:
    """Tensors by name (a later file's name wins), read through one Reader.

    Failed construction drains only a reader created here; a supplied reader
    remains caller-owned. Explicit close() drains the associated reader.
    """

    def __init__(self, files, reader: Reader | None = None) -> None:
        self.reader = reader or Reader()
        self.where: dict[str, tuple[Path, int, int, str, list[int]]] = {}   # name -> (file, begin, bytes, dtype, shape)
        try:
            for path in files:
                base, header = read_header(path)
                for name, e in header.items():
                    if name != "__metadata__":
                        begin, end = e["data_offsets"]
                        self.where[name] = (Path(path), base + begin, end - begin, e["dtype"], list(e["shape"]))

        except BaseException as primary:
            if self.reader is not reader:
                try:
                    self.reader.close()
                except BaseException as cleanup:
                    BaseException.add_note(primary, "checkpoint reader construction cleanup also failed")
                    raise primary from cleanup
            raise

    def keys(self) -> list[str]:
        return list(self.where)

    def __contains__(self, name: str) -> bool:
        return name in self.where

    def close(self) -> None:
        self.reader.close()

    def get(self, name: str, device: str | torch.device = "cpu") -> torch.Tensor:
        path, begin, n, dtype, shape = self.where[name]
        if dtype not in DTYPES:
            raise ValueError(f"{name}: safetensors dtype {dtype} is not supported")
        return self.reader.read(path, begin, n, device).view(DTYPES[dtype]).reshape(shape)


class ReadAhead:
    """Operation-owned reads and uploads; reusable after a completed close.

    Every submitted future stays owned until its result is consumed or shutdown
    observes it. Dropped/unconsumed read failures raise at close; failures already
    returned by take are not raised again. Close drains accepted work before
    releasing the upload stream. Failed shutdown retains owners for retry and
    refuses new work. Control methods serialize; close never joins under their
    lock or from the worker that would have to finish itself.
    """

    def __init__(self, reader: Reader | None = None, threads: int = 8, run: int = 128 << 20,
                 gap: int = 1 << 20) -> None:
        if (type(threads) is not int or threads <= 0 or type(run) is not int or run <= 0
                or type(gap) is not int or gap < 0):
            raise ValueError("read-ahead threads/run must be positive integers and gap nonnegative")
        self.reader = reader or Reader()
        self.threads, self.run, self.gap = threads, run, gap
        self.ahead: dict = {}                              # key -> the read's future: (upload event or None, tensors)
        self.pool = None
        self.stream = None
        self._lock = threading.RLock()
        self._condition = threading.Condition(self._lock)
        self._workers = threading.local()
        self._owned = set()
        self._observed = set()
        self._discard_error = None
        self._closing = self._shutdown_pending = False

    def _available(self):
        if self._closing or self._shutdown_pending:
            raise RuntimeError("read-ahead shutdown is in progress or incomplete")

    def _worker_control(self):
        if getattr(self._workers, "active", False):
            raise RuntimeError("a read-ahead worker cannot reenter its own control methods")

    def _retire(self):
        referenced = set(self.ahead.values())
        for future in tuple(self._owned):
            if future not in referenced and future.done() and (
                    future.cancelled() or future in self._observed or future.exception() is None):
                self._owned.remove(future)
                self._observed.discard(future)

    def queue(self, items, device=None, cut=None) -> None:
        """Start reading ``items`` (key, path, first byte, end byte, meta) not queued yet; ``cut(raw, meta)`` copies a tensor out of a shared read."""
        self._worker_control()             # before locking: take may be waiting for this worker
        with self._lock:
            self._available()
            self._retire()
            self._queue(items, device, cut)

    def _queue(self, items, device, cut):

        from concurrent.futures import ThreadPoolExecutor

        cut = cut or (lambda raw, meta: raw.clone())
        by_path: dict[str, list] = {}
        accepted = set()
        for item in items:
            if not isinstance(item, (tuple, list)) or len(item) != 5:
                raise ValueError("queued checkpoint reads require key/path/begin/end/metadata")
            key, path, begin, end, _ = item
            if (type(begin) is not int or type(end) is not int or begin < 0 or end < begin):
                raise ValueError("queued checkpoint byte ranges must be nonnegative ordered integers")
            if key in self.ahead:
                continue
            if key in accepted:
                raise ValueError("queued checkpoint lookup keys must be distinct")
            accepted.add(key)
            by_path.setdefault(str(path), []).append(item)
        if not by_path:
            return
        if device is not None and torch.device(device).type != "cuda":
            device = None
        if device is not None:
            device = torch.device(device)
            if device.index is None:
                device = torch.device("cuda", torch.cuda.current_device())
            if self.stream is not None and self.stream.device != device:
                raise ValueError("read-ahead upload device cannot change before close")
        if self.pool is None:
            self.pool = ThreadPoolExecutor(self.threads, thread_name_prefix="read-ahead")
        if device is not None and self.stream is None:
            self.stream = torch.cuda.Stream(torch.device(device))
        for path, group in by_path.items():
            group.sort(key=lambda item: item[2])
            run: list = []
            hi = 0
            for item in group + [None]:
                if run and (item is None or item[2] - hi > self.gap or item[3] - run[0][2] > self.run):
                    future = self.pool.submit(self._read, path, run[0][2], hi, run, device, cut)
                    try:
                        self._owned.add(future)
                        # Prepare all lookup aliases before publication. A host
                        # allocation failure still drains the locally held job.
                        ahead = dict(self.ahead)
                        for queued in run:
                            ahead[queued[0]] = future
                        self.ahead = ahead
                    except BaseException as primary:
                        try:
                            if not future.cancel():
                                future.result()
                            if self.stream is not None:
                                self.stream.synchronize()
                        except BaseException as cleanup:
                            BaseException.add_note(primary, "unpublished read cleanup failed; owner remains retained")
                            raise primary from cleanup
                        raise
                    run = []
                if item is not None:
                    hi = max(hi, item[3]) if run else item[3]
                    run.append(item)

    def take(self, key):
        """``key``'s tensor, once read (on a device, the caller's stream waits for its upload); None if not queued."""
        self._worker_control()
        with self._lock:
            self._available()
            return self._take(key)

    def _take(self, key):
        future = self.ahead.pop(key, None)
        if future is None:
            return None
        try:
            uploaded, tensors = future.result()
        except BaseException as error:
            try:
                if future.done() and not future.cancelled() and future.exception(timeout=0) is error:
                    self._observed.add(future)
                self._retire()
            except BaseException as cleanup:
                BaseException.add_note(error, "read result retirement failed; owner remains retained")
                raise error from cleanup
            raise
        out = tensors.pop(key)
        if uploaded is not None:
            stream = torch.cuda.current_stream(out.device)
            stream.wait_event(uploaded)
            out.record_stream(stream)
        self._retire()
        return out

    def drop(self, keys) -> None:
        """Forget queued tensors nobody will take, so their copies are freed once read."""
        self._worker_control()
        with self._lock:
            self._available()
            for key in keys:
                future = self.ahead.pop(key, None)
                if future is not None:
                    def discard(done, key=key):
                        try:
                            with self._lock:
                                if not done.cancelled() and done.exception() is None:
                                    done.result()[1].pop(key, None)
                                self._retire()
                        except BaseException as error:
                            # Future callbacks otherwise only log exceptions.
                            # Publish the original object without formatting;
                            # close observes it before releasing owned state.
                            with self._lock:
                                if self._discard_error is None:
                                    self._discard_error = error
                    future.add_done_callback(discard)
            self._retire()

    def close(self) -> None:
        """Cancel the reads not started, wait for the rest, and give the uploads' pinned buffers back."""
        if getattr(self._workers, "active", False):
            raise RuntimeError("a read-ahead worker cannot close its own owner")
        with self._condition:
            while self._closing:
                self._condition.wait()
            self._closing = self._shutdown_pending = True
            pool, stream, pending = self.pool, self.stream, tuple(self._owned)
            observed = frozenset(self._observed)
        errors, remaining = [], set()
        joined = pool is None
        try:
            if pool is not None:
                try:
                    pool.shutdown(cancel_futures=True)
                    joined = True
                except BaseException as error:
                    errors.append(error)
            for future in pending:
                try:
                    if not future.cancelled():
                        future.result()
                except BaseException as error:
                    if future not in observed:
                        errors.append(error)
                    try:
                        if not future.done():
                            remaining.add(future)
                        elif not future.cancelled():
                            actual = future.exception(timeout=0)
                            if actual is not None and actual is not error and future not in observed:
                                errors.append(actual)
                    except BaseException as status_error:
                        errors.append(status_error)
                        remaining.add(future)
            synced = stream is None
            if joined and not remaining and stream is not None:
                try:
                    stream.synchronize()
                    synced = True
                    getattr(torch._C, "_host_emptyCache", lambda: None)()
                except BaseException as error:
                    errors.append(error)
            complete = joined and not remaining and synced
            with self._condition:
                if self._discard_error is not None:
                    errors.append(self._discard_error)
                if joined:
                    self.pool = None
                if complete:
                    self.stream = None
                    self.ahead.clear()
                    self._owned.clear()
                    self._observed.clear()
                    self._discard_error = None
                    self._shutdown_pending = False
            if not complete and not errors:
                errors.append(RuntimeError("read-ahead shutdown did not drain its owned work"))
            if errors:
                for _ in errors[1:]:
                    BaseException.add_note(errors[0], "additional read-ahead shutdown failure; owners retained if incomplete")
                raise errors[0]
        finally:
            with self._condition:
                self._closing = False
                self._condition.notify_all()

    def _read(self, path: str, lo: int, hi: int, run: list, device, cut) -> tuple:
        self._workers.active = True
        try:
            return self._read_payload(path, lo, hi, run, device, cut)
        finally:
            self._workers.active = False

    def _read_payload(self, path, lo, hi, run, device, cut):
        if device is None:
            raw = self.reader.read(path, lo, hi - lo)
            return None, {key: cut(raw[b - lo:e - lo], meta) for key, _, b, e, meta in run}
        host = self.reader.read(path, lo, hi - lo, pinned=True)
        with torch.cuda.device(torch.device(device)), torch.cuda.stream(self.stream):
            raw = host.to(device, non_blocking=True)
            out = {key: cut(raw[b - lo:e - lo], meta) for key, _, b, e, meta in run}
            uploaded = torch.cuda.Event()
            uploaded.record(self.stream)
        return uploaded, out


def in_background(job, futures: list) -> None:
    """Start ``job`` on a thread of its own and append its future to ``futures`` (``wait_all`` waits for them)."""

    from concurrent.futures import ThreadPoolExecutor

    pool = ThreadPoolExecutor(1, thread_name_prefix="background-read")
    future = None
    try:
        future = pool.submit(job)
        futures.append(future)
    except BaseException as primary:
        try:
            pool.shutdown(cancel_futures=True)
            if future is not None and not future.cancelled():
                future.result()
        except BaseException as cleanup:
            BaseException.add_note(primary, "unpublished background read cleanup failed")
            raise primary from cleanup
        raise
    else:
        pool.shutdown(wait=False)                    # the thread ends with its one job


def wait_all(futures: list) -> None:
    """Wait for every future (none is left running), then raise the first one's error, if any."""

    errors, remaining = [], []
    for future in futures:
        try:
            future.result()
        except BaseException as error:
            errors.append(error)
            if not future.done():
                remaining.append(future)
    futures[:] = remaining
    if errors:
        for _ in errors[1:]:
            BaseException.add_note(errors[0], "additional background read failure")
        raise errors[0]


__all__ = ["ALIGN", "DTYPES", "PIECE", "ReadAhead", "Reader", "SafeTensors", "in_background", "read_header", "wait_all"]

"""Checkpoint byte ranges -> device tensors through a small ring of pinned staging buffers (the weight load).

Reader threads ``preadv`` file ranges straight into pinned buffers (one kernel copy out of the page cache: no
memory-map page faults, no intermediate host arrays), and the caller's thread issues each buffer's host-to-device
copy asynchronously on the current stream. A buffer is refilled only after the event recorded behind its last copy
has completed. Packed rows use disjoint input and output slices in the same staging buffer.
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

CHUNK = 64 << 20          # bytes per staging buffer
BUFFERS = 4               # staging buffers = reader threads


class Staging:
    """One loader's staging ring. A failed copy ends the load and the caller closes the ring."""

    def __init__(self, chunk: int = CHUNK, buffers: int = BUFFERS):
        self.chunk = chunk
        self.buffers = [torch.empty(chunk, dtype=torch.uint8, pin_memory=True) for _ in range(buffers)]
        self.views = [memoryview(b.numpy()) for b in self.buffers]
        self.events: list[torch.cuda.Event | None] = [None] * buffers
        self.next = 0                                   # the buffer the next batch fills (round robin)
        self.pool = ThreadPoolExecutor(buffers, thread_name_prefix="ds41-load")
        self.fds: dict = {}

    def _fd(self, path) -> int:
        fd = self.fds.get(path)
        if fd is None:
            fd = self.fds[path] = os.open(path, os.O_RDONLY | os.O_CLOEXEC)
        return fd

    def _read(self, slot, fd, pieces, after, pack):
        if after is not None:
            after.synchronize()                         # the buffer's previous copy has read it
        view, at = self.views[slot], 0
        for offset, n in pieces:
            done = 0
            while done < n:
                got = os.preadv(fd, [view[at + done:at + n]], offset + done)
                if got <= 0:
                    raise OSError(f"short read of the checkpoint at byte {offset + done}")
                done += got
            at += n
        if pack is None:
            return 0, at
        return at, pack(self.buffers[slot], at)

    def copy(self, path, ranges, out: torch.Tensor) -> torch.Tensor:
        """Fill contiguous uint8 ``out`` with the file's byte ``ranges`` [(offset, nbytes)], concatenated in order."""
        flat = out.view(-1)
        if out.dtype != torch.uint8 or not out.is_contiguous() or flat.numel() != sum(n for _, n in ranges):
            raise ValueError("staging copies need a contiguous uint8 destination of the ranges' total size")
        batches, batch, size = [], [], 0                # pieces packed into buffer-sized batches, in order
        for offset, n in ranges:
            while n:
                take = min(n, self.chunk - size)
                batch.append((offset, take))
                size += take
                offset, n = offset + take, n - take
                if size == self.chunk:
                    batches.append(batch)
                    batch, size = [], 0
        if batch:
            batches.append(batch)
        self._copy_batches(self._fd(path), batches, flat)
        return out

    def copy_packed_rows(self, path, offset, source_row_bytes, out: torch.Tensor, pack) -> torch.Tensor:
        """Read contiguous source rows and upload packed rows from a disjoint pinned slice."""
        if offset < 0 or source_row_bytes <= 0 or out.ndim < 2 or out.dtype != torch.uint8 or not out.is_contiguous():
            raise ValueError("packed staging requires a row layout and a contiguous uint8 destination")
        row_bytes = out.shape[-1]
        if not row_bytes or self.chunk < source_row_bytes + row_bytes:
            raise ValueError("a source and packed row must fit in one staging buffer")
        rows = out.numel() // row_bytes
        per_batch = self.chunk // (source_row_bytes + row_bytes)
        batches = [[(offset + first * source_row_bytes, min(rows - first, per_batch) * source_row_bytes)]
                   for first in range(0, rows, per_batch)]

        def transform(buffer, source_bytes):
            packed_bytes = source_bytes // source_row_bytes * row_bytes
            pack(buffer[:source_bytes], buffer[source_bytes:source_bytes + packed_bytes])
            return packed_bytes

        self._copy_batches(self._fd(path), batches, out.view(-1), transform)
        return out

    def _copy_batches(self, fd, batches, flat, pack=None):
        pending = deque()

        def submit(i):
            slot = self.next
            self.next = (slot + 1) % len(self.buffers)
            pending.append((slot, self.pool.submit(self._read, slot, fd, batches[i], self.events[slot], pack)))

        ahead = min(len(batches), len(self.buffers))
        for i in range(ahead):
            submit(i)
        position = 0
        for i in range(len(batches)):
            slot, future = pending.popleft()
            start, n = future.result()
            flat[position:position + n].copy_(self.buffers[slot][start:start + n], non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            self.events[slot] = event
            position += n
            if i + ahead < len(batches):
                submit(i + ahead)

    def close(self) -> None:
        """Waits for every copy, then releases the buffers (and torch's cached pinned blocks) and the files."""
        for event in self.events:
            if event is not None:
                event.synchronize()
        self.pool.shutdown(wait=True)
        self.views.clear()
        self.buffers.clear()
        torch._C._host_emptyCache()
        for fd in self.fds.values():
            os.close(fd)
        self.fds.clear()

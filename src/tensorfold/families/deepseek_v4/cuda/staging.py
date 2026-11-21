"""Checkpoint byte ranges -> device tensors through a small ring of pinned staging buffers (the weight load).

Reader threads ``preadv`` file ranges straight into pinned buffers (one kernel copy out of the page cache: no
memory-map page faults, no intermediate host arrays), and the caller's thread issues each buffer's host-to-device
copy asynchronously on the current stream. A buffer is refilled only after the event recorded behind its last copy
has completed, so a copy never reads bytes being overwritten; the bytes uploaded are exactly the file's.
"""

from __future__ import annotations

import os
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import torch

CHUNK = 64 << 20          # bytes per staging buffer
BUFFERS = 4               # staging buffers = reader threads


class Staging:
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

    def _read(self, slot, fd, pieces, after):
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
        return at

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
                    batches.append((batch, size))
                    batch, size = [], 0
        if batch:
            batches.append((batch, size))
        fd = self._fd(path)
        pending = deque()

        def submit(i):
            slot = self.next
            self.next = (slot + 1) % len(self.buffers)
            pending.append((slot, self.pool.submit(self._read, slot, fd, batches[i][0], self.events[slot])))

        ahead = min(len(batches), len(self.buffers))
        for i in range(ahead):
            submit(i)
        position = 0
        for i in range(len(batches)):
            slot, future = pending.popleft()
            n = future.result()
            flat[position:position + n].copy_(self.buffers[slot][:n], non_blocking=True)
            event = torch.cuda.Event()
            event.record()
            self.events[slot] = event
            position += n
            if i + ahead < len(batches):
                submit(i + ahead)
        return out

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

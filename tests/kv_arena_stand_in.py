"""A host stand-in for ``tensorfold.cuda.kv_arena.Arena`` (tests that run qwen3_5's cache policy on CPU tensors).

It keeps the arena contract the policy relies on: a capacity, rows mapped exactly as asked (granularity one row),
``extra`` bytes, an owner, and views that hold the arena alive; each ``view`` registers the arena under its address,
so ``kv_arena.owned`` itself grants in-place growth only to the owner's current view. Its storage moves on
``ensure`` (CPU memory has no virtual-memory mapping here), so tests that check fixed addresses run the real arena
on a GPU (``tests/cuda/test_kv_arena.py``).
"""

from __future__ import annotations

import weakref
from types import SimpleNamespace

import torch


class HostArena:
    def __init__(self, capacity, row_shape, dtype, device, owner=None):
        self.capacity, self.row_shape, self.dtype, self.device, self.owner = capacity, tuple(row_shape), dtype, \
            torch.device(device), owner
        self.row_bytes = int(torch.Size(self.row_shape).numel()) * torch.empty((), dtype=dtype).element_size()
        self.buf = torch.empty((0, *self.row_shape), dtype=dtype, device=self.device)
        self._view = None
        self.drv = SimpleNamespace(check_retirement=lambda: None)

    @property
    def rows(self) -> int:
        return self.buf.shape[0]

    @property
    def mapped(self) -> int:
        return self.rows * self.row_bytes

    def extra(self, rows: int) -> int:
        if rows > self.capacity:
            raise ValueError(f"{rows} rows exceed the arena's {self.capacity}-row reservation")
        return max(0, rows - self.rows) * self.row_bytes

    def ensure(self, rows: int) -> None:
        if self.extra(rows):
            grown = torch.empty((rows, *self.row_shape), dtype=self.dtype, device=self.device)
            grown[:self.rows] = self.buf
            self.buf = grown

    def view(self) -> torch.Tensor:
        from tensorfold.cuda import kv_arena

        out = self.buf[:]
        out._arena = self                     # the view holds the arena alive, as a real arena's views do
        self._view = weakref.ref(out)
        kv_arena._ARENAS[out.data_ptr()] = self
        return out


def install(monkeypatch) -> None:
    """Route ``kv_arena.Arena`` to the host stand-in, with a fresh arena registry, for the calling test."""

    from tensorfold.cuda import kv_arena

    monkeypatch.setattr(kv_arena, "_ARENAS", weakref.WeakValueDictionary())
    monkeypatch.setattr(kv_arena, "Arena", HostArena)

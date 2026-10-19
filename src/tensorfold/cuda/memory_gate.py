"""Stream caches take memory as they grow: the room the startup plan left them, what they hold, and a reserve."""

from __future__ import annotations

from typing import Callable

from .capacity import cuda_limit_bytes


class NoRoom(RuntimeError):
    """A request that can't start until a live stream finishes and frees its caches."""


class MemoryGate:
    """``room`` bytes for every stream's growing caches; ``fits`` also asks the host (MemAvailable on a unified GPU)."""

    def __init__(self, room: int, reserve: int, live: Callable[[], int] | None = None, *,
                 allow_lone_growth: bool = True) -> None:
        self.room, self.reserve, self.held = int(room), int(reserve), 0
        self.live = live
        # Flash Next's default admits a whole lone window separately. An
        # explicitly shared reservation must never borrow outside this gate.
        self.allow_lone_growth = allow_lone_growth
        self.waits = self.ends = 0

    def fits(self, extra: int) -> bool:
        """Whether ``extra`` more bytes can be allocated now (a resize holds its old buffers until the copy lands)."""

        if self.held + int(extra) > self.room - self.reserve:
            return False
        return self.live is None or self.live() >= int(extra) + self.reserve

    def take(self, extra: int) -> None:
        self.held += int(extra)

    def give(self, freed: int) -> None:
        self.held = max(0, self.held - int(freed))


def torch_live(torch, available: Callable, *, unavailable_spare_bytes: int = 0, owner=None) -> Callable[[], int]:
    """Reusable allocator bytes plus free memory, capped by the budget left after current allocations.

    Attention-cache arenas (``kv_arena``) map memory outside the allocator: what they map counts as allocated, and
    gone arenas' queued releases as reusable (the next arena operation releases them before mapping).
    A prepared immutable graph pool can retain inactive blocks that ordinary
    service allocations cannot reuse. Its owner supplies that constant and
    outlives this callable; default callers retain the original accounting.
    """

    from .kv_arena import mapped_bytes, pending_bytes

    if not unavailable_spare_bytes and owner is None:
        # Keep the established zero-exclusion HOT path unchanged for other
        # decoders; select the owned-pool arithmetic once during preparation.
        def room() -> int:
            queued = pending_bytes()
            torch_allocated = int(torch.cuda.memory_allocated())
            allocated = torch_allocated + mapped_bytes() - queued
            free = int(available(torch)) + int(torch.cuda.memory_reserved()) - torch_allocated + queued
            limit = cuda_limit_bytes()
            return max(0, min(free, limit - allocated)) if limit is not None else max(0, free)
        return room

    def room(_owner=owner) -> int:  # the default retains a prepared pool owner without a HOT lookup/callback
        queued = pending_bytes()
        torch_allocated = int(torch.cuda.memory_allocated())
        spare = max(0, int(torch.cuda.memory_reserved()) - torch_allocated - unavailable_spare_bytes)
        free = int(available(torch)) + spare + queued
        allocated = torch_allocated + mapped_bytes() - queued
        limit = cuda_limit_bytes()
        return max(0, min(free, limit - allocated)) if limit is not None else max(0, free)

    return room


__all__ = ["MemoryGate", "NoRoom", "torch_live"]

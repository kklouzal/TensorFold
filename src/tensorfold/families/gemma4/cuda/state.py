"""One sequence's persistent Gemma 4 caches and explicit speculative ownership.

A State has one exclusive forward owner. CUDA operations execute on the owner's
stream; cross-stream handoff requires the caller's event synchronization. All
cache storage is allocated once for the admitted context, so a forward never
reallocates or copies retained keys and captured graphs keep stable addresses.

Sliding layers own a ring of ``ring`` slots (slot = position % ring) holding
RoPE'd K and normalized V. A forward publishes its rows before attending, so the
ring retains window - 1 earlier keys plus the widest forward (``rows``).
Full-attention layers share K and V: only raw key rows and their FP32
reciprocal RMS are stored, one row per position. Snapshots own detached copies,
so branches never alias mutable cache storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .weights import Config

RING_ALIGN = 64
PREFILL_ROWS = 1024      # widest text prefill forward of a request lane
# Attention geometry (see attention.py): absolute key chunks and query tiles.
QUERY_TILE = 16          # (row, head) query pairs a program carries; fixed for every row count
SLIDING_CHUNK = 256      # absolute key chunk of sliding layers
GLOBAL_CHUNK = 512       # absolute key chunk of full-attention layers
SEQUENTIAL_ROWS = 17     # full attention folds in one program per query tile from this many rows


def widest_forward(verify_rows: int, longest_block: int = 0) -> int:
    """Rows a lane's State must accept: a prefill chunk, a verification window or
    one whole visual block (a block is never split across forwards)."""
    return max(PREFILL_ROWS, verify_rows, longest_block)


def sliding_slots(window: int, rows: int, group: int, longest_block: int = 0) -> int:
    """Chunk slots a query tile can touch: its first row's earliest window key
    through its last visible key, which a visual block may extend."""
    span = window - 1 + min(rows, QUERY_TILE // group) + longest_block
    return (span - 2 + SLIDING_CHUNK) // SLIDING_CHUNK + 1


def companion_rings(layer_types, sliding_window: int, block: int, rows: int, window: int) -> tuple[int, ...]:
    """DraftState slots per companion layer (the allocation and admission share this formula): sliding
    rings hold the window plus a proposal block and one append's rows on distinct slots; the full layer
    holds every admitted position."""
    ring = -(-max(sliding_window + block, rows) // RING_ALIGN) * RING_ALIGN
    full = -(-window // RING_ALIGN) * RING_ALIGN
    return tuple(ring if kind == "sliding_attention" else full for kind in layer_types)


@dataclass(frozen=True)
class Snapshot:
    """Read-only owner of detached cache tensors; callers must not mutate _layers.

    ``high_water`` keeps the retained-history bound: ring slots of positions
    below high_water - ring may hold rejected rows written after them.
    """
    position: int
    high_water: int
    context: int
    rows: int
    config: Config
    _layers: tuple[Any, Any, Any, Any]


def ring_slots(config: Config, context: int, rows: int) -> int:
    return min(-(-(config.window + rows) // RING_ALIGN) * RING_ALIGN, -(-context // RING_ALIGN) * RING_ALIGN)


class State:
    def __init__(self, config: Config, device, context: int, rows: int = 16) -> None:
        import torch
        if type(context) is not int or not 1 <= context <= config.max_position:
            raise ValueError("Gemma4 state context must fit the checkpoint's position limit")
        if type(rows) is not int or not 1 <= rows <= context:
            raise ValueError("Gemma4 forward rows must be positive and fit the admitted context")
        self.config, self.device, self.context, self.rows = config, device, context, rows
        self.position = 0
        self.high_water = 0
        self.poisoned = False
        self.ring = ring_slots(config, context, rows)
        sliding = [i for i in range(config.layers) if config.sliding(i)]
        full = [i for i in range(config.layers) if not config.sliding(i)]
        # Layer -> (sliding?, index into that kind's stacked storage).
        self.slot = {i: (True, n) for n, i in enumerate(sliding)}
        self.slot.update({i: (False, n) for n, i in enumerate(full)})
        bf16 = torch.bfloat16
        self.keys = torch.empty((len(sliding), self.ring, config.kv_heads, config.head_dim), dtype=bf16, device=device)
        self.values = torch.empty_like(self.keys)
        self.raw = torch.empty((len(full), context, config.global_kv_heads, config.global_head_dim), dtype=bf16,
                               device=device)
        self.inv = torch.empty((len(full), context, config.global_kv_heads), dtype=torch.float32, device=device)
        # Device copy of the forward's first position: kernels and graphs read it.
        self.pos = torch.zeros((1,), dtype=torch.int32, device=device)

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in (self.keys, self.values, self.raw, self.inv, self.pos))

    def check(self, rows: int) -> None:
        if self.poisoned:
            raise RuntimeError("Gemma4 state is poisoned; restore a snapshot or replace the state")
        if type(rows) is not int or rows <= 0 or rows > self.context - self.position:
            raise ValueError("Gemma4 forward rows exceed the admitted context")
        if rows > self.rows:
            raise ValueError(f"Gemma4 forward of {rows} rows exceeds the state's {self.rows}-row ring reserve")

    def reset(self) -> None:
        """Begin a new sequence in place; stale slots are never read before being rewritten."""
        self.position = self.high_water = 0
        self.poisoned = False

    def _ring_index(self, position: int):
        import torch
        count = min(position, self.ring)
        return torch.arange(position - count, position, device=self.device) % self.ring

    def snapshot(self) -> Snapshot:
        if self.poisoned:
            raise RuntimeError("cannot snapshot a poisoned Gemma4 state")
        p = self.position
        index = self._ring_index(p)
        layers = (self.keys.index_select(1, index), self.values.index_select(1, index),
                  self.raw[:, :p].clone(), self.inv[:, :p].clone())
        return Snapshot(p, self.high_water, self.context, self.rows, self.config, layers)

    def restore(self, snapshot: Snapshot) -> None:
        if (not isinstance(snapshot, Snapshot) or snapshot.config != self.config or snapshot.context != self.context
                or snapshot.rows != self.rows):
            raise ValueError("Gemma4 snapshot belongs to a different cache contract")
        keys, values, raw, inv = snapshot._layers
        p = snapshot.position
        # Publication can be interrupted between copies. Such a state must
        # remain unusable until a complete restore succeeds.
        self.poisoned = True
        index = self._ring_index(p)
        self.keys.index_copy_(1, index, keys)
        self.values.index_copy_(1, index, values)
        self.raw[:, :p].copy_(raw)
        self.inv[:, :p].copy_(inv)
        # The restored ring holds exactly the snapshot's slots, so it inherits its rollback bound.
        self.position, self.high_water = p, snapshot.high_water
        self.poisoned = False

    def advance(self, rows: int) -> None:
        """Commit a completed forward's rows (callers roll back rejected suffixes)."""
        self.position += rows
        self.high_water = max(self.high_water, self.position)

    def rollback(self, position: int) -> None:
        """Keep a committed prefix; reject rollback beyond retained sliding history."""
        if self.poisoned:
            raise RuntimeError("cannot roll back a poisoned Gemma4 state")
        if type(position) is not int or not 0 <= position <= self.position:
            raise ValueError("Gemma4 rollback position must be a committed prefix")
        if self.high_water > self.ring and max(0, position - self.config.window + 1) < self.high_water - self.ring:
            raise ValueError("Gemma4 rollback exceeds retained sliding-window history; restore a snapshot")
        self.position = position


def capacity_bytes(config: Config, context: int, rows: int = 16) -> int:
    """Persistent KV residency of one State; forward scratch and snapshots are additional."""
    ring = ring_slots(config, context, rows)
    total = 0
    for i in range(config.layers):
        if config.sliding(i):
            total += ring * config.kv_heads * config.head_dim * 4
        else:
            total += context * config.global_kv_heads * (config.global_head_dim * 2 + 4)
    return total

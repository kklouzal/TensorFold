"""One sequence's bounded Gemma 4 caches and explicit speculative ownership.

A State has one exclusive forward owner. CUDA operations execute on the owner's
stream; cross-stream handoff requires the caller's event synchronization. Sliding
rings preserve max_rows speculative tokens in addition to the visible window.
Snapshots own detached copies, so branches never alias mutable cache storage.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .weights import Config


@dataclass(frozen=True)
class Snapshot:
    """Read-only owner of detached cache tensors; callers must not mutate _layers."""
    position: int
    context: int
    max_rows: int
    config: Config
    _layers: tuple[tuple[Any, Any, int], ...]


class _Cache:
    def __init__(self, heads: int, dim: int, ring: int, device) -> None:
        import torch
        self.heads, self.dim, self.ring, self.device = heads, dim, ring, device
        self.keys = torch.empty((0, heads, dim), dtype=torch.bfloat16, device=device)
        self.values = torch.empty_like(self.keys)
        self.high_water = 0

    def before(self, position: int, window: int) -> tuple[Any, Any, int]:
        """Chronological committed rows, with the absolute position of row zero."""
        import torch
        first = max(0, position - window + 1) if self.ring else 0
        if not self.ring or position <= self.ring:
            return self.keys[first:position], self.values[first:position], first
        slot, count = first % self.ring, position - first
        if slot + count <= self.ring:
            return self.keys[slot:slot + count], self.values[slot:slot + count], first
        end = slot + count - self.ring
        return (torch.cat((self.keys[slot:], self.keys[:end])),
                torch.cat((self.values[slot:], self.values[:end])), first)

    def write(self, keys, values, position: int, context: int) -> None:
        import torch
        end = position + keys.shape[0]
        need = min(self.ring, end) if self.ring else end
        if self.keys.shape[0] < need:
            capacity = min(self.ring or context, max(128, ((need + 511) // 512) * 512))
            new_k = torch.empty((capacity, self.heads, self.dim), dtype=self.keys.dtype, device=self.device)
            new_v = torch.empty_like(new_k)
            keep = min(position, self.keys.shape[0])
            new_k[:keep].copy_(self.keys[:keep])
            new_v[:keep].copy_(self.values[:keep])
            self.keys, self.values = new_k, new_v
        if not self.ring:
            self.keys[position:end].copy_(keys)
            self.values[position:end].copy_(values)
        else:
            count = min(keys.shape[0], self.ring)
            start = end - count
            slot = start % self.ring
            first = min(count, self.ring - slot)
            self.keys[slot:slot + first].copy_(keys[-count:][:first])
            self.values[slot:slot + first].copy_(values[-count:][:first])
            if first < count:
                self.keys[:count - first].copy_(keys[-count + first:])
                self.values[:count - first].copy_(values[-count + first:])
        self.high_water = max(self.high_water, end)


class State:
    def __init__(self, config: Config, device, context: int, max_rows: int = 16) -> None:
        if type(context) is not int or not 1 <= context <= config.max_position:
            raise ValueError("Gemma4 state context must fit the checkpoint's position limit")
        if type(max_rows) is not int or not 1 <= max_rows <= context:
            raise ValueError("Gemma4 max_rows must be positive and fit the admitted context")
        self.config, self.device, self.context, self.max_rows = config, device, context, max_rows
        self.position = 0
        self.poisoned = False
        self.kv = []
        for layer in range(config.layers):
            heads, dim = config.attention_geometry(layer)
            ring = min(context, config.window + max_rows) if config.sliding(layer) else 0
            self.kv.append(_Cache(heads, dim, ring, device))

    @property
    def pos(self) -> int:
        return self.position

    @property
    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for c in self.kv for t in (c.keys, c.values))

    def check(self, rows: int) -> None:
        if self.poisoned:
            raise RuntimeError("Gemma4 state is poisoned; restore a snapshot or replace the state")
        if type(rows) is not int or rows <= 0 or rows > self.context - self.position:
            raise ValueError("Gemma4 forward rows exceed the admitted context")

    def _detached_layers(self) -> tuple[tuple[Any, Any, int], ...]:
        if self.poisoned:
            raise RuntimeError("cannot snapshot a poisoned Gemma4 state")
        held = []
        for cache in self.kv:
            count = min(self.position, cache.ring) if cache.ring else self.position
            held.append((cache.keys[:count].clone(), cache.values[:count].clone(), cache.high_water))
        return tuple(held)

    def snapshot(self) -> Snapshot:
        return Snapshot(self.position, self.context, self.max_rows, self.config, self._detached_layers())

    def restore(self, snapshot: Snapshot) -> None:
        if (not isinstance(snapshot, Snapshot) or snapshot.config != self.config or snapshot.context != self.context
                or snapshot.max_rows != self.max_rows or len(snapshot._layers) != len(self.kv)):
            raise ValueError("Gemma4 snapshot belongs to a different cache contract")
        replacements = [(keys.clone().to(self.device), values.clone().to(self.device), high)
                        for keys, values, high in snapshot._layers]
        # Publication can be interrupted between layers. Such a state must
        # remain unusable until a complete restore succeeds.
        self.poisoned = True
        for cache, (keys, values, high) in zip(self.kv, replacements):
            cache.keys, cache.values, cache.high_water = keys, values, high
        self.position, self.poisoned = snapshot.position, False

    def fork(self) -> "State":
        other = State(self.config, self.device, self.context, self.max_rows)
        # Transfer newly detached copies directly; going through a public
        # snapshot plus restore would clone the complete prefix twice.
        for cache, (keys, values, high) in zip(other.kv, self._detached_layers()):
            cache.keys, cache.values, cache.high_water = keys, values, high
        other.position = self.position
        return other

    def rollback(self, position: int) -> None:
        """Keep a committed prefix; reject rollback beyond retained sliding history."""
        if self.poisoned:
            raise RuntimeError("cannot roll back a poisoned Gemma4 state")
        if type(position) is not int or not 0 <= position <= self.position:
            raise ValueError("Gemma4 rollback position must be a committed prefix")
        for cache in self.kv:
            if cache.ring and cache.high_water > cache.ring:
                first_needed = max(0, position - self.config.window + 1)
                if first_needed < cache.high_water - cache.ring:
                    raise ValueError("Gemma4 rollback exceeds retained sliding-window history; restore a snapshot")
        self.position = position


def capacity_bytes(config: Config, context: int, max_rows: int = 16) -> int:
    """Worst-case BF16 KV residency; current forward scratch/snapshots are additional."""
    total = 0
    for i in range(config.layers):
        heads, dim = config.attention_geometry(i)
        rows = min(context, config.window + max_rows) if config.sliding(i) else context
        total += rows * heads * dim * 4
    return total

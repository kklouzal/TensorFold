"""Independent scalar reference for the bounded expert-cache replacement contract.

This test/benchmark oracle keeps the original per-layer byte arrays, Python
recency list and tuple comparison. It shares no ranking or history layout with
the optimized policy. Cache callers own validation of touched keys and slot
operations, and retain the same bounded-clock and immutable-source contract.
"""

from __future__ import annotations

import torch

_HALVE = bytes(value // 2 for value in range(256))


def same_tensor_bits(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare logical tensor bytes, including signed zero and exact NaN payloads."""

    if left.shape != right.shape or left.dtype != right.dtype:
        return False
    left_bytes = left.detach().contiguous().reshape(-1).view(torch.uint8)
    right_bytes = right.detach().contiguous().reshape(-1).view(torch.uint8)
    return torch.equal(left_bytes, right_bytes)


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


class ScalarPolicy:
    """Aging LFU, then LRU, then physical slot; exact saturating byte scores."""

    def __init__(self, capacity: int) -> None:
        self.capacity = _integer(capacity, "capacity")
        self.frequency: dict[int, bytearray] = {}
        self.keys: list[tuple[int, int] | None] = [None] * capacity
        self.resident: dict[tuple[int, int], int] = {}
        self.recency = [0] * capacity
        self.tick = self.touches = 0
        self.interval = 64 * capacity

    def register(self, layer: int, count: int) -> None:
        _integer(layer, "layer_id", 0)
        _integer(count, "expert count")
        if layer in self.frequency:
            raise ValueError(f"expert layer {layer} is already registered")
        self.frequency[layer] = bytearray(count)

    def touch(self, keys: list[tuple[int, int]]) -> None:
        if not keys:
            return
        for layer, expert in keys:
            self.touches += 1
            if self.touches == self.interval:
                for counters in self.frequency.values():
                    counters[:] = counters.translate(_HALVE)
                self.touches = 0
                ranks = {stamp: rank for rank, stamp in enumerate(sorted(set(self.recency)))}
                self.recency[:] = [ranks[stamp] for stamp in self.recency]
                self.tick = len(ranks)
            counters = self.frequency[layer]
            value = counters[expert]
            if value < 255:
                counters[expert] = value + 1
        self.tick += 1

    def victim(self, protected: set[int]) -> int:
        candidates = [slot for slot in range(self.capacity) if slot not in protected]
        if not candidates:
            raise ValueError("expert lease exceeds the fixed hot-pool capacity")
        return min(candidates, key=lambda slot: (-1 if self.keys[slot] is None else
            self.frequency[self.keys[slot][0]][self.keys[slot][1]], self.recency[slot], slot))

    def remove(self, slot: int) -> None:
        key = self.keys[slot]
        if key is not None:
            del self.resident[key]
        self.keys[slot] = None

    def install(self, slot: int, key: tuple[int, int]) -> None:
        self.keys[slot] = key
        self.resident[key] = slot
        self.recency[slot] = self.tick


__all__ = ["ScalarPolicy", "same_tensor_bits"]

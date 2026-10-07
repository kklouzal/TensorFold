"""A model-owned host-authoritative expert cache with a fixed CUDA hot pool.

Registered payloads share one layout; their expert counts may differ. CPU
tensors and all their aliases must remain immutable until ``close``. Leases
serialize host metadata and lend CUDA views only until their context exits;
callers enqueue every consumer on that context's current stream. A completion
event orders a different stream before reuse. CPU staging waits for its own
H2D event before overwrite. No quantization, routing or graph capture occurs.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import threading
from typing import Iterator

import numpy as np
import torch

_HALVE = bytes(value >> 1 for value in range(256))
# Matched GB10 CPU replay crosses from scalar to NumPy ranking at 27 cells.
# Complete forward comparisons include the cost of protecting each window.
_VECTOR_MIN_CAPACITY = 64
_BLOCKED = np.iinfo(np.int64).max


def _integer(value: int, name: str, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class _Field:
    shape: tuple[int, ...]
    dtype: torch.dtype
    offset: int
    size: int


def _layout(payloads: tuple[torch.Tensor, ...]) -> tuple[int, tuple[_Field, ...], int]:
    if not isinstance(payloads, tuple) or not payloads:
        raise ValueError("payloads must be a nonempty tuple of CPU tensors")
    count, offset, fields = None, 0, []
    for tensor in payloads:
        if not isinstance(tensor, torch.Tensor) or tensor.device.type != "cpu" or tensor.ndim < 2 or \
                tensor.layout != torch.strided or tensor.is_quantized or not tensor.is_contiguous() or tensor.requires_grad:
            raise ValueError("expert payloads must be contiguous, non-gradient CPU tensors with an expert dimension")
        if tensor.shape[0] <= 0 or tensor[0].numel() <= 0:
            raise ValueError("expert payloads cannot have empty dimensions")
        if count is not None and count != tensor.shape[0]:
            raise ValueError("expert payload counts disagree")
        count = tensor.shape[0]
        offset = -(-offset // 16) * 16
        size = tensor[0].numel() * tensor.element_size()
        fields.append(_Field(tuple(tensor.shape[1:]), tensor.dtype, offset, size))
        offset += size
    return -(-offset // 16) * 16, tuple(fields), count


def entry_bytes_for(payloads: tuple[torch.Tensor, ...]) -> int:
    """16-byte-aligned per-expert bytes, including payload padding; allocates nothing."""

    return _layout(payloads)[0]


class _Policy:
    """Bounded aging LFU with deterministic LRU/physical-slot tie breaks.

    One contiguous byte per logical expert retains evicted history, plus a
    zero sentinel. Registration rebinds private layer views; callers cannot
    retain those views across registration. A small pool uses list recencies
    and scalar ranking. Larger pools use bounded int64 recencies and 26 bytes
    of resident arrays per cell, with one byte per cell temporarily buffered
    by ``np.take`` and at most capacity native indices for bulk protection.
    The victim strategy is selected once at construction.

    Aging every 64*capacity touches avoids permanent saturation. Ranking
    recencies then preserves every tie/order; touches < interval and tick <=
    interval+capacity+1. The composite frequency/recency key fits int64 under
    the cache's signed-int32 capacity bound. Cache ownership serializes all
    mutations, and touched/installed keys are validated at its lease boundary.
    """

    def __init__(self, capacity: int) -> None:
        self.capacity = _integer(capacity, "capacity")
        if capacity > 2**31 - 1:
            raise ValueError("policy capacity exceeds signed 32-bit expert indexing bounds")
        self.frequency: dict[int, memoryview] = {}
        self._ranges: dict[int, tuple[int, int]] = {}
        self._flat = bytearray(1)
        self._flat_view = None
        self.keys: list[tuple[int, int] | None] = [None] * capacity
        self.resident: dict[tuple[int, int], int] = {}
        self.tick = self.touches = 0
        self.interval = 64 * capacity
        self._stride = self.interval + capacity + 2
        self._logical = self._scores = self._empty = self._scratch = None
        if capacity < _VECTOR_MIN_CAPACITY:
            self.recency = [0] * capacity
            self.victim = self._scalar_victim
        else:
            self.recency = np.zeros(capacity, dtype=np.int64)
            self._logical = np.zeros(capacity, dtype=np.int64)
            self._scores = np.empty(capacity, dtype=np.uint8)
            self._empty = np.empty(capacity, dtype=np.bool_)
            self._scratch = np.empty(capacity, dtype=np.int64)
            self.victim = self._vector_victim

    def register(self, layer: int, count: int) -> None:
        _integer(layer, "layer_id", 0)
        _integer(count, "expert count")
        if layer in self.frequency:
            raise ValueError(f"expert layer {layer} is already registered")
        # Build before publishing: allocation failure preserves all old state.
        # Offsets stay stable so resident logical indices survive rebinding.
        ranges = dict(self._ranges)
        ranges[layer] = (len(self._flat), count)
        flat = self._flat + bytearray(count)
        view = None if self._logical is None else np.frombuffer(flat, dtype=np.uint8)
        frequency = {layer: memoryview(flat)[start:start + count] for layer, (start, count) in ranges.items()}
        self._flat, self._flat_view, self._ranges, self.frequency = flat, view, ranges, frequency

    def touch(self, keys: list[tuple[int, int]]) -> None:
        if not keys:
            return
        for layer, expert in keys:
            self.touches += 1
            if self.touches == self.interval:
                self._flat[:] = self._flat.translate(_HALVE)
                self.touches = 0
                if self._logical is None:
                    ranks = {stamp: rank for rank, stamp in enumerate(sorted(set(self.recency)))}
                    self.recency[:] = [ranks[stamp] for stamp in self.recency]
                    self.tick = len(ranks)
                else:
                    _, ranks = np.unique(self.recency, return_inverse=True)
                    self.recency[:] = ranks
                    self.tick = int(ranks.max()) + 1
            counters = self.frequency[layer]
            value = counters[expert]
            if value < 255:
                counters[expert] = value + 1
        self.tick += 1

    def _scalar_victim(self, protected: set[int]) -> int:
        candidates = [slot for slot in range(self.capacity) if slot not in protected]
        if not candidates:
            raise ValueError("expert lease exceeds the fixed hot-pool capacity")
        return min(candidates, key=lambda slot: (-1 if self.keys[slot] is None else
            self.frequency[self.keys[slot][0]][self.keys[slot][1]], self.recency[slot], slot))

    def _vector_victim(self, protected: set[int]) -> int:
        # Flat indices are proven in-range by register/install and lease
        # validation. mode='raise' retains invariant-failure detection.
        np.take(self._flat_view, self._logical, out=self._scores)
        np.multiply(self._scores, self._stride, out=self._scratch, dtype=np.int64)
        np.equal(self._logical, 0, out=self._empty)
        self._scratch[self._empty] = -self._stride
        np.add(self._scratch, self.recency, out=self._scratch)
        if len(protected) < 16:
            for slot in protected:
                self._scratch[slot] = _BLOCKED
        else:
            indices = np.fromiter(protected, dtype=np.intp, count=len(protected))
            self._scratch[indices] = _BLOCKED
        # argmin chooses the first physical cell on equal composite keys.
        slot = int(self._scratch.argmin())
        if self._scratch[slot] == _BLOCKED:
            raise ValueError("expert lease exceeds the fixed hot-pool capacity")
        return slot

    def remove(self, slot: int) -> None:
        key = self.keys[slot]
        if key is not None:
            del self.resident[key]
        self.keys[slot] = None
        if self._logical is not None:
            self._logical[slot] = 0

    def install(self, slot: int, key: tuple[int, int]) -> None:
        self.keys[slot] = key
        self.resident[key] = slot
        self.recency[slot] = self.tick
        if self._logical is not None:
            self._logical[slot] = self._ranges[key[0]][0] + key[1]


@dataclass
class _Layer:
    source: tuple[torch.Tensor, ...]
    versions: tuple[int | None, ...]
    views: tuple[torch.Tensor, ...]
    fields: tuple[_Field, ...]
    count: int


class HostExpertCache:
    """Fixed global CUDA cells and a bounded pinned staging ring.

    ``gpu_bytes`` is a hard bound on owned tensor storage, not allocator or
    CUDA-context overhead. Actual allocation is ``capacity * entry_bytes``;
    pinned storage is exactly ``staging_slots * entry_bytes``. All layers use
    one payload layout so each CUDA payload stays fully contiguous, including
    its expert dimension. Registration transfers immutable CPU authority.

    A scheduling/copy failure poisons this cache; further leases fail rather
    than serve incomplete data. ``close`` waits for owned CUDA work and drops
    buffers. Borrowed views must never be retained or consumed outside a lease.
    """

    capturable = False

    def __init__(self, gpu_bytes: int, entry_bytes: int, device: torch.device | str, *,
                 staging_slots: int = 2) -> None:
        self.budget_bytes = _integer(gpu_bytes, "gpu_bytes")
        self.entry_bytes = _integer(entry_bytes, "entry_bytes")
        self.staging_slots = _integer(staging_slots, "staging_slots")
        if entry_bytes % 16 or gpu_bytes < entry_bytes:
            raise ValueError("entry_bytes must be 16-byte aligned and fit the GPU budget")
        self.capacity = gpu_bytes // entry_bytes
        self.gpu_bytes = self.capacity * entry_bytes
        self.pinned_bytes = staging_slots * entry_bytes
        if max(self.gpu_bytes, self.pinned_bytes, entry_bytes) > 2**63 - 1 or self.capacity > 2**31 - 1:
            raise ValueError("expert cache exceeds tensor-storage or signed 32-bit expert indexing bounds")
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("host expert cache requires a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if device.index is None else device.index)
        self._lock = threading.RLock()
        self._active = self._closed = False
        self._failure: str | None = None
        self._layers: dict[int, _Layer] = {}
        self._fields: tuple[_Field, ...] | None = None
        self._policy = _Policy(self.capacity)
        # Mutable cache buffers survive caller inference-mode scopes; create
        # ordinary tensors so a later lease can update them in either mode.
        with torch.inference_mode(False):
            self._pool = torch.empty(self.gpu_bytes, dtype=torch.uint8, device=self.device)
            self._staging = torch.empty((staging_slots, entry_bytes), dtype=torch.uint8, pin_memory=True)
        self._stage_events = [torch.cuda.Event() for _ in range(staging_slots)]
        self._stage_pending = [False] * staging_slots
        self._next_stage = 0
        self._last_use = torch.cuda.Event()
        self._last_stream = None
        self.hits = self.misses = self.copied_bytes = self.evictions = 0

    @property
    def host_bytes(self) -> int:
        """Authoritative CPU payload bytes; counts each registration once."""

        return sum(t.numel() * t.element_size() for layer in self._layers.values() for t in layer.source)

    @property
    def score_history_entries(self) -> int:
        """Exactly one byte of frequency history per registered logical expert."""

        return sum(len(values) for values in self._policy.frequency.values())

    def _usable(self) -> None:
        if self._closed:
            raise RuntimeError("host expert cache is closed")
        if self._failure is not None:
            raise RuntimeError(f"host expert cache is unusable after scheduling failure: {self._failure}")
        if self._active:
            raise RuntimeError("nested expert leases or registration during a lease are unsupported")

    def register(self, layer_id: int, payloads: tuple[torch.Tensor, ...]) -> None:
        size, fields, count = _layout(payloads)
        with self._lock:
            self._usable()
            if size > self.entry_bytes or (self._fields is not None and fields != self._fields):
                raise ValueError("expert layer layout differs from the fixed hot-pool layout")
            # Views created in inference mode inherit that mode even when the
            # arena is ordinary. They must remain writable in future leases.
            with torch.inference_mode(False):
                views = tuple(self._pool.narrow(0, self.capacity * field.offset, self.capacity * field.size)
                              .view(field.dtype).view(self.capacity, *field.shape) for field in fields)
            self._policy.register(layer_id, count)
            # Inference tensors deliberately have no version counter. Their
            # aliases obey the same immutable-authority contract as all sources.
            versions = tuple(None if torch.is_inference(t) else t._version for t in payloads)
            self._layers[layer_id] = _Layer(payloads, versions, views, fields, count)
            self._fields = fields

    def _copy(self, layer: _Layer, expert: int, slot: int, stream) -> None:
        staging = self._next_stage
        self._next_stage = (staging + 1) % self.staging_slots
        event = self._stage_events[staging]
        if self._stage_pending[staging] and not event.query():
            event.synchronize()  # CPU must not overwrite an in-flight pinned H2D source
        for source, target, field in zip(layer.source, layer.views, layer.fields):
            pinned = self._staging[staging].narrow(0, field.offset, field.size).view(field.dtype).view(field.shape)
            pinned.copy_(source[expert])
            target[slot].copy_(pinned, non_blocking=True)
        event.record(stream)
        self._stage_pending[staging] = True

    @contextmanager
    def lease(self, layer_id: int, expert_ids: list[int]) -> Iterator[tuple[tuple[torch.Tensor, ...], dict[int, int]]]:
        """Lend requested logical experts until every consumer is queued on the current stream.

        IDs must be distinct and fit ``capacity``; the caller windows larger
        unions. A shared expert is an ordinary logical ID and consumes one cell.
        Empty leases are valid. Nested leases fail explicitly rather than hang.
        """

        with self._lock, torch.cuda.device(self.device):
            self._usable()
            _integer(layer_id, "layer_id", 0)
            layer = self._layers.get(layer_id)
            if layer is None:
                raise ValueError(f"unknown expert layer {layer_id}")
            if not isinstance(expert_ids, list) or len(expert_ids) > self.capacity:
                raise ValueError("expert IDs must be a list fitting the fixed hot-pool capacity")
            if any(isinstance(i, bool) or not isinstance(i, int) or
                    not 0 <= i < layer.count for i in expert_ids) or len(set(expert_ids)) != len(expert_ids):
                raise ValueError("expert IDs must be a list of distinct in-range integers")
            if tuple(None if torch.is_inference(t) else t._version for t in layer.source) != layer.versions:
                raise ValueError("registered CPU expert tensors were mutated")
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("host-backed expert leases cannot be captured in a CUDA graph")
            stream = torch.cuda.current_stream(self.device)
            self._active = True
            keys = [(layer_id, expert) for expert in expert_ids]
            mapping: dict[int, int] = {}
            primary = None
            try:
                try:
                    if self._last_stream is not None and stream != self._last_stream:
                        stream.wait_event(self._last_use)
                    self._policy.touch(keys)
                    protected = {self._policy.resident[key] for key in keys if key in self._policy.resident}
                    for key in keys:
                        slot = self._policy.resident.get(key)
                        if slot is None:
                            slot = self._policy.victim(protected)
                            if self._policy.keys[slot] is not None:
                                self.evictions += 1
                            self._policy.remove(slot)
                            self._copy(layer, key[1], slot, stream)
                            self._policy.install(slot, key)
                            self.misses += 1
                            self.copied_bytes += sum(field.size for field in layer.fields)
                        else:
                            self.hits += 1
                            self._policy.recency[slot] = self._policy.tick
                        protected.add(slot)
                        mapping[key[1]] = slot
                except BaseException as error:
                    self._failure = repr(error)
                    raise
                yield layer.views, mapping
            except BaseException as error:
                primary = error
                raise
            finally:
                try:
                    self._last_use.record(stream)
                except BaseException as error:
                    self._failure = repr(error)
                    if primary is not None:
                        raise primary from error
                    raise
                finally:
                    self._last_stream = stream
                    self._active = False

    def close(self) -> None:
        """Finish owned GPU work before releasing staging and cache storage; idempotent."""

        with self._lock:
            if self._active:
                raise RuntimeError("cannot close an active expert lease")
            if self._closed:
                return
            if self._last_stream is not None:
                if self._failure is not None:
                    self._last_stream.synchronize()
                else:
                    self._last_use.synchronize()
            self._layers.clear()
            self._pool = self._staging = None
            self._closed = True


class CachedExperts:
    """Affine4 grouped-kernel adapter; CPU ``Experts`` remains authoritative."""

    kernel = "qmm"
    capturable = False

    def __init__(self, experts, cache: HostExpertCache, layer_id: int) -> None:
        self.cache, self.layer_id = cache, layer_id
        self.gs, self.width, self.dims, self.limit = experts.gs, experts.width, experts.dims, experts.limit
        self.swiglu, self.logical_experts = experts.swiglu, experts.count
        cache.register(layer_id, (experts.up, experts.down))

    @property
    def count(self) -> int:
        return self.logical_experts

    @property
    def capacity(self) -> int:
        return self.cache.capacity

    def bytes_per_expert(self) -> int:
        return self.cache.entry_bytes

    @contextmanager
    def lease(self, expert_ids: list[int]):
        from tensorfold.cuda.experts import Experts

        with self.cache.lease(self.layer_id, expert_ids) as (payloads, mapping):
            hot = Experts(*payloads, self.gs, self.width, self.dims, self.limit)
            yield hot, mapping


__all__ = ["HostExpertCache", "CachedExperts", "entry_bytes_for"]

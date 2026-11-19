"""Host-RAM tier for the automatic prefix cache: spill kept prompt points instead of dropping them.

The concurrent decoder keeps prompt points (``MultiDecoder.kept``: ids, slot State, snapshot, MTP tail) in idle
full-window slots. When value-based eviction drops a point (its slot is reused, it falls past ``keep``, or a resume
in place rewrites it), ``spill`` copies that slot's committed rows below the point and the point's complete snapshot
into a bounded, pinned per-rank host slab. ``restore`` copies them back into a free slot whose unique-history page
ownership is identical (same slot residue for KV owners and the same indexer owner of every layer), so every rank
copies only its own pages: no collective, and the restored rows and snapshot are bit-identical to the VRAM-resident
point (the request then runs exactly as a resume of that point would).

Determinism across tensor-parallel ranks: every decision here is a pure function of the replicated kept tables, the
request and rank-identical byte counts (rows from the page placement; each point shape's size agreed once by an
all-gather), so all ranks hold the same entries in the same blocks.
The cache key is the ordered token IDs, the request principal (FIFO), and the State ``kv_identity`` (model KV format,
history/indexer layout, logical context); the process's weights, prefix policy and tokenizer are fixed for its
lifetime and nothing is persisted. Memory: one pinned slab of ``capacity`` bytes per rank, allocated at startup and
charged to the host commitment; entries take fixed blocks, so allocation never fails once admitted.

Off unless ``--prefix-host-bytes`` is set: under a stream of unique prompts every eviction spills, which cost 1-2%
of prefill/decode throughput on the 8xV100 production benchmark even with the copies on a side stream (the
coordinator's packing work), while a repeated or extended prompt restores in milliseconds instead of seconds.

Cost model (SC117, Q4 KV, TP8): restoring n tokens copies about 1 KiB/token/rank plus a ~14.6 MB recurrent snapshot
per point over PCIe (~12 GB/s), about 1.2 ms + 85 ns/token, against ~364 us/token of prefill: any point a request can
use beyond its best VRAM point by ``MIN_GAIN`` tokens is restored.
"""

from __future__ import annotations

import contextlib
import copy

import torch

BLOCK_BYTES = 1 << 21        # fixed allocation unit of the pinned slab
MIN_GAIN = 64                # tokens a host point must save beyond the best VRAM point (~23 ms of prefill)


class _Region:
    """A byte range of an entry's blocks, laid out sequentially; copies split at block boundaries."""

    __slots__ = ("dtype", "nbytes", "offset", "shape")

    def __init__(self, offset, nbytes, dtype, shape):
        self.offset, self.nbytes, self.dtype, self.shape = offset, nbytes, dtype, shape


def _bytes_of(tensor):
    return tensor.reshape(-1).view(torch.uint8)


def await_spill(st):
    """Order a slot's next write after a pending spill of its rows (a GPU-side wait, no host sync)."""
    ready = getattr(st, "host_spill_ready", None)
    if ready is not None:
        torch.cuda.current_stream().wait_event(ready)
        st.host_spill_ready = None


class HostPrefixTier:
    """One rank's pinned host slab and the rank-replicated host entry table."""

    def __init__(self, capacity_bytes: int, *, device, comm=None, block_bytes: int = BLOCK_BYTES, pin: bool = True):
        if type(capacity_bytes) is not int or capacity_bytes < block_bytes:
            raise ValueError("a host prefix tier needs at least one block of explicit bytes")
        self.block_bytes = block_bytes
        self.blocks = capacity_bytes // block_bytes
        self.capacity = self.blocks * block_bytes
        self.slab = torch.empty(self.capacity, dtype=torch.uint8, pin_memory=pin)
        self.device = torch.device(device)
        # Spills copy on their own stream, after everything already queued on the coordinator's stream, so decode
        # and prefill keep running; a slot's next writer and a restore of the entry wait on the spill's event.
        self.stream = torch.cuda.Stream(self.device) if self.device.type == "cuda" else None
        self.free = list(range(self.blocks - 1, -1, -1))      # pop() hands out ascending blocks
        self.entries: list[dict] = []
        self.device, self.comm = torch.device(device), comm
        self.clock = 0.0
        self._point_bounds = {}
        self.counters = {"spills": 0, "spill_bytes": 0, "spill_skipped": 0, "restores": 0, "restore_bytes": 0,
                         "restored_tokens": 0, "evictions": 0, "lookups": 0, "hits": 0}

    # ------------------------------------------------------------------------------------------------- layout
    def _views(self, blocks, offset, nbytes):
        """(slab view, start within the region) pieces covering ``nbytes`` at ``offset`` of the entry's blocks."""
        done = 0
        while done < nbytes:
            index, within = divmod(offset + done, self.block_bytes)
            take = min(nbytes - done, self.block_bytes - within)
            start = blocks[index] * self.block_bytes + within
            yield self.slab[start:start + take], done
            done += take

    def _put(self, blocks, cursor, tensor):
        source = _bytes_of(tensor.contiguous())
        if self.stream is not None:
            # The copy reads this device memory on the side stream: keep it from reuse until that work is done.
            source.record_stream(self.stream)
        region = _Region(cursor[0], source.numel(), tensor.dtype, tuple(tensor.shape))
        for view, at in self._views(blocks, region.offset, region.nbytes):
            view.copy_(source[at:at + view.numel()], non_blocking=True)
        cursor[0] += region.nbytes
        return region

    def _get_into(self, blocks, region, target):
        if not target.is_contiguous():
            raise RuntimeError("host prefix restore targets must be contiguous cache rows")
        flat = _bytes_of(target)
        if flat.numel() != region.nbytes:
            raise RuntimeError("host prefix region differs from its restore target")
        for view, at in self._views(blocks, region.offset, region.nbytes):
            flat[at:at + view.numel()].copy_(view, non_blocking=True)

    def _get(self, blocks, region):
        target = torch.empty(region.shape, dtype=region.dtype, device=self.device)
        self._get_into(blocks, region, target)
        return target

    # ------------------------------------------------------------------------------------------- snapshots
    def _local(self, tensor):
        """A tensor on this rank's device, which the slab stores (others are host values, copied as such)."""
        return tensor.device.type == self.device.type and (
            self.device.index is None or tensor.device.index in (None, self.device.index))

    def _pack(self, blocks, cursor, value):
        """Deep structure of a snapshot with every tensor stored in the slab (device tensors only)."""
        if isinstance(value, torch.Tensor):
            if not self._local(value):
                return ("value", value.clone())
            return ("tensor", self._put(blocks, cursor, value))
        if isinstance(value, dict):
            return ("dict", {key: self._pack(blocks, cursor, item) for key, item in value.items()})
        if isinstance(value, tuple):
            return ("tuple", tuple(self._pack(blocks, cursor, item) for item in value))
        if isinstance(value, list):
            return ("list", [self._pack(blocks, cursor, item) for item in value])
        return ("value", copy.deepcopy(value))

    def _unpack(self, blocks, packed):
        kind, value = packed
        if kind == "tensor":
            return self._get(blocks, value)
        if kind == "dict":
            return {key: self._unpack(blocks, item) for key, item in value.items()}
        if kind == "tuple":
            return tuple(self._unpack(blocks, item) for item in value)
        if kind == "list":
            return [self._unpack(blocks, item) for item in value]
        return copy.deepcopy(value)

    def _tensor_bytes(self, value):
        if isinstance(value, torch.Tensor):
            return value.numel() * value.element_size() if self._local(value) else 0
        if isinstance(value, dict):
            return sum(self._tensor_bytes(item) for item in value.values())
        if isinstance(value, (list, tuple)):
            return sum(self._tensor_bytes(item) for item in value)
        return 0

    # ------------------------------------------------------------------------------------------- history
    @staticmethod
    def _layers(st, rows, mtp_rows):
        layers = [(ordinal, st.kc[ordinal], st.pooled[ordinal], rows) for ordinal in range(len(st.kc))]
        if hasattr(st, "mtp_kc") and mtp_rows:
            layers.append((len(st.kc), st.mtp_kc, st.mtp_pooled, mtp_rows))
        return layers

    @staticmethod
    def _runs(cache, rows, page_tokens):
        """Maximal runs of this rank's pages below ``rows`` that are contiguous in its arena: (first row, offset,
        count)."""
        runs = []
        for page in range((rows + page_tokens - 1) // page_tokens):
            offset = cache.offsets[page]
            if offset < 0:
                continue
            count = min(page_tokens, rows - page * page_tokens)
            if runs and runs[-1][0] + runs[-1][2] == page * page_tokens and runs[-1][1] + runs[-1][2] == offset:
                runs[-1][2] += count
            else:
                runs.append([page * page_tokens, offset, count])
        return runs

    @staticmethod
    def _cache_tensors(cache):
        tensors = [cache.k, cache.v]
        if cache.key_format.quantized:
            tensors.append(cache.ks)
        if cache.value_format.quantized:
            tensors.append(cache.vs)
        return tensors

    @staticmethod
    def owner_class(placement, slot, layers):
        """What decides which rank owns each page/pool of a slot: equal classes restore rank-locally."""
        return ((slot % placement.group_size, placement.degree)
                + tuple(placement.indexer_owner(slot, layer, include_mtp=layers > placement.target_layers)
                        for layer in range(layers)))

    @staticmethod
    def _history_bound(st, rows, mtp_rows):
        """Bytes the largest rank copies for these rows: the same on every rank, from the page placement alone."""
        import math

        placement = st.history_placement
        total = 0
        for ordinal, cache, pool, count in HostPrefixTier._layers(st, rows, mtp_rows):
            row_bytes = sum(math.prod(tensor.shape[1:]) * tensor.element_size()
                            for tensor in HostPrefixTier._cache_tensors(cache))
            owned = max(placement.local_rows(st.history_slot, ordinal, count, rank) for rank in range(placement.world))
            total += owned * row_bytes + (count // st.ratio) * pool.shape[-1] * pool.element_size()
        return total

    def _point_bound(self, point):
        """A point's snapshot/tail bytes: fixed-size tensors (recurrent state, rows, compact tails on their owner
        rank), so one all-gather per point shape agrees the largest rank's size; later spills reuse it."""
        local = self._tensor_bytes(point[2]) + self._tensor_bytes(point[3])
        key = ("prompt_end" in point[2], point[3] is None)
        bound = self._point_bounds.get(key)
        if bound is None:
            bound = self._agree(local)
            self._point_bounds[key] = bound
        elif local > bound:
            raise RuntimeError("host prefix point outgrew its rank-agreed size")
        return bound

    # ------------------------------------------------------------------------------------------- policy
    def _value(self, entry):
        """GreedyDual-Size-Frequency: recency (clock at last use) + uses x tokens saved per block held."""
        return entry["used"] + entry["uses"] * max(len(point[0]) for point in entry["points"]) / len(entry["blocks"])

    def _evict_one(self):
        victim = min(self.entries, key=self._value)
        self.clock = max(self.clock, self._value(victim))
        self.entries = [entry for entry in self.entries if entry is not victim]
        self.free.extend(reversed(victim["blocks"]))
        self.counters["evictions"] += 1

    def _agree(self, local_bytes):
        comm = self.comm
        if comm is None or comm.world == 1:
            return local_bytes
        from tensorfold.cuda.capacity import gather_ints

        return max(row[0] for row in gather_ints(torch, comm.all_gather, [local_bytes], world=comm.world))

    # ------------------------------------------------------------------------------------------- spill
    def spill(self, points, principal):
        """Copy one slot's kept ``points`` [(ids, State, snapshot, tail)] before the decoder drops them.

        Called at the same ordered point on every rank. A point whose ids and principal an entry already holds is
        not copied again (its use is counted); the rest share one copy of the slot's rows below the longest point.
        """
        fresh = []
        for point in points:
            same = next((entry for entry in self.entries if entry["principal"] == principal
                         and any(old[0] == point[0] for old in entry["points"])), None)
            if same is None:
                fresh.append(point)
            else:
                same["uses"] += 1
                same["used"] = self.clock
        if not fresh:
            return
        st = fresh[0][1]
        rows = max(max(point[2]["pos"], point[2].get("prompt_end", {}).get("state", {}).get("pos", 0))
                   for point in fresh)
        mtp_rows = max(max(point[2]["mtp_len"], point[2].get("prompt_end", {}).get("state", {}).get("mtp_len", 0))
                       for point in fresh)
        for reader in tuple(st.history_native_readers):
            reader.wait_native_reuse()
        # Rank-identical without a per-spill collective: rows from the placement, points from agreed sizes.
        need = self._history_bound(st, rows, mtp_rows) + sum(self._point_bound(point) for point in fresh)
        blocks_needed = -(-need // self.block_bytes)
        if blocks_needed > self.blocks:
            self.counters["spill_skipped"] += 1
            return
        while len(self.free) < blocks_needed:
            self._evict_one()
        blocks = [self.free.pop() for _ in range(blocks_needed)]
        cursor = [0]
        placement = st.history_placement
        layers = []
        context = contextlib.nullcontext()
        if self.stream is not None:
            # After the rows' writers and any queued restore that read these blocks (stream order).
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            context = torch.cuda.stream(self.stream)
        ready = None
        with context:
            for ordinal, cache, pool, count in self._layers(st, rows, mtp_rows):
                runs = []
                for start, offset, length in self._runs(cache, count, placement.page_tokens):
                    runs.append((start, length, [self._put(blocks, cursor, tensor[offset:offset + length])
                                                 for tensor in self._cache_tensors(cache)]))
                pooled = None
                if st.history_rank == placement.indexer_owner(st.history_slot, ordinal,
                                                              include_mtp=hasattr(st, "mtp_kc")):
                    pooled = self._put(blocks, cursor, pool[:count // st.ratio])
                layers.append((ordinal, count, tuple(cache.owners[:(count + placement.page_tokens - 1)
                                                                   // placement.page_tokens]), runs, pooled))
            packed = [(list(point[0]), self._pack(blocks, cursor, point[2]),
                       None if point[3] is None else self._pack(blocks, cursor, point[3])) for point in fresh]
            if self.stream is not None:
                ready = torch.cuda.Event()
                ready.record(self.stream)
        if cursor[0] > len(blocks) * self.block_bytes:
            raise RuntimeError("host prefix spill exceeded its rank-agreed allocation")
        if ready is not None:
            st.host_spill_ready = ready              # prefixes.slot_for: the slot's next writer waits on it
        self.entries.append({"principal": principal, "points": packed, "blocks": blocks, "layers": layers,
                             "ready": ready,
                             "rows": rows, "mtp_rows": mtp_rows, "kv_identity": st.kv_identity,
                             "class": self.owner_class(placement, st.history_slot, len(st.kc) + hasattr(st, "mtp_kc")),
                             "used": self.clock, "uses": 1})
        self.counters["spills"] += 1
        self.counters["spill_bytes"] += need

    # ------------------------------------------------------------------------------------------- lookup/restore
    def best(self, prompt, principal):
        """The (entry, point index) with the longest point the prompt extends, or None."""
        self.counters["lookups"] += 1
        found, length = None, 0
        for entry in self.entries:
            if entry["principal"] != principal:
                continue
            for index, (ids, _snap, _tail) in enumerate(entry["points"]):
                n = len(ids)
                # The O(1) end/start checks reject almost every other prompt before the full comparison.
                if length < n < len(prompt) and prompt[n - 1] == ids[-1] and prompt[0] == ids[0] and prompt[:n] == ids:
                    found, length = (entry, index), n
        return found

    def restorable(self, entry, st):
        placement = st.history_placement
        return (st.kv_identity == entry["kv_identity"] and st.limit >= entry["rows"]
                and self.owner_class(placement, st.history_slot, len(st.kc) + hasattr(st, "mtp_kc")) == entry["class"])

    def restore(self, entry, index, st):
        """Copy ``entry``'s rows into the free, already admitted ``st`` and return (ids, snapshot, tail) on device."""
        placement = st.history_placement
        for reader in tuple(st.history_native_readers):
            reader.wait_native_reuse()
        await_spill(st)
        if entry["ready"] is not None:
            torch.cuda.current_stream(self.device).wait_event(entry["ready"])
        blocks = entry["blocks"]
        caches = {ordinal: (cache, pool) for ordinal, cache, pool, _count in
                  self._layers(st, entry["rows"], entry["mtp_rows"])}
        for ordinal, count, owners, runs, pooled in entry["layers"]:
            cache, pool = caches[ordinal]
            if tuple(cache.owners[:len(owners)]) != owners:
                raise RuntimeError("host prefix restore target owns different history pages")
            targets = {start: offset for start, offset, _length in self._runs(cache, count, placement.page_tokens)}
            for start, length, regions in runs:
                offset = targets.get(start)
                if offset is None:
                    raise RuntimeError("host prefix restore target lacks an owned history run")
                for tensor, region in zip(self._cache_tensors(cache), regions, strict=True):
                    self._get_into(blocks, region, tensor[offset:offset + length])
            if pooled is not None:
                self._get_into(blocks, pooled, pool[:count // st.ratio])
        ids, snap, tail = entry["points"][index]
        entry["uses"] += 1
        entry["used"] = self.clock
        self.counters["hits"] += 1
        self.counters["restores"] += 1
        self.counters["restored_tokens"] += len(ids)
        self.counters["restore_bytes"] += len(blocks) * self.block_bytes
        return list(ids), self._unpack(blocks, snap), None if tail is None else self._unpack(blocks, tail)

    def record(self):
        return {"capacity_bytes_per_rank": self.capacity, "used_bytes_per_rank": (self.blocks - len(self.free))
                * self.block_bytes, "entries": len(self.entries),
                "points": sum(len(entry["points"]) for entry in self.entries), **self.counters}

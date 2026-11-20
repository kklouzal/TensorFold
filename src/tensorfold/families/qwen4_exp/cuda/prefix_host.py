"""Host-RAM tier for the automatic prefix cache: spill kept prompt points instead of dropping them.

The concurrent decoder keeps prompt points (``MultiDecoder.kept``: ids, slot State, snapshot, MTP tail) in idle
full-window slots. When value-based eviction drops a point (its slot is reused, it falls past ``keep``, or a resume
in place rewrites it), ``spill`` copies that slot's committed rows below the point and the point's complete snapshot
into a bounded, pinned per-rank host slab. Rows the slab already holds for the slot's current history (an earlier
point of the same chain, the entry it was restored from, the chain it was forked from) are not copied again: a spill
stores only the rows past them as a row extent that references them, so a conversation spilled point by point, turn
by turn, holds each row once. ``restore`` copies them back into a free slot whose unique-history page
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

On by default (``DEFAULT_BYTES`` across the ranks, at most a quarter of MemAvailable; ``--prefix-host-bytes 0``
disables; a default slab that cannot be pinned starts without the tier): a repeated or extended prompt restores in
milliseconds instead of seconds (11.4K tokens: 0.16 s against 3.8 s cold). Under a stream of unique prompts every
eviction spills and nothing is restored, which measured about -0.9% prefill and -0.8..-1.9% decode (trimmed
geomeans) on the 8xV100 production fastbench before shared rows (coordinator time ~3 ms/spill): the cost is the
device-to-host traffic per spill (~48 MB/rank then), mostly recurrent snapshots.

Cost model (SC117, Q4 KV, TP8): restoring n tokens copies about 1 KiB/token/rank plus a ~14.6 MB recurrent snapshot
per point over PCIe (~12 GB/s), about 1.2 ms + 85 ns/token, against ~364 us/token of prefill: any point a request can
use beyond its best VRAM point by ``MIN_GAIN`` tokens is restored.
"""

from __future__ import annotations

import contextlib
import copy
import math
import time

import torch

BLOCK_BYTES = 1 << 21        # fixed allocation unit of the pinned slab
DEFAULT_BYTES = 64 << 30     # all ranks together when --prefix-host-bytes is unset (capped at MemAvailable / 4)


def pinned_slab_bytes(per_rank: int, block: int = BLOCK_BYTES) -> int:
    """The largest per-rank slab not above ``per_rank`` that pinning allocates exactly, or 0 below one block.

    PyTorch's pinned (CachingHostAllocator) allocation rounds every request up to a power of two (torch 2.16:
    pinned_max_round_threshold and pinned_max_cached_size are unbounded by default), so a 6 GiB request pins 8 GiB.
    A power-of-two slab (``block`` is one) keeps the pinned bytes equal to the bytes the host budget charges.
    """
    if type(per_rank) is not int or per_rank < block:
        return 0
    return 1 << (per_rank.bit_length() - 1)
MAX_CHAIN = 4                # row extents a restore reads at most (``_base`` merges longer chains)
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
        self.extents = 0                                      # live row extents (shared history rows)
        self.device, self.comm = torch.device(device), comm
        self.clock = 0.0
        self._point_bounds = {}
        self.counters = {"spills": 0, "spill_bytes": 0, "spill_skipped": 0, "restores": 0, "restore_bytes": 0,
                         "restored_tokens": 0, "evictions": 0, "lookups": 0, "hits": 0, "shared_bytes": 0,
                         "spill_seconds": 0.0}           # coordinator-thread time issuing spills (telemetry)

    # ------------------------------------------------------------------------------------------------- layout
    def _views(self, blocks, offset, nbytes):
        """(slab view, start within the region) pieces covering ``nbytes`` at ``offset`` of the entry's blocks; a run
        of consecutive slab blocks is one piece (one copy)."""
        done = 0
        while done < nbytes:
            index, within = divmod(offset + done, self.block_bytes)
            last = index
            while last + 1 < len(blocks) and blocks[last + 1] == blocks[last] + 1 \
                    and (last + 1 - index) * self.block_bytes - within < nbytes - done:
                last += 1
            take = min(nbytes - done, (last + 1 - index) * self.block_bytes - within)
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
    def _runs(cache, rows, page_tokens, first=0):
        """This rank's pages from page ``first`` below ``rows`` grouped into maximal runs contiguous in its arena:
        (arena offset, rows, ((page, rows), ...)). A rank packs its owned pages back to back, so with pages striped
        over the group (every fourth page per rank) a layer is still one run: one copy per tensor, not one per page."""
        runs = []
        for page in range(first, (rows + page_tokens - 1) // page_tokens):
            offset = cache.offsets[page]
            if offset < 0:
                continue
            count = min(page_tokens, rows - page * page_tokens)
            if runs and runs[-1][0] + runs[-1][1] == offset:
                runs[-1][1] += count
                runs[-1][2].append((page, count))
            else:
                runs.append([offset, count, [(page, count)]])
        return [(offset, count, tuple(pages)) for offset, count, pages in runs]

    @staticmethod
    def _targets(cache, pages):
        """(source row within the run, target arena offset, rows) copies placing a run's ``pages`` in ``cache``:
        consecutive pages whose target offsets are contiguous share one copy."""
        copies, at = [], 0
        for page, count in pages:
            offset = cache.offsets[page] if page < len(cache.offsets) else -1
            if offset < 0:
                raise RuntimeError("host prefix restore target lacks an owned history run")
            if copies and copies[-1][1] + copies[-1][2] == offset and copies[-1][0] + copies[-1][2] == at:
                copies[-1][2] += count
            else:
                copies.append([at, offset, count])
            at += count
        return copies

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
    def _history_bound(st, rows, mtp_rows, start=(0, 0)):
        """Bytes the largest rank copies for the rows from ``start`` (page-aligned (rows, MTP rows); at or past a
        layer's end: none of it) below these rows: the same on every rank, from the page placement alone."""
        placement = st.history_placement
        total = 0
        for ordinal, cache, pool, count in HostPrefixTier._layers(st, rows, mtp_rows):
            first = start[ordinal == len(st.kc)]
            if first >= count:
                continue
            row_bytes = sum(math.prod(tensor.shape[1:]) * tensor.element_size()
                            for tensor in HostPrefixTier._cache_tensors(cache))
            owned = max(placement.local_rows(st.history_slot, ordinal, count, rank)
                        - placement.local_rows(st.history_slot, ordinal, first, rank) for rank in range(placement.world))
            total += owned * row_bytes + (count // st.ratio - first // st.ratio) * pool.shape[-1] * pool.element_size()
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
        """GreedyDual-Size-Frequency: recency (clock at last use) + uses x tokens saved per block held. An entry holds
        its snapshot blocks and an equal share of every row extent its restore reads (shared rows split their cost)."""
        held = len(entry["blocks"])
        extent = entry["extent"]
        while extent is not None:
            held += len(extent["blocks"]) / extent["users"]
            extent = extent["parent"]
        return entry["used"] + entry["uses"] * max(len(point[0]) for point in entry["points"]) / max(held, 1)

    def _evict_one(self):
        victim = min(self.entries, key=self._value)
        self.clock = max(self.clock, self._value(victim))
        self.entries = [entry for entry in self.entries if entry is not victim]
        self.free.extend(reversed(victim["blocks"]))
        extent = victim["extent"]
        while extent is not None:
            extent["users"] -= 1
            extent = extent["parent"]
        self._release(victim["extent"])
        self.counters["evictions"] += 1

    def _release(self, extent):
        """Drop one reference to ``extent``: unreferenced, it frees its blocks and its own reference to its parent."""
        while extent is not None:
            extent["refs"] -= 1
            if extent["refs"]:
                return
            extent["alive"] = False
            self.free.extend(reversed(extent["blocks"]))
            self.extents -= 1
            extent = extent["parent"]

    @staticmethod
    def _chain_blocks(extent):
        blocks = 0
        while extent is not None:
            blocks += len(extent["blocks"])
            extent = extent["parent"]
        return blocks

    def _agree(self, local_bytes):
        comm = self.comm
        if comm is None or comm.world == 1:
            return local_bytes
        from tensorfold.cuda.capacity import gather_ints

        return max(row[0] for row in gather_ints(torch, comm.all_gather, [local_bytes], world=comm.world))

    # ------------------------------------------------------------------------------------------- shared rows
    # A spill stores a slot's history rows as a row extent: its rows from ``start`` (rows, MTP rows; page-aligned,
    # or ``end`` in a dimension it holds none of) below ``end``, the rows below ``start`` coming from its parent's
    # chain. An extent's chain (each extent's rows below the next one's start) equals the slot's rows below ``end``
    # when it was spilled. ``st.host_rows`` records which chain the slot's rows still equal, below which rows, for one
    # admission generation of the slot: every admission begins a new one, and ``prefixes`` carries the record only
    # where the rows stay (a resume in place, a fork's copied rows, a restore), truncated where the request writes.
    # A later spill of the slot copies only the rows past it; an entry restores its chain below its own rows.
    def _owner_class_of(self, st):
        return self.owner_class(st.history_placement, st.history_slot, len(st.kc) + hasattr(st, "mtp_kc"))

    def held(self, st):
        """(extent, rows, MTP rows): the host chain ``st``'s rows equal below those rows in its current admission
        generation, or None."""
        stamp = getattr(st, "host_rows", None)
        if stamp is None:
            return None
        tier, extent, rows, mtp_rows, generation = stamp
        if tier is not self or generation != st.slot_generation or not extent["alive"]:
            return None
        # A history degree migration (MultiDecoder: migrate_history_bundle) permutes the slot's page owners without
        # a new admission generation: rows held under another page ownership no longer match this rank's pages.
        if extent["kv_identity"] != st.kv_identity or extent["class"] != self._owner_class_of(st):
            return None
        return extent, rows, mtp_rows

    def hold(self, st, held, rows=None, mtp_rows=None):
        """Record that ``st``'s rows equal ``held`` (``held``'s rows, cut to ``rows``/``mtp_rows`` when given: where
        the slot's next request writes) for its current generation; None, or another page ownership, clears it."""
        if held is None:
            st.host_rows = None
            return
        extent, have, have_mtp = held
        rows = have if rows is None else min(rows, have)
        mtp_rows = have_mtp if mtp_rows is None else min(mtp_rows, have_mtp)
        if extent["kv_identity"] != st.kv_identity or extent["class"] != self._owner_class_of(st):
            st.host_rows = None
            return
        while extent is not None and extent["start"][0] >= rows and extent["start"][1] >= mtp_rows:
            extent = extent["parent"]                   # it holds nothing below the rows that stay
        st.host_rows = None if extent is None else (self, extent, rows, mtp_rows, st.slot_generation)

    @staticmethod
    def _parts(extent, upto, weights):
        """Layer-weighted rows ``extent`` stores below ``upto`` and at or past it."""
        below = past = 0
        for first, end, limit, weight in zip(extent["start"], extent["end"], upto, weights):
            below += max(0, min(end, limit) - first) * weight
            past += max(0, end - max(first, limit)) * weight
        return below, past

    def _base(self, st, ids, rows, mtp_rows):
        """What a spill of ``st``'s rows below (rows, MTP rows) builds on: (extent or None for a new root, start), or
        (extent, None) when that extent's chain already holds them (a reference, nothing copied). An extent whose
        stored rows are mostly past what this spill can use is not built on (nor referenced): its part this spill
        needs is copied again instead, so shared rows never pin more unusable rows than they save."""
        held = self.held(st)
        if held is None:
            return None, (0, 0)
        extent, have, have_mtp = held
        common = min(len(extent["ids"]), len(ids), have)
        if extent["ids"][:common] != ids[:common]:
            raise RuntimeError("a slot's history changed within one admission generation: its host rows are stale")
        need, weights = (rows, mtp_rows), (len(st.kc), 1)
        if rows <= have and mtp_rows <= have_mtp:
            while extent["parent"] is not None and extent["start"][0] >= rows and extent["start"][1] >= mtp_rows:
                extent = extent["parent"]
            below, past = self._parts(extent, need, weights)
            if past <= below:
                return extent, None
            base, valid = extent["parent"], tuple(min(first, n) for first, n in zip(extent["start"], need))
        else:
            below, past = self._parts(extent, (have, have_mtp), weights)
            if past > below:
                base, valid = extent["parent"], (min(extent["start"][0], have), min(extent["start"][1], have_mtp))
            else:
                base, valid = extent, (have, have_mtp)
        if base is None:
            valid = (0, 0)
        align = math.lcm(st.history_placement.page_tokens, st.ratio)
        start = tuple(n if n <= v else v // align * align for n, v in zip(need, valid))
        # A restore copies once per extent of its chain: this extent takes in every base extent no larger than what
        # it copies (and its base past MAX_CHAIN), so chains stay logarithmic in their spills and each row is copied
        # again only a logarithmic number of times as they merge.
        while base is not None and base["parent"] is not None and (
                base["depth"] >= MAX_CHAIN or start[0] - base["start"][0] <= rows - start[0]):
            start = tuple(min(first, own) for first, own in zip(start, base["start"]))
            base = base["parent"]
        return base, start

    def _copy_rows(self, st, rows, mtp_rows, start, parent, blocks):
        """Queue the copies of ``st``'s rows from ``start`` below (rows, MTP rows) into ``blocks``: a new extent."""
        cursor = [0]
        placement = st.history_placement
        layers = []
        for ordinal, cache, pool, count in self._layers(st, rows, mtp_rows):
            first = start[ordinal == len(st.kc)]
            if first >= count:
                continue
            runs = []
            for offset, length, pages in self._runs(cache, count, placement.page_tokens, first // placement.page_tokens):
                runs.append((pages, [self._put(blocks, cursor, tensor[offset:offset + length])
                                     for tensor in self._cache_tensors(cache)]))
            pooled = None
            if st.history_rank == placement.indexer_owner(st.history_slot, ordinal,
                                                          include_mtp=hasattr(st, "mtp_kc")):
                pooled = self._put(blocks, cursor, pool[first // st.ratio:count // st.ratio])
            layers.append((ordinal, first, count, tuple(cache.owners[:(count + placement.page_tokens - 1)
                                                                     // placement.page_tokens]), runs, pooled))
        if cursor[0] > len(blocks) * self.block_bytes:
            raise RuntimeError("host prefix spill exceeded its rank-agreed allocation")
        self.extents += 1
        return {"parent": parent, "start": start, "end": (rows, mtp_rows), "blocks": blocks, "layers": layers,
                "depth": 1 if parent is None else parent["depth"] + 1, "refs": 1, "users": 0, "alive": True, "kv_identity": st.kv_identity, "class": self._owner_class_of(st)}

    # ------------------------------------------------------------------------------------------- spill
    def spill(self, points, principal):
        """Copy one slot's kept ``points`` [(ids, State, snapshot, tail)] before the decoder drops them.

        Called at the same ordered point on every rank. A point whose ids and principal an entry already holds is
        not copied again (its use is counted); the rest share one copy of the slot's rows below the longest point,
        past the rows the host already holds for this slot (``held``).
        """
        began = time.perf_counter()
        try:
            self._spill(points, principal)
        finally:
            self.counters["spill_seconds"] += time.perf_counter() - began

    def _spill(self, points, principal):
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
        # Rank-identical without a per-spill collective: rows from the placement, points from agreed sizes, the
        # base from the replicated host tables and slot generations.
        ids = max((point[0] for point in fresh), key=len)
        base, start = self._base(st, ids, rows, mtp_rows)
        full = self._history_bound(st, rows, mtp_rows)
        row_need = 0 if start is None else full if base is None else self._history_bound(st, rows, mtp_rows, start)
        point_need = sum(self._point_bound(point) for point in fresh)
        row_blocks, point_blocks = -(-row_need // self.block_bytes), -(-point_need // self.block_bytes)
        if base is not None:
            base["refs"] += 1                       # the new extent's (or entry's) reference: no eviction frees it
            if row_blocks + point_blocks > self.blocks - self._chain_blocks(base):
                self._release(base)                 # no room beside it: copy every row
                base, start, row_need = None, (0, 0), full
                row_blocks = -(-full // self.block_bytes)
        if row_blocks + point_blocks > self.blocks:
            self.counters["spill_skipped"] += 1
            return
        while len(self.free) < row_blocks + point_blocks:
            self._evict_one()
        extent_blocks = [self.free.pop() for _ in range(row_blocks)] if start is not None else []
        blocks = [self.free.pop() for _ in range(point_blocks)]
        context = contextlib.nullcontext()
        if self.stream is not None:
            # After the rows' writers and any queued restore that read these blocks (stream order).
            self.stream.wait_stream(torch.cuda.current_stream(self.device))
            context = torch.cuda.stream(self.stream)
        ready = rows_ready = None
        extent = base
        with context:
            if start is not None:
                extent = self._copy_rows(st, rows, mtp_rows, start, base, extent_blocks)
                extent["ids"] = list(ids)
                if self.stream is not None:
                    # The slot's own rows are read by now: its next writer waits only for these copies. The snapshots
                    # below are the kept points' own device copies (record_stream keeps them alive), not slot memory.
                    rows_ready = torch.cuda.Event()
                    rows_ready.record(self.stream)
            cursor = [0]
            packed = [(list(point[0]), self._pack(blocks, cursor, point[2]),
                       None if point[3] is None else self._pack(blocks, cursor, point[3])) for point in fresh]
            if self.stream is not None:
                ready = torch.cuda.Event()
                ready.record(self.stream)
        if cursor[0] > len(blocks) * self.block_bytes:
            raise RuntimeError("host prefix spill exceeded its rank-agreed allocation")
        if rows_ready is not None:
            st.host_spill_ready = rows_ready         # prefixes.slot_for: the slot's next writer waits on it
        if start is not None:
            st.host_rows = (self, extent, rows, mtp_rows, st.slot_generation)
        chain = extent
        while chain is not None:
            chain["users"] += 1
            chain = chain["parent"]
        # A restore waits on ``ready``: the chain's earlier extents were copied before it on the same stream.
        self.entries.append({"principal": principal, "points": packed, "blocks": blocks, "extent": extent,
                             "ready": ready, "point_bytes": point_need, "rows": rows, "mtp_rows": mtp_rows,
                             "kv_identity": st.kv_identity, "class": self._owner_class_of(st),
                             "used": self.clock, "uses": 1})
        self.counters["spills"] += 1
        self.counters["spill_bytes"] += row_need + point_need
        self.counters["shared_bytes"] += full - row_need

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
        return (st.kv_identity == entry["kv_identity"] and st.limit >= entry["rows"]
                and self._owner_class_of(st) == entry["class"])

    def restore(self, entry, index, st):
        """Copy ``entry``'s rows into the free, already admitted ``st`` and return (ids, snapshot, tail) on device."""
        for reader in tuple(st.history_native_readers):
            reader.wait_native_reuse()
        await_spill(st)
        if entry["ready"] is not None:
            torch.cuda.current_stream(self.device).wait_event(entry["ready"])
        rows, mtp_rows = entry["rows"], entry["mtp_rows"]
        caches = {ordinal: (cache, pool) for ordinal, cache, pool, _count in self._layers(st, rows, mtp_rows)}
        page_tokens = st.history_placement.page_tokens
        upto, moved = [rows, mtp_rows], 0
        extent = entry["extent"]
        while extent is not None:                   # each extent's rows below the next one's start: disjoint copies
            for ordinal, first, count, owners, runs, pooled in extent["layers"]:
                stop = min(count, upto[ordinal == len(st.kc)])
                if first >= stop:
                    continue
                cache, pool = caches[ordinal]
                pages = -(-stop // page_tokens)
                if tuple(cache.owners[:pages]) != owners[:pages]:
                    raise RuntimeError("host prefix restore target owns different history pages")
                tensors = self._cache_tensors(cache)
                for run, regions in runs:
                    total = regions[0].shape[0]                     # the run's rows
                    if run[-1][0] * page_tokens + run[-1][1] > stop:  # the restore ends inside this run
                        run = tuple((page, min(n, stop - page * page_tokens)) for page, n in run
                                    if page * page_tokens < stop)
                        if not run:
                            continue
                    for at, offset, length in self._targets(cache, run):
                        for tensor, region in zip(tensors, regions, strict=True):
                            row = region.nbytes // total
                            part = _Region(region.offset + at * row, length * row, region.dtype,
                                           (length, *region.shape[1:]))
                            self._get_into(extent["blocks"], part, tensor[offset:offset + length])
                            moved += part.nbytes
                low, high = first // st.ratio, stop // st.ratio
                if pooled is not None and high > low:
                    row = pooled.nbytes // pooled.shape[0]
                    part = _Region(pooled.offset, (high - low) * row, pooled.dtype, (high - low, *pooled.shape[1:]))
                    self._get_into(extent["blocks"], part, pool[low:high])
                    moved += part.nbytes
            upto = [min(upto[0], extent["start"][0]), min(upto[1], extent["start"][1])]
            extent = extent["parent"]
        st.host_rows = None if entry["extent"] is None else (self, entry["extent"], rows, mtp_rows, st.slot_generation)
        ids, snap, tail = entry["points"][index]
        entry["uses"] += 1
        entry["used"] = self.clock
        self.counters["hits"] += 1
        self.counters["restores"] += 1
        self.counters["restored_tokens"] += len(ids)
        self.counters["restore_bytes"] += moved + entry["point_bytes"]
        blocks = entry["blocks"]
        return list(ids), self._unpack(blocks, snap), None if tail is None else self._unpack(blocks, tail)

    def record(self):
        return {"capacity_bytes_per_rank": self.capacity, "used_bytes_per_rank": (self.blocks - len(self.free))
                * self.block_bytes, "entries": len(self.entries), "extents": self.extents,
                "points": sum(len(entry["points"]) for entry in self.entries), **self.counters}

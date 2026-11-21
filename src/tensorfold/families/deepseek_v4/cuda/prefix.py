"""DeepSeek-V4.1 prompt-prefix reuse: the Gemma 4 SM70 ``PrefixOwner`` (the Flash Next retention policy, qwen4_exp
``prefixes``) and host-RAM tier (``prefix_host``, shared row extents) over this lane's packed KV pages and slots.

What a kept point at position h must hold. A slot's per-request state after a forward that ends at h is a function
of its first h tokens only (patch 85's slot semantics; every kernel resolves its own request's rows):

  * KV rows below h in the slot's 64-token pages: per source layer (2, 8, 14 at ratio 2; 20 at ratio 1) a compressed
    latent row (288 B) and an indexer key row (68 B) at every position p with (p + 1) % ratio == 0, row p // ratio.
    They stay in the slot (resume in place) or are copied page by page into another slot (a fork, ``copy``).
    Rows at positions >= h may hold anything (a later reply, a cancelled chunk, another slot's old data in a reused
    page): a forward writes a position's rows before any read of them (insert precedes select/attend per layer,
    and reads are causal), exactly as a cold prefill into reused pages relies on.
  * The window rings ([slots, 128, 512] FP16 per layer): after a forward ending at h a ring holds positions
    h-128..h-1 at p % 128 (patch 87 writes each request's last 128 positions); copied whole.
  * The ratio-2 compressor partials (``MuC2State``: partial [2, 512] FP32 + committed position per source) and the
    Engram n-gram history (``MuEngramHashState.history`` [4] int64: three compressed ids + committed position);
    copied whole, so any h is a valid point (a continuing forward checks committed == its start: the trap words).
  * No other per-slot state: the decode/prefill indexers read only the pages (Full mode, no patch-170 reuse
    state); DeepSeek-V4.1's compress ratios are 0/1/2 (no ratio 4 / 128 groups to align to).

So the state at any forward boundary of a slot is exact. What decides where a point may be kept is the arithmetic of
the rows a resumed request still computes: its prompt chunks must be the chunks a cold prefill of the same prompt
runs (same rows per forward, so the same GEMM shapes, MoE path, indexer path and reductions). Every prompt, cold or
resumed, is therefore chunked at its message boundaries (``chunk_bounds``: the renderer's ``message_points`` -- the
system prompt's end, each user turn or tool-result group, each assistant reply, each generation header's end --
at least ``MIN_CHUNK`` rows apart, and every ``prompt_rows`` inside a longer message), and points are kept only on
those boundaries plus the exact prompt end. The boundaries below any position are a function of the ids before it
(and the id at it), so a later prompt sharing a kept point's ids has the same boundaries below it; a resume lands only
on a boundary of the new prompt (``_resume_at``, which otherwise falls back to the best kept boundary below) and then
computes exactly the cold chunks. The exact end is kept twice: one token early with its last-row logits for exact
repeats (an exact repeat computes no prompt row; its reply is the cold reply), and at its end for a prompt that
extends it on a boundary (a next turn re-renders the generation header the end stops on). Under these rules a cached
request's every computed row is bitwise the cold request's (verified with tools/quality_eval cachecheck).

Ranks: rank 0's owner decides and announces the Gemma ``ops`` codes; every rank runs ``SlotState`` on its own
tensors in announcement order (identical shapes and page ids, so snapshot ids and slab offsets are rank-replicated).
Snapshots live in pool pages, so the KV ledger (pages) bounds them with the slots' caches.
"""

from __future__ import annotations

import torch

from tensorfold.families.gemma4.sm70 import prefix as _gemma
from tensorfold.families.gemma4.sm70 import prefix_host as _gemma_host
from tensorfold.families.gemma4.sm70.ops import (OP_COPY, OP_DROP, OP_GROW, OP_HRESTORE, OP_HSPILL, OP_RESTORE,
                                                 OP_SNAP)
from tensorfold.families.qwen4_exp.cuda import prefixes
from tensorfold.families.qwen4_exp.cuda.prefix_host import BLOCK_BYTES

from .kernels.kv_pack import C_ROW, I_ROW
from .target import BLOCK

RING = 128
WIDTH = 512
OPS = (OP_SNAP, OP_RESTORE, OP_DROP, OP_COPY, OP_GROW, OP_HSPILL, OP_HRESTORE)


def snapshot_bytes(layers: int, c2_sources: int) -> int:
    """Raw bytes of one kept point's slot state on one rank: every layer's ring rows, the C2 partials and committed
    positions, the Engram history."""
    return layers * RING * WIDTH * 2 + c2_sources * (2 * WIDTH * 4 + 8) + 4 * 8


def extent_bytes(layout, first: int, end: int) -> int:
    """Bytes of the KV rows of positions [first, end) (every source's rows first // r .. end // r)."""
    return sum((end // ratio - first // ratio) * (C_ROW + I_ROW) for ratio, _, _, _ in layout.regions.values())


MIN_CHUNK = 256          # rows: a message boundary closer than this to the previous one is not a chunk boundary
SPACING = 4096          # rows: kept points inside a prompt besides its first and last two boundaries


def chunk_bounds(points, n: int, rows: int) -> list[int]:
    """A prompt's chunk boundaries in (0, n) from its message ``points`` (sorted): a point at least MIN_CHUNK rows
    past the previous boundary, and every ``rows`` past the previous boundary inside a longer message. Read left to
    right from position 0, so the boundaries below a position depend on the ids before it only (and the id at it)."""
    out: list[int] = []
    prev = 0
    for p in (*points, n):
        while p - prev > rows:
            prev += rows
            out.append(prev)
        if p < n and p - prev >= MIN_CHUNK:
            out.append(p)
            prev = p
    return out


def chunk_end(bounds, start: int, n: int) -> int:
    """Where the prompt chunk at ``start`` ends: the next boundary, or the prompt end."""
    from bisect import bisect_right
    k = bisect_right(bounds, start)
    return bounds[k] if k < len(bounds) else n


def default_keep_slots(streams: int) -> int:
    """Idle slots that keep conversations' prefix chains in VRAM. Their caches come from the page budget (evicted
    first when a live stream needs pages); a slot's own fixed cost is its rings (5.2 MiB/rank) and the kept points'
    last-row logits on rank 0 (517 KB each, outside the page ledger), so the count stays near the stream count."""
    return max(8, streams)


def host_slab(comm, rank: int, tp: int, requested: int | None):
    """The host tier's pinned slab of this rank (Gemma 4 SM70's sizing: ``requested`` bytes across the ranks, unset:
    Flash Next's ``default_bytes``; a power of two per rank, agreed as the smallest rank's) -> (bytes, slab)."""
    from tensorfold.families.qwen4_exp.cuda.prefix_host import default_bytes, pin_agreed, pinned_slab_bytes
    if requested is not None and (type(requested) is not int or requested < 0):
        raise ValueError("prefix_host_bytes must be a nonnegative byte count")
    explicit = requested is not None
    total = requested if explicit else default_bytes()
    mine = torch.tensor([pinned_slab_bytes(total // tp)], dtype=torch.int64, device="cuda")
    agreed = torch.empty(tp, dtype=torch.int64, device="cuda")
    comm.all_gather(mine, agreed)
    per_rank = int(agreed.min())
    if not per_rank:
        return 0, None

    def every_rank(ok):
        flags = torch.empty(tp, dtype=torch.int64, device="cuda")
        comm.all_gather(torch.tensor([ok], dtype=torch.int64, device="cuda"), flags)
        return flags.tolist()

    slab = pin_agreed(lambda: torch.empty(per_rank, dtype=torch.uint8, pin_memory=True), every_rank, per_rank,
                      explicit=explicit, requested=total, rank=rank, lane="DeepSeek-V4.1 SM70: ")
    return (per_rank, slab) if slab is not None else (0, None)


# ---------------------------------------------------------------------------------------------- rank 0: the owner
class Slot(_gemma.Slot):
    @property
    def limit(self) -> int:
        """Resident window: every slot can grow to the admitted context."""
        return self.owner.e.context


class PrefixOwner(_gemma.PrefixOwner):
    """The Gemma 4 SM70 prefix owner with this lane's surface: page-granular slot capacities, the page ledger
    (``e.budget``, ``e.slot_bytes``, ``e.snap_bytes``: snapshots are stored in pool pages) and the DeepSeek host
    tier. The engine supplies ``e.context``, ``e.prompt_rows``, ``e.streams``, ``e.keep_slots``,
    ``e.host_per_rank``, ``e.host_slab``, ``e.token_bytes`` and ``e.announce``."""

    def _init_prefix(self, engine, points) -> None:
        # The Gemma owner's fields (its _init_prefix reads Gemma's pool and drafter for the snapshot size).
        self.slots = [Slot(self, i) for i in range(engine.streams + engine.keep_slots)]
        self.free = list(reversed(self.slots))          # prefixes pops from the end
        self.kept: list = []
        self.used: dict[int, int] = {}
        self.clock = 0
        self.keep = 6 * max(1, engine.keep_slots)        # kept points (a prompt: its first, last two and spaced stops, two ends)
        self.depth = 0                                   # no drafter rows past the prompt
        self.prefix_counters = dict.fromkeys(prefixes.COUNTERS, 0)
        self.points = points
        self.held = 0                                    # ledger bytes: attached slot pages + snapshot pages
        self.snap_bytes = engine.snap_bytes
        self.snap_ids: set[int] = set()
        self.next_snap = 0
        self._extent = 0
        self._counted = None
        self.host_tier = HostTier(self, engine.host_per_rank) if engine.host_per_rank else None

    def _cap(self, rows: int) -> int:
        return min(self.e.context, -(-rows // BLOCK) * BLOCK)

    def bounds(self, prompt) -> list[int]:
        """``prompt``'s chunk boundaries (every request's chunks, cold or resumed, end on them)."""
        points = self.points(prompt) if self.points is not None else []
        return chunk_bounds(points, len(prompt), self.e.prompt_rows)

    def _stops(self, s) -> list[int]:
        """The boundaries past the resume point this prompt keeps states at: its first (the system prompt's end, a
        prefix other conversations share), its last two (the latest message and reply, where the next turn
        diverges) and one about every SPACING rows between (a long prompt edited midway)."""
        ahead = [b for b in s.bounds if b > s.cached]
        out = []
        for i, b in enumerate(ahead):
            if i == 0 or i >= len(ahead) - 2 or b - out[-1] >= SPACING:
                out.append(b)
        return out

    def _resume_at(self, s, st, resume, cached: int, sid):
        """A resume lands on a boundary of ``s``'s prompt, from a state kept exactly there: else (an exact end
        extended, or a kept point that is not a boundary of this prompt) on the longest kept boundary point below
        it, or cold. Points of ``st`` past that landing go: this request rewrites their rows in other chunks."""
        if s.head is not None or (cached in s.bounds and not resume["state"].get("exact")):
            return cached, sid
        aligned, sid = 0, None
        for ids, _st, snap, _tail in self.kept:
            n = len(ids)
            if aligned < n <= cached and n in s.bounds and not snap.get("exact") and s.prompt[:n] == ids:
                aligned, sid = n, snap["sid"]
        self._recount(cached - aligned)
        stale = [k for k in self.kept if k[1] is st and len(k[0]) > aligned]
        if stale:
            self.kept = [k for k in self.kept if not (k[1] is st and len(k[0]) > aligned)]
            self._sweep()
        return aligned, sid

    def keep_end(self, s, logits) -> None:
        """The exact prompt end (state at n): for exact repeats as the point one token early with the last-row
        logits (``prefixes``' prompt-end convention: the state at n cannot resume at n - 1, so ``_resume_at`` never
        lands on it), and as the point at n for a prompt extending it on a boundary. When n - 1 is a kept boundary
        the exact end joins that point (whose state is at n - 1) instead."""
        n = len(s.prompt)
        if n < 2:
            return
        sid = self._snapshot(s.st, n)
        if sid is None:
            return
        ids = list(s.prompt[:n - 1])
        end = {"token": s.prompt[-1], "state": {"sid": sid, "pos": n, "mtp_len": n}, "tail": None,
               "head": logits.clone(), "last": None}
        point = next((k for k in self.kept if k[1] is s.st and k[0] == ids and not k[2].get("exact")), None)
        snap = ({**point[2], "prompt_end": end} if point is not None else
                {"sid": sid, "pos": n - 1, "mtp_len": n - 1, "exact": True, "prompt_end": end})
        prefixes.remember(self, ids, s.st, snap, None)
        prefixes.remember(self, list(s.prompt), s.st, {"sid": sid, "pos": n, "mtp_len": n}, None)
        self._sweep()


class HostTier(_gemma_host.HostTier):
    """Gemma 4 SM70's host tier; a row extent [first, end) holds every source's rows first // r .. end // r."""

    def _rows_bytes(self, first: int, rows: int) -> int:
        return extent_bytes(self.owner.e.layout, first, rows)


# ---------------------------------------------------------------------------------------------- every rank: copies
class SlotState:
    """One rank's prefix copies over ``target`` (its pool pages, rings, C2 partials, Engram history) and the host
    tier's pinned ``slab``. Snapshot ``sid`` occupies pool pages owned under the key ("snap", sid)."""

    def __init__(self, target, slab=None) -> None:
        self.target, self.pool, self.slab = target, target.pool, slab
        self.raw = snapshot_bytes(len(self.pool.rings), len(target.c2))
        self.pages = -(-self.raw // self.pool.layout.page_bytes)     # pool pages per snapshot
        self.snaps: dict[int, int] = {}                              # sid -> position
        self.host_layout: dict[int, list] = {}                       # entry id -> [(slab offset, position)]
        self.host_extents: dict[int, tuple] = {}                     # extent id -> (first, [segment offsets])

    @property
    def snap_bytes(self) -> int:
        return self.pages * self.pool.layout.page_bytes

    def run(self, op, a, b, c, d, ids):
        if op == OP_SNAP:                      # slot a at position b -> snapshot c
            return self.snap(a, b, c)
        if op == OP_RESTORE:                   # snapshot b into slot a
            return self.restore(a, b)
        if op == OP_DROP:
            return self.drop(a)
        if op == OP_COPY:                      # KV rows [0, c) of slot a into slot b
            return self.copy(a, b, c)
        if op == OP_GROW:                      # slot a to capacity b, its rows kept
            return self.pool.attach(a, b)
        if op == OP_HSPILL:
            return self.hspill(a, b, c, d, ids)
        if op == OP_HRESTORE:
            return self.hrestore(a, b, c, d, ids)
        raise RuntimeError(f"unknown DeepSeek-V4.1 prefix operation {op}")

    # ---- slot state
    def _parts(self, slot: int) -> list:
        t = self.target
        parts = [ring[slot] for ring in self.pool.rings]
        for layer in sorted(t.c2):
            parts += [t.c2[layer].partial[slot], t.c2[layer].committed[slot]]
        parts.append(t.hash_state.history[slot])
        return [p.reshape(-1).view(torch.uint8) for p in parts]      # views of the live tensors

    def _index(self, pages) -> torch.Tensor:
        return torch.tensor(pages, dtype=torch.long).to(self.pool.cache.device, non_blocking=True)

    def _pack(self, slot: int) -> torch.Tensor:
        buf = torch.zeros(self.snap_bytes, dtype=torch.uint8, device=self.pool.cache.device)
        torch.cat(self._parts(slot), out=buf[:self.raw])
        return buf

    def _unpack(self, slot: int, buf: torch.Tensor) -> None:
        at = 0
        for part in self._parts(slot):
            part.copy_(buf[at:at + part.numel()])
            at += part.numel()

    def snap(self, slot: int, pos: int, sid: int) -> None:
        """Keep ``slot``'s state (at a forward boundary, position ``pos``) in fresh pool pages."""
        if sid in self.snaps:
            raise ValueError(f"prefix snapshot {sid} exists")
        pages = self.pool.attach(("snap", sid), self.pages * BLOCK)
        self.pool.cache[self._index(pages)] = self._pack(slot).view(self.pages, -1)
        self.snaps[sid] = pos

    def restore(self, slot: int, sid: int) -> int:
        """Snapshot ``sid`` into ``slot`` (any slot: every value is slot-independent); returns its position."""
        pages = self.pool.owned[("snap", sid)]
        self._unpack(slot, self.pool.cache[self._index(pages)].reshape(-1))
        return self.snaps[sid]

    def drop(self, sid: int) -> None:
        del self.snaps[sid]
        self.pool.detach(("snap", sid))

    def copy(self, src: int, dst: int, rows: int) -> None:
        k = -(-rows // BLOCK)
        have, want = self.pool.owned.get(src, []), self.pool.owned.get(dst, [])
        if len(have) < k or len(want) < k:
            raise RuntimeError(f"prefix fork of {rows} rows: slot {src} holds {len(have)} pages, slot {dst} "
                               f"{len(want)}")
        if k:
            cache = self.pool.cache
            cache[self._index(want[:k])] = cache[self._index(have[:k])]

    # ---- KV rows by position (host extents)
    def _segments(self, first: int, end: int):
        """(region view [pages, rows per page, row bytes], first row, end row) of every source's latent and index
        rows of positions [first, end), in a fixed order."""
        cache = self.pool.cache
        for layer in sorted(self.pool.layout.regions):
            ratio, per, c_offset, i_offset = self.pool.layout.regions[layer]
            for offset, size in ((c_offset, C_ROW), (i_offset, I_ROW)):
                view = cache[:, offset:offset + per * size].unflatten(1, (per, size))
                yield view, per, first // ratio, end // ratio

    def _where(self, slot: int, per: int, lo: int, hi: int):
        k = torch.arange(lo, hi, dtype=torch.long)
        table = torch.tensor(self.pool.owned[slot], dtype=torch.long)
        device = self.pool.cache.device
        return table[k // per].to(device, non_blocking=True), (k % per).to(device, non_blocking=True)

    def _put(self, at: int, tensor: torch.Tensor) -> int:
        size = tensor.numel()
        self.slab[at:at + size].view(tensor.shape).copy_(tensor, non_blocking=True)
        return at + size

    def hspill(self, hid: int, slot: int, end: int, block: int, ids) -> None:
        """Host entry ``hid`` <- slot: its KV rows [first, end) as extent xid at slab block xblock (xid -1: none) and
        the snapshots ``sids`` from slab block ``block``; ids: n, sids, xid, first, xblock, ids freed meanwhile."""
        n = ids[0]
        sids, (xid, first, xblock), gone = ids[1:1 + n], ids[1 + n:4 + n], ids[4 + n:]
        for old in gone:
            self.host_layout.pop(old, None)
            self.host_extents.pop(old, None)
        if xid >= 0:
            at, offsets = xblock * BLOCK_BYTES, []
            for view, per, lo, hi in self._segments(first, end):
                offsets.append(at)
                if hi > lo:
                    pages, rows = self._where(slot, per, lo, hi)
                    at = self._put(at, view[pages, rows])
            if at > block * BLOCK_BYTES:
                raise RuntimeError("a host prefix extent overran its blocks")
            self.host_extents[xid] = (first, offsets)
        at, regions = block * BLOCK_BYTES, []
        for sid in sids:
            regions.append((at, self.snaps[sid]))
            at = self._put(at, self.pool.cache[self._index(self.pool.owned[("snap", sid)])].reshape(-1))
        self.host_layout[hid] = regions

    def hrestore(self, hid: int, slot: int, copies: int, pairs: int, ids) -> None:
        """Host entry ``hid`` -> slot: ids holds ``pairs`` (entry snapshot index, new snapshot id) pairs, then
        ``copies`` (extent id, first position, end position) row copies."""
        regions = self.host_layout[hid]
        device = self.pool.cache.device
        for xid, lo, hi in zip(ids[2 * pairs::3], ids[2 * pairs + 1::3], ids[2 * pairs + 2::3]):
            first, offsets = self.host_extents[xid]
            for (view, per, r0, r1), (_v, _p, base, _e), at in zip(self._segments(lo, hi), self._segments(first, lo),
                                                                    offsets):
                if r1 > r0:
                    size = view.shape[2]
                    start = at + (r0 - base) * size
                    rows = self.slab[start:start + (r1 - r0) * size].view(r1 - r0, size)
                    pages, index = self._where(slot, per, r0, r1)
                    view[pages, index] = rows.to(device, non_blocking=True)
        for k, sid in zip(ids[0:2 * pairs:2], ids[1:2 * pairs:2]):
            at, pos = regions[k]
            pages = self.pool.attach(("snap", sid), self.pages * BLOCK)
            data = self.slab[at:at + self.snap_bytes].to(device, non_blocking=True)
            self.pool.cache[self._index(pages)] = data.view(self.pages, -1)
            self.snaps[sid] = pos

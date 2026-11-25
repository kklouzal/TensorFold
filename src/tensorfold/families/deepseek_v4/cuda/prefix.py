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
  * The window rings (128 FP16 or RotorQuant rows per slot/layer): after a forward ending at h a ring holds positions
    h-128..h-1 at p % 128 (patch 87 writes each request's last 128 positions); copied whole.
  * The ratio-2 compressor partials (``MuC2State``: partial [2, 512] FP32 + committed position per source) and the
    Engram n-gram history (``MuEngramHashState.history`` [4] int64: three compressed ids + committed position);
    copied whole, so any h is a valid point (a continuing forward checks committed == its start: the trap words).
  * No other per-slot state: the decode/prefill indexers read only the pages (Full mode, no patch-170 reuse
    state); DeepSeek-V4.1's compress ratios are 0/1/2 (no ratio 4 / 128 groups to align to).

So the state at any forward boundary of a slot is exact, and a point may be kept wherever a prompt chunk ends.
Points are kept only at natural breaks (the user's rule): message boundaries (``message_points``) and the exact prompt
end. Where chunks end is a cost choice: a prompt forward costs about 0.21 s plus 0.45 ms per row at TP8 (pfx2-a), so a
prompt is split only where a later request is likely to resume (``chunk_splits``), each split keeping a point: the
system prompt's end (the first of the renderer's ``message_points``: a prefix other conversations share) and the last
message's start (a regenerated or re-rendered turn -- thinking without tools re-renders the previous header --
resumes there); a next turn that extends the prompt resumes at its exact end. ``prompt_rows`` (8192) only caps a
chunk's rows: those splits keep no point. Chunking every message boundary instead (bitwise cached == cold by
construction, pfx2-a cachecheck 22/22 identical) cost 1.28-1.85x the one-chunk cold prefill on multi-turn prompts.

Cached vs cold: the capped splits continue from the previous split, so a resume from the system prompt's end or a
last message's start computes exactly the chunks a cold prefill of its prompt computes from there (bitwise equal).
A next turn resumed at the previous prompt's exact end computes its rows in other chunks than its cold prefill would:
equal up to the reductions' rounding, gated by tools/quality_eval cachecheck compare (cached vs cold NLL). An exact
repeat computes no prompt row at all (its reply is the cold reply).

Retention (``prefixes``' GreedyDual with this lane's hooks): the points a chain's latest request kept (its last
message's start, its exact end, a cold prompt's system point) are ``prefix_protected``: never a ladder's spare and
worth all their tokens, so a conversation's newest points outlive its older ones and other chains' history points;
``prefix_reach`` lets the host tier compare against where a VRAM point actually resumes (an exact end a prompt does
not continue reaches only the point below it).

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

from .kernels.kv_pack import I_ROW
from .target import BLOCK

WIDTH = 512
OPS = (OP_SNAP, OP_RESTORE, OP_DROP, OP_COPY, OP_GROW, OP_HSPILL, OP_HRESTORE)


def snapshot_bytes(ring_bytes: int, c2_sources: int) -> int:
    """Raw bytes of one kept point's slot state on one rank: every layer's ring rows, the C2 partials and committed
    positions, the Engram history."""
    return ring_bytes + c2_sources * (2 * WIDTH * 4 + 8) + 4 * 8


def extent_bytes(layout, first: int, end: int) -> int:
    """Bytes of the KV rows of positions [first, end) (every source's rows first // r .. end // r)."""
    return sum((end // ratio - first // ratio) * (layout.compressed_row_bytes + I_ROW)
               for ratio, _, _, _ in layout.regions.values())


MIN_CHUNK = 256          # rows: no split closer than this to the previous one (a forward costs ~470 rows' work)


def chunk_splits(points, start: int, n: int, rows: int) -> list[int]:
    """Where a prompt computed from ``start`` splits its chunks in (start, n): its first message point at least
    MIN_CHUNK rows into the prompt (the system prompt's end, when past ``start``) and its last one (``points``
    sorted), each at least MIN_CHUNK rows past the previous split (``PrefixOwner._stops`` keeps points there), and
    every ``rows`` rows past the previous split (a chunk cap only). The caps continue from the previous split, so the
    splits past a kept message point are the same whether a prompt starts cold or resumes there."""
    first = next((p for p in points if p >= MIN_CHUNK), None)
    ahead = [p for p in points if start < p < n]
    wanted = sorted({first, ahead[-1]}) if ahead and first is not None and start < first < n else ahead[-1:]
    out: list[int] = []
    prev = start
    for p in (*wanted, n):
        while p - prev > rows:
            prev += rows
            out.append(prev)
        if p < n and p - prev >= MIN_CHUNK:
            out.append(p)
            prev = p
    return out


def chunk_end(bounds, start: int, n: int) -> int:
    """Where the prompt chunk at ``start`` ends: the next split, or the prompt end."""
    from bisect import bisect_right
    k = bisect_right(bounds, start)
    return bounds[k] if k < len(bounds) else n


def default_keep_slots(streams: int) -> int:
    """Idle slots that keep conversations' prefix chains in VRAM. Their caches come from the page budget (evicted
    first when a live stream needs pages); a slot's own fixed cost is its rings (5.2 MiB/rank) and the kept points'
    last-row logits on rank 0 (517 KB each, outside the page ledger), so the count stays near the stream count."""
    return max(8, streams)


class HostSlab:
    """The host tier's pinned slab of this rank (Gemma 4 SM70's sizing: ``requested`` bytes across the ranks, unset:
    Flash Next's ``default_bytes``; a power of two per rank, agreed as the smallest rank's). The size is agreed at
    construction (one collective); the pinning then runs on its own thread (host and driver work only) so it overlaps
    the rest of the startup, until ``finish`` (one collective, every rank) -> (bytes, slab). No CUDA graph may be
    captured in between (a pinned allocation is not allowed during a global-mode capture)."""

    def __init__(self, comm, rank: int, tp: int, requested: int | None):
        from tensorfold.families.qwen4_exp.cuda.prefix_host import default_bytes, pinned_slab_bytes
        if requested is not None and (type(requested) is not int or requested < 0):
            raise ValueError("prefix_host_bytes must be a nonnegative byte count")
        self.comm, self.rank, self.tp = comm, rank, tp
        self.explicit = requested is not None
        self.total = requested if self.explicit else default_bytes()
        mine = torch.tensor([pinned_slab_bytes(self.total // tp)], dtype=torch.int64, device="cuda")
        agreed = torch.empty(tp, dtype=torch.int64, device="cuda")
        comm.all_gather(mine, agreed)
        self.per_rank = int(agreed.min())
        self._pin = None
        if self.per_rank:
            from concurrent.futures import ThreadPoolExecutor
            pool = ThreadPoolExecutor(1, thread_name_prefix="ds41-slab")
            self._pin = pool.submit(lambda: torch.empty(self.per_rank, dtype=torch.uint8, pin_memory=True))
            pool.shutdown(wait=False)

    def finish(self):
        from tensorfold.families.qwen4_exp.cuda.prefix_host import pin_agreed
        if not self.per_rank:
            return 0, None
        tp = self.tp

        def every_rank(ok):
            flags = torch.empty(tp, dtype=torch.int64, device="cuda")
            self.comm.all_gather(torch.tensor([ok], dtype=torch.int64, device="cuda"), flags)
            return flags.tolist()

        slab = pin_agreed(self._pin.result, every_rank, self.per_rank, explicit=self.explicit, requested=self.total,
                          rank=self.rank, lane="DeepSeek-V4.1 SM70: ")
        return (self.per_rank, slab) if slab is not None else (0, None)


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
        self.keep = 6 * max(1, engine.keep_slots)        # kept points (a prompt: its splits and its end)
        self.depth = 0                                   # no drafter rows past the prompt
        self.prefix_counters = dict.fromkeys(prefixes.COUNTERS, 0)
        self.points = points
        self.held = 0                                    # ledger bytes: attached slot pages + snapshot pages
        self.snap_bytes = engine.snap_bytes
        self.snap_ids: set[int] = set()
        self.next_snap = 0
        self.turns = 0                                   # requests placed (``Slot.turn``: a chain's latest)
        self.turn_of: dict[int, int] = {}                # snapshot id -> the request (turn) that kept it
        self._extent = 0
        self._counted = None
        self.host_tier = HostTier(self, engine.host_per_rank) if engine.host_per_rank else None

    def retire_placement_prefixes(self) -> None:
        """Retire the old placement's retained state on the drained scheduler owner.

        Admission, running model work and the issuing stream must already be
        fenced by the caller. Complete normal snapshot/slot releases before
        asking every rank to retire its host reachability tables.
        """
        if self.streams or self.filling:
            raise RuntimeError("DeepSeek placement retirement requires an idle prefix owner")
        all_slots = {id(slot) for slot in self.slots}
        free_slots = {id(slot) for slot in self.free}
        kept_slots = {id(entry[1]) for entry in self.kept}
        if (len(all_slots) != len(self.slots) or len(free_slots) != len(self.free)
                or free_slots & kept_slots or free_slots | kept_slots != all_slots):
            raise RuntimeError("DeepSeek placement retirement found inconsistent prefix slot ownership")
        for _, slot, point, _ in self.kept:
            end = point.get("prompt_end")
            if (not slot.capacity or point["sid"] not in self.snap_ids
                    or (end is not None and end["state"]["sid"] not in self.snap_ids)):
                raise RuntimeError("DeepSeek placement retirement found an unowned retained prefix")
        expected = sum(self._bytes(slot) for slot in self.slots) + len(self.snap_ids) * self.snap_bytes
        if self.held != expected:
            raise RuntimeError("DeepSeek placement retirement found inconsistent retained bytes")
        # Regular eviction may leave attached pages in the free slot list.
        # Drop every reachable point and release every attached slot.
        self.kept.clear()
        self._sweep()
        for slot in self.slots:
            if slot.capacity:
                self._release(slot)
            else:
                slot.host_rows = None
        self.free[:] = reversed(self.slots)
        self.used.clear()
        self._counted = None
        self._extent = 0
        if self.host_tier is not None:
            self.host_tier.retire_placement_prefixes()
        if self.held or self.snap_ids or self.turn_of:
            raise RuntimeError("DeepSeek placement retirement did not release the GPU prefix ledger")

    def _cap(self, rows: int) -> int:
        return min(self.e.context, -(-rows // BLOCK) * BLOCK)

    def place(self, s, capacity: int, reuse: bool) -> None:
        super().place(s, capacity, reuse)
        self.turns += 1
        s.st.turn = self.turns                           # the points this request keeps: its chain's latest
        if not reuse:
            s.bounds = self._splits(s)

    def _splits(self, s) -> list[int]:
        if getattr(s, "message_points", None) is None:
            s.message_points = self.points(s.prompt) if self.points is not None else []
        return chunk_splits(s.message_points, s.cached, len(s.prompt), self.e.prompt_rows)

    def _stops(self, s) -> list[int]:
        """The splits of this request's chunks (``s.bounds``) that are message boundaries keep a point; the
        ``prompt_rows`` cap's splits do not (points only at natural breaks)."""
        s.bounds = self._splits(s)
        marks = set(s.message_points)
        return [b for b in s.bounds if b in marks]

    # ---- retention hooks (``prefixes``)
    def _snapshot(self, st, pos: int):
        sid = super()._snapshot(st, pos)
        if sid is not None:
            self.turn_of[sid] = getattr(st, "turn", 0)
        return sid

    def _sweep(self) -> None:
        super()._sweep()
        for sid in [sid for sid in self.turn_of if sid not in self.snap_ids]:
            del self.turn_of[sid]

    def prefix_protected(self, entry) -> bool:
        """Whether ``entry`` is one of the points its chain's latest request kept (a host restore's snapshots carry
        no request: older)."""
        turn = self.turn_of.get(entry[2]["sid"])
        return turn is not None and turn == max(self.turn_of.get(k[2]["sid"], -1) for k in self.kept
                                                 if k[1] is entry[1])

    def prefix_reach(self, entry, prompt) -> int:
        """Where a resume from VRAM point ``entry`` lands for ``prompt`` (``_resumed`` then ``_resume_at``)."""
        ids, snap, n = entry[0], entry[2], len(entry[0])
        end = snap.get("prompt_end")
        if end is not None and len(prompt) == n + 1 and prompt[-1] == end["token"]:
            return n + 1
        if state_at(snap) == n:
            return n
        if state_at(snap) == n + 1 < len(prompt) and snap.get("next") == prompt[n]:
            return n + 1
        return max((len(k[0]) for k in self.kept if len(k[0]) <= n and state_at(k[2]) == len(k[0])
                    and prompt[:len(k[0])] == k[0]), default=0)

    def _resume_at(self, s, st, resume, cached: int, sid):
        """A kept point's state is exact at its length, except an exact end's: kept one token early (``prefixes``'
        prompt-end convention), its snapshot holds the state at the prompt end (``at``), one token (``next``)
        further. A prompt continuing with that token resumes there (a next turn: the exact end extended; a fork
        copies that row too); any other lands on the longest kept point below whose state is at its length, or
        cold, and the points of ``st`` past the landing go (this request rewrites their rows), so no point whose
        state is not at its length is resumed there. (``prefixes`` demotes an end it resumes without its end:
        ``at`` and ``next`` stay; such a point never spills, ``HostTier._servable``.)"""
        state = resume["state"]
        if s.head is not None or state_at(state) == cached:
            return cached, sid
        if state_at(state) == cached + 1 < len(s.prompt) and state.get("next") == s.prompt[cached]:
            src = next((k[1] for k in self.kept if k[2] is state), st)
            if src is not st:                            # a fork copied the point's rows: also the end's last
                self.e.announce(OP_COPY, src.index, st.index, cached + 1)
            self._recount(-1)
            return cached + 1, sid
        aligned, sid = 0, None
        for ids, _st, snap, _tail in self.kept:
            n = len(ids)
            if aligned < n <= cached and state_at(snap) == n and s.prompt[:n] == ids:
                aligned, sid = n, snap["sid"]
        self._recount(cached - aligned)
        if any(k[1] is st and len(k[0]) > aligned for k in self.kept):
            self.kept = [k for k in self.kept if not (k[1] is st and len(k[0]) > aligned)]
            self._sweep()
        return aligned, sid

    def keep_end(self, s, logits) -> None:
        """The exact prompt end (state at n) as the point one token early with the last-row logits (an exact
        repeat computes nothing) and the token past it (``next``: a prompt continuing it resumes at n). When n - 1
        is a kept split, that point serves both (one row computed)."""
        n = len(s.prompt)
        ids = list(s.prompt[:n - 1])
        if n < 2 or any(k[1] is s.st and k[0] == ids and state_at(k[2]) == n - 1 for k in self.kept):
            return
        sid = self._snapshot(s.st, n)
        if sid is None:
            return
        end = {"token": s.prompt[-1], "state": {"sid": sid, "pos": n, "mtp_len": n}, "tail": None,
               "head": logits.clone(), "last": None}
        prefixes.remember(self, ids, s.st, {"sid": sid, "pos": n - 1, "mtp_len": n - 1, "exact": True, "at": n,
                                            "next": s.prompt[-1], "prompt_end": end}, None)
        self._sweep()


def state_at(snap) -> int:
    """The position a kept point's snapshot holds the state of (an exact end's ``at``, else its ``pos``)."""
    return snap.get("at", snap["pos"])


class HostTier(_gemma_host.HostTier):
    """Gemma 4 SM70's host tier; a row extent [first, end) holds every source's rows first // r .. end // r."""

    def retire_placement_prefixes(self) -> None:
        """Release every entry and its extent chain on the drained host-tier owner."""
        for entry in tuple(self.entries):
            self._drop(entry)
        if self.extents or any(self.taken):
            raise RuntimeError("DeepSeek placement retirement found an outstanding host extent")
        self._plan = None
        # The caller clears every rank's host namespace before new work.
        self._gone.clear()

    def _rows_bytes(self, first: int, rows: int) -> int:
        return extent_bytes(self.owner.e.layout, first, rows)

    def _servable(self, point) -> bool:
        """Any point whose state is at its length (a message boundary: a resume needs no alignment here) and an exact end
        with its end; an exact end ``prefixes`` demoted (its state one row past it) serves nothing from here."""
        snap = point[2]
        return snap.get("prompt_end") is not None if snap.get("exact") else state_at(snap) == len(point[0])

    def best(self, prompt, principal):
        """Gemma 4 SM70's lookup, where an exact end also serves a prompt continuing it (``_resume_at``)."""
        found, length = None, 0
        for entry in self.entries:
            if entry["principal"] != principal:
                continue
            for index, (ids, meta) in enumerate(entry["points"]):
                n = len(ids)
                if meta["exact"] and (len(prompt) <= n or prompt[n] != meta["prompt_end"]["token"]):
                    continue
                if length < n < len(prompt) and prompt[n - 1] == ids[-1] and prompt[:n] == ids:
                    found, length = (entry, index), n
        return found

    @staticmethod
    def _snap(meta, sid) -> dict:
        snap = _gemma_host.HostTier._snap(meta, sid)
        if meta["exact"]:                    # the snapshot holds the end's state (``PrefixOwner.keep_end``)
            snap.update(at=meta["prompt_end"]["pos"], next=meta["prompt_end"]["token"])
        return snap


# ---------------------------------------------------------------------------------------------- every rank: copies
class SlotState:
    """One rank's prefix copies over ``target`` (its pool pages, rings, C2 partials, Engram history) and the host
    tier's pinned ``slab``. Snapshot ``sid`` occupies pool pages owned under the key ("snap", sid)."""

    def __init__(self, target, slab=None) -> None:
        self.target, self.pool, self.slab = target, target.pool, slab
        ring_bytes = sum(ring.nbytes // ring.shape[0] for ring in self.pool.rings)
        self.raw = snapshot_bytes(ring_bytes, len(target.c2))
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
    def retire_placement_prefixes(self) -> None:
        """Clear host reachability after ordered GPU retirement and a device fence."""
        if self.snaps or self.pool.owned or len(self.pool.free) != self.pool.pages:
            raise RuntimeError("DeepSeek placement retirement requires released rank-local KV borrowers")
        self.host_layout.clear()
        self.host_extents.clear()

    def _segments(self, first: int, end: int):
        """(region view [pages, rows per page, row bytes], first row, end row) of every source's latent and index
        rows of positions [first, end), in a fixed order."""
        cache = self.pool.cache
        for layer in sorted(self.pool.layout.regions):
            ratio, per, c_offset, i_offset = self.pool.layout.regions[layer]
            for offset, size in ((c_offset, self.pool.layout.compressed_row_bytes), (i_offset, I_ROW)):
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

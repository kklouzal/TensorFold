"""Host-RAM tier for Gemma 4 SM70 prefix reuse (on by default, ``--prefix-host-bytes`` sizes it, 0 disables): spill
kept points instead of dropping them, restore them into a free slot on a later hit.

``prefixes`` (qwen4_exp) calls ``spill`` with the points of one slot chain it is about to drop (evicted, past
``keep``, or rewritten by a resume in place) while that slot's rows are still intact, and ``best``/``restore`` when
a host point saves more than ``MIN_GAIN`` tokens beyond the best VRAM point. An entry holds every point's ring
snapshot and references the slot's full-cache rows below its longest point (target and drafter) as a chain of row
extents: a spill copies only the rows past those the slab already holds for the slot's current history (an earlier
spill of the chain, or the entry it was restored from), so a conversation's turns hold each row once. A spill takes
one run of fixed blocks of each rank's pinned slab (allocated at startup, so the HTTP host pool is sized without it)
for its new extent and its entry; an extent is freed with the last entry or extent that references it. Copies are
ordered on the device stream (no host synchronization): a later write to the slot or the slab region follows them.
Restored rows and snapshots are byte copies, so a restored point resumes exactly as the VRAM-resident point would.

A restore also keeps the entry's shorter points along the restored one (a conversation's system block and earlier
message starts), so they travel with the chain to its next spill; the entry, left holding only older requests' exact
ends, drops to the lowest value and is evicted before any other entry.

Rank 0 owns the entry table and decides; every rank executes the same copies (``OP_HSPILL``/``OP_HRESTORE``) and
derives identical slab offsets from identical tensor shapes. Nothing is persisted.
"""

from __future__ import annotations

from tensorfold.families.qwen4_exp.cuda.prefix_host import BLOCK_BYTES

from .ops import OP_HRESTORE, OP_HSPILL

MAX_CHAIN = 4                # row extents a restore reads at most (``_base`` merges longer chains)
ALIGN = 256                  # byte alignment of an exact end's logits inside its entry


def _to_slab(slab, offset, tensor):
    """``tensor``'s bytes into the pinned slab at ``offset``, ordered on the device stream (no synchronization); a
    restore reads them back on the same stream after the copy, and a later spill reusing these blocks writes after
    that read. Returns the slab view."""
    size = tensor.numel() * tensor.element_size()
    return slab[offset:offset + size].view(tensor.dtype).view(tensor.shape).copy_(tensor, non_blocking=True)


class HostTier:
    def __init__(self, owner, capacity_bytes: int) -> None:
        if capacity_bytes < BLOCK_BYTES:
            raise ValueError("the host prefix tier needs at least one 2 MiB block per rank")
        self.owner = owner
        self.blocks = capacity_bytes // BLOCK_BYTES
        self.taken = bytearray(self.blocks)          # 1: block holds an entry's snapshots or a row extent
        self.entries: list[dict] = []
        self.extents = 0                             # live row extents
        self.next_id = 0                             # entry and extent ids share one space (the ranks' layouts)
        self._gone: list[int] = []                   # freed ids the ranks drop with the next spill
        self.counters = dict.fromkeys(("spills", "spill_skipped", "spill_bytes", "shared_bytes", "restores",
                                       "restore_bytes", "evictions"), 0)

    # ------------------------------------------------------------------------------------------ allocation
    def _run(self, k: int) -> int | None:
        run = 0
        for i, used in enumerate(self.taken):
            run = 0 if used else run + 1
            if run == k:
                return i - k + 1
        return None

    def _drop(self, entry) -> None:
        self.taken[entry["start"]:entry["start"] + entry["blocks"]] = bytes(entry["blocks"])
        self.entries = [x for x in self.entries if x is not entry]
        self._gone.append(entry["hid"])
        self._release(entry["extent"])

    def _release(self, extent) -> None:
        """Drop one reference to ``extent``: unreferenced, it frees its blocks and its own reference to its parent."""
        while extent is not None:
            extent["refs"] -= 1
            if extent["refs"]:
                return
            extent["alive"] = False
            self.taken[extent["start"]:extent["start"] + extent["blocks"]] = bytes(extent["blocks"])
            self._gone.append(extent["xid"])
            self.extents -= 1
            extent = extent["parent"]

    def _place(self, k: int) -> int | None:
        start = self._run(k)
        while start is None and self.entries:
            victim = min(self.entries, key=lambda x: x["value"])
            self._drop(victim)
            self.counters["evictions"] += 1
            start = self._run(k)
        return start

    def _rows_bytes(self, first: int, rows: int) -> int:
        """Slab bytes of a row extent holding rows [first, rows) (per-row caches: every row the same size)."""
        return (rows - first) * self.owner.e.token_bytes

    # ------------------------------------------------------------------------------------------ shared rows
    # As Flash Next's tier (qwen4_exp prefix_host): a spill stores the slot's rows past those the slab already holds
    # for its current history as a row extent (rows [first, end) after its parent chain's), and ``st.host_rows``
    # records which chain the slot's rows equal below which row, for one admission generation of the slot.
    def held(self, st):
        """(extent, rows, rows): the host chain ``st``'s rows equal below that row in its current admission
        generation, or None."""
        stamp = getattr(st, "host_rows", None)
        if stamp is None:
            return None
        tier, extent, rows, generation = stamp
        if tier is not self or generation != st.slot_generation or not extent["alive"]:
            return None
        return extent, rows, rows

    def hold(self, st, held, rows=None, _mtp_rows=None) -> None:
        """Record that ``st``'s rows equal ``held``'s (cut to ``rows``: where the slot's next request writes) for its
        current generation; None clears it."""
        if held is None:
            st.host_rows = None
            return
        extent, have, _ = held
        rows = have if rows is None else min(rows, have)
        while extent is not None and extent["first"] >= rows:
            extent = extent["parent"]                   # it holds nothing below the rows that stay
        st.host_rows = None if extent is None else (self, extent, rows, st.slot_generation)

    def _base(self, st, ids, rows):
        """What a spill of ``st``'s rows below ``rows`` builds on: (extent or None, first row it copies), or
        (extent, None) when that extent's chain already holds them. An extent mostly past what this spill can use
        is not built on (its needed part is copied again), so shared rows never pin more unusable rows than they
        save."""
        held = self.held(st)
        if held is None:
            return None, 0
        extent, have, _ = held
        common = min(len(extent["ids"]), len(ids), have)
        if extent["ids"][:common] != ids[:common]:
            raise RuntimeError("a slot's history changed within one admission generation: its host rows are stale")
        if rows <= have:
            while extent["parent"] is not None and extent["first"] >= rows:
                extent = extent["parent"]
            if extent["end"] - rows <= rows - extent["first"]:
                return extent, None
            base, first = extent["parent"], extent["first"]
        elif extent["end"] - have > have - extent["first"]:
            base, first = extent["parent"], min(extent["first"], have)
        else:
            base, first = extent, have
        if base is None:
            return None, 0
        # A restore copies once per extent of its chain: this extent takes in every base extent no larger than what
        # it copies (and its base past MAX_CHAIN): logarithmic chains, logarithmic copies again.
        while base["parent"] is not None and (base["depth"] >= MAX_CHAIN or first - base["first"] <= rows - first):
            first = min(first, base["first"])
            base = base["parent"]
        return base, first

    @staticmethod
    def _chain_blocks(extent) -> int:
        blocks = 0
        while extent is not None:
            blocks += extent["blocks"]
            extent = extent["parent"]
        return blocks

    # ------------------------------------------------------------------------------------------ prefixes surface
    def spill(self, points, principal) -> None:
        o = self.owner
        # A host point serves exact repeats (an exact end) or extensions from an aligned chunk boundary (``best``,
        # ``PrefixOwner.place``'s exact chunk arithmetic). A former exact end whose end a resume rewrote (``prefixes``
        # drops both) is neither unless aligned: its rows would be copied for nothing.
        points = [k for k in points if k[2].get("exact") or len(k[0]) % o.e.prompt_rows == 0]
        if not points:
            return
        st = points[0][1]
        if not st.capacity:
            return
        sids, metas, rows, heads = [], [], 0, []
        for ids, _st, snap, _tail in points:
            end = snap.get("prompt_end")
            rows = max(rows, len(ids), end["state"]["pos"] if end is not None else 0)
            if snap["sid"] not in sids:
                sids.append(snap["sid"])
            metas.append((list(ids), {"pos": snap["pos"], "snap": sids.index(snap["sid"]), "exact": bool(snap.get("exact")),
                                      "prompt_end": None if end is None else
                                      {"token": end["token"], "pos": end["state"]["pos"], "head": None}}))
            if end is not None:
                heads.append((metas[-1][1]["prompt_end"], end["head"]))
        rows = min(rows, st.capacity)
        ids = max((meta[0] for meta in metas), key=len)
        base, first = self._base(st, ids, rows)
        # One run of blocks: the new extent's rows (every rank) from its first block, then the snapshots (every
        # rank) and rank 0's exact-end logits after them (the tier's capacity bounds them too).
        x_bytes = 0 if first is None else self._rows_bytes(first, rows)
        xk = -(-x_bytes // BLOCK_BYTES)
        need = -(-(len(sids) * o.snap_bytes) // ALIGN) * ALIGN
        at = []
        for _meta, head in heads:
            at.append(need)
            need += -(-(head.numel() * head.element_size()) // ALIGN) * ALIGN
        k = -(-need // BLOCK_BYTES)
        if base is not None:
            base["refs"] += 1                        # the new extent's (or entry's) reference: no eviction frees it
        start = None
        if xk + k <= self.blocks - self._chain_blocks(base):
            start = self._place(xk + k)
        if start is None and base is not None:
            self._release(base)                      # no room beside it: copy every row
            base, first = None, 0
            x_bytes = self._rows_bytes(0, rows)
            xk = -(-x_bytes // BLOCK_BYTES)
            if xk + k <= self.blocks:
                start = self._place(xk + k)
        if start is None:
            self.counters["spill_skipped"] += 1
            return
        extent = base
        xid = -1
        if first is not None:
            xid, self.next_id = self.next_id, self.next_id + 1
            extent = {"xid": xid, "parent": base, "first": first, "end": rows, "start": start, "blocks": xk,
                      "depth": 1 if base is None else base["depth"] + 1, "refs": 1, "alive": True, "ids": list(ids)}
            self.extents += 1
        hid, self.next_id = self.next_id, self.next_id + 1
        gone, self._gone = self._gone, []
        o.e.announce(OP_HSPILL, hid, st.index, rows, start + xk,
                     [len(sids), *sids, xid, 0 if first is None else first, start, *gone])
        for (meta, head), offset in zip(heads, at):
            meta["head"] = _to_slab(o.e.host_slab, (start + xk) * BLOCK_BYTES + offset, head)
        self.taken[start:start + xk + k] = b"\x01" * (xk + k)
        if first is not None:
            st.host_rows = (self, extent, rows, st.slot_generation)
        self.entries.append({"hid": hid, "points": metas, "rows": rows, "start": start + xk, "blocks": k,
                             "extent": extent, "value": o.clock + max(len(ids) for ids, _ in metas),
                             "principal": principal})
        self.counters["spills"] += 1
        self.counters["spill_bytes"] += x_bytes + need
        self.counters["shared_bytes"] += self._rows_bytes(0, rows) - x_bytes

    def best(self, prompt, principal):
        found, length = None, 0
        for entry in self.entries:
            if entry["principal"] != principal:
                continue
            for index, (ids, meta) in enumerate(entry["points"]):
                n = len(ids)
                if meta["exact"] and not (len(prompt) == n + 1 and prompt[-1] == meta["prompt_end"]["token"]):
                    continue                    # an exact end serves only exact repeats (aligned points the rest)
                if length < n < len(prompt) and prompt[n - 1] == ids[-1] and prompt[:n] == ids:
                    found, length = (entry, index), n
        return found

    def restorable(self, entry, st) -> bool:
        """Every slot can hold any entry (uniform slots). A pure check: ``prefixes`` calls it while scanning the
        free list and the kept chains, so it must not evict or release anything."""
        return True

    def _shorter(self, entry, index) -> list[int]:
        """The entry's other points the restored point extends (their rows are below it): kept again with it."""
        ids = entry["points"][index][0]
        return [j for j, (other, _meta) in enumerate(entry["points"])
                if j != index and len(other) < len(ids) and ids[:len(other)] == other]

    def prepare(self, entry, index, st) -> bool:
        """After ``st`` grew: room for the snapshots ``restore`` charges, the restored point's and those of the
        entry's shorter points along it (a conversation's system block and earlier messages stay resumable and
        travel with the chain to its next spill), else the restored point's alone (False: none; the caller then
        also re-checks that ``entry`` survived any spill this made)."""
        o = self.owner
        points = [index] + self._shorter(entry, index)
        snaps = {entry["points"][j][1]["snap"] for j in points}
        if len(snaps) > 1 and o._room(len(snaps) * o.snap_bytes, st):
            self._plan = (entry["hid"], index, points)
            return True
        self._plan = (entry["hid"], index, [index])
        return o._room(o.snap_bytes, st)

    def restore(self, entry, index, st):
        """The entry's rows (each extent of its chain below the next one's first row) and the planned points'
        snapshots into ``st`` (one ``OP_HRESTORE``). The shorter points are kept on the chain here; the restored
        point is returned for ``prefixes`` to keep. What the chain now keeps is spilled again with it (its rows
        shared with this entry's), so an entry left holding only older exact ends (exact repeats of superseded
        requests) drops to the lowest value: other entries' points outlive it."""
        from tensorfold.families.qwen4_exp.cuda.prefixes import remember
        o = self.owner
        plan = getattr(self, "_plan", None)
        points = plan[2] if plan is not None and plan[:2] == (entry["hid"], index) else [index]
        self._plan = None
        sids, pairs = {}, []
        for j in points:
            k = entry["points"][j][1]["snap"]
            if k not in sids:
                sids[k], o.next_snap = o.next_snap, o.next_snap + 1
                pairs += [k, sids[k]]
        copies, upto, extent = [], entry["rows"], entry["extent"]
        while extent is not None:
            if extent["first"] < upto:
                copies += [extent["xid"], extent["first"], min(extent["end"], upto)]
            upto = min(upto, extent["first"])
            extent = extent["parent"]
        o.e.announce(OP_HRESTORE, entry["hid"], st.index, len(copies) // 3, len(sids), pairs + copies)
        if entry["extent"] is not None:
            st.host_rows = (self, entry["extent"], entry["rows"], st.slot_generation)
        for sid in sids.values():
            o.snap_ids.add(sid)
        o.held += len(sids) * o.snap_bytes
        for j in points[1:]:
            ids, meta = entry["points"][j]
            remember(o, list(ids), st, self._snap(meta, sids[meta["snap"]]), None)
        rest = [m for j, (_ids, m) in enumerate(entry["points"]) if j not in points]
        if len(points) > 1 and all(m["exact"] for m in rest):
            # Everything else this entry holds serves only exact repeats of its older requests: evicted first.
            entry["value"] = 0
        else:
            entry["value"] = o.clock + len(entry["points"][index][0])
        ids, meta = entry["points"][index]
        self.counters["restores"] += 1
        self.counters["restore_bytes"] += self._rows_bytes(0, entry["rows"]) + len(sids) * o.snap_bytes
        return list(ids), self._snap(meta, sids[meta["snap"]]), None

    @staticmethod
    def _snap(meta, sid) -> dict:
        snap = {"sid": sid, "pos": meta["pos"], "mtp_len": meta["pos"], **({"exact": True} if meta["exact"] else {})}
        end = meta["prompt_end"]
        if end is not None:
            snap["prompt_end"] = {"token": end["token"], "state": {"sid": sid, "pos": end["pos"], "mtp_len": end["pos"]},
                                  "tail": None, "head": end["head"].to("cuda", non_blocking=True), "last": None}
        return snap

    def record(self) -> dict:
        """The /stats ``prefix_cache.host`` record (keys as Flash Next's tier, read by server.metrics)."""
        return {"capacity_bytes_per_rank": self.blocks * BLOCK_BYTES, "used_bytes_per_rank": sum(self.taken) * BLOCK_BYTES,
                "entries": len(self.entries), "extents": self.extents,
                "points": sum(len(entry["points"]) for entry in self.entries), **dict(self.counters)}

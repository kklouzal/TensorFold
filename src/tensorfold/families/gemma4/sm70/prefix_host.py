"""Host-RAM tier for Gemma 4 SM70 prefix reuse (opt-in ``--prefix-host-bytes``): spill kept points instead of
dropping them, restore them into a free slot on a later hit.

``prefixes`` (qwen4_exp) calls ``spill`` with the points of one slot chain it is about to drop (evicted, past
``keep``, or rewritten by a resume in place) while that slot's rows are still intact, and ``best``/``restore`` when
a host point saves more than ``MIN_GAIN`` tokens beyond the best VRAM point. An entry holds the slot's full-cache
rows below its longest point (target and drafter) and every point's ring snapshot, in one contiguous run of fixed
blocks of each rank's pinned slab (allocated at startup, so the HTTP host pool is sized without it). Copies are
ordered on the device stream (no host synchronization): a later write to the slot or the slab region follows them.
Restored rows and snapshots are byte copies, so a restored point resumes exactly as the VRAM-resident point would.

A restore also keeps the entry's shorter points along the restored one (a conversation's system block and earlier
message starts), so they travel with the chain to its next spill; the entry, left holding only older requests' exact
ends, drops to the lowest value and is evicted before any other entry.

Rank 0 owns the entry table and decides; every rank executes the same copies (``OP_HSPILL``/``OP_HRESTORE``) and
derives identical slab offsets from identical tensor shapes. Nothing is persisted.
"""

from __future__ import annotations

from .ops import OP_HRESTORE, OP_HSPILL

BLOCK_BYTES = 1 << 21


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
        self.taken = bytearray(self.blocks)          # 1: block holds an entry
        self.entries: list[dict] = []
        self.next_id = 0
        self.counters = dict.fromkeys(("spills", "spill_skipped", "spill_bytes", "restores", "restore_bytes",
                                       "evictions"), 0)

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

    def _place(self, k: int) -> tuple[int | None, list[int]]:
        gone = []
        start = self._run(k)
        while start is None and self.entries:
            victim = min(self.entries, key=lambda x: x["value"])
            self._drop(victim)
            gone.append(victim["hid"])
            self.counters["evictions"] += 1
            start = self._run(k)
        return start, gone

    # ------------------------------------------------------------------------------------------ prefixes surface
    def spill(self, points, principal) -> None:
        o = self.owner
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
        # Every rank lays out rows then snapshots from the entry's first block (_host_op). Rank 0 alone keeps the exact
        # ends' last-row logits, after them in the same blocks, so the tier's capacity bounds them too.
        need = -(-(rows * o.e.token_bytes + len(sids) * o.snap_bytes) // ALIGN) * ALIGN
        at = []
        for _meta, head in heads:
            at.append(need)
            need += -(-(head.numel() * head.element_size()) // ALIGN) * ALIGN
        k = -(-need // BLOCK_BYTES)
        if k > self.blocks:
            self.counters["spill_skipped"] += 1
            return
        start, gone = self._place(k)
        hid, self.next_id = self.next_id, self.next_id + 1
        o.e.announce(OP_HSPILL, hid, st.index, rows, start, [len(sids), *sids, *gone])
        for (meta, head), offset in zip(heads, at):
            meta["head"] = _to_slab(o.e.host_slab, start * BLOCK_BYTES + offset, head)
        self.taken[start:start + k] = b"\x01" * k
        self.entries.append({"hid": hid, "points": metas, "rows": rows, "start": start, "blocks": k,
                             "value": o.clock + max(len(ids) for ids, _ in metas), "principal": principal})
        self.counters["spills"] += 1
        self.counters["spill_bytes"] += need

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
        """Rows below the entry's longest point and the planned points' snapshots into ``st`` (one ``OP_HRESTORE``).
        The shorter points are kept on the chain here; the restored point is returned for ``prefixes`` to keep. What
        the chain now keeps is spilled again with it, so an entry left holding only older exact ends (exact repeats of
        superseded requests) drops to the lowest value: other entries' points outlive it."""
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
        o.e.announce(OP_HRESTORE, entry["hid"], st.index, entry["rows"], len(sids), pairs)
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
        self.counters["restore_bytes"] += entry["rows"] * o.e.token_bytes + len(sids) * o.snap_bytes
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
                "entries": len(self.entries), "points": sum(len(entry["points"]) for entry in self.entries),
                **dict(self.counters)}

"""Gemma 4 SM70 prompt-prefix reuse: Flash Next's retention policy (qwen4_exp ``prefixes``) over this engine's slots.

A kept point is (prompt ids, slot, snapshot record, None): the slot's full-attention caches hold every row below the
point (rows past it are rewritten before any read) and a snapshot holds the sliding-window ring rows the point needs
(target and drafter rings are overwritten as the slot moves on). Points are the template's message starts
(``cuda.markers.resume_points``: the second message, the last assistant and the last other message) and the prompt's
end less one token (``entry_end``: a next turn re-renders the generation prompt's tail), which also keeps the exact
prompt end with its last-row logits, so an exact repeat resumes with no prompt row at all and replies exactly as the
original request did. Placement (resume in place, fork into a free slot, cold), GreedyDual-Size values and the
chain ladder are ``prefixes``' own; this mixin supplies the owner surface it reads (``kept``, ``free``, ``used``,
``clock``, ``keep``, ``depth``, ``_busy``, ``_grow``, ``_drop_kept`` ...) and the memory ledger: full caches and
snapshots are charged to the engine's KV budget; a request that needs room first releases free slots' caches,
then evicts the least valuable idle chains.

Every operation is announced to all ranks (rank 0 decides); snapshot ids are rank-replicated.
"""

from __future__ import annotations

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.markers import MIN_GAP
from tensorfold.families.qwen4_exp.cuda import prefixes

from .ops import OP_ATTACH, OP_COPY, OP_DETACH, OP_DROP, OP_GROW, OP_RESTORE, OP_SNAP


def entry_end(prompt) -> int:
    """Where a prompt's kept point ends: one token early (a next turn re-renders the generation prompt's tail)."""
    return max(1, len(prompt) - 1)


class Slot:
    """One KV slot as ``prefixes`` sees it: identity, resident window (``limit``) and the fork copy."""

    compact_indexer = None

    def __init__(self, owner, index: int) -> None:
        self.owner, self.index = owner, index
        self.capacity = 0                  # attached full-cache rows (0: detached)

    @property
    def limit(self) -> int:
        return self.capacity

    def begin_sequence_generation(self) -> None:
        """Nothing slot-generation-scoped to retire here: every write is ordered on the one device stream."""

    def copy_prefix(self, src: "Slot", n: int, mtp_len: int, **_ignored):
        self.owner.e.announce(OP_COPY, src.index, self.index, n)
        return None

    def __repr__(self) -> str:
        return f"Slot({self.index}, {self.capacity})"


class PrefixOwner:
    """Mixin for the engine's Decoder (rank 0): slots, the prefix-cache owner surface and the KV ledger."""

    def _init_prefix(self, engine, points) -> None:
        self.slots = [Slot(self, i) for i in range(engine.streams + engine.keep_slots)]
        self.free = list(reversed(self.slots))          # prefixes pops from the end
        self.kept: list = []
        self.used: dict[int, int] = {}
        self.clock = 0
        self.keep = 4 * max(1, engine.keep_slots)        # kept points (a prompt keeps up to four)
        self.depth = engine.block - 1
        self.host_tier = None
        self.prefix_counters = dict.fromkeys(("lookups", "resume_hits", "resume_tokens", "fork_hits", "fork_tokens",
                                              "host_hits", "host_tokens"), 0)
        self.points = points                             # message starts of a prompt (markers), or None
        self.held = 0                                    # ledger bytes: attached full caches + snapshots
        self.snap_bytes = engine.target.pool.snapshot_bytes() + (
            engine.head.snapshot_bytes() if engine.head is not None else 0)
        self.snap_ids: set[int] = set()
        self.next_snap = 0
        self._extent = 0

    # ------------------------------------------------------------------------------------------ owner surface
    def _busy(self) -> set[int]:
        return {id(s.st) for s in (*self.streams.values(), *self.filling) if s.st is not None}

    def _cap(self, rows: int) -> int:
        from .target import CHUNK
        return min(self.e.target.alloc, -(-rows // CHUNK) * CHUNK)

    def _bytes(self, st: Slot) -> int:
        return self.e.slot_bytes(st.capacity) if st.capacity else 0

    def _room(self, need: int, *spared) -> bool:
        """Make ``need`` more ledger bytes available: free slots' caches first, then the least valuable idle chain
        (never ``spared``). False when the live streams hold the rest."""
        budget = self.e.budget
        while self.held + need > budget:
            idle = next((st for st in self.free if st.capacity and all(st is not x for x in spared)), None)
            if idle is None:
                victim = prefixes.evict(self, *(spared + (None, None))[:2])
                if victim is None:
                    return False
                self._sweep()
                self.free.append(victim)
                idle = victim
            self._release(idle)
        return True

    def _release(self, st: Slot) -> None:
        self.held -= self._bytes(st)
        st.capacity = 0
        self.e.announce(OP_DETACH, st.index)

    def _grow(self, st: Slot, rows: int, protect=None) -> bool:
        """The slot holds ``max(rows, the admitted request's extent)`` rows, its rows kept; False without room."""
        cap = self._cap(max(rows, self._extent))
        if st.capacity >= cap:
            return True
        need = self.e.slot_bytes(cap) - self._bytes(st)
        if not self._room(need, st, protect):
            return False
        if st.capacity:
            self.e.announce(OP_GROW, st.index, cap, st.capacity)
        else:
            self.e.announce(OP_ATTACH, st.index, cap)
        self.held += need
        st.capacity = cap
        return True

    def _drop_kept(self, st: Slot) -> None:
        self.kept = [k for k in self.kept if k[1] is not st]
        self.used.pop(id(st), None)
        self._sweep()

    def _quarantine(self, st: Slot) -> None:
        """A failed placement: the ranks' slot state is unknown; the decoder stops serving (``broken``)."""
        self.broken = self.broken or RuntimeError(f"a Gemma 4 prefix placement failed in slot {st.index}")

    def _retire_failed_slot(self, st: Slot) -> None:
        self._drop_kept(st)

    # ------------------------------------------------------------------------------------------ snapshots
    def _snapshot(self, st: Slot, pos: int) -> int | None:
        """Snapshot ``st``'s rings at ``pos`` (charged to the ledger), or None without room."""
        if not self._room(self.snap_bytes, st):
            return None
        sid, self.next_snap = self.next_snap, self.next_snap + 1
        self.e.announce(OP_SNAP, st.index, pos, sid)
        self.snap_ids.add(sid)
        self.held += self.snap_bytes
        return sid

    def _sweep(self) -> None:
        """Drop snapshots no kept point (or kept prompt end) refers to."""
        live = set()
        for _ids, _st, snap, _tail in self.kept:
            live.add(snap["sid"])
            end = snap.get("prompt_end")
            if end is not None:
                live.add(end["state"]["sid"])
        for sid in sorted(self.snap_ids - live):
            self.snap_ids.discard(sid)
            self.held -= self.snap_bytes
            self.e.announce(OP_DROP, sid)

    # ------------------------------------------------------------------------------------------ admission
    def place(self, s, capacity: int, reuse: bool) -> None:
        """Choose ``s``'s slot (reuse a kept point, or cold) with ``capacity`` full-cache rows; NoRoom when the
        live streams hold the memory. Sets ``s.st``, ``s.slot``, ``s.cached``, ``s.pos`` and ``s.head`` (an exact
        repeat's stored last-row logits)."""
        self._extent = capacity
        try:
            st, resume, cached = prefixes.slot_for(self, s.prompt, reuse)
        except RuntimeError as error:
            raise NoRoom(f"a {len(s.prompt)}-token prompt waits for a free slot") from error
        finally:
            self._sweep()
        if resume is None:
            cached = 0
            if st.capacity != capacity:
                need = self.e.slot_bytes(capacity) - self._bytes(st)
                if need > 0 and not self._room(need, st):
                    self.free.append(st)
                    raise NoRoom(f"a {len(s.prompt)}-token prompt waits for KV memory until a live stream finishes")
                self.held += need
                st.capacity = capacity
            self.e.announce(OP_ATTACH, st.index, capacity)
            s.head = None
        else:
            if not self._grow(st, capacity):
                if all(k[1] is not st for k in self.kept):
                    self.free.append(st)
                raise NoRoom(f"a {len(s.prompt)}-token prompt waits for KV memory until a live stream finishes")
            self.e.announce(OP_RESTORE, st.index, resume["state"]["sid"])
            s.head = resume.get("head")
        s.st, s.slot, s.cached, s.pos = st, st.index, cached, cached
        s.stops = self._stops(s) if reuse else []

    def _stops(self, s) -> list[int]:
        """Positions whose states this prompt keeps: message starts past the reused prefix, then its end."""
        n = len(s.prompt)
        end = entry_end(s.prompt)
        found = self.points(s.prompt) if self.points is not None else []
        stops = sorted({p for p in found if s.cached + MIN_GAP <= p < end})
        return stops + ([end] if s.cached < end and n >= 2 else [])

    def keep_points(self, s, start: int, stop: int, logits) -> None:
        """After the prompt chunk [start, stop): keep the stream's points inside it (``logits``: the prompt's
        last-row logits when the chunk ends the prompt)."""
        n = len(s.prompt)
        while s.stops and s.stops[0] <= stop:
            p = s.stops.pop(0)
            if p < start:
                continue
            last = p == entry_end(s.prompt) and stop == n and n >= 2
            at = n if last else p
            sid = self._snapshot(s.st, at)
            if sid is None:
                return
            snap = {"sid": sid, "pos": p, "mtp_len": p}
            if last:
                snap["prompt_end"] = {"token": s.prompt[-1], "state": {"sid": sid, "pos": n, "mtp_len": n},
                                      "tail": None, "head": logits.clone(), "last": None}
            prefixes.remember(self, list(s.prompt[:p]), s.st, snap, None)
            self._sweep()

    def retire_slot(self, s) -> None:
        """A finished stream's slot: idle kept when it holds points, else free (its caches released)."""
        st, s.st = s.st, None
        if st is None:
            return
        if any(k[1] is st for k in self.kept):
            return
        if all(x is not st for x in self.free):
            self.free.append(st)
        self._release(st)

    def prefix_usage(self) -> dict:
        kept = self.kept
        ends: dict[int, int] = {}
        for ids, slot, _snap, _tail in kept:
            ends[id(slot)] = max(ends.get(id(slot), 0), len(ids))
        return {"entries": len(kept), "capacity": self.keep, "slots": len(ends), "tokens": sum(ends.values()),
                "snapshots": len(self.snap_ids), **dict(self.prefix_counters), "host": None}

"""Flash Next's concurrent rounds on one GPU: every stream keeps exactly its own accepted prefix."""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda import pdl
from tensorfold.cuda.capacity import LIMIT_ENV, available_bytes, cuda_limit_bytes
from tensorfold.cuda.kernels import gdn
from tensorfold.cuda.logprobs import capture
from tensorfold.cuda.markers import MIN_GAP
from tensorfold.cuda.memory_gate import MemoryGate, NoRoom, torch_live
from tensorfold.cuda.sampling import (
    TokenMap,
    collect_streams,
    launch_streams,
    repaired_choose,
    validate_policy,
    validate_rows,
)
from tensorfold.cuda.streams import Stream, accept
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import GrammarError

from ..cuda import CONFIDENCE, DEPTH
from . import attn_multi, gdn_multi, image_rows, multi_graphs, prefixes
from .decode import PREFILL_ROWS, WARM_TAIL, Engine, draft, entry_end, prefill_begin
from .forward import Cut, KVWords, _kv_check, commit, compute, compute_mixed, converges, cut_snapshot, read_ahead, stage
from .mtp import mtp_compute, mtp_stage
from .prompt_plan import CUTS, pass_limit
from .state import ENDS, Buffers, KVNumericError, State

FIRST, STEP = 256, 8192          # rows an idle slot keeps; rows a stream's caches grow by at a time
GIB = 1024**3
SHARE = 0.0                      # --decode-share: a round alone takes this share of its pass's time (0: whole passes)
PASS_MIN = 128                   # the fewest prompt rows a round's pass takes
COPY_MATCH = 8                   # copy drafts: the context's last this many tokens seen before, and as many after them
FILL_GUARD = 8                   # a prompt passed over this many passes takes the next one (no starvation), as on Macs


def _slot(w, st: State, buf: Buffers, mbuf: Buffers, pbuf: Buffers, capacity: int, prefill_rows: int) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs)."""

    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, buf.rows, prefill_rows
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    e.kv_pair = st.kv_pair
    e.stops = ()
    return e


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``capacity`` tokens of context."""

    def __init__(self, w, *, slots: int, capacity: int, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 stop_eos: bool = True, keep: int = 8, kv_dtype: str = "bf16", prefill_rows: int = PREFILL_ROWS,
                 share: float = SHARE, points=None, vision=None, workspace_bytes: int = 0, kv_pair=None,
                 kv_key_dtype: str | None = None, kv_value_dtype: str | None = None, graphs: bool = False) -> None:
        if w.comm is not None:
            raise ValueError("concurrent Flash Next runs on one GPU for now")
        self.w, self.depth, self.confidence, self.capacity = w, depth, confidence, capacity
        from .rotor_decoder import RotorDecoderPolicy
        self.rotor_decoder_policy = RotorDecoderPolicy()
        self.points = points                         # a prompt's message starts to keep states at, or None
        self.vision = vision
        self.copy = os.environ.get("TENSORFOLD_MTP_COPY", "0") == "1"
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows = slots * (depth + 1)
        # a round's window and a prompt pass share each layer's expert launch: the pass's buffers hold both
        self.converged, self.prefill_rows = converges(w), prefill_rows
        # rounds beside a filling prompt size its pass so decoding keeps ``share`` of the pass's time (0: whole passes)
        self.share, self.round_s, self.row_s = share, None, None
        self.buf = Buffers(w, rows, capacity, moe_prefill=True)
        self.mbuf = Buffers(w, rows, capacity) if w.mtp is not None else None
        self.pbuf = Buffers(w, prefill_rows + (rows if self.converged else 0), capacity, prefill=True)
        from .hc_plans import enroll

        enroll(w, self.buf)
        enroll(w, self.mbuf, mtp=True)
        self.gdn = gdn_multi.Scratch(w, rows)            # every stream's DeltaNet rows, one launch a step
        # a frame's codec status words, read back with its sampling packet (KVWords): one host wait a step
        self.kv_host = torch.zeros((slots, 2), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        # every step's tables at fixed addresses and its kernels at a fixed geometry, so captured forwards replay
        self.round_tables = gdn_multi.RoundTables(w, self.gdn, slots, depth + 1)
        self.step_tables = attn_multi.StepTables(w, slots, depth + 1, mtp=False)
        self.mtp_tables = attn_multi.StepTables(w, slots, depth + 1, mtp=True) if w.mtp is not None else None
        # captured rounds (multi_graphs; ``capture_graphs`` at warm-up): None until then, or when not captured
        self.graphs: multi_graphs.RoundGraphs | None = None
        reason = None if graphs else "not requested"
        budget = multi_graphs.budget_bytes() if graphs else 0
        if graphs and not budget:
            reason = f"{multi_graphs.ENV}=0"
        elif graphs:
            reason = multi_graphs.capturable(w)
        self.graph_budget = budget if reason is None else 0
        self.graph_record: dict = {"captured": False, "reason": reason}
        # programmatic dependent launch of the decode kernels: this chip's policy, once (pdl.ON is set each round)
        self.pdl = pdl.select(w.device.index or 0) if torch.cuda.is_available() else False
        pdl.ON = self.pdl
        self.held: dict[int, list[int]] = {}             # stream id -> last round's kept rows, folded in next round
        # slots start small and grow with their stream's context, up to the window, while the gate has room
        self.free = [State(w, min(capacity, FIRST), depth + 1, kv_dtype, limit=capacity, kv_pair=kv_pair,
                           kv_key_dtype=kv_key_dtype, kv_value_dtype=kv_value_dtype) for _ in range(slots)]
        self.failed_slots: list[State] = []          # bounded by slots; exclusive owners until retirement succeeds
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.free[0]))
        self.window_bytes = self.free[0].cache_bytes(capacity)          # one stream's caches at the full window
        free = torch_live(torch, available_bytes) if torch.cuda.is_available() else None
        # the mapped n-gram tables are not held back (they barely fit on a Spark); lookups page from disk instead
        live = free
        self.memory_gate = MemoryGate(live() if live is not None else 1 << 62,
                                      reserve=max(2 * GIB, workspace_bytes), live=live)
        self._memory_error: BaseException | None = None
        self.streams: dict[int, Stream] = {}
        self.filling: list[Stream] = []                  # admitted, prompts still prefilling (oldest first)
        # stream id -> [its engine, drafts?, next row, kept states, the streams row before the next (its MTP tail)]
        self.fills: dict[int, list] = {}
        self.passed: dict[int, int] = {}                 # stream id -> prompt passes since its last rows
        self.arrived = lambda: False                     # a request waits to be admitted (the scheduler sets this)
        self.next_id = 0
        self._draft_map = TokenMap(w.draft_ids, int(w.draft_ids.numel()), device=w.draft_ids.device) \
            if w.draft_ids is not None else None
        self.draft_host = self._draft_map.ids(int(w.draft_ids.numel())) if self._draft_map is not None else None
        self.kept: list[tuple[list[int], State, dict, torch.Tensor | None]] = []   # (ids, slot, snapshot, tail)
        self.keep = keep
        self.used: dict[int, int] = {}                   # retention (prefixes.py): id(slot) -> clock at its last use
        self.clock = 0

    def _rotor_lookup(self, request, participants=1):
        return self.rotor_decoder_policy.request(request, owned_requests=self.live(),
                                                 participants=participants, copy=self.copy)

    def _busy(self) -> set[int]:
        return {id(s.st) for s in [*self.streams.values(), *self.filling]}

    def _drop_kept(self, st: State) -> None:
        self.kept = [k for k in self.kept if k[1] is not st]

    def _grow(self, st: State, rows: int, *, alone: bool = False, protect: State | None = None) -> bool:
        """Grow caches while the gate has room; one stream may use its startup allowance within an explicit cap."""

        if rows <= st.capacity or st.capacity >= st.limit:       # admission's count keeps a stream within its window
            return True
        size = min(st.limit, -(-rows // STEP) * STEP)
        before = st.cache_bytes()
        grow = st.cache_bytes(size) - before
        while not self.memory_gate.fits(grow + st.layer_bytes(size)):     # a layer's old buffers stay until its copy
            if not self._evict_kept(st, protect=protect):
                if alone:
                    if not self.memory_gate.allow_lone_growth:
                        raise NoRoom(
                            f"Growing this request's attention caches to {size} tokens cannot fit the shared "
                            "KV reservation and remaining device memory. Retry it, shorten the prompt or "
                            "max_tokens, reduce --context, or increase --kv-full-windows.")
                    limit = cuda_limit_bytes() if torch.cuda.is_available() else None
                    if limit is not None:
                        # A lone stream may use its startup reserve without exceeding the copy peak cap.
                        peak = int(torch.cuda.memory_allocated()) + grow + st.layer_bytes(st.capacity)
                        if peak > limit:
                            if self.filling:
                                return False             # a filling request can finish and release its slot
                            raise NoRoom(
                                f"Growing this request's attention caches to {size} tokens would exceed "
                                f"{LIMIT_ENV}={limit / GIB:g}: the estimated copy peak is {peak / GIB:.2f} GiB. "
                                "Shorten the prompt or max_tokens, reduce --context or --parallel, or raise "
                                f"{LIMIT_ENV} if more GPU memory is available.")
                    break
                return False
        # This worker is the gate's sole writer. Capture after any evictions,
        # then reconcile from owned storage even if replacement or its return
        # is interrupted; a partially executed take() cannot double-charge.
        gate, held_before, primary = self.memory_gate, self.memory_gate.held, None
        try:
            st.resize(size)
        except BaseException as error:  # noqa: BLE001 - charge attached storage even after interrupted replacement
            primary = error
            raise
        finally:
            if getattr(st, "_resize_frames_pending", None) is not None:
                # Uncommitted traceback-held storage is outside the attached sum.
                self._memory_error = primary
            try:
                gate.held = held_before + st.cache_bytes() - before
            except BaseException as cleanup:  # noqa: BLE001 - preserve failed resize and accounting outcomes
                self._memory_error = cleanup
                raise_failures(primary, [cleanup])
        if torch.cuda.is_available():
            torch.cuda.empty_cache()             # the old buffers back to the system: MemAvailable stays true
        return True

    def _shrink(self, st: State, *, force: bool = False) -> None:
        """An idle slot back to its first rows: its caches' memory returns to the gate."""

        gate, held_before = self.memory_gate, self.memory_gate.held
        before, primary, errors = st.cache_bytes(), None, []
        try:
            if st._resize_error is not None:
                st.recover_resize(self.w, min(FIRST, st.limit))
            else:
                st.reset(self.w)
                if force or st.capacity > FIRST:
                    st.resize(min(FIRST, st.limit))
        except BaseException as error:  # noqa: BLE001 - retire owned storage even on cancellation
            primary = error
            try:
                self._quarantine(st)
            except BaseException as cleanup:  # noqa: BLE001 - preserve every cleanup failure with its primary
                errors.append(cleanup)
        try:
            # Even a failed rebuild/reset may have changed attached storage.
            if getattr(st, "_resize_frames_pending", None) is not None:
                self._memory_error = primary
            gate.held = held_before + st.cache_bytes() - before
        except BaseException as cleanup:  # noqa: BLE001 - preserve every cleanup failure with its primary
            self._memory_error = cleanup
            errors.append(cleanup)
            try:
                self._quarantine(st)
            except BaseException as ownership:  # noqa: BLE001 - retain failures while preserving slot ownership
                errors.append(ownership)
        raise_failures(primary, errors)

    def _quarantine(self, st: State) -> None:
        """Retain a failed exclusive slot; neither admission nor kept prefixes may borrow it."""

        if all(owned is not st for owned in self.failed_slots):
            self.failed_slots.append(st)
        self._drop_kept(st)
        self.free = [owned for owned in self.free if owned is not st]

    def _retire_failed_slot(self, st: State) -> None:
        self._shrink(st, force=True)
        try:
            self.free.append(st)
            self.failed_slots.remove(st)
        except BaseException as primary:  # noqa: BLE001 - undo partial publication before a later retry
            try:
                self._quarantine(st)
            except BaseException as cleanup:  # noqa: BLE001 - preserve publication and ownership-repair failures
                raise_failures(primary, [cleanup])
            raise

    def recover_failed_slots(self) -> None:
        """Retry exclusive retirement after request/worker quiescence; failed owners remain retained."""

        errors = []
        for st in tuple(self.failed_slots):
            try:
                self._retire_failed_slot(st)
            except BaseException as error:  # noqa: BLE001 - retire owned storage even on cancellation
                errors.append(error)
        raise_failures(None, errors)

    def _evict_kept(self, keep: State, *, protect: State | None = None) -> bool:
        """Free the least valuable idle kept chain (never ``keep`` or ``protect``); False when none is left."""

        st = prefixes.evict(self, keep, protect)
        if st is None:
            return False
        self._shrink(st)
        if all(f is not st for f in self.free):
            self.free.append(st)
        return True

    def _flush(self, s) -> None:
        """Materialize deferred recurrent rows before waiting releases the shared scratch."""

        rows = self.held.pop(s.sid, [])
        if not rows:
            return
        sc, st = self.gdn, s.st
        parity = 1 - sc.parity
        if not sc.lin:
            return
        k, v, g, beta = [[getattr(sc, name)[parity, li] for li in range(sc.lin)]
                         for name in ("k", "v", "g", "beta")]
        ptrs = gdn.replay_table(k, v, g, beta, [[st.rec[st.cur[li], li] for li in range(sc.lin)]],
                                in_place=True)
        table = gdn.to_device(ptrs, torch.int64, self.w.device)
        kept = gdn.to_device(rows, torch.int32, self.w.device).view(1, -1)
        counts = gdn.to_device([len(rows)], torch.int32, self.w.device)
        gdn.replay(table, sc.lin, 1, kept, counts, k[0], v[0], in_place=True)

    def _make_room(self) -> list[Stream]:
        """Before a round: grow each live window oldest-first; a stream that can't grow makes the newest end."""

        live = sorted((s for s in self.streams.values() if not s.done), key=lambda s: s.sid)
        alone = len(live) == 1 and (self.memory_gate.allow_lone_growth or
                                   not any(not s.done for s in self.filling))
        blocked, ended = False, []
        for s in live:
            rows = max(s.st.pos, s.st.mtp_len) + len(s.drafts) + self.depth + 2
            try:
                s.waiting = rows > s.st.capacity if blocked else not self._grow(s.st, rows, alone=alone)
            except NoRoom as exc:
                s.error, s.done, s.waiting = exc, True, False
                self.held.pop(s.sid, None)
                self._drop_kept(s.st)
                self.memory_gate.ends += 1
                ended.append(s)                          # finish() releases only this request's slot
                continue
            blocked = blocked or s.waiting
        if live and live[0].waiting and len(live) > 1:        # even the oldest can't grow: the newest ends
            newest = live[-1]
            newest.error = RuntimeError(
                f"This server ran out of memory with {len(live)} streams decoding, so the newest (this request, after "
                f"{len(newest.out)} tokens) was stopped for the older ones to finish. Retry it, shorten the prompt or "
                "max_tokens, or start the server with a smaller --parallel.")
            newest.done, newest.waiting = True, False
            self.memory_gate.ends += 1
            self.streams.pop(newest.sid, None)
            self.held.pop(newest.sid, None)
            self._drop_kept(newest.st)
            self._shrink(newest.st)
            self.free.append(newest.st)
            return [*ended, newest, *self._make_room()]
        for s in live:
            if s.waiting:
                self._flush(s)                 # shared scratch will be reused while this stream waits
        self.memory_gate.waits += any(s.waiting for s in live)
        return ended

    def _slot_for(self, prompt: list[int], reuse: bool):
        """Reuse the longest kept point, copying a fork into a free slot when the memory gate permits it."""

        best = prefixes._best(self.kept, prompt) if reuse else None
        if best is not None:
            source, snap = best[1:3]
            try:
                source.kv_validate_snapshot(snap)
            except KVNumericError:
                self._drop_kept(source)
                if id(source) not in self._busy():
                    self._shrink(source)
                    if all(st is not source for st in self.free):
                        self.free.append(source)
                raise
        return prefixes.slot_for(self, prompt, reuse)

    def _remember(self, ids: list[int], st: State, snap: dict, tail) -> None:
        st.kv_validate_snapshot(snap)
        prefixes.remember(self, ids, st, snap, tail)

    def live(self) -> int:
        return len(self.streams) + len(self.filling)

    def usage(self) -> dict:
        """/stats' KV-memory and prefix-cache readings, from another thread, without a lock, each on its own: the gate's
        ints, ``kept`` as published (this worker replaces it, never changes it in place), and ``streams``/``filling``
        each copied whole by one unpacking before it is iterated (one C loop under the GIL: ``server.stats``).

        ``kv``: the memory gate's budget for growing stream caches (``room`` less ``reserve``), what they hold (kept
        prefix-cache slots included), and what the slots of running requests hold beyond their first rows (vLLM's
        usage: kept entries are evicted for a request that needs room, so they count as free). ``free_tokens_estimate``
        is the context that budget less the running requests' share would hold; host memory and the resize copy can
        still refuse it, and admission stays the authority."""

        gate = self.memory_gate
        budget = gate.room - gate.reserve
        busy = [s.st for s in [*self.streams.values(), *self.filling] if s.st is not None]
        first = min(self.capacity, FIRST)
        active = sum(max(0, st.cache_bytes() - st.cache_bytes(first)) for st in {id(st): st for st in busy}.values())
        per_token = -(-self.window_bytes // self.capacity) if self.capacity else 0
        kept = self.kept
        ends: dict[int, int] = {}
        for ids, slot, _snap, _tail in kept:
            ends[id(slot)] = max(ends.get(id(slot), 0), len(ids))
        return {"kv": {"budget_bytes": budget, "held_bytes": gate.held, "active_bytes": active,
                       "usage_ratio": round(active / budget, 6) if budget > 0 else None,
                       "bytes_per_token": per_token,
                       "free_tokens_estimate": max(0, budget - active) // per_token if per_token else None},
                "prefix_cache": {"entries": len(kept), "capacity": self.keep, "slots": len(ends),
                                 "tokens": sum(ends.values())},
                "memory_waits": gate.waits, "memory_stops": gate.ends}

    @torch.no_grad()
    def warm(self) -> None:
        """A synthetic greedy request through prefill (a full chunk, then a partial one cut at the kept point), its drafts and one round, then forgotten, so no request compiles or loads a kernel."""

        s = Stream([0] * min(self.prefill_rows + WARM_TAIL + 1, self.capacity - self.depth - 2), 2)
        self.admit(s)
        if not s.done:
            self.round()                                 # the whole prompt (nothing else decodes), then a round
        self.streams.pop(s.sid, None)
        self._drop_kept(s.st)
        self._shrink(s.st)
        if all(f is not s.st for f in self.free):
            self.free.append(s.st)
        if self.graph_budget and self.graphs is None:
            self.capture_graphs()

    @torch.no_grad()
    def capture_graphs(self) -> None:
        """Capture the rounds' verify forwards and MTP steps on the idle slots (``multi_graphs``) and take their
        measured memory (``room_bytes``) from the stream caches' room. Before any request: every slot must be free."""

        if self.streams or self.filling or self.kept or len(self.free) != self.round_tables.slots:
            raise RuntimeError("graphs are captured on idle slots, before any request")
        graphs = multi_graphs.RoundGraphs(self, self.graph_budget)
        graphs.capture()
        self.graphs = graphs
        self.memory_gate.room -= graphs.room_bytes
        self.graph_record = {"captured": True, **graphs.record()}

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Queue a request in a free slot (a kept prompt end it extends, if any); rounds prefill its prompt."""

        error = getattr(self, "_memory_error", None)
        if error is not None:
            raise error
        room = self.capacity - len(s.prompt) - self.depth - 1
        if room < 1:
            raise ValueError(f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.capacity}-token context")
        s.count = max(1, min(s.count, room))
        if any(x.waiting for x in self.streams.values()):
            raise NoRoom("streams already wait for memory; a new request waits until one finishes")
        t0 = time.perf_counter()
        st, resume, s.cached = self._slot_for(list(s.prompt), s.draft and s.vision is None)
        try:
            if not self._grow(st, len(s.prompt) + self.depth + 2, alone=not self.streams and not self.filling):
                raise NoRoom(f"a {len(s.prompt)}-token prompt waits for memory until a live stream finishes")
        except NoRoom:
            if resume is None:
                self.free.append(st)
            else:
                self._remember(list(s.prompt[:s.cached]), st, resume["state"], resume["tail"])
            raise
        except BaseException as primary:  # noqa: BLE001 - recover the popped slot even after interrupted growth
            try:
                self._quarantine(st)
                self._retire_failed_slot(st)
            except BaseException as cleanup:  # noqa: BLE001 - preserve every cleanup failure with its primary
                raise_failures(primary, [cleanup])
            raise
        e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity, self.prefill_rows)
        e._draft_map = self._draft_map
        mtp = s.draft and self.depth > 0 and self.mbuf is not None
        try:
            begin = prefill_begin(e, s.prompt, mtp=mtp, resume=resume)
            image = s.vision is not None
            image_rows.begin(e, s, self.vision)
            if image:
                torch.cuda.empty_cache()                 # the tower's scratch back before the prompt's passes
        except Exception:
            self._drop_kept(st)
            self._shrink(st)
            self.free.append(st)
            raise
        e.stops = sorted({p for p in self.points(s.prompt) if begin + MIN_GAP <= p < entry_end(s.prompt)}) \
            if s.draft and st.image_positions is None and self.points is not None else []
        s.sid, s.st = self.next_id, st
        self.next_id += 1
        if self.copy and s.draft:
            from tensorfold.families.qwen3_5.cuda.decode import CopyIndex

            s.copies = CopyIndex(COPY_MATCH)
        s.prefill_s = time.perf_counter() - t0
        # the resumed point stays kept (slot_for keeps it in place, or a fork's source holds it): no fill re-keeps it
        self.fills[s.sid] = [e, mtp, begin, [], None]
        self.filling.append(s)

    def _fill(self) -> list[Stream]:
        """Fill ordered prompt pieces until a stream decodes, a waiting request needs admission or a filling prompt's
        client has left (the scheduler then calls ``cancel`` before the next round)."""

        ended: list[Stream] = []
        while self.filling:
            ended += self._pass()
            if any(not x.done and not x.waiting for x in self.streams.values()) or self.arrived() or \
                    any(s.cancel is not None and s.cancel.is_set() for s in self.filling):
                break
        return ended

    def _pass_rows(self) -> int:
        """A round's prompt rows: its decode (a round alone) takes ``share`` of the pass's time, by the last rounds."""

        live = any(not s.done for s in self.streams.values())
        return pass_limit(self.prefill_rows, live, self.share, self.round_s, self.row_s, PASS_MIN)

    def _timed(self, seconds: float, rows: int) -> None:
        """A round's wall time: a round alone updates its estimate, a round with a pass the seconds a row adds."""

        if rows:
            extra = max(0.0, seconds - (self.round_s or 0.0)) / rows
            self.row_s = extra if self.row_s is None else 0.7 * self.row_s + 0.3 * extra
        else:
            self.round_s = seconds if self.round_s is None else 0.7 * self.round_s + 0.3 * seconds

    def _order(self) -> list[Stream]:
        """Order overdue prompts first, then foreground, fewest rows left and stable arrival order."""

        def key(s: Stream):
            passed = self.passed.get(s.sid, 0)
            due = passed >= FILL_GUARD
            return (not due, -passed if due else 0, s.background, len(s.prompt) - self.fills[s.sid][2])

        return sorted(self.filling, key=key)                         # stable: ties keep arrival order

    def _pieces(self, rows: int | None = None) -> list[tuple[Stream, int, int]]:
        """The next pass: rows from the filling prompts in ``_order``, up to ``rows`` and ENDS ending prompts."""

        pieces, room = [], self._pass_rows() if rows is None else rows
        for s in self._order():
            start = self.fills[s.sid][2]
            n = min(len(s.prompt) - start, room)
            inside = [p for p in self._points(s, start, n) if p < start + n]
            if len(inside) > CUTS:                       # a piece holds CUTS kept states at most: the next one ends it
                n = inside[CUTS] - start
            ends = sum(1 for x, a, k in pieces if a + k == len(x.prompt))
            if n == 0 or (start + n == len(s.prompt) and ends == ENDS):
                break
            pieces.append((s, start, n))
            room -= n
        return pieces

    def _pass(self) -> list[Stream]:
        """One prompt pass alone; prompts that end sample their first token, draft and join the rounds."""

        pieces = self._pieces()
        self._note_passed(pieces)
        self.pbuf.rotor_lookup = bool(pieces) and self._rotor_lookup(pieces[0][0], len(pieces))
        t0 = time.perf_counter()
        try:
            self._read_ahead(pieces)
            segs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
            ends, cuts = self._end_rows(pieces, segs), self._cuts(pieces, segs)
            logits = compute(self.w, segs, self.pbuf, logits=bool(ends), ends=ends, cuts=cuts)
            heads = logits[:len(ends)].clone() if ends else None
            lasts = self._absorb(pieces, segs, cuts)
        except Exception as exc:                         # noqa: BLE001  (these requests fail, the others go on)
            return self._failed(pieces, exc)
        return self._joined(pieces, heads, lasts, (time.perf_counter() - t0) / len(pieces))

    def _read_ahead(self, pieces) -> None:
        """Overlap the next piece's n-gram reads with this pass."""

        ngram = self.w.cfg.ngram_size - 1
        for s, a, n in pieces:
            if a + n >= len(s.prompt) or s.st.ple_history is None:
                continue
            hist = np.concatenate([np.asarray(s.st.ple_history, dtype=np.int64),
                                   np.asarray(s.prompt[a:a + n], dtype=np.int64)])[-ngram:]
            read_ahead(self.w, hist, s.prompt[a + n:a + n + self.prefill_rows])

    def _note_passed(self, pieces) -> None:
        """Count a pass against every filling prompt it left out; one it took starts over."""

        took = {s.sid for s, _, _ in pieces}
        for s in self.filling:
            self.passed[s.sid] = 0 if s.sid in took else self.passed.get(s.sid, 0) + 1
        for sid in [k for k in self.passed if k not in {s.sid for s in self.filling}]:
            del self.passed[sid]

    @staticmethod
    def _end_rows(pieces, segs) -> list[int]:
        """The pass rows that end a prompt (each gets the head)."""

        return [a1 - 1 for (s, a, n), (_, _, a1) in zip(pieces, segs) if a + n == len(s.prompt)]

    @staticmethod
    def _keep_at(s: Stream) -> int | None:
        """Where a drafting stream's prompt state is kept: one token before its end, which a next turn extends."""

        return entry_end(s.prompt) if s.draft and s.st.image_positions is None else None

    def _points(self, s: Stream, start: int, n: int) -> list[int]:
        """The message-start and prompt-end snapshots in rows ``start`` + 1 .. ``start`` + ``n``, ascending."""

        if not s.draft or s.st.image_positions is not None:
            return []
        return [p for p in (*self.fills[s.sid][0].stops, self._keep_at(s)) if start < p <= start + n]

    def _cuts(self, pieces, segs) -> list[Cut]:
        """The kept points strictly inside the pass's pieces, where their DeltaNet chains split."""

        return [Cut(k - a, at=a0) for (s, a, n), (_, a0, _) in zip(pieces, segs)
                for k in self._points(s, a, n) if k < a + n]

    def _absorb(self, pieces, segs, cuts=()) -> list[torch.Tensor]:
        """After a pass's forward: each prompt's last row and kept point, the MTP head's absorb, the commits."""

        _kv_check(segs)
        lasts = [self.pbuf.streams[a1 - 1:a1].clone() for _, _, a1 in segs]
        at, points = {(cut.at, cut.row): cut for cut in cuts}, []
        for (s, a, n), (st, a0, _) in zip(pieces, segs):      # before the MTP head writes the pass's streams
            mtp = self.fills[s.sid][1]
            for k in self._points(s, a, n):
                row = k - a
                mtp_len = st.mtp_len + row - 1 if mtp else st.mtp_len   # every row but the point's last
                tail = self.pbuf.streams[a0 + row - 1:a0 + row].clone() if mtp else None
                cut = at.get((a0, row))
                snap = None if cut is None else cut_snapshot(self.w, st, self.pbuf, cut, mtp_len)
                points.append((s, mtp_len, tail, snap))
        absorb = [(s.st, s.prompt[a + 1:a + n + 1], self.pbuf.streams[a0:a0 + n])
                  for (s, a, n), (_, a0, _) in zip(pieces, segs) if self.fills[s.sid][1] and a + 1 < len(s.prompt)]
        if absorb:                   # the MTP head absorbs each prompt's rows (its cache in position order)
            absorb = [(st, nxt, streams[:len(nxt)]) for st, nxt, streams in absorb]
            mtp_compute(self.w, mtp_stage(self.pbuf, absorb), self.pbuf, logits=False)
            for st, nxt, _ in absorb:
                st.set_mtp_len(st.mtp_len + len(nxt))
        for (s, a, n), (st, a0, _) in zip(pieces, segs):
            commit(self.w, st, self.pbuf, n, n, at=a0)
        for s, mtp_len, tail, snap in points:            # a point that ends its piece: the state as committed
            self.fills[s.sid][3].append((snap if snap is not None else {**s.st.snapshot(), "mtp_len": mtp_len}, tail))
        return lasts

    def _failed(self, pieces, exc: Exception) -> list[Stream]:
        failed = [s for s, _, _ in pieces]
        for s in failed:
            s.error, s.done = exc, True
            self.filling.remove(s)
            self.fills.pop(s.sid)
            self._drop_kept(s.st)
        return failed                                    # finish() frees their slots

    def _joined(self, pieces, heads, lasts, spent: float) -> list[Stream]:
        """Prompts that ended sample their first token, draft and join the rounds; returns those already done."""

        _kv_check([(s.st, 0, n) for s, _, n in pieces])
        joined, head = [], 0
        for (s, a, n), last in zip(pieces, lasts):
            s.prefill_s += spent
            fill = self.fills[s.sid]
            e, mtp, _, kept, _ = fill
            fill[2], fill[4] = a + n, last if mtp else None
            if s.on_fill is not None:                        # a request that asked for its prefill progress
                s.on_fill(a + n)
            for snap, tail in kept:
                if a < snap["pos"] <= a + n:
                    self._remember(list(s.prompt[:snap["pos"]]), s.st, snap, tail)
            fill[3] = []                                   # only the bounded cache owns a stored snapshot
            if a + n < len(s.prompt):
                continue
            self.filling.remove(s)
            self.fills.pop(s.sid)
            st, e.last_streams = s.st, last
            logits = heads[head:head + 1]
            head += 1
            image_rows.finish(st)
            try:                                             # this request's own policy, grammar and capture
                if s.constraint is not None:                 # a reply's grammar: the first token too
                    logits = s.constraint.mask(logits, None, self.w.meta.get("vocab_offset", 0))
                first = e.sample(logits, [len(s.prompt)], s.sampling)[0]
                if s.probabilities is not None:
                    capture(logits, [first], [len(s.prompt)], s.probabilities)
                if s.constraint is not None:
                    s.constraint.advance([first])
            except Exception as exc:                         # noqa: BLE001  it fails alone; the others go on
                s.error, s.done = exc, True                  # returned, so its reply is the error and finish() frees it
                joined.append(s)
                continue
            s.context = list(s.prompt)
            s.drafts = []
            s.started = time.perf_counter()
            self.streams[s.sid] = s
            s.take([first], self._ends(s))                   # the first token goes out before the next draft
            e.buf.rotor_lookup = self._rotor_lookup(s)
            if e.mbuf is not None:
                e.mbuf.rotor_lookup = e.buf.rotor_lookup
            if mtp and not s.done and s.count > 1:
                s.drafts = draft(e, last, [first], st.pos + 1, min(self.depth, s.count - 1), s.sampling,
                                 self.confidence)
            if s.done:
                joined.append(s)
        return joined

    def cancel(self, streams: list[Stream]) -> list[Stream]:
        """Filling prompts whose clients left stop; each keeps the state its passes committed, so a retry of the same
        prompt resumes there rather than from its last message start. Returns them, done, for ``finish``. A state that
        fails its checks (``snapshot``, ``kv_validate_snapshot``) is not kept: that stream ends with the error and its
        slot keeps nothing, so ``finish`` frees it; the others still stop."""

        for s in streams:
            self.filling.remove(s)
            self.passed.pop(s.sid, None)
            _, mtp, at, _, tail = self.fills.pop(s.sid)
            s.done, s.finished = True, time.perf_counter()
            # only rows a pass committed past the resume point: before any pass the kept entry it resumed (its slot's,
            # with its MTP tail) already holds this state, and remembering it again would displace that entry
            if self._keep_at(s) is not None and max(MIN_GAP, s.cached + 1) <= at < len(s.prompt):
                mtp_len = s.st.mtp_len - 1 if mtp else s.st.mtp_len        # as a point ending a piece keeps it
                try:
                    self._remember(list(s.prompt[:at]), s.st, {**s.st.snapshot(), "mtp_len": mtp_len}, tail)
                except Exception as exc:                 # noqa: BLE001  its reply is the error; its slot holds nothing
                    s.error = exc
                    self._drop_kept(s.st)
        return streams

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (``ignore_eos``)."""

        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """One round over the live streams, with the next prompt pass in the same forward while prompts fill."""

        error = getattr(self, "_memory_error", None)
        if error is not None:
            raise error
        pdl.ON = self.pdl                                # this decoder's launch policy (one process global)
        ended = self._make_room()                      # every stream's caches hold this round, or the newest wait
        live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if self.filling and (not live or not self.converged):
            ended += self._fill()                      # passes alone; a prompt that ends here joins this round
            live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if not live:
            return ended
        grammars, failed = {}, []
        for s in live:                                   # a grammar cuts the drafts no accepted path can hold
            if s.constraint is not None:
                tokens = [s.out[-1]] + list(s.drafts)
                try:
                    grammars[s.sid] = s.constraint.window(tokens, list(range(-1, len(tokens) - 1)))
                except GrammarError as exc:              # this request ends with its error, the others go on
                    s.error, s.done = exc, True
                    self.held.pop(s.sid, None)
                    failed.append(s)
                    continue
                s.drafts = grammars[s.sid].tokens[1:]
        live = [s for s in live if not s.done]
        if not live:
            return failed + ended
        self.buf.rotor_lookup = self._rotor_lookup(live[0], len(live))
        self.pbuf.rotor_lookup = False  # any overlapping prompt invalidates the exclusive-request Region
        t0 = time.perf_counter()
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        # the n-gram rows are read once the forward reaches their layer, behind the layers before it
        segs = stage(self.w, self.buf, windows, late=True)
        # a pass shares the round's forward only where their experts share a launch; else _fill ran it between rounds
        pieces, psegs = (self._pieces(self._pass_rows()) if self.filling and self.converged else []), None
        cuts = []
        if pieces:
            self._note_passed(pieces)
            try:
                self._read_ahead(pieces)
                psegs = stage(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces])
                cuts = self._cuts(pieces, psegs)
            except Exception as exc:                     # noqa: BLE001  (the pass's requests fail, the round goes on)
                ended += self._failed(pieces, exc)
                pieces = []
        held = [self.held.pop(s.sid, []) for s in live]
        tables = self.buf.gdn_tables = gdn_multi.Tables(self.gdn, segs, held, static=self.round_tables)
        self.buf.attn_step = attn_multi.Step(self.w, segs, mtp=False, tables=self.step_tables,
                                             rotor_lookup=self.buf.rotor_lookup)
        try:
            if pieces:                                 # the window and the pass: each layer's experts once for both
                pends = self._end_rows(pieces, psegs)
                logits, heads = compute_mixed(self.w, segs, self.buf, psegs, self.pbuf, ends=pends, cuts=cuts)
                heads = heads[:len(pends)].clone() if pends else None
            else:                                      # a captured forward when one serves this round, else eager
                logits = self.graphs.verify_forward(segs) if self.graphs is not None else None
                if logits is None:
                    logits = compute(self.w, segs, self.buf, check=False)     # the frame's status: checked below
        finally:
            self.buf.gdn_tables = self.buf.attn_step = None
        lasts = self._absorb(pieces, psegs, cuts) if pieces else None
        starts = [a0 for _, a0, _ in segs] + [segs[-1][2]]
        positions = [[st.pos + 1 + r for r in range(a1 - a0)] for st, a0, a1 in segs]
        samplings = [s.sampling for s in live]
        # one host wait for the packets and the frame's status words, latched before any score is used; a failure
        # before that wait reports a failed frame first, as a check right after the forward would
        words = KVWords(segs, self.kv_host)
        try:
            for s, (_, a0, a1) in zip(live, segs):
                if s.sid in grammars:
                    s.constraint.mask(logits[a0:a1], grammars[s.sid])
            launched = launch_streams(logits, starts, positions, samplings)
        except BaseException:
            words.settle()
            raise
        sampled = collect_streams(launched, words)
        paths = [accept(tokens, list(range(-1, len(tokens) - 1)), rows, s.count - len(s.out), self._ends(s))
                 for s, (_, tokens), rows in zip(live, windows, sampled)]
        for s, (_, tokens), (_, a0, _), (path, end), pos in zip(live, windows, segs, paths, positions):
            if s.probabilities is not None:
                capture(logits, [tokens[r] for r in path[1:]] + [end], [pos[r] for r in path],
                        s.probabilities, rows=[a0 + r for r in path])
        for s, rows in zip(live, gdn_multi.keep(tables, [len(path) for path, _ in paths])):
            self.held[s.sid] = rows                      # the next round's trees fold these rows in first
        kept = []
        for s, (_, tokens), (st, a0, a1), rows, (path, end) in zip(live, windows, segs, sampled, paths):
            commit(self.w, st, self.buf, a1 - a0, len(path), at=a0, states=False)
            s.committed.extend(tokens[:len(path)])
            s.counted(len(tokens))
            new = [tokens[r] for r in path[1:]] + [end]
            if s.constraint is not None:
                try:
                    s.constraint.advance(new)
                except GrammarError as exc:
                    s.error = exc
            last = s.error is not None or len(s.out) + len(new) >= s.count or end in self._ends(s)
            kept.append((s, a0, rows[:len(path)], new, last))
        self._draft_all([(s, a0, keep) for s, a0, keep, _, last in kept if s.draft and not last],
                        {s.sid: fresh for s, _, _, fresh, last in kept
                         if getattr(s, "copies", None) is not None and not last})
        for s, _, _, new, _ in kept:
            if s.error is not None:
                s.done = True
                continue
            s.take(new, self._ends(s))
        spent = time.perf_counter() - t0
        self._timed(spent, sum(n for _, _, n in pieces))
        if pieces:                                     # prompts that ended in this round's pass join the next
            ended += self._joined(pieces, heads, lasts, spent / len(pieces))
        done = [s for s in live if s.done]
        for s in done:                                   # a finished stream's state is never read again
            self.held.pop(s.sid, None)
        return failed + done + ended

    def _draft_all(self, streams: list, new: dict | None = None) -> None:
        """Every drafting stream absorbs its kept rows and chains drafts, all streams in one step a depth.
        A stream whose context repeats earlier text drafts that continuation instead of an MTP row. A step's codec
        status words are read back in the next draft readback (``KVWords``); its MTP entries count once they are."""

        for s, _, _ in streams:
            s.drafts = []
        room = {s.sid: min(self.depth, s.count - len(s.out) - len(keep)) for s, _, keep in streams}
        todo = [(s, a0, keep) for s, a0, keep in streams if room[s.sid] > 0 and self.mbuf is not None]
        if not todo:
            return
        for s, _, _ in todo:
            st = s.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
        windows = [(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo]
        segs = mtp_stage(self.mbuf, windows)
        logits = self._mtp(segs)

        def absorbed():                                  # the absorb's entries, once its status words are valid
            for (s, _, keep), (st, a0, a1) in zip(todo, segs):
                st.set_mtp_len(st.mtp_len + len(keep))

        words = KVWords(segs, self.kv_host, absorbed)    # its status words join the first draft's readback
        active = [(s, a1 - 1) for s, (_, _, a1) in zip([t[0] for t in todo], segs)]
        if new:
            try:                                         # a failure here still reports a failed absorb first
                for s, _ in active:
                    if s.sid in new and getattr(s, "copies", None) is not None:
                        s.context.extend(new[s.sid])
                        s.drafts = s.copies.propose(s.context, max(room[s.sid], COPY_MATCH))[:room[s.sid]]
                        del s.context[len(s.context) - len(new[s.sid]):]
                keep_rows = [i for i, (s, _) in enumerate(active) if not s.drafts]
                if len(keep_rows) != len(active):
                    logits = logits[keep_rows]
                active = [active[i] for i in keep_rows]
            except BaseException:
                words.settle()
                raise
            if not active:                               # every stream copies: no draft readback checks the absorb
                words.settle()
                return
        for j in range(self.depth):
            picks = self._picks(logits, [s.st.pos + 1 + j for s, _ in active], [s.sampling for s, _ in active], words)
            nxt = []
            for (s, row), (d, p) in zip(active, picks):
                low = self.confidence > 0 and p < self.confidence
                if low and j > 0:
                    continue
                s.drafts.append(d)
                if not low and j + 1 < room[s.sid]:
                    nxt.append((s, row, d))
            if not nxt:
                return
            windows = [(s.st, [d], self.mbuf.streams[row:row + 1]) for s, row, d in nxt]
            segs = mtp_stage(self.mbuf, windows)
            logits = self._mtp(segs)

            def chained(nxt=nxt):                        # the step's entries, once its status words are valid
                for s, _, _ in nxt:
                    s.st.set_mtp_len(s.st.mtp_len + 1)
                    s.st.mtp_drafted += 1

            words = KVWords(segs, self.kv_host, chained)  # the next pick's readback checks them (a step has one)
            active = [(s, a0) for (s, _, _), (_, a0, _) in zip(nxt, segs)]

    def _mtp(self, segs: list) -> torch.Tensor:
        """An MTP step over every drafting stream, its attention one launch a kernel for all of them."""

        self.mbuf.rotor_lookup = False
        if self.live() == 1:
            request = next(iter(self.streams.values())) if self.streams else self.filling[0]
            self.mbuf.rotor_lookup = self._rotor_lookup(request, len(segs))
        self.mbuf.attn_step = attn_multi.Step(self.w, segs, mtp=True, tables=self.mtp_tables,
                                              rotor_lookup=self.mbuf.rotor_lookup)
        try:                                             # the caller checks the step's status
            out = self.graphs.mtp_step(segs) if self.graphs is not None else None
            return mtp_compute(self.w, segs, self.mbuf, check=False) if out is None else out
        finally:
            self.mbuf.attn_step = None

    def _picks(self, logits: torch.Tensor, positions: list[int], samplings: list,
               words: KVWords) -> list[tuple[int, float]]:
        """Batch drafts and original confidence; greedy keeps the first physical FP32 maximum.

        Positive keyed policies use global-ID ties. Borrow full rows only to
        repair truncation or apply the complete-vocabulary top-k-off rule.

        ``words``: the producing step's codec status words, read back in this readback's wait and latched before
        anything here is validated or used (a failure before that wait checks them first).
        """

        try:
            if len(samplings) != len(positions):
                raise ValueError("one draft policy and absolute position per source row required")
            validate_rows(logits, positions, None)
            for i, sampling in enumerate(samplings):
                validate_policy(positions[i:i + 1], sampling)
            if self._draft_map is not None:
                self._draft_map.ids(logits.shape[1])
        except BaseException:
            words.settle()
            raise
        if not positions:
            words.settle()
            return []
        try:                                             # a failure before the words' check settles them first
            row = logits.float()
            # Greedy proposals only consume the original maximum and log-sum-exp.
            # k=0 retains their common read-back offsets without unused top-k work.
            k, candidates = 0, []
            if not all(s is None or s.temperature <= 0 for s in samplings):
                k = max([int(s.top_k) + MARGIN for s in samplings
                         if s is not None and s.temperature > 0 and s.top_k] or [1])
                k = min(k, row.shape[1])
                vals, idx = torch.topk(row, k, dim=-1, sorted=False)
                candidates = [vals.contiguous().view(torch.int32).to(torch.int64), idx]
            top, col = row.max(dim=-1, keepdim=True)
            lse = torch.logsumexp(row, dim=-1, keepdim=True)
            got = torch.cat([*candidates, top.view(torch.int32).to(torch.int64), col,
                             lse.view(torch.int32).to(torch.int64)], dim=1)
            words.queue()
            got = got.to("cpu", non_blocking=True)
            torch.cuda.current_stream(logits.device).synchronize()
        except BaseException:
            words.settle()
            raise
        words.check()
        got = got.numpy()
        normalizers = got[:, 2 * k + 2].astype(np.int32).view(np.float32)
        if not np.isfinite(normalizers).all():
            raise ValueError("FP32 draft confidence normalization is nonfinite")
        out = []
        for i, (pos, smp) in enumerate(zip(positions, samplings)):
            g = got[i]
            lse_i = float(normalizers[i])
            if smp is None or smp.temperature <= 0:
                c = int(g[2 * k + 1])
                out.append((int(self.draft_host[c]) if self.draft_host is not None else c,
                            float(np.exp(float(g[2 * k:2 * k + 1].astype(np.int32).view(np.float32)[0]) - lse_i))))
                continue
            values = g[:k].astype(np.int32).view(np.float32)
            cols = g[k:2 * k]
            ids = self.draft_host[cols] if self.draft_host is not None else cols
            complete = None
            complete_ids = self.draft_host

            def full_row(_):
                nonlocal complete, complete_ids
                if complete is None:
                    complete = row[i].cpu().numpy()
                    if complete_ids is None:
                        complete_ids = np.arange(row.shape[1], dtype=np.int64)
                return complete, complete_ids

            if smp.top_k:
                tok = repaired_choose(values[None], ids[None], [pos], smp, row.shape[1], full_row)[0]
            else:
                full_values, full_ids = full_row(0)
                tok = choose_rows(full_values[None], full_ids[None], np.asarray([pos], dtype=np.uint64), smp)[0]
            hit = np.nonzero(ids == tok)[0]
            value = values[hit[0]] if len(hit) else complete[int(np.nonzero(complete_ids == tok)[0][0])]
            probability = float(np.exp(float(value) - lse_i))
            if not np.isfinite(probability):
                raise ValueError("FP32 draft confidence produced a nonfinite probability")
            out.append((int(tok), probability))
        return out

    def finish(self, done: list[Stream]) -> None:
        """Drop finished streams; a slot whose prompt state is kept stays with it, the rest are free again."""

        for s in done:
            self.streams.pop(s.sid, None)
            if not any(k[1] is s.st for k in self.kept) and all(f is not s.st for f in self.free):
                self._shrink(s.st)
                self.free.append(s.st)

    def drop(self) -> list[Stream]:
        prior, errors = tuple(self.failed_slots), []
        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.filling, self.fills = [], {}
        for s in live:
            self.streams.pop(s.sid, None)
            self.held.pop(s.sid, None)
            self._drop_kept(s.st)
            try:
                self._shrink(s.st)
                self.free.append(s.st)
            except BaseException as error:  # noqa: BLE001 - retire owned storage even on cancellation
                errors.append(error)
        for st in prior:
            try:
                self._retire_failed_slot(st)
            except BaseException as error:  # noqa: BLE001 - retire owned storage even on cancellation
                errors.append(error)
        raise_failures(None, errors)
        return live


def _tensors(st: State):
    for value in vars(st).values():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, torch.Tensor):
                yield v
            elif hasattr(v, "__dict__"):                  # scratch and KV cache objects, the MTP head's too
                yield from (t for t in vars(v).values() if isinstance(t, torch.Tensor))

"""Flash Next's concurrent rounds on one GPU: every stream keeps exactly its own accepted prefix."""

from __future__ import annotations

import os
import time

import numpy as np
import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda import pdl
from tensorfold.cuda.capacity import LIMIT_ENV, available_bytes, cuda_limit_bytes, gather_ints
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
from tensorfold.cuda.streams import Stream, accept, prefix_limit
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import GrammarError

from ..cuda import CONFIDENCE, DEPTH
from . import attn_multi, gdn_multi, image_rows, multi_graphs, prefixes
from .decode import (
    PREFILL_ROWS, WARM_TAIL, Engine, SamplingShard, _gathered_fits, choose_gathered,
    draft, entry_end, prefill_begin, tp_sample_rows,
)
from .forward import Cut, KVWords, _kv_check, commit, compute, compute_mixed, converges, cut_snapshot, read_ahead, stage
from .mtp import mtp_compute, mtp_stage
from .prompt_plan import CUTS, POINTS, pass_limit
from .service_memory import automatic_graph_bytes, cache_resize_overlap, context_reservation, future_service, materialized_graph_room
from .speculation import MethodGeometry
from .state import ENDS, Buffers, KVNumericError, State

FIRST, STEP = 256, 8192          # rows an idle slot keeps; rows a stream's caches grow by at a time
GIB = 1024**3
SHARE = 0.0                      # --decode-share: a round alone takes this share of its pass's time (0: whole passes)
PASS_MIN = 128                   # the fewest prompt rows a round's pass takes
COPY_MATCH = 8                   # copy drafts: the context's last this many tokens seen before, and as many after them
FILL_GUARD = 8                   # a prompt passed over this many passes takes the next one (no starvation), as on Macs


def _slot(w, st: State, buf: Buffers, mbuf: Buffers, pbuf: Buffers, capacity: int, prefill_rows: int, *,
          logical_context: int | None = None) -> Engine:
    """A one-sequence engine over a slot's state and the shared buffers (eager: no CUDA graphs)."""

    e = object.__new__(Engine)
    e.w, e.capacity, e.rows, e.prefill_rows = w, capacity, buf.rows, prefill_rows
    e.logical_context = logical_context
    e.buf, e.mbuf, e.pbuf, e.st, e.graphs = buf, mbuf, pbuf, st, None
    e.kv_pair = st.kv_pair
    e._sample_plan = getattr(buf, "sample_plan", None)
    e._draft_plan = getattr(mbuf, "sample_plan", None) if mbuf is not None else None
    e._draft_map = None
    e.stops = ()
    return e


class MultiDecoder:
    """Rounds over the live streams; ``slots`` streams at most, each with ``capacity`` tokens of context."""

    def __init__(self, w, *, slots: int, capacity: int, depth: int = DEPTH, confidence: float = CONFIDENCE,
                 stop_eos: bool = True, keep: int = 8, kv_dtype: str = "bf16", prefill_rows: int = PREFILL_ROWS,
                 share: float = SHARE, points=None, vision=None, workspace_bytes: int = 0, kv_pair=None,
                 kv_key_dtype: str | None = None, kv_value_dtype: str | None = None, graphs: bool = False,
                 graph_budget=..., graph_header_room: int | None = None, vision_workspace_bytes: int = 0,
                 full_windows: int | None = None, execution_rows: int | None = None, memory_live=None) -> None:
        if w.comm is not None and w.meta.get("execution_provider") != "sm70":
            raise ValueError("concurrent tensor-parallel Flash Next requires the SM70 execution provider")
        budget = (multi_graphs.budget_bytes() if graphs else 0) if graph_budget is ... else graph_budget
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError("captured-round budget must be nonnegative integer bytes or automatic")
        if full_windows is not None and (type(full_windows) is not int or full_windows < 1):
            raise ValueError("full KV windows must be a positive integer")
        if memory_live is not None and not callable(memory_live):
            raise ValueError("prepared memory admission must provide a callable live-byte reader")
        self.full_windows = full_windows
        self.w, self.depth, self.confidence, self.capacity = w, depth, confidence, capacity
        from .rotor_decoder import RotorDecoderPolicy
        self.rotor_decoder_policy = RotorDecoderPolicy()
        self.points = points                         # a prompt's message starts to keep states at, or None
        self.vision = vision
        self.copy = os.environ.get("TENSORFOLD_MTP_COPY", "0") == "1" and getattr(w, "dflash2", None) is None
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows = slots * (depth + 1) if execution_rows is None else execution_rows
        if type(rows) is not int or not depth + 1 <= rows <= slots * (depth + 1):
            raise ValueError("execution_rows must hold one complete verification window within the resident envelope")
        self.controller = None
        self.execution_manifest_installer = None
        self.poison_control = None
        self._execution_plan = None
        self._round_migration_s = 0.0
        if w.meta.get("execution_provider") == "sm70":
            from .adaptive_controller import AdaptiveController

            self.controller = AdaptiveController(slots=slots, row_budget=rows, proposal_limit=depth,
                                                 prefill_tokens=prefill_rows)
            self.controller.work_observations.configure_indexer(
                ratio=w.cfg.index_ratio, budget=w.cfg.index_budget, heads=w.cfg.index_heads,
                target_caches=sum(not layer.linear for layer in w.layers))
        # a round's window and a prompt pass share each layer's expert launch: the pass's buffers hold both
        self.converged, self.prefill_rows = converges(w), prefill_rows
        # rounds beside a filling prompt size its pass so decoding keeps ``share`` of the pass's time (0: whole passes)
        self.share, self.round_s, self.row_s = share, None, None
        self.buf = Buffers(w, rows, capacity, moe_prefill=True)
        self.mbuf = Buffers(w, rows, capacity, owner_role="mtp") if w.mtp is not None else None
        self.speculation = MethodGeometry("mtp" if self.mbuf is not None else
                                          "dspark" if getattr(w, "dspark", None) is not None else
                                          "dflash2" if getattr(w, "dflash2", None) is not None else "none",
                                          depth)
        self.pbuf = Buffers(w, prefill_rows + (rows if self.converged else 0), capacity, prefill=True)
        self._sample_plan = self._draft_plan = None
        if w.comm is not None:
            self._sample_plan = self.buf.sample_plan = SamplingShard(w, w.head.n, offset=int(w.meta["vocab_offset"]))
            if self.mbuf is not None:
                width = w.draft_head.n if w.draft_head is not None else w.head.n
                self._draft_plan = self.mbuf.sample_plan = SamplingShard(
                    w, width, offset=int(w.meta["vocab_offset"]), id_map=w.draft_ids)
        from .hc_plans import enroll

        enroll(w, self.buf)
        enroll(w, self.mbuf, mtp=True)
        self.gdn = gdn_multi.Scratch(w, rows)            # every stream's DeltaNet rows, one launch a step
        # a frame's codec status words, read back with its sampling packet (KVWords): one host wait a step
        self.kv_host = torch.zeros((slots, 2), dtype=torch.int32, pin_memory=torch.cuda.is_available())
        self.kv_cohort = None
        if w.comm is not None and w.meta.get("execution_provider") == "sm70":
            from .kv_status import KVStatusCohort

            self.kv_cohort = KVStatusCohort(slots, w.comm, w.device)
        # every step's tables at fixed addresses and its kernels at a fixed geometry, so captured forwards replay
        self.round_tables = gdn_multi.RoundTables(w, self.gdn, slots, depth + 1)
        self.step_tables = attn_multi.StepTables(w, slots, depth + 1, mtp=False)
        self.mtp_tables = attn_multi.StepTables(w, slots, depth + 1, mtp=True) if w.mtp is not None else None
        # captured rounds (multi_graphs; ``capture_graphs`` at warm-up): None until then, or when not captured
        self.graphs: multi_graphs.RoundGraphs | None = None
        self._failed_graphs: multi_graphs.RoundGraphs | None = None
        reason = None if graphs else "not requested"
        if graphs and budget == 0:
            reason = f"{multi_graphs.ENV}=0"
        elif graphs and getattr(w, "dspark", None) is not None and not getattr(w.dspark, "target_capturable", False):
            reason = "DSpark target feature capture runs eagerly"
        elif graphs and getattr(w, "dflash2", None) is not None:
            reason = "DFlash2 target feature capture runs eagerly"
        elif graphs:
            reason = multi_graphs.capturable(w)
        self.graph_request = budget
        self.graph_budget = (budget or 0) if reason is None else 0
        self.graph_header_room = graph_header_room
        self.graph_floor = FIRST
        self.graph_floor_extra_bytes = 0
        self.vision_workspace_bytes = vision_workspace_bytes
        self.graph_record: dict = {"captured": False, "reason": reason}
        # programmatic dependent launch of the decode kernels: this chip's policy, once (pdl.ON is set each round)
        self.pdl = pdl.select(w.device.index or 0) if torch.cuda.is_available() else False
        pdl.ON = self.pdl
        self.held: dict[int, list[int]] = {}             # stream id -> last round's kept rows, folded in next round
        # A full-window promise owns every selected cache before startup returns.
        # Legacy and explicitly shared modes keep the original growing floors.
        self.full_resident = full_windows is not None and full_windows >= slots
        initial_capacity = capacity if self.full_resident else min(capacity, FIRST)
        self.free = [State(w, initial_capacity, depth + 1, kv_dtype, limit=capacity, kv_pair=kv_pair,
                           kv_key_dtype=kv_key_dtype, kv_value_dtype=kv_value_dtype, slot=slot)
                     for slot in range(slots)]
        self.states = tuple(self.free)
        self._migration_cursor = 0
        if (self.controller is not None and self.free[0].history_placement is not None
                and self.free[0].history_placement.target_layers % 4 == 0
                and self.free[0].kc[0].arena is not None
                and all(cache.arena is self.free[0].kc[0].arena for cache in self.free[0].kc)):
            self.controller.enable_migration(self.free[0].history_placement.degree)
        self.failed_slots: list[State] = []          # bounded by slots; exclusive owners until retirement succeeds
        self.slot_bytes = sum(t.numel() * t.element_size() for t in _tensors(self.free[0]))
        self.window_bytes = self.free[0].cache_bytes(capacity)          # one stream's caches at the full window
        free = memory_live if memory_live is not None else (
            torch_live(torch, available_bytes) if torch.cuda.is_available() else None)
        # the mapped n-gram tables are not held back (they barely fit on a Spark); lookups page from disk instead
        live = free
        reserve = max(2 * GIB, workspace_bytes)
        self.dflash_resources = None
        if getattr(w, "dflash2", None) is not None:
            self.dflash_resources = w.dflash2.plan.serving_resources(
                slots=slots, context=w.dflash2.context, prefill_rows=self.pbuf.rows, execution_rows=rows,
                keep=keep, points=POINTS, target_vocab=w.cfg.vocab)
            resource = self.dflash_resources
            # Physical free already excludes loaded weights, per-slot rings
            # and both feature frames. Protect future snapshots/temporary
            # work, including full target p extraction, exactly once here.
            reserve += (resource["snapshot_bytes"] + resource["execution_workspace_bytes"]
                        - resource["feature_frame_bytes"] + resource["target_probability_device_bytes"])
        room = live() if live is not None else 1 << 62
        already_held = 0
        if full_windows is not None:
            promise = context_reservation(self.free, full_windows, preserve_overlap=self.full_resident)
            required = promise["growth_bytes"] + promise["copy_overlap_bytes"] + reserve
            if room < required:
                raise RuntimeError("admitted full KV windows and workspace no longer fit available device memory")
            already_held = sum(st.cache_bytes() - st.cache_bytes(min(FIRST, st.limit)) for st in self.free)
            room = already_held + required
        self.memory_gate = MemoryGate(room, reserve=reserve, live=live,
                                      allow_lone_growth=full_windows is None)
        self.memory_gate.held = already_held
        if w.comm is not None:
            from .tp_scheduler import SharedMemoryGate

            self.memory_gate = SharedMemoryGate(self.memory_gate, w.comm)
        self._memory_error: BaseException | None = None
        # All slots share this validated immutable descriptor. Prefix reuse is
        # qualified by numerical format as well as the request's sampler.
        self.kv_pair = self.free[0].kv_pair
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

    def _cache_overlap(self, st, rows):
        return cache_resize_overlap(st, rows, auxiliary=getattr(self, "full_windows", None) is not None)

    def _grow(self, st: State, rows: int, *, alone: bool = False, protect: State | None = None) -> bool:
        """Grow caches while the gate has room; one stream may use its startup allowance within an explicit cap."""

        if rows <= st.capacity or st.capacity >= st.limit:       # admission's count keeps a stream within its window
            return True
        size = min(st.limit, -(-rows // STEP) * STEP)
        before = st.cache_bytes()
        grow = st.cache_bytes(size) - before
        while not self.memory_gate.fits(grow + self._cache_overlap(st, size)):     # a layer's old buffers stay until its copy
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
                        peak = int(torch.cuda.memory_allocated()) + grow + self._cache_overlap(st, st.capacity)
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
            if self.graphs is not None:
                self.graphs.invalidate_state(st)
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
        """Reset an idle slot; full resident windows persist unless forced/error retirement releases them."""

        gate, held_before = self.memory_gate, self.memory_gate.held
        before, primary, errors = st.cache_bytes(), None, []
        floor = (st.limit if getattr(self, "full_resident", False) else getattr(self, "graph_floor", FIRST))
        target = min(FIRST if force or st._resize_error is not None else floor, st.limit)
        try:
            if self.graphs is not None and (force or st._resize_error is not None or st.capacity > target):
                self.graphs.invalidate_state(st)
            if st._resize_error is not None:
                st.recover_resize(self.w, min(FIRST, st.limit))
            else:
                st.reset(self.w)
                if force or st.capacity > target:
                    st.resize(target)
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
        if getattr(self, "full_resident", False) and not self._grow(st, st.limit, alone=True):
            raise NoRoom("retired slot cannot restore its promised full context window")
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
        if "media_state" in snap:
            return  # cross-request media borrowing remains unqualified; recovery snapshots stay valid
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
        if self.graph_record["reason"] is None and self.graphs is None:
            self.capture_graphs()
        if self.controller is not None:
            self.controller.reset_observations()

    @torch.no_grad()
    def _prepare_graph_floor(self):
        """Admit persistent bound slots without counting one promised full window twice."""
        floor = min(8192, self.capacity)
        growth = [max(0, state.cache_bytes(floor) - state.cache_bytes()) for state in self.free]
        promised = min(len(growth), getattr(self, "full_windows", None) or 1)
        extra = sum(growth) - sum(sorted(growth, reverse=True)[:promised])
        if self.graph_header_room is not None and extra > self.graph_header_room:
            self.graph_budget = 0
            self.graph_record = {"captured": False, "reason": "bound graph cache floors exceed admitted header room",
                                 "floor_tokens": floor, "floor_extra_bytes": extra, "budget_bytes": 0}
            return False
        self.graph_floor = floor
        try:
            for state in self.free:
                if not self._grow(state, floor):
                    raise NoRoom("bound graph cache floors cannot retain the promised full window")
        except BaseException:
            self.graph_floor = FIRST
            for state in self.free:
                self._shrink(state, force=True)
            raise
        self.graph_floor_extra_bytes = extra
        if self.graph_header_room is not None:
            self.graph_header_room -= extra
        return True

    @torch.no_grad()
    def capture_graphs(self) -> None:
        """Capture the rounds' verify forwards and MTP steps on the idle slots (``multi_graphs``) and take their
        measured memory (``room_bytes``) from the stream caches' room. Before any request: every slot must be free."""

        if self.streams or self.filling or self.kept or len(self.free) != self.round_tables.slots:
            raise RuntimeError("graphs are captured on idle slots, before any request")
        if self.graphs is not None or self._failed_graphs is not None:
            raise RuntimeError("captured-round pools are immutable until this decoder retires")
        if self.graph_record["reason"] is not None:
            return
        if getattr(self.w, "meta", {}).get("execution_provider") == "sm70" and not self._prepare_graph_floor():
            return
        multi_graphs.prepare_capture()
        slots = len(self.free)
        # The legacy resident policy promises one full native window. Explicit
        # full windows also protect every selected slot's target/draft caches.
        promise = context_reservation(self.free, getattr(self, "full_windows", None),
                                      preserve_overlap=getattr(self, "full_resident", False))
        growth = promise["growth_bytes"]
        future = future_service(self.w, self.free, streams=slots, retained=self.keep + slots * POINTS,
                                growth_bytes=growth, vision_workspace_bytes=self.vision_workspace_bytes,
                                minimum_workspace_bytes=self.memory_gate.reserve,
                                copy_overlap_bytes=promise["copy_overlap_bytes"])
        live = multi_graphs.live_bytes
        automatic = self.graph_request is None
        observed = (automatic and getattr(self, "full_resident", False)
                    and getattr(self.w, "meta", {}).get("execution_provider") == "sm70"
                    and getattr(self.w, "meta", {}).get("indexer_history") == "ram")
        refinement = None
        if observed:
            initial_live, error = 0, None
            packet = [0, 0, 0, future["future_bytes"]]
            try:
                initial_live = live()
                room = materialized_graph_room(self.free, self.capacity, initial_live, promise, future)
                packet = [1, room, initial_live, future["future_bytes"]]
            except Exception as failure:
                error = failure
            world = int(self.w.meta.get("world", 1))
            observations = (gather_ints(torch, self.w.comm.all_gather, packet, world=world)
                            if self.w.comm is not None else [packet])
            if (len(observations) != world or any(len(row) != 4 or any(type(value) is not int for value in row)
                                                  for row in observations)):
                if error is not None:
                    raise error
                raise RuntimeError("invalid all-rank materialized graph observation packet")
            if any(row[0] != 1 for row in observations):
                if error is not None:
                    raise error
                raise RuntimeError("a peer could not prove observed full-window graph room")
            if any(row[1] < 0 or row[3] < 2 * GIB or row[2] < row[3] or row[1] != row[2] - row[3]
                   for row in observations):
                raise RuntimeError("inconsistent all-rank materialized graph room proof")
            self.graph_budget = min(row[1] for row in observations)
            refinement = {"policy": "ram-full-resident-observed-v1", "agreed_graph_bytes": self.graph_budget,
                "observations": [{"rank": rank, "graph_room_bytes": row[1], "live_after_floor_bytes": row[2],
                                  "future_service_bytes": row[3]} for rank, row in enumerate(observations)],
                "full_slots": slots, "capacity": self.capacity, "future_service": dict(future),
                "header_room_bytes": self.graph_header_room, "private_spare_credit_bytes": 0}
        else:
            initial_live = live()
        if automatic and not observed:
            self.graph_budget = automatic_graph_bytes(initial_live, future["future_bytes"], self.graph_header_room)
        resolution = {"automatic": automatic, "initial_live_bytes": initial_live,
                      "header_room_bytes": self.graph_header_room, "future_service": future}
        if refinement is not None:
            resolution["observed_graph_admission"] = refinement
        if getattr(self, "full_resident", False):
            resolution["materialized_full_windows"] = {
                "slots": slots, "capacities": [st.capacity for st in self.free],
                "cache_bytes": sum(st.cache_bytes() for st in self.free)}
        if getattr(self.w, "meta", {}).get("execution_provider") == "sm70":
            resolution.update(floor_tokens=self.graph_floor, floor_extra_bytes=self.graph_floor_extra_bytes)
        if not self.graph_budget:
            if getattr(self, "graph_floor_extra_bytes", 0):
                released = self.graph_floor_extra_bytes
                self.graph_floor, self.graph_floor_extra_bytes = FIRST, 0
                for state in self.free:
                    self._shrink(state, force=True)
                if self.graph_header_room is not None:
                    self.graph_header_room += released
                resolution.update(floor_tokens=FIRST, floor_extra_bytes=0, declined_floor_extra_bytes=released)
            self.graph_record = {"captured": False, "reason": "automatic graph budget has no room after required service",
                                 "budget_bytes": 0, **resolution}
            return
        graphs = multi_graphs.RoundGraphs(self, self.graph_budget, spare=lambda: live() - future["future_bytes"])
        cleanup_errors = [None]  # preserve a capture primary even if failure accounting/publication also fails
        try:
            graphs.capture(prepared=True)  # no allocations into the graph pool since prepare_capture above
        except BaseException as primary:
            self._failed_graphs = graphs
            graphs.retain_failure(primary)
            if graphs.admitted:
                try:
                    self._publish_graphs(graphs, resolution, captured=False)
                except BaseException as cleanup:
                    self._memory_error = primary
                    cleanup_errors[0] = cleanup
                    raise_failures(primary, cleanup_errors)
            else:
                # A failed drain/reset/accounting leaves no admissible eager
                # continuation. Its owner remains explicit until retirement.
                self._memory_error = primary
            raise
        self._publish_graphs(graphs, resolution, captured=True)

    def _publish_graphs(self, graphs, resolution, *, captured):
        """Prepare fallible metadata before publishing one accounted pool; interrupted publication poisons reuse."""
        self._failed_graphs = graphs
        try:
            live = multi_graphs.service_live_bytes(graphs)
            record = {"captured": captured,
                      "reason": None if captured else "capture failed; retained pool accounted, decoder remains eager",
                      **resolution, **graphs.record()}
            # The explicit gate already contains only promised cache/workspace
            # bytes; graph capture protected that promise instead of borrowing it.
            if (getattr(self, "full_windows", None) is not None and
                    self.memory_gate.held + live() < self.memory_gate.room):
                raise RuntimeError("captured graph pool consumed the promised full KV windows or workspace")
            room = (self.memory_gate.room if getattr(self, "full_windows", None) is not None else
                    self.memory_gate.room - graphs.room_bytes)
            self.memory_gate.live = live
            self.memory_gate.room = room
            self.graph_record = record
            if captured:
                self.graphs = graphs
                self._failed_graphs = None
        except BaseException as error:
            self._memory_error = error
            raise

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
        if getattr(self.w, "dflash2", None) is not None and len(s.owed) > s.count:
            raise ValueError("committed replay exceeds the current context/reply allowance")
        if any(x.waiting for x in self.streams.values()):
            raise NoRoom("streams already wait for memory; a new request waits until one finishes")
        t0 = time.perf_counter()
        from .prefix_policy import reuse_allowed

        reuse = reuse_allowed(self.w, s.sampling, draft=s.draft, vision=s.vision,
                              position=len(s.prompt), kv_pair=self.kv_pair)
        st, resume, s.cached = self._slot_for(list(s.prompt), reuse)
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
        e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity, self.prefill_rows,
                  logical_context=getattr(self, "logical_context", None))
        e._draft_map = self._draft_map
        mtp = s.draft and self.depth > 0 and self.mbuf is not None
        try:
            begin = prefill_begin(e, s.prompt, mtp=mtp, resume=resume, new_sequence=False)
            st.media_mtp_enabled = mtp
            if getattr(self.w, "dflash2", None) is not None:
                self.w.dflash2.set_enabled(st, s.draft)
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
        if self.controller is not None:
            observation = self.controller.work_observations
            try:
                if mtp and resume is not None and resume.get("tail") is not None:
                    observation.admission(s, prefix_mtp_position=resume["state"]["mtp_len"],
                                          zero_proposals=self._proposal_limit() == 0)
                else:
                    observation.admission(s)
            except BaseException as primary:
                observation.failed_observation = True
                try:
                    self._quarantine(st)
                    self._retire_failed_slot(st)
                except BaseException as cleanup:
                    raise_failures(primary, [cleanup])
                raise
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
            if self.controller is not None:
                break  # Every bounded adaptive pass returns to the globally frozen control boundary.
            if any(not x.done and not x.waiting for x in self.streams.values()) or self.arrived() or \
                    any(s.cancel is not None and s.cancel.is_set() for s in self.filling):
                break
        return ended

    def _pass_rows(self) -> int:
        """A round's prompt rows: its decode (a round alone) takes ``share`` of the pass's time, by the last rounds."""

        return self.pass_rows_for(None if self.controller is None else self.controller.effective)

    def pass_rows_for(self, settings):
        """The prefill quota of a frozen operation, including a newly selected profile."""

        live = any(not s.done for s in self.streams.values())
        limit = pass_limit(self.prefill_rows, live, self.share, self.round_s, self.row_s, PASS_MIN)
        if settings is not None:
            limit = min(limit, settings["prefill_tokens"])
        return limit

    def freeze_execution(self):
        if self.controller is None:
            return None
        active = [s for s in (*self.streams.values(), *self.filling) if not s.done]
        sessions = sorted({s.sid for s in active})
        lengths = [s.st.pos for s in active if s.st is not None]
        workload = {"active_requests": len(sessions),
                    "decode_requests": sum(not s.done and not s.waiting for s in self.streams.values()),
                    "prefill_requests": sum(not s.done for s in self.filling),
                    "max_history_tokens": max(lengths, default=0), "history_tokens": sum(lengths),
                    "retained_sessions": 0,
                    "pending_prompt_tokens": sum(len(s.prompt) - self.fills[s.sid][2]
                                                 for s in self.filling if not s.done),
                    "pending_arrivals": bool(self.arrived()), "indexer_work": None, "expert_group_max": None}
        return self.controller.freeze(sessions, workload=workload)

    def accept_execution(self, plan):
        if self.controller is None:
            if plan is not None:
                raise ValueError("this decoder has no adaptive execution controller")
            return
        self.controller.accept(plan)
        self.apply_execution_family()
        self._execution_plan = plan

    def apply_execution_family(self):
        if self.controller is None:
            return
        family = self.controller.effective_family
        for buffers in (self.buf, self.mbuf, self.pbuf):
            if buffers is not None:
                buffers.owner_kernel_family = family["kernel_family"]
                buffers.owner_graph_family = family["graph_family"]

    def _proposal_limit(self):
        return self.depth if self.controller is None else self.controller.effective["proposal_limit"]

    def _proposal_count(self, stream, remaining):
        logical = getattr(self, "logical_context", self.capacity - self.depth - 1)
        return self.speculation.cap_proposals(
            self._proposal_limit(), reply_tokens=max(0, remaining),
            context_tokens=max(0, logical - stream.st.pos - 1), target_rows=self.buf.rows)

    @staticmethod
    def _output_marks(streams):
        return [(stream, len(stream.out), len(stream.owed)) for stream in streams]

    @staticmethod
    def _useful_outputs(marks):
        # Stream.take consumes owed background replay before invoking emit.
        # A failed grammar advance or replay mismatch contributes no output.
        return sum(max(0, len(stream.out) - before - owed) for stream, before, owed in marks)

    def _prefix_limits(self, streams, news):
        active = any(s.token_limit is not None or getattr(s, "_tp_token_limit", False) for s in streams)
        if not active:
            return [(len(new), False) for new in news]
        decisions, error = None, None
        if self.w.comm is None or self.w.comm.rank == 0:
            try:
                decisions = [prefix_limit(s.token_limit, new, s.out) for s, new in zip(streams, news)]
            except BaseException as caught:
                error = caught
        if self.w.comm is not None:
            decisions = self.prefix_decisions(decisions, error)
        elif error is not None:
            raise error
        if len(decisions) != len(streams):
            raise ValueError("stop-boundary decision count differs from the planned microbatch")
        if self.w.comm is None or self.w.comm.rank == 0:
            return decisions
        # Revalidate received metadata before it can govern target/draft state.
        return [prefix_limit(lambda _new, _out, decision=decision: tuple(decision), new, s.out)
                for s, new, decision in zip(streams, news, decisions)]

    def history_record(self):
        records = [state.history_record() for state in self.states]
        active = self._busy()
        cached = {id(point[1]) for point in self.kept}
        for state, record in zip(self.states, records):
            record["request_active"] = id(state) in active
            record["derived_prefix_cache"] = id(state) in cached
            if getattr(getattr(self, "w", None), "dflash2", None) is not None:
                record["dflash2"] = self.w.dflash2.record(state)
        return {"slots": records, "active_request_slots": len(active),
                "intentional_retained_sessions": 0, "derived_prefix_slots": len(cached),
                "derived_prefix_entries": len(self.kept),
                "migration": None if self.controller is None else
                self.controller.record()["migration"]}

    def install_execution_manifest(self, relative_name):
        if self.execution_manifest_installer is None:
            raise ValueError("execution-profile manifest verification was not enabled at startup")
        return self.execution_manifest_installer(relative_name)

    def maintenance_pending(self):
        migration = None if self.controller is None else self.controller.migration
        return migration is not None and migration["pending"] and migration["blocked"] is None

    def _migrate_boundary(self, migration):
        if self.controller is None or migration is None or not migration["pending"]:
            return
        started = time.perf_counter()
        try:
            self._migration_bundle(migration)
        except BaseException as error:
            self._memory_error = error
            poison = getattr(self, "poison_control", None)
            if poison is not None:
                poison(error)
            raise
        finally:
            self._round_migration_s = getattr(self, "_round_migration_s", 0.0) + time.perf_counter() - started

    def _migration_bundle(self, migration):
        if self.controller is None or migration is None or not migration["pending"]:
            return
        target, generation = migration["requested_degree"], migration["request_epoch"]
        for offset in range(len(self.states)):
            index = (self._migration_cursor + offset) % len(self.states)
            state = self.states[index]
            pending = getattr(state, "_history_migration", None)
            if state.history_placement.degree == target and pending is None:
                continue
            for stream in self.streams.values():
                if stream.st is state:
                    self._flush(stream)
            if state.mtp_drafted:
                state.set_mtp_len(state.mtp_len - state.mtp_drafted)
                state.mtp_drafted = 0
            if getattr(state, "_migration_request", None) != generation:
                state.begin_history_migration(target)
                state._migration_request = generation
            moved = state.migrate_history_bundle()
            self._migration_cursor = (index + 1) % len(self.states)
            pending = getattr(state, "_history_migration", None)
            blocked = ((pending.get("blocked_reason") or "next balanced bundle exceeds admitted migration overlap")
                       if not moved and pending else None)
            self.controller.migration_status(request_epoch=generation, pending=True, blocked=blocked)
            return
        self.controller.migration_status(request_epoch=generation, pending=False)

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
        # Independent prompts retain their standalone bulk GEMM/reduction shape.
        # Decode windows still batch together in the existing round/graph path.
        meta = self.w.meta
        separate = meta.get("execution_provider") == "sm70" and meta.get("sm70_profile") == "optimized"
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
            if separate:
                break
        return pieces

    def _pass(self) -> list[Stream]:
        """One prompt pass alone; prompts that end sample their first token, draft and join the rounds."""

        pieces = self._pieces()
        self._note_passed(pieces)
        self.pbuf.rotor_lookup = bool(pieces) and self._rotor_lookup(pieces[0][0], len(pieces))
        t0 = time.perf_counter()
        marks = self._output_marks([s for s, _, _ in pieces]) if self.controller is not None else None
        active_count = sum(not s.done for s in (*self.streams.values(), *self.filling))
        completed = False
        primary = None
        try:
            try:
                self._read_ahead(pieces)
                windows = [(s.st, s.prompt[a:a + n]) for s, a, n in pieces]
                segs = stage(self.w, self.pbuf, windows)
                ends, cuts = self._end_rows(pieces, segs), self._cuts(pieces, segs)
                try:
                    from .media_mtp import target_inputs

                    with target_inputs(self.w, self.pbuf, windows, segs) as merged:
                        logits = compute(self.w, segs, self.pbuf, logits=bool(ends), ends=ends, cuts=cuts,
                                         **({"input_embeddings": merged} if merged is not None else {}))
                except BaseException as primary:
                    if self.controller is not None:
                        self.controller.work_observations.incomplete(primary)
                    raise
                if self.controller is not None:
                    self.controller.work_observations.target(
                        segs, prefill=True, replay_rows=sum(n for stream, _, n in pieces if stream.carry is not None))
                heads = logits[:len(ends)].clone() if ends else None
                lasts = self._absorb(pieces, segs, cuts)
                completed = True
            except Exception as exc:                         # noqa: BLE001  (these requests fail, the others go on)
                return self._failed(pieces, exc)
            return self._joined(pieces, heads, lasts, (time.perf_counter() - t0) / len(pieces))
        except BaseException as error:
            primary = error
            raise
        finally:
            if self.controller is not None:
                try:
                    self.controller.work_observations.publish()
                    self.controller.observe(
                        active=active_count, scheduled=0, target_rows=0,
                        prefill_tokens=sum(n for _, _, n in pieces) if completed else 0,
                        committed_tokens=self._useful_outputs(marks),
                        max_history_tokens=max((s.st.pos for s, _, _ in pieces), default=0),
                        elapsed_s=time.perf_counter() - t0)
                except BaseException as observation:
                    raise_failures(primary, [observation])

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

        _kv_check(segs, comm=self.w.comm)
        if any(getattr(s.st, "media_input_span", None) is not None for s, _a, _n in pieces):
            return self._absorb_ready(pieces, segs, cuts)
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
            segments = mtp_stage(self.pbuf, absorb)
            try:
                mtp_compute(self.w, segments, self.pbuf, logits=False)
            except BaseException as primary:
                if self.controller is not None:
                    self.controller.work_observations.incomplete(primary)
                raise
            if self.controller is not None:
                self.controller.work_observations.mtp(segments, role="prefill")
            for st, nxt, _ in absorb:
                st.set_mtp_len(st.mtp_len + len(nxt))
        for s, mtp_len, _, snap in points:
            if snap is not None and getattr(s.st, "indexer_history_owner", None) is not None:
                snap.update(s.st.indexer_snapshot_metadata(snap["pos"], mtp_len))
        dflash = getattr(self.w, "dflash2", None)
        for (s, a, n), (st, a0, _) in zip(pieces, segs):
            commit(self.w, st, self.pbuf, n, n, at=a0,
                   **({"commit_dflash": False} if dflash is not None else {}))
        if dflash is not None:
            self._commit_dflash([(state, end - begin) for state, begin, end in segs], prefill=True)
        for s, mtp_len, tail, snap in points:            # a point that ends its piece: the state as committed
            state = snap if snap is not None else {**s.st.snapshot(), "mtp_len": mtp_len}
            if snap is None and getattr(s.st, "indexer_history_owner", None) is not None:
                state.update(s.st.indexer_snapshot_metadata(s.st.pos, mtp_len))
            self.fills[s.sid][3].append((state, tail))
        return lasts

    def _absorb_ready(self, pieces, segs, cuts):
        """Commit actual merged input, then rebuild each method's own frontier.

        The target's pending HC views are copied by one bounded MTP staging
        operation before that MTP pass reuses pbuf.streams. Saved/cut boundaries
        own one complete HC row and its coordinates, never a full-prefix array.
        """
        from .media_mtp import CommittedMTPChunk, condition_chunks, snapshot_state

        at = {(cut.at, cut.row): cut for cut in cuts}
        lasts, points, chunks, chunk_streams = [], [], [], []
        starts = {}
        for (s, start, count), (st, begin, end) in zip(pieces, segs):
            mtp = self.fills[s.sid][1]
            media = st.media_input_span is not None
            checkpoints = tuple(self._points(s, start, count))
            lasts.append(self.pbuf.streams[end - 1:end].clone() if mtp and not media else None)
            for cut_at in checkpoints:
                row = cut_at - start
                valid = max(0, cut_at - 1) if media and mtp else st.mtp_len + row - 1 if mtp else st.mtp_len
                tail = self.pbuf.streams[begin + row - 1:begin + row].clone() if mtp and not media else None
                cut = at.get((begin, row))
                snap = (None if cut is None else cut_snapshot(self.w, st, self.pbuf, cut, valid,
                                                               **({"defer_media": True} if media else {})))
                points.append((s, cut_at, valid, tail, snap, media))
            if mtp:
                starts[s.sid] = st.mtp_len
                chunks.append(CommittedMTPChunk(st, s.prompt, start, start + count,
                                                self.pbuf.streams[begin:end], tuple(k for k in checkpoints if k < start + count)))
                chunk_streams.append(s)
        dflash = getattr(self.w, "dflash2", None)
        for (_s, _a, count), (st, begin, _end) in zip(pieces, segs):
            commit(self.w, st, self.pbuf, count, count, at=begin,
                   **({"commit_dflash": False} if dflash is not None else {}))
        if dflash is not None:
            self._commit_dflash([(st, end - begin) for st, begin, end in segs], prefill=True)
        prepared = {}
        if chunks:
            try:
                results = condition_chunks(self.w, self.pbuf, chunks)
            except BaseException as primary:
                if self.controller is not None:
                    self.controller.work_observations.incomplete(primary)
                raise
            for s, result in zip(chunk_streams, results):
                prepared[s.sid] = result
                if self.controller is not None and s.st.mtp_len > starts[s.sid]:
                    self.controller.work_observations.mtp([(s.st, 0, s.st.mtp_len - starts[s.sid])],
                                                         role="prefill", position=starts[s.sid])
            for index, (s, _a, _n) in enumerate(pieces):
                if s.sid in prepared and prepared[s.sid][0] is not None:
                    lasts[index] = prepared[s.sid][0].hidden  # explicit alias to the State-owned boundary
        for s, cut_at, valid, tail, snap, media in points:
            if snap is None:
                snap = {**s.st.snapshot(), "mtp_len": valid}
            elif media:
                boundary = prepared[s.sid][1][cut_at] if self.fills[s.sid][1] else None
                snap.update(snapshot_state(s.st, position=cut_at, mtp_len=valid, boundary=boundary, take_boundary=True))
            snap.update(s.st.indexer_snapshot_metadata(snap["pos"], valid))
            if media and self.fills[s.sid][1]:
                tail = snap["media_state"]["mtp_boundary"]["hidden"]  # one protected copy, two logical references
            self.fills[s.sid][3].append((snap, tail))
        for s, _a, _n in pieces:
            if s.st.media_input_span is not None:
                s.st.retire_media_input_span(cancelled=False)
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

        _kv_check([(s.st, 0, n) for s, _, n in pieces], comm=self.w.comm)
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
                replay = getattr(self.w, "dflash2", None) is not None and bool(s.owed)
                if s.constraint is not None and not replay:  # replay advances the canonical grammar without a draw
                    logits = s.constraint.mask(logits, None, self.w.meta.get("vocab_offset", 0))
                first = (s.owed[0] if replay else e.sample(logits, [len(s.prompt)], s.sampling,
                         **({"gathered": False} if getattr(self.w, "dflash2", None) is not None else {}))[0])
                limit, terminal = (1, False) if replay else self._prefix_limits([s], [[first]])[0]
                if not limit:
                    s.done = True
                    s.started = s.finished = time.perf_counter()
                    joined.append(s)
                    continue
                if s.probabilities is not None and not replay:
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
            if terminal:
                s.done, s.finished = True, time.perf_counter()
            e.buf.rotor_lookup = self._rotor_lookup(s)
            if e.mbuf is not None:
                e.mbuf.rotor_lookup = e.buf.rotor_lookup
            has_draft = s.draft and (mtp or self.depth > 0 and getattr(self.w, "dspark", None) is not None)
            if has_draft and not s.done and s.count > 1:
                count = self._proposal_count(s, s.count - 1) if has_draft and not s.done else 0
                mtp_position = st.mtp_len
                try:
                    s.drafts = draft(e, last, [first], st.pos + 1, count, s.sampling,
                                     self.confidence)
                except BaseException as primary:
                    if self.controller is not None and mtp:
                        self.controller.work_observations.incomplete(primary)
                    raise
                if self.controller is not None and mtp:
                    observation = self.controller.work_observations
                    observation.mtp([(st, 0, 1)], role="refresh", position=mtp_position,
                                    zero_proposal_rows=int(count == 0))
                    if st.mtp_drafted:
                        observation.mtp([(st, 0, st.mtp_drafted)], role="proposal", position=mtp_position + 1)
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
                    state = {**s.st.snapshot(), "mtp_len": mtp_len}
                    if getattr(s.st, "indexer_history_owner", None) is not None:
                        state.update(s.st.indexer_snapshot_metadata(s.st.pos, mtp_len))
                    self._remember(list(s.prompt[:at]), s.st, state, tail)
                except Exception as exc:                 # noqa: BLE001  its reply is the error; its slot holds nothing
                    s.error = exc
                    self._drop_kept(s.st)
        return streams

    def _ends(self, s: Stream) -> tuple[int, ...]:
        """The end tokens that end this stream: none when its request ignores them (``ignore_eos``)."""

        return self.eos if s.stop_eos else ()

    @torch.no_grad()
    def round(self) -> list[Stream]:
        """One complete coordinator iteration, including bounded maintenance."""
        if self.controller is None:
            return self._round()
        started = time.perf_counter()
        before = self.controller.round_started()
        self._round_migration_s = 0.0
        outcome = "ok"
        primary, result = None, None
        try:
            result = self._round()
        except BaseException as error:
            outcome = type(error).__name__
            primary = error
        errors = []
        try:
            self.controller.work_observations.publish()
            self.controller.observe_round(before=before, elapsed_s=time.perf_counter() - started,
                                          migration_s=self._round_migration_s, outcome=outcome)
        except BaseException as error:
            errors.append(error)
        raise_failures(primary, errors)
        return result

    def _round(self) -> list[Stream]:
        """One round over the live streams, with the next prompt pass in the same forward while prompts fill."""

        error = getattr(self, "_memory_error", None)
        if error is not None:
            raise error
        plan = self._execution_plan
        self._execution_plan = None
        if self.controller is not None and plan is None:
            plan = self.freeze_execution()
            self.controller.accept(plan)
            self.apply_execution_family()
        self._migrate_boundary(None if plan is None else plan["migration"])
        pdl.ON = self.pdl                                # this decoder's launch policy (one process global)
        cancelled = []
        if getattr(self.w, "dflash2", None) is not None:
            for s in self.streams.values():
                if not s.done and s.cancel is not None and s.cancel.is_set():
                    # Checked reconstruction emits no fresh token callback.
                    # Observe cancellation at the same quiescent boundary so
                    # a long owed prefix cannot defer request retirement.
                    s.done, s.finished = True, time.perf_counter()
                    cancelled.append(s)
        ended = cancelled + self._make_room()          # all issued work from the previous round has resolved
        live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if self.filling and (not live or not self.converged):
            ended += self._fill()                      # passes alone; a prompt that ends here joins this round
            live = [s for s in self.streams.values() if not s.done and not s.waiting]
        if not live:
            return ended
        dflash = getattr(self.w, "dflash2", None)
        packets = None
        draft_started = time.perf_counter() if dflash is not None else None
        if dflash is not None:
            # A ChainPacket owns draws from this operation's frozen settings.
            # Nothing from a prior round may be shortened after those draws.
            live, packets = self._dflash_proposals(live, plan)
        elif self.controller is not None:
            for s in live:
                del s.drafts[plan["proposal_limit"]:]
        grammars, failed = {}, []
        for s in live:                                   # a grammar cuts the drafts no accepted path can hold
            if s.constraint is not None and not (dflash is not None and s.owed):
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
        if self.controller is not None and dflash is None:
            selected = self.controller.select([(s.sid, 1 + len(s.drafts)) for s in live], plan)
            by_id = {s.sid: s for s in live}
            # Older BF16 backends defer recurrent commits in shared scratch.
            # Retire every unselected borrower before another round reuses it.
            for s in live:
                if s.sid not in selected:
                    self._flush(s)
            live = [by_id[sid] for sid in selected]
            if not live:
                return failed + ended
        active_count = len(plan["order"]) if plan is not None else len(live)
        self.buf.rotor_lookup = self._rotor_lookup(live[0], len(live))
        self.pbuf.rotor_lookup = False  # any overlapping prompt invalidates the exclusive-request Region
        t0 = time.perf_counter() if draft_started is None else draft_started
        windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
        # the n-gram rows are read once the forward reaches their layer, behind the layers before it
        segs = stage(self.w, self.buf, windows, late=True)
        # a pass shares the round's forward only where their experts share a launch; else _fill ran it between rounds
        pieces, psegs = (self._pieces(self._pass_rows()) if self.filling and self.converged else []), None
        marks = self._output_marks([*live, *(s for s, _, _ in pieces)]) if self.controller is not None else None
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
        half = self.buf.h.dtype == torch.float16
        tables = None if half else gdn_multi.Tables(self.gdn, segs, held, static=self.round_tables)
        self.buf.gdn_tables = tables
        self.buf.attn_step = (None if half else attn_multi.Step(
            self.w, segs, mtp=False, tables=self.step_tables, rotor_lookup=self.buf.rotor_lookup))
        try:
            if pieces:                                 # the window and the pass: each layer's experts once for both
                pends = self._end_rows(pieces, psegs)
                from .media_mtp import target_inputs

                with target_inputs(self.w, self.pbuf, [(s.st, s.prompt[a:a + n]) for s, a, n in pieces], psegs) as merged:
                    logits, heads = compute_mixed(self.w, segs, self.buf, psegs, self.pbuf, ends=pends, cuts=cuts,
                                                  **({"prefill_embeddings": merged} if merged is not None else {}))
                heads = heads[:len(pends)].clone() if pends else None
            else:                                      # a captured forward when one serves this round, else eager
                logits = self.graphs.verify_forward(segs) if self.graphs is not None else None
                if logits is None:
                    logits = compute(self.w, segs, self.buf, check=False)     # the frame's status: checked below
            if self.controller is not None:
                self.controller.work_observations.target(segs, replay_rows=sum(
                    end - begin for s, (_state, begin, end) in zip(live, segs) if dflash is not None and s.owed))
                if pieces:
                    self.controller.work_observations.target(
                        psegs, prefill=True,
                        replay_rows=sum(n for stream, _, n in pieces if stream.carry is not None))
        except BaseException as primary:
            if self.controller is not None:
                self.controller.work_observations.incomplete(primary)
            raise
        finally:
            self.buf.gdn_tables = self.buf.attn_step = None
        if self.w.comm is not None:
            # Delayed KVWords checks are local. Every TP owner must learn a
            # failed peer's status before any rank enters the sampler gather.
            _kv_check(segs, comm=self.w.comm, workspace=getattr(self, "kv_cohort", None))
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
                    s.constraint.mask(logits[a0:a1], grammars[s.sid], self.w.meta.get("vocab_offset", 0))
            launched = (launch_streams(logits, starts, positions, samplings)
                        if self.w.comm is None and dflash is None else None)
        except BaseException:
            words.settle()
            raise
        if dflash is not None:
            from .dflash2_verify import verify_batch
            from tensorfold.cuda.sampling import sample_rows

            words.settle()
            paths, sampled = [], []
            for s, (state, begin, end), positions0 in zip(live, segs, positions):
                if s.owed:
                    # These outputs already belong to the canonical transcript.
                    # Consume the pending canonical input, force its known next
                    # token, and let Stream.take check it without re-emission.
                    path, terminal = [0], s.owed[0]
                    rows = [terminal]
                elif not s.draft:
                    rows = (sample_rows(logits[begin:end], positions0, s.sampling) if self.w.comm is None else
                            tp_sample_rows(self.w, logits[begin:end], positions0, s.sampling, plan=self._sample_plan))
                    path, terminal = [0], rows[0]
                else:
                    row = verify_batch(self.w, logits[begin:end], [(state, 0, end - begin)], [packets[s.sid]],
                        [s.sampling], [s.count - len(s.out)], [self._ends(s)], plan=self._sample_plan)[0]
                    path, terminal, rows = list(row.path), row.terminal, list(row.outputs)
                paths.append((path, terminal))
                sampled.append(rows)
        elif self.w.comm is None:
            sampled = collect_streams(launched, words)
        else:
            words.settle()
            if (not grammars and self.w.meta.get("execution_provider") == "sm70"
                    and all(sampling == samplings[0] for sampling in samplings) and _gathered_fits(samplings[0])):
                chosen = choose_gathered(self.w, self.buf.cand_all, logits.shape[0],
                                         [position for group in positions for position in group], samplings[0],
                                         logits=logits, plan=self._sample_plan)
                sampled = [chosen[a0:a1] for a0, a1 in zip(starts, starts[1:])]
            else:
                sampled = [tp_sample_rows(self.w, logits[a0:a1], pos, sampling, plan=self._sample_plan)
                           for a0, a1, pos, sampling in zip(starts, starts[1:], positions, samplings)]
        if dflash is None:
            paths = [accept(tokens, list(range(-1, len(tokens) - 1)), rows, s.count - len(s.out), self._ends(s))
                     for s, (_, tokens), rows in zip(live, windows, sampled)]
        if self.controller is not None:
            for s, (_, tokens), (path, _end) in zip(live, windows, paths):
                if dflash is None or not s.owed:
                    self.controller.work_observations.verification(len(tokens) - 1, max(0, len(path) - 1))
        proposed = [[tokens[r] for r in path[1:]] + [end]
                    for (_, tokens), (path, end) in zip(windows, paths)]
        if dflash is None:
            boundaries = self._prefix_limits(live, proposed)
        else:
            fresh = [(s, new) for s, new in zip(live, proposed) if not s.owed]
            decisions = iter(self._prefix_limits([s for s, _ in fresh], [new for _, new in fresh]))
            boundaries = [(1, False) if s.owed else next(decisions) for s in live]
        paths = [(path[:count], new[count - 1] if count else end)
                 for (path, end), new, (count, _stop) in zip(paths, proposed, boundaries)]
        for s, (_, tokens), (_, a0, _), (path, end), pos in zip(live, windows, segs, paths, positions):
            if s.probabilities is not None and path and not (dflash is not None and s.owed):
                capture(logits, [tokens[r] for r in path[1:]] + [end], [pos[r] for r in path],
                        s.probabilities, rows=[a0 + r for r in path])
        if not half:
            for s, rows in zip(live, gdn_multi.keep(tables, [len(path) for path, _ in paths])):
                self.held[s.sid] = rows                  # the next round's trees fold these rows in first
        kept = []
        for s, (_, tokens), (st, a0, a1), rows, (path, end), (count, terminal), new in zip(
                live, windows, segs, sampled, paths, boundaries, proposed):
            if path:
                commit(self.w, st, self.buf, a1 - a0, len(path), at=a0, states=False,
                       **({"commit_dflash": False} if dflash is not None else {}))
            else:
                st.discard_target_tentative()
            s.committed.extend(tokens[:len(path)])
            s.counted(len(tokens))
            new = new[:count]
            if s.constraint is not None:
                try:
                    s.constraint.advance(new)
                except GrammarError as exc:
                    s.error = exc
            last = terminal or s.error is not None or len(s.out) + len(new) >= s.count or end in self._ends(s)
            kept.append((s, a0, rows[:len(path)], new, last))
        if dflash is not None:
            # Resolve every accepted target/feature prefix before publishing
            # any output, including a warm zero-proposal operation.
            self._commit_dflash([(s.st, len(path)) for s, (path, _end) in zip(live, paths) if path],
                zero_proposal_rows=sum(len(path) for s, (path, _end) in zip(live, paths)
                                       if s.draft and not s.owed and not packets[s.sid].chain.tokens))
            for s in live:
                s.drafts = []
        else:
            self._draft_all([(s, a0, keep) for s, a0, keep, _, last in kept if keep and s.draft and not last],
                            {s.sid: fresh for s, _, _, fresh, last in kept
                             if getattr(s, "copies", None) is not None and not last})
        for (s, _, _, new, _), (_count, terminal) in zip(kept, boundaries):
            if s.error is not None:
                s.done = True
                continue
            if new:
                s.take(new, self._ends(s))
            if terminal:
                s.done, s.finished = True, time.perf_counter()
        spent = time.perf_counter() - t0
        self._timed(spent, sum(n for _, _, n in pieces))
        if pieces:                                     # prompts that ended in this round's pass join the next
            ended += self._joined(pieces, heads, lasts, spent / len(pieces))
        if self.controller is not None:
            self.controller.observe(active=active_count, scheduled=len(live), target_rows=segs[-1][2],
                                    prefill_tokens=sum(n for _, _, n in pieces),
                                    committed_tokens=self._useful_outputs(marks),
                                    max_history_tokens=max(s.st.pos for s in live),
                                    elapsed_s=time.perf_counter() - t0)
        done = [s for s in live if s.done]
        for s in done:                                   # a finished stream's state is never read again
            self.held.pop(s.sid, None)
        return failed + done + ended

    def _dflash_proposals(self, live, plan):
        """Choose useful rows before any selector/RNG work in this operation.

        The fair order comes from the accepted frozen controller plan. A small
        remainder can shorten the last sequence's count before drawing, and
        zero proposals still owns one target row and maintains conditioning.
        """
        from .dflash2_verify import bind_packet

        by_id = {s.sid: s for s in live}
        order = list(by_id) if plan is None else plan["order"]
        available = self.buf.rows if plan is None else min(self.buf.rows, plan["row_budget"])
        limit = self.depth if plan is None else plan["proposal_limit"]
        logical = getattr(self, "logical_context", self.capacity - self.depth - 1)
        selected, counts = [], []
        for sid in order:
            if sid not in by_id:
                continue
            if available < 1:
                break
            s = by_id[sid]
            count = self.speculation.cap_proposals(limit if s.draft and not s.owed else 0,
                reply_tokens=max(0, s.count - len(s.out)),
                context_tokens=max(0, logical - s.st.pos - 1), target_rows=available)
            selected.append(s)
            counts.append(count)
            available -= count + 1
        if self.controller is not None:
            chosen = self.controller.select([(s.sid, count + 1) for s, count in zip(selected, counts)], plan)
            if chosen != [s.sid for s in selected]:
                raise RuntimeError("DFlash2 proposal rows differ from the frozen controller selection")
        selected_ids = {item.sid for item in selected}
        for s in live:
            if s.sid not in selected_ids:
                self._flush(s)
        fresh = [(s, count) for s, count in zip(selected, counts) if s.draft and not s.owed]
        for s in selected:
            s.drafts = []
        try:
            chains = (self.w.dflash2.propose_many([s.st for s, _ in fresh], [s.out[-1] for s, _ in fresh],
                [count for _, count in fresh], [s.sampling for s, _ in fresh], [s.st.pos + 1 for s, _ in fresh])
                if fresh else [])
        except BaseException as primary:
            if self.controller is not None:
                self.controller.work_observations.incomplete(primary)
            raise
        packets = {}
        for (s, _count), chain in zip(fresh, chains):
            packets[s.sid] = bind_packet(s.st, chain)
            s.drafts = list(chain.tokens)
        if len(chains) != len(fresh):
            raise RuntimeError("DFlash2 returned a different frozen proposal batch")
        if self.controller is not None:
            self.controller.work_observations.dflash_proposals([len(chain.tokens) for chain in chains])
        return selected, packets

    def _commit_dflash(self, accepted, *, prefill=False, zero_proposal_rows=0):
        accepted = [(state, rows) for state, rows in accepted if state.dflash2.enabled]
        if not accepted:
            return
        try:
            self.w.dflash2.commit_many(accepted)
        except BaseException as primary:
            if self.controller is not None:
                self.controller.work_observations.incomplete(primary)
            raise
        if self.controller is not None:
            self.controller.work_observations.dflash_conditioning(sum(rows for _state, rows in accepted),
                prefill=prefill, zero_proposal_rows=zero_proposal_rows)

    def _draft_all(self, streams: list, new: dict | None = None) -> None:
        """Every drafting stream absorbs its kept rows and chains drafts, all streams in one step a depth.
        A stream whose context repeats earlier text drafts that continuation instead of an MTP row. A step's codec
        status words are read back in the next draft readback (``KVWords``); its MTP entries count once they are."""

        provider = getattr(self.w, "dspark", None)
        if provider is not None:
            for s, _at, keep in streams:
                s.drafts = []
                count = self._proposal_count(s, s.count - len(s.out) - len(keep))
                if count <= 0:
                    continue
                if new and s.sid in new and getattr(s, "copies", None) is not None:
                    before = len(s.context)
                    try:
                        s.context.extend(new[s.sid])
                        s.drafts = s.copies.propose(s.context, max(count, COPY_MATCH))[:count]
                    finally:
                        del s.context[before:]
                    if s.drafts:
                        continue
                e = _slot(self.w, s.st, self.buf, None, self.pbuf, self.capacity, self.prefill_rows,
                          logical_context=getattr(self, "logical_context", None))
                s.drafts = provider.draft(e, int(keep[-1]), count, s.sampling, self.confidence)
            return

        bound = getattr(getattr(self, "graphs", None), "bound", None)
        handoff = getattr(bound, "greedy", None)
        room = {s.sid: self._proposal_count(s, s.count - len(s.out) - len(keep)) for s, _, keep in streams}
        if (handoff is not None and self._proposal_limit() == self.depth
                and all(count == self.depth for count in room.values()) and handoff.try_draft(streams, new)):
            return

        for s, _, _ in streams:
            s.drafts = []
        # A zero-proposal step still maintains the selected MTP capability.
        # Otherwise resuming proposals would read stale conditioning history.
        todo = [(s, a0, keep) for s, a0, keep in streams if self.mbuf is not None]
        if not todo:
            return
        for s, _, _ in todo:
            st = s.st
            if st.mtp_drafted:
                st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                st.mtp_drafted = 0
        windows = [(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo]
        segs = mtp_stage(self.mbuf, windows)
        proposing = any(room[s.sid] > 0 for s, _, _ in todo)
        logits = self._mtp(segs) if proposing else self._mtp(segs, logits=False)
        if self.controller is not None:
            self.controller.work_observations.mtp(
                segs, role="refresh", zero_proposal_rows=sum(len(keep) for stream, _, keep in todo
                                                            if room[stream.sid] == 0))

        def absorbed():                                  # the absorb's entries, once its status words are valid
            for (s, _, keep), (st, a0, a1) in zip(todo, segs):
                st.set_mtp_len(st.mtp_len + len(keep))

        words = KVWords(segs, self.kv_host, absorbed)    # its status words join the first draft's readback
        active = [(s, a1 - 1) for (s, _, _), (_, _, a1) in zip(todo, segs)]
        taking = [index for index, (s, _) in enumerate(active) if room[s.sid] > 0]
        if not taking:
            words.settle()
            return
        if len(taking) != len(active):
            logits = logits[taking]
            active = [active[index] for index in taking]
        if not active:
            words.settle()
            return
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
        for j in range(self._proposal_limit()):
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
            if self.controller is not None:
                self.controller.work_observations.mtp(segs, role="proposal")

            def chained(nxt=nxt):                        # the step's entries, once its status words are valid
                for s, _, _ in nxt:
                    s.st.set_mtp_len(s.st.mtp_len + 1)
                    s.st.mtp_drafted += 1

            words = KVWords(segs, self.kv_host, chained)  # the next pick's readback checks them (a step has one)
            active = [(s, a0) for (s, _, _), (_, a0, _) in zip(nxt, segs)]

    def _mtp(self, segs: list, *, logits: bool = True, next_inputs: torch.Tensor | None = None) -> torch.Tensor | None:
        """An MTP step over every drafting stream, its attention one launch a kernel for all of them."""

        self.mbuf.rotor_lookup = False
        if self.live() == 1:
            request = next(iter(self.streams.values())) if self.streams else self.filling[0]
            self.mbuf.rotor_lookup = self._rotor_lookup(request, len(segs))
        self.mbuf.attn_step = (None if self.mbuf.h.dtype == torch.float16 else attn_multi.Step(
            self.w, segs, mtp=True, tables=self.mtp_tables, rotor_lookup=self.mbuf.rotor_lookup))
        try:                                             # the caller checks the step's status
            out = (self.graphs.mtp_step(segs)
                   if logits and self.graphs is not None and next_inputs is None
                   and not any(getattr(st, "media_profile_identity", None) is not None for st, _a, _z in segs) else None)
            if out is None:
                out = mtp_compute(self.w, segs, self.mbuf, check=False, logits=logits,
                                  **({"next_inputs": next_inputs} if next_inputs is not None else {}))
            if self.w.comm is not None:
                _kv_check(segs, comm=self.w.comm, workspace=getattr(self, "kv_cohort", None))
            return out
        except BaseException as primary:
            if self.controller is not None:
                self.controller.work_observations.incomplete(primary)
            raise
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
        if self.w.comm is not None:
            words.settle()
            if (self.w.meta.get("execution_provider") == "sm70"
                    and logits.data_ptr() == self.mbuf.logits.data_ptr()
                    and all(sampling == samplings[0] for sampling in samplings) and _gathered_fits(samplings[0])):
                chosen, probabilities = choose_gathered(
                    self.w, self.mbuf.cand_all, len(positions), positions, samplings[0], with_prob=True,
                    logits=logits, plan=self._draft_plan)
                return list(zip(chosen, probabilities))
            picks = []
            for row, (position, sampling) in enumerate(zip(positions, samplings)):
                tokens, probabilities = tp_sample_rows(
                    self.w, logits[row:row + 1], [position], sampling, with_prob=True, plan=self._draft_plan,
                    offset=int(self.w.meta["vocab_offset"]), id_map=self.w.draft_ids)
                picks.append((tokens[0], probabilities[0]))
            return picks
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
        """Retire each request; only bounded derived prompt prefixes can keep its slot's bytes."""

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
    from .indexer_history import IndexerHistory, HistoryLayer

    for value in vars(st).values():
        for v in value if isinstance(value, list) else [value]:
            if isinstance(v, IndexerHistory):
                yield from v.device_tensors()
            elif isinstance(v, HistoryLayer):
                # The State's explicit owner above accounts for each shared
                # ring/stage once; a layer handle owns no additional storage.
                continue
            elif isinstance(v, torch.Tensor):
                yield v
            elif hasattr(v, "__dict__"):                  # scratch and KV cache objects, the MTP head's too
                yield from (t for t in vars(v).values() if isinstance(t, torch.Tensor))

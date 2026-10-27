"""Flash Next's concurrent rounds on one GPU: every stream keeps exactly its own accepted prefix."""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import torch

from tensorfold.cleanup import raise_failures, rollback
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
from tensorfold.cuda.streams import Stream, accept, mark_error_state_retired, prefix_limit
from tensorfold.engine.exact_sampling import MARGIN, choose_rows
from tensorfold.engine.grammar import GrammarError

from ..cuda import CONFIDENCE, DEPTH
from . import attn_multi, gdn_multi, image_rows, media_requests, multi_graphs, prefixes
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
                 full_windows: int | None = None, execution_rows: int | None = None, memory_live=None,
                 logical_context: int | None = None) -> None:
        if w.comm is not None and w.meta.get("execution_provider") != "sm70":
            raise ValueError("concurrent tensor-parallel Flash Next requires the SM70 execution provider")
        budget = (multi_graphs.budget_bytes() if graphs else 0) if graph_budget is ... else graph_budget
        if budget is not None and (type(budget) is not int or budget < 0):
            raise ValueError("captured-round budget must be nonnegative integer bytes or automatic")
        if full_windows is not None and (type(full_windows) is not int or full_windows < 1):
            raise ValueError("full KV windows must be a positive integer")
        if memory_live is not None and not callable(memory_live):
            raise ValueError("prepared memory admission must provide a callable live-byte reader")
        if logical_context is not None:
            if type(logical_context) is not int or not 0 < logical_context <= capacity:
                raise ValueError("decoder logical context must fit its physical cache capacity")
            self.logical_context = logical_context
        self.full_windows = full_windows
        self.w, self.depth, self.confidence, self.capacity = w, depth, confidence, capacity
        from .rotor_decoder import RotorDecoderPolicy
        self.rotor_decoder_policy = RotorDecoderPolicy()
        self.points = points                         # a prompt's message starts to keep states at, or None
        self.vision = vision
        self.media_runtime = media_requests.runtime(vision)
        self.admission_policy = "legacy"
        self.retention_catalog = None
        if self.media_runtime is not None and (
                type(vision_workspace_bytes) is not int or vision_workspace_bytes < 0
                or vision_workspace_bytes != self.media_runtime.resources["media_transient_bytes"]):
            raise ValueError("media live-memory reserve differs from its admitted full transient resource plan")
        self._media_limits = None
        self.copy = os.environ.get("TENSORFOLD_MTP_COPY", "0") == "1" and getattr(w, "dflash2", None) is None
        self.eos = tuple(w.cfg.eos) if stop_eos else ()
        rows = slots * (depth + 1) if execution_rows is None else execution_rows
        if type(rows) is not int or not depth + 1 <= rows <= slots * (depth + 1):
            raise ValueError("execution_rows must hold one complete verification window within the resident envelope")
        self.controller = None
        self.execution_manifest_installer = None
        self.poison_control = None
        self.input_preparation_context = w.meta.get("input_preparation_context")
        if self.input_preparation_context is not None:
            self.input_preparation_context.bind_decoder(self)
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
                           kv_key_dtype=kv_key_dtype, kv_value_dtype=kv_value_dtype, slot=slot,
                           logical_context=logical_context)
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
        if self.media_runtime is not None:
            # The resident producer/weights are already excluded from live
            # free bytes. Protect all remaining media transient work once,
            # additively with language work absent a serialized-max proof.
            reserve += vision_workspace_bytes
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

            self.memory_gate = SharedMemoryGate(self.memory_gate, w.comm, rank_local=self.media_runtime is not None)
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
        self._kept: list[tuple[list[int], State, dict, torch.Tensor | None]] = []  # legacy until catalog authority binds
        self.keep = keep
        self.used: dict[int, int] = {}                   # retention (prefixes.py): id(slot) -> clock at its last use
        self.clock = 0

    def _rotor_lookup(self, request, participants=1):
        return self.rotor_decoder_policy.request(request, owned_requests=self.live(),
                                                 participants=participants, copy=self.copy)

    @property
    def kept(self):
        catalog = getattr(self, "retention_catalog", None)
        return self._kept if catalog is None else catalog.prefix_entries

    @kept.setter
    def kept(self, values):
        catalog = getattr(self, "retention_catalog", None)
        if catalog is None:
            self._kept = values
        else:
            catalog.prefix_entries = values

    def bind_retention_catalog(self, catalog):
        """Bind the one actual catalog before request/prefix publication."""
        from .retained_catalog import RetainedCatalog

        if type(catalog) is not RetainedCatalog or self.retention_catalog is not None:
            raise ValueError("retention requires one actual catalog issued for this decoder")
        if catalog.prefix_limit != self.keep or catalog.recovery_versions != max(self.keep, POINTS):
            raise ValueError("retention catalog differs from the decoder's actual prefix/recovery geometry")
        if self.streams or self.filling or self.kept or catalog.prefix_entries:
            raise RuntimeError("retention authority binds only to the empty warmed request/prefix state")
        catalog.bind_decoder(self)
        catalog.prefix_entries = self._kept
        self.retention_catalog = catalog
        self._kept = None  # The bound catalog is the sole prefix authority.

    def _busy(self) -> set[int]:
        busy = {id(s.st) for s in [*self.streams.values(), *self.filling]}
        catalog = getattr(self, "retention_catalog", None)
        if catalog is not None:
            busy.update(id(state) for state in catalog.protected_states())
        return busy

    def enqueue_retained(self, stream, *, parent=None, prefix_serial=None):
        """Actor-only ownership before expiry controls or a GPU admission claim."""
        from .retained_host import RetainedHost

        catalog = self.retention_catalog
        if catalog is None or catalog.policy is None:
            return
        intent = stream.retention_intent
        stream.retention_request = None
        stream.retention_handle = None
        stream.retention_window = None
        if intent is not None and intent.target is not None:
            handle = catalog.reserve_pending(intent.target, future_bytes=catalog.future_claim())
            stream.retention_handle = handle  # Journal before aliases/old CPU retirement.
            if handle.phase != "reserved" or handle.host is not None:
                stream.retention_window = catalog.replace_pending(handle, retire_host=RetainedHost.retire_backend)
            else:
                stream.retention_window = handle.state
            for alias in intent.aliases:
                if alias not in handle.owners:
                    catalog.associate(handle, alias)
        if parent is not None or prefix_serial is not None:
            if prefix_serial is not None:
                from .retained_catalog import RetentionExpired

                try:
                    source = catalog.by_serial(prefix_serial)
                except RetentionExpired:
                    return  # Required target reservation may reclaim an unleased optional cache candidate.
                if source.role != "prefix":
                    raise RuntimeError("optional prefix binding changed its issued catalog role")
            else:
                source = catalog.lookup(parent)
            if source is None:
                raise RuntimeError("selected retained prefix vanished before its replicated consumer claim")
            def register(request):
                stream.retention_request = request
            if stream.retention_handle is None:
                from .retained_catalog import LOCAL_PRINCIPAL

                catalog.borrow_prefix(source, principal=stream.principal or LOCAL_PRINCIPAL, register=register)
            else:
                catalog.borrow(source, continuation="new-turn", target_handle=stream.retention_handle, register=register)

    def retention_eligibility(self, stream):
        request = getattr(stream, "retention_request", None)
        return None if request is None else request.eligibility()

    def _capacity_refusal(self, error):
        """Only clean cache/backing exhaustion may leave an accepted generation warm."""
        try:
            from .execution_identity import _unretired

            uncertain = _unretired(error)
            self.retention_catalog.check_health()
        except BaseException as inspection:
            try:
                self.poison_control(error)
            except BaseException as containment:
                raise_failures(error, (inspection, containment))
            raise_failures(error, (inspection,))
        if uncertain:
            try:
                self.poison_control(error)
            except BaseException as containment:
                raise_failures(error, (containment,))
            raise error

    def suspend_request(self, stream):
        """Quiescent live request only; preserve its actual host and drawn method state."""
        from .retained_catalog import LOCAL_PRINCIPAL, RetentionCapacity
        from .retained_host import FillContinuation, RetainedHost

        catalog = self.retention_catalog
        if (catalog is None or catalog.policy is None or catalog.policy.tier != "ram"
                or stream.done or stream.cancel is not None and stream.cancel.is_set()
                or getattr(stream, "blocked_input", None) is None):
            return False
        handle = getattr(stream, "retention_handle", None)
        if handle is None or handle.transient:
            try:
                handle = catalog.reserve_temporary(stream.principal or LOCAL_PRINCIPAL,
                    register=lambda owner: setattr(stream, "retention_handle", owner))
            except RetentionCapacity as error:
                self._capacity_refusal(error)
                return False  # GPU remains authoritative; no recomputation or lost accepted generation.
        fill = self.fills.get(stream.sid)
        continuation, hidden = None, None
        if fill is not None:
            engine, enabled, cursor, unpromoted, hidden = fill
            if unpromoted:
                raise RuntimeError("live retention must follow publication of the bounded prompt cuts")
            continuation = FillContinuation(enabled, cursor, tuple(engine.stops))
        self._flush(stream)
        state, previous = stream.st, handle.host
        if not state.mtp_hidden_width or state.mtp_len - state.mtp_drafted != state.pos - 1:
            hidden = None
        if state.mtp_boundary is not None:
            hidden = state.mtp_boundary.hidden
        host = RetainedHost.capture(stream, leader=self.w.comm is None or self.w.comm.rank == 0,
                                    fill=continuation, frozen=True)
        recoveries = tuple((snapshot, tail) for _ids, owner, snapshot, tail in self.kept if owner is state)
        catalog.capture_idle(stream, pending_hc=hidden, recoveries=recoveries, host=host, handle=handle)
        request = getattr(stream, "retention_request", None)
        if request is not None:
            request.close()  # Positive previous restore/copy completion precedes active execution.
        if previous is not None and previous is not host:
            previous.close()  # This snapshot view retires; the original CPU caller still owns the same Stream.
        if fill is not None:
            self.filling.remove(stream)
            self.fills.pop(stream.sid)
        else:
            self.streams.pop(stream.sid)
        host.detach(state)
        stream.retained_host = host
        stream.retention_request = None
        catalog.suspend(handle)
        catalog.borrow(handle, continuation="same-request",
                       register=lambda request: setattr(stream, "retention_request", request))
        return True

    def _resume_live(self, stream):
        request, host = stream.retention_request, stream.retained_host
        if stream.cancel is not None and stream.cancel.is_set():
            return False  # The owning queue retires cancellation, without issuing a new restore/model step.
        state = getattr(stream, "retention_window", None)
        if state is None:
            if not self.free and prefixes._idle(self, self._busy()) is None:
                raise mark_error_state_retired(NoRoom("live continuation waits for a complete recyclable GPU window"))
            state, _resume, _cached = self._slot_for(list(stream.prompt), False, principal=stream.principal)
            stream.retention_window = state
            if not self._grow(state, state.limit, alone=not self.streams and not self.filling):
                raise NoRoom("live continuation waits for its complete future GPU claim")
            # Restore preflight checks the current admitted media/numerical binding.
            if stream.vision is not None:
                from .media_mtp import bind_context

                bind_context(state, self.media_runtime.identity, stream._media_request.plan.media_identity)
            state.cache_principal = stream.principal
            state.media_mtp_enabled = stream.draft and self.w.mtp is not None and self.mbuf is not None
            if getattr(self.w, "dflash2", None) is not None:
                self.w.dflash2.set_enabled(state, stream.draft)
        handoff = request.claim(state, stream=stream, ticket=stream.admission_ticket)
        if handoff is None:
            return False
        _original, fill = host.attach(state, handoff=handoff, frozen=True)
        if fill is not None:
            engine = _slot(self.w, state, self.buf, self.mbuf, self.pbuf, self.capacity, self.prefill_rows,
                           logical_context=getattr(self, "logical_context", None))
            engine._draft_map, engine.stops = self._draft_map, list(fill.stops)
            engine.last_streams = handoff["pending_hc"]
            self.fills[stream.sid] = [engine, fill.mtp, fill.cursor, [], handoff["pending_hc"]]
            self.filling.append(stream)
        else:
            self.streams[stream.sid] = stream
        stream.retention_window = None
        stream.retained_host = None
        return None

    def _release_retained_states(self, *, already_reset=()):
        catalog = self.retention_catalog
        if catalog is None:
            return
        active = {id(stream.st) for stream in [*self.streams.values(), *self.filling]}
        for state in catalog.take_released_states():
            if id(state) in active or any(state is protected for protected in catalog.protected_states()):
                raise RuntimeError("catalog released a State still owned by a request or transfer")
            self._drop_kept(state)
            if all(state is not reset for reset in already_reset):
                self._shrink(state)
            if all(state is not free for free in self.free):
                self.free.append(state)

    def _capture_retained(self, stream, pending_hc=None, *, frame=None, row=None):
        """Before the shared target frame is reused; the catalog clones the bounded HC row."""
        handle = getattr(stream, "retention_handle", None)
        if (stream.error is not None or getattr(stream, "retention_captured", False)
                or stream.cancel is not None and stream.cancel.is_set()):
            return
        if handle is None:
            catalog = getattr(self, "retention_catalog", None)
            if catalog is None or catalog.policy is None or catalog.policy.tier != "ram":
                return
            from .prefix_policy import reuse_allowed
            from .retained_catalog import LOCAL_PRINCIPAL, RetentionCapacity

            if not reuse_allowed(
                    self.w, stream.sampling, draft=stream.draft, vision=stream.vision,
                    position=stream.st.pos, kv_pair=self.kv_pair):
                return
        from .retained_host import RetainedHost

        self._flush(stream)
        state = stream.st
        if (pending_hc is None and frame is not None and row is not None and state.mtp_boundary is None
                and state.mtp_hidden_width and state.mtp_len - state.mtp_drafted == state.pos - 1):
            pending_hc = frame.streams[row:row + 1]
        if not state.mtp_hidden_width or state.mtp_len - state.mtp_drafted != state.pos - 1:
            pending_hc = None
        if state.mtp_boundary is not None:
            pending_hc = state.mtp_boundary.hidden
        recoveries = tuple((snapshot, tail) for _ids, owner, snapshot, tail in self.kept if owner is state)
        if handle is None:
            try:
                handle = catalog.capture_prefix(stream, principal=stream.principal or LOCAL_PRINCIPAL,
                    pending_hc=pending_hc, recoveries=recoveries,
                    register=lambda owner: setattr(stream, "retention_handle", owner))
            except RetentionCapacity as error:
                self._capacity_refusal(error)
                return  # Optional cache acquisition never fails an accepted generation.
            stream.retention_captured = True
            return
        host = RetainedHost.capture(stream, leader=self.w.comm is None or self.w.comm.rank == 0, completed=True)
        stream.retirement_host = host  # Existing logical input journal retires it after the actual worker proof.
        self.retention_catalog.capture_idle(stream, pending_hc=pending_hc, recoveries=recoveries, host=host, handle=handle)
        stream.retention_captured = True
        # The actor still needs stream.st for finish/ACK. Detachment occurs in finish.

    def retire_retained_request(self, stream):
        """Consumer-only retirement; logical owners and issued DMA remain catalog-owned."""
        primary = stream.error
        try:
            from .retained_catalog import RetentionError
            from .execution_identity import _unretired

            outcome = stream.error if isinstance(stream.error, RetentionError) else None
            request = getattr(stream, "retention_request", None)
            owner = None if request is None else request.restore_handle
            if owner is not None and owner.failure is not None:
                outcome = owner.failure
                primary = owner.failure
            if primary is not None:
                if _unretired(primary):
                    raise primary
                self.retention_catalog.check_health()
            return self._retire_retained_request(stream, outcome)
        except BaseException as failure:
            if primary is None or failure is primary:
                raise
            try:
                # Latch the actual TP owner before fallible diagnostic owner
                # retention. Even marker-allocation failure stays terminal.
                self.poison_control(primary)
            except BaseException as containment:
                def uncertain_retirement(failure=failure, containment=containment):
                    raise_failures(failure, (containment,))
                rollback(stream, primary, uncertain_retirement)
            def uncertain_retirement(failure=failure):
                raise failure
            rollback(stream, primary, uncertain_retirement)

    def _retire_retained_request(self, stream, outcome):
        if stream.sid in self.streams or any(stream is item for item in self.filling):
            return True  # Actual rank finish releases GPU readers after removing the active owner.
        request = getattr(stream, "retention_request", None)
        if request is not None:
            owner = request.restore_handle
            if owner is not None and owner.transfer is not None:
                return False
            if owner is not None and owner.failed_transfer is not None:
                outcome = owner.failure
                # A corrupt source is a completed pre-write refusal. Fence the
                # actor's actual destination before releasing its plan/frames.
                if owner.state is not None and owner.state.pos_dev.device.type == "cuda":
                    torch.cuda.current_stream(owner.state.pos_dev.device).synchronize()
                self.retention_catalog.retire_failed_destination(request, fenced=True)
            request.close()
            stream.retention_request = None
        host = getattr(stream, "retained_host", None)
        if host is not None:
            stream.retirement_host = host
            stream.retained_host = None
        handle = getattr(stream, "retention_handle", None)
        reset = ()
        if stream.error is not None and handle is not None and handle.state is stream.st and stream.st is not None:
            state = stream.st
            self._flush(stream)
            if state.pos_dev.device.type == "cuda":
                torch.cuda.current_stream(state.pos_dev.device).synchronize()
            self._drop_kept(state)
            self._shrink(state)
            self.retention_catalog.retire_backend(handle, state=state, fenced=True, error=stream.error,
                detach_host=lambda owner: setattr(stream, "retirement_host", owner))
            reset = (state,)
        if handle is not None and handle.role == "temporary" and not getattr(stream, "retention_captured", False):
            if handle.host is not None:
                stream.retirement_host = handle.host
            self.retention_catalog.release_temporary(handle)
            stream.retention_handle = None
        self._release_retained_states(already_reset=reset)
        if all(stream is not item for item in self.filling) and stream.sid not in self.streams:
            window = getattr(stream, "retention_window", None)
            if window is not None:
                stream.retention_window = None
                if (all(window is not state for state in self.retention_catalog.protected_states())
                        and all(window is not state for state in self.free)):
                    self._drop_kept(window)
                    self._shrink(window)
                    self.free.append(window)
                stream.st = None
        if outcome is not None:
            # Reuse the existing native-descriptor transport after the actual
            # catalog destination/readers retired. Never clear unknown owners.
            from tensorfold.server.scheduler import Scheduler as HostScheduler

            HostScheduler._clear_failure_frames(outcome)
            mark_error_state_retired(outcome)
        return True

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

    def _slot_for(self, prompt: list[int], reuse: bool, *, principal=None):
        """Reuse the longest kept point, copying a fork into a free slot when the memory gate permits it."""

        fifo = getattr(self, "admission_policy", "legacy") == "fifo"
        best = (prefixes._best(self.kept, prompt, principal=principal) if fifo else
                prefixes._best(self.kept, prompt)) if reuse else None
        if best is not None:
            source, snap = best[1:3]
            if source.compact_indexer is not None:
                from .history_owner import check_prefix_retirement

                check_prefix_retirement(source, source, self.poison_control)
            try:
                source.kv_validate_snapshot(snap)
            except KVNumericError:
                self._drop_kept(source)
                if id(source) not in self._busy():
                    self._shrink(source)
                    if all(st is not source for st in self.free):
                        self.free.append(source)
                raise
        return prefixes.slot_for(self, prompt, reuse, principal=principal) if fifo else prefixes.slot_for(self, prompt, reuse)

    def _remember(self, ids: list[int], st: State, snap: dict, tail) -> None:
        st.kv_validate_snapshot(snap)
        if "media_state" in snap:
            return  # cross-request media borrowing remains unqualified; recovery snapshots stay valid
        if getattr(self, "admission_policy", "legacy") == "fifo":
            snap = {**snap, "prefix_principal": st.cache_principal}
            prefixes.remember(self, ids, st, snap, tail, principal=st.cache_principal)
        else:
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
        catalog = getattr(self, "retention_catalog", None)
        protected = () if catalog is None else catalog.protected_states()
        first = min(self.capacity, FIRST)
        active_states = {id(st): st for st in busy}
        active = sum(max(0, st.cache_bytes() - st.cache_bytes(first)) for st in active_states.values())
        retained = sum(max(0, st.cache_bytes() - st.cache_bytes(first)) for st in protected if id(st) not in active_states)
        per_token = -(-self.window_bytes // self.capacity) if self.capacity else 0
        kept = self.kept
        ends: dict[int, int] = {}
        for ids, slot, _snap, _tail in kept:
            ends[id(slot)] = max(ends.get(id(slot), 0), len(ids))
        kv = {"budget_bytes": budget, "held_bytes": gate.held, "active_bytes": active,
              "usage_ratio": round(active / budget, 6) if budget > 0 else None, "bytes_per_token": per_token,
              "free_tokens_estimate": max(0, budget - active - retained) // per_token if per_token else None}
        if catalog is not None:
            kv["protected_idle_bytes"] = retained
        return {"kv": kv,
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

    def media_runnable(self):
        """Readiness only: expected input failures are handled at the next round boundary."""
        if getattr(self, "admission_policy", "legacy") == "fifo":
            if any(self._runnable(stream) for stream in self.streams.values()):
                return True
            return bool(self._pieces())
        if getattr(self, "media_runtime", None) is None:
            return True
        if any(not stream.done and not stream.waiting for stream in self.streams.values()):
            return True
        limits, failures = self.prepare_media_round()
        self._media_limits = limits
        return bool(failures or self._pieces())

    def validate_request(self, stream):
        """Pure selected-profile feasibility check, before a FIFO ticket or device claim."""
        if not stream.prompt:
            raise mark_error_state_retired(ValueError("prefill requires at least one token"))
        room = self.capacity - len(stream.prompt) - self.depth - 1
        if room < 1:
            raise mark_error_state_retired(ValueError(
                f"a prompt of {len(stream.prompt)} tokens leaves no room in the {self.capacity}-token context"))
        if any(type(token) is not int or not 0 <= token < self.w.cfg.vocab for token in stream.prompt):
            raise mark_error_state_retired(ValueError("prompt token IDs differ from the selected model vocabulary"))
        return room

    def input_reservation_bytes(self, stream=None):
        """Actual complete prompt growth for one logical App operation.

        Thinking/gating may issue larger backend prompts under that same owner.
        Native token IDs and the list allocation are charged, not serialized
        int32 extents. Fixed replay sources remain the same physical lease.
        With no stream, return the mandatory text-only floor of every claim;
        endpoint commitment uses the same formula as actual enqueue admission.
        """
        rows = self.capacity - self.depth - 2
        header = sys.getsizeof([])
        pointer = sys.getsizeof([None]) - header
        integer = sys.getsizeof(min(self.w.cfg.vocab - 1, (1 << 31) - 1))
        replay = None if stream is None else getattr(stream.vision, "replay", None)
        sources = 0 if replay is None else sum(reference.byte_length for reference in replay.refs)
        return header + (rows + 1) * pointer + rows * integer + sources

    def prepare_media_round(self):
        """Leader-only CPU errors become per-request outcomes before GPU publication."""
        from tensorfold.vision.images_http import ImageInputError

        owner = getattr(self, "media_runtime", None)
        if owner is None:
            return {}, []
        limits, failures = {}, []
        for stream in self.filling:
            request = getattr(stream, "_media_request", None)
            if request is None:
                continue
            start = self.fills[stream.sid][2]
            try:
                request.prime(start)
                count = request.limit(start, min(self.prefill_rows, owner.profile.execution.input_rows,
                                                  len(stream.prompt) - start))
            except ImageInputError as error:
                try:
                    from .execution_identity import _unretired

                    uncertain = _unretired(error)
                    if not uncertain:
                        owner.check_media_health()
                except BaseException as inspection:
                    try:
                        self.poison_control(error)
                    except BaseException as containment:
                        raise_failures(error, (inspection, containment))
                    raise_failures(error, (inspection,))
                if uncertain:
                    raise
                failures.append((stream, error))
                count = 0
            limits[stream.sid] = (start, count)
        return limits, failures

    @torch.no_grad()
    def admit(self, s: Stream) -> None:
        """Queue a request in a free slot (a kept prompt end it extends, if any); rounds prefill its prompt."""

        error = getattr(self, "_memory_error", None)
        if error is not None:
            raise error
        if getattr(s, "retained_host", None) is not None:
            return self._resume_live(s)
        if getattr(self, "admission_policy", "legacy") == "fifo":
            principal = s.principal
            room = self.validate_request(s)
            ticket = getattr(s, "admission_ticket", None)
            if type(ticket) is not int or not 0 <= ticket < 1 << 63:
                raise mark_error_state_retired(ValueError("FIFO admission requires its validated enqueue ticket"))
        else:
            room = self.capacity - len(s.prompt) - self.depth - 1
            if room < 1:
                raise mark_error_state_retired(ValueError(
                    f"a prompt of {len(s.prompt)} tokens leaves no room in the {self.capacity}-token context"))
        s.count = max(1, min(s.count, room))
        if getattr(self.w, "dflash2", None) is not None and len(s.owed) > s.count:
            raise ValueError("committed replay exceeds the current context/reply allowance")
        if any(x.waiting for x in self.streams.values()):
            raise mark_error_state_retired(NoRoom("streams already wait for memory; a new request waits until one finishes"))
        t0 = time.perf_counter()
        from .prefix_policy import reuse_allowed

        reuse = reuse_allowed(self.w, s.sampling, draft=s.draft, vision=s.vision,
                              position=len(s.prompt), kv_pair=self.kv_pair)
        retained = getattr(s, "retention_request", None)
        window = getattr(s, "retention_window", None)
        if (window is None and getattr(self, "admission_policy", "legacy") == "fifo"
                and not self.free and prefixes._idle(self, self._busy()) is None):
            raise mark_error_state_retired(NoRoom("all stream slots remain owned by active or retained generations"))
        if window is not None:
            st, resume, s.cached = window, None, 0
        elif retained is not None:
            st, resume, s.cached = self._slot_for(list(s.prompt), False,
                                                principal=principal)
        elif getattr(self, "admission_policy", "legacy") == "fifo":
            st, resume, s.cached = self._slot_for(list(s.prompt), reuse,
                                                principal=principal)
        else:
            st, resume, s.cached = self._slot_for(list(s.prompt), reuse)
        try:
            if not self._grow(st, st.limit if retained is not None else len(s.prompt) + self.depth + 2,
                              alone=not self.streams and not self.filling):
                raise NoRoom(f"a {len(s.prompt)}-token prompt waits for memory until a live stream finishes")
        except NoRoom as error:
            if resume is None:
                self.free.append(st)
            else:
                self._remember(list(s.prompt[:s.cached]), st, resume["state"], resume["tail"])
            mark_error_state_retired(error)
            raise
        except BaseException as primary:  # noqa: BLE001 - recover the popped slot even after interrupted growth
            try:
                self._quarantine(st)
                self._retire_failed_slot(st)
            except BaseException as cleanup:  # noqa: BLE001 - preserve every cleanup failure with its primary
                raise_failures(primary, [cleanup])
            mark_error_state_retired(primary)
            raise
        e = _slot(self.w, st, self.buf, self.mbuf, self.pbuf, self.capacity, self.prefill_rows,
                  logical_context=getattr(self, "logical_context", None))
        e._draft_map = self._draft_map
        mtp = s.draft and self.depth > 0 and self.mbuf is not None
        handoff = None
        try:
            if retained is not None:
                from .retained_catalog import RetentionMismatch

                s.retention_window, s.st = st, st  # Owned even while asynchronous restore parks this ticket.
                if retained.restore_handle is None and retained.handoff is None:
                    prefill_begin(e, s.prompt, mtp=mtp, new_sequence=False)
                    image_rows.begin(e, s, self.vision)
                try:
                    handoff = retained.claim(st, stream=s, ticket=s.admission_ticket)
                except RetentionMismatch:
                    # A verified pre-copy cache miss permits ordinary prefill
                    # for a new turn; its protected source stays authoritative.
                    retained.close()
                    s.retention_request = None
                    retained = None
                    handoff = None
                if retained is not None:
                    if handoff is None:
                        return False
                    tail = handoff["pending_hc"] if st.mtp_len == st.pos - 1 else None
                    resume = {"state": st.snapshot(), "tail": tail}
                    s.cached = st.pos
            elif window is not None:
                st.begin_sequence_generation()
            begin = prefill_begin(e, s.prompt, mtp=mtp, resume=resume, new_sequence=False)
            if getattr(self, "admission_policy", "legacy") == "fifo":
                st.cache_principal = principal
            if handoff is not None:
                for snapshot, tail in handoff["recoveries"]:
                    self._remember(list(s.prompt[:snapshot["pos"]]), st, snapshot, tail)
            st.media_mtp_enabled = mtp
            if getattr(self.w, "dflash2", None) is not None:
                self.w.dflash2.set_enabled(st, s.draft)
            image = s.vision is not None
            image_rows.begin(e, s, self.vision)
            if image and self.media_runtime is None:
                torch.cuda.empty_cache()                 # the tower's scratch back before the prompt's passes
        except Exception as primary:
            if getattr(s, "retention_window", None) is not None:
                # The queued actor/catalog retirement path owns this window,
                # including completed pre-write refusals and issued DMA.
                raise
            try:
                self._drop_kept(st)
                self._shrink(st)
                self.free.append(st)
            except BaseException as cleanup:
                raise_failures(primary, [cleanup])
            mark_error_state_retired(primary)
            raise
        e.stops = sorted({p for p in self.points(s.prompt) if begin + MIN_GAP <= p < entry_end(s.prompt)}) \
            if s.draft and st.image_positions is None and getattr(st, "media_profile_identity", None) is None \
            and self.points is not None else []
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
        if getattr(s, "retention_window", None) is not None:
            s.retention_window = None

    def _fill(self) -> list[Stream]:
        """Fill ordered prompt pieces until a stream decodes, a waiting request needs admission or a filling prompt's
        client has left (the scheduler then calls ``cancel`` before the next round)."""

        ended: list[Stream] = []
        while self.filling:
            if getattr(self, "media_runtime", None) is not None and not self._pieces():
                break  # CPU readiness wakes the scheduler; never spin on an unfinished decoder.
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
        limit = self._proposal_limit()
        capacity = getattr(stream, "output_capacity", None)
        if capacity is not None:
            limit = min(limit, capacity - 1)
        return self.speculation.cap_proposals(
            limit, reply_tokens=max(0, remaining),
            context_tokens=max(0, logical - (stream.st.pos if stream.st is not None else stream.retained_host.position) - 1),
            target_rows=self.buf.rows)

    def output_need(self, stream):
        """The whole legal next emission window; already drawn proposals stay intact."""
        host = getattr(stream, "retained_host", None)
        if stream.sid in self.fills or host is not None and host.fill is not None:
            return 1
        remaining = stream.count - len(stream.out)
        if getattr(self.w, "dflash2", None) is not None and stream.draft and not stream.owed:
            return min(remaining, self._proposal_count(stream, remaining) + 1)
        return min(remaining, 1 + len(stream.drafts))

    def _runnable(self, stream):
        budget = getattr(stream, "production_budget", None)
        return (not stream.done and not stream.waiting and getattr(stream, "blocked_input", None) is None
                and (budget is None or budget >= self.output_need(stream)))

    def _take(self, stream, new):
        before = stream.committed_output_tokens
        stream.take(new, self._ends(stream))
        budget = getattr(stream, "production_budget", None)
        if budget is not None:
            used = stream.committed_output_tokens - before
            if used > budget:
                raise RuntimeError("committed output exceeded its frozen production window")
            stream.production_budget = budget - used

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
        active = {id(stream.st) for stream in [*self.streams.values(), *self.filling] if stream.st is not None}
        catalog = getattr(self, "retention_catalog", None)
        protected = set() if catalog is None else {id(state) for state in catalog.protected_states()}
        retention = None if catalog is None else catalog.record()
        cached = {id(point[1]) for point in self.kept}
        for state, record in zip(self.states, records):
            record["request_active"] = id(state) in active
            record["derived_prefix_cache"] = id(state) in cached
            if catalog is not None:
                record["retained_gpu"] = id(state) in protected
            if getattr(getattr(self, "w", None), "dflash2", None) is not None:
                record["dflash2"] = self.w.dflash2.record(state)
        result = {"slots": records, "active_request_slots": len(active),
                "intentional_retained_sessions": 0 if retention is None else retention["sessions"],
                "derived_prefix_slots": len(cached),
                "derived_prefix_entries": len(self.kept),
                "migration": None if self.controller is None else
                self.controller.record()["migration"]}
        if catalog is not None:
            result.update(retained_gpu_slots=len(protected), retention=retention)
        return result

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
        catalog = getattr(self, "retention_catalog", None)
        retained_blocked = False
        for offset in range(len(self.states)):
            index = (self._migration_cursor + offset) % len(self.states)
            state = self.states[index]
            pending = getattr(state, "_history_migration", None)
            if state.history_placement.degree == target and pending is None:
                continue
            if catalog is not None and catalog.migration_blocked(state):
                retained_blocked = True
                continue  # Issued MTP rows/transfer readers cannot be trimmed or mutated.
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
        self.controller.migration_status(request_epoch=generation, pending=retained_blocked,
            **({"blocked": "retained transfer or issued MTP rows protect the pending state"} if retained_blocked else {}))

    def _timed(self, seconds: float, rows: int) -> None:
        """A round's wall time: a round alone updates its estimate, a round with a pass the seconds a row adds."""

        if rows:
            extra = max(0.0, seconds - (self.round_s or 0.0)) / rows
            self.row_s = extra if self.row_s is None else 0.7 * self.row_s + 0.3 * extra
        else:
            self.round_s = seconds if self.round_s is None else 0.7 * self.round_s + 0.3 * seconds

    def _order(self) -> list[Stream]:
        """Order overdue prompts first, then foreground, fewest rows left and stable arrival order."""

        if getattr(self, "admission_policy", "legacy") == "fifo":
            def fifo(s: Stream):
                passed = self.passed.get(s.sid, 0)
                due = passed >= FILL_GUARD
                return (not due, -passed if due else 0, s.admission_ticket)

            return sorted((stream for stream in self.filling if self._runnable(stream)), key=fifo)

        def key(s: Stream):
            passed = self.passed.get(s.sid, 0)
            due = passed >= FILL_GUARD
            return (not due, -passed if due else 0, s.background, len(s.prompt) - self.fills[s.sid][2])

        return sorted(self.filling, key=key)                         # stable: ties keep arrival order

    def _pieces(self, rows: int | None = None) -> list[tuple[Stream, int, int]]:
        """The next pass: rows from the filling prompts in ``_order``, up to ``rows`` and ENDS ending prompts."""

        pieces, room = [], self._pass_rows() if rows is None else rows
        owner = getattr(self, "media_runtime", None)
        media_room = None if owner is None else owner.profile.execution.input_rows
        # Independent prompts retain their standalone bulk GEMM/reduction shape.
        # Decode windows still batch together in the existing round/graph path.
        meta = self.w.meta
        separate = meta.get("execution_provider") == "sm70" and meta.get("sm70_profile") == "optimized"
        for s in self._order():
            start = self.fills[s.sid][2]
            n = min(len(s.prompt) - start, room)
            request = getattr(s, "_media_request", None)
            if request is not None:
                limits = getattr(self, "_media_limits", None)
                n = min(n, media_room)
                if limits is None:
                    n = request.limit(start, n)
                else:
                    expected, count = limits.get(s.sid, (start, 0))
                    if expected != start:
                        raise ValueError("frozen media rows belong to another prompt boundary")
                    n = min(n, count)
                if not n:
                    continue
            inside = [p for p in self._points(s, start, n) if p < start + n]
            if len(inside) > CUTS:                       # a piece holds CUTS kept states at most: the next one ends it
                n = inside[CUTS] - start
            ends = sum(1 for x, a, k in pieces if a + k == len(x.prompt))
            if n == 0 or (start + n == len(s.prompt) and ends == ENDS):
                break
            pieces.append((s, start, n))
            room -= n
            if request is not None:
                media_room -= n
            if separate:
                break
        return pieces

    def _pass(self) -> list[Stream]:
        """One prompt pass alone; prompts that end sample their first token, draft and join the rounds."""

        pieces = self._pieces()
        if not pieces:
            return []
        self._note_passed(pieces)
        self.pbuf.rotor_lookup = bool(pieces) and self._rotor_lookup(pieces[0][0], len(pieces))
        t0 = time.perf_counter()
        marks = self._output_marks([s for s, _, _ in pieces]) if self.controller is not None else None
        active_count = sum(not s.done for s in (*self.streams.values(), *self.filling))
        completed = False
        primary = None
        try:
            try:
                media_requests.publish(getattr(self, "media_runtime", None), pieces)
                context = getattr(self.pbuf, "input_preparation_context", None)

                def operation():
                    return self._prepare_pass(pieces)

                windows, segs, ends, cuts = operation() if context is None else context.prepare("prefill-window", operation)
                try:
                    from .media_mtp import target_inputs

                    with target_inputs(self.w, self.pbuf, windows, segs) as merged:
                        logits = compute(self.w, segs, self.pbuf, logits=bool(ends), ends=ends, cuts=cuts,
                                         **({"input_embeddings": merged} if merged is not None else {}))
                except BaseException as failure:
                    if self.controller is not None:
                        self.controller.work_observations.incomplete(failure)
                    raise
                if self.controller is not None:
                    self.controller.work_observations.target(
                        segs, prefill=True, replay_rows=sum(n for stream, _, n in pieces if stream.carry is not None))
                heads = logits[:len(ends)].clone() if ends else None
                lasts = self._absorb(pieces, segs, cuts)
                completed = True
            except Exception as exc:                         # noqa: BLE001  (these requests fail, the others go on)
                primary = exc
                if getattr(self.pbuf, "input_preparation_context", None) is not None:
                    from .input_preparation import reject_prepared

                    reject_prepared(self.pbuf, exc)
                return self._failed(pieces, exc)
            return self._joined(pieces, heads, lasts, (time.perf_counter() - t0) / len(pieces))
        except BaseException as error:
            primary = error
            raise
        finally:
            try:
                media_requests.retire_batch(getattr(self, "media_runtime", None), pieces, cancelled=not completed)
            except BaseException as cleanup:
                raise_failures(primary, [cleanup])
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

    def _prepare_pass(self, pieces):
        """Build a complete local prompt frame inside its one cohort phase."""
        self._read_ahead(pieces)
        windows = [(s.st, s.prompt[a:a + n]) for s, a, n in pieces]
        segs = stage(self.w, self.pbuf, windows)
        return windows, segs, self._end_rows(pieces, segs), self._cuts(pieces, segs)

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

        return entry_end(s.prompt) if s.draft and s.st.image_positions is None \
            and getattr(s.st, "media_profile_identity", None) is None else None

    def _points(self, s: Stream, start: int, n: int) -> list[int]:
        """The message-start and prompt-end snapshots in rows ``start`` + 1 .. ``start`` + ``n``, ascending."""

        if not s.draft or s.st.image_positions is not None or getattr(s.st, "media_profile_identity", None) is not None:
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
            if snap is not None and (s.st.compact_indexer is not None or getattr(s.st, "indexer_history_owner", None) is not None):
                snap.update(s.st.indexer_snapshot_metadata(snap["pos"], mtp_len))
        dflash = getattr(self.w, "dflash2", None)
        for (s, a, n), (st, a0, _) in zip(pieces, segs):
            commit(self.w, st, self.pbuf, n, n, at=a0,
                   **({"commit_dflash": False} if dflash is not None else {}))
        if dflash is not None:
            self._commit_dflash([(state, end - begin) for state, begin, end in segs], prefill=True)
        for s, mtp_len, tail, snap in points:            # a point that ends its piece: the state as committed
            state = snap if snap is not None else {**s.st.snapshot(), "mtp_len": mtp_len}
            if snap is None and (s.st.compact_indexer is not None or getattr(s.st, "indexer_history_owner", None) is not None):
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
                    self._capture_retained(s, last)
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
            self._take(s, [first])                          # the first token goes out before the next draft
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
                self._capture_retained(s, last)
                joined.append(s)
        return joined

    def cancel(self, streams: list[Stream]) -> list[Stream]:
        """Filling prompts whose clients left stop; each keeps the state its passes committed, so a retry of the same
        prompt resumes there rather than from its last message start. Returns them, done, for ``finish``. A state that
        fails its checks (``snapshot``, ``kv_validate_snapshot``) is not kept: that stream ends with the error and its
        slot keeps nothing, so ``finish`` frees it; the others still stop."""

        for s in streams:
            if getattr(self, "admission_policy", "legacy") == "fifo" and s.sid in self.streams:
                self._flush(s)
                s.done, s.finished = True, time.perf_counter()
                continue
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
                    if s.st.compact_indexer is not None or getattr(s.st, "indexer_history_owner", None) is not None:
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
        live = [s for s in self.streams.values() if self._runnable(s)]
        if self.filling and (not live or not self.converged):
            ended += self._fill()                      # passes alone; a prompt that ends here joins this round
            live = [s for s in self.streams.values() if self._runnable(s)]
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
        def prepare_decode():
            windows = [(s.st, [s.out[-1]] + list(s.drafts)) for s in live]
            # N-gram rows are read behind the target layers before their layer.
            return windows, stage(self.w, self.buf, windows, late=True)

        context = getattr(self.buf, "input_preparation_context", None)
        windows, segs = prepare_decode() if context is None else context.prepare("decode-window", prepare_decode)
        # a pass shares the round's forward only where their experts share a launch; else _fill ran it between rounds
        pieces, psegs = (self._pieces(self._pass_rows()) if self.filling and self.converged else []), None
        marks = self._output_marks([*live, *(s for s, _, _ in pieces)]) if self.controller is not None else None
        cuts, pwindows, pends = [], None, None
        if pieces:
            self._note_passed(pieces)
            try:
                context = getattr(self.pbuf, "input_preparation_context", None)

                def operation():
                    return self._prepare_pass(pieces)

                pwindows, psegs, pends, cuts = operation() if context is None else context.prepare("mixed-prefill-window", operation)
            except Exception as exc:                     # noqa: BLE001  (the pass's requests fail, the round goes on)
                if getattr(self.pbuf, "input_preparation_context", None) is not None:
                    from .input_preparation import reject_prepared

                    reject_prepared(self.pbuf, exc)
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
                from .media_mtp import target_inputs

                with target_inputs(self.w, self.pbuf, pwindows, psegs) as merged:
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
                self._take(s, new)
            if terminal:
                s.done, s.finished = True, time.perf_counter()
        spent = time.perf_counter() - t0
        for s, a0, keep, _new, _last in kept:
            if s.done:
                self._capture_retained(s, frame=self.buf, row=a0 + len(keep) - 1 if keep else None)
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
            capacity = getattr(s, "output_capacity", None)
            if capacity is not None:
                count = min(count, capacity - 1)
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
        def prepare_refresh():
            for s, _, _ in todo:
                st = s.st
                if st.mtp_drafted:
                    st.set_mtp_len(st.mtp_len - st.mtp_drafted)
                    st.mtp_drafted = 0
            windows = [(s.st, keep, self.buf.streams[a0:a0 + len(keep)]) for s, a0, keep in todo]
            return mtp_stage(self.mbuf, windows)

        context = getattr(self.mbuf, "input_preparation_context", None)
        segs = prepare_refresh() if context is None else context.prepare("mtp-refresh-window", prepare_refresh)
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
            for stream, _, _ in nxt:
                stream.st.begin_mtp_proposals()
            def prepare_proposal():
                windows = [(s.st, [d], self.mbuf.streams[row:row + 1]) for s, row, d in nxt]
                return mtp_stage(self.mbuf, windows)

            segs = prepare_proposal() if context is None else context.prepare("mtp-proposal-window", prepare_proposal)
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
            try:
                if getattr(s, "retention_request", None) is not None:
                    self.retire_retained_request(s)
                if s.error is not None:
                    self._drop_kept(s.st)  # A failed request does not retain deferred conditioning through prefix aliases.
                handle = getattr(s, "retention_handle", None)
                if getattr(s, "retention_captured", False):
                    if handle.role == "temporary":
                        from .prefix_policy import reuse_allowed
                        from .retained_catalog import LOCAL_PRINCIPAL, RetentionCapacity

                        def detach_host(host):
                            s.retirement_host = host
                            host.detach(s.st)
                        reusable = reuse_allowed(self.w, s.sampling, draft=s.draft, vision=s.vision,
                                                 position=s.st.pos, kv_pair=self.kv_pair)
                        if reusable:
                            try:
                                hidden = None if s.st.mtp_boundary is not None else handle.ready_handoff["pending_hc"]
                                recoveries = tuple((snapshot, tail) for _ids, owner, snapshot, tail in self.kept if owner is s.st)
                                self.retention_catalog.capture_prefix(s, principal=s.principal or LOCAL_PRINCIPAL,
                                    pending_hc=hidden, recoveries=recoveries, from_handle=handle, detach_host=detach_host)
                            except RetentionCapacity as error:
                                self._capacity_refusal(error)
                                reusable = False
                        if not reusable:
                            if handle.host is not None:
                                detach_host(handle.host)
                            self.retention_catalog.release_temporary(handle)
                            s.retention_handle = None
                            self._release_retained_states()
                    elif handle.host is not None:
                        handle.host.detach(s.st)
                    else:
                        s.st = None
                    host = getattr(s, "retirement_host", None)
                    if host is not None and host.delivery is None:
                        host.retire_backend()  # Followers have no caller/worker; this exact rank finish is their proof.
                        s.retirement_host = None
                    # RetainedHost owns the original Stream without a strong
                    # GPU link; the catalog alone keeps the completed window.
                if (not getattr(s, "retention_captured", False)
                        and not any(k[1] is s.st for k in self.kept) and all(f is not s.st for f in self.free)
                        and (getattr(self, "retention_catalog", None) is None or all(
                            state is not s.st for state in self.retention_catalog.protected_states()))):
                    self._shrink(s.st)
                    self.free.append(s.st)
            except BaseException as cleanup:
                if s.error is not None:
                    raise_failures(s.error, [cleanup])
                raise
            if s.error is not None:
                mark_error_state_retired(s.error)

    def drop(self, error=None) -> list[Stream]:
        prior, errors = tuple(self.failed_slots), []
        live = [s for s in self.streams.values() if not s.done] + self.filling
        self.filling, self.fills = [], {}
        for s in live:
            self.streams.pop(s.sid, None)
            self.held.pop(s.sid, None)
            self._drop_kept(s.st)
            try:
                if (getattr(s, "retention_handle", None) is not None
                        or getattr(s, "retention_request", None) is not None):
                    if error is None:
                        raise RuntimeError("retained backend cleanup requires its explicit owning failure outcome")
                    s.error, s.done = error, True
                    self.retire_retained_request(s)
                    if s.st is None or any(s.st is state for state in self.free):
                        continue
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

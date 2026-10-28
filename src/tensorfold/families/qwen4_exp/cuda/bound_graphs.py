"""SM70 round captures bound to existing State slots and immutable buffer owners.

The half provider uses ordinary State pointers rather than the modern kernels'
pointer tables. Views borrow each slot's caches, convolution/history and replay
scratch; only recurrent input/output has private, stable storage. This removes
mixed physical parity from graph keys without changing the State commit rule.
"""

from __future__ import annotations

import copy
from functools import partial

from .profile_construction import replay as profile_replay

import torch

from . import history_runtime
from .forward import _kv_begin, _kv_check, compute, stage, stage_late
from .mtp import mtp_compute, mtp_stage
from .state import CAND, Buffers


class BoundLane:
    def __init__(self, state, index, *, provisional_mtp=False, exact_rows=False):
        self.state, self.index = state, index
        self.exact_rows = exact_rows
        self.identity = (state.kv_identity, state.version)
        self.view = copy.copy(state)
        self.recurrent = torch.empty_like(state.rec)
        self.mtp_position = torch.empty_like(state.mtp_pos) if provisional_mtp else None
        self.keep_rows = torch.ones((1,), dtype=torch.int32, device=state.rec.device)
        self.history_rows = 1
        self.scratch = [copy.copy(scratch) for scratch in state.scratch]
        for scratch in self.scratch:
            scratch.keep_rows = self.keep_rows
        self.refresh()

    def refresh(self, rows=None, *, recurrent=True, mtp_position=None):
        state, view = self.state, self.view
        if (state.kv_identity, state.version) != self.identity:
            raise ValueError("bound graphs require their original cache storage and KV identity")
        view.__dict__.update(vars(state))
        view.rec = self.recurrent
        if self.mtp_position is not None:
            view.mtp_pos = self.mtp_position
        if mtp_position is not None and self.mtp_position is not None:
            view.mtp_len = mtp_position
            self.mtp_position.fill_(mtp_position)
        elif mtp_position is not None and mtp_position != state.mtp_len:
            raise ValueError("provisional MTP positions require a qualified bound lane")
        view.cur = [0] * len(state.cur)
        view.scratch = self.scratch
        if self.exact_rows:
            view.__dict__.pop("graph_rows", None)
        else:
            view.graph_rows = self.keep_rows
        if rows is not None:
            self.history_rows = rows
            self.keep_rows.fill_(rows)
        view.history_rows = self.history_rows
        if state.cur and any(parity != state.cur[0] for parity in state.cur):
            raise ValueError("bound graph recurrent layers must share one committed parity")
        parity = state.cur[0] if state.cur else 0
        if recurrent:
            self.recurrent[0].copy_(state.rec[parity])
        return view

    def publish(self):
        state = self.state
        parity = state.cur[0] if state.cur else 0
        state.rec[1 - parity].copy_(self.recurrent[1])


class BoundStateGraphs:
    """The SM70 implementation inside RoundGraphs; prefill and commit remain eager."""

    def __init__(self, owner):
        self.owner, self.dec = owner, owner.dec
        self.unique = self.dec.w.meta.get("history_layout") == "unique"
        self.native_owner = self.captured_owner = False
        if self.unique:
            from .multi_graphs import UNIQUE_CAPTURE_CONTRACT, UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT, UNIQUE_NATIVE_OWNER_GRAPH_CAPTURE_CONTRACT

            contract = self.dec.w.meta.get("unique_capture_contract")
            if contract not in (UNIQUE_CAPTURE_CONTRACT, UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT, UNIQUE_NATIVE_OWNER_GRAPH_CAPTURE_CONTRACT):
                raise ValueError("unique graph owners require the exact N1 piecewise capture contract")
            self.native_owner = contract in (UNIQUE_NATIVE_OWNER_CAPTURE_CONTRACT, UNIQUE_NATIVE_OWNER_GRAPH_CAPTURE_CONTRACT)
            self.captured_owner = contract == UNIQUE_NATIVE_OWNER_GRAPH_CAPTURE_CONTRACT
        self.states = tuple(self.dec.free)
        self.indices = {id(state): index for index, state in enumerate(self.states)}
        self.lanes = {}
        self.buffers = {}
        self.row_maps = {}
        self.segment_maps = {}
        self.map_events = {}
        self.bindings = {}
        self.closed = False
        self.hits = self.misses = 0
        self.greedy = None
        self.greedy_reason = "not qualified"
        self.lane_declines = 0
        from .lane_admission import prepare

        fixed = all(state.capacity == state.limit for state in self.states)
        self.lane_allocator = prepare(torch) if fixed else None
        # One cold owner-setup vote, never a hot replay/readback. All ranks
        # select the same feature even if one allocator cannot prove bounds.
        self.lane_preflight = self._agree(fixed and self.lane_allocator is not None)
        self.lane_catalogue_retired = False

    def _new_lane_bytes(self, state):
        """Pure requested extents; allocator rounding/segments are proved separately."""
        sizes = [state.rec.numel() * state.rec.element_size(), 4]
        if self.dec.w.meta.get("device_greedy_mtp_ready", False):
            sizes.append(state.mtp_pos.numel() * state.mtp_pos.element_size())
        return sizes

    def _admit_lanes(self, ordered):
        # This catalogue is cohort-symmetric: slots/invalidations follow the
        # shared decoder protocol, creations vote/charge together, and any
        # partial allocation poisons the owner before it can be replayed.
        # Existing lanes require no vote, allocator query or device readback.
        missing = []
        for state, _, _ in ordered:
            index = self.indices.get(id(state))
            if index is None or self.states[index] is not state:
                raise ValueError("a captured round must borrow one of its decoder's original State slots")
            lane = self.lanes.get(index)
            if lane is not None and lane.identity != (state.kv_identity, state.version):
                raise ValueError("bound graphs require their original cache storage and KV identity")
            if state.cur and any(parity != state.cur[0] for parity in state.cur):
                raise ValueError("bound graph recurrent layers must share one committed parity")
            if lane is None:
                missing.append(state)
        if not missing:
            return not self.lane_catalogue_retired
        if not self.lane_preflight:
            return True                    # Movable/unsupported geometry retains C16 behavior.
        if self.lane_catalogue_retired:
            return False                   # Common force/recovery retirement never rearms this owner.
        sizes = [size for state in missing for size in self._new_lane_bytes(state)]
        bounds = self.lane_allocator.bounds(sizes)
        remaining = None if self.owner.spare is None else self.owner.spare()
        allowed = (bounds is not None and self.lane_allocator.unchanged(torch)
                   and self.owner.room_bytes + bounds[0] <= self.owner.budget
                   and (remaining is None or remaining >= max(bounds)))
        if not self._agree(allowed):
            self.lane_declines += 1
            self.owner.stopped = "new bound lane staging does not fit its graph budget and required service memory"
            return False
        return True

    def _lane(self, state):
        index = self.indices.get(id(state))
        if index is None or self.states[index] is not state:
            raise ValueError("a captured round must borrow one of its decoder's original State slots")
        lane = self.lanes.get(index)
        if lane is None:
            before = torch.cuda.memory_allocated(self.dec.w.device)
            lane = self.lanes[index] = object.__new__(BoundLane)
            lane.__init__(state, index, provisional_mtp=bool(
                self.dec.w.meta.get("device_greedy_mtp_ready", False)), exact_rows=self.unique)
            self._resident(max(0, torch.cuda.memory_allocated(self.dec.w.device) - before))
        return lane

    def _resident(self, size):
        self.owner.bytes += size
        self.owner.room_bytes += size
        if not self._agree(self.owner.room_bytes <= self.owner.budget):
            raise MemoryError("bound graph persistent state exceeds its shared rank budget")

    def _buffer(self, kind):
        result = self.buffers.get(kind)
        if result is None:
            before = torch.cuda.memory_allocated(self.dec.w.device)
            source = self.dec.buf if kind == "verify" else self.dec.mbuf
            rows, segments = source.rows, len(self.states)
            if self.unique:
                from .multi_graphs import UNIQUE_CAPTURE_MAX_ROWS

                rows, segments = min(rows, UNIQUE_CAPTURE_MAX_ROWS), 1
            result = self.buffers[kind] = Buffers(self.dec.w, rows, self.dec.capacity,
                                                 moe_prefill=source.moe.plan.prefill,
                                                 owner_role="mtp" if kind == "mtp" else None)
            if self.native_owner:
                role = "mtp" if kind == "mtp" else "decode"
                if (self.dec.w.meta.get("owner_workspaces", {}).get(role) is not source.history_owner
                        or result.history_owner is not source.history_owner):
                    raise ValueError("owner-piece graphs require their canonical prepared attention workspace")
                for name in ("pa", "attn_o"):
                    canonical, private = getattr(source, name), getattr(result, name)
                    if (canonical.shape[0] < rows or canonical.shape[1:] != private.shape[1:]
                            or canonical.dtype != private.dtype or canonical.device != private.device
                            or not canonical.is_contiguous()):
                        raise ValueError("owner-piece activation alias differs from its canonical layout")
                    # Source Buffers allocate these tensors once and serving
                    # never resizes/rebinds them. Views retain their storage;
                    # graph teardown precedes canonical owner teardown.
                    setattr(result, name, canonical[:rows])
            result.dspark_graph = kind == "verify" and getattr(self.dec.w, "dspark", None) is not None
            if result.dspark_graph:
                self.dec.w.dspark.prepare_graph_context(result, self)
            # HC plans are optional modern-provider operators; the half branch
            # uses the same prepared linears as its ordinary eager buffers.
            result.hc_plans = None
            result.dynamic_last = True
            # Whole-cohort reductions follow the real eager row geometry even
            # when their captured buffers include speculative padded rows.
            result.graph_rows = torch.empty((1,), dtype=torch.int32, device=self.dec.w.device)
            from tensorfold.cuda.sm70 import family_ops

            if family_ops._NATIVE_FAMILY and family_ops._NATIVE_ATTENTION:
                from tensorfold.cuda.sm70.attention_native import Workspace

                c = self.dec.w.cfg
                result.attn.native_workspace = Workspace(result.attn.out.shape[0], c.heads, c.head_dim,
                                                        self.dec.capacity, self.dec.w.device)
            self.map_events[kind] = [torch.cuda.Event(), False]
            if self.unique:
                # _plan admits one compact lane at its exact width: its row
                # inverse is the identity and its only segment index is zero.
                # These maps never change, including across State-slot reuse.
                self.row_maps[kind] = (None, torch.arange(rows, dtype=torch.int64, device=self.dec.w.device))
                self.segment_maps[kind] = (None, torch.zeros(segments, dtype=torch.int64, device=self.dec.w.device))
                self.map_events[kind][0].record(torch.cuda.current_stream(self.dec.w.device))
                self.map_events[kind][1] = True
            else:
                self.row_maps[kind] = (torch.empty(rows, dtype=torch.int64, pin_memory=True),
                                       torch.empty(rows, dtype=torch.int64, device=self.dec.w.device))
                self.segment_maps[kind] = (torch.empty(segments, dtype=torch.int64, pin_memory=True),
                                           torch.empty(segments, dtype=torch.int64, device=self.dec.w.device))
            self._resident(max(0, torch.cuda.memory_allocated(self.dec.w.device) - before))
        return result

    def _maps(self, kind, rows, segments):
        event = self.map_events[kind]
        if self.unique:
            device, sdevice = self.row_maps[kind][1], self.segment_maps[kind][1]
            # A different consumer stream still joins the constructor's writes;
            # no pinned mirror is mutated or borrowed by a repeated upload.
            torch.cuda.current_stream(device.device).wait_event(event[0])
            return device[:len(rows)], sdevice[:len(segments)]
        if event[1]:
            event[0].synchronize()             # pinned mirrors remain borrowed until their last upload completes
        host, device = self.row_maps[kind]
        host[:len(rows)].numpy()[:] = rows
        device[:len(rows)].copy_(host[:len(rows)], non_blocking=True)
        shost, sdevice = self.segment_maps[kind]
        if segments:
            shost[:len(segments)].numpy()[:] = segments
            sdevice[:len(segments)].copy_(shost[:len(segments)], non_blocking=True)
        event[0].record()
        event[1] = True
        return device[:len(rows)], sdevice[:len(segments)]

    def _width(self, kind, segs):
        if self.unique and len(segs) == 1:
            return segs[0][2] - segs[0][1]
        # Chained MTP steps append one row per participant. Keep their physical
        # expert/attention work at that exact width; absorbs and main verifies
        # retain the admitted speculative width in the same bounded pool.
        return 1 if kind == "mtp" and segs and all(a1 - a0 == 1 for _, a0, a1 in segs) else self.dec.depth + 1

    def _plan(self, kind, segs, mtp_positions=None, *, device_tokens=None):
        from .multi_graphs import UNIQUE_CAPTURE_MAX_ROWS

        if self.unique and (len(segs) != 1 or mtp_positions is not None or device_tokens is not None
                            or not 1 <= segs[0][2] - segs[0][1] <= min(self.dec.depth + 1, UNIQUE_CAPTURE_MAX_ROWS)):
            return None
        width = self._width(kind, segs)
        expected = 0
        for _state, a0, a1 in segs:
            if a0 != expected or not 1 <= a1 - a0 <= width:
                raise ValueError("bound graph segments require compact rows within the admitted per-lane width")
            expected = a1
        if not expected:
            raise ValueError("bound graphs require at least one live row")
        ordered = sorted(segs, key=lambda segment: self.indices.get(id(segment[0]), -1))
        counts = tuple(a1 - a0 for _, a0, a1 in ordered)
        if mtp_positions is not None:
            if (kind != "mtp" or len(mtp_positions) != len(segs)
                    or any(type(value) is not int or value < 0 for value in mtp_positions)):
                raise ValueError("provisional MTP positions require one nonnegative integer per segment")
            supplied = {id(state): value for (state, _, _), value in zip(segs, mtp_positions)}
            positions = [supplied[id(state)] for state, _, _ in ordered]
        else:
            positions = [state.pos if kind == "verify" else state.mtp_len for state, _, _ in ordered]
        if device_tokens is not None:
            source = self.dec.mbuf
            if (kind != "mtp" or len(segs) != 1 or counts != (1,) or device_tokens.shape != (1,)
                    or device_tokens.dtype != torch.int32 or device_tokens.device != source.ids.device
                    or not device_tokens.is_contiguous()):
                raise ValueError("device MTP tokens require one contiguous int32 token on the bound device")
        from tensorfold.cuda.sm70.attention_qualification import allows

        end = max(position + width for position in positions)
        context = None if self.unique else max(256, 1 << (end - 1).bit_length())
        if (self.unique and any(position + width > state.logical_context
                                for (state, _, _), position in zip(ordered, positions))
                or not self.unique and not allows(self.dec.w, positions, width, context)):
            # Preserve invalid-owner/state errors before an optional decline;
            # no allocation, collective, pointer binding or staging occurs.
            for state, _, _ in ordered:
                index = self.indices.get(id(state))
                if index is None or self.states[index] is not state:
                    raise ValueError("a captured round must borrow one of its decoder's original State slots")
                lane = self.lanes.get(index)
                if lane is not None and lane.identity != (state.kv_identity, state.version):
                    raise ValueError("bound graphs require their original cache storage and KV identity")
                if state.cur and any(parity != state.cur[0] for parity in state.cur):
                    raise ValueError("bound graph recurrent layers must share one committed parity")
            return None
        if not self._admit_lanes(ordered):
            return None
        lanes = [self._lane(state) for state, _, _ in ordered]
        key = (tuple((lane.index, lane.identity) for lane in lanes), width, context)
        # source compact rows in canonical lane order, and their inverse for
        # publishing the captured values back into the ordinary round buffers.
        inverse = [None] * sum(counts)
        for lane, (_, a0, a1) in enumerate(ordered):
            inverse[a0:a1] = range(lane * width, lane * width + a1 - a0)
        return key, lanes, ordered, context, counts, inverse

    def _prepare(self, kind, segs, source, *, mtp_positions=None, device_tokens=None):
        if kind != "mtp" and device_tokens is not None:
            raise ValueError("device token staging applies only to MTP steps")
        plan = self._plan(kind, segs, mtp_positions, device_tokens=device_tokens)
        if plan is None:
            return None
        key, lanes, ordered, context, counts, inverse = plan
        b = self._buffer(kind)
        order = [self.indices[id(state)] for state, _, _ in segs]
        canonical = sorted(order)
        indices, segment_indices = self._maps(kind, inverse, [canonical.index(index) for index in order])
        b.graph_rows.fill_(len(inverse))
        tokens = source.ids_host[:segs[-1][2]].tolist()
        width = key[1]
        recurrent = kind == "verify" or self.dec.w.mtp.layer.linear
        positions = ({id(state): value for (state, _, _), value in zip(segs, mtp_positions)}
                     if mtp_positions is not None else {})
        windows = [(lane.refresh(a1 - a0, recurrent=recurrent,
                                 mtp_position=positions.get(id(lane.state), lane.state.mtp_len) if kind == "mtp" else None),
                    tokens[a0:a1] + [0] * (width - (a1 - a0)))
                   for lane, (_, a0, a1) in zip(lanes, ordered)]
        if kind == "verify":
            bound = stage(self.dec.w, b, windows, late=True)
            # Host PLE rows are prepared before the full bound-state capture.
            # Their GPU projections and State-owned convolution are captured.
            if b.ple_late is not None:
                stage_late(self.dec.w, b)
            from .prefix_policy import OWNER_TARGET_GREEDY_PROFILE, OWNER_TARGET_GUARD_PROFILE, OWNER_TARGET_PRELUDE_PROFILE, OWNER_ATTENTION_GRAPHS_PROFILE

            compact_target = self.dec.w.meta.get("sm70_profile") in (OWNER_TARGET_GREEDY_PROFILE, OWNER_TARGET_GUARD_PROFILE, OWNER_TARGET_PRELUDE_PROFILE, OWNER_ATTENTION_GRAPHS_PROFILE)
            run = partial(compute, self.dec.w, bound, b, context=context, check=False,
                          **({"gather_candidates": False} if compact_target else {}))
        else:
            padded = b.mtp_in[:len(lanes) * width]
            padded.zero_()
            padded.index_copy_(0, indices, source.mtp_in[:len(inverse)])
            windows = [(view, ids, padded[index * width:(index + 1) * width])
                       for index, (view, ids) in enumerate(windows)]
            bound = mtp_stage(b, windows)
            if device_tokens is not None:
                if (len(segs) != 1 or counts != (1,) or device_tokens.shape != (1,)
                        or device_tokens.dtype != torch.int32 or device_tokens.device != b.ids.device
                        or not device_tokens.is_contiguous()):
                    raise ValueError("device MTP tokens require one contiguous int32 token on the bound device")
                b.ids[:1].copy_(device_tokens)
            b.last_host[:len(lanes)].numpy()[:] = [index * width + count - 1 for index, count in enumerate(counts)]
            b.last[:len(lanes)].copy_(b.last_host[:len(lanes)], non_blocking=True)
            b.staged.record()
            run = partial(mtp_compute, self.dec.w, bound, b, context=context, check=False)
        b.rotor_lookup = False
        b.gdn_tables = b.attn_step = b.split = None
        return key, lanes, b, bound, run, counts, indices, segment_indices

    def _agree(self, allowed):
        comm = self.dec.w.comm
        if comm is None:
            return bool(allowed)
        send = torch.tensor([int(allowed)], dtype=torch.int64, device=self.dec.w.device)
        receive = torch.empty(comm.world, dtype=torch.int64, device=send.device)
        comm.all_gather(send, receive)
        return bool(receive.min().item())

    def _capture(self, kind, key, lanes, b, segs, run):
        owner = self.owner
        allowed = owner.room_bytes < owner.budget and owner._has_room()
        if not self._agree(allowed):
            return None
        if self.unique:
            # Attention stays eager between graph pieces. Warming its host
            # operation here would leave an unresolved compact-indexer span
            # before the actual replay. Initialize the omitted output and warm
            # only captured work; the ordinary synthetic request warmed attention.
            # Warm the actual capture stream after joining staged inputs.
            # Its normal State guards join real transfer events before capture;
            # the serialized coordinator preserves that frontier through run().
            _kv_begin(segs)  # Surface an existing poisoned transfer before submitting a new stream dependency.
            owner.side.wait_stream(torch.cuda.current_stream(self.dec.w.device))
            with torch.cuda.stream(owner.side):
                b.attn_o.zero_()
                b.split = lambda _insert: None
                try:
                    run()
                finally:
                    b.split = None
                _kv_check(segs, comm=self.dec.w.comm)
        else:
            run()                                  # load kernels and exact launch geometry before capture
            _kv_check(segs, comm=self.dec.w.comm)
        torch.cuda.synchronize(self.dec.w.device)
        from .multi_graphs import free_bytes

        before = free_bytes()
        pieces = owner._pieces(run, b, split=self.unique)
        torch.cuda.synchronize(self.dec.w.device)
        took = max(0, before - free_bytes())
        allowed = owner.room_bytes + took <= owner.budget
        if not self._agree(allowed):
            # An unadmitted private pool cannot be credited as free memory or
            # silently reused by an eager continuation. Retain it explicitly.
            owner.retain_failure(MemoryError("bound graph capture exceeded the shared rank budget"))
            self.bindings[(kind, key)] = (pieces, tuple(lanes))
            raise owner.failure
        owner.bytes += took
        owner.room_bytes += took
        owner.measured.append({"key": [kind, repr(key)], "pieces": len(pieces.graphs), "bytes": took})
        self.bindings[(kind, key)] = (pieces, tuple(lanes))
        target = owner.verify if kind == "verify" else owner.mtp
        target[key] = pieces
        return pieces

    def _account(self, previous):
        if not self.owner.admitted:
            return
        self.owner._account_pool()
        from .multi_graphs import service_live_bytes

        self.dec.memory_gate.room -= self.owner.room_bytes - previous
        self.dec.memory_gate.live = service_live_bytes(self.owner)
        self.dec.graph_record = {**self.dec.graph_record, **self.owner.record(),
                                 "bound_hits": self.hits, "bound_misses": self.misses}

    def forward(self, kind, segs, *, mtp_positions=None, device_tokens=None):
        try:
            return self._forward(kind, segs, mtp_positions=mtp_positions, device_tokens=device_tokens)
        except BaseException as error:
            self.owner.retain_failure(error)
            self.dec._failed_graphs, self.dec._memory_error = self.owner, error
            raise

    def _forward(self, kind, segs, *, mtp_positions=None, device_tokens=None):
        if self.closed:
            raise RuntimeError("bound graph owner is closed")
        source = self.dec.buf if kind == "verify" else self.dec.mbuf
        if self.native_owner and (len(segs) != 1 or self.dec.live() > 1):
            return None  # All other cohorts keep their ordinary eager validation/work, without private staging.
        if self.unique and not self.native_owner and (source.owner_kernel_family != "torch-reference"
                            or source.owner_graph_family != "uncaptured"):
            return None
        if source.rotor_lookup or any(state.image_positions is not None for state, _, _ in segs):
            return None
        width = self._width(kind, segs)
        if len(segs) * width > source.rows:
            # Graph padding consumes real storage and work. A compact eager
            # microbatch can fit while its full-width captured lanes do not.
            return None
        positions = mtp_positions if mtp_positions is not None else [
            state.pos if kind == "verify" else state.mtp_len for state, _, _ in segs]
        if len(positions) != len(segs) or any(type(value) is not int or value < 0 for value in positions):
            raise ValueError("bound graph positions require one nonnegative integer per segment")
        if any(position + width > state.capacity for (state, _, _), position in zip(segs, positions)):
            return None
        if self.native_owner:
            from .owner_execution import KERNEL_FAMILY, REFERENCE_KERNEL, UNCAPTURED, GRAPH_FAMILY, FULL_COLLECTIVE
            from .compact_indexer import validate_target_indexer_frame, validate_mtp_indexer_frame

            attention = source.history_owner
            if getattr(self, "captured_owner", False):
                manager = source.owner_graphs
                reference_setup = (source.owner_kernel_family == REFERENCE_KERNEL
                    and source.owner_graph_family == UNCAPTURED and manager is None
                    and attention.native_states is None)
                captured_setup = (source.owner_kernel_family == KERNEL_FAMILY
                    and source.owner_graph_family == GRAPH_FAMILY and manager is not None
                    and manager.workspace is attention and manager.prepared
                    and not manager.closed and manager.failure is None)
                if not (reference_setup or captured_setup) or attention.native_collective != FULL_COLLECTIVE:
                    raise ValueError("captured owner pieces require their prepared canonical full8 attention graphs")
            elif (source.owner_kernel_family not in (REFERENCE_KERNEL, KERNEL_FAMILY)
                    or source.owner_graph_family != UNCAPTURED or source.owner_graphs is not None
                    or attention.native_collective != FULL_COLLECTIVE):
                raise ValueError("owner-piece graphs require canonical eager full8 attention")
            if (source.owner_kernel_family == KERNEL_FAMILY and
                    (attention.native_states is None or any(id(state) not in attention.native_states for state, _, _ in segs))):
                raise RuntimeError("owner-piece attention requires original prepared State owners")
            validate = validate_mtp_indexer_frame if kind == "mtp" else validate_target_indexer_frame
            validate(source, segs)  # Before any graph may mutate recurrent/convolution/cache state.
            attention.wait_native_reuse()  # The shared activation producer cannot overtake prior readers.
        runtime = getattr(self.dec.w, "dspark", None) if kind == "verify" else None
        if runtime is not None:
            runtime.check_graph(segs)
        previous = self.owner.room_bytes
        try:
            prepared = self._prepare(
                kind, segs, source, mtp_positions=mtp_positions, device_tokens=device_tokens)
        except BaseException as error:
            self.owner.retain_failure(error)
            self.dec._failed_graphs, self.dec._memory_error = self.owner, error
            raise
        if prepared is None:
            self.misses += 1
            return None
        key, lanes, b, bound, run, rows, index, segment_indices = prepared
        if runtime is not None:
            runtime.check_graph(segs, buffers=b)
        pieces = (self.owner.verify if kind == "verify" else self.owner.mtp).get(key)
        if pieces is None:
            self.misses += 1
            try:
                pieces = self._capture(kind, key, lanes, b, bound, run)
            except BaseException as error:
                self.owner.retain_failure(error)
                self.dec._failed_graphs, self.dec._memory_error = self.owner, error
                raise
            if pieces is None:
                self._account(previous)
                return None
            self._account(previous)
        else:
            self.hits += 1
        archive = getattr(self.dec.w, "meta", {}).get("indexer_history") == "ram"
        if archive:
            # Bound views carry provisional MTP positions and logical counts;
            # their owners are shared with the original States, not duplicated.
            history_runtime.begin(bound, prefill=b.prefill, mtp=kind == "mtp", counts=rows)
        _kv_begin(segs)
        if self.unique:
            if self.native_owner:
                self.owner._replay_pieces(pieces, bound, b, mtp=kind == "mtp", owner_frame=(segs, source))
            else:
                self.owner._replay_pieces(pieces, bound, b, mtp=kind == "mtp")
        else:
            for graph in pieces.graphs:
                profile_replay(graph, self.dec.w.comm)
        if archive:
            history_runtime.submitted(bound, prefill=b.prefill, mtp=kind == "mtp")
        # Match RoundGraphs' ordinary contract: status remains pending for the
        # caller's mandatory check, after exports and before sampling/publication.
        if kind == "verify":
            for lane in lanes:
                lane.publish()
        n = index.numel()
        torch.index_select(b.streams, 0, index, out=source.streams[:n])
        if kind == "verify":
            torch.index_select(b.proj, 1, index, out=source.proj[:, :n])
            torch.index_select(b.ple_nrow, 0, index, out=source.ple_nrow[:n])
            torch.index_select(b.logits, 0, index, out=source.logits[:n])
            from .prefix_policy import OWNER_TARGET_GREEDY_PROFILE, OWNER_TARGET_GUARD_PROFILE, OWNER_TARGET_PRELUDE_PROFILE, OWNER_ATTENTION_GRAPHS_PROFILE

            if (self.dec.w.comm is not None
                    and self.dec.w.meta.get("sm70_profile") not in (OWNER_TARGET_GREEDY_PROFILE, OWNER_TARGET_GUARD_PROFILE, OWNER_TARGET_PRELUDE_PROFILE, OWNER_ATTENTION_GRAPHS_PROFILE)):
                packet = 2 * CAND + 1
                world, physical = self.dec.w.comm.world, len(lanes) * width
                gathered = b.cand_all[:world * physical * packet].view(world, physical, packet)
                compact = source.cand_all[:world * n * packet].view(world, n, packet)
                torch.index_select(gathered, 1, index, out=compact)
            if runtime is not None:
                runtime.stage_graph(source, b, segs, index)
            return source.logits[:n]
        # MTP returns one head row per segment; restore the caller's segment order.
        n = len(segs)
        result = source.logits.view(-1)[:n * pieces.out.shape[1]].view(n, pieces.out.shape[1])
        torch.index_select(pieces.out, 0, segment_indices, out=result)
        if self.dec.w.comm is not None:
            packet, world = 2 * CAND + 1, self.dec.w.comm.world
            gathered = b.cand_all[:world * n * packet].view(world, n, packet)
            compact = source.cand_all[:world * n * packet].view(world, n, packet)
            torch.index_select(gathered, 1, segment_indices, out=compact)
        return result

    def mtp_ready(self, state, rows, position):
        """Pure lookup before a device handoff; a miss must use ordinary staging."""
        if self.unique:
            return False  # This contract preserves the ordinary sampling/proposal protocol.
        index = self.indices.get(id(state))
        lane = self.lanes.get(index)
        if (self.closed or lane is None or self.owner.failure is not None
                or lane.mtp_position is None
                or lane.state is not state or lane.identity != (state.kv_identity, state.version)
                or type(position) is not int or position < 0):
            return False
        width = 1 if rows == 1 else self.dec.depth + 1
        if not 1 <= rows <= width or position + width > state.capacity:
            return False
        context = max(256, 1 << (position + width - 1).bit_length())
        from tensorfold.cuda.sm70.attention_qualification import allows

        if not allows(self.dec.w, [position], width, context):
            return False
        return (((lane.index, lane.identity),), width, context) in self.owner.mtp

    def warm(self, states):
        """Prepare representative full-width keys; later exact shapes share the bounded pool."""
        # This owner serves every admitted key. Reserve it before optional
        # captures consume the shared allowance or transient warm-up headroom.
        if not self.unique:
            self._prepare_greedy()
        runtime = getattr(self.dec.w, "dspark", None)
        if runtime is not None:
            runtime.prepare_proposal_graphs(self)
        width = self.dec.depth + 1
        if self.unique:
            from .multi_graphs import UNIQUE_CAPTURE_MAX_ROWS

            width = min(width, UNIQUE_CAPTURE_MAX_ROWS)
        maximum = 1 if self.unique else min(len(states), self.dec.buf.rows // width)
        for n in range(1, maximum + 1):
            active = list(reversed(states))[:n]
            for rows in range(1, width + 1) if self.unique else (width,):
                segs = stage(self.dec.w, self.dec.buf, [(state, [0] * rows) for state in active], late=True)
                self.forward("verify", segs)
                runtime = getattr(self.dec.w, "dspark", None)
                if runtime is not None:
                    # Warm-up never commits target positions. Validate its frame,
                    # then discard independently staged draft context before the
                    # next synthetic key borrows these same slots.
                    _kv_check(segs, comm=self.dec.w.comm)
                    runtime.discard_graph(segs)
                if self.unique:
                    _kv_check(segs, comm=self.dec.w.comm)
                    for state in active:
                        state.reset(self.dec.w, new_sequence=False)
                if self.dec.mbuf is not None:
                    windows = [(state, [0] * rows, self.dec.buf.streams[a0:a1]) for state, a0, a1 in segs]
                    mseg = mtp_stage(self.dec.mbuf, windows)
                    self.forward("mtp", mseg)
                    if self.unique:
                        _kv_check(mseg, comm=self.dec.w.comm)
                        for state in active:
                            state.reset(self.dec.w, new_sequence=False)
                    elif width > 1:
                        singles = [(state, [0], self.dec.buf.streams[a0:a0 + 1]) for state, a0, _ in segs]
                        self.forward("mtp", mtp_stage(self.dec.mbuf, singles))

        if runtime is not None:
            runtime.warm_proposals(states)

    def _prepare_greedy(self):
        # Every rank participates, including ranks whose metadata declines the
        # capability. A differing preparation flag cannot split collectives.
        requested = (bool(self.dec.w.meta.get("device_greedy_mtp_ready", False))
                     and self.dec.depth > 0 and self.dec.w.comm is not None
                     and self.dec.w.mtp is not None and not self.dec.w.mtp.layer.linear)
        if not self._agree(requested):
            return
        from .device_greedy import GreedyHandoff, workspace_bytes

        device, host = workspace_bytes(self.dec._draft_plan.width, self.dec.depth, self.dec.w.comm.world)
        # Retained host bytes conservatively consume the same already-admitted
        # graph ceiling. Actual device allocation is measured after creation.
        allowed = self.owner.room_bytes + device + host + 32 * 512 <= self.owner.budget
        if not self._agree(allowed and self.owner._has_room()):
            self.greedy_reason = "workspace exceeds the captured-round budget"
            return
        before = torch.cuda.memory_allocated(self.dec.w.device)
        workspace = object.__new__(GreedyHandoff)
        self.greedy = workspace                 # own partial construction before any fallible allocation
        workspace.__init__(self)
        self._resident(max(0, torch.cuda.memory_allocated(self.dec.w.device) - before) + host)
        self.greedy_reason = None

    def invalidate(self, state):
        index = self.indices.get(id(state))
        if index is None:
            return
        if self.lane_preflight:
            # Fixed-limit slots never resize during ordinary service. Explicit
            # force/recovery retirement is a cohort control operation; disable
            # this owner's optional catalogue permanently on that operation.
            self.lane_catalogue_retired = True
        torch.cuda.synchronize(self.dec.w.device)
        runtime = getattr(self.dec.w, "dspark", None)
        if runtime is not None:
            runtime.invalidate_proposals(state)
        for (kind, key), (pieces, lanes) in tuple(self.bindings.items()):
            if any(lane.state is state for lane in lanes):
                for graph in pieces.graphs:
                    graph.reset()
                del self.bindings[(kind, key)]
                (self.owner.verify if kind == "verify" else self.owner.mtp).pop(key, None)
        self.lanes.pop(index, None)

    def close(self):
        for state in self.states:
            self.invalidate(state)
        runtime = getattr(self.dec.w, "dspark", None)
        if runtime is not None:
            runtime.retire_proposals(self)
            for buffer in self.buffers.values():
                if getattr(buffer, "dspark_graph", False):
                    runtime.retire_graph(buffer)
        self.buffers.clear()
        self.row_maps.clear()
        self.segment_maps.clear()
        self.map_events.clear()
        self.greedy = None
        self.closed = True

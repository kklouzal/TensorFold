"""Exact-row SM70 graph pieces with live unique-history handoffs between them.

The graph borrows the decoder's original buffers. Expert and linear GPU
operations are captured, while unique attention and host PLE
staging execute against the current live State between pieces. Captures retain
no direct State pointer, padded row, coordinate grant or host position. The
private native-GDN variant reads freshly staged registered pointers through
fixed descriptor buffers; the original variant keeps recurrent chains live.
Only idle startup creates graph keys; an unseen serving key runs eagerly.
"""

from functools import partial
from copy import deepcopy
import json
import os

import torch

from tensorfold.cleanup import raise_failures

from .forward import _kv_begin, _kv_check, compute, gdn_chain, ple_conv_streams, stage, stage_late
from .mtp import mtp_compute, mtp_stage
from .profile_construction import replay


class ExactStateGraphs:
    def __init__(self, owner):
        self.owner, self.dec = owner, owner.dec
        self.states = tuple(self.dec.free)
        self.indices = {id(state): index for index, state in enumerate(self.states)}
        self.bindings = {}
        self.closed = self.preparing = False
        self.hits = self.misses = self.owner_inserts = 0
        self.serving_hits = {"verify": 0, "mtp": 0}
        self.serving_misses = {"verify": 0, "mtp": 0}
        self.greedy = None
        diagnostic = os.environ.get("TENSORFOLD_EXACT_GRAPH_PIECE_DIAGNOSTIC", "0")
        if diagnostic not in ("0", "1"):
            raise ValueError("exact graph piece diagnostic must be 0 or 1")
        self.piece_diagnostic = diagnostic == "1"
        self.native_gdn_capture = bool(getattr(self.dec, 'native_gdn_capture', False))
        self.value8_prepared = getattr(self.dec, 'gdn_value8_prepared', None)
        self.value8_bindings = {}
        workspace = getattr(self.dec, 'gdn_value8_workspace', None)
        if self.value8_prepared is not None and workspace is None:
            raise ValueError('value8 exact capture requires its existing decoder Q workspace')
        self.value8_workspace_record = (None if self.value8_prepared is None else {
            'source': 'MultiDecoder.gdn.q', 'reused_bytes': workspace.q.numel()*workspace.q.element_size(),
            'additional_device_bytes': 0, 'scope': 'target-exact-capture-only; eager/prefill/MTP retain incumbent'})
        self.requested_policy = deepcopy(getattr(self.dec, 'exact_graph_policy', None))
        if self.requested_policy is not None:
            self.requested_policy['keys'] = tuple(tuple(key) for key in self.requested_policy['keys'])
        self.available_keys = self.requested_keys = ()
        self.captured_keys = []
        self.selected_keys = None
        self.declined_keys = {}

    def _catalogue(self, states):
        """Original finite shape order; selection never changes live capacity."""
        width = self.dec.depth + 1
        maximum = min(len(states), self.dec.buf.rows // width)
        counts = sorted({value for value in (1, 2, 4, 8, 16, 24, maximum) if 0 < value <= maximum})
        shapes = tuple((count, rows) for count in counts
                       for rows in (range(1, width + 1) if count == 1 else sorted({1, width})))
        available = tuple((role, count * rows, count) for count, rows in shapes
                          for role in (('verify', 'mtp') if self.dec.mbuf is not None else ('verify',)))
        requested = available if self.requested_policy is None else self.requested_policy['keys']
        if not requested or not set(requested).issubset(available):
            raise ValueError('exact graph subset contains a key outside the actual startup catalogue')
        self.available_keys, self.requested_keys = available, tuple(requested)
        self.selected_keys = frozenset(requested)
        self.declined_keys = {key: 'excluded by paid-subset policy' for key in available if key not in self.selected_keys}
        return shapes

    def _native_frame(self, segs, source, *, capture=False):
        """Startup-only publication; serving uses its decode preparation phase."""
        if not self.preparing:
            raise RuntimeError('exact GDN warm descriptor publication requires idle startup')
        def prepare():
            frame = self.dec.round_tables.native_frame(segs, source, allow_single=True)
            if frame is None:
                raise ValueError('fixed exact GDN registration changed during startup')
            source.gdn_tables = frame
            return frame
        context = getattr(source, 'input_preparation_context', None)
        frame = prepare() if context is None else context.prepare('exact-gdn-warm-frame', prepare)
        source.gdn_capture_ready_operands = None
        source.gdn_value8_ready = None
        rows = segs[-1][2]
        if capture:
            # NativeTables.operands validates current frame and actor stream.
            # Capture consumes these cold views without asking a frame created
            # on that actor stream to validate against the separate side stream.
            if self.value8_prepared is not None:
                source.gdn_value8_ready = tuple(self.value8_prepared.batch_prepare(frame, layer,
                    source.proj[layer, :rows], source.gout[:rows], source.gxs[:rows], self.dec.gdn_value8_workspace)
                    for layer in range(self.dec.round_tables.lin))
            else:
                source.gdn_capture_ready_operands = tuple(frame.operands(layer,
                    source.proj[layer, :rows], source.gout[:rows], source.gxs[:rows])
                    for layer in range(self.dec.round_tables.lin))
        elif self.dec.round_tables.lin:
            frame.operands(0, source.proj[0, :rows], source.gout[:rows], source.gxs[:rows])
        return frame

    def _key(self, kind, segs):
        if kind not in ("verify", "mtp") or not segs:
            raise ValueError("exact graph key requires a target or MTP frame")
        source = self.dec.buf if kind == "verify" else self.dec.mbuf
        if source is None:
            raise ValueError("MTP graphs require the selected MTP buffers")
        cursor = 0
        for state, first, last in segs:
            index = self.indices.get(id(state))
            if index is None or self.states[index] is not state:
                raise ValueError("exact graphs require their original decoder State slots")
            if (type(first) is not int or type(last) is not int or first != cursor
                    or not first < last <= source.rows):
                raise ValueError("exact graph segments must cover compact nonempty live rows")
            if getattr(state, "graph_rows", None) is not None:
                raise ValueError("unique graph pieces cannot use padded State envelopes")
            cursor = last
        # Every operation consuming State storage executes between pieces.
        # Only these dimensions affect the captured GPU program; the MTP head
        # has one output row per participant and uses its staged last-row map.
        return cursor, len(segs)

    def _insert(self, insert, segs, buffers):
        kind, layer, *arguments = insert
        if kind == "owner_attention":
            from .history_owner import attention_layer

            rows, mtp = arguments
            attention_layer(layer, self.dec.w, segs, buffers, rows, mtp=mtp)
            self.owner_inserts += 1
        elif kind == "ple_conv":
            ple_conv_streams(layer, self.dec.w, segs, buffers)
        elif kind == "gdn_chain":
            gdn_chain(layer, self.dec.w, segs, buffers, arguments[0])
        elif kind == "ple_stage":
            if buffers.ple_late is not None:
                stage_late(self.dec.w, buffers)
        else:
            raise ValueError("unsupported exact graph handoff")

    def _agree(self, value):
        comm = self.dec.w.comm
        if comm is None:
            return bool(value)
        send = torch.tensor([int(value)], dtype=torch.int64, device=self.dec.w.device)
        result = torch.empty(comm.world, dtype=torch.int64, device=send.device)
        comm.all_gather(send, result)
        return bool(result.min().item())

    def _capture(self, kind, key, segs, source):
        if not self.preparing:
            raise RuntimeError("exact graph capture is restricted to idle startup")
        if self.selected_keys is None or (kind, *key) not in self.selected_keys:
            raise RuntimeError('exact graph capture requires a selected startup catalogue key')
        owner = self.owner
        if not self._agree(owner.room_bytes < owner.budget and owner._has_room()):
            self.declined_keys[(kind, *key)] = owner.stopped or 'no admitted graph room before capture'
            return None
        run = partial(compute if kind == "verify" else mtp_compute,
                      self.dec.w, segs, source, check=False)
        run()
        # The eager warm program performed real tentative appends. They cannot
        # be overwritten by a replay at the same frontier: retire and reset the
        # private idle slots, then issue fresh coordinate grants and windows.
        segs = self._restage_warm(kind, segs, source)
        run = partial(compute if kind == 'verify' else mtp_compute,
                      self.dec.w, segs, source, check=False)
        from .multi_graphs import free_bytes

        before = free_bytes()
        before_allocated = int(torch.cuda.memory_allocated(self.dec.w.device))
        before_reserved = int(torch.cuda.memory_reserved(self.dec.w.device))
        try:
            source.exact_graph_handoffs = True
            source.gdn_capture_operands = (source.gdn_capture_ready_operands
                if kind == 'verify' and self.native_gdn_capture else None)
            source.gdn_value8_capture = (source.gdn_value8_ready
                if kind == 'verify' and self.value8_prepared is not None else None)
            if source.gdn_value8_capture is not None:
                self.value8_bindings[key] = source.gdn_value8_capture
            if self.piece_diagnostic:
                def observe(index, insert, accounting):
                    record = {"schema": "tensorfold.exact-graph-piece-v1",
                        "rank": getattr(self.dec.w.comm, 'rank', 0),
                        "role": kind, "rows": key[0], "participants": key[1],
                        "piece": index, "terminating_insert": (None if insert is None else
                            {"kind": insert[0], "layer": insert[1].index,
                             "rows": insert[2] if len(insert) > 2 else None}),
                        "diagnostic_only": True, "P3_valid": False,
                        "accounting": accounting}
                    payload = (json.dumps(record, separators=(',', ':')) + '\n').encode('ascii')
                    if os.write(2, payload) != len(payload):
                        raise OSError('exact graph piece diagnostic receipt was truncated')
                pieces = owner._pieces(run, source, observer=observe, retain_pending=True)
            else:
                pieces = owner._pieces(run, source, retain_pending=True)
            # _pieces keeps its original borrower journal until this transfer.
            # Publish before any flag cleanup, device fence or memory query.
            self.bindings[(kind, key)] = pieces
            owner.pending.remove(pieces)
        finally:
            source.exact_graph_handoffs = False
            source.gdn_capture_operands = None
            source.gdn_value8_capture = None
            source.gdn_value8_ready = None
        torch.cuda.synchronize(self.dec.w.device)
        physical_delta = before - free_bytes()
        took = max(0, physical_delta)
        admitted = self._agree(owner.room_bytes + took <= owner.budget)
        error = None
        if not admitted:
            # A graph pool can retain reserved memory after reset; do not
            # advertise that unproven memory as an admissible eager fallback.
            error = MemoryError(f"exact graph budget exceeded: role={kind} rows={key[0]} participants={key[1]} "
                f"pieces={len(pieces.graphs)} prior_keys={len(owner.measured)} "
                f"prior_bytes={owner.room_bytes} delta_bytes={took} budget_bytes={owner.budget}")
            owner.retain_failure(error)
        try:
            if not admitted:
                self.declined_keys[(kind, *key)] = 'captured key exceeded admitted graph budget'
            observation = {"schema": "tensorfold.exact-graph-key-v1", "rank": getattr(self.dec.w.comm, 'rank', 0),
                "role": kind, "rows": key[0], "participants": key[1], "pieces": len(pieces.graphs),
                "prior_keys": len(owner.measured), "prior_bytes": owner.room_bytes,
                "physical_delta_bytes": physical_delta, "allocated_delta_bytes": int(torch.cuda.memory_allocated(self.dec.w.device)) - before_allocated,
                "reserved_delta_bytes": int(torch.cuda.memory_reserved(self.dec.w.device)) - before_reserved,
                "budget_bytes": owner.budget, "admitted": admitted}
            # Startup only: preserve the decisive allocation/driver split even if
            # constructor failure prevents the final graph-owner receipt publishing.
            payload = (json.dumps(observation, separators=(',', ':')) + '\n').encode('ascii')
            if os.write(2, payload) != len(payload):
                raise OSError('exact graph startup accounting receipt was truncated')
            owner.measured.append({"key": [kind, repr(key)], "pieces": len(pieces.graphs), "bytes": took,
                                   "accounting": observation})
        except BaseException as diagnostic:
            if error is not None:
                raise_failures(error, [diagnostic])
            raise
        if error is not None:
            raise error
        owner.bytes += took
        owner.room_bytes += took
        (owner.verify if kind == "verify" else owner.mtp)[key] = pieces
        self.captured_keys.append((kind, *key))
        return pieces

    def forward(self, kind, segs, *, mtp_positions=None, device_tokens=None):
        if kind not in self.serving_hits:
            raise ValueError("exact graph role must be verify or mtp")
        if self.closed:
            raise RuntimeError("exact graph owner is retired")
        if self.owner.failure is not None:
            raise self.owner.failure
        # Device-greedy staging has a separate preparation/position contract.
        # Ordinary MTP4 uses the same public staging path as eager execution.
        if mtp_positions is not None or device_tokens is not None:
            self.misses += 1
            if not self.preparing:
                self.serving_misses[kind] += 1
            return None
        source = self.dec.buf if kind == "verify" else self.dec.mbuf
        key = self._key(kind, segs)
        if source.rotor_lookup or any(state.image_positions is not None for state, _, _ in segs):
            self.misses += 1
            if not self.preparing:
                self.serving_misses[kind] += 1
            return None
        from .compact_indexer import validate_target_indexer_frame, validate_mtp_indexer_frame

        validate = validate_target_indexer_frame if kind == "verify" else validate_mtp_indexer_frame
        validate(source, segs)
        if kind == 'verify' and self.native_gdn_capture:
            if self.preparing:
                self._native_frame(segs, source)
            else:
                try:
                    # Published inside the existing all-rank decode-window
                    # preparation phase. The fixed region rejects registration
                    # changes there rather than taking a rank-local fallback.
                    value8 = self.value8_bindings.get(key) if self.value8_prepared is not None else None
                    if value8 is not None:
                        # Contains the original current-frame/actor check plus
                        # Q/descriptor/output binding facts. All layers share
                        # these fixed owners, so validate once, not per layer.
                        value8[0].validate_frame(source.gdn_tables, segs=segs)
                    else:
                        source.gdn_tables.validate_current(segs, source)
                except BaseException as error:
                    from .input_preparation import reject_prepared

                    reject_prepared(source, error)
        pieces = (self.owner.verify if kind == "verify" else self.owner.mtp).get(key)
        if pieces is None and self.preparing and self.selected_keys is not None and (kind, *key) in self.selected_keys:
            pieces = self._capture(kind, key, segs, source)
            if pieces is not None:
                # Capture preparation reset/restaged these same actual slots.
                # Validate the newly issued frame before the first replay too.
                validate(source, segs)
        if pieces is None:
            self.misses += 1
            if not self.preparing:
                self.serving_misses[kind] += 1
            return None
        try:
            _kv_begin(segs)
            for graph, insert in zip(pieces.graphs, pieces.inserts + [None]):
                replay(graph, self.dec.w.comm)
                if insert is not None:
                    self._insert(insert, segs, source)
            self.hits += 1
            if not self.preparing:
                self.serving_hits[kind] += 1
            return pieces.out
        except BaseException as error:
            self.owner.retain_failure(error)
            self.dec._failed_graphs, self.dec._memory_error = self.owner, error
            raise

    def _reset_warm_states(self, segs):
        if (not self.preparing or self.dec.streams or self.dec.filling or self.dec.kept):
            raise RuntimeError('synthetic graph reset requires the globally idle startup catalogue')
        _kv_check(segs, comm=self.dec.w.comm)
        torch.cuda.synchronize(self.dec.w.device)
        seen = set()
        for state, _, _ in segs:
            if id(state) in seen:
                continue
            index = self.indices.get(id(state))
            if index is None or self.states[index] is not state:
                raise ValueError('synthetic graph reset requires its original idle State slots')
            seen.add(id(state))
            state.reset(self.dec.w, new_sequence=False)

    def _restage_warm(self, kind, segs, source):
        tokens = source.ids_host[:segs[-1][2]].tolist()
        hidden = source.mtp_in if kind == 'mtp' else None
        self._reset_warm_states(segs)
        if kind == 'verify':
            rebound = stage(self.dec.w, source, [(state, tokens[first:last]) for state, first, last in segs], late=True)
            if source.ple_late is not None:
                stage_late(self.dec.w, source)
        else:
            # State reset does not mutate the decoder's owned HC input buffer;
            # mtp_stage owns any copies/aliases and issues fresh position grants.
            rebound = mtp_stage(source, [(state, tokens[first:last], hidden[first:last])
                                         for state, first, last in segs])
        from .compact_indexer import validate_target_indexer_frame, validate_mtp_indexer_frame

        (validate_target_indexer_frame if kind == 'verify' else validate_mtp_indexer_frame)(source, rebound)
        if kind == 'verify' and self.native_gdn_capture:
            self._native_frame(rebound, source, capture=True)
        _kv_begin(rebound)
        torch.cuda.synchronize(self.dec.w.device)  # Reset/copies retire before side-stream capture begins.
        return rebound

    def warm(self, states):
        if self.preparing or self.bindings:
            raise RuntimeError("exact graph catalogue is prepared once")
        # Validate the entire requested shape/role set before resetting States,
        # uploading inputs, or attempting any capture.
        shapes = self._catalogue(states)
        self.preparing = True
        try:
            for count, rows in shapes:
                verify = ('verify', count * rows, count) in self.selected_keys
                mtp = ('mtp', count * rows, count) in self.selected_keys
                if not verify and not mtp:
                    continue
                active = list(reversed(states))[:count]
                private = [(state, index * rows, (index + 1) * rows) for index, state in enumerate(active)]
                self._reset_warm_states(private)
                segs = stage(self.dec.w, self.dec.buf,
                             [(state, [0] * rows) for state in active], late=True)
                if self.dec.buf.ple_late is not None:
                    stage_late(self.dec.w, self.dec.buf)
                result = self.forward('verify', segs) if verify else None
                if result is None:
                    # MTP needs the actual target HC rows even when that target
                    # key was excluded or declined before capture admission.
                    if self.native_gdn_capture:
                        self._native_frame(segs, self.dec.buf)
                    compute(self.dec.w, segs, self.dec.buf, check=False)
                _kv_check(segs, comm=self.dec.w.comm)
                if mtp:
                    windows = [(state, [0] * rows, self.dec.buf.streams[first:last])
                               for state, first, last in segs]
                    self._reset_warm_states(segs)
                    mseg = mtp_stage(self.dec.mbuf, windows)
                    if self.forward('mtp', mseg) is None:
                        mtp_compute(self.dec.w, mseg, self.dec.mbuf, check=False)
                    _kv_check(mseg, comm=self.dec.w.comm)
        finally:
            self.preparing = False

    def invalidate(self, state):
        if id(state) not in self.indices:
            return
        # Attention cache replacement is validated by the live handoff. Native
        # GDN captures use fresh descriptor contents; registration retains every
        # pointee. This private fixed region rejects recurrent registration
        # changes in decode preparation before any rank submits its replay.

    def close(self):
        torch.cuda.synchronize(self.dec.w.device)
        for (kind, key), pieces in tuple(self.bindings.items()):
            for graph in pieces.graphs:
                graph.reset()
            (self.owner.verify if kind == "verify" else self.owner.mtp).pop(key, None)
            del self.bindings[(kind, key)]
        # The outer owner retires pending/partial captures before this method.
        # Keep every plan if any graph reset fails, including failed-capture
        # keys that never reached the published self.bindings catalogue.
        self.value8_bindings.clear()
        self.closed = True

    def record(self):
        catalogue = {'policy': 'full-original-v1' if self.requested_policy is None else self.requested_policy['policy'],
            'evidence_sha256': None if self.requested_policy is None else self.requested_policy['evidence_sha256'],
            'available_keys': [list(key) for key in self.available_keys],
            'requested_keys': [list(key) for key in self.requested_keys],
            'selected_keys': [list(key) for key in self.available_keys if self.selected_keys is not None and key in self.selected_keys],
            'captured_keys': [list(key) for key in self.captured_keys],
            'declined_keys': [{'key': list(key), 'reason': reason} for key, reason in self.declined_keys.items()]}
        return {"capture_scope": "exact-row-gpu-pieces-with-eager-state-handoffs",
                "replay_hits": self.hits, "replay_misses": self.misses,
                "serving_hits": dict(self.serving_hits), "serving_misses": dict(self.serving_misses),
                "native_gdn_capture": self.native_gdn_capture,
                "gdn_value8_capture": self.value8_prepared is not None,
                "gdn_value8_workspace": deepcopy(self.value8_workspace_record),
                "exact_graph_catalogue": catalogue,
                "eager_owner_inserts": self.owner_inserts, "capture_during_serving": False}

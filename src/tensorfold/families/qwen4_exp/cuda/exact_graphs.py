"""Exact-row SM70 graph pieces with live unique-history handoffs between them.

The graph borrows the decoder's original buffers. Expert and linear GPU
operations are captured, while recurrent chains, unique attention and host PLE
staging execute against the current live State between pieces. Captures retain
no State pointer, padded row, coordinate grant or host position.
Only idle startup creates graph keys; an unseen serving key runs eagerly.
"""

from functools import partial

import torch

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
        owner = self.owner
        if not self._agree(owner.room_bytes < owner.budget and owner._has_room()):
            return None
        run = partial(compute if kind == "verify" else mtp_compute,
                      self.dec.w, segs, source, check=False)
        run()
        _kv_check(segs, comm=self.dec.w.comm)
        torch.cuda.synchronize(self.dec.w.device)
        from .multi_graphs import free_bytes

        before = free_bytes()
        source.exact_graph_handoffs = True
        try:
            pieces = owner._pieces(run, source)
        finally:
            source.exact_graph_handoffs = False
        torch.cuda.synchronize(self.dec.w.device)
        took = max(0, before - free_bytes())
        self.bindings[(kind, key)] = pieces
        if not self._agree(owner.room_bytes + took <= owner.budget):
            # A graph pool can retain reserved memory after reset; do not
            # advertise that unproven memory as an admissible eager fallback.
            error = MemoryError("exact graph pieces exceed their admitted shared budget")
            owner.retain_failure(error)
            raise error
        owner.bytes += took
        owner.room_bytes += took
        owner.measured.append({"key": [kind, repr(key)], "pieces": len(pieces.graphs), "bytes": took})
        (owner.verify if kind == "verify" else owner.mtp)[key] = pieces
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
        pieces = (self.owner.verify if kind == "verify" else self.owner.mtp).get(key)
        if pieces is None and self.preparing:
            pieces = self._capture(kind, key, segs, source)
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

    def warm(self, states):
        if self.preparing or self.bindings:
            raise RuntimeError("exact graph catalogue is prepared once")
        self.preparing = True
        try:
            width = self.dec.depth + 1
            # Full-width verifier rows and one-row proposal steps cover the
            # regular MTP round. C1 also prepares every acceptance-tail width.
            maximum = min(len(states), self.dec.buf.rows // width)
            counts = sorted({value for value in (1, 2, 4, 8, 16, 24, maximum) if 0 < value <= maximum})
            for count in counts:
                active = list(reversed(states))[:count]
                widths = range(1, width + 1) if count == 1 else sorted({1, width})
                for rows in widths:
                    segs = stage(self.dec.w, self.dec.buf,
                                 [(state, [0] * rows) for state in active], late=True)
                    if self.dec.buf.ple_late is not None:
                        stage_late(self.dec.w, self.dec.buf)
                    self.forward("verify", segs)
                    _kv_check(segs, comm=self.dec.w.comm)
                    if self.dec.mbuf is not None:
                        windows = [(state, [0] * rows, self.dec.buf.streams[first:last])
                                   for state, first, last in segs]
                        self.forward("mtp", mtp_stage(self.dec.mbuf, windows))
                        _kv_check(segs, comm=self.dec.w.comm)
        finally:
            self.preparing = False

    def invalidate(self, state):
        if id(state) not in self.indices:
            return
        # No captured kernel borrows State storage. Cache replacement does
        # not invalidate these buffer-only graphs; live handoffs validate the
        # replacement cache and its current coordinate frame independently.

    def close(self):
        torch.cuda.synchronize(self.dec.w.device)
        for (kind, key), pieces in tuple(self.bindings.items()):
            for graph in pieces.graphs:
                graph.reset()
            (self.owner.verify if kind == "verify" else self.owner.mtp).pop(key, None)
            del self.bindings[(kind, key)]
        self.closed = True

    def record(self):
        return {"capture_scope": "exact-row-gpu-pieces-with-eager-state-handoffs",
                "replay_hits": self.hits, "replay_misses": self.misses,
                "serving_hits": dict(self.serving_hits), "serving_misses": dict(self.serving_misses),
                "eager_owner_inserts": self.owner_inserts, "capture_during_serving": False}

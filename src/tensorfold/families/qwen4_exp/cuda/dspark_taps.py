"""Owned, pending-safe target features for an explicitly attached DSpark adapter.

This module does not install a target hook or change State. Capture before the
configured layer, before its PLE, using the original pending HC writeback into
private scratch. Feature leases remain owned until a consumer releases them.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import glue

MAX_ROWS = 2048
STREAMS = 4


@dataclass(frozen=True)
class TapPlan:
    device_bytes: int
    host_bytes: int
    transient_bytes: int

    @property
    def admission_bytes(self):
        return self.device_bytes + self.transient_bytes


def _geometry(config, rows, packed_rows):
    hidden = config.hidden
    before = tuple(config.capture_before)
    if (type(rows) is not int or not 1 <= rows <= MAX_ROWS or type(packed_rows) is not int
            or not 0 <= packed_rows <= rows or type(hidden) is not int or hidden <= 0 or hidden % 256
            or not before or any(type(i) is not int or i < 0 for i in before)
            or tuple(sorted(set(before))) != before
            or before != tuple(i + 1 for i in config.taps)):
        raise ValueError("DSpark taps require ordered output+1 layers, complete HC groups and at most2048 rows")
    return hidden, before


def byte_plan(config, rows, *, packed_rows=0):
    hidden, before = _geometry(config, rows, packed_rows)
    features = rows * len(before) * hidden * 2
    materialized = rows * STREAMS * hidden * 2
    sums = rows * (hidden // 256) * STREAMS * 4
    mean = rows * hidden * 4
    packed = packed_rows * (len(before) * hidden * 2 + 8)
    # The maintained out= reduction avoids an explicit FP32 HC cast. Until
    # native GPU allocation qualification, retain this conservative library
    # reduction/staging bound separately from the exact owned tensor bytes.
    transient = rows * hidden * (STREAMS + 1) * 4
    return TapPlan(features + materialized + sums + mean + packed, packed_rows * 8, transient)


@dataclass(frozen=True)
class Segment:
    state: object
    version: int
    identity: object
    history: object
    start: int
    a0: int
    a1: int

    def live(self):
        if ((self.state.version, self.state.kv_identity) != (self.version, self.identity)
                or getattr(self.state, "cur", None) is not self.history
                or getattr(self.state, "_kv_error", None) is not None
                or getattr(self.state, "_resize_error", None) is not None):
            raise ValueError("DSpark tap segment no longer owns its validated target storage")


def _segments(rows, segments):
    result, end, seen = [], 0, set()
    for state, a0, a1 in segments:
        if (type(a0) is not int or a0 != end or type(a1) is not int or not a0 < a1 <= rows or id(state) in seen
                or type(state.pos) is not int or state.pos < 0):
            raise ValueError("DSpark tap segments require one compact row range per target State")
        seen.add(id(state))
        segment = Segment(state, state.version, state.kv_identity, getattr(state, "cur", None), state.pos, a0, a1)
        segment.live()
        result.append(segment)
        end = a1
    if result and end != rows:
        raise ValueError("DSpark tap segments do not cover the captured rows")
    return tuple(result)


class TapWorkspace:
    """One worker/graph owner, static tensor addresses, and one active frame lease."""

    def __init__(self, config, rows, device, *, packed_rows=0):
        self.hidden, self.before = _geometry(config, rows, packed_rows)
        self.rows, self.packed_rows, self.device = rows, packed_rows, torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.plan = byte_plan(config, rows, packed_rows=packed_rows)
        self.features = torch.empty((rows, len(self.before), self.hidden), dtype=torch.float16, device=self.device)
        self.materialized = torch.empty((rows, STREAMS * self.hidden), dtype=torch.float16, device=self.device)
        self.pss = torch.empty((rows, self.hidden // 256, STREAMS), dtype=torch.float32, device=self.device)
        self.mean = torch.empty((rows, self.hidden), dtype=torch.float32, device=self.device)
        self.packed = torch.empty((packed_rows, len(self.before) * self.hidden), dtype=torch.float16, device=self.device)
        self.indices = torch.empty(packed_rows, dtype=torch.int64, device=self.device)
        self.host_indices = torch.empty(packed_rows, dtype=torch.int64, pin_memory=self.device.type == "cuda")
        self._ready = torch.cuda.Event() if self.device.type == "cuda" else None
        self._done = torch.cuda.Event() if self.device.type == "cuda" else None
        self._uploaded = torch.cuda.Event() if self.device.type == "cuda" else None
        self._pending_done = self._pending_upload = False
        self._active, self._closed, self._generation = None, False, 0
        self._graphs = set()
        self._graph_sink = None
        self._graph_recorded = False

    @property
    def nbytes(self):
        if self._closed:
            return 0
        return sum(t.numel() * t.element_size() for t in
                   (self.features, self.materialized, self.pss, self.mean, self.packed, self.indices))

    def begin(self, rows, segments=()):
        if self._closed or self._active is not None or type(rows) is not int or not 1 <= rows <= self.rows:
            raise ValueError("DSpark tap workspace is closed, borrowed, or exceeds admitted rows")
        bindings = _segments(rows, segments)
        if self._pending_done:
            torch.cuda.current_stream(self.device).wait_event(self._done)
            self._pending_done = False
        self._generation += 1
        capture = TapCapture(self, rows, bindings)
        self._active = capture
        return capture

    def close(self):
        if self._active is not None or self._graphs:
            raise ValueError("DSpark tap workspace still has an active producer, consumer or graph plan")
        if self._pending_done:
            self._done.synchronize()
        if self._pending_upload:
            self._uploaded.synchronize()
        self._closed = True
        self.features = self.materialized = self.pss = self.mean = None
        self.packed = self.indices = self.host_indices = None

    def prepare_replay(self):
        """Outside capture, wait for a prior export before overwriting graph features."""
        if self._closed or self._active is not None:
            raise ValueError("DSpark graph tap workspace is closed or borrowed")
        if self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise ValueError("DSpark replay ownership must be checked outside capture")
        if self._pending_done:
            torch.cuda.current_stream(self.device).wait_event(self._done)
            self._pending_done = False

    def begin_graph(self, rows):
        """Record only feature writes; target State bindings belong to each replay."""
        if self._pending_done and self.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
            raise ValueError("DSpark graph consumer must be fenced before capture")
        capture = self.begin(rows)
        self._graph_recorded = False
        if self._graph_sink is None:
            self._graph_sink = object()
            self._graphs.add(id(self._graph_sink))
        capture.graph_sink = self._graph_sink
        return capture

    def export_graph(self, destination, segments, index):
        """Copy an owner's trusted target export map into a compact, independent lease.

        The bound graph owner supplies the same immutable int64 row map it uses
        to export target logits/streams. It guarantees each map entry names a
        computed physical row, including the leading real prefix of every lane.
        This API never reads that device map back or invents a second row order.
        """
        if (self._closed or self._active is not None or self._graph_sink is None or not self._graph_recorded
                or destination is self or destination.device != self.device
                or destination.hidden != self.hidden or destination.before != self.before
                or not isinstance(index, torch.Tensor) or index.dtype != torch.int64
                or index.device != self.device or index.ndim != 1 or not index.is_contiguous()
                or not 1 <= index.numel() <= destination.rows):
            raise ValueError("DSpark graph export requires its owned row map and distinct matching storage")
        self._check_aliases((index,))
        destination._check_aliases((self.features, index))
        rows = index.numel()
        capture = destination.begin(rows, segments)
        try:
            torch.index_select(self.features.flatten(1), 0, index,
                               out=destination.features[:rows].flatten(1))
            capture.seen.update(destination.before)
            return capture.finish()
        except BaseException:
            capture.abort()
            raise
        finally:
            # The graph feature storage is borrowed only until this copy;
            # context projection reads destination's independent feature lease.
            if self._done is not None:
                self._done.record(torch.cuda.current_stream(self.device))
                self._pending_done = True

    def retire_graph(self):
        """After the owning graph pool resets ALL graphs borrowing this buffer."""
        if self._active is not None:
            raise ValueError("DSpark graph tap still has an active producer or consumer")
        if self._graph_sink is not None:
            self._graphs.discard(id(self._graph_sink))
            self._graph_sink = None
            self._graph_recorded = False

    def _check_aliases(self, values):
        for value in values:
            if value is not None and any(torch._C._overlaps(value, owned) for owned in
                    (self.features, self.materialized, self.pss, self.mean, self.packed, self.indices)):
                raise ValueError("DSpark tap inputs must not alias their owned output/scratch")


class TapCapture:
    def __init__(self, owner, rows, segments):
        self.owner, self.rows, self.segments = owner, rows, segments
        self.generation, self.seen, self.captured = owner._generation, set(), set()
        self.closed = False

    def _open(self):
        if self.closed or self.owner._closed or self.owner._active is not self:
            raise ValueError("DSpark tap producer is stale or finished")

    def _live(self):
        self._open()
        for segment in self.segments:
            segment.live()

    def before_layer(self, index, h, pending=None):
        self._live()
        o, rows = self.owner, self.rows
        if index not in o.before:
            return
        if index in self.seen:
            raise ValueError("DSpark duplicate layer capture")
        if (h.device != o.device or h.dtype != torch.float16 or not h.is_contiguous()
                or h.shape not in ((rows, STREAMS * o.hidden), (rows, STREAMS, o.hidden))):
            raise ValueError("DSpark taps require matching contiguous FP16 four-branch HC rows")
        flat = h.view(rows, STREAMS * o.hidden)
        values = [h]
        if pending is not None:
            if not isinstance(pending, tuple) or len(pending) != 4:
                raise ValueError("DSpark pending HC branch requires the existing four-field tuple")
            mode, branch, weights, inject = pending
            if type(mode) is not int or mode not in (1, 2, 3, 4) or inject is None:
                raise ValueError("DSpark pending HC branch has an unsupported mode")
            if (not isinstance(inject, torch.Tensor) or inject.device != o.device or inject.dtype != torch.float16
                    or inject.ndim != 2 or inject.shape[0] < rows or inject.shape[1] != STREAMS
                    or not inject.is_contiguous() or not isinstance(branch, torch.Tensor)
                    or branch.device != o.device or not branch.is_contiguous()):
                raise ValueError("DSpark pending HC operands do not match its target rows/device")
            if mode == 2:
                if (branch.ndim != 3 or branch.shape[0] != rows or branch.shape[2] != o.hidden
                        or branch.dtype not in (torch.float16, torch.float32) or not isinstance(weights, torch.Tensor)
                        or weights.device != o.device or weights.dtype != torch.float32
                        or weights.shape != branch.shape[:2] or not weights.is_contiguous()):
                    raise ValueError("DSpark pending MoE slots/weights do not match its target rows")
            elif (weights is not None or branch.dtype not in (torch.float16, torch.float32)
                  or (mode == 1 and branch.shape != (rows, o.hidden))
                  or (mode in (3, 4) and (branch.ndim != 3 or branch.shape[0] < 1
                                        or branch.shape[1:] != (rows, o.hidden) or branch.dtype != torch.float32))):
                raise ValueError("DSpark pending branch/reduction geometry is invalid")
            values.extend((branch, weights, inject))
        o._check_aliases(values)
        if pending is not None:
            if h.is_cuda:
                from tensorfold.cuda.sm70 import family_ops
                if not family_ops._NATIVE_FAMILY:
                    raise ValueError("CUDA pending taps require the qualified native FP16 HC writeback")
            if mode == 2:
                glue.hc_writeback(flat, o.materialized[:rows], o.pss[:rows], STREAMS, mode,
                                  inject=inject[:rows], y=branch, wts=weights)
            else:
                glue.hc_writeback(flat, o.materialized[:rows], o.pss[:rows], STREAMS, mode,
                                  branch=branch, inject=inject[:rows])
            flat = o.materialized[:rows]
        torch.mean(flat.view(rows, STREAMS, o.hidden), dim=1, dtype=torch.float32, out=o.mean[:rows])
        o.features[:rows, o.before.index(index)].copy_(o.mean[:rows])
        self.seen.add(index)
        if h.is_cuda and torch.cuda.is_current_stream_capturing():
            self.captured.add(index)

    def finish(self):
        self._live()
        if self.seen != set(self.owner.before) or self.captured:
            raise ValueError("DSpark taps are incomplete or require an explicit captured-graph plan")
        self.closed = True
        segments, self.segments = self.segments, ()
        return TapBatch(self.owner, self.rows, segments)

    def finish_capture(self, graph):
        self._live()
        if self.captured != set(self.owner.before) or not callable(getattr(graph, "replay", None)):
            raise ValueError("DSpark graph plan requires every configured hook in its actual CUDA capture")
        self.closed = True
        self.owner._active = None
        plan = TapGraphPlan(self.owner, self.rows, self.segments, graph)
        self.segments = ()
        self.owner._graphs.add(id(plan))
        return plan

    def finish_graph_sink(self):
        """End the Python capture/warm call without any context publication."""
        self._live()
        if (getattr(self, "graph_sink", None) is None
                or self.graph_sink is not self.owner._graph_sink
                or self.seen != set(self.owner.before)
                or (self.captured and self.captured != self.seen)):
            raise ValueError("DSpark graph sink requires every configured target hook")
        self.closed = True
        self.segments = ()
        self.owner._active = None
        self.owner._graph_recorded = True

    def abort(self):
        self._open()  # failed/invalidated target metadata must not block cleanup
        if self.owner._done is not None:
            self.owner._done.record(torch.cuda.current_stream(self.owner.device))
            self.owner._pending_done = True
        self.closed = True
        self.segments = ()
        self.owner._active = None


@dataclass
class TapGraphPlan:
    owner: TapWorkspace
    rows: int
    bindings: tuple[Segment, ...]
    graph: object

    def replay(self, segments=()):
        if self.owner is None or id(self) not in self.owner._graphs:
            raise ValueError("DSpark tap graph plan is retired")
        current = _segments(self.rows, segments)
        def identity(part):
            return id(part.state), part.version, part.identity, part.a0, part.a1
        if tuple(map(identity, current)) != tuple(map(identity, self.bindings)):
            raise ValueError("DSpark captured taps require their original target slots and row geometry")
        capture = self.owner.begin(self.rows, segments)
        try:
            self.graph.replay()
            capture.seen.update(self.owner.before)
            return capture.finish()
        except BaseException:
            capture.abort()
            raise

    def retire(self):
        """Call after the target graph owner stops replaying this graph."""
        if self.owner is None:
            return
        if self.owner._active is not None:
            raise ValueError("DSpark tap graph still has a producer or feature consumer")
        self.owner._graphs.discard(id(self))
        self.owner, self.graph, self.bindings = None, None, ()


class TapBatch:
    """Borrowed owned features; enqueue consumer reads before release on that stream."""

    def __init__(self, owner, rows, segments):
        self.owner, self.rows, self.segments = owner, rows, segments
        self.generation, self.released, self.compacted = owner._generation, False, False
        self._features = owner.features[:rows].flatten(1)
        owner._active = self
        if owner._ready is not None:
            owner._ready.record(torch.cuda.current_stream(owner.device))

    def _open(self):
        if self.released or self.owner._closed or self.owner._active is not self:
            raise ValueError("DSpark tap feature lease is released or stale")

    def _live(self):
        self._open()
        for segment in self.segments:
            segment.live()

    def wait_ready(self):
        self._live()
        if self.owner._ready is not None:
            torch.cuda.current_stream(self.owner.device).wait_event(self.owner._ready)

    @property
    def features(self):
        self.wait_ready()
        if any(part.state.pos != part.start for part in self.segments):
            raise ValueError("advanced target taps require an explicit accepted-prefix segment")
        return self._features

    def segment(self, state, *, accepted=None):
        self.wait_ready()
        part = next((part for part in self.segments if part.state is state), None)
        if part is None:
            raise ValueError("DSpark tap segment belongs to a different target State")
        rows = part.a1 - part.a0 if accepted is None else accepted
        if (type(rows) is not int or not 0 <= rows <= part.a1 - part.a0
                or state.pos != part.start + (0 if accepted is None else rows)):
            raise ValueError("DSpark tap rows differ from the committed target prefix")
        return self._features[part.a0:part.a0 + rows]

    def compact(self, row_indices, segments=()):
        self._live()
        indices = tuple(row_indices)
        if (self.compacted or not indices or len(indices) > self.owner.packed_rows
                or any(type(i) is not int or not 0 <= i < self.rows for i in indices)
                or len(set(indices)) != len(indices)):
            raise ValueError("DSpark compact rows exceed their owned map/output or source frame")
        bindings = _segments(len(indices), segments)
        if bool(bindings) != bool(self.segments):
            raise ValueError("DSpark compaction cannot add or erase target ownership")
        old = {(id(part.state), part.version, part.identity, part.start) for part in self.segments}
        if any((id(part.state), part.version, part.identity, part.start) not in old for part in bindings):
            raise ValueError("DSpark compact rows changed their target owners or source cursor")
        for part in bindings:
            source = next(source for source in self.segments if source.state is part.state)
            expected = tuple(range(source.a0, source.a0 + part.a1 - part.a0))
            if indices[part.a0:part.a1] != expected:
                raise ValueError("DSpark compact rows must retain each target's original leading prefix")
        self.wait_ready()
        o, count = self.owner, len(indices)
        if o._pending_upload:
            o._uploaded.synchronize()  # the pinned index mirror remains borrowed until its last upload
        o.host_indices[:count].numpy()[:] = indices
        o.indices[:count].copy_(o.host_indices[:count], non_blocking=o.device.type == "cuda")
        if o._uploaded is not None:
            o._uploaded.record()
            o._pending_upload = True
        torch.index_select(self._features, 0, o.indices[:count], out=o.packed[:count])
        self.rows, self.segments, self._features, self.compacted = count, bindings, o.packed[:count], True
        if o._ready is not None:
            o._ready.record()
        return self

    def release(self):
        self._open()  # a failed target must still be able to retire its feature storage
        if self.owner._ready is not None:
            torch.cuda.current_stream(self.owner.device).wait_event(self.owner._ready)
        if self.owner._done is not None:
            self.owner._done.record(torch.cuda.current_stream(self.owner.device))
            self.owner._pending_done = True
        self.released = True
        self.owner._active = None
        self.owner, self._features, self.segments = None, None, ()

"""Unique pooled history with bounded, transactional unrotated raw tails.

Logical cache rows define grouping. RoPE coordinates are separate row data;
especially MTP's shifted input embedding does not define its cache coordinates.
The target execution arena retains every locally owned cache ordinal until the
accepted-prefix decision. Each cache's four-row ring contains only its pending
members; completed raw keys are never a context-growing history.
"""

from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager

import torch

from tensorfold.cuda.sm70 import compact_indexer as native
from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.cleanup import raise_failures

SCHEMA = "tensorfold.compact-indexer-v1"
RATIO = 4


def coordinate_rows(state, positions):
    """Resolve the existing query-coordinate route independently of cache grouping."""
    values = positions.long()[:, None].expand(-1, 3)
    if state.image_positions is not None or state.rope_delta:
        values = values + state.rope_delta_dev.reshape(-1)[0]
    if state.image_positions is not None and len(state.image_positions):
        length = len(state.image_positions)
        selected = state.image_positions.index_select(0, positions.long().clamp(0, length - 1)).long()
        values = torch.where((positions < length)[:, None], selected, values)
    return values.contiguous()


def resolve_indexer_coordinates(state, cache_start, rows):
    if (type(cache_start) is not int or type(rows) is not int or cache_start < 0 or rows < 0
            or cache_start > state.capacity or rows > state.capacity - cache_start):
        raise ValueError("indexer coordinate rows must fit their explicit cache span")
    positions = torch.arange(rows, dtype=torch.int64, device=state.pos_dev.device) + cache_start
    return coordinate_rows(state, positions)


@dataclass(frozen=True)
class IndexerFrame:
    frame_id: int
    token_rows: int
    segments: tuple
    mtp: bool


def _stage_frame(buffers, segments, coordinates, *, mtp):
    prefix = "mtp" if mtp else "target"
    if coordinates is None:
        setattr(buffers, prefix + "_indexer_coordinates", None)
        setattr(buffers, prefix + "_indexer_frame", None)
        return
    count = segments[-1][2] if segments else 0
    storage = getattr(buffers, prefix + "_indexer_coordinate_storage")
    if (storage is None or not isinstance(coordinates, torch.Tensor) or coordinates.dtype != torch.int64
            or coordinates.device != storage.device or coordinates.shape != (count, 3) or count > buffers.rows):
        raise ValueError("indexer coordinates require owned int64[n,3] storage on the consumer device")
    rows, cursor = [], 0
    for state, first, last in segments:
        if first != cursor or not first < last <= count:
            raise ValueError("coordinate segments must cover the staged rows")
        start = state.mtp_len if mtp else state.pos
        rows.append((state, state.slot_generation, state.history_generation, start, first, last))
        cursor = last
    frame_id = getattr(buffers, prefix + "_indexer_frame_id") + 1
    buffers.compact_indexer_stage.begin(mtp=mtp)
    if coordinates.data_ptr() != storage.data_ptr() or coordinates.stride() != storage.stride():
        storage[:count].copy_(coordinates)
    buffers.compact_indexer_stage.mark(mtp=mtp)
    setattr(buffers, prefix + "_indexer_frame_id", frame_id)
    setattr(buffers, prefix + "_indexer_coordinates", storage[:count])
    setattr(buffers, prefix + "_indexer_frame", IndexerFrame(frame_id, count, tuple(rows), mtp))


def _validate_frame(buffers, segments, *, mtp):
    prefix = "mtp" if mtp else "target"
    frame = getattr(buffers, prefix + "_indexer_frame")
    values = getattr(buffers, prefix + "_indexer_coordinates")
    storage = getattr(buffers, prefix + "_indexer_coordinate_storage")
    if frame is None and values is None:
        return None
    if (not isinstance(frame, IndexerFrame) or frame.mtp is not mtp
            or frame.frame_id != getattr(buffers, prefix + "_indexer_frame_id")
            or values is None or values.shape != (frame.token_rows, 3) or values.dtype != torch.int64
            or storage is None or values.device != storage.device or len(frame.segments) != len(segments)
            or type(frame.token_rows) is not int or not 0 <= frame.token_rows <= buffers.rows
            or frame.token_rows != (segments[-1][2] if segments else 0)
            or values.data_ptr() != storage.data_ptr() or values.stride() != storage.stride()):
        raise ValueError("indexer coordinate frame is stale or malformed")
    for recorded, (state, first, last) in zip(frame.segments, segments):
        start = state.mtp_len if mtp else state.pos
        if (recorded[0] is not state or recorded[1:] !=
                (state.slot_generation, state.history_generation, start, first, last)):
            raise ValueError("indexer coordinates no longer belong to their cache row/generation")
    return values


def stage_mtp_indexer_frame(buffers, segments, coordinates):
    _stage_frame(buffers, segments, coordinates, mtp=True)


def validate_mtp_indexer_frame(buffers, segments):
    return _validate_frame(buffers, segments, mtp=True)


def stage_target_indexer_frame(buffers, segments, coordinates):
    _stage_frame(buffers, segments, coordinates, mtp=False)


def validate_target_indexer_frame(buffers, segments):
    return _validate_frame(buffers, segments, mtp=False)


class Staging:
    """At most one owned ordinal per eight-ordinal channel and execution row."""
    def __init__(self, rows, caches, device, *, mtp, mtp_only=False):
        self.rows = rows
        channels = 0 if mtp_only else (caches + 7) // 8
        self.target = torch.empty((channels, rows, 128), dtype=torch.float16, device=device)
        self.coordinates = torch.empty((0 if mtp_only else rows, 3), dtype=torch.int64, device=device)
        self.mtp = torch.empty((rows if mtp else 0, 128), dtype=torch.float16, device=device)
        self.mtp_coordinates = torch.empty((rows if mtp else 0, 3), dtype=torch.int64, device=device)
        self.target_epoch = self.mtp_epoch = 0
        self._events, self._streams = [None, None], [None, None]
        self._errors = [None, None]
        for role, values in enumerate((self.target, self.mtp)):
            if values.is_cuda and values.numel():
                self._events[role] = torch.cuda.Event()
                stream = torch.cuda.current_stream(values.device)
                self._events[role].record(stream)
                self._streams[role] = stream.cuda_stream

    def wait(self, *, mtp):
        role = int(mtp)
        if self._errors[role] is not None:
            raise self._errors[role]
        event = self._events[role]
        if event is not None:
            values = self.mtp if mtp else self.target
            stream = torch.cuda.current_stream(values.device)
            if stream.cuda_stream != self._streams[role]:
                stream.wait_event(event)

    def mark(self, *, mtp):
        role = int(mtp)
        if self._errors[role] is not None:
            raise self._errors[role]
        event = self._events[role]
        if event is not None:
            try:
                values, coordinates = (self.mtp, self.mtp_coordinates) if mtp else (self.target, self.coordinates)
                stream = torch.cuda.current_stream(values.device)
                values.record_stream(stream)
                coordinates.record_stream(stream)
                event.record(stream)
                self._streams[role] = stream.cuda_stream
            except BaseException as error:
                self._errors[role] = error
                raise

    @contextmanager
    def reading(self, *, mtp):
        """Every explicit-coordinate consumer participates, not only raw owners."""
        self.wait(mtp=mtp)
        primary = prior = secondary = None
        try:
            yield
        except BaseException as error:
            primary = error
            prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                     BaseException.__suppress_context__.__get__(error))
        if self._errors[int(mtp)] is None:
            try:
                self.mark(mtp=mtp)
            except BaseException as error:
                secondary = error
        elif primary is None:
            secondary = self._errors[int(mtp)]
        if primary is not None:
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])
        raise_failures(primary, () if secondary is None else (secondary,))

    def begin(self, *, mtp):
        self.wait(mtp=mtp)
        if mtp:
            self.mtp_epoch += 1
        else:
            self.target_epoch += 1

    def span(self, ordinal, first, last, *, mtp):
        if not 0 <= first < last <= self.rows:
            raise ValueError("compact raw span exceeds its execution arena")
        if mtp:
            return self.mtp[first:last], self.mtp_coordinates[first:last], self.mtp_epoch
        return self.target[ordinal // 8, first:last], self.coordinates[first:last], self.target_epoch

    @property
    def nbytes(self):
        return sum(value.nbytes for value in (self.target, self.coordinates, self.mtp, self.mtp_coordinates))


@dataclass
class Span:
    start: int
    raw: torch.Tensor
    coordinates: torch.Tensor
    staging: Staging | None
    epoch: int
    mtp: bool
    expected: tuple
    resolved: int | None = None

    @property
    def end(self):
        return self.start + len(self.raw)

    def check(self):
        if self.staging is not None and self.epoch != (self.staging.mtp_epoch if self.mtp else self.staging.target_epoch):
            raise ValueError("compact append staging was reused before its cut/commit resolved")


class Cache:
    def __init__(self, capacity, dim, device, generations, status, *, layer, proposal_rows=0, mtp=False):
        self.capacity, self.dim, self.layer = capacity, dim, layer
        self.generations, self.status = generations, status
        self.raw = torch.zeros((RATIO, dim), dtype=torch.float16, device=device)
        self.coordinates = torch.zeros((RATIO, 3), dtype=torch.int64, device=device)
        self.protected = torch.zeros_like(self.raw)
        self.protected_coordinates = torch.zeros_like(self.coordinates)
        self.proposal_raw = torch.zeros((proposal_rows, dim), dtype=torch.float16, device=device)
        self.proposal_coordinates = torch.zeros((proposal_rows, 3), dtype=torch.int64, device=device)
        self.base_raw = torch.zeros((RATIO if mtp else 0, dim), dtype=torch.float16, device=device)
        self.base_coordinates = torch.zeros((RATIO if mtp else 0, 3), dtype=torch.int64, device=device)
        self.n = self.origin = 0
        self.base = None
        self.base_expected = None
        self.proposal_count = 0
        self.span = None
        self.done = torch.cuda.Event() if self.raw.is_cuda else None
        self.pending = False
        self._stream = None
        self._retirement_error = None
        if self.done is not None:
            self.done.record(torch.cuda.current_stream(self.raw.device))  # Materialize every admitted event before publication.
            self._stream = torch.cuda.current_stream(self.raw.device).cuda_stream
            self.pending = True

    def _copy(self, source, destination, expected, at=0):
        self.wait_native_reuse()
        ready = source.contiguous()
        try:
            native.publish(ready, destination, at, self.generations, expected, self.status, self.layer)
            self._mark()
        except BaseException as error:
            self._retirement_error = error
            raise

    def _mark(self):
        if self.done is not None:
            stream = torch.cuda.current_stream(self.raw.device)
            self.done.record(stream)
            self._stream = stream.cuda_stream
            self.pending = True

    def _read_done(self, tensors, *, span=None, mark_cache=True):
        if self.raw.is_cuda:
            stream = torch.cuda.current_stream(self.raw.device)
            for tensor in tensors:
                tensor.record_stream(stream)
            if mark_cache:
                self._mark()
            if span is not None and span.staging is not None:
                span.staging.mark(mtp=span.mtp)

    @contextmanager
    def reading(self, tensors, *, span=None):
        """Retire issued reads even when a later allocation/copy fails.

        Calls are serialized by the coordinator. The event chain supports a
        different current stream without making allocation lifetime stand in
        for mutation ordering. Failure to record retirement poisons this owner.
        """
        marked = [False]
        self.wait_native_reuse()
        if span is not None and span.staging is not None:
            span.staging.wait(mtp=span.mtp)
        primary = prior = secondary = None
        try:
            yield marked
        except BaseException as error:
            primary = error
            prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                     BaseException.__suppress_context__.__get__(error))
        if self._retirement_error is not None:
            # A writer/mark already failed. Retain the borrower rather than
            # retrying uncertain retirement or exposing the shared role.
            if span is not None and span.staging is not None:
                span.staging._errors[int(span.mtp)] = self._retirement_error
            if primary is None:
                primary = self._retirement_error
                prior = (BaseException.__cause__.__get__(primary), BaseException.__context__.__get__(primary),
                         BaseException.__suppress_context__.__get__(primary))
        else:
            try:
                self._read_done(tensors, span=span, mark_cache=not marked[0])
            except BaseException as error:
                self._retirement_error = primary if primary is not None else error
                if span is not None and span.staging is not None:
                    span.staging._errors[int(span.mtp)] = error
                secondary = error
        if primary is not None:
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])
        raise_failures(primary, () if secondary is None else (secondary,))

    def wait_native_reuse(self):
        if self._retirement_error is not None:
            raise self._retirement_error
        if self.pending:
            stream = torch.cuda.current_stream(self.raw.device)
            if stream.cuda_stream != self._stream:
                stream.wait_event(self.done)

    def validate_append(self, start, count, expected):
        if (type(start) is not int or type(count) is not int or count < 1
                or start < 0 or start > self.capacity or start != self.n or count > self.capacity - start):
            raise ValueError("compact append must start at its own valid cache frontier")
        if self.span is not None and self.span.resolved is None:
            raise ValueError("compact append would replace unresolved raw staging")
        if self.base is not None and (self.base_expected != expected or start < self.base
                                     or count > len(self.proposal_raw) - (start - self.base)):
            raise ValueError("MTP compact proposal span exceeds its allocated envelope/generation")

    def begin(self, span, raw, coordinates):
        self.validate_append(span.start, len(span.raw), span.expected)
        span.staging.wait(mtp=span.mtp)
        span.raw.copy_(raw)
        if span.coordinates.data_ptr() != coordinates.data_ptr() or span.coordinates.stride() != coordinates.stride():
            span.coordinates.copy_(coordinates)
        span.staging.mark(mtp=span.mtp)
        self._copy(self.raw, self.protected, span.expected)
        self._copy(self.coordinates, self.protected_coordinates, span.expected)
        self.span = span
        if self.base is not None:
            offset = span.start - self.base
            with self.reading((span.raw, span.coordinates), span=span) as reader:
                self._copy(span.raw, self.proposal_raw, span.expected, offset)
                self._copy(span.coordinates, self.proposal_coordinates, span.expected, offset)
                reader[0] = True
            self.proposal_count = offset + len(span.raw)

    def begin_proposals(self, expected):
        if self.base is not None:
            if self.base_expected != expected:
                raise ValueError("compact proposal base belongs to an older sequence/owner generation")
            return
        if not len(self.proposal_raw):
            raise ValueError("this compact cache has no speculative method span")
        self._copy(self.raw, self.base_raw, expected)
        self._copy(self.coordinates, self.base_coordinates, expected)
        self.base, self.proposal_count = self.n, 0
        self.base_expected = expected

    def tail(self, length):
        if type(length) is not int or not 0 <= length <= self.capacity:
            raise ValueError("compact tail length must fit the selected cache")
        if length == self.n:
            with self.reading((self.raw, self.coordinates)):
                return self.raw.clone(), self.coordinates.clone()
        if length % RATIO == 0 and length < self.n:
            return torch.zeros_like(self.raw), torch.zeros_like(self.coordinates)
        span = self.span
        active_span = None
        if self.base is not None and self.base <= length <= self.base + self.proposal_count:
            start, raw, coordinates = self.base, self.proposal_raw[:self.proposal_count], self.proposal_coordinates[:self.proposal_count]
            protected, protected_coordinates = self.base_raw, self.base_coordinates
        elif span is not None and span.start <= length <= span.end:
            span.check()
            if span.resolved is not None and length > span.resolved:
                raise ValueError("compact cut contains rejected cache rows")
            start, raw, coordinates = span.start, span.raw, span.coordinates
            protected, protected_coordinates = self.protected, self.protected_coordinates
            active_span = span
        else:
            raise ValueError("compact cut needs its captured raw tail; pooled means cannot reconstruct it")
        count = length % RATIO
        result, coords = torch.zeros_like(self.raw), torch.zeros_like(self.coordinates)
        if count:
            with self.reading((protected, protected_coordinates, raw, coordinates), span=active_span):
                positions = torch.arange(length - count, length, device=self.raw.device, dtype=torch.int64)
                old = positions < start
                source = (positions - start).clamp(0, max(0, len(raw) - 1))
                values = torch.where(old[:, None], protected.index_select(0, positions % RATIO), raw.index_select(0, source))
                axes = torch.where(old[:, None], protected_coordinates.index_select(0, positions % RATIO), coordinates.index_select(0, source))
                result.index_copy_(0, positions % RATIO, values)
                coords.index_copy_(0, positions % RATIO, axes)
        return result, coords

    def pool(self, pooled, weight, inv_freq, eps, sections, scale):
        span = self.span
        span.check()
        first, end = span.start // RATIO, span.end // RATIO
        if first == end:
            return
        with self.reading((self.protected, self.protected_coordinates, span.raw, span.coordinates), span=span) as reader:
            positions = torch.arange(first * RATIO, end * RATIO, device=self.raw.device, dtype=torch.int64)
            previous = positions < span.start
            source = (positions - span.start).clamp(0, len(span.raw) - 1)
            raw = torch.where(previous[:, None], self.protected.index_select(0, positions % RATIO), span.raw.index_select(0, source))
            raw = raw.reshape(end - first, RATIO, self.dim).float()
            total = raw[:, 0]
            for member in range(1, RATIO):
                total = total + raw[:, member]
            value = (total / RATIO).half()
            normalization_rows = max(len(value), min(self.capacity // RATIO, 16)) if value.is_cuda else len(value)
            if normalization_rows != len(value):
                padded = torch.zeros((normalization_rows, self.dim), dtype=torch.float16, device=self.raw.device)
                padded[:len(value)].copy_(value)
                value = ops.rms(padded, weight, eps)[:len(value)]
            else:
                value = ops.rms(value, weight, eps)
            member = positions.reshape(-1, RATIO)[:, 0]
            old = member < span.start
            at = (member - span.start).clamp(0, len(span.raw) - 1)
            coordinates = torch.where(old[:, None], self.protected_coordinates.index_select(0, member % RATIO), span.coordinates.index_select(0, at))
            value = ops.rotary(value, inv_freq, torch.arange(len(value), device=value.device),
                               rope=coordinates, length=len(value), sections=sections, scale=scale)
            value = ops._guard(value, self.status, 16, self.layer)
            self._copy(value, pooled, span.expected, first)
            reader[0] = True  # _copy recorded the final cache event; only staging needs another mark.

    def prepare_commit(self, length, expected):
        if type(length) is not int or not 0 <= length <= self.capacity:
            raise ValueError("compact commit length must fit its cache")
        if length == self.n:
            return None  # Another cache can still have a pending append.
        if self.base is not None and length == self.base:
            if self.base_expected != expected:
                raise ValueError("compact proposal rollback belongs to an older generation")
            return self.base_raw, self.base_coordinates, True
        span = self.span
        if span is None or span.expected != expected or not span.start <= length <= span.end:
            raise ValueError("compact commit does not belong to its produced cache span/generation")
        raw, coordinates = self.tail(length)
        return raw, coordinates, False

    def publish_commit(self, plan, length, expected):
        if plan is None:
            return
        raw, coordinates, rollback = plan
        self._copy(raw, self.raw, expected)
        self._copy(coordinates, self.coordinates, expected)
        self.n = length
        if self.span is not None:
            self.span.resolved = length
        if rollback:
            self.base, self.base_expected, self.proposal_count, self.span = None, None, 0, None

    def snapshot(self, length):
        raw, coordinates = self.tail(length)
        return {"origin": self.origin, "length": length, "pooled_valid": length // RATIO,
                "raw": raw, "coordinates": coordinates}

    def validate_snapshot(self, snapshot, length):
        if (not isinstance(snapshot, dict) or set(snapshot) != {"origin", "length", "pooled_valid", "raw", "coordinates"}
                or any(type(snapshot.get(name)) is not int for name in ("origin", "length", "pooled_valid"))
                or snapshot["origin"] != self.origin or snapshot["length"] != length
                or snapshot["pooled_valid"] != length // RATIO):
            raise ValueError("compact snapshot frontier/grouping does not match this cache")
        for name, shape, dtype in (("raw", (RATIO, self.dim), torch.float16), ("coordinates", (RATIO, 3), torch.int64)):
            value = snapshot[name]
            if not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != dtype or value.device != self.raw.device:
                raise ValueError("compact snapshot tail representation differs from its cache")

    def restore(self, snapshot, length, expected):
        self.validate_snapshot(snapshot, length)
        self._copy(snapshot["raw"], self.raw, expected)
        self._copy(snapshot["coordinates"], self.coordinates, expected)
        self.n, self.base, self.proposal_count, self.span = length, None, 0, None
        self.base_expected = None

    def reset(self, expected):
        self._copy(torch.zeros_like(self.raw), self.raw, expected)
        self._copy(torch.zeros_like(self.coordinates), self.coordinates, expected)
        self.n, self.origin, self.base, self.proposal_count, self.span = 0, 0, None, 0, None
        self.base_expected = None

    @property
    def nbytes(self):
        owned = sum(value.nbytes for value in vars(self).values() if isinstance(value, torch.Tensor)
                    and value is not self.generations and value is not self.status)
        if self.span is not None and self.span.staging is None:
            owned += self.span.raw.nbytes + self.span.coordinates.nbytes
        return owned

    def clone(self, generations, status):
        import copy

        reads = tuple(value for value in vars(self).values() if isinstance(value, torch.Tensor))
        if self.span is not None:
            self.span.check()
            reads += (self.span.raw, self.span.coordinates)
        with self.reading(reads, span=self.span):
            other = copy.copy(self)
            for name, value in vars(self).items():
                if isinstance(value, torch.Tensor) and name not in ("generations", "status"):
                    setattr(other, name, value.clone())
            other.generations, other.status = generations, status
            if self.span is not None:
                span = self.span
                span.check()
                other.span = Span(span.start, span.raw.clone(), span.coordinates.clone(), None,
                                  span.epoch, span.mtp, span.expected, span.resolved)
            other.done = torch.cuda.Event() if self.raw.is_cuda else None
            other.pending = other.done is not None
            if other.done is not None:
                other.done.record(torch.cuda.current_stream(self.raw.device))
                other._stream = torch.cuda.current_stream(self.raw.device).cuda_stream
        return other


class CompactIndexer:
    def __init__(self, placement, slot, rank, capacity, caches, dim, device, generations, status, *, proposal_rows, layer_ids):
        if dim != 128 or caches not in (placement.target_layers, placement.target_layers + 1):
            raise ValueError("compact indexer requires the declared target/MTP cache geometry")
        self.placement, self.slot, self.rank = placement, slot, rank
        self.capacity, self.caches, self.dim = capacity, caches, dim
        self.generations, self.status = generations, status
        if len(layer_ids) != caches:
            raise ValueError("compact indexer requires every actual cache layer identity")
        self.entries = [Cache(capacity, dim, device, generations, status,
                              layer=layer_ids[ordinal],
                              proposal_rows=proposal_rows if ordinal == placement.target_layers else 0,
                              mtp=ordinal == placement.target_layers)
                        if rank == placement.indexer_owner(slot, ordinal, include_mtp=caches > placement.target_layers)
                        else None for ordinal in range(caches)]
        self.empty = torch.empty((0, dim), dtype=torch.float16, device=device)
        self._init_transfer()

    def _init_transfer(self):
        self._transfer_done = None
        self._transfer_stream = None
        self._transfer_error = None
        self._transfer_borrowers = ()
        if self.empty.is_cuda:
            self._transfer_done = torch.cuda.Event()
            stream = torch.cuda.current_stream(self.empty.device)
            self._transfer_done.record(stream)  # Materialize before State publication/capacity measurement.
            self._transfer_stream = stream.cuda_stream

    def _wait_transfer(self):
        if self._transfer_error is not None:
            raise self._transfer_error
        if self._transfer_done is not None:
            try:
                stream = torch.cuda.current_stream(self.empty.device)
                if stream.cuda_stream != self._transfer_stream:
                    stream.wait_event(self._transfer_done)
                    self._transfer_stream = stream.cuda_stream
            except BaseException as error:
                self._transfer_error = error
                raise

    @contextmanager
    def transferring(self, other, tensors):
        """Fence every prefix KV/pool read and write on both actual State owners.

        The coordinator serializes calls but may change CUDA streams. Cache
        tail events alone do not cover pooled/KV transfers on non-indexer ranks.
        An uncertain retirement retains all borrowers and forbids either reuse.
        """
        if self is other or self.empty.device != other.empty.device:
            raise ValueError("prefix transfers require distinct same-device compact State owners")
        self.wait_native_reuse()
        other.wait_native_reuse()
        self._transfer_borrowers = other._transfer_borrowers = tensors
        primary = prior = secondary = None
        try:
            yield
        except BaseException as error:
            primary = error
            prior = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error),
                     BaseException.__suppress_context__.__get__(error))
        try:
            if self._transfer_done is not None:
                stream = torch.cuda.current_stream(self.empty.device)
                for tensor in tensors:
                    tensor.record_stream(stream)
                self._transfer_done.record(stream)
                self._transfer_stream = stream.cuda_stream
                other._transfer_done.record(stream)
                other._transfer_stream = stream.cuda_stream
        except BaseException as error:
            self._transfer_error = other._transfer_error = primary if primary is not None else error
            secondary = error
        else:
            self._transfer_borrowers = other._transfer_borrowers = ()
        if primary is not None:
            BaseException.__cause__.__set__(primary, prior[0])
            BaseException.__context__.__set__(primary, prior[1])
            BaseException.__suppress_context__.__set__(primary, prior[2])
        raise_failures(primary, () if secondary is None else (secondary,))

    def layer(self, ordinal):
        entry = self.entries[ordinal]
        return self.empty if entry is None else entry.raw

    def append(self, ordinal, start, raw, coordinates, staging, first, last, *, mtp, expected):
        self._wait_transfer()
        entry = self.entries[ordinal]
        if entry is None:
            raise ValueError("compact indexer append was routed to a non-owner")
        destination, axes, epoch = staging.span(ordinal, first, last, mtp=mtp)
        if (raw.shape != destination.shape or coordinates.shape != axes.shape or raw.dtype != torch.float16
                or coordinates.dtype != torch.int64 or raw.device != destination.device or coordinates.device != axes.device):
            raise ValueError("compact input rows/coordinates do not match their bounded execution span")
        entry.begin(Span(start, destination, axes, staging, epoch, mtp, expected), raw, coordinates)
        return entry

    def begin_mtp_proposals(self, expected):
        self._wait_transfer()
        if self.caches == self.placement.target_layers:
            raise ValueError("compact target-only profile has no MTP cache")
        entry = self.entries[-1]
        if entry is not None:
            entry.begin_proposals(expected)

    def lengths(self, pos, mtp_len):
        if any(type(value) is not int or not 0 <= value <= self.capacity for value in (pos, mtp_len)):
            raise ValueError("compact cache frontiers must be integer rows within the selected allowance")
        return (pos,) * self.placement.target_layers + ((mtp_len,) if self.caches > self.placement.target_layers else ())

    def commit(self, pos, mtp_len, expected):
        self._wait_transfer()
        lengths = self.lengths(pos, mtp_len)
        plans = [None if entry is None else entry.prepare_commit(length, expected)
                 for entry, length in zip(self.entries, lengths)]
        for entry, length, plan in zip(self.entries, lengths, plans):
            if entry is not None:
                entry.publish_commit(plan, length, expected)
        return any(plan is not None for plan in plans)

    def discard_target(self):
        for entry in self.entries[:self.placement.target_layers]:
            if entry is not None:
                entry.span = None

    def has_pending_target(self):
        return any(entry is not None and entry.span is not None and entry.span.resolved is None
                   for entry in self.entries[:self.placement.target_layers])

    def snapshot(self, pos, mtp_len):
        self.wait_native_reuse()
        lengths = self.lengths(pos, mtp_len)
        return {"schema": SCHEMA, "lengths": lengths,
                "tails": tuple(None if entry is None else entry.snapshot(length)
                               for entry, length in zip(self.entries, lengths))}

    def validate_snapshot(self, snapshot, pos, mtp_len):
        lengths = self.lengths(pos, mtp_len)
        if (not isinstance(snapshot, dict) or set(snapshot) != {"schema", "lengths", "tails"}
                or snapshot["schema"] != SCHEMA or snapshot["lengths"] != lengths
                or type(snapshot["lengths"]) is not tuple or any(type(value) is not int for value in snapshot["lengths"])
                or not isinstance(snapshot["tails"], tuple) or len(snapshot["tails"]) != self.caches):
            raise ValueError("compact snapshot belongs to another cache inventory/frontier")
        for entry, length, tail in zip(self.entries, lengths, snapshot["tails"]):
            if entry is not None:
                entry.validate_snapshot(tail, length)
            elif tail is not None:
                raise ValueError("compact snapshot tail was not relocated to its canonical owner")

    def restore(self, snapshot, pos, mtp_len, expected):
        self.wait_native_reuse()
        self.validate_snapshot(snapshot, pos, mtp_len)
        for entry, length, tail in zip(self.entries, self.lengths(pos, mtp_len), snapshot["tails"]):
            if entry is not None:
                entry.restore(tail, length, expected)

    def reset(self, expected):
        self.wait_native_reuse()
        for entry in self.entries:
            if entry is not None:
                entry.reset(expected)

    def wait_native_reuse(self):
        self._wait_transfer()
        for entry in self.entries:
            if entry is not None:
                entry.wait_native_reuse()

    def check(self):
        self._wait_transfer()
        for entry in self.entries:
            if entry is not None and entry._retirement_error is not None:
                raise entry._retirement_error

    @property
    def retirement_failure(self):
        """Actual unretired ownership, independent of the exception's type."""
        if self._transfer_error is not None:
            return self._transfer_error
        for entry in self.entries:
            if entry is None:
                continue
            if entry._retirement_error is not None:
                return entry._retirement_error
            span = entry.span
            if span is not None and span.staging is not None:
                failure = span.staging._errors[int(span.mtp)]
                if failure is not None:
                    return failure
        return None

    def clone(self, generations, status):
        import copy

        self.wait_native_reuse()
        other = copy.copy(self)
        other.generations, other.status = generations, status
        other.entries = [None if entry is None else entry.clone(generations, status) for entry in self.entries]
        other.empty = self.empty.clone()
        other._init_transfer()
        return other

    @property
    def nbytes(self):
        return sum(entry.nbytes for entry in self.entries if entry is not None)

    def record(self):
        return {"schema": SCHEMA, "raw_ring_rows": RATIO, "state_bytes": self.nbytes,
                "owned_ordinals": [ordinal for ordinal, entry in enumerate(self.entries) if entry is not None],
                "pooled_valid": {ordinal: entry.n // RATIO for ordinal, entry in enumerate(self.entries) if entry is not None},
                "pending_raw": {ordinal: entry.n % RATIO for ordinal, entry in enumerate(self.entries) if entry is not None}}

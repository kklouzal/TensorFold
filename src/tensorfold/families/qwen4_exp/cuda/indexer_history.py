"""Lossless RAM indexer history with bounded captured device/pinned staging.

The ordinary ``gpu`` history remains outside this module. In RAM mode a State
owns full NUMA-local FP16 raw history, a bounded GPU ring, and fixed prefill and
decode staging addresses. Pooled centroids remain the existing GPU-owned data.

``begin`` records host-known absolute rows before executing/capturing a frame.
Prep/pool kernels address its fixed stage through a device origin scalar. This
also retains every chained MTP row when the same one-row graph is replayed.
``enqueue`` is capturable asynchronous D2H; ``validated(fenced=True)`` is called
ONLY after the existing producing-stream numeric/sampling readback completed.
It copies ready RAM to its full archive, without another CUDA wait. Committed
lengths are published separately. All CPU copies run on the allocating NUMA
target through libc, never an inherited ATen intra-op worker pool.

The caller retires graphs before closing owners. Reset, prefix restore/copy,
and close may fence pending uploads at those cold lifecycle boundaries. There
is no per-layer/per-token synchronization, worker queue, or GC-owned resource.
The engine actor is the single writer of each session/owner; callers serialize
forward registration, validation, commits, and lifecycle operations.
"""
from __future__ import annotations

from dataclasses import dataclass
import ctypes
import os

from tensorfold.cleanup import raise_failures


FORMAT = "fp16-indexer-archive-v1"


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} requires an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class HistoryPlan:
    """Pure exact payload/backing forecast; ``layers`` includes optional MTP."""

    layers: int
    index_dim: int
    ratio: int
    prefill_rows: int
    decode_rows: int
    page_size: int = os.sysconf("SC_PAGE_SIZE")

    def __post_init__(self):
        for name in ("layers", "index_dim", "ratio", "prefill_rows", "decode_rows", "page_size"):
            _integer(getattr(self, name), name, 1)

    @property
    def prefill_stage_rows(self):
        return self.prefill_rows

    @property
    def decode_stage_rows(self):
        # A verification's accepted MTP absorb plus its entire proposal chain
        # can precede the one existing validation readback.
        return 2 * self.decode_rows

    @property
    def ring_rows(self):
        rows = max(self.prefill_stage_rows, self.decode_stage_rows) + self.ratio - 1
        return -(-rows // self.ratio) * self.ratio

    @property
    def device_bytes_per_layer(self):
        return 2 * self.index_dim * (self.ring_rows + self.prefill_stage_rows + self.decode_stage_rows) + 8

    @property
    def device_nbytes(self):
        return self.layers * self.device_bytes_per_layer

    def _backing(self, size):
        return -(-size // self.page_size) * self.page_size

    def host_shapes(self, limit):
        _integer(limit, "history limit", 1)
        return ((self.layers, limit, self.index_dim),
                (self.layers, self.prefill_stage_rows, self.index_dim),
                (self.layers, self.decode_stage_rows, self.index_dim),
                (self.layers, self.ring_rows, self.index_dim), (self.layers,))

    def host_bytes(self, limit):
        from tensorfold.cuda.numa_memory import backing_bytes

        shapes = self.host_shapes(limit)
        return (sum(backing_bytes(2 * a * b * c, page_size=self.page_size) for a, b, c in shapes[:-1])
                + backing_bytes(8 * self.layers, page_size=self.page_size))

    @property
    def pinned_bytes(self):
        return (sum(self._backing(2 * self.layers * rows * self.index_dim)
                    for rows in (self.prefill_stage_rows, self.decode_stage_rows, self.ring_rows))
                + self._backing(8 * self.layers))

    def rows(self, kind):
        if kind == "prefill":
            return self.prefill_stage_rows
        if kind == "decode":
            return self.decode_stage_rows
        raise ValueError("indexer frame kind must be prefill or decode")


@dataclass(frozen=True)
class HistoryLayer:
    """Explicit ring/stage handle, never a fake full-capacity CUDA tensor."""

    owner: IndexerHistory
    index: int

    @property
    def ring(self):
        self.owner.check()
        return self.owner.ring[self.index]

    @property
    def origin(self):
        self.owner.check()
        return self.owner.origins[self.index:self.index + 1]

    @property
    def device_nbytes(self):
        return self.owner.plan.device_bytes_per_layer

    def stage(self, kind):
        self.owner.check()
        return self.owner.frames[kind].gpu[self.index]


class HistoryFrame:
    """Fixed captured stage addresses; host journals are updated outside capture."""

    def __init__(self, owner, kind, gpu, host):
        self.owner, self.kind, self.gpu, self.host = owner, kind, gpu, host
        self.layers = ()
        self.event = None
        self.submitted = False

    def stage(self, layer):
        self.owner._layer(layer)
        return self.gpu[layer]

    def enqueue(self, layers=None):
        """Capture-safe D2H, with one contiguous copy for each layer interval."""
        self.owner.check()
        indices = self.owner._indices(self.layers if layers is None else layers)
        if not indices:
            return
        first = previous = indices[0]
        for index in (*indices[1:], None):
            if index is not None and index == previous + 1:
                previous = index
                continue
            self.host[first:previous + 1].copy_(self.gpu[first:previous + 1], non_blocking=True)
            first = previous = index

    def validated(self, *, fenced):
        self.owner.validated(self.kind, fenced=fenced)

    def mark_submitted(self, stream=None):
        """Record completion OUTSIDE capture, after this real eager/graph submission.

        Captured internal event nodes are deliberately not used as host-query
        proof for a later replay. The existing mandatory readback should make
        this external, same-stream event ready without another wait.
        """
        self.owner.check()
        if self.gpu.is_cuda:
            import torch
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("archive completion must be recorded after real graph replay, outside capture")
            if self.event is None:
                self.event = torch.cuda.Event()
            self.event.record(stream or torch.cuda.current_stream(self.gpu.device))
        self.submitted = True

    def abort(self, *, fenced=False):
        self.owner.abort(self.kind, fenced=fenced)


class FrameLease:
    """One phase's immutable layer/origin selection over persistent stage storage."""

    def __init__(self, frame, layers, origins):
        self.frame, self.layers, self.origins = frame, layers, origins
        self.owner, self.kind = frame.owner, frame.kind
        self.epoch = self.owner._epoch

    def _check(self):
        self.owner.check()
        if self.epoch != self.owner._epoch:
            raise RuntimeError("stale indexer frame lease after a lifecycle transition")

    @property
    def gpu(self):
        return self.frame.gpu

    @property
    def event(self):
        return self.frame.event

    @event.setter
    def event(self, value):
        self.frame.event = value

    def stage(self, layer):
        self._check()
        return self.frame.stage(layer)

    def enqueue(self, layers=None):
        self._check()
        self.frame.enqueue(self.layers if layers is None else layers)

    def mark_submitted(self, stream=None):
        self._check()
        self.frame.mark_submitted(stream)

    def validated(self, *, fenced):
        self._check()
        self.frame.validated(fenced=fenced)

    def commit(self, keep):
        self._check()
        _integer(keep, "accepted rows")
        lengths = list(self.owner._published)
        for index, origin in zip(self.layers, self.origins):
            last_origin, end = self.owner._last[self.kind].get(index, (-1, -1))
            if last_origin != origin or origin + keep > end:
                raise ValueError("accepted indexer rows exceed this validated frame")
            lengths[index] = origin + keep
        self.owner.commit(lengths)

    def abort(self, *, fenced=False):
        self._check()
        self.frame.abort(fenced=fenced)


class HistorySession:
    """Engine-owned bounded journal; close after all captured borrowers retire."""

    def __init__(self, plan, allocator, device, *, maximum_owners):
        if type(plan) is not HistoryPlan or not callable(allocator):
            raise ValueError("history requires its immutable plan and NUMA host allocator")
        self.plan, self.allocator, self.device = plan, allocator, device
        self.maximum_owners = _integer(maximum_owners, "maximum history owners", 1)
        self._owners = []
        self.closed = False
        self._closing = False

    def new(self, limit):
        if self.closed or self._closing:
            raise RuntimeError("indexer history session is closed")
        self._owners = [owner for owner in self._owners if not owner.closed]
        if len(self._owners) >= self.maximum_owners:
            raise RuntimeError("bounded indexer history owner catalogue is full; retire unused clones")
        owner = object.__new__(IndexerHistory)
        owner._initialize(self, limit)
        self._owners.append(owner)       # Journal before any fallible host/device allocation.
        try:
            owner._allocate()
        except BaseException as error:
            owner._poison = error
            raise
        return owner

    def close(self, *, fenced=False):
        if self.closed:
            return
        self._closing = True
        errors = []
        if not fenced and any(owner._outstanding or owner._upload_pending for owner in self._owners):
            try:
                next(owner for owner in self._owners if owner._outstanding or owner._upload_pending)._synchronize()
                fenced = True
            except BaseException as error:
                errors.append(error)
        for owner in self._owners:
            try:
                owner.close(fenced=fenced)
            except BaseException as error:
                errors.append(error)
        self._owners = [owner for owner in self._owners if not owner.closed]
        if not self._owners:
            try:
                self.allocator.close()
            except BaseException as error:
                errors.append(error)
            else:
                self.closed = True
        raise_failures(None, errors)


class IndexerHistory:
    """One State's lossless raw archive; pooled centroids remain external."""

    def _initialize(self, session, limit):
        self.session, self.plan, self.allocator = session, session.plan, session.allocator
        self.limit, self.device = _integer(limit, "history limit", 1), session.device
        self.closed, self._ready, self._poison = False, False, None
        self._resources, self.frames = [], {}
        self.ring = self.origins = None
        self._published = [0] * self.plan.layers
        self._available = [0] * self.plan.layers
        self._pending = {"prefill": {}, "decode": {}}
        self._last = {"prefill": {}, "decode": {}}
        self._outstanding = self._upload_pending = False
        self._epoch = 0

    def _host(self, shape, dtype, *, pinned, tag):
        from tensorfold.cuda.numa_memory import backing_bytes

        resource = self.allocator(shape, dtype, pinned=pinned, tag=tag)
        size = resource.tensor.numel() * resource.tensor.element_size()
        self._resources.append((resource, backing_bytes(size, page_size=self.plan.page_size)))
        if (tuple(resource.tensor.shape) != tuple(shape) or resource.tensor.dtype != dtype
                or resource.tensor.device.type != "cpu" or not resource.tensor.is_contiguous()):
            raise ValueError("NUMA allocator returned incompatible indexer storage")
        return resource.tensor

    def _allocate(self):
        import torch

        p = self.plan
        self.archive = self._host((p.layers, self.limit, p.index_dim), torch.float16,
                                  pinned=False, tag="indexer-archive")
        self.restore_host = self._host((p.layers, p.ring_rows, p.index_dim), torch.float16,
                                       pinned=True, tag="indexer-ring-restore")
        self.origins_host = self._host((p.layers,), torch.int64, pinned=True, tag="indexer-frame-origins")
        self.ring = torch.zeros((p.layers, p.ring_rows, p.index_dim), dtype=torch.float16, device=self.device)
        self.origins = torch.zeros((p.layers,), dtype=torch.int64, device=self.device)
        self._origin_event = torch.cuda.Event() if self.ring.is_cuda else None
        for kind in ("prefill", "decode"):
            shape = (p.layers, p.rows(kind), p.index_dim)
            host = self._host(shape, torch.float16, pinned=True, tag="indexer-" + kind)
            gpu = torch.zeros(shape, dtype=torch.float16, device=self.device)
            self.frames[kind] = HistoryFrame(self, kind, gpu, host)
        with self.allocator.target.scope():
            for resource, _ in self._resources:
                tensor = resource.tensor
                if tensor is not self.archive:
                    ctypes.memset(tensor.data_ptr(), 0, tensor.numel() * tensor.element_size())
        if self.ring.is_cuda:
            for tensor in (self.restore_host, self.origins_host, *(frame.host for frame in self.frames.values())):
                if not tensor.is_pinned():
                    raise ValueError("NUMA staging must be registered pinned host memory")
        self._ready = True

    @property
    def device_nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.device_tensors())

    @property
    def host_nbytes(self):
        return sum(size for _, size in self._resources)

    def device_tensors(self):
        return [value for value in (self.ring, self.origins, *(frame.gpu for frame in self.frames.values()))
                if value is not None]

    def check(self):
        if self.closed:
            raise RuntimeError("indexer history is closed")
        if self._poison is not None:
            raise self._poison
        if not self._ready:
            raise RuntimeError("indexer history allocation did not complete")

    def _layer(self, layer):
        _integer(layer, "indexer layer")
        if layer >= self.plan.layers:
            raise ValueError("indexer layer is outside its owner")
        return layer

    def _indices(self, layers):
        indices = tuple(self._layer(index) for index in layers)
        if tuple(sorted(set(indices))) != indices:
            raise ValueError("history layers require unique increasing indices")
        return indices

    def layer(self, index):
        self.check()
        return HistoryLayer(self, self._layer(index))

    def _lengths(self, lengths):
        if type(lengths) not in (list, tuple) or len(lengths) != self.plan.layers:
            raise ValueError("one committed raw-history length per indexer layer required")
        result = tuple(_integer(value, "history length") for value in lengths)
        if any(value > self.limit for value in result):
            raise ValueError("history length exceeds its full RAM reservation")
        return result

    def begin(self, kind, position, rows, *, layers=None):
        """Register a phase outside capture; repeated MTP calls append to its origin."""
        self.check()
        self.plan.rows(kind)
        _integer(position, "history frame position")
        _integer(rows, "history frame rows")
        indices = self._indices(range(self.plan.layers) if layers is None else layers)
        if self.ring.is_cuda:
            import torch
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("archive frame registration belongs before capture/replay")
        changed = False
        updates = []
        for index in indices:
            other = "decode" if kind == "prefill" else "prefill"
            if index in self._pending[other]:
                raise RuntimeError("indexer layer already has an unvalidated frame in the other staging owner")
            previous = self._pending[kind].get(index)
            origin, end = previous if previous is not None else (position, position)
            if (position < origin or position > (end if previous is not None else self._available[index])
                    or position + rows > self.limit or position + rows - origin > self.plan.rows(kind)):
                raise ValueError("indexer frame has a gap, invalid origin, or exceeds bounded staging")
            if previous is None:
                changed = True
            updates.append((index, origin, max(end, position + rows), previous is None))
        if changed and self._origin_event is not None and self._origin_event.query() is False:
            raise RuntimeError("pinned indexer origins are still borrowed by an earlier upload")
        with self.allocator.target.scope():
            for index, origin, end, new in updates:
                if new:
                    ctypes.c_int64.from_address(self.origins_host.data_ptr() + 8 * index).value = origin
                    self._last[kind].pop(index, None)
                if rows:
                    self._pending[kind][index] = (origin, end)
        self._outstanding = self._outstanding or bool(indices)
        if changed:
            self.origins.copy_(self.origins_host, non_blocking=True)
            if self._origin_event is not None:
                self._origin_event.record(torch.cuda.current_stream(self.ring.device))
        frame = self.frames[kind]
        frame.layers = indices
        frame.submitted = False
        return FrameLease(frame, indices, tuple(origin for _, origin, _, _ in updates))

    def validated(self, kind=None, *, fenced):
        """Archive ready staged bytes after the caller's existing numeric wait."""
        self.check()
        if fenced is not True:
            raise ValueError("archive publication requires the completed producing-stream validation fence")
        kinds = ("prefill", "decode") if kind is None else (kind,)
        for selected in kinds:
            self.plan.rows(selected)
            frame = self.frames[selected]
            if self._pending[selected] and (not frame.submitted or
                    (frame.event is not None and not frame.event.query())):
                raise RuntimeError("archive DMA has not completed for the submitted producing-stream frame")
        with self.allocator.target.scope():
            for selected in kinds:
                self.plan.rows(selected)
                frame = self.frames[selected]
                for index, (origin, end) in self._pending[selected].items():
                    count = end - origin
                    source = frame.host[index].data_ptr()
                    target = self.archive[index].data_ptr() + origin * self.plan.index_dim * 2
                    if count:
                        ctypes.memmove(target, source, count * self.plan.index_dim * 2)
                    # A rewritten branch invalidates its former suffix, just
                    # as the existing cache ownership/kept-prefix protocol does.
                    self._available[index] = end
                self._last[selected].update(self._pending[selected])
                self._pending[selected].clear()
                frame.submitted = False
        self._outstanding = any(self._pending.values())
        if not self._outstanding:
            self._upload_pending = False

    def commit(self, lengths):
        self.check()
        values = self._lengths(lengths)
        if any(value > available for value, available in zip(values, self._available)):
            raise ValueError("committed indexer prefix is not present in validated RAM history")
        self._published[:] = values

    def _synchronize(self):
        if self.ring is not None and self.ring.is_cuda:
            import torch
            torch.cuda.synchronize(self.ring.device)

    def abort(self, kind=None, *, fenced=False):
        if not fenced and (self._outstanding or self._upload_pending):
            self._synchronize()
        for selected in (("prefill", "decode") if kind is None else (kind,)):
            self.plan.rows(selected)
            self._pending[selected].clear()
        self._outstanding = any(self._pending.values())
        if not self._outstanding:
            self._upload_pending = False

    def _idle(self):
        self.check()
        if any(self._pending.values()):
            raise RuntimeError("indexer frame must be validated or explicitly aborted before a lifecycle operation")
        if self._upload_pending:
            self._synchronize()
            self._upload_pending = False

    def _reload(self, lengths):
        p = self.plan
        with self.allocator.target.scope():
            ctypes.memset(self.restore_host.data_ptr(), 0, self.restore_host.numel() * 2)
            for index, length in enumerate(lengths):
                start = max(0, length - p.ring_rows)
                count = length - start
                first = start % p.ring_rows
                for source, target, rows in ((start, first, min(count, p.ring_rows - first)),
                                             (start + min(count, p.ring_rows - first), 0,
                                              max(0, count - (p.ring_rows - first)))):
                    if rows:
                        ctypes.memmove(self.restore_host[index].data_ptr() + target * p.index_dim * 2,
                                       self.archive[index].data_ptr() + source * p.index_dim * 2,
                                       rows * p.index_dim * 2)
        self._upload_pending = self.ring.is_cuda
        self.ring.copy_(self.restore_host, non_blocking=True)

    def snapshot(self, lengths):
        self.check()
        values = self._lengths(lengths)
        if any(value > available for value, available in zip(values, self._available)):
            raise ValueError("snapshot asks for unvalidated raw history")
        return {"format": FORMAT, "layers": self.plan.layers, "index_dim": self.plan.index_dim,
                "ratio": self.plan.ratio, "lengths": values}

    def validate_snapshot(self, metadata, lengths):
        self.check()
        values = self._lengths(lengths)
        expected = self.snapshot(values)
        if type(metadata) is not dict or metadata != expected:
            raise ValueError("snapshot has incompatible or unvalidated indexer history")

    def restore(self, metadata, lengths):
        self._idle()
        self.validate_snapshot(metadata, lengths)
        values = self._lengths(lengths)
        self._epoch += 1
        self._last = {"prefill": {}, "decode": {}}
        self._reload(values)
        self._published[:] = values

    def validate_copy(self, source, lengths):
        self._idle()
        source._idle()
        values = self._lengths(lengths)
        if (self.plan.layers, self.plan.index_dim, self.plan.ratio, str(self.device)) != (
                source.plan.layers, source.plan.index_dim, source.plan.ratio, str(source.device)):
            raise ValueError("indexer prefix copy requires matching model/device geometry")
        if source.allocator.target != self.allocator.target:
            raise ValueError("indexer prefix copies must retain the same NUMA-local allocation target")
        source.snapshot(values)

    def copy_prefix(self, source, lengths):
        self._copy(source, lengths, lengths)

    def _copy(self, source, lengths, restore_lengths):
        self.validate_copy(source, lengths)
        values = self._lengths(lengths)
        restored = self._lengths(restore_lengths)
        self._epoch += 1
        self._last = {"prefill": {}, "decode": {}}
        with self.allocator.target.scope():
            for index, rows in enumerate(values):
                if rows:
                    ctypes.memmove(self.archive[index].data_ptr(), source.archive[index].data_ptr(),
                                   rows * self.plan.index_dim * 2)
        self._available[:] = values
        self._published[:] = restored
        self._reload(restored)

    def clone(self):
        self._idle()
        result = self.session.new(self.limit)
        result._copy(self, self._available, self._published)
        return result

    def reset(self):
        self.check()
        self.abort()
        self._epoch += 1
        self._last = {"prefill": {}, "decode": {}}
        self._published[:] = [0] * self.plan.layers
        # Raw bytes and their availability survive a rewind until overwritten;
        # a valid prior snapshot can therefore still be restored losslessly.

    def close(self, *, fenced=False):
        if self.closed:
            return
        if not fenced and (self._outstanding or self._upload_pending):
            self._synchronize()
        self._pending = {"prefill": {}, "decode": {}}
        self._outstanding = self._upload_pending = False
        self.ring = self.origins = None
        for frame in self.frames.values():
            frame.gpu = frame.host = None
        self.archive = self.restore_host = self.origins_host = None
        failed, errors = [], []
        for resource, size in reversed(self._resources):
            try:
                resource.close()
            except BaseException as error:
                failed.append((resource, size))
                errors.append(error)
        self._resources = list(reversed(failed))
        self.closed = not self._resources
        if errors:
            self._poison = errors[0]
        raise_failures(None, errors)

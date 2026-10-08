"""Compact EXL3 CPU authority and upload-only logical-expert caching.

Original gate/up/down trellis bytes are retained without dequantization,
requantization, uniform CPU padding or logical route remapping. Prepared FP16
scale tables stay logically indexed on the execution device. The inherited
HostExpertCache owns aging LFU, weight staging, stream ordering and eviction.
Borrowed logical pointer tables are usable only for the IDs in a live lease.

The --vram-experts budget bounds packed GPU weight cells. Bounded publication
controls (32 bytes/device cell and 64 pinned bytes/cell) are separately admitted,
as are per-forward wave controls. Host authority includes compact payload plus
28 bytes/logical expert of tracked CPU starts, half-bit widths and byte counts.
"""

from __future__ import annotations

from bisect import bisect_left
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
import math
import struct
import threading
import weakref

import numpy as np
import torch
import triton
import triton.language as tl

from tensorfold.cuda.expert_cache import HostExpertCache
from tensorfold.cuda.direct_read import ReadAhead
from . import experts as native
from . import format as fmt

_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")
_DTYPE = {"I16": torch.int16, "I32": torch.int32, "U32": torch.int32, "F16": torch.float16}
_READ_WINDOW = 16 << 20
_READ_RUN = _READ_WINDOW - 3 * 4096
_READ_EXPERTS = 64


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class CompactAuthority:
    """Transferred immutable CPU ownership, including every alias of its tensors."""

    data: torch.Tensor
    starts: torch.Tensor
    k2: torch.Tensor
    trellis_bytes: torch.Tensor
    dims: int
    width: int
    codebook: str

    @property
    def count(self):
        return self.starts.numel()

    @property
    def source(self):
        return self.data, self.starts, self.k2, self.trellis_bytes

    @property
    def payload_bytes(self):
        return self.data.nbytes

    @property
    def metadata_bytes(self):
        return self.starts.nbytes + self.k2.nbytes + self.trellis_bytes.nbytes

    @property
    def max_entry_bytes(self):
        return int(self.trellis_bytes.max())

    def projection(self, expert, projection):
        """Read-only-by-contract CPU view for byte oracles, not a mutation API."""

        if type(expert) is not int or not 0 <= expert < self.count or type(projection) is not int or \
                projection not in (0, 1, 2):
            raise ValueError("invalid logical expert or projection")
        sizes = self.k2[expert].to(torch.int64) * (self.dims * self.width // 16)
        offset = int(self.starts[expert]) + int(sizes[:projection].sum())
        size = int(sizes[projection])
        k, n = (self.width, self.dims) if projection == 2 else (self.dims, self.width)
        return self.data.narrow(0, offset, size).view(torch.int16).view(k // 16, n // 16, int(self.k2[expert, projection]) * 8)


@dataclass
class _CompactLayer:
    authority: CompactAuthority
    source: tuple[torch.Tensor, ...]
    versions: tuple[int, ...]
    views: tuple[torch.Tensor, ...]
    count: int
    fields: tuple = ()


def _validate_authority(authority):
    if not isinstance(authority, CompactAuthority):
        raise ValueError("compact authority must be an owned CompactAuthority")
    if any(not isinstance(t, torch.Tensor) or t.device.type != "cpu" or t.layout != torch.strided or
           t.is_quantized or not t.is_contiguous() or t.requires_grad or torch.is_inference(t)
           for t in authority.source):
        raise ValueError("compact authority needs ordinary immutable contiguous CPU tensors")
    if authority.data.dtype != torch.uint8 or authority.data.ndim != 1 or authority.data.is_pinned():
        raise ValueError("compact trellis authority must be a pageable CPU byte vector")
    count = authority.count
    if not 0 < count <= 1024 or authority.starts.dtype != torch.int64 or tuple(authority.starts.shape) != (count,):
        raise ValueError("compact authority requires 1..1024 experts and int64 starts")
    if authority.k2.dtype != torch.int32 or tuple(authority.k2.shape) != (count, 3):
        raise ValueError("compact authority half-bit widths must be int32 [E,3]")
    if authority.trellis_bytes.dtype != torch.int64 or tuple(authority.trellis_bytes.shape) != (count,):
        raise ValueError("compact authority byte counts must be int64 [E]")
    _integer(authority.dims, "model dimensions")
    _integer(authority.width, "expert width")
    if authority.dims % 128 or authority.width % 128:
        raise ValueError("compact expert dimensions must be positive multiples of 128")
    native.codebook_id(authority.codebook)
    if bool(((authority.k2 < 2) | (authority.k2 > 16)).any()):
        raise ValueError("compact authority half-bit widths are unsupported")
    for half_bits in torch.unique(authority.k2).tolist():
        fmt.check_bits(half_bits / 2)
        if half_bits % 2 and authority.codebook != "mul1":
            raise ValueError("half-integer EXL3 widths require the mul1 codebook")
    expected = authority.k2.to(torch.int64).sum(-1) * (authority.dims * authority.width // 16)
    if not torch.equal(expected, authority.trellis_bytes):
        raise ValueError("compact authority byte counts disagree with original widths")
    end = 0
    for start, size in zip(authority.starts.tolist(), expected.tolist()):
        if start != end or size <= 0 or size % 16:
            raise ValueError("compact authority entries must be consecutive, aligned and nonempty")
        end += size
    if end != authority.data.numel():
        raise ValueError("compact authority does not exactly cover its payload")


@triton.jit
def _publish(CONTROL, GATE, UP, DOWN, N, B: tl.constexpr):
    i = tl.program_id(0) * B + tl.arange(0, B)
    valid = i < N
    logical = tl.load(CONTROL + i * 4, valid, other=0)
    gate = tl.load(CONTROL + i * 4 + 1, valid, other=0)
    up = tl.load(CONTROL + i * 4 + 2, valid, other=0)
    down = tl.load(CONTROL + i * 4 + 3, valid, other=0)
    tl.store(GATE + logical, gate, valid)
    tl.store(UP + logical, up, valid)
    tl.store(DOWN + logical, down, valid)


class Exl3HostExpertCache(HostExpertCache):
    """Variable-size compact payloads using the existing cache lease protocol."""

    def __init__(self, gpu_bytes, entry_bytes, device, *, staging_slots=2):
        super().__init__(gpu_bytes, entry_bytes, device, staging_slots=staging_slots)
        if self._pool.data_ptr() % 16 or self._pool.data_ptr() + self.gpu_bytes > 2**63 - 1:
            raise ValueError("EXL3 cells require aligned signed-int64 CUDA addresses")
        with torch.inference_mode(False):
            self._cells = self._pool.view(self.capacity, self.entry_bytes)
            self._publication_host = torch.empty((2, self.capacity, 4), dtype=torch.int64, pin_memory=True)
            self._publication_device = torch.empty((self.capacity, 4), dtype=torch.int64, device=self.device)
        self._publication_events = [torch.cuda.Event(), torch.cuda.Event()]
        self._publication_pending = [False, False]
        self._publication_stage = 0
        self._header_keys = None
        self._pack_ref = None
        self._sealed = False

    @property
    def control_device_bytes(self):
        return self.capacity * 32

    def _replacement_views(self, pool, capacity):
        if pool.data_ptr() % 16 or pool.data_ptr() + pool.numel() > 2**63 - 1:
            raise ValueError("EXL3 cells require aligned signed-int64 CUDA addresses")
        cells = pool.view(capacity, self.entry_bytes)
        host = torch.empty((2, capacity, 4), dtype=torch.int64, pin_memory=True)
        device = torch.empty((capacity, 4), dtype=torch.int64, device=self.device)
        return {key: (cells,) for key in self._layers}, (cells, host, device)

    def _commit_replacement(self, controls):
        self._cells, self._publication_host, self._publication_device = controls

    @property
    def control_host_bytes(self):
        return self.capacity * 64

    @property
    def host_payload_bytes(self):
        return sum(layer.authority.payload_bytes for layer in self._layers.values())

    @property
    def metadata_host_bytes(self):
        return sum(layer.authority.metadata_bytes for layer in self._layers.values())

    def register(self, layer_id, authority):
        _validate_authority(authority)
        with self._lock:
            self._usable()
            if self._sealed:
                raise RuntimeError("EXL3 cache registrations are sealed")
            if authority.max_entry_bytes > self.entry_bytes:
                raise ValueError("an original EXL3 expert does not fit the fixed GPU cell")
            self._policy.register(layer_id, authority.count)
            source = authority.source
            self._layers[layer_id] = _CompactLayer(authority, source, tuple(t._version for t in source),
                                                    (self._cells,), authority.count)

    def _copy(self, layer, expert, slot, stream):
        stage = self._next_stage
        self._next_stage = (stage + 1) % self.staging_slots
        event = self._stage_events[stage]
        if self._stage_pending[stage] and not event.query():
            event.synchronize()
        start = int(layer.authority.starts[expert])
        size = int(layer.authority.trellis_bytes[expert])
        pinned = self._staging[stage, :size]
        pinned.copy_(layer.authority.data.narrow(0, start, size))
        self._cells[slot, :size].copy_(pinned, non_blocking=True)
        event.record(stream)
        self._stage_pending[stage] = True
        return size

    def projection_keys(self, pk, prefix, shared):
        """Sorted header index resolves each layer without repeated whole-pack scans."""

        if self._sealed:
            raise RuntimeError("EXL3 cache registrations are sealed")
        if self._header_keys is None:
            self._header_keys = tuple(sorted(pk.where))
            self._pack_ref = weakref.ref(pk)
        elif self._pack_ref() is not pk:
            raise ValueError("one EXL3 host cache belongs to one checkpoint reader")
        result = {}
        for base in (prefix + ".", shared + "."):
            # The successor of the final '.' bounds the entire prefix,
            # including every valid Unicode tensor-name suffix.
            start, end = bisect_left(self._header_keys, base), bisect_left(self._header_keys, base[:-1] + "/")
            for key in self._header_keys[start:end]:
                projection, _, part = key.rpartition(".")
                result.setdefault(projection, set()).add(part)
        return result

    def finish_loading(self):
        """Release startup-only header references and freeze the authority registry."""

        with self._lock:
            self._usable()
            self._header_keys = self._pack_ref = None
            self._sealed = True

    def publish(self, layer_id, tables, ids, mapping):
        """Called inside a live parent lease, before its consumers are queued."""

        if not self._active:
            raise RuntimeError("EXL3 pointer publication requires a live cache lease")
        if not ids:
            return
        stage = self._publication_stage
        self._publication_stage = (stage + 1) % 2
        event = self._publication_events[stage]
        if self._publication_pending[stage] and not event.query():
            event.synchronize()
        host = self._publication_host[stage, :len(ids)]
        destination = host.numpy()
        authority = self._layers[layer_id].authority
        unit = authority.dims * authority.width // 16
        widths = authority.k2.numpy()
        base = self._pool.data_ptr()
        for row, logical in enumerate(ids):
            gate = base + mapping[logical] * self.entry_bytes
            up = gate + unit * int(widths[logical, 0])
            down = up + unit * int(widths[logical, 1])
            destination[row] = logical, gate, up, down
        self._publication_device[:len(ids)].copy_(host, non_blocking=True)
        stream = torch.cuda.current_stream(self.device)
        event.record(stream)
        self._publication_pending[stage] = True
        _publish[(triton.cdiv(len(ids), 128),)](self._publication_device, tables.gate_ptr, tables.up_ptr,
                                             tables.down_ptr, len(ids), B=128, num_warps=4)

    def close(self):
        super().close()
        self._cells = self._publication_host = self._publication_device = None
        self._header_keys = self._pack_ref = None


class CachedExl3Experts:
    """Native logical-expert metadata; every grouped consumer needs a live lease."""

    kernel = "exl3"
    capturable = False

    def __init__(self, authority, tables, cache, layer_id):
        self.cache, self.layer_id, self._tables = cache, layer_id, tables
        for field in ("count", "dims", "width", "cb", "k2_gu", "k2_d", "trellis_bytes"):
            setattr(self, field, getattr(tables, field))
        self._immutable = tuple(getattr(tables, field) for field in
                                ("gate_k2", "up_k2", "down_k2", "suh_g", "suh_u", "svh_g", "svh_u", "suh_d", "svh_d"))
        self._versions = tuple(t._version for t in self._immutable)
        cache.register(layer_id, authority)

    @property
    def capacity(self):
        return self.cache.capacity

    def bytes_per_expert(self):
        return self.cache.entry_bytes

    def nbytes_read(self, ids):
        return self._tables.nbytes_read(ids)

    def device_tensors(self):
        """Report shared device storage ownership for deduplicated model accounting."""

        for owner in (self.cache._pool, self.cache._publication_device, self._tables):
            if owner is not None:
                yield owner

    @contextmanager
    def lease(self, ids):
        if tuple(t._version for t in self._immutable) != self._versions:
            raise ValueError("logical EXL3 scale/width tables were mutated")
        with self.cache.lease(self.layer_id, ids) as (_, mapping):
            try:
                self.cache.publish(self.layer_id, self._tables, ids, mapping)
            except BaseException as error:
                self.cache._failure = repr(error)
                raise
            yield self._tables


def _parts(pk, projection, keys):
    if set(keys) - set(fmt.PARTS):
        raise ValueError(f"{projection}: unexpected EXL3 projection parts")
    if {"su", "suh"} <= set(keys) or {"sv", "svh"} <= set(keys):
        raise ValueError(f"{projection}: ambiguous duplicate scale representations")
    result = {}
    for key in keys:
        _, begin, end, dtype, shape = pk.entry(projection + "." + key)
        if dtype not in _DTYPE or any(type(size) is not int or size <= 0 for size in shape):
            raise ValueError(f"{projection}.{key}: invalid EXL3 dtype or dimensions")
        expected = math.prod(shape) * (4 if dtype in ("I32", "U32") else 2)
        if type(begin) is not int or type(end) is not int or begin < 0 or end - begin != expected:
            raise ValueError(f"{projection}.{key}: payload length disagrees with shape")
        result[key] = {"dtype": dtype, "shape": shape}
    return result


class _PackRanges:
    """Borrow checkpoint range transport; the owning Pack closes its Reader."""

    def __init__(self, pk):
        self.pk = pk

    def read(self, file, offset, size):
        value = self.pk.read(file, offset, offset + size)
        if not isinstance(value, torch.Tensor) or value.device.type != "cpu" or value.dtype != torch.uint8 or \
                value.ndim != 1 or not value.is_contiguous() or value.numel() != size:
            raise ValueError(f"{file}@{offset}: checkpoint range must return exact contiguous CPU bytes")
        return value


def _read_bound(items):
    """Match gap0 ReadAhead groups, including Reader alignment-copy storage."""

    files = {}
    for _, file, begin, end, _ in items:
        files.setdefault(file, []).append((begin, end))
    total = 0
    for spans in files.values():
        start = end = None
        for begin, stop in sorted(spans):
            if start is not None and (begin > end or stop - start > _READ_RUN):
                total += ((end + 4095) // 4096 - start // 4096 + 1) * 4096
                start = None
            if start is None:
                start, end = begin, stop
            else:
                end = max(end, stop)
        if start is not None:
            total += ((end + 4095) // 4096 - start // 4096 + 1) * 4096
    return total


def _finish_reads(ahead, queued, *, close=True):
    # A take interrupted while waiting has already removed its lookup key.
    # Retain every queued future until the entire window has been consumed.
    pending = dict.fromkeys((*queued, *ahead.ahead.values()))
    errors = []
    for future in pending:
        try:
            future.cancel()
        except BaseException as error:
            errors.append(error)
    try:
        if close:
            ahead.close()
        else:
            ahead.ahead.clear()
    except BaseException as error:
        errors.append(error)
    remaining = {}
    for future in pending:
        try:
            if not future.cancelled():
                future.result()  # also joins work if shutdown itself was interrupted
        except BaseException as error:
            errors.append(error)
            try:
                if not future.done():
                    remaining[future] = None
            except BaseException as status_error:
                errors.append(status_error)
                remaining[future] = None
    queued.clear()
    queued.update(remaining)  # interrupted waits remain owned through final shutdown
    return errors


class CompactReadSession:
    """One operation-owned CPU reader; each layer drains before the next starts.

    The owning loader keeps Pack alive and closes this session before returning
    weights. A failed layer poisons reuse. Standalone load_compact owns a session
    for its one layer. No serving worker or cross-layer queue is retained.
    """

    def __init__(self, pk):
        self.thread = threading.current_thread()
        self.pk = pk
        self.ahead = ReadAhead(reader=_PackRanges(pk), threads=1, run=_READ_RUN, gap=0)
        self.queued = {}
        self.active = False
        self.closed = False
        self.poisoned = False

    def __enter__(self):
        self.validate(self.pk)
        return self

    def validate(self, pk):
        if threading.current_thread() is not self.thread:
            raise RuntimeError("EXL3 read session is owned by another thread")
        if self.closed or self.poisoned or self.active or self.ahead.ahead or self.queued:
            raise RuntimeError("EXL3 read session is unavailable or retains pending layer reads")
        if self.pk is not pk:
            raise ValueError("EXL3 read session must borrow this checkpoint Pack")

    def __exit__(self, kind, primary, traceback):
        try:
            self.close()
        except BaseException as cleanup:
            if primary is None:
                raise
            if cleanup is not primary:
                primary.add_note(f"EXL3 read session cleanup also failed: {cleanup!r}")
                for note in getattr(cleanup, "__notes__", ()):
                    primary.add_note(f"EXL3 read session cleanup detail: {note}")

    def close(self):
        if threading.current_thread() is not self.thread:
            raise RuntimeError("EXL3 read session is owned by another thread")
        if self.closed:
            return
        if self.active:
            raise RuntimeError("cannot close EXL3 reader while a layer is consuming payloads")
        pending = tuple(dict.fromkeys((*self.queued, *self.ahead.ahead.values())))
        errors = []
        try:
            errors.extend(_finish_reads(self.ahead, self.queued))
        except BaseException as error:
            errors.append(error)
        # shutdown(wait=True) can be interrupted before returning. Retain the
        # owner and retry that supported shutdown once, preserving the failure.
        if self.ahead.pool is not None:
            try:
                errors.extend(_finish_reads(self.ahead, self.queued))
            except BaseException as error:
                errors.append(error)
        if self.ahead.pool is not None:
            try:
                self.ahead.close()  # joining cannot depend on a failed drain helper
            except BaseException as error:
                errors.append(error)
        # Joining is separate from observing asynchronous result statuses.
        # The original snapshot survives a failed helper and its partial clears.
        # A status observation failure leaves that future owned for retry; never
        # release Pack while work or an unobserved result remains outstanding.
        remaining = {}
        for future in pending:
            try:
                if not future.done():
                    remaining[future] = None
                elif not future.cancelled():
                    failure = future.exception(timeout=0)
                    if failure is not None and not any(failure is error for error in errors):
                        errors.append(failure)
            except BaseException as error:
                errors.append(error)
                remaining[future] = None
        self.queued.clear()
        self.queued.update(remaining)
        if not remaining:
            self.ahead.ahead.clear()
        self.closed = self.ahead.pool is None and not self.queued and not self.ahead.ahead
        self.poisoned |= bool(errors)
        if self.closed:
            self.pk = None
            self.ahead.reader.pk = None
        if not self.closed:
            errors.append(RuntimeError("EXL3 reader shutdown or future status observation is incomplete"))
        if errors:
            primary = errors[0]
            for secondary in errors[1:]:
                if secondary is not primary:
                    primary.add_note(f"secondary EXL3 read cleanup failure: {secondary!r}")
            raise primary


def _payload_iterator(pk, metadata, ahead, queued):
    """Bounded complete bundles; borrowed views die after the consuming copy.

    At most16MiB of Reader blocks and64 experts are queued. One worker plus
    possible Reader alignment relocation needs at most32MiB payload scratch;
    view/control metadata is separately bounded. No per-part clone is made.
    This iterator must be closed if its consumer fails.
    """

    bundles = []
    for expert, metas in enumerate(metadata):
        items = []
        for meta in metas:
            parts = ("trellis", meta.in_scales, meta.out_scales)
            if meta.codebook in fmt.MARKERS:
                parts += (meta.codebook,)
            for part in parts:
                name = f"{meta.prefix}.{part}"
                file, begin, end, dtype, shape = pk.entry(name)
                items.append((name, file, begin, end, (dtype, tuple(shape))))
        bundles.append((expert, items))
    bundles.sort(key=lambda value: min((item[1], item[2]) for item in value[1]))
    oversized = [_read_bound(items) > _READ_WINDOW for _, items in bundles]
    at = 0
    while at < len(bundles):
        if oversized[at]:
            # Few large sequential projections preserve the original streaming
            # region and its admitted2*largest temporary bound. No new size floor.
            expert, entries = bundles[at]
            for name, _, _, _, _ in entries:
                part = name.rsplit(".", 1)[1]
                if part in fmt.MARKERS:
                    marker = pk.get(name)
                    if marker.device.type != "cpu" or marker.dtype != torch.int32 or marker.numel() != 1 or \
                            int(marker.item()) & 0xFFFFFFFF != fmt.MARKERS[part]:
                        raise ValueError(f"{name}: invalid EXL3 codebook marker payload")
                    del marker
            for projection, meta in enumerate(metadata[expert]):
                payload = {f"{meta.prefix}.{part}": pk.get(f"{meta.prefix}.{part}")
                           for part in ("trellis", meta.in_scales, meta.out_scales)}
                yield expert, projection, payload
                del payload
            at += 1
            continue
        selected, items = [], []
        while at < len(bundles) and not oversized[at] and len(selected) < _READ_EXPERTS:
            expert, addition = bundles[at]
            proposed = items + addition
            if _read_bound(proposed) > _READ_WINDOW:
                if not selected:
                    raise ValueError("an EXL3 expert exceeds bounded joint-read scratch")
                break
            selected.append((expert, addition))
            items = proposed
            at += 1
        # Each returned slice owns a reference to its immutable Reader
        # block; no block/stage is reused while a borrowed view survives.
        ahead.queue(items, cut=lambda raw, _: raw)
        queued.update((future, None) for future in ahead.ahead.values())
        for name, _, _, _, (dtype, _) in items:
            part = name.rsplit(".", 1)[1]
            if part in fmt.MARKERS:
                raw = ahead.take(name)
                if raw is None or raw.numel() != 4 or dtype not in ("I32", "U32") or \
                        struct.unpack("<I", memoryview(raw.numpy()))[0] != fmt.MARKERS[part]:
                    raise ValueError(f"{name}: invalid EXL3 codebook marker payload")
                # Scalar markers can be unaligned in serialized files.
                # struct.unpack accepts bytes without a typed tensor view.
                del raw
        for expert, entries in selected:
            by_name = {item[0]: item for item in entries}
            for projection, meta in enumerate(metadata[expert]):
                payload = {}
                for part in ("trellis", meta.in_scales, meta.out_scales):
                    name = f"{meta.prefix}.{part}"
                    _, _, _, _, (dtype, shape) = by_name[name]
                    raw = ahead.take(name)
                    if raw is None or raw.storage_offset() % 2:
                        raise ValueError(f"{name}: unaligned or missing typed EXL3 payload")
                    payload[name] = raw.view(_DTYPE[dtype]).reshape(shape)
                    del raw
                yield expert, projection, payload
                del payload
        if ahead.ahead:
            raise RuntimeError("an EXL3 joint-read window retained unconsumed parts")
        queued.clear()


@contextmanager
def _joint_payloads(pk, metadata, reads=None):
    """Own and drain the bounded read pipeline through consumer failure."""

    owner = CompactReadSession(pk) if reads is None else nullcontext(reads)
    with owner as session:
        if not isinstance(session, CompactReadSession):
            raise ValueError("EXL3 read session must be an owned CompactReadSession")
        session.validate(pk)
        session.active = True
        iterator = _payload_iterator(pk, metadata, session.ahead, session.queued)
        primary = None
        try:
            yield iterator
        except BaseException as error:
            primary = error
            raise
        finally:
            errors = []
            try:
                iterator.close()
            except BaseException as error:
                errors.append(error)
            try:
                errors.extend(_finish_reads(session.ahead, session.queued, close=False))
            except BaseException as error:
                errors.append(error)
            finally:
                session.active = False
                session.poisoned |= primary is not None or bool(errors)
            if errors:
                if primary is None:
                    primary = errors[0]
                    for secondary in errors[1:]:
                        if secondary is not primary:
                            primary.add_note(f"secondary EXL3 read cleanup failure: {secondary!r}")
                    raise primary
                for secondary in errors:
                    if secondary is not primary:
                        primary.add_note(f"secondary EXL3 read cleanup failure: {secondary!r}")


def load_compact(pk, prefix, count, shared, *, device="cpu", keys=None, reads=None):
    """Validated compact authority plus native scale/width tables, without GPU trellises.

    CPU ``device`` is an isolated constructor/oracle mode, not a serving fallback.
    Markers and all headers are validated before allocating expert payload storage.
    One layer's FP16 scales and bounded joint-read windows are temporary. The
    reader implements where/entry/get/read; read returns exact CPU byte ranges.
    """

    if reads is not None:
        if not isinstance(reads, CompactReadSession):
            raise ValueError("EXL3 read session must be an owned CompactReadSession")
        reads.validate(pk)
    _integer(count, "routed expert count")
    if count >= 1024:
        raise ValueError("EXL3 routing supports at most 1023 routed experts plus shared")
    names = [f"{prefix}.{expert}" for expert in range(count)] + [shared]
    if keys is None:
        keys = {}
        for key in pk.where:
            if key.startswith((prefix + ".", shared + ".")):
                projection, _, part = key.rpartition(".")
                keys.setdefault(projection, set()).add(part)
    metadata, codebooks = [], set()
    wanted = {f"{name}.{projection}" for name in names for projection in _PROJECTIONS}
    if set(keys) != wanted:
        raise ValueError("missing or unexpected EXL3 expert projection")
    dims = width = None
    total, starts, bits, counts = 0, [], [], []
    for expert, name in enumerate(names):
        starts.append(total)
        widths, metas, size = [], [], 0
        for projection in _PROJECTIONS:
            group = f"{name}.{projection}"
            parts = _parts(pk, group, keys[group])
            meta = fmt.parse_group(group, parts)
            if meta.bias:
                raise ValueError(f"{group}: biased EXL3 experts are unsupported")
            if dims is None:
                dims, width = meta.k, meta.n
            expected = (width, dims) if projection == "down_proj" else (dims, width)
            if (meta.k, meta.n) != expected:
                raise ValueError(f"{group}: routed/shared expert dimensions disagree")
            codebooks.add(meta.codebook)
            if meta.codebook in fmt.MARKERS:
                marker_name = f"{group}.{meta.codebook}"
                info = parts[meta.codebook]
                marker = pk.get(marker_name)
                if info["dtype"] not in ("I32", "U32") or info["shape"] not in ([], [1]) or \
                        marker.device.type != "cpu" or marker.dtype != torch.int32 or marker.numel() != 1 or \
                        int(marker.item()) & 0xFFFFFFFF != fmt.MARKERS[meta.codebook]:
                    raise ValueError(f"{marker_name}: invalid EXL3 codebook marker payload")
            widths.append(int(meta.bits * 2))
            metas.append(meta)
            size += meta.trellis_bytes
        total += size
        if total > 2**63 - 1:
            raise ValueError("EXL3 compact payload exceeds tensor-storage bounds")
        bits.append(widths)
        counts.append(size)
        metadata.append(metas)
    if len(codebooks) != 1:
        raise ValueError("routed/shared EXL3 experts mix codebooks")
    with torch.inference_mode(False):
        data = torch.empty((total,), dtype=torch.uint8, device="cpu")
        start_tensor = torch.tensor(starts, dtype=torch.int64)
        bit_tensor = torch.tensor(bits, dtype=torch.int32)
        count_tensor = torch.tensor(counts, dtype=torch.int64)
        scale_tables = [torch.empty((count + 1, n), dtype=torch.float16) for n in
                        (dims, dims, width, width, width, dims)]
    with _joint_payloads(pk, metadata, reads) as payloads:
        for expert, projection, payload in payloads:
            offset = starts[expert] + (dims * width // 16) * sum(bits[expert][:projection])
            meta = metadata[expert][projection]
            group = meta.prefix
            raw = payload[group + ".trellis"]
            if raw.device.type != "cpu" or raw.dtype != torch.int16 or not raw.is_contiguous() or \
                    tuple(raw.shape) != (meta.k // 16, meta.n // 16, 8 * bits[expert][projection]):
                raise ValueError(f"{group}: original trellis payload disagrees with validated metadata")
            data[offset:offset + meta.trellis_bytes].copy_(raw.view(torch.uint8).reshape(-1))
            for scale_index, part, length in ((0, meta.in_scales, meta.k), (1, meta.out_scales, meta.n)):
                value = payload[group + "." + part]
                if value.device.type != "cpu" or not value.is_contiguous():
                    raise ValueError(f"{group}.{part}: scales must be contiguous CPU values")
                if part in ("su", "sv"):
                    if value.dtype != torch.int16 or value.numel() != length // 16:
                        raise ValueError(f"{group}.{part}: invalid packed sign scales")
                    value = torch.from_numpy(fmt.unpack_signs(value.numpy()))
                elif value.dtype != torch.float16 or tuple(value.shape) != (length,):
                    raise ValueError(f"{group}.{part}: invalid FP16 scale payload")
                native.validate_scale_payload(value, f"{group}.{part}")
                target = ((0, 2), (1, 3), (4, 5))[projection][scale_index]
                scale_tables[target][expert].copy_(value)
            del raw, value, payload
    with torch.inference_mode(False):
        authority = CompactAuthority(data, start_tensor, bit_tensor, count_tensor, dims, width, codebooks.pop())
        _validate_authority(authority)
        # Blocking transfer finishes every CPU source use before temporary tables leave scope.
        scales = [value.to(device) for value in scale_tables]
        ptrs = [torch.zeros((count + 1,), dtype=torch.int64, device=device) for _ in _PROJECTIONS]
        widths = [bit_tensor[:, i].contiguous().to(device) for i in range(3)]
        gu = bit_tensor[:, :2]
        down = bit_tensor[:, 2]
        tables = native.Exl3RoutedExperts(*ptrs, *widths, *scales, count + 1, dims, width,
                                        native.codebook_id(authority.codebook), (int(gu.min()), int(gu.max())),
                                        (int(down.min()), int(down.max())), count_tensor, [])
    return authority, tables


def load_cached(pk, prefix, count, shared, cache, layer_id, device, *, reads=None):
    """The loader adapter: original compact CPU trellises, GPU logical metadata."""

    if not isinstance(cache, Exl3HostExpertCache):
        raise ValueError("EXL3 cached loading requires the compact EXL3 cache")
    destination = torch.device(device)
    if destination.type != "cuda" or destination.index not in (None, cache.device.index):
        raise ValueError("EXL3 metadata and weight cache must share a CUDA device")
    keys = cache.projection_keys(pk, prefix, shared)
    authority, tables = load_compact(pk, prefix, count, shared, device=cache.device, keys=keys, reads=reads)
    return CachedExl3Experts(authority, tables, cache, layer_id)


def logical_waves(pick, count, capacity):
    """Bounded deterministic logical-ID partitions; invalid native picks are skipped."""

    _integer(count, "expert count")
    _integer(capacity, "capacity")
    if not isinstance(pick, np.ndarray) or pick.dtype != np.int32 or pick.ndim != 2:
        raise ValueError("wave picks must be a CPU int32 [rows,slots] array")
    valid = pick[(pick >= 0) & (pick < count)]
    ids = np.unique(valid).tolist()
    return [ids[start:start + capacity] for start in range(0, len(ids), capacity)]


def _validate_native_picks(pick, count):
    """Reject duplicate valid per-row IDs before native member-buffer writes.

    The caller supplies the already validated, bounded CPU int32 snapshot.
    Invalid native IDs may repeat because their output slots are skipped.
    """

    ordered = np.sort(pick, axis=1)
    previous = ordered[:, :-1]
    duplicate = (previous == ordered[:, 1:]) & (previous >= 0) & (previous < count)
    if bool(duplicate.any()):
        raise ValueError("native EXL3 routing requires distinct valid experts within each row")


class WaveScratch:
    """One bounded D2H snapshot and reusable masked-pick H2D staging per forward."""

    def __init__(self, rows, slots, device):
        self.rows, self.slots = _integer(rows, "rows"), _integer(slots, "slots")
        if self.slots > 32:
            raise ValueError("native EXL3 members support at most 32 slots per row")
        destination = torch.device(device)
        if destination.type != "cuda":
            raise ValueError("EXL3 wave controls require a CUDA device")
        self.device = torch.device("cuda", torch.cuda.current_device() if destination.index is None else destination.index)
        with torch.inference_mode(False):
            self.original = torch.empty((rows, slots), dtype=torch.int32, pin_memory=True)
            self.masked = torch.empty((rows, slots), dtype=torch.int32, pin_memory=True)
            self.device_pick = torch.empty((rows, slots), dtype=torch.int32, device=self.device)
        self.ready, self.uploaded = torch.cuda.Event(), torch.cuda.Event()
        self.pending = False

    @property
    def device_bytes(self):
        return self.rows * self.slots * 4

    @property
    def host_bytes(self):
        return self.rows * self.slots * 8

    def read(self, pick, rows):
        if not 0 < rows <= self.rows or pick.dtype != torch.int32 or pick.device != self.device or \
                tuple(pick.shape) != (rows, self.slots) or not pick.is_contiguous():
            raise ValueError("EXL3 wave source does not match its bounded device window")
        self.original[:rows].copy_(pick, non_blocking=True)
        self.ready.record(torch.cuda.current_stream(self.device))
        self.ready.synchronize()
        return self.original[:rows].numpy()

    def stage(self, source, active, count):
        if self.pending:
            self.uploaded.synchronize()
        rows = source.shape[0]
        target = self.masked[:rows].numpy()
        target[:] = source
        target[~np.isin(source, active)] = count
        self.device_pick[:rows].copy_(self.masked[:rows], non_blocking=True)
        self.uploaded.record(torch.cuda.current_stream(self.device))
        self.pending = True
        return self.device_pick[:rows]


def routed_cached(x, pick, cached, scratch, rows, *, limit=math.inf, act_mode=native.ACT_F32):
    """Compute each original row/slot once, retaining native output and ACT_F32 bits.

    Scratch has one host owner. Its caller must order borrowed y consumers
    before the next scratch write, including explicit cross-stream dependencies.
    Inactive wave picks use E; skipped slots retain scratch.y.
    Every valid original pair is written by one wave. Routing weights are applied
    once by the existing consuming writeback, after all expert waves complete.
    The bounded CPU routing snapshot verifies distinct valid experts within
    each row before leases or native kernels. Invalid native IDs may repeat.
    Wave controls additionally enforce the native limit of 32 slots per row.
    """

    if not isinstance(cached, CachedExl3Experts) or scratch.count_experts != cached.count:
        raise ValueError("cached EXL3 experts and native scratch disagree")
    _integer(rows, "rows")
    if x.device != cached.cache.device or x.dtype not in (torch.bfloat16, torch.float16) or x.ndim != 2 or \
            tuple(x.shape) != (rows, cached.dims) or x.stride(-1) != 1 or \
            scratch.xg.shape[-1] != cached.dims or scratch.xd.shape[-1] != cached.width:
        raise ValueError("EXL3 wave inputs and native scratch dimensions/device disagree")
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError("host-backed EXL3 forwards cannot be captured")
    controls = getattr(scratch, "host_waves", None)
    if controls is None:
        controls = scratch.host_waves = WaveScratch(scratch.rows, scratch.slots, x.device)
    source = controls.read(pick, rows)
    _validate_native_picks(source, cached.count)
    for ids in logical_waves(source, cached.count, cached.capacity):
        with cached.lease(ids) as hot:
            masked = controls.stage(source, ids, cached.count)
            native.routed(x, masked, None, hot, scratch, None, rows, limit=limit, act_mode=act_mode)
    return scratch.y[:rows * scratch.slots]


__all__ = ["CompactAuthority", "Exl3HostExpertCache", "CachedExl3Experts", "load_compact", "load_cached",
           "logical_waves", "WaveScratch", "routed_cached"]

# SPDX-License-Identifier: Apache-2.0
# DSpark block/Markov equations follow SGLang v0.5.21 and the attributed
# dspark_reference.py. This adapter adds explicit SM70/TP ownership boundaries.
"""Unregistered FP16 dense DSpark adapter; eager execution, no target mutation.

CUDA dense products borrow Half weights through the existing fixed-M32 Dense
interface (FP32 accumulation). Attention/normalization retain explicit FP32
arithmetic. CPU execution exists for independent tiny equation/TP oracles only;
it is not CUDA qualification. No CLI, State, target hook or provider selects it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from . import dspark_reference as ref
from .dspark_config import DSparkConfig
from .dspark_provider import DraftSample
from .dspark_weights import LOAD_CHUNK, SOURCE_BYTES, plan_weights, tensor_shapes
from .sm70_mtp_weights import Dense

EXECUTION_ID = "fp16-m32-f32acc-tp-rank-order-f32-attention-dspark-v1"
MAX_ROWS = 2048


def _int(value, name, lo, hi):
    if type(value) is not int or not lo <= value <= hi:
        raise ValueError(f"DSpark {name} must be an integer in [{lo},{hi}]")
    return value


def _plans(c, rank, world):
    header, offset = {}, 0
    for name, shape in tensor_shapes(c).items():
        count = math.prod(shape) * 2
        header[name] = {"shape": list(shape), "dtype": "F16", "data_offsets": [offset, offset + count]}
        offset += count
    return plan_weights(c, header, tp=(rank, world))


@dataclass(frozen=True)
class VocabShard:
    rank: int
    world: int
    start: int
    stop: int
    vocab: int

    @property
    def width(self):
        return self.stop - self.start

    @classmethod
    def make(cls, config, rank, world):
        config.partition(rank, world)
        if config.vocab < world:
            raise ValueError("DSpark TP vocabulary shards must be nonempty")
        width, extra = divmod(config.vocab, world)
        start = rank * width + min(rank, extra)
        result = cls(rank, world, start, start + width + int(rank < extra), config.vocab)
        if not result.width:
            raise ValueError("DSpark TP vocabulary shards must be nonempty")
        return result


@dataclass
class DeviceWeights:
    config: DSparkConfig
    tensors: dict
    checkpoint_identity: str
    rank: int = 0
    world: int = 1
    piece_hash: str = ""

    def __post_init__(self):
        plans = _plans(self.config, self.rank, self.world)
        if not isinstance(self.checkpoint_identity, str) or not self.checkpoint_identity:
            raise ValueError("DSpark requires a trusted common checkpoint identity")
        if self.tensors.keys() != {p.name for p in plans}:
            raise ValueError("DSpark local weight namespace differs")
        self.device = self.tensors[plans[0].name].device
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("DSpark dense weights require CPU oracle or CUDA storage")
        for plan in plans:
            t = self.tensors[plan.name]
            dtype = torch.float32 if plan.f32 else torch.float16
            if (t.device != self.device or t.dtype != dtype or tuple(t.shape) != plan.shape
                    or not t.is_contiguous()):
                raise ValueError(f"DSpark local weight geometry/device/dtype differs: {plan.name}")
        # The common identity deliberately excludes per-rank pieces; preload
        # consensus must include this identity and the partition policy.
        self.identity = hashlib.sha256(
            (self.config.identity + "|" + self.checkpoint_identity + "|" + EXECUTION_ID).encode()
        ).hexdigest()

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def device_tensors(self):
        return tuple(self.tensors.values())


def load_shard(directory, *, target, checkpoint_identity, tp=(0, 1), device="cuda", chunk_bytes=LOAD_CHUNK,
               _journal=None):
    """Bounded local safetensors slices; no download or full FP32 weight copy.

    checkpoint_identity is an already verified full source identity supplied by
    admission. piece_hash additionally attests the exact source slices read by
    this rank. A CPU device permits loader/oracle tests without exposing a GPU.
    """
    from safetensors import safe_open
    from tensorfold.cuda.tensor_file import read_header

    c = DSparkConfig.read(directory)
    c.check_target(target)
    rank, world = tp
    file = Path(directory) / "model.safetensors"
    _, header = read_header(file, SOURCE_BYTES)
    plans = plan_weights(c, header, tp=tp)
    minimum = max(math.prod(p.source_shape[1:]) * SOURCE_BYTES[header[p.name]["dtype"]] for p in plans)
    _int(chunk_bytes, "load chunk", minimum, 1 << 30)
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("DSpark loader requires CUDA or CPU oracle storage")
    if not isinstance(checkpoint_identity, str) or not checkpoint_identity:
        raise ValueError("DSpark requires a trusted common checkpoint identity")
    values = {}
    digest = hashlib.sha256((c.identity + f"|{rank}/{world}").encode())
    with safe_open(file, framework="pt", device="cpu") as source:
        for plan in plans:
            dtype = torch.float32 if plan.f32 else torch.float16
            result = torch.empty(plan.shape, dtype=dtype, device=device)
            if _journal is not None:
                _journal[plan.name] = result
            first, last = plan.rows or (0, plan.source_shape[0])
            row_bytes = math.prod(plan.source_shape[1:]) * SOURCE_BYTES[header[plan.name]["dtype"]]
            step = max(1, chunk_bytes // row_bytes)
            value = source.get_slice(plan.name)
            digest.update((plan.name + "|" + header[plan.name]["dtype"] + str((plan.rows, plan.columns))).encode())
            for start in range(first, last, step):
                stop = min(start + step, last)
                slices = (slice(start, stop),)
                if len(plan.source_shape) == 2:
                    slices += (slice(*(plan.columns or (0, plan.source_shape[1]))),)
                raw = value[slices].contiguous()
                if (not torch.isfinite(raw).all() or
                        (dtype == torch.float16 and raw.abs().max().float() > torch.finfo(torch.float16).max)):
                    raise ValueError(f"DSpark nonfinite/FP16-overflow source tensor: {plan.name}")
                digest.update(raw.view(torch.uint8).numpy().tobytes())
                result[start - first:stop - first].copy_(raw.to(dtype))
            values[plan.name] = result
    return DeviceWeights(c, values, checkpoint_identity, rank, world, digest.hexdigest())


def byte_plan(config, *, tp=(0, 1), capacity, slots=1, prefill_rows=256, block=None):
    """Exact owned weights/cache/staged KV, conservative eager operation bound.

    Target embedding/head and TapWorkspace are separately owned and charged.
    No graph budget is accepted by this eager adapter. Library workspace is a
    bound for admission review, not a measured allocator minimum.
    """
    rank, world = tp
    p = config.partition(rank, world)
    shard = VocabShard.make(config, rank, world)
    capacity = _int(capacity, "capacity", 1, config.context)
    slots = _int(slots, "slots", 1, 64)
    rows = _int(prefill_rows, "prefill rows", 1, MAX_ROWS)
    block = _int(config.block if block is None else block, "block", 1, min(32, config.block))
    h, d = config.hidden, config.head_dim
    q, k, wide = p["heads"] * d, p["kv_heads"] * d, p["intermediate"]
    resident = sum(x.nbytes for x in _plans(config, rank, world))
    cache = slots * config.layers * capacity * k * 4
    stage = slots * config.layers * rows * k * 4
    # Include padded Half input, FP32 product/result, caller normalization and
    # collective partials; CUDA never materializes a full FP32 weight matrix.
    dense = 32 * (max(h * len(config.taps), h, q, wide, config.rank) * 2
                  + max(h, q, k, wide, shard.width) * 4)
    dense += max(rows, block) * max(h, q, k, wide, shard.width) * 8
    context = rows * h * 24 + config.layers * rows * k * 4
    length = capacity + block
    attention = length * k * 12 + p["heads"] * block * length * 12 + block * q * 16
    proposal = block * (shard.width * 12 + h * 32 + wide * 12 + config.rank * 8)
    gather = world * max(rows, block) * h * 4 + world * 128
    transient = dense + max(context, attention + proposal) + gather
    return {"resident_bytes": resident, "state_bytes": cache, "context_staging_bytes": stage,
            "eager_transient_bound": transient, "device_bytes": resident + cache + stage + transient,
            "target_shared_weight_bytes": 0, "tap_workspace_bytes": 0,
            "full_f32_weight_cast_bytes": 0, "graph_bytes": 0,
            "planned_rank": rank, "planned_world": world}


def _event(device):
    return torch.cuda.Event() if device.type == "cuda" else None


def _wait(event, device):
    if event is not None:
        torch.cuda.current_stream(device).wait_event(event)


def _record(event, device):
    if event is not None:
        event.record(torch.cuda.current_stream(device))


def _borrow(tensors):
    for t in tensors:
        if t.is_cuda:
            t.record_stream(torch.cuda.current_stream(t.device))


@dataclass(frozen=True)
class _Stamp:
    version: object
    codec: object
    history: object

    @classmethod
    def read(cls, target):
        if getattr(target, "_kv_error", None) is not None or getattr(target, "_resize_error", None) is not None:
            raise ValueError("DSpark target storage is poisoned")
        return cls(getattr(target, "version", None), getattr(target, "kv_identity", None), getattr(target, "cur", None))

    def check(self, target):
        other = self.read(target)
        if self.version != other.version or self.codec != other.codec or self.history is not other.history:
            raise ValueError("DSpark target storage/reset identity changed; explicit reset/resize/restore required")


@dataclass
class DeviceContextUpdate:
    start: int
    keys: torch.Tensor | None
    values: torch.Tensor | None
    version: int
    owner: object
    rows: int
    ready: object = None

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.keys, self.values) if t is not None)

    def retire(self):
        self.keys = self.values = self.owner = self.ready = None


@dataclass
class DeviceState:
    target: object
    identity: str
    keys: torch.Tensor | None
    values: torch.Tensor | None
    stamp: _Stamp
    length: int = 0
    version: int = 0
    pending: DeviceContextUpdate | None = None
    ready: object = None
    closed: bool = False
    valid_length: int = 0  # physical prefix retained, including after a logical reset/rewind

    @property
    def capacity(self):
        return 0 if self.closed else self.keys.shape[1]

    @property
    def nbytes(self):
        return 0 if self.closed else sum(t.numel() * t.element_size() for t in (self.keys, self.values))

    def device_tensors(self):
        """Flat committed cache ownership for the existing slot memory gate."""
        return () if self.closed else (self.keys, self.values)


@dataclass
class DeviceSnapshot:
    identity: str
    keys: torch.Tensor
    values: torch.Tensor
    length: int
    ready: object = None

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.keys, self.values))


@dataclass(frozen=True)
class DeviceProposal:
    ids: torch.Tensor
    verify_ids: torch.Tensor
    corrected_logits: torch.Tensor  # rank-local vocabulary shard, Half
    probabilities: torch.Tensor | None  # actual rank-local q, or absent for explicit keyed choices
    confidence: torch.Tensor | None
    start: int
    state_version: int
    identity: str
    shard: VocabShard
    owner: object = field(repr=False, compare=False)
    ready: object = field(default=None, repr=False, compare=False)

    @property
    def nbytes(self):
        values = (self.ids, self.verify_ids, self.corrected_logits, self.probabilities, self.confidence)
        return sum(t.numel() * t.element_size() for t in values if t is not None)

    def wait(self):
        """Borrow exported tensors on the consumer's current CUDA stream."""
        _wait(self.ready, self.ids.device)
        _borrow(tuple(t for t in (self.ids, self.verify_ids, self.corrected_logits, self.probabilities) if t is not None))
        if self.confidence is not None:
            _borrow((self.confidence,))
        return self


@dataclass(frozen=True)
class PositionKeyedChoice:
    """Existing target-coupled sampler choice; no fabricated proposal q.

    Only the existing position-keyed target verifier may consume this mode.
    Standard rejection requires DraftSample with its actual probabilities.
    """
    ids: torch.Tensor


def gqa_attention(q, keys, values):
    """Committed prefix plus whole proposal block, intentionally bidirectional.

    FP32 score/probability/value arithmetic; full FP16 output boundary. Keeping
    this separate from BF16 draft_attention avoids dtype reinterpretation.
    """
    rows, heads, dim = q.shape
    kv = keys.shape[1]
    grouped = q.float().reshape(rows, kv, heads // kv, dim)
    scores = torch.einsum("rhgd,chd->hgrc", grouped, keys.float()) * (dim ** -0.5)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.einsum("hgrc,chd->rhgd", probabilities, values.float()).reshape(rows, heads * dim).half()


def context_norm(x, scale, eps):
    """Fixed32 token rows, including KV heads, for regrouped committed context.

    Torch may choose a different reduction tree from its output count. Context
    arriving as a tiny retained suffix must use the same tree as a long piece.
    Dummy rows never persist or participate in another row's normalization.
    """
    out = torch.empty_like(x)
    for start in range(0, x.shape[0], 32):
        count = min(32, x.shape[0] - start)
        padded = torch.zeros((32, *x.shape[1:]), dtype=x.dtype, device=x.device)
        padded[:count].copy_(x[start:start + count])
        out[start:start + count].copy_(ref.rms(padded, scale, eps)[:count])
    return out


class DSparkCUDA:
    """One explicit dense draft provider; caller retains target verification.

    embed(ids) shares target FP16 embedding; head(hidden, shard) shares the
    target full-vocabulary head and returns this rank's contiguous vocab shard.
    sampler(logits, step=..., position=..., shard=...) returns DraftSample with
    global device int64[1] IDs and actual local FP32 q[1,shard.width]. Each rank
    invokes callbacks/collectives in identical order. Eager status consensus
    rejects invalid IDs before any Markov index_select; graph capture is refused.
    """
    capturable = False

    def __init__(self, weights, *, target, embed, head, target_identity, comm=None, prefill_rows=256):
        c = weights.config
        c.check_target(target)
        if not callable(embed) or not callable(head) or not isinstance(target_identity, str) or not target_identity:
            raise ValueError("DSpark requires explicit shared target callbacks/identity")
        self.prefill_rows = _int(prefill_rows, "prefill rows", 1, MAX_ROWS)
        self.weights, self.config, self.device = weights, c, weights.device
        self.rank, self.world = weights.rank, weights.world
        self.local = c.partition(self.rank, self.world)
        self.shard = VocabShard.make(c, self.rank, self.world)
        if ((self.world > 1 and comm is None) or (comm is not None and
                (comm.rank != self.rank or comm.world != self.world))):
            raise ValueError("DSpark communicator differs from its weight partition")
        if self.device.type == "cuda" and torch.cuda.get_device_capability(self.device) != (7, 0):
            raise ValueError("DSpark FP16 adapter is currently scoped to SM70")
        self.comm, self.embed, self.head = comm, embed, head
        self.identity = hashlib.sha256((weights.identity + "|" + target_identity).encode()).hexdigest()
        self.distributed = self.world > 1
        self.linears = {name: Dense(t) for name, t in weights.tensors.items()
                        if t.ndim == 2 and not name.startswith("confidence_head.")}

    def _eager(self):
        if self.device.type == "cuda":
            with torch.cuda.device(self.device):
                if torch.cuda.is_current_stream_capturing():
                    raise RuntimeError("DSpark CUDA adapter is eager-only; capture is unqualified")

    def _gather(self, value):
        value = value.contiguous()
        if self.world == 1:
            return value.unsqueeze(0)
        out = torch.empty((self.world, *value.shape), dtype=value.dtype, device=self.device)
        self.comm.all_gather(value, out)
        return out

    def _finite(self, *values):
        # All ranks complete this status collective before any rank raises.
        flags = torch.stack([torch.isfinite(v).all() for v in values]).all().to(torch.int32).reshape(1)
        if not bool(self._gather(flags).all().cpu()):
            raise ValueError("DSpark nonfinite device computation")

    def _linear(self, x, name, *, reduce=False):
        result = self.linears[name].matmul(x, f32=True)
        if reduce and self.world > 1:
            parts = self._gather(result)
            result = parts[0].clone()
            for rank in range(1, self.world):
                result.add_(parts[rank])
        return result.half()

    def _check(self, state, *, aligned=True, stamp=True):
        self._eager()
        c = self.config
        if not isinstance(state, DeviceState) or state.closed or state.identity != self.identity:
            raise ValueError("DSpark closed/foreign state identity")
        shape = (c.layers, state.capacity, self.local["kv_heads"], c.head_dim)
        if (state.keys.dtype != torch.float16 or state.values.dtype != torch.float16 or
                state.keys.device != self.device or state.values.device != self.device or
                tuple(state.keys.shape) != shape or state.keys.shape != state.values.shape or
                not state.keys.is_contiguous() or not state.values.is_contiguous() or
                torch._C._overlaps(state.keys, state.values) or
                type(state.length) is not int or not 0 <= state.length <= state.capacity <= c.context):
            raise ValueError("DSpark state storage differs from its admitted geometry")
        if type(state.valid_length) is not int or not state.length <= state.valid_length <= state.capacity:
            raise ValueError("DSpark state lost its retained physical prefix")
        if stamp:
            state.stamp.check(state.target)
        elif stamp is not None:
            _Stamp.read(state.target)
        if aligned and getattr(state.target, "pos", None) != state.length:
            raise ValueError("DSpark target/context cursor differs")
        if aligned and getattr(state.target, "_kv_pending", False):
            raise ValueError("DSpark requires a validated committed target prefix")
        _wait(state.ready, self.device)
        _borrow((state.keys, state.values))

    def _new(self, target, capacity, length=0):
        _int(capacity, "capacity", max(1, length), self.config.context)
        if type(getattr(target, "capacity", None)) is not int or target.capacity < capacity:
            raise ValueError("DSpark target capacity is smaller than its draft admission")
        shape = (self.config.layers, capacity, self.local["kv_heads"], self.config.head_dim)
        return DeviceState(target, self.identity, torch.empty(shape, dtype=torch.float16, device=self.device),
                           torch.empty(shape, dtype=torch.float16, device=self.device), _Stamp.read(target),
                           length=length, ready=_event(self.device))

    def make_state(self, target_state, *, capacity):
        self._eager()
        if getattr(target_state, "pos", None) != 0:
            raise ValueError("DSpark state starts beside an empty target State")
        result = self._new(target_state, capacity)
        _record(result.ready, self.device)
        return result

    def _context(self, state, features):
        self._check(state, aligned=False)
        c, t = self.config, self.weights.tensors
        if (state.pending is not None or features.device != self.device or features.dtype != torch.float16 or
                features.ndim != 2 or features.shape[1] != len(c.taps) * c.hidden or
                not 0 < features.shape[0] <= self.prefill_rows or state.length + features.shape[0] > state.capacity):
            raise ValueError("DSpark pending context/feature geometry exceeds admission")
        self._finite(features)
        projected = context_norm(self._linear(features, "fc.weight"), t["hidden_norm.weight"], c.eps)
        positions = torch.arange(state.length, state.length + features.shape[0], device=self.device)
        keys, values = [], []
        for layer in range(c.layers):
            base = f"layers.{layer}.self_attn."
            k = self._linear(projected, base + "k_proj.weight").view(-1, self.local["kv_heads"], c.head_dim)
            keys.append(ref.rope(context_norm(k, t[base + "k_norm.weight"], c.eps), positions, c.theta))
            values.append(self._linear(projected, base + "v_proj.weight").view(-1, self.local["kv_heads"], c.head_dim))
        keys, values = torch.stack(keys), torch.stack(values)
        self._finite(keys, values)
        result = DeviceContextUpdate(state.length, keys, values, state.version, state, features.shape[0], _event(self.device))
        _record(result.ready, self.device)
        return result

    @torch.no_grad()
    def stage_context(self, state, features):
        update = self._context(state, features)
        state.pending = update
        return update

    @torch.no_grad()
    def stage_taps(self, states, batch, *, accepted=None):
        """Consume/release one typed TapBatch after independent KV reads enqueue.

        accepted=None stages the uncommitted target frame. Otherwise each count
        names an already committed prefix, using TapBatch's guarded accessor.
        Returns updates in input state order, with None for an empty prefix.
        No partial update is published if another lane fails validation/math.
        """
        from .dspark_taps import TapBatch

        if not isinstance(batch, TapBatch):
            raise ValueError("DSpark requires the typed owned target tap lease")
        states = tuple(states)
        try:
            if not states or len({id(s) for s in states}) != len(states):
                raise ValueError("DSpark tap lanes must be nonempty and unique")
            counts = (None,) * len(states) if accepted is None else tuple(accepted)
            if len(counts) != len(states):
                raise ValueError("DSpark accepted counts must cover all tap lanes")
            pieces = []
            for state, count in zip(states, counts, strict=True):
                self._check(state, aligned=False)
                binding = next((p for p in batch.segments if p.state is state.target), None)
                if binding is None or binding.start != state.length or state.pending is not None:
                    raise ValueError("DSpark taps belong to a different draft prefix")
                pieces.append(batch.segment(state.target, accepted=count))
            updates = [self._context(state, x) if x.shape[0] else None
                       for state, x in zip(states, pieces, strict=True)]
            for state, update in zip(states, updates, strict=True):
                state.pending = update
            return tuple(updates)
        finally:
            if not batch.released:
                batch.release()

    @torch.no_grad()
    def commit(self, state, update, accepted):
        self._check(state, aligned=False)
        if (not isinstance(update, DeviceContextUpdate) or update is not state.pending or update.owner is not state
                or update.version != state.version or update.start != state.length or
                type(accepted) is not int or not 0 <= accepted <= update.rows):
            raise ValueError("DSpark stale/foreign context update or invalid accepted prefix")
        shape = (self.config.layers, update.rows, self.local["kv_heads"], self.config.head_dim)
        if (any(not isinstance(t, torch.Tensor) or t.device != self.device or t.dtype != torch.float16 or
                tuple(t.shape) != shape or not t.is_contiguous() for t in (update.keys, update.values)) or
                torch._C._overlaps(update.keys, update.values) or
                any(torch._C._overlaps(a, b) for a in (update.keys, update.values)
                    for b in (state.keys, state.values))):
            raise ValueError("DSpark context update lost its independent admitted storage")
        end = state.length + accepted
        if getattr(state.target, "pos", None) != end or getattr(state.target, "_kv_pending", False):
            raise ValueError("DSpark commit requires the matching verified target prefix")
        _wait(update.ready, self.device)
        _borrow((update.keys, update.values))
        state.keys[:, state.length:end].copy_(update.keys[:, :accepted])
        state.values[:, state.length:end].copy_(update.values[:, :accepted])
        state.length, state.version, state.pending = end, state.version + 1, None
        state.valid_length = max(state.valid_length, end)
        _record(state.ready, self.device)
        update.retire()

    def discard(self, state):
        self._check(state, aligned=False, stamp=None)
        if state.pending is not None:
            _wait(state.pending.ready, self.device)
            _borrow((state.pending.keys, state.pending.values))
            state.pending.retire()
        state.pending, state.version = None, state.version + 1
        _record(state.ready, self.device)

    def rewind(self, state, position):
        self._check(state, aligned=False, stamp=False)
        _int(position, "retained prefix", 0, state.length)
        if getattr(state.target, "pos", None) != position or state.stamp.codec != _Stamp.read(state.target).codec:
            raise ValueError("DSpark rewind requires the same retained target prefix/codec")
        if state.pending is not None:
            _wait(state.pending.ready, self.device)
            _borrow((state.pending.keys, state.pending.values))
            state.pending.retire()
        state.length, state.pending, state.version = position, None, state.version + 1
        state.stamp = _Stamp.read(state.target)
        _record(state.ready, self.device)

    def reset(self, state):
        self.rewind(state, 0)

    @torch.no_grad()
    def copy_prefix(self, source, target_state, *, capacity):
        self._check(source)
        if (source.pending is not None or getattr(target_state, "pos", None) != source.length or
                _Stamp.read(target_state).codec != source.stamp.codec):
            raise ValueError("DSpark prefix copy requires matching committed target cursors")
        result = self._new(target_state, capacity, source.length)
        try:
            result.keys[:, :source.length].copy_(source.keys[:, :source.length])
            result.values[:, :source.length].copy_(source.values[:, :source.length])
        finally:
            _record(source.ready, self.device)
        _record(result.ready, self.device)
        result.valid_length = source.length
        return result

    @torch.no_grad()
    def resize(self, state, capacity):
        self._check(state, stamp=False)
        current = _Stamp.read(state.target)
        if (state.pending is not None or state.stamp.codec != current.codec
                or state.stamp.history is not current.history):
            raise ValueError("DSpark resize requires committed target storage with the same codec")
        _int(capacity, "capacity", max(1, state.length), self.config.context)
        replacement = self._new(state.target, capacity, state.length)
        try:
            replacement.keys[:, :state.length].copy_(state.keys[:, :state.length])
            replacement.values[:, :state.length].copy_(state.values[:, :state.length])
        finally:
            _record(state.ready, self.device)
        state.keys, state.values, state.stamp = replacement.keys, replacement.values, replacement.stamp
        state.valid_length = state.length
        state.version += 1
        _record(state.ready, self.device)

    @torch.no_grad()
    def snapshot(self, state):
        self._check(state)
        if state.pending is not None:
            raise ValueError("DSpark snapshot requires a committed target prefix")
        try:
            result = DeviceSnapshot(self.identity, state.keys[:, :state.length].clone(),
                                    state.values[:, :state.length].clone(), state.length, _event(self.device))
        finally:
            _record(state.ready, self.device)
        _record(result.ready, self.device)
        return result

    @torch.no_grad()
    def restore(self, state, snapshot):
        self._check(state, aligned=False, stamp=False)
        if not isinstance(snapshot, DeviceSnapshot):
            raise ValueError("DSpark snapshot differs from its target prefix/storage")
        _int(snapshot.length, "snapshot prefix", 0, state.capacity)
        shape = (self.config.layers, snapshot.length, self.local["kv_heads"], self.config.head_dim)
        if (snapshot.identity != self.identity or
                snapshot.length > state.capacity or getattr(state.target, "pos", None) != snapshot.length or
                getattr(state.target, "_kv_pending", False) or
                snapshot.keys.shape != shape or snapshot.values.shape != shape or
                any(t.device != self.device or t.dtype != torch.float16 for t in (snapshot.keys, snapshot.values)) or
                state.stamp.codec != _Stamp.read(state.target).codec):
            raise ValueError("DSpark snapshot differs from its target prefix/storage")
        self._retire_pending(state)
        _wait(snapshot.ready, self.device)
        _borrow((snapshot.keys, snapshot.values))
        state.keys[:, :snapshot.length].copy_(snapshot.keys)
        state.values[:, :snapshot.length].copy_(snapshot.values)
        state.length, state.version, state.stamp = snapshot.length, state.version + 1, _Stamp.read(state.target)
        state.valid_length = max(state.valid_length, snapshot.length)
        _record(state.ready, self.device)

    def _retire_pending(self, state):
        if state.pending is not None:
            _wait(state.pending.ready, self.device)
            _borrow((state.pending.keys, state.pending.values))
            state.pending.retire()
            state.pending = None

    def close_state(self, state):
        self._check(state, aligned=False, stamp=None)
        self._retire_pending(state)
        _record(state.ready, self.device)
        if state.ready is not None:
            state.ready.synchronize()
        state.closed = True
        state.keys = state.values = state.target = state.ready = None

    def _hidden(self, state, embeds):
        c, t = self.config, self.weights.tensors
        positions = torch.arange(state.length, state.length + embeds.shape[0], device=self.device)
        hidden, residual = embeds, None
        for layer in range(c.layers):
            base = f"layers.{layer}."
            x, residual = ref.add_norm(hidden, residual, t[base + "input_layernorm.weight"], c.eps)
            ab = base + "self_attn."
            q = self._linear(x, ab + "q_proj.weight").view(-1, self.local["heads"], c.head_dim)
            k = self._linear(x, ab + "k_proj.weight").view(-1, self.local["kv_heads"], c.head_dim)
            v = self._linear(x, ab + "v_proj.weight").view(-1, self.local["kv_heads"], c.head_dim)
            q = ref.rope(ref.rms(q, t[ab + "q_norm.weight"], c.eps), positions, c.theta)
            k = ref.rope(ref.rms(k, t[ab + "k_norm.weight"], c.eps), positions, c.theta)
            all_k = torch.cat((state.keys[layer, :state.length], k))
            all_v = torch.cat((state.values[layer, :state.length], v))
            hidden = self._linear(gqa_attention(q, all_k, all_v), ab + "o_proj.weight", reduce=True)
            x, residual = ref.add_norm(hidden, residual, t[base + "post_attention_layernorm.weight"], c.eps)
            gate = self._linear(x, base + "mlp.gate_proj.weight")
            up = self._linear(x, base + "mlp.up_proj.weight")
            hidden = self._linear((F.silu(gate.float()) * up.float()).half(), base + "mlp.down_proj.weight", reduce=True)
        return ref.add_norm(hidden, residual, t["norm.weight"], c.eps)[0]

    def _greedy(self, logits, **_):
        score, index = logits.float().max(-1)
        ids = index.to(torch.int64) + self.shard.start
        scores, candidates = self._gather(score), self._gather(ids)
        best = scores.max(0).values
        valid = torch.where(scores == best, candidates, self.config.vocab)
        token = valid.min(0).values
        q = torch.zeros_like(logits, dtype=torch.float32)
        local = (token - self.shard.start).clamp(0, self.shard.width - 1)
        q.scatter_(1, local[:, None], ((token >= self.shard.start) & (token < self.shard.stop)).float()[:, None])
        return DraftSample(token, q)

    def _sample(self, logits, sampler, step, position):
        error = None
        try:
            sample = sampler(logits, step=step, position=position, shard=self.shard)
            coupled = isinstance(sample, PositionKeyedChoice)
            if (not isinstance(sample, (DraftSample, PositionKeyedChoice)) or not isinstance(sample.ids, torch.Tensor) or
                    sample.ids.device != self.device or
                    sample.ids.dtype != torch.int64 or sample.ids.shape != (1,) or
                    (not coupled and (not isinstance(sample.probabilities, torch.Tensor) or
                     sample.probabilities.device != self.device or sample.probabilities.dtype != torch.float32 or
                     sample.probabilities.shape != (1, self.shard.width)))):
                raise ValueError("DSpark sampler must return global device IDs and its actual local FP32 distribution")
        except Exception as fault:
            # Malformed/failed callbacks on one rank must not leave peers at
            # the subsequent sampling collective. Publish a safe invalid row.
            error = fault
            sample = DraftSample(torch.zeros(1, dtype=torch.int64, device=self.device),
                                 torch.zeros_like(logits, dtype=torch.float32))
            coupled = False
        ids = self._gather(sample.ids)
        token = ids[0]
        valid = (ids == token).all() & (token >= 0).all() & (token < self.config.vocab).all() & (error is None)
        q = None if coupled else sample.probabilities
        mass = torch.zeros((), dtype=torch.float32, device=self.device)
        selected = mass.clone()
        if q is not None:
            safe_local = (token - self.shard.start).clamp(0, self.shard.width - 1)
            selected = q.gather(1, safe_local[:, None]).flatten()
            selected *= ((token >= self.shard.start) & (token < self.shard.stop))
            valid &= torch.isfinite(q).all() & (q >= 0).all()
            mass = q.sum()
        status = torch.stack((valid.float(), torch.full_like(mass, float(coupled)), mass, selected.sum()))
        gathered = self._gather(status).cpu()
        if (not bool(gathered[:, 0].all()) or not bool((gathered[:, 1] == gathered[0, 1]).all()) or
                (not coupled and (abs(float(gathered[:, 2].sum()) - 1) > 2e-5
                                  or not float(gathered[:, 3].sum()) > 0))):
            raise ValueError("DSpark sampler returned invalid/disagreeing global IDs or proposal probabilities") from error
        # Sampling callbacks may reuse their IDs/q scratch on the next step.
        return token.clone(), None if q is None else q.clone()

    def check_proposal(self, state, proposal):
        self._check(state)
        if (not isinstance(proposal, DeviceProposal) or proposal.owner is not state or proposal.identity != self.identity
                or proposal.start != state.length or proposal.state_version != state.version
                or state.pending is not None or proposal.shard != self.shard):
            raise ValueError("DSpark proposal belongs to a stale/foreign target prefix")

    @torch.no_grad()
    def propose(self, state, bonus, *, block=None, sampler=None):
        self._check(state)
        try:
            return self._propose(state, bonus, block=block, sampler=sampler)
        finally:
            # Even a failed callback/finite gate may have enqueued cache reads.
            # A different stream's later writer must wait those readers too.
            _record(state.ready, self.device)

    def _propose(self, state, bonus, *, block=None, sampler=None):
        c, t = self.config, self.weights.tensors
        block = _int(c.block if block is None else block, "block", 1, min(32, c.block))
        bonus = _int(bonus, "known bonus", 0, c.vocab - 1)
        if state.pending is not None or state.length + block + 1 > state.capacity:
            raise ValueError("DSpark proposal has pending context or exceeds its admitted block/cache")
        inputs = torch.full((block,), c.mask, dtype=torch.int64, device=self.device)
        inputs[0] = bonus
        embeds = self.embed(inputs)
        if embeds.device != self.device or embeds.dtype != torch.float16 or embeds.shape != (block, c.hidden):
            raise ValueError("DSpark target embedding callback must return shared full FP16 rows")
        self._finite(embeds)
        hidden = self._hidden(state, embeds)
        self._finite(hidden)
        base = self.head(hidden, self.shard)
        if base.device != self.device or base.dtype not in (torch.float16, torch.float32) or base.shape != (block, self.shard.width):
            raise ValueError("DSpark target head callback must return its declared local vocabulary shard")
        base = base.half()
        self._finite(base)
        chosen, corrected, qs, prevs = [], [], [], []
        previous = inputs[:1]
        for step in range(block):
            # Previous is the checked common global ID, never a rank-local ID.
            prev = t["markov_head.markov_w1.weight"].index_select(0, previous)
            bias = self._linear(prev, "markov_head.markov_w2.weight")
            logits = (base[step:step + 1].float() + bias.float()).half()
            self._finite(logits)
            # Preserve raw corrected logits before a sampler applies in-place
            # penalties/filters to its borrowed input.
            corrected.append(logits.clone())
            token, q = self._sample(logits, self._greedy if sampler is None else sampler, step, state.length + 1 + step)
            prevs.append(prev)
            chosen.append(token)
            qs.append(q)
            previous = token
        ids = torch.cat(chosen)
        confidence = None
        if c.confidence:
            joined = torch.cat((hidden, torch.cat(prevs)), -1).float()
            confidence = (joined @ t["confidence_head.proj.weight"].T + t["confidence_head.proj.bias"]).flatten().sigmoid()
            self._finite(confidence)
        # Own every exported buffer; target/head callback scratch is not retained.
        if any(q is None for q in qs) and not all(q is None for q in qs):
            raise ValueError("DSpark sampling probability contract changed inside a proposal")
        verify, corrected = torch.cat((inputs[:1], ids)), torch.cat(corrected)
        qs = None if qs[0] is None else torch.cat(qs)
        ready = _event(self.device)
        _record(ready, self.device)
        result = DeviceProposal(ids, verify, corrected, qs, confidence,
                                state.length, state.version, self.identity, self.shard, state, ready)
        return result

"""Unique GPU history storage and bounded owner attention for the SM70 path.

Weights stay TP8. Only packed KV pages and the canonical raw/pooled indexer
are retained; communication scratch is bounded by the current execution rows.
This source implementation requires separate native/full-model qualification.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass
import traceback
from types import SimpleNamespace

import torch

from tensorfold.cuda.sm70 import family_ops as ops

from .kvcache import KVCache, KVPairFormat


@dataclass(frozen=True)
class HistoryPageClass:
    """One allocator class: codec pair, geometry and a full fixed token page."""
    pair: KVPairFormat
    head_dim: int
    tokens: int

    def __post_init__(self):
        if type(self.tokens) is not int or self.tokens <= 0 or self.tokens % 4:
            raise ValueError("typed history pages need complete positive pooling groups")
        self.pair.row_bytes(1, self.head_dim)

    @property
    def nbytes(self):
        return self.tokens * self.pair.row_bytes(1, self.head_dim)


class HistoryArena(KVCache):
    """Preallocated per-rank typed pages, reserved before model execution."""
    __slots__ = ("page_class", "page_capacity", "reserved_pages", "rank")

    def __init__(self, page_class, pages, device, *, rank):
        if type(pages) is not int or pages < 0 or type(rank) is not int or not 0 <= rank < 8:
            raise ValueError("typed history arena page count must be nonnegative")
        self.rank = rank
        self.page_class, self.page_capacity, self.reserved_pages = page_class, pages, 0
        super().__init__(pages * page_class.tokens, 1, page_class.head_dim, device, pair=page_class.pair)

    @property
    def free_pages(self):
        return self.page_capacity - self.reserved_pages

    def reserve(self, page_class, pages):
        if page_class != self.page_class or type(pages) is not int or not 0 <= pages <= self.free_pages:
            raise ValueError("matching typed history pages are unavailable in the admitted rank arena")
        first = self.reserved_pages * self.page_class.tokens
        self.reserved_pages += pages
        return first


class OwnedKVCache(KVCache):
    """Packed local pages; logical positions remain independent of rank offsets."""

    __slots__ = ("placement", "slot", "layer", "rank", "logical_capacity", "offsets", "page_offsets",
                 "arena", "reserved_rows", "owners", "page_class")

    def __init__(self, rows, kv_heads, head_dim, device, *, pair, placement, slot, layer, rank,
                 arena=None, arena_offset=0, owners=None):
        if kv_heads != 1:
            raise ValueError("unique history expects one weight-shard KV head")
        self.placement, self.slot, self.layer, self.rank = placement, slot, layer, rank
        self.page_class = HistoryPageClass(pair, head_dim, placement.page_tokens)
        self.logical_capacity = rows
        offsets, owned = [], 0
        head = rank // placement.group_size
        self.owners = ([placement.kv_owner(slot, layer, head, page)
                        for page in range((rows + placement.page_tokens - 1) // placement.page_tokens)]
                       if owners is None else list(owners))
        if (len(self.owners) != (rows + placement.page_tokens - 1) // placement.page_tokens
                or any(type(owner) is not int or not head * 4 <= owner < (head + 1) * 4 for owner in self.owners)):
            raise ValueError("owned page table must describe the complete local KV-head group")
        for page in range((rows + placement.page_tokens - 1) // placement.page_tokens):
            if self.owners[page] == rank:
                offsets.append(arena_offset + owned)
                owned += placement.physical_page_rows(rows, page)
            else:
                offsets.append(-1)
        self.offsets = tuple(offsets)
        self.page_offsets = torch.tensor(offsets, dtype=torch.int64, device=device)
        self.arena, self.reserved_rows = arena, owned
        if arena is None:
            super().__init__(owned, kv_heads, head_dim, device, pair=pair)
        else:
            if not pair.key.quantized or not pair.value.quantized or arena.pair != pair:
                raise ValueError("shared history arenas require matching quantized K/V sides")
            if isinstance(arena, HistoryArena) and (arena.rank != rank or arena.page_class != self.page_class):
                raise ValueError("shared typed history arena differs from its admitted rank/page class")
            if arena_offset < 0 or arena_offset + owned > arena.capacity:
                raise ValueError("history share exceeds the admitted rank arena")
            self._pair, self._capacity, self._kv_heads, self._head_dim = pair, arena.capacity, 1, head_dim
            self.k, self.v, self.ks, self.vs = arena.k, arena.v, arena.ks, arena.vs

    @property
    def nbytes(self):
        return self.reserved_rows * self.pair.row_bytes(1, self.head_dim) + self.pair.dummy_bytes + self.page_offsets.nbytes

    def _spans(self, start, count):
        if type(start) is not int or type(count) is not int or not 0 <= start <= self.logical_capacity - count:
            raise ValueError("append range must fit the promised logical history")
        width = self.placement.page_tokens
        consumed = 0
        while consumed < count:
            position = start + consumed
            page, within = divmod(position, width)
            take = min(count - consumed, width - within)
            if self.offsets[page] >= 0:
                yield consumed, self.offsets[page] + within, take
            consumed += take

    def append(self, start, key, value, key_scales, value_scales):
        """Current-stream copies from a bounded packed frame, never a history mirror."""
        for source, target, count in self._spans(start, key.shape[0]):
            self.k[target:target + count].copy_(key[source:source + count])
            self.v[target:target + count].copy_(value[source:source + count])
            if self.key_format.quantized:
                self.ks[target:target + count].copy_(key_scales[source:source + count])
            if self.value_format.quantized:
                self.vs[target:target + count].copy_(value_scales[source:source + count])

    def resized(self, capacity, keep):
        if type(keep) is not int or keep < 0:
            raise ValueError("retained history length must be a nonnegative integer")
        pages = (capacity + self.placement.page_tokens - 1) // self.placement.page_tokens
        owners = self.owners[:pages] + [self.placement.kv_owner(self.slot, self.layer, self.rank // 4, page)
                                      for page in range(len(self.owners), pages)]
        other = OwnedKVCache(capacity, 1, self.head_dim, self.k.device, pair=self.pair,
                             placement=self.placement, slot=self.slot, layer=self.layer, rank=self.rank, owners=owners)
        kept = min(keep, capacity, self.logical_capacity)
        for page in range((kept + self.placement.page_tokens - 1) // self.placement.page_tokens):
            source, target = self.offsets[page], other.offsets[page]
            if source < 0:
                if target >= 0:
                    raise ValueError("resize cannot reset a partially migrated owner table")
                continue
            if target < 0:
                raise ValueError("resize cannot reset a partially migrated owner table")
            count = min(self.placement.page_tokens, kept - page * self.placement.page_tokens)
            other.k[target:target + count].copy_(self.k[source:source + count])
            other.v[target:target + count].copy_(self.v[source:source + count])
            if self.key_format.quantized:
                other.ks[target:target + count].copy_(self.ks[source:source + count])
            if self.value_format.quantized:
                other.vs[target:target + count].copy_(self.vs[source:source + count])
        if not self.key_format.quantized:
            other.ks.copy_(self.ks)
        if not self.value_format.quantized:
            other.vs.copy_(self.vs)
        return other

    def clone(self):
        other = OwnedKVCache(self.logical_capacity, 1, self.head_dim, self.k.device, pair=self.pair,
                             placement=self.placement, slot=self.slot, layer=self.layer,
                             rank=self.rank, owners=self.owners)
        for page, source in enumerate(self.offsets):
            if source < 0:
                continue
            target, count = other.offsets[page], self.placement.page_tokens
            for original, copied in ((self.k, other.k), (self.v, other.v)):
                copied[target:target + count].copy_(original[source:source + count])
            for original, copied, format in ((self.ks, other.ks, self.key_format), (self.vs, other.vs, self.value_format)):
                if format.quantized:
                    copied[target:target + count].copy_(original[source:source + count])
                else:
                    copied.copy_(original)
        return other

    def local_indices(self, logical):
        safe = logical.clamp(0, max(0, self.logical_capacity - 1))
        page = torch.div(safe, self.placement.page_tokens, rounding_mode="floor")
        offsets = self.page_offsets[page]
        valid = (logical >= 0) & (logical < self.logical_capacity) & (offsets >= 0)
        return (offsets + safe % self.placement.page_tokens).clamp_min(0), valid


def allocate_target_caches(rows, head_dim, device, *, pair, placement, slot, rank):
    """One bounded rank arena permits balanced page swaps without full copies."""
    if not pair.key.quantized or not pair.value.quantized:
        return [OwnedKVCache(rows, 1, head_dim, device, pair=pair, placement=placement,
                             slot=slot, layer=layer, rank=rank) for layer in range(placement.target_layers)]
    counts = [placement.local_rows(slot, layer, rows, rank) for layer in range(placement.target_layers)]
    page_class = HistoryPageClass(pair, head_dim, placement.page_tokens)
    arena = HistoryArena(page_class, sum(counts) // placement.page_tokens, device, rank=rank)
    result = []
    for layer, count in enumerate(counts):
        offset = arena.reserve(page_class, count // placement.page_tokens)
        result.append(OwnedKVCache(rows, 1, head_dim, device, pair=pair, placement=placement,
                                   slot=slot, layer=layer, rank=rank, arena=arena, arena_offset=offset))
    return result


def _encode(value, format, status, layer, *, error):
    if not format.quantized:
        return value.half(), None
    if format.codec:
        rows, heads, dims = value.shape
        codes = torch.empty((rows, heads, dims * format.bits // 8), dtype=torch.uint8, device=value.device)
        scales = torch.empty((rows, heads, dims // format.group),
                             dtype=getattr(torch, format.scale_dtype), device=value.device)
        position = torch.zeros((1,), dtype=torch.int32, device=value.device)
        ops._rotor_encode(value, codes, scales, position, codec=format.codec, status=status, layer=layer, error=error)
        return codes, scales
    if format.bits != 8:
        raise ValueError("unique SM70 history requires ordinary Q8 or an admitted RotorQuant Q4 codec")
    return ops.quantize_int8(value, status=status, layer=layer)




def _native_prelude_position(position):
    """An already bound native role is allowed; extended/foreign roles are not."""
    from ..position_profile import NATIVE_ALGORITHM, PositionOperands, PositionRole

    if not isinstance(position, PositionOperands) or not isinstance(position.domain, PositionRole):
        return False
    role = position.domain
    return (role.algorithm == NATIVE_ALGORITHM and role.rope.rope_type == "default"
            and role.max_rows <= role.rope.native_context
            and role.rope.context_limit <= role.rope.native_context
            and position.amplitude == role.rope.attention_factor
            and position.sections == role.rope.mrope_section)


def _native_prelude_binding(binding):
    """Startup capability check over the actual resolver/binder's immutable roles."""
    from ..position_profile import BoundPositionProfile

    return (binding is None or isinstance(binding, BoundPositionProfile)
            and _native_prelude_position(binding.target)
            and (binding.mtp is None or _native_prelude_position(binding.mtp)))


def _native_frame_coordinates(state, buffers, start, stop, *, mtp):
    """Borrow issued axes without rewriting the frame's mutation-tracked storage."""
    explicit = buffers.mtp_indexer_coordinates if mtp else buffers.target_indexer_coordinates
    if explicit is not None:
        return explicit[start:stop]
    from .compact_indexer import coordinate_rows

    position = state.mtp_len if mtp else state.pos
    return coordinate_rows(state, torch.arange(stop - start, device=state.pos_dev.device) + position)


def _record_native_prelude(weights, *, mtp):
    """Serialized submitted-frame diagnostics; an exhausted MTP count is unknown."""
    weights.meta["sm70_owner_prelude_submitted_frames"] += 1
    if mtp:
        value = weights.meta["sm70_owner_prelude_submitted_mtp_frames"]
        if value < (1 << 63) - 1:
            weights.meta["sm70_owner_prelude_submitted_mtp_frames"] = value + 1
        else:
            weights.meta["sm70_owner_prelude_mtp_counter_overflowed"] = True

def _publish_native_frame(state, layer, weights, buffers, start, stop, key, value, ks, vs, ik,
                          frequency, sections, amplitude, *, mtp, frame_first=0):
    c = weights.cfg
    a = layer.attn
    ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
    cache = state.mtp_kc if mtp else state.kc[ai]
    rows = stop - start
    last = frame_first + rows
    pos = state.mtp_len if mtp else state.pos
    cache.append(pos, key[frame_first:last], value[frame_first:last], ks[frame_first:last], vs[frame_first:last])
    if state.history_rank == state.history_placement.indexer_owner(state.history_slot, ai,
                                                                  include_mtp=hasattr(state, "mtp_kc")):
        coordinates = _native_frame_coordinates(state, buffers, start, stop, mtp=mtp)
        entry = state.compact_indexer.append(ai, pos, ik[frame_first:last], coordinates,
            buffers.compact_indexer_stage, start, stop, mtp=mtp,
            expected=(state.slot_generation, state.history_generation))
        entry.pool(state.mtp_pooled if mtp else state.pooled[ai], a.ik_scale, frequency,
                   c.eps, sections, amplitude)
        if mtp:
            expected = (state.slot_generation, state.history_generation)
            entry.publish_commit(entry.prepare_commit(pos + rows, expected), pos + rows, expected)
    if not mtp:
        state.target_tentative_length = pos + rows
    _record_native_prelude(weights, mtp=mtp)

def _native_frame_prepare(projection, state, layer, weights, buffers, start, stop, *, mtp):
    """Reuse the qualified scalar-position prep into a bounded frame, not a history.

    The model owns a separate initialized immutable zero. Position zero addresses
    frame outputs; the original State position is the existing RoPE delta input.
    Only plain native-context Q8 text uses this specialization. Existing cache
    and CompactIndexer transactions copy the frame before the actor reuses gated.
    """
    if (state.image_positions is not None or state.rope_delta != 0
            or getattr(state, "media_profile_identity", None) is not None
            or getattr(state, "media_input_span", None) is not None):
        return False  # No native writes; preserve the ordinary coordinate/codec route.
    c, a = weights.cfg, layer.attn
    ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
    cache = state.mtp_kc if mtp else state.kc[ai]
    if (cache.key_format.bits != 8 or cache.value_format.bits != 8
            or cache.key_format.codec or cache.value_format.codec):
        return False
    if (not projection.is_cuda or projection.dtype != torch.float16 or not projection.is_contiguous()
            or not buffers.gated.is_contiguous() or buffers.gated.dtype != torch.float16):
        return False
    if not 0 <= start < stop <= buffers.rows or stop > projection.shape[0]:
        raise ValueError("native owner frame must fit its prepared input/output rows")
    position = buffers.mtp_indexer_position if mtp else buffers.target_indexer_position
    binding = getattr(weights, "positions", None)
    if binding is not None:
        expected = binding.mtp if mtp else binding.target
        from .compact_indexer import _scalar_text_frame

        frame = buffers.mtp_indexer_frame if mtp else buffers.target_indexer_frame
        if (position is not expected or not _native_prelude_position(position) or not _scalar_text_frame(frame)):
            return False  # Arbitrary issued THW coordinates retain their ordinary path.
        frequency, sections, amplitude = position.inv_freq, position.sections, position.amplitude
    else:
        if position is not None:
            return False
        frequency, sections, amplitude = weights.inv_freq, c.mrope_section, c.rope_attention_factor
    capacity, dim = buffers.rows, c.head_dim
    # The original pre-gate scratch is idle until attention returns. Contiguous
    # frame arrays occupy 800 bytes/row in the admitted 1536-byte/row slab.
    storage = buffers.gated.view(torch.uint8).view(-1)
    payload, scale, raw = capacity * dim, capacity * (dim // 32) * 2, capacity * c.index_dim * 2
    if 2 * payload + 2 * scale + raw > storage.numel():
        raise ValueError("native owner frame exceeds its existing gated scratch")
    key = storage[:payload].view(torch.int8).view(capacity, 1, dim)
    value = storage[payload:2 * payload].view(torch.int8).view(capacity, 1, dim)
    ks = storage[2 * payload:2 * payload + scale].view(torch.float16).view(capacity, 1, dim // 32)
    vs = storage[2 * payload + scale:2 * payload + 2 * scale].view(torch.float16).view(capacity, 1, dim // 32)
    ik = storage[2 * payload + 2 * scale:2 * payload + 2 * scale + raw].view(torch.float16).view(capacity, c.index_dim)
    from tensorfold.cuda.sm70.family_native import extension

    zero = weights.meta["sm70_owner_prelude_zero"]
    absolute = state.mtp_pos if mtp else state.pos_dev
    extension().prep(projection[start:stop], zero, a.q_scale, a.k_scale, a.iq_scale, frequency,
        buffers.q[start:stop], key, value, buffers.iq[start:stop], ik, ks, vs, state.kv_status,
        absolute, None, c.heads, 1, dim, c.index_heads, c.index_dim, 8, 8, layer.index,
        c.eps, amplitude, not buffers.prefill)
    _publish_native_frame(state, layer, weights, buffers, start, stop, key, value, ks, vs, ik,
                          frequency, sections, amplitude, mtp=mtp)
    return True

def prepare(projection, state, layer, weights, buffers, start, stop, *, mtp):
    """Prepare one bounded frame; only owners publish persistent rows."""
    from .prefix_policy import OWNER_TARGET_PRELUDE_PROFILE, OWNER_TARGET_DENSE_PROFILE, OWNER_BATCH_PRELUDE_PROFILE, ROUTER_BLAS_PROFILE, SHARED_FUSION_PROFILE, HC_SMALLROW_VENDOR_PROFILE

    if (weights.meta.get("sm70_profile") in (OWNER_TARGET_PRELUDE_PROFILE, OWNER_TARGET_DENSE_PROFILE, OWNER_BATCH_PRELUDE_PROFILE, ROUTER_BLAS_PROFILE, SHARED_FUSION_PROFILE, HC_SMALLROW_VENDOR_PROFILE)
            and _native_frame_prepare(projection, state, layer, weights, buffers, start, stop, mtp=mtp)):
        return
    c, a = weights.cfg, layer.attn
    ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
    cache = state.mtp_kc if mtp else state.kc[ai]
    pos = state.mtp_len if mtp else state.pos
    rows = stop - start
    p = projection[start:stop]
    # These actor-owned arrays have no live consumers during input preparation.
    # q/iq are overwritten again after normalization/rotation. gated's key,
    # value and index-key slices stay disjoint until append/pool submissions
    # retire in current-stream order; the attention gate writes gated only
    # after this entire preparation/attention call returns. Raw projection
    # gate columns remain untouched. No new device allocation or history exists.
    guarded_out = weights.meta.get("sm70_native_guard_trial") is True
    query_out = buffers.q[start:stop] if guarded_out else None
    iq_out = buffers.iq[start:stop] if guarded_out else None
    if guarded_out:
        if c.heads * c.head_dim < 2 * c.head_dim + c.index_dim:
            raise ValueError("native guard requires the admitted disjoint gated scratch geometry")
        scratch = buffers.gated[start:stop]
        key_out = scratch[:, :c.head_dim].reshape(rows, 1, c.head_dim)
        value_out = scratch[:, c.head_dim:2 * c.head_dim].reshape(rows, 1, c.head_dim)
        ik_out = scratch[:, 2 * c.head_dim:2 * c.head_dim + c.index_dim]
    else:
        key_out = value_out = ik_out = None
    positions = torch.arange(rows, device=p.device) + pos
    at = c.heads * 2 * c.head_dim
    query = ops._guard(p[:, :at].reshape(rows, c.heads, 2, c.head_dim)[:, :, 0], state.kv_status, 1, layer.index, out=query_out)
    key = ops._guard(p[:, at:at + c.head_dim].reshape(rows, 1, c.head_dim), state.kv_status, 2, layer.index, out=key_out)
    at += c.head_dim
    value = ops._guard(p[:, at:at + c.head_dim].reshape(rows, 1, c.head_dim), state.kv_status, 4, layer.index, out=value_out)
    at += c.head_dim
    iq = ops._guard(p[:, at:at + c.index_heads * c.index_dim].reshape(rows, c.index_heads, c.index_dim),
                    state.kv_status, 16, layer.index, out=iq_out)
    at += c.index_heads * c.index_dim
    ik = ops._guard(p[:, at:at + c.index_dim], state.kv_status, 16, layer.index, out=ik_out)

    def normalize(value, weight):
        if buffers.prefill:
            return ops.rms(value, weight, c.eps)
        return torch.cat([ops.rms(value[row:row + 1], weight, c.eps) for row in range(rows)], 0)

    from .compact_indexer import coordinate_rows

    if state.compact_indexer is None:
        raise RuntimeError("unique history requires its compact raw/pool transaction owner")
    explicit = buffers.mtp_indexer_coordinates if mtp else buffers.target_indexer_coordinates
    coordinates = explicit[start:stop] if explicit is not None else coordinate_rows(state, positions)
    position = buffers.mtp_indexer_position if mtp else buffers.target_indexer_position
    frequencies = weights.inv_freq if position is None else position.inv_freq
    sections = c.mrope_section if position is None else position.sections
    amplitude = c.rope_attention_factor if position is None else position.amplitude
    options = dict(rope=coordinates, length=rows, sections=sections, scale=amplitude)
    rotary_positions = torch.arange(rows, device=p.device, dtype=torch.int64)
    query = ops.rotary(normalize(query, a.q_scale), frequencies, rotary_positions, **options)
    key = ops.rotary(normalize(key, a.k_scale), frequencies, rotary_positions, **options)
    query = ops._guard(query, state.kv_status, 1, layer.index, out=query_out)
    key = ops._guard(key, state.kv_status, 2, layer.index, out=key_out)
    if cache.key_format.codec:
        ops._rotor_rotate(query, buffers.q[start:stop], codec=cache.key_format.codec,
                          status=state.kv_status, layer=layer.index)
    else:
        buffers.q[start:stop].copy_(ops.h32(query).half() if cache.key_format.quantized else query)
    prepared_iq = ops._guard(
        ops.rotary(normalize(iq, a.iq_scale), frequencies, rotary_positions, **options),
        state.kv_status, 16, layer.index, out=iq_out)
    if not guarded_out or prepared_iq is not iq_out:
        buffers.iq[start:stop].copy_(prepared_iq)  # status=None retains the original passthrough contract.
    key_codes, key_scales = _encode(key, cache.key_format, state.kv_status, layer.index, error=2)
    value_codes, value_scales = _encode(value, cache.value_format, state.kv_status, layer.index, error=4)
    cache.append(pos, key_codes, value_codes, key_scales, value_scales)
    if state.history_rank == state.history_placement.indexer_owner(state.history_slot, ai,
                                                                  include_mtp=hasattr(state, "mtp_kc")):
        entry = state.compact_indexer.append(ai, pos, ik.half(), coordinates,
            buffers.compact_indexer_stage, start, stop, mtp=mtp,
            expected=(state.slot_generation, state.history_generation))
        entry.pool(state.mtp_pooled if mtp else state.pooled[ai], a.ik_scale, frequencies,
                   c.eps, sections, amplitude)
        if mtp:
            expected = (state.slot_generation, state.history_generation)
            entry.publish_commit(entry.prepare_commit(pos + rows, expected), pos + rows, expected)
    if not mtp:
        state.target_tentative_length = pos + rows


def _decode(payload, scales, format, dims):
    if not format.quantized:
        return payload
    if format.codec:
        return ops._rotor_decode(payload, scales, dims, format.codec)
    return ops.dequantize_int8(payload, scales)


def _attention_tiles(query, cache, selected, visible, scale, *, values=False):
    """Bounded selected-domain tiles, excluding future/unowned operands."""
    dims = query.shape[1]
    for start in range(0, selected.numel(), 128):
        positions = selected[start:start + 128]
        offsets, owned = cache.local_indices(positions)
        mask = owned & visible[start:start + 128]
        key_payload = torch.where(mask[:, None, None], cache.k[offsets], 0)
        key_scales = (torch.where(mask[:, None, None], cache.ks[offsets], 0)
                      if cache.key_format.quantized else None)
        key = _decode(key_payload, key_scales,
                      cache.key_format, dims)[:, 0]
        key = torch.where(mask[:, None], key, 0)
        score = ops._matmul32(query, key.T) * scale
        score.masked_fill_(~mask[None], -torch.inf)
        value = None
        if values:
            value_payload = torch.where(mask[:, None, None], cache.v[offsets], 0)
            value_scales = (torch.where(mask[:, None, None], cache.vs[offsets], 0)
                            if cache.value_format.quantized else None)
            value = _decode(value_payload, value_scales,
                            cache.value_format, dims)[:, 0]
            value = torch.where(mask[:, None], value, 0)
        yield score, mask, value


def partial_statistics(query, cache, selected, visible, scale):
    """A shard's unnormalized FP32 maximum and denominator; no PV rounding."""
    heads = query.shape[0]
    maximum = torch.full((heads,), -torch.inf, dtype=torch.float32, device=query.device)
    denominator = torch.zeros_like(maximum)
    if not cache.capacity:
        return maximum, denominator
    for score, mask, _ in _attention_tiles(query, cache, selected, visible, scale):
        tile_max = score.amax(-1)
        next_max = torch.maximum(maximum, tile_max)
        finite = torch.isfinite(next_max)
        safe_max = torch.where(finite, next_max, 0)
        alpha = torch.where(torch.isfinite(maximum), torch.exp(maximum - safe_max), 0)
        probabilities = torch.where(mask[None], torch.exp(score - safe_max[:, None]), 0)
        denominator = denominator * alpha + probabilities.sum(-1)
        maximum = next_max
    return maximum, denominator


def merge_statistics(maxima, denominators, *, status=None, layer=0):
    """Global softmax normalization before the target's FP16 probability cast."""
    maximum = maxima.amax(0)
    safe = torch.where(torch.isfinite(maximum), maximum, 0)
    factor = torch.where(torch.isfinite(maxima), torch.exp(maxima - safe[None]), 0)
    denominator = (factor * denominators).sum(0)
    # The original attention domain includes the current token. An all-empty
    # or non-finite state is a contained invariant failure. The existing KV
    # status readback fences and rejects the frame before its publication.
    bad = (~torch.isfinite(denominator) | (denominator <= 0)).any()
    if status is None:
        if denominator.is_cuda:
            torch._assert_async(~bad, "unique attention has no finite visible softmax mass")
        elif bool(bad):
            raise ValueError("unique attention has no finite visible softmax mass")
    else:
        status[0].bitwise_or_(bad.int() * 16)
        status[1].copy_(torch.where(bad & (status[1] == -2), layer, status[1]))
    return safe, torch.where(denominator > 0, denominator, 1)


def normalized_partial(query, cache, selected, visible, scale, maximum, denominator):
    """PV with globally normalized FP16 probabilities, matching target rounding.

    A local unnormalized numerator cannot be rescaled after casting its
    probabilities to FP16. Re-reading bounded packed tiles preserves that
    rounding boundary without retaining a full FP16 history mirror.
    """
    result = torch.zeros(query.shape, dtype=torch.float32, device=query.device)
    if not cache.capacity:
        return result
    for score, mask, value in _attention_tiles(query, cache, selected, visible, scale, values=True):
        probability = torch.where(mask[None], torch.exp(score - maximum[:, None]) / denominator[:, None], 0).half()
        result = result + ops._matmul32(probability, value)
    return result


class OwnerBuffers:
    """Layer-batched communication slabs shared across every session/cache."""

    def _empty_resources(self):
        self.rows = self.heads = self.head_dim = self.idw = self.rank = self.device = self.prefill = self.comm = None
        self.queries = self.selection = self.selections = self.partial = self.partials = self.position = None
        self.native_states, self.native_owners, self.native_pair, self.native_buckets = None, (), None, ()
        self.descriptors = self.descriptors_host = self.global_stats = self.lane_rows = self.native_layer = None
        self.descriptor_host_words = None
        self.descriptor_upload = self.native_done = None
        self._upload_pending = self._native_pending = self._base_ready = self._construction_started = False
        self.native_comm = self.native_collective = None
        self._native_transport_configured = False
        self.reference_transport_rows = 0
        self.closed = False

    def __init__(self, weights, rows, scratch, *, prefill=False):
        if vars(self).get("_construction_started", False):
            raise RuntimeError("owner workspace construction is one-shot; retire and construct a new owner")
        self._empty_resources()
        self._construction_started = True
        c, device = weights.cfg, weights.device
        world, group = 8, 4
        self.rows, self.heads, self.prefill = rows, c.heads, prefill
        self.head_dim, self.rank, self.idw = c.head_dim, int(weights.meta.get("rank", 0)), scratch.idw
        self.device, self.comm = torch.device(device), getattr(weights, "comm", None)
        from .owner_execution import FULL_COLLECTIVE

        self.native_comm, self.native_collective = self.comm, FULL_COLLECTIVE
        self.queries = torch.empty((world, rows, c.heads, c.head_dim), dtype=torch.float16, device=device)
        self.device = self.queries.device
        self.selection = torch.zeros((rows, scratch.idw + 2), dtype=torch.int32, device=device)
        self.selections = torch.empty((world, rows, scratch.idw + 2), dtype=torch.int32, device=device)
        # Statistics and normalized PV are sequential phases. Every consumer
        # uses a flat prefix; the two stats words never coexist with PV here.
        extent = max(c.head_dim, 2)
        self.partial = torch.empty((rows, group * c.heads, extent), dtype=torch.float32, device=device)
        self.partials = torch.empty((world, rows, group * c.heads, extent), dtype=torch.float32, device=device)
        self.position = torch.zeros((1,), dtype=torch.int32, device=device)
        self._base_ready = True

    @classmethod
    def from_text(cls, text, *, rows, prefill, idw, world, rank, device, comm, register=None):
        """Header-only factory: no target/draft weights or State allocation."""
        if (type(world) is not int or world != 8 or type(rank) is not int or not 0 <= rank < world
                or type(rows) is not int or rows < 1 or type(prefill) is not bool):
            raise ValueError("owner preflight requires the actual TP8 rank and bounded execution role")
        if comm is None or comm.world != world or comm.rank != rank:
            raise ValueError("owner preflight requires the active matching TP8 communicator")
        if register is not None and not callable(register):
            raise ValueError("owner preflight registration must be callable")
        heads, kv_heads = text["num_attention_heads"], text["num_key_value_heads"]
        if type(heads) is not int or heads <= 0:
            raise ValueError("native owner header requires positive integer query heads")
        dim = text.get("head_dim") or text["hidden_size"] // heads
        budget, ratio = text.get("indexer_budget", 2048), text.get("indexer_compress_ratio", 4)
        if (any(type(value) is not int or value <= 0 for value in (heads, kv_heads, dim, budget, ratio))
                or heads % world or kv_heads != 2 or (heads // world, dim) != (3, 256)
                or type(idw) is not int or idw != budget + ratio):
            raise ValueError("native owner header geometry differs from TP8 H3/D256 and its canonical index width")
        weights = SimpleNamespace(cfg=SimpleNamespace(heads=heads // world, head_dim=dim),
                                  device=torch.device(device), meta={"rank": rank, "world": world}, comm=comm)
        owner = object.__new__(cls)
        owner._empty_resources()
        if register is not None:
            register(owner)  # Journal the canonical partial owner before any fallible device allocation.
        owner.__init__(weights, min(rows, 256), SimpleNamespace(idw=idw), prefill=prefill)
        return owner

    def allocation_record(self):
        """Owned tensor extents, separated from observed driver/graph residency.

        A pre-capacity owner is already present in the physical-free baseline;
        its adoption removes the matching future allocation from geometry. This
        receipt does not credit unused allocator or CUDA graph-pool reservation.
        """
        native = self.native_record()
        extras = sum(native[name] for name in ("descriptor_bytes", "global_statistics_bytes",
                                               "lane_control_bytes", "layer_control_bytes"))
        return {"rows": self.rows, "prefill": self.prefill, "heads": self.heads,
                "head_dim": self.head_dim, "index_width": self.idw,
                "rank": self.rank, "device": str(self.device), "complete": self._base_ready,
                "device_tensor_bytes": self.nbytes,
                "communication_tensor_bytes": self.nbytes - extras,
                "native_tensor_bytes": extras, "host_staging_bytes": native["host_descriptor_bytes"],
                "pinned_host_staging_bytes": native["pinned_descriptor_bytes"],
                "collective_family": native["collective_family"], "attention_members": native["attention_members"],
                "qualification": "construction extents; not physical allocator or execution qualification"}

    def prepare_native(self, states=None, *, pair=None, page_tokens=128, buckets=None):
        """Prepare fixed pointer tables before capture/admission; never select a kernel.

        A header-only caller may supply ``pair`` and no States. Zero descriptor
        rows are executable inactive work for startup capture. The canonical
        pool is bound later without replacing table/workspace addresses.
        """
        if not self._base_ready or self.closed:
            raise RuntimeError("native preparation requires a complete open owner workspace")
        from tensorfold.cuda.sm70.paged_attention import DESCRIPTOR_WORDS
        from tensorfold.cuda.sm70.paged_attention_descriptor import ABI_VERSION, WORD_BYTES, FIELDS
        from .owner_execution import row_buckets

        # The hot tuple below is intentionally direct. Its semantic field order
        # is checked once against the generated ABI authority before any pointer
        # table exists; a reordered interface cannot quietly reinterpret addresses.
        expected = ("K_DATA", "V_DATA", "K_METADATA", "V_METADATA", "PAGE_OFFSETS", "GENERATIONS",
                    "STATUS", "POSITION", "LIVE_ROWS", "EXPECTED_SLOT", "EXPECTED_OWNER",
                    "LOGICAL_CAPACITY", "PHYSICAL_ROWS", "PAGE_COUNT", "ROW_OFFSET", "INDEXER_OWNER")
        if ABI_VERSION != 1 or WORD_BYTES != 8 or FIELDS != expected:
            raise RuntimeError("native descriptor packing differs from its generated schema")
        prepared_buckets = self.native_buckets if buckets is None and self.native_buckets else row_buckets(self.rows, buckets)
        if self.native_buckets and self.native_buckets != prepared_buckets:
            raise ValueError("prepared native row catalogue cannot change during serving")
        if self.heads != 3 or self.head_dim != 256:
            raise ValueError("native paged owner attention requires local H3/D256")
        states = None if states is None else tuple(states)
        if pair is None and states:
            pair = states[0].kv_pair
        if pair is None:
            raise ValueError("native preparation needs the selected homogeneous KV format")
        fields = (pair.key.bits, pair.key.codec, pair.value.bits, pair.value.codec)
        if fields not in ((8, 0, 8, 0), (4, 9, 4, 9), (4, 10, 4, 10)):
            raise ValueError("native paged owner attention requires homogeneous ordinary Q8 or compact Q4")
        if self.native_pair is not None and self.native_pair != pair:
            raise ValueError("a prepared native owner workspace cannot change its codec identity")
        if type(page_tokens) is not int or page_tokens < 1 or page_tokens % 4:
            raise ValueError("native history pages must contain complete positive pooling groups")
        if self.native_pair is not None and self.native_page_tokens != page_tokens:
            raise ValueError("prepared native graphs cannot change their page geometry")
        if self.descriptors is None:
            device = self.queries.device
            self.descriptors = torch.zeros((self.rows, DESCRIPTOR_WORDS), dtype=torch.int64, device=device)
            self.descriptors_host = torch.zeros((self.rows, DESCRIPTOR_WORDS), dtype=torch.int64,
                                                pin_memory=self.queries.is_cuda)
            # Retain a zero-copy CPU view of the same pinned DMA mirror. Its writes
            # remain behind descriptor_upload retirement in stage_native_descriptors.
            self.descriptor_host_words = self.descriptors_host.numpy()
            self.global_stats = torch.empty((self.rows, 12, 2), dtype=torch.float32, device=device)
            self.lane_rows = torch.zeros((self.rows,), dtype=torch.int32, device=device)
            self.native_layer = torch.zeros((1,), dtype=torch.int32, device=device)
            if self.queries.is_cuda:
                self.descriptor_upload = torch.cuda.Event()
                self.native_done = torch.cuda.Event()
        self.native_pair = pair
        self.native_page_tokens = page_tokens
        self.native_buckets = prepared_buckets
        if self.queries.is_cuda:
            from tensorfold.cuda.sm70.paged_attention import extension

            extension()  # Compilation/module initialization is outside native_run/capture.
        if states is not None:
            self.bind_native_states(states)

    @staticmethod
    def _state_storage(state):
        tensors = [state.history_generations, state.kv_status, state.pos_dev]
        if hasattr(state, "mtp_kc"):
            tensors.append(state.mtp_pos)
        for cache in [*state.kc, *([state.mtp_kc] if hasattr(state, "mtp_kc") else [])]:
            tensors.extend((cache.k, cache.v, cache.ks, cache.vs, cache.page_offsets))
        return tuple(tensors)

    @staticmethod
    def _storage_signature(tensors):
        return tuple((tensor.data_ptr(), tuple(tensor.shape), tensor.dtype, tensor.device) for tensor in tensors)

    def bind_native_states(self, states):
        """Retain every address owner through graph/operation retirement.

        The catalogue is the already bounded canonical State pool, not a cache
        of arbitrary encountered requests. Replacing storage requires explicit
        drain/retirement before rebinding; same-pointer publication is not enough.
        """
        if self.descriptors is None:
            raise RuntimeError("prepare native owner tables before binding State storage")
        catalogue, owners = {}, []
        for state in states:
            if (id(state) in catalogue or state.history_placement is None or state.kv_pair != self.native_pair
                    or state.history_placement.page_tokens != self.native_page_tokens):
                raise ValueError("native catalogue needs distinct unique-history States of one codec")
            tensors = self._state_storage(state)
            if any(not isinstance(tensor, torch.Tensor) or tensor.device != self.queries.device for tensor in tensors):
                raise ValueError("native catalogue storage must share its workspace device")
            controls = [(state.history_generations, torch.int64, (2,)), (state.kv_status, torch.int32, (2,)),
                        (state.pos_dev, torch.int32, (1,))]
            if hasattr(state, "mtp_kc"):
                controls.append((state.mtp_pos, torch.int32, (1,)))
            if any(tensor.dtype != dtype or tuple(tensor.shape) != shape or not tensor.is_contiguous()
                   for tensor, dtype, shape in controls):
                raise ValueError("native State controls require the exact contiguous generation/status/position schema")
            for cache in [*state.kc, *([state.mtp_kc] if hasattr(state, "mtp_kc") else [])]:
                pages = (state.capacity + self.native_page_tokens - 1) // self.native_page_tokens
                if (cache.logical_capacity != state.capacity or cache.pair != self.native_pair
                        or cache.page_offsets.dtype != torch.int64 or cache.page_offsets.shape != (pages,)
                        or not cache.page_offsets.is_contiguous()):
                    raise ValueError("native cache page table differs from its immutable logical/codec plan")
                for payload, scales, fmt in ((cache.k, cache.ks, cache.key_format),
                                             (cache.v, cache.vs, cache.value_format)):
                    if (payload.dtype != getattr(torch, fmt.payload_dtype)
                            or scales.dtype != getattr(torch, fmt.scale_dtype)
                            or payload.shape != (cache.capacity, 1, 256 * fmt.bits // 8)
                            or scales.shape != (cache.capacity, 1, 256 // fmt.group)
                            or not payload.is_contiguous() or not scales.is_contiguous()):
                        raise ValueError("native packed cache tensors differ from their admitted physical row schema")
            catalogue[id(state)] = (state, self._storage_signature(tensors))
            owners.extend((state, *tensors))
        if self.native_states and self.native_states != catalogue:
            raise RuntimeError("native State storage changed; retire its borrowers before rebinding")
        self.native_states, self.native_owners = catalogue, tuple(owners)
        for state, _ in catalogue.values():
            state.history_native_readers.add(self)

    def wait_native_reuse(self):
        if self._native_pending:
            torch.cuda.current_stream(self.queries.device).wait_event(self.native_done)

    def stage_native_descriptors(self, segments, layer, base, count, *, mtp=False, bucket_rows=None):
        if self.descriptors is None or self.native_states is None:
            raise RuntimeError("native State catalogue is not prepared")
        bucket = count if bucket_rows is None else bucket_rows
        if (type(base) is not int or base < 0 or type(count) is not int or type(bucket) is not int
                or not 0 < count <= bucket <= self.rows):
            raise ValueError("native descriptor frame exceeds its bounded row envelope")
        if bucket not in self.native_buckets:
            raise ValueError("native descriptor bucket was not prepared before admission")
        spans, cursor = [], base
        for state, first, last in segments:
            if type(first) is not int or type(last) is not int or not 0 <= first < last or last - first > 2**31 - 1:
                raise ValueError("native lanes need bounded nonempty contiguous row ranges")
            begin, end = max(base, first), min(base + count, last)
            if begin >= end:
                continue
            if begin != cursor:
                raise ValueError("native descriptor lanes have a gap or overlap")
            registered = self.native_states.get(id(state))
            current = self._state_storage(state)
            if registered is None or registered[0] is not state or registered[1] != self._storage_signature(current):
                raise RuntimeError("native descriptor points outside its retained canonical storage catalogue")
            if (type(state.slot_generation) is not int or type(state.history_generation) is not int
                    or not 0 <= state.slot_generation <= 2**63 - 1 or not 0 <= state.history_generation <= 2**63 - 1):
                raise OverflowError("native descriptor generation exceeds signed int64")
            if mtp and not hasattr(state, "mtp_kc"):
                raise ValueError("native MTP descriptors require the selected resident MTP cache")
            ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
            cache = state.mtp_kc if mtp else state.kc[ai]
            position = state.mtp_pos if mtp else state.pos_dev
            index_owner = state.history_placement.indexer_owner(state.history_slot, ai,
                                                               include_mtp=hasattr(state, "mtp_kc"))
            spans.append((state, first, last, begin, end, cache, position, index_owner))
            cursor = end
        if cursor != base + count:
            raise ValueError("native descriptor lanes do not cover the live execution prefix")
        if self._upload_pending:
            self.descriptor_upload.synchronize()  # Retire only host DMA, not the complete model iteration.
        self.wait_native_reuse()
        host = self.descriptors_host[:bucket]
        host.zero_()
        self.lane_rows[:bucket].zero_()
        self.native_layer.fill_(layer.index)
        for lane, (state, first, last, begin, end, cache, position, index_owner) in enumerate(spans):
            self.lane_rows[lane].fill_(last - first)
            live_pointer = self.lane_rows.data_ptr() + lane * self.lane_rows.element_size()
            for row in range(begin, end):
                words = (cache.k.data_ptr(), cache.v.data_ptr(), cache.ks.data_ptr(), cache.vs.data_ptr(),
                         cache.page_offsets.data_ptr(), state.history_generations.data_ptr(), state.kv_status.data_ptr(),
                         position.data_ptr(), live_pointer, state.slot_generation, state.history_generation,
                         state.capacity, cache.capacity, cache.page_offsets.numel(), row - first, index_owner)
                self.descriptor_host_words[row - base] = words
        self.descriptors[:bucket].copy_(host, non_blocking=self.queries.is_cuda)
        if self.descriptor_upload is not None:
            self.descriptor_upload.record(torch.cuda.current_stream(self.queries.device))
            self._upload_pending = True

    def native_views(self, rows):
        if self.descriptors is None or type(rows) is not int or not 1 <= rows <= self.rows:
            raise ValueError("native views require a prepared bounded row bucket")
        q = self.queries.reshape(-1)[:8 * rows * 3 * 256].view(8, rows, 3, 256)
        selection = self.selections.reshape(-1)[:8 * rows * self.selection.shape[1]].view(8, rows, self.selection.shape[1])
        local_stats = self.partial.reshape(-1)[:rows * 12 * 2].view(rows, 12, 2)
        all_stats = self.partials.reshape(-1)[:8 * rows * 12 * 2].view(8, rows, 12, 2)
        local_values = self.partial.reshape(-1)[:rows * 12 * 256].view(rows, 12, 256)
        all_values = self.partials.reshape(-1)[:8 * rows * 12 * 256].view(8, rows, 12, 256)
        # Final value merge runs after normalized PV has consumed all queries.
        # Reuse that FP16 slab for output without another row-output allocation.
        out = self.queries.reshape(-1)[:rows * 3 * 256].view(rows, 3, 256)
        return {"q_all": q, "selected_all": selection, "row_table": self.descriptors[:rows],
                "local_stats": local_stats, "all_stats": all_stats, "global_stats": self.global_stats[:rows],
                "local_values": local_values, "all_values": all_values, "out": out}

    def graph_tensors(self):
        """Stable direct graph inputs; State pointers are separately leased by descriptors."""
        return {"queries": self.queries, "selections": self.selections, "partial": self.partial,
                "partials": self.partials, "descriptors": self.descriptors, "global_stats": self.global_stats,
                "lane_rows": self.lane_rows, "native_layer": self.native_layer}

    def native_bucket(self, actual_rows):
        if type(actual_rows) is not int or not 1 <= actual_rows <= self.rows or not self.native_buckets:
            raise ValueError("native owner rows exceed the prepared execution catalogue")
        return next(bucket for bucket in self.native_buckets if actual_rows <= bucket)

    def clear_native_inputs(self, rows):
        self.wait_native_reuse()
        views = self.native_views(rows)
        for name in ("q_all", "selected_all", "row_table"):
            views[name].zero_()
        self.selection[:rows].zero_()
        self.lane_rows[:rows].zero_()
        self.native_layer.zero_()
        if self.descriptor_upload is not None:
            self.descriptor_upload.record(torch.cuda.current_stream(self.queries.device))
            self._upload_pending = True

    def wait_native_inputs(self):
        if self._upload_pending:
            torch.cuda.current_stream(self.queries.device).wait_event(self.descriptor_upload)

    def configure_native_transport(self, policy, comm):
        """Fix an initialized startup communicator; never switch with live work."""
        from .owner_execution import FULL_COLLECTIVE, HEAD_COLLECTIVE

        if self._native_transport_configured or self.native_states is not None or self._native_pending:
            raise RuntimeError("native owner transport is immutable after startup configuration")
        world = 8 if policy == FULL_COLLECTIVE else 4 if policy == HEAD_COLLECTIVE else 0
        if (not world or comm is None or comm.world != world or comm.rank != self.rank % world
                or (world == 8 and comm is not self.comm)):
            raise ValueError("native owner transport does not match its fixed head/rank group")
        self.native_comm, self.native_collective = comm, policy
        self._native_transport_configured = True

    def gather_native(self, send, destination):
        # The native kernels consume only the four logical ranks sharing this
        # KV head. Preserve global rank indices/merge order in the same slabs;
        # the other head's entries are neither evaluated nor merged.
        if self.native_comm.world == 4:
            first = self.rank // 4 * 4
            destination = destination[first:first + 4]
        self.native_comm.all_gather(send, destination)

    def warm_native_transport(self, rows):
        """Inactive startup only: initialize every used wire datatype/shape."""
        if self.native_states is not None or self._native_pending:
            raise RuntimeError("native transport warmup requires the unbound startup workspace")
        if rows not in self.native_buckets:
            raise ValueError("native transport shape was not prepared before startup warmup")
        views = self.native_views(rows)
        self.gather_native(views["q_all"][self.rank], views["q_all"])
        # Canonical selection remains global even under the four rank policy.
        self.comm.all_gather(self.selection[:rows], views["selected_all"])

    def warm_reference_transport(self):
        """Warm every still-eligible exact reference wire shape before capacity.

        Reference attention does not pad its rows. It remains selectable after
        native preparation, so head4 bucket warmup alone cannot bound full8
        datatype/shape lazy residency. The owner envelope is at most256 rows.
        """
        if self.native_states is not None or self._native_pending or self.reference_transport_rows:
            raise RuntimeError("reference transport warmup requires the untouched startup workspace")
        for rows in range(1, self.rows + 1):
            views = self.native_views(rows)
            # Reference q is out-of-place. Reuse a disjoint, sufficiently large
            # partial slab before its statistics/PV phases; no new tensor data
            # allocation is needed. NCCL overwrites the complete receive view.
            query = self.partial.view(torch.float16).reshape(-1)[:rows * self.heads * self.head_dim].view(
                rows, self.heads, self.head_dim)
            query.zero_()
            self.selection[:rows].zero_()
            self.comm.all_gather(query, views["q_all"])
            self.comm.all_gather(self.selection[:rows], views["selected_all"])
            views["local_stats"].zero_()
            self.comm.all_gather(views["local_stats"], views["all_stats"])
            views["local_values"].zero_()
            self.comm.all_gather(views["local_values"], views["all_values"])
        if self.queries.is_cuda:
            torch.cuda.synchronize(self.device)
        self.reference_transport_rows = self.rows

    def native_run(self, rows):
        """The same preallocated partial-attention pipeline for eager and graphs."""
        from tensorfold.cuda.sm70 import paged_attention as native

        if rows not in self.native_buckets:
            raise ValueError("native owner execution shape was not prepared before admission")
        views = self.native_views(rows)
        if self.comm is None or self.comm.world != 8 or self.comm.rank != self.rank:
            raise ValueError("native owner execution requires its active TP8 communicator")
        bits, codec = self.native_pair.key.bits, self.native_pair.key.codec
        common = {"group_start": self.rank // 4 * 4, "page_tokens": self.native_page_tokens, "layer": self.native_layer}
        native.local_statistics(views["q_all"], views["selected_all"], views["row_table"], views["local_stats"],
                                key_bits=bits, key_codec=codec, scale=256 ** -0.5, **common)
        self.gather_native(views["local_stats"], views["all_stats"])
        native.merge_statistics(views["all_stats"], views["row_table"], views["global_stats"], **common)
        native.normalized_partial(views["q_all"], views["selected_all"], views["row_table"], views["global_stats"],
                                  views["local_values"], key_bits=bits, key_codec=codec,
                                  value_bits=bits, value_codec=codec, scale=256 ** -0.5, **common)
        self.gather_native(views["local_values"], views["all_values"])
        native.merge_values(views["all_values"], views["row_table"], views["out"], rank=self.rank,
                             page_tokens=self.native_page_tokens, layer=self.native_layer, value_bits=bits, value_codec=codec)
        return views["out"]

    def mark_native_submitted(self):
        if self.native_done is not None:
            self.native_done.record(torch.cuda.current_stream(self.queries.device))
            self._native_pending = True

    def retire_native(self):
        """Caller stops graph replays first; storage remains owned if a fence fails."""
        device_owned = any(isinstance(value, torch.Tensor) and value.is_cuda for value in vars(self).values())
        if device_owned:
            torch.cuda.synchronize(self.device)
        if self.descriptors is not None:
            self.descriptors.zero_()
            if self.lane_rows is not None:
                self.lane_rows.zero_()
            if device_owned:
                torch.cuda.synchronize(self.device)
        if self.native_states is not None:
            for state, _ in self.native_states.values():
                state.history_native_readers.discard(self)
        self.native_states, self.native_owners = None, ()
        self._upload_pending = self._native_pending = False

    def close(self):
        """After graph retirement, release even partially initialized tensor owners.

        A constructor failure may remain in a traceback. Clearing these slots
        after proven retirement releases their allocations despite that reference.
        Failed fences retain the complete owner for terminal cleanup reporting.
        """
        if self.closed:
            return
        self.retire_native()
        self.queries = self.selection = self.selections = self.partial = self.partials = self.position = None
        self.descriptors = self.descriptors_host = self.global_stats = self.lane_rows = self.native_layer = None
        self.descriptor_host_words = None
        self.descriptor_upload = self.native_done = None
        self.native_comm = None
        self.closed, self._base_ready = True, False

    def native_record(self):
        return {"prepared": self.descriptors is not None,
                "collective_family": self.native_collective,
                "attention_members": list(range(self.rank // 4 * 4, self.rank // 4 * 4 + 4))
                                     if self.native_comm is not None and self.native_comm.world == 4 else list(range(8)),
                "native_rows": list(self.native_buckets),
                "reference_transport_rows": self.reference_transport_rows,
                "bound_states": 0 if self.native_states is None else len(self.native_states),
                "descriptor_bytes": 0 if self.descriptors is None else self.descriptors.nbytes,
                "global_statistics_bytes": 0 if self.global_stats is None else self.global_stats.nbytes,
                "lane_control_bytes": 0 if self.lane_rows is None else self.lane_rows.nbytes,
                "layer_control_bytes": 0 if self.native_layer is None else self.native_layer.nbytes,
                "host_descriptor_bytes": 0 if self.descriptors_host is None else self.descriptors_host.nbytes,
                "pinned_descriptor_bytes": (self.descriptors_host.nbytes if self.descriptors_host is not None
                                             and self.descriptors_host.is_pinned() else 0)}

    @property
    def nbytes(self):
        return sum(value.nbytes for name, value in vars(self).items()
                   if isinstance(value, torch.Tensor) and name != "descriptors_host")


def attention_layer(layer, weights, segments, buffers, total_rows, *, mtp=False):
    """Retire all-rank coordinate reads, including ranks without this indexer."""
    explicit = buffers.mtp_indexer_coordinates if mtp else buffers.target_indexer_coordinates
    if explicit is None:
        return _attention_layer(layer, weights, segments, buffers, total_rows, mtp=mtp)
    with buffers.compact_indexer_stage.reading(mtp=mtp):
        return _attention_layer(layer, weights, segments, buffers, total_rows, mtp=mtp)


def _attention_layer(layer, weights, segments, buffers, total_rows, *, mtp=False):
    """Execute owner history with four ordered batched collectives per chunk.

    Per-row causal selection stays canonical. The two KV-head groups produce
    disjoint attention partials while all eight ranks retain their TP work.
    Packed histories are never gathered or expanded in full.
    """
    c, rank, comm = weights.cfg, weights.meta["rank"], weights.comm
    if comm is None or comm.world != 8 or comm.rank != rank:
        raise ValueError("unique history requires its live TP8 communicator")
    owner = buffers.history_owner
    family = buffers.owner_kernel_family
    native_selected = family != "torch-reference"
    if buffers.owner_graph_family not in ("uncaptured", "owner-attention-v1"):
        raise ValueError("unsupported owner attention graph family")
    if buffers.owner_graph_family != "uncaptured" and (not native_selected or buffers.owner_graphs is None):
        raise RuntimeError("owner attention graph family was not prepared before serving")
    if native_selected:
        from tensorfold.cuda.sm70.paged_attention import KERNEL_FAMILY

        if family != KERNEL_FAMILY:
            raise ValueError("unsupported owner attention kernel family")
        if owner.native_states is None or any(id(state) not in owner.native_states for state, _, _ in segments):
            raise RuntimeError("native owner kernel and its canonical State catalogue were not prepared before serving")
    from .compact_indexer import validate_mtp_indexer_frame, validate_target_indexer_frame

    if mtp:
        validate_mtp_indexer_frame(buffers, segments)
    else:
        validate_target_indexer_frame(buffers, segments)
    owner.wait_native_reuse()  # Prior raw-pointer readers retire before any new persistent cache write.
    if any(getattr(state, "graph_rows", None) is not None for state, _, _ in segments):
        raise ValueError("unique owner attention needs exact live rows; padded graph envelopes are not qualified")
    # All input rows belong to exactly one live segment; reject gaps/overlaps
    # before any cache write or distributed submission.
    cursor = 0
    for state, first, last in segments:
        if first != cursor or not first < last <= total_rows or state.history_placement is None:
            raise ValueError("unique history requires contiguous live session segments")
        cursor = last
    if cursor != total_rows:
        raise ValueError("unique history segments do not cover the execution rows")
    if mtp or segments[0][0].att_index[layer.index] == 0:
        buffers.compact_indexer_stage.begin(mtp=mtp)
    batch = getattr(buffers, "owner_batch_prelude", None)
    # Keep the selected scalar N1 route; batch only actual multi-State frames.
    prepared = batch is not None and len(segments) > 1 and batch.submit(
        weights, buffers, layer, segments, total_rows, mtp=mtp)
    if not prepared:
        for state, first, last in segments:
            prepare(buffers.pa, state, layer, weights, buffers, first, last, mtp=mtp)
    for base in range(0, total_rows, owner.rows):
        count = min(owner.rows, total_rows - base)
        graph_manager = buffers.owner_graphs if native_selected else None
        bucket = (owner.native_bucket(count) if graph_manager is None else graph_manager.bucket(count)) if native_selected else count
        query_all = owner.queries.reshape(-1)[:8 * bucket * c.heads * c.head_dim].view(8, bucket, c.heads, c.head_dim)
        if native_selected:
            owner.wait_native_reuse()
            send = query_all[rank]
            send[:count].copy_(buffers.q[base:base + count])
            send[count:].zero_()
            owner.gather_native(send, query_all)  # Exactly this rank's legal in-place receive slice.
        else:
            comm.all_gather(buffers.q[base:base + count], query_all)
        selection = owner.selection[:bucket]
        selection.zero_()
        selected_segments = []
        for state, first, last in segments:
            begin, end = max(base, first), min(base + count, last)
            if begin >= end:
                continue
            ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
            index_owner = state.history_placement.indexer_owner(state.history_slot, ai,
                                                               include_mtp=hasattr(state, "mtp_kc"))
            host_pos = (state.mtp_len if mtp else state.pos) + begin - first
            selected_segments.append((state, ai, begin - base, end - base, host_pos, index_owner))
            if rank == index_owner:
                scratch = copy(buffers.attn)
                # qsa_rows expects contiguous native destinations. Canonical
                # scratch is contiguous; copy its bounded result to the wire slab.
                scratch.ids = buffers.attn.ids[:end - begin]
                scratch.nk = buffers.attn.nk[:end - begin]
                scratch.sparse = buffers.attn.sparse[:end - begin]
                owner.position.fill_(host_pos)
                pooled = state.mtp_pooled if mtp else state.pooled[ai]
                ops.qsa_rows(buffers.iq[begin:end], pooled, owner.position, scratch, end - begin,
                             context=host_pos + end - begin)
                selection[begin - base:end - base, :buffers.attn.idw].copy_(scratch.ids)
                selection[begin - base:end - base, buffers.attn.idw].copy_(scratch.nk)
                selection[begin - base:end - base, buffers.attn.idw + 1].copy_(scratch.sparse)
        selections = owner.selections.reshape(-1)[:8 * bucket * selection.shape[1]].view(8, bucket, selection.shape[1])
        comm.all_gather(selection, selections)
        if native_selected:
            owner.stage_native_descriptors(segments, layer, base, count, mtp=mtp, bucket_rows=bucket)
            if graph_manager is None:
                owner.wait_native_inputs()
                output = owner.native_run(bucket)[:count]
            else:
                output = graph_manager.run(count, graph=buffers.owner_graph_family
                                           == "owner-attention-v1")
            owner.wait_native_reuse()
            buffers.attn_o[base:base + count].copy_(output)
            owner.mark_native_submitted()  # The lease includes this last consumer of the reused query/output slab.
            continue
        partial = owner.partial.reshape(-1)[:count * 4 * c.heads * 2].view(count, 4 * c.heads, 2)
        group_start = rank // 4 * 4
        jobs = []
        for state, ai, first, last, host_pos, index_owner in selected_segments:
            cache = state.mtp_kc if mtp else state.kc[ai]
            for row in range(first, last):
                visible_end = host_pos + row - first + 1
                complete = visible_end // c.index_ratio
                if complete > c.index_budget // c.index_ratio:
                    n = (c.index_budget // c.index_ratio) * c.index_ratio + visible_end % c.index_ratio
                    ids = selections[index_owner, row, :n].long()
                else:
                    ids = torch.arange(visible_end, dtype=torch.int64, device=weights.device)
                query = query_all[group_start:group_start + 4, row].reshape(4 * c.heads, c.head_dim)
                visible = ids < visible_end
                maximum, denominator = partial_statistics(query, cache, ids, visible, c.head_dim ** -0.5)
                partial[row, :, 0].copy_(maximum)
                partial[row, :, 1].copy_(denominator)
                jobs.append((state, cache, row, query, ids, visible))
        partials = owner.partials.reshape(-1)[:8 * partial.numel()].view(8, *partial.shape)
        comm.all_gather(partial, partials)
        normalized = []
        for state, cache, row, query, ids, visible in jobs:
            pieces = partials[group_start:group_start + 4, row]
            maximum, denominator = merge_statistics(pieces[..., 0], pieces[..., 1],
                                                     status=state.kv_status, layer=layer.index)
            normalized.append((state, cache, row, query, ids, visible, maximum, denominator))
        value_part = owner.partial.reshape(-1)[:count * 4 * c.heads * c.head_dim].view(count, 4 * c.heads, c.head_dim)
        for _, cache, row, query, ids, visible, maximum, denominator in normalized:
            value_part[row].copy_(normalized_partial(query, cache, ids, visible, c.head_dim ** -0.5,
                                                     maximum, denominator))
        value_parts = owner.partials.reshape(-1)[:8 * value_part.numel()].view(8, *value_part.shape)
        comm.all_gather(value_part, value_parts)
        local_heads = slice((rank % 4) * c.heads, (rank % 4 + 1) * c.heads)
        for state, ai, first, last, _, _ in selected_segments:
            cache = state.mtp_kc if mtp else state.kc[ai]
            for row in range(first, last):
                pieces = value_parts[group_start:group_start + 4, row, local_heads]
                result = pieces[0]
                for peer in range(1, 4):
                    result = result + pieces[peer]
                if cache.value_format.codec:
                    ops._rotor_rotate(result[None], buffers.attn_o[base + row:base + row + 1],
                                      codec=cache.value_format.codec, inverse=True,
                                      status=state.kv_status, layer=layer.index)
                elif cache.value_format.quantized:
                    buffers.attn_o[base + row].copy_(ops.h32(result).half())
                else:
                    buffers.attn_o[base + row].copy_(result.half())
    return buffers.attn_o[:total_rows]


def prefix_retirement_failure(destination, source):
    for state in (source, destination):
        if state.compact_indexer is not None:
            failure = state.compact_indexer.retirement_failure
            if failure is not None:
                return failure
    return None


def prefix_failure(destination, error, poison_control):
    from tensorfold.cleanup import raise_failures

    destination._history_error = error
    try:
        if poison_control is not None:
            poison_control(error)
        else:
            from tensorfold.cuda.cohort import notify_failure

            notify_failure()
    except BaseException as notification:
        raise_failures(error, (notification,))


def check_prefix_retirement(destination, source, poison_control):
    failure = prefix_retirement_failure(destination, source)
    if failure is not None:
        prefix_failure(destination, failure, poison_control)
        raise failure


def copy_prefix(destination, source, target_rows, mtp_rows, *, indexer_snapshot=None, poison_control=None):
    """Move pooled prefixes and the exact captured pending tails, never old raw history.

    All scratch and restore-tail payloads are acquired/voted before the first
    transfer. An uncertain transfer poisons the cohort before ordinary ACKs.
    """
    from tensorfold.cleanup import raise_failures
    from tensorfold.cuda.memory_gate import NoRoom
    from .compact_indexer import SCHEMA

    placement, comm = destination.history_placement, destination.history_comm
    if comm is None or any(not hasattr(comm, name) for name in ("broadcast", "all_gather")):
        raise ValueError("unique prefix copies require the TP8 data/status communicator")
    def fatal(error):
        prefix_failure(destination, error, poison_control)

    def unretired():
        return prefix_retirement_failure(destination, source)

    failure = unretired()
    if failure is not None:
        fatal(failure)
        raise failure
    try:
        for state in (source, destination):
            for reader in tuple(state.history_native_readers):
                reader.wait_native_reuse()
    except BaseException as error:
        fatal(error)  # A failed prior-reader fence is never fresh staging exhaustion.
        raise
    local_error, local_kind = None, 0
    try:
        if source.compact_indexer is None or destination.compact_indexer is None:
            raise ValueError("unique prefix copies require matching compact indexer owners")
        snapshot = (source.compact_indexer.snapshot(target_rows, mtp_rows)
                    if indexer_snapshot is None else indexer_snapshot)
        source.compact_indexer.validate_snapshot(snapshot, target_rows, mtp_rows)
        lengths = destination.compact_indexer.lengths(target_rows, mtp_rows)
        tails = tuple(None if entry is None else {
            "origin": entry.origin, "length": length, "pooled_valid": length // destination.ratio,
            "raw": torch.empty_like(entry.raw), "coordinates": torch.empty_like(entry.coordinates)}
            for entry, length in zip(destination.compact_indexer.entries, lengths))
        relocated = {"schema": SCHEMA, "lengths": lengths, "tails": tails}
        model = destination.kc[0]
        frame = KVCache(placement.page_tokens, 1, model.head_dim, model.k.device, pair=model.pair)
        frame_index = torch.empty((placement.page_tokens, destination.index_dim), dtype=torch.float16, device=model.k.device)
        frame_coordinates = torch.empty((4, 3), dtype=torch.int64, device=model.k.device)
        layers = [(ordinal, destination.kc[ordinal], source.kc[ordinal], target_rows,
                   destination.pooled[ordinal], source.pooled[ordinal]) for ordinal in range(len(destination.kc))]
        if hasattr(destination, "mtp_kc"):
            layers.append((placement.target_layers, destination.mtp_kc, source.mtp_kc, mtp_rows,
                           destination.mtp_pooled, source.mtp_pooled))
        transfer_tensors = [frame.k, frame.v, frame_index, frame_coordinates]
        if frame.key_format.quantized:
            transfer_tensors.append(frame.ks)
        if frame.value_format.quantized:
            transfer_tensors.append(frame.vs)
        for _, dst, src, _, dpool, spool in layers:
            transfer_tensors.extend((dst.k, dst.v, src.k, src.v, dpool, spool))
            if dst.key_format.quantized:
                transfer_tensors.extend((dst.ks, src.ks))
            if dst.value_format.quantized:
                transfer_tensors.extend((dst.vs, src.vs))
        for collection in (snapshot["tails"], tails):
            for tail in collection:
                if tail is not None:
                    transfer_tensors.extend((tail["raw"], tail["coordinates"]))
        transfer_tensors = tuple(transfer_tensors)
    except (MemoryError, torch.cuda.OutOfMemoryError) as error:
        failure = unretired()
        if failure is not None:
            fatal(failure)
            raise_failures(failure, (error,))
        local_error, local_kind = error, 1
    except ValueError as error:
        failure = unretired()
        if failure is not None:
            fatal(failure)
            raise_failures(failure, (error,))
        local_error, local_kind = error, 2
    except BaseException as error:
        fatal(error)
        raise
    try:
        destination.kv_status[0].fill_(local_kind)
        destination.kv_status[1].fill_(-2)
        comm.all_gather(destination.kv_status, destination._kv_peer_status)
        kinds = [int(row[0]) for row in destination._kv_peer_status.cpu().tolist()]
        destination.kv_status[0].zero_()
    except BaseException as error:
        fatal(error)
        raise
    if any(kinds):
        ranks = [rank for rank, kind in enumerate(kinds) if kind]
        if any(kind == 2 for kind in kinds):
            raise ValueError(f"compact prefix snapshot refused on ranks {ranks}") from local_error
        raise NoRoom(f"compact prefix staging unavailable on ranks {ranks}") from local_error
    try:
        with source.compact_indexer.transferring(destination.compact_indexer, transfer_tensors):
            for layer, dst, src, rows, dpool, spool in layers:
                for page in range((rows + placement.page_tokens - 1) // placement.page_tokens):
                    count = min(placement.page_tokens, rows - page * placement.page_tokens)
                    for head in range(placement.global_kv_heads):
                        root = src.owners[page] % placement.group_size + head * placement.group_size
                        if destination.history_rank == root:
                            offset = src.offsets[page]
                            frame.k[:count].copy_(src.k[offset:offset + count])
                            frame.v[:count].copy_(src.v[offset:offset + count])
                            if dst.key_format.quantized:
                                frame.ks[:count].copy_(src.ks[offset:offset + count])
                            if dst.value_format.quantized:
                                frame.vs[:count].copy_(src.vs[offset:offset + count])
                        for tensor in (frame.k[:count], frame.v[:count]):
                            comm.broadcast(tensor, root)
                        if dst.key_format.quantized:
                            comm.broadcast(frame.ks[:count], root)
                        if dst.value_format.quantized:
                            comm.broadcast(frame.vs[:count], root)
                        if destination.history_rank // placement.group_size == head:
                            dst.append(page * placement.page_tokens, frame.k[:count], frame.v[:count],
                                       frame.ks[:count], frame.vs[:count])
                source_owner = placement.indexer_owner(source.history_slot, layer, include_mtp=hasattr(source, "mtp_kc"))
                target_owner = placement.indexer_owner(destination.history_slot, layer, include_mtp=hasattr(destination, "mtp_kc"))
                for first in range(0, rows // destination.ratio, placement.page_tokens):
                    count = min(placement.page_tokens, rows // destination.ratio - first)
                    view = frame_index[:count]
                    if destination.history_rank == source_owner:
                        view.copy_(spool[first:first + count])
                    comm.broadcast(view, source_owner)
                    if destination.history_rank == target_owner:
                        dpool[first:first + count].copy_(view)
                if destination.history_rank == source_owner:
                    frame_index[:4].copy_(snapshot["tails"][layer]["raw"])
                    frame_coordinates.copy_(snapshot["tails"][layer]["coordinates"])
                comm.broadcast(frame_index[:4], source_owner)
                comm.broadcast(frame_coordinates, source_owner)
                if destination.history_rank == target_owner:
                    tails[layer]["raw"].copy_(frame_index[:4])
                    tails[layer]["coordinates"].copy_(frame_coordinates)
            return relocated
    except BaseException as error:
        fatal(error)
        raise


def begin_migration(state, degree, *, overlap_bytes=None):
    """Prepare a degree transition; execute only at serialized batch boundaries.

    The CPU/GPU worker that owns this State must serialize this API with
    forward, commit, restore, reset, prefix copy, resize and retirement. It does
    not submit background work. Automatic degree selection requires measured
    profiles and is intentionally separate from this transfer mechanism.
    """
    from dataclasses import replace
    from .history_placement import MIGRATION_OVERLAP_BYTES

    placement = state.history_placement
    if placement is None or placement.target_layers % 4:
        raise ValueError("live degree migration requires matching target-layer quartets")
    if any(cache.arena is None or cache.arena is not state.kc[0].arena for cache in state.kc):
        raise ValueError("live migration requires one admitted shared quantized target arena")
    cap = MIGRATION_OVERLAP_BYTES if overlap_bytes is None else overlap_bytes
    if type(cap) is not int or not 0 < cap <= MIGRATION_OVERLAP_BYTES:
        raise ValueError("migration overlap must fit the admitted 64 MiB/rank candidate cap")
    target = replace(placement, degree=degree)
    state.kv_check()
    if state.mtp_drafted or getattr(state, "target_tentative_length", state.pos) != state.pos:
        raise ValueError("resolve speculative rows before changing history ownership")
    # Coalescing begins from actual published page maps; never reset a partial
    # transition to its original formula after those source slots were reused.
    state._history_migration = {"target": target, "next_bundle": 0, "overlap_bytes": cap}


def _fence(state):
    cache = state.kc[0]
    if cache.k.is_cuda:
        # Initial safe serial implementation. Native event-scoped overlap is
        # a later qualified candidate; a callback or timeout is not completion.
        torch.cuda.synchronize(cache.k.device)


def migrate_next_bundle(state):
    """Publish at most one balanced page/quartet bundle, retaining mixed maps.

    Each rank owns one same-sized source page in a matching quartet. A degree
    change permutes those owners; the destination reuses its vacated arena
    slot. Two compressed page frames bound all additional payload occupancy,
    independent of retained context length. MTP/indexer ownership is unchanged.
    """
    pending = getattr(state, "_history_migration", None)
    if pending is None:
        return False
    if pending.get("blocked_reason") is not None:
        return False
    state.kv_check()
    if state.mtp_drafted or getattr(state, "target_tentative_length", state.pos) != state.pos:
        raise ValueError("resolve speculative rows before migrating a history bundle")
    if state.compact_indexer is not None and state.compact_indexer.has_pending_target():
        raise ValueError("resolve compact indexer appends before migrating history")
    comm, rank, placement = state.history_comm, state.history_rank, state.history_placement
    if comm is None or any(not hasattr(comm, name) for name in ("broadcast", "barrier", "all_gather")):
        raise ValueError("migration requires the active TP8 broadcast/barrier/status communicator")
    pages = len(state.kc[0].offsets)
    quartets = placement.target_layers // 4
    limit = pages * quartets
    while pending["next_bundle"] < limit:
        bundle = pending["next_bundle"]
        page, quartet = divmod(bundle, quartets)
        layers = tuple(range(quartet * 4, quartet * 4 + 4))
        target = pending["target"]
        old_local = [state.kc[layer].owners[page] % 4 for layer in layers]
        new_local = [target.kv_owner(state.history_slot, layer, 0, page) for layer in layers]
        if old_local == new_local:
            pending["next_bundle"] += 1
            continue
        if state.history_generation >= 2**63 - 1:
            raise OverflowError("history ownership generation exhausted; retire the session without migration")
        if sorted(old_local) != list(range(4)) or sorted(new_local) != list(range(4)):
            raise RuntimeError("history bundle is not a balanced quartet permutation")
        count = placement.physical_page_rows(state.capacity, page)
        model_cache = state.kc[layers[0]]
        temporary_bytes = 2 * model_cache.pair.nbytes(count, 1, model_cache.head_dim)
        if temporary_bytes > pending["overlap_bytes"]:
            # Keep serving from the actual current table. No allocation or
            # publication occurs when the next bundle cannot fit its promise.
            pending["blocked_reason"] = "next balanced bundle exceeds admitted migration overlap"
            return False
        _fence(state)
        comm.barrier()
        source_layer = layers[old_local.index(rank % 4)]
        target_layer = layers[new_local.index(rank % 4)]
        source_cache = state.kc[source_layer]
        source_offset = source_cache.offsets[page]
        source = transport = None
        source_staged = False
        completed = False
        try:
            local_error = None
            try:
                if source_offset < 0:
                    raise RuntimeError("published source page has no local arena slot")
                # Charge before construction: a failed constructor's traceback
                # can retain its partially initialized tensor owner.
                state.history_migration_bytes = temporary_bytes
                source = KVCache(count, 1, model_cache.head_dim, model_cache.k.device, pair=model_cache.pair)
                transport = KVCache(count, 1, model_cache.head_dim, model_cache.k.device, pair=model_cache.pair)
                for original, staged in ((source_cache.k, source.k), (source_cache.v, source.v),
                                         (source_cache.ks, source.ks), (source_cache.vs, source.vs)):
                    staged.copy_(original[source_offset:source_offset + count])
                source_staged = True
            except BaseException as error:
                local_error = error
                resource = isinstance(error, (MemoryError, torch.cuda.OutOfMemoryError))
                state.kv_status[0].fill_(1 if resource else 2)
            # A rank-local allocation/staging failure must be known everywhere
            # before any peer enters payload collectives. This reuses already
            # admitted status storage; the readback also fences local staging.
            comm.all_gather(state.kv_status, state._kv_peer_status)
            votes = state._kv_peer_status.cpu().tolist()
            failed = [peer for peer, words in enumerate(votes) if words[0]]
            if failed:
                if all(words[0] in (0, 1) for words in votes):
                    # No arena/table write or payload submission has occurred.
                    # The status readback fences every local stage. Release
                    # partial constructors too; expected exhaustion keeps the
                    # current executable mapping and never becomes a retry loop.
                    if local_error is not None:
                        traceback.clear_frames(local_error.__traceback__)
                    source = transport = local_error = None
                    state.kv_status[0].zero_()
                    state.history_migration_bytes = 0
                    pending["blocked_reason"] = f"migration staging resource exhaustion on rank(s) {failed}"
                    return False
                if local_error is not None:
                    raise local_error
                raise RuntimeError(f"history migration staging failed on rank(s) {failed}")
            for layer, old, new in zip(layers, old_local, new_local):
                if old == new:
                    continue
                for head in range(2):
                    root, destination = head * 4 + old, head * 4 + new
                    if rank == root:
                        for staged, wire in ((source.k, transport.k), (source.v, transport.v),
                                             (source.ks, transport.ks), (source.vs, transport.vs)):
                            wire.copy_(staged)
                    for wire in (transport.k, transport.v, transport.ks, transport.vs):
                        comm.broadcast(wire, root)
                    if rank == destination:
                        for arena, wire in ((source_cache.k, transport.k), (source_cache.v, transport.v),
                                            (source_cache.ks, transport.ks), (source_cache.vs, transport.vs)):
                            arena[source_offset:source_offset + count].copy_(wire)
            _fence(state)
            comm.barrier()  # Every source stays staged until all copies acknowledge completion.
            for layer, new in zip(layers, new_local):
                cache = state.kc[layer]
                offsets = list(cache.offsets)
                offsets[page] = source_offset if layer == target_layer else -1
                cache.page_offsets[page].fill_(offsets[page])
                cache.offsets = tuple(offsets)
                cache.owners[page] = rank // 4 * 4 + new
            next_generation = state.history_generation + 1
            generations = getattr(state, "history_generations", None)
            if generations is not None:
                generations[1].fill_(next_generation)
            _fence(state)
            comm.barrier()  # Publish the ownership generation only after all device tables agree.
            state.history_generation = next_generation
            pending["next_bundle"] += 1
            completed = True
        except BaseException as error:
            # The source staging frame is retained until the bundle resolves.
            # Restore this rank's original authoritative bytes before declaring
            # the session poisoned; never execute from an ambiguous table.
            if source_staged:
                try:
                    for arena, staged in ((source_cache.k, source.k), (source_cache.v, source.v),
                                          (source_cache.ks, source.ks), (source_cache.vs, source.vs)):
                        arena[source_offset:source_offset + count].copy_(staged)
                    _fence(state)
                except BaseException as cleanup:
                    error.add_note(f"history source restoration failed: {cleanup!r}")
            state._history_error = RuntimeError("history migration failed; this session requires exclusive retirement")
            state._history_error.__cause__ = error
            state._kv_error = state._history_error
            # The original failure traceback can retain submitted copies and
            # staging frames. Keep explicit ownership and honest occupancy
            # until the failed distributed operation is exclusively retired.
            state._history_failed_frames = (source, transport)
            raise
        finally:
            if completed:
                state.history_migration_bytes = 0
        return True
    state.history_placement = pending["target"]
    for cache in state.kc:
        cache.placement = pending["target"]
    state._history_migration = None
    return False

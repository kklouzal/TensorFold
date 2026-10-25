"""Unique GPU history storage and bounded owner attention for the SM70 path.

Weights stay TP8. Only packed KV pages and the canonical raw/pooled indexer
are retained; communication scratch is bounded by the current execution rows.
This source implementation requires separate native/full-model qualification.
"""

from __future__ import annotations

from copy import copy
import traceback

import torch

from tensorfold.cuda.sm70 import family_ops as ops

from .kvcache import KVCache


class OwnedKVCache(KVCache):
    """Packed local pages; logical positions remain independent of rank offsets."""

    __slots__ = ("placement", "slot", "layer", "rank", "logical_capacity", "offsets", "page_offsets",
                 "arena", "reserved_rows", "owners")

    def __init__(self, rows, kv_heads, head_dim, device, *, pair, placement, slot, layer, rank,
                 arena=None, arena_offset=0, owners=None):
        if kv_heads != 1:
            raise ValueError("unique history expects one weight-shard KV head")
        self.placement, self.slot, self.layer, self.rank = placement, slot, layer, rank
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
                owned += placement.page_rows(rows, page)
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
        other = self.resized(self.logical_capacity, self.logical_capacity)
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
    arena = KVCache(sum(counts), 1, head_dim, device, pair=pair)
    result, offset = [], 0
    for layer, count in enumerate(counts):
        result.append(OwnedKVCache(rows, 1, head_dim, device, pair=pair, placement=placement,
                                   slot=slot, layer=layer, rank=rank, arena=arena, arena_offset=offset))
        offset += count
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


def prepare(projection, state, layer, weights, buffers, start, stop, *, mtp):
    """Prepare one bounded frame; only owners publish persistent rows."""
    c, a = weights.cfg, layer.attn
    ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
    cache = state.mtp_kc if mtp else state.kc[ai]
    pos = state.mtp_len if mtp else state.pos
    rows = stop - start
    p = projection[start:stop]
    positions = torch.arange(rows, device=p.device) + pos
    at = c.heads * 2 * c.head_dim
    query = ops._guard(p[:, :at].reshape(rows, c.heads, 2, c.head_dim)[:, :, 0], state.kv_status, 1, layer.index)
    key = ops._guard(p[:, at:at + c.head_dim].reshape(rows, 1, c.head_dim), state.kv_status, 2, layer.index)
    at += c.head_dim
    value = ops._guard(p[:, at:at + c.head_dim].reshape(rows, 1, c.head_dim), state.kv_status, 4, layer.index)
    at += c.head_dim
    iq = ops._guard(p[:, at:at + c.index_heads * c.index_dim].reshape(rows, c.index_heads, c.index_dim),
                    state.kv_status, 16, layer.index)
    at += c.index_heads * c.index_dim
    ik = ops._guard(p[:, at:at + c.index_dim], state.kv_status, 16, layer.index)

    def normalize(value, weight):
        if buffers.prefill:
            return ops.rms(value, weight, c.eps)
        return torch.cat([ops.rms(value[row:row + 1], weight, c.eps) for row in range(rows)], 0)

    rope = state.image_positions
    options = dict(rope=rope, delta=state.rope_delta_dev if rope is not None or state.rope_delta else None,
                   length=0 if rope is None else rope.shape[0], sections=c.mrope_section,
                   scale=c.rope_attention_factor)
    query = ops.rotary(normalize(query, a.q_scale), weights.inv_freq, positions, **options)
    key = ops.rotary(normalize(key, a.k_scale), weights.inv_freq, positions, **options)
    query = ops._guard(query, state.kv_status, 1, layer.index)
    key = ops._guard(key, state.kv_status, 2, layer.index)
    if cache.key_format.codec:
        ops._rotor_rotate(query, buffers.q[start:stop], codec=cache.key_format.codec,
                          status=state.kv_status, layer=layer.index)
    else:
        buffers.q[start:stop].copy_(ops.h32(query).half() if cache.key_format.quantized else query)
    buffers.iq[start:stop].copy_(ops._guard(
        ops.rotary(normalize(iq, a.iq_scale), weights.inv_freq, positions, **options), state.kv_status, 16, layer.index))
    key_codes, key_scales = _encode(key, cache.key_format, state.kv_status, layer.index, error=2)
    value_codes, value_scales = _encode(value, cache.value_format, state.kv_status, layer.index, error=4)
    cache.append(pos, key_codes, value_codes, key_scales, value_scales)
    if state.history_rank == state.history_placement.indexer_owner(state.history_slot, ai,
                                                                  include_mtp=hasattr(state, "mtp_kc")):
        raw = state.mtp_ikc if mtp else state.ikc[ai]
        raw[pos:pos + rows].copy_(ik.half())
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

    def __init__(self, weights, rows, scratch):
        c, device = weights.cfg, weights.device
        world, group = 8, 4
        self.rows, self.heads = rows, c.heads
        self.queries = torch.empty((world, rows, c.heads, c.head_dim), dtype=torch.float16, device=device)
        self.selection = torch.zeros((rows, scratch.idw + 2), dtype=torch.int32, device=device)
        self.selections = torch.empty((world, rows, scratch.idw + 2), dtype=torch.int32, device=device)
        self.partial = torch.empty((rows, group * c.heads, c.head_dim + 2), dtype=torch.float32, device=device)
        self.partials = torch.empty((world, rows, group * c.heads, c.head_dim + 2), dtype=torch.float32, device=device)
        self.position = torch.zeros((1,), dtype=torch.int32, device=device)

    @property
    def nbytes(self):
        return sum(value.nbytes for value in vars(self).values() if isinstance(value, torch.Tensor))


def attention_layer(layer, weights, segments, buffers, total_rows, *, mtp=False):
    """Execute owner history with four ordered batched collectives per chunk.

    Per-row causal selection stays canonical. The two KV-head groups produce
    disjoint attention partials while all eight ranks retain their TP work.
    Packed histories are never gathered or expanded in full.
    """
    c, rank, comm = weights.cfg, weights.meta["rank"], weights.comm
    if comm is None or comm.world != 8 or comm.rank != rank:
        raise ValueError("unique history requires its live TP8 communicator")
    owner = buffers.history_owner
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
    for state, first, last in segments:
        prepare(buffers.pa, state, layer, weights, buffers, first, last, mtp=mtp)
        ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
        if rank == state.history_placement.indexer_owner(state.history_slot, ai,
                                                         include_mtp=hasattr(state, "mtp_kc")):
            raw, pooled = ((state.mtp_ikc, state.mtp_pooled) if mtp else (state.ikc[ai], state.pooled[ai]))
            pos = state.mtp_pos if mtp else state.pos_dev
            ops.qsa_pool(raw, pooled, pos, layer.attn.ik_scale, weights.inv_freq, c.eps,
                         buffers.attn, last - first, rope=state.image_positions,
                         delta=state.rope_delta_dev if state.image_positions is not None or state.rope_delta else None,
                         length=0 if state.image_positions is None else state.image_positions.shape[0],
                         sections=c.mrope_section, rope_scale=c.rope_attention_factor)
    for base in range(0, total_rows, owner.rows):
        count = min(owner.rows, total_rows - base)
        query_all = owner.queries.reshape(-1)[:8 * count * c.heads * c.head_dim].view(8, count, c.heads, c.head_dim)
        comm.all_gather(buffers.q[base:base + count], query_all)
        selection = owner.selection[:count]
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
        selections = owner.selections.reshape(-1)[:8 * count * selection.shape[1]].view(8, count, selection.shape[1])
        comm.all_gather(selection, selections)
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


def copy_prefix(destination, source, target_rows, mtp_rows):
    """Relocate a reusable prefix through one bounded page, preserving codes.

    All TP ranks call this operation in the same order. It is outside graph
    capture and the coordinator publishes the destination only after its normal
    stream completion/state restore. No whole-history conversion is performed.
    """
    placement, comm = destination.history_placement, destination.history_comm
    if comm is None or not hasattr(comm, "broadcast"):
        raise ValueError("unique prefix copies require the TP8 broadcast communicator")
    layers = [(layer, destination.kc[layer], source.kc[layer], target_rows,
               destination.ikc[layer], source.ikc[layer], destination.pooled[layer], source.pooled[layer])
              for layer in range(len(destination.kc))]
    if mtp_rows:
        layers.append((placement.target_layers, destination.mtp_kc, source.mtp_kc, mtp_rows,
                       destination.mtp_ikc, source.mtp_ikc, destination.mtp_pooled, source.mtp_pooled))
    for layer, dst, src, rows, draw, sraw, dpool, spool in layers:
        frame = KVCache(placement.page_tokens, 1, dst.head_dim, dst.k.device, pair=dst.pair)
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
        source_owner = placement.indexer_owner(source.history_slot, layer,
                                               include_mtp=hasattr(source, "mtp_kc"))
        target_owner = placement.indexer_owner(destination.history_slot, layer,
                                               include_mtp=hasattr(destination, "mtp_kc"))
        # Raw incomplete groups must be retained; pooled rows only cover fully
        # formed groups. Their positional/rotation basis is unchanged.
        frame_index = torch.empty((placement.page_tokens, destination.index_dim), dtype=draw.dtype, device=draw.device)
        for original, copied, length in ((sraw, draw, rows), (spool, dpool, rows // destination.ratio)):
            for start in range(0, length, placement.page_tokens):
                count = min(placement.page_tokens, length - start)
                view = frame_index[:count]
                if destination.history_rank == source_owner:
                    view.copy_(original[start:start + count])
                comm.broadcast(view, source_owner)
                if destination.history_rank == target_owner:
                    copied[start:start + count].copy_(view)


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
        if sorted(old_local) != list(range(4)) or sorted(new_local) != list(range(4)):
            raise RuntimeError("history bundle is not a balanced quartet permutation")
        count = placement.page_rows(state.capacity, page)
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
            _fence(state)
            comm.barrier()  # Publish the ownership generation only after all device tables agree.
            state.history_generation += 1
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

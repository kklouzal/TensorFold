"""DeepSeek-V4.1-Flash forward on one SM70 rank: the 1Cat-vLLM DS41 forward with the friend's multi-request
packed attention (overlay patches 85/87), on TensorFold-owned KV pages, slots and collectives, in the TP8 x (ETP4 x EP2) layout of the Flash Next and Gemma 4 lanes.

Order and arithmetic follow 1Cat-vLLM 4d57473d ``vllm/models/deepseek_v41/model.py`` (``_forward_fused``:
VLLM_V41_HC_FUSED=1, VLLM_V41_ROUTER_FUSED=1) with patch 87's ``attention_forward`` (packed FP4 KV rows on the
GPU, per-slot window rings, row-wise decode indexer, single-request prefill indexer). Kernels are the vendored
1Cat/overlay Triton kernels (``.kernels``), the 1Cat SM70 GGUF expert ops (``tensorfold.cuda.sm70.onecat_ds41``) and,
below the grouped-prefill threshold, the friend's expert-major decode experts (``ds41_decode_moe``, patch 240 EM3).

TensorFold changes, recorded: TP8 attention/dense (8 query heads, 1 output group, 4 indexer heads per rank)
instead of TP4, and experts ETP4 x EP2 (rank r holds experts [192 (r // 4), +192) at TP slice r % 4) instead of
PP2 x TP4; a layer's routed and shared partials form one 8-rank sum. Every cross-rank sum is accumulated in FP32
and rounded once (upstream sums FP16 partials of 4 ranks); the decode indexer's per-rank head sums stay FP32.
Q8_0 / Q5_K dense projections stay quantized as 1Cat dmv13 planes (``tensorfold.cuda.sm70.ds41_dense``: dmv13 /
patch 112 for <= 32 rows, bounded restore + cuBLAS above); the embedding is vocabulary-sharded (one FP16 sum). The Engram rows come from mapped host memory (``engram.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .collectives import Collectives
from .weights import EP, Geometry, Weights

DECODE_MAX = 8                 # patch 85: requests with more tokens take the single-request prefill indexer
BLOCK = 64                     # KV page: tokens per page (the friend's production --block-size 64)
# Fewest step tokens for the fused sparse attention (ds41_sparse_attention), from single-GPU timings at 180 W
# (attn-1 eager, attn-2 inside CUDA graphs): compressed layers keep patch 87's gather + bmm for graph-replayed decode
# buckets (<= 32 rows: 19-140 us vs the fused 43-147 us split / 108-168 us unsplit) and take the fused kernel above
# (eager 8192-row chunks 36 ms -> 8 ms; 48 graph rows 224 -> 199 us); window-only layers take it from 2 rows.
FUSED_MIN_COMPRESSED = 33
FUSED_MIN_WINDOW = 2
# Fewest step tokens for sequence-parallel HC (Shard): prompt chunks from here keep the FP32 HC residual, its
# transitions, the Engram gate and routing on 1/8 of the rows per rank; decode rows and graphs never take it.
SP_MIN = 1024


@dataclass(frozen=True)
class Shard:
    """Megatron-style sequence parallelism of one forward's rows: rank r owns rows [first, first + rows) with
    ``first`` = r * size, size = ceil(tokens / 8) (the last ranks' shards may be shorter; every rank has >= 1 row
    above 56 tokens). Cross-rank sums of row partials are FP32 reduce-scatters (rounded once to FP16 per shard,
    as the FP32 all-reduce rounds), followed by FP16 all-gathers of the normalized next-sublayer inputs."""

    tokens: int
    size: int
    first: int
    rows: int

    @classmethod
    def of(cls, tokens: int, rank: int, world: int):
        size = -(-tokens // world)
        first = rank * size
        return cls(tokens, size, first, max(0, min(tokens, first + size) - first))

    def take(self, x):
        return x[self.first:self.first + self.rows]

    def padded(self, partial):
        """[world * size, dims] FP32 copy of [tokens, dims] partials, zero rows past ``tokens``."""
        out = torch.empty(self.size * 8, partial.shape[-1], dtype=torch.float32, device=partial.device)
        out[:self.tokens].copy_(partial)
        out[self.tokens:].zero_()
        return out


# --------------------------------------------------------------------------------------------- batch view
@dataclass
class Step:
    """One forward's batch: requests in row order, each a contiguous run of tokens (patch 85's MuStep)."""

    ids: torch.Tensor            # [T] int64
    positions: torch.Tensor      # [T] int64
    valid: torch.Tensor          # [T] bool
    token_req: torch.Tensor      # [T] int32 request row of each token (-1 padding)
    qsl: torch.Tensor            # [R + 1] int32 query start offsets
    qsl_cpu: list
    num_reqs: int
    req_slot: torch.Tensor       # [rows] int32 slot of each request row (scratch for padding rows)
    block_table: torch.Tensor    # [R, pages] int32 page ids of each request row
    prefill: list                # [(r, s0, s1, last position)]: requests with > DECODE_MAX tokens (host values)
    decode_index: torch.Tensor | None   # token rows of the other requests (None: all rows)
    # Image input (vision.py): rows of image spans (ids IMAGE_TOKEN_ID), their mask and their FP16 features; None
    # in every text step, whose forward is then exactly the text model's
    image: torch.Tensor | None = None          # [T] bool
    image_rows: torch.Tensor | None = None     # [V] int64, ascending
    features: torch.Tensor | None = None       # [V, 5120] FP16
    single_token_slots: bool = False           # decode graph: one row per live slot, scratch rows never commit


def window_ring_bytes(rotorquant: bool) -> int:
    from .kernels.kv_pack import C_DIM, ROTOR_ROW
    from .kernels.mu_packed import RING

    return RING * (ROTOR_ROW if rotorquant else C_DIM * 2)


class Pool:
    """Packed latent/index pages and one FP16 or RotorQuant window ring per layer and slot."""

    def __init__(self, geometry: Geometry, pages: int, slots: int, device, *, layout=None):
        from .kernels.kv_pack import C_DIM, ROTOR_ROW
        from .kernels.mu_packed import PackedLayout, RING
        if layout is None:
            sources = {layer: geometry.compress_ratios[layer] for layer in geometry.kv_sources}
            layout = PackedLayout(sources, BLOCK)
        self.layout = layout
        self.pages = pages
        self.cache = torch.zeros(pages, self.layout.page_bytes, dtype=torch.uint8, device=device)
        width, dtype = (ROTOR_ROW, torch.uint8) if layout.rotorquant else (C_DIM, torch.float16)
        self.rings = [torch.zeros(slots, RING, width, dtype=dtype, device=device)
                      for _ in range(geometry.layers)]
        self.free = list(range(pages - 1, -1, -1))    # lowest page first (pop from the end)
        self.owned: dict[int, list[int]] = {}

    def token_bytes(self) -> int:
        return self.layout.bytes_per_token

    def attach(self, slot: int, tokens: int) -> list[int]:
        need = -(-tokens // BLOCK)
        have = self.owned.setdefault(slot, [])
        if need > len(have) + len(self.free):
            raise MemoryError(f"KV pool exhausted: {need - len(have)} more pages needed, {len(self.free)} free")
        while len(have) < need:
            have.append(self.free.pop())
        return have

    def detach(self, slot: int) -> None:
        for page in reversed(self.owned.pop(slot, [])):
            self.free.append(page)
        self.free.sort(reverse=True)

    def reset_ring(self, slot: int) -> None:
        for ring in self.rings:
            ring[slot].zero_()


# ------------------------------------------------------------------------------------------------ the model
class Target:
    def __init__(self, w: Weights, comm: Collectives, *, slots: int, pages: int, max_tokens: int, context: int,
                 engram, device, lattice, rope, layout, bias_vl=None):
        from .kernels import quant_rope, mu_state
        self.w, self.c, self.g = w, comm, w.geometry
        self.device = device
        self.slots = slots                    # the last slot is scratch (padding rows)
        self.context = context
        # positions the decode indexer's score rows cover (its FP32 cross-rank sum and top-512 read width //
        # ratio): the admitted context, or a smaller startup graph class (graphs.py) every row of the batch fits
        self.index_context = context
        self.max_tokens = max_tokens
        self.pool = Pool(self.g, pages, slots, device, layout=layout)
        self.rotor_cache = None
        if layout.rotorquant:
            from .rotor_cache import RotorCache

            self.rotor_cache = RotorCache(device)
        self.engram = engram
        self.window_rope = quant_rope.rope_table(rope=rope, device=device)
        self.compressed_rope = quant_rope.rope_table(compressed=True, rope=rope, device=device)
        g = self.g
        self.hash_state = mu_state.MuEngramHashState(
            w.engram_token_map, w.engram_primes, w.engram_offsets, w.engram_multipliers, slots=slots - 1,
            pad_token_id=g.engram_pad, max_tokens=max_tokens, max_context=max(16384, context), trap=True)
        self.c2 = {}
        for layer in w.layers:
            if layer.owns_kv and layer.ratio == 2:
                self.c2[layer.index] = mu_state.MuC2State(layer.comp_norm, slots=slots - 1, trap=True)
        self.local = range(w.group * (g.experts // EP), (w.group + 1) * (g.experts // EP))
        self.engram_rows = torch.empty(max_tokens, 24 * 256, dtype=torch.float16, device=device)
        from tensorfold.cuda.sm70 import ds41_decode_moe
        self.decode_moe = ds41_decode_moe.Workspace(device)   # routed experts below the grouped-prefill threshold
        # Prefill routed experts (>= MIN_TOKENS rows): the friend's TurboMind lattice path (ds41_lattice_moe); its
        # canonical-bank workspace is the caller's (allocated before the KV pool is sized).
        self.lattice = lattice
        self.bias_vl = bias_vl             # [layers, experts] FP32: image rows' router bias (--vision), else None
        # Engram lookup telemetry for the deferred hot-row VRAM cache candidate (off unless the variable names a
        # file; rank 0 only, since every rank looks up the same rows): appends each forward's valid hashed row ids
        # (int64 [tokens, 2 tables, 24 heads], table-local) for offline hit-rate / churn analysis.
        import os
        trace = os.environ.get("TENSORFOLD_DS41_ENGRAM_TRACE") if w.rank == 0 else None
        self.engram_trace = open(trace, "ab") if trace else None

    # ---- slot lifecycle (every rank, same order)
    def reset_slot(self, slot: int) -> None:
        self.pool.reset_ring(slot)
        self.hash_state.reset_slot(slot)
        for state in self.c2.values():
            state.reset_slot(slot)

    # ---- small pieces
    def _norm(self, x, weight):
        from .kernels.norm import rms_norm
        return rms_norm(x, weight, self.g.eps)

    @staticmethod
    def _linear(x, weight):
        from tensorfold.cuda.sm70.ds41_dense import Group
        if isinstance(weight, Group):
            return weight(x)[0]
        return torch.matmul(x, weight.t())

    def embed(self, ids):
        """Vocabulary-shard lookup: this rank decodes the rows of the ids it owns (others +0), one exact FP16 sum
        over the 8 ranks (exactly one rank contributes each row)."""
        from tensorfold.cuda.sm70 import ds41_dense as dense
        rows = torch.empty(ids.shape[0], self.g.hidden, dtype=torch.float16, device=ids.device)
        dense.q5k_rows(self.w.embed, ids, self.w.embed_first, rows)
        return self.c.exact_sum_half(rows)

    # ---- attention (patch 87 attention_forward)
    def attention(self, ly, x, step: Step, table, shared, shard=None):
        from .kernels import quant_rope as qr_ops
        from .kernels import mu_packed as pk
        from .kernels.rope_qdq import rope_qdq
        tokens = x.shape[0]
        positions = step.positions
        q_low, kv_low = ly.qa_kv(x)
        qr = self._norm(q_low, ly.q_norm)
        query = qr_ops.apply_rope(self._linear(qr, ly.wq_b).view(tokens, -1, 512), positions, table)
        pool = self.pool
        ring = pool.rings[ly.index]
        rotor = self.rotor_cache
        cold = shared.get("rotor_cold_legacy")
        valid = step.valid
        # A one-row request reads the current value from window; the prior
        # RING-1 positions never alias the ring row being written here.
        decode_ring = (ring, step.req_slot, step.valid) if rotor is None and step.single_token_slots else None
        if rotor is not None:
            valid = shared["rotor_valid"]
            if cold is None:
                query = rotor.rotate(query, valid, ly.index)
            normalized_window = self._norm(kv_low, ly.kv_norm)
            physical_window = qr_ops.apply_rope(normalized_window, positions, table)
            window = rotor.pack(physical_window, valid, ly.index)
        elif decode_ring is not None and 1 <= tokens <= 64:
            window = rope_qdq(kv_low, positions, table, decode_ring=decode_ring,
                              decode_norm=(ly.kv_norm, self.g.eps))
        else:
            window = rope_qdq(self._norm(kv_low, ly.kv_norm), positions, table, decode_ring=decode_ring)
        extra = {}
        if ly.ratio:
            if ly.owns_kv:
                latent = self._compress(ly, x, step)
                key = self._norm(self._linear(latent, ly.idx_wk), ly.idx_k_norm)
                key = qr_ops.apply_rope(key, (positions - (ly.ratio - 1)).clamp_min(0), table)
                pk.insert_rows(pool.cache, key, positions, valid, step.token_req, step.block_table,
                               layout=pool.layout, layer=ly.index, compressed=False)
                compact = qr_ops.apply_rope(latent, (positions - ly.ratio + 1).clamp_min(0), table)
                if rotor is not None:
                    live = valid if ly.ratio == 1 else valid & ((positions + 1) % ly.ratio == 0)
                    if cold is not None:
                        # Consumers read only the latest source. Keep its scratch pages across a layer pause.
                        scratch = pk.PackedLayout({ly.index: ly.ratio}, pool.layout.block_size)
                        pages = torch.empty(cold["table"].numel(), scratch.page_bytes, device=x.device,
                                            dtype=torch.uint8)
                        pk.insert_rows(pages, compact, positions, valid, step.token_req, cold["table"],
                                       layout=scratch, layer=ly.index, compressed=True)
                        cold["source"] = (ly.index, pages, scratch)
                    compact = rotor.pack(compact, live, ly.index)
                pk.insert_rows(pool.cache, compact, positions, valid, step.token_req, step.block_table,
                               layout=pool.layout, layer=ly.index, compressed=True)
            if ly.has_indexer:
                shared["topk"] = self._select(ly, x, qr, step, table)
            if cold is None:
                extra = dict(cache=pool.cache, block_table=step.block_table, layout=pool.layout, layer=ly.source,
                             topk=shared["topk"])
            else:
                source, pages, scratch = cold["source"]
                if source != ly.source:
                    raise RuntimeError("DeepSeek cold prefill lost its compact source")
                extra = dict(cache=pages, block_table=cold["table"], layout=scratch, layer=source,
                             topk=shared["topk"])
        if cold is None:
            output = self._attend(query, step, ring, window, ly.sink, **extra)
        else:
            # Retain the earlier FP8 window QDQ and E2M1 compact attention operand boundaries.
            window_operand = rope_qdq(normalized_window, positions, table)
            torch._assert_async(torch.isfinite(query).all(), f"DeepSeek layer {ly.index}: nonfinite cold attention query")
            output = self._attend(query, step, cold["ring"], window_operand, ly.sink, **extra)
            torch._assert_async(torch.isfinite(output).all(), f"DeepSeek layer {ly.index}: nonfinite cold attention output")
        if decode_ring is None:
            pk.insert_ring(ring, window, positions, valid, step.token_req, step.qsl, step.req_slot)
        if rotor is not None and cold is None:
            output = rotor.rotate(output, valid, ly.index, inverse=True)
        output = rope_qdq(output, positions, table, inverse=True)   # TP8: one output group per rank; qdq(rope^-1)
        projected = self._linear(output.reshape(tokens, -1), ly.wo_a)
        partial = self._linear(projected, ly.wo_b)
        if shard is not None:
            return self.c.scatter_sum(shard.padded(partial), shard)
        return self.c.sum(partial)

    def _compress(self, ly, x, step):
        if ly.ratio == 1:
            return self._norm(self._linear(x, ly.comp_wkv), ly.comp_norm)
        xf = x.float()
        values, scores = self._linear(xf, ly.comp_wkv), self._linear(xf, ly.comp_wgate)
        state = self.c2[ly.index]
        latent, _ = state(values, scores, step.positions, step.token_req, step.qsl, step.req_slot,
                          commit_single=step.single_token_slots)
        if not step.single_token_slots:
            state.commit_all(values, scores, step.positions, step.qsl, step.req_slot, step.num_reqs, step.token_req)
        return latent

    def _select(self, ly, x, qr, step, table):
        """top-512 compressed positions per token (patch 87 ``_select``: Full layers, segmented top-k)."""
        from .kernels import index_shard
        from .kernels import mu_packed as pk
        from .kernels.rope_qdq import rope_qdq
        from .weights import WORLD
        tokens = x.shape[0]
        query = self._linear(qr, ly.idx_wq_b).view(tokens, -1, 128)
        query = rope_qdq(query, step.positions, table, fp4=True, group=32)
        weights = self._linear(x, ly.idx_proj) * (1.0 / 64.0)
        width = self.index_context // ly.ratio
        if (tokens > 1 and not step.prefill and step.decode_index is None
                and index_shard.eligible(query.shape[1], WORLD, width)):
            # decode rows: every rank scores all 32 heads over its 1/8 of the key blocks (K4, index_shard; one row
            # keeps the head-sharded scores: 98-118 vs 104-123 us per rank, dec-k4r / dec-k4b)
            return index_shard.select_decode(query, weights, step.positions, step.valid, step.token_req,
                                             self.pool.cache, step.block_table, layout=self.pool.layout,
                                             source=ly.source, ratio=ly.ratio, width=width, rank=self.w.rank,
                                             comm=self.c.world)
        return pk.select_positions(query, weights, step.positions, step, self.pool.cache, step.block_table,
                                   layout=self.pool.layout, source=ly.source, ratio=ly.ratio,
                                   width=self.index_context // ly.ratio, all_reduce=self._sum_half,
                                   score_dtype=torch.float32, row_split=(self.w.rank, 8, self.c.world.all_gather))

    def _sum_half(self, scores):
        """The indexer's TP sum: FP32 (decode) or FP16 (prefill tiles) per-rank scores, one FP16 rounding."""
        return self.c.sum(scores, torch.float16)

    def _attend(self, query, step, ring, window, sink, *, cache=None, block_table=None, layout=None, layer=None,
                topk=None):
        """Sparse attention [T, 8, 512]: the fused kernel (patch 181 at TP8, ``ds41_sparse_attention``: keys
        streamed from the ring / pages, no gathered key buffer) from FUSED_MIN_* step tokens, else patch 87's
        gather + bmm passes (decode buckets of compressed layers); both read the same keys and FP16 probabilities."""
        if query.shape[0] >= (FUSED_MIN_COMPRESSED if cache is not None else FUSED_MIN_WINDOW):
            from tensorfold.cuda.sm70 import ds41_sparse_attention as fused
            return fused.attend(query.contiguous(), step.positions, step, step.req_slot, ring, window.contiguous(),
                                sink, cache=cache, block_table=block_table, layout=layout, source=layer, topk=topk)
        from .kernels import mu_packed as pk
        return pk.attend(query, step.positions, step, step.req_slot, ring, window, sink, cache=cache,
                         block_table=block_table, layout=layout, source=layer, topk=topk)

    # ---- MoE
    def moe(self, ly, x, image=None, shard=None, x_rows=None):
        """Experts for ``x`` [T, H]; ``image``: bool [T] rows that route with the vision bias; with ``shard``,
        routing runs on this rank's rows ``x_rows`` (gathered after) and the result is this rank's FP16 shard rows."""
        from tensorfold.cuda.sm70 import ds41_lattice_moe as lattice_moe
        from tensorfold.cuda.sm70 import onecat_ds41 as native
        from .kernels.band import add_halves, route_local
        from .kernels.mhc import fp32_projection
        g = self.g
        logits = fp32_projection(x if shard is None else x_rows, ly.gate)
        mask = image if image is None or shard is None else shard.take(image)
        if mask is None and shard is None:
            # the router epilogue and this rank's EP2 mapping in one launch (local id, weight 0 for the other group)
            local, weights = route_local(logits, ly.gate_bias, g.top_k, g.routed_scale, self.local.start,
                                         len(self.local))
        else:
            if mask is None:
                from .kernels.router import route_topk

                probabilities, ids = route_topk(logits, ly.gate_bias, g.top_k, g.routed_scale)
            else:       # image rows route with the vision bias (upstream Gate.bias_vl; weights stay unbiased)
                from .vision import route_with_vision

                probabilities, ids = route_with_vision(logits, ly.gate_bias, self.bias_vl[ly.index], mask,
                                                       g.top_k, g.routed_scale)
            ids = ids.to(torch.int32)
            if shard is not None:
                # The row owner can belong to either EP group. Share global routes before applying the
                # receiving rank's local expert mapping, which discards routes owned by the other group.
                ids = self.c.gather_rows(ids, shard)
                probabilities = self.c.gather_rows(probabilities, shard)
            first = self.local.start
            local = ids - first
            mine = (local >= 0) & (local < len(self.local))
            local = torch.where(mine, local, torch.zeros_like(local)).contiguous()
            weights = torch.where(mine, probabilities.float(),
                                  torch.zeros_like(probabilities, dtype=torch.float32)).contiguous()
        if x.shape[0] >= lattice_moe.MIN_TOKENS:   # grouped lattice GEMMs (no host sync)
            out = lattice_moe.experts(x.contiguous(), local, weights, ly.exp_gate, ly.exp_up, ly.exp_down,
                                      ly.gate_up_type, ly.down_type, ly.down_left, self.lattice)
        else:  # expert-major decode experts (the friend's EM3), bit-equal to the 6-row op, graph-capturable
            out = self.decode_moe.experts(x.contiguous(), local, weights, ly.exp_gate, ly.exp_up, ly.exp_down,
                                          ly.gate_up_type, ly.down_type, ly.down_left)
        hidden = native.shared_swiglu(*ly.sh_gu(x))
        if shard is not None:
            total = shard.padded(out)
            total[:shard.tokens] += self._linear(hidden, ly.sh_down)
            return self.c.scatter_sum(total, shard)
        return self.c.sum(add_halves(out, self._linear(hidden, ly.sh_down)), torch.float16)

    # ---- Engram
    def engram_layer(self, ly, hidden, hashes, valid, shard=None):
        from .kernels.engram import apply_engram_gate
        tokens = hashes.shape[0]        # every row of the step (with a shard, hidden holds only this rank's)
        channel = self.g.engram_layers.index(ly.index)
        rows = self.engram.lookup(hashes, channel, self.engram_rows[:tokens])
        projected = self.c.gather(self._linear(rows, ly.engram_wkv))
        if shard is not None:       # every rank projects all rows (column-sharded weight); gates its own rows
            projected, valid = shard.take(projected), shard.take(valid)
        return apply_engram_gate(hidden, projected, ly.engram_q, ly.engram_k, valid, inplace=True)

    # ---- the whole forward (begin / layers / end: a prompt chunk may pause between layers while decode rounds run)
    def forward(self, step: Step, logits_rows: torch.Tensor | None):
        """Run every layer for ``step``; return this rank's vocabulary-shard logits [rows, vocab / 8] (FP32) of
        ``logits_rows`` (None: no logits)."""
        state = self.begin(step)
        self.layers(state, self.units)
        return self.end(state, logits_rows)

    def begin(self, step: Step) -> dict:
        """Hash and embed ``step``'s tokens; the returned state runs through ``layers`` then ``end``. A paused
        state holds only its own tensors: per-slot KV, rings, Engram history and C2 partials are written in layer
        order, so other forwards may run between its layers (they touch other slots)."""
        g = self.g
        tokens = step.ids.shape[0]
        # Image-span rows are dead Engram n-gram members (upstream: valid & ~image into the hash state, its commit
        # and the Engram gate); KV and attention keep every row.
        valid = step.valid if step.image is None else step.valid & ~step.image
        if step.single_token_slots:
            hashes = self.hash_state(step.ids, step.positions, valid, step.token_req, step.qsl, step.req_slot,
                                     commit_single=True)
        else:
            hashes = self.hash_state(step.ids, step.positions, valid, step.token_req, step.qsl, step.req_slot)
            self.hash_state.commit_all(step.ids, step.positions, valid, step.qsl, step.req_slot, step.num_reqs,
                                       step.token_req)
        if self.engram_trace is not None:
            self.engram_trace.write(hashes[valid].cpu().numpy().tobytes())
        hidden = self.embed(step.ids)
        if step.image is not None:        # the tower's FP16 span rows replace the image token's embedding
            if self.bias_vl is None:
                raise RuntimeError("an image step on an engine without the vision router bias")
            hidden.index_copy_(0, step.image_rows, step.features)
        shard = Shard.of(tokens, self.w.rank, 8) if tokens >= SP_MIN else None
        if shard is not None:       # this rank's rows of the HC residual
            hidden = shard.take(hidden)
        # FP16 cannot carry the HC residual range of valid special tokens: the stream stays FP32 (upstream)
        hidden = hidden.float()[:, None].expand(-1, g.hc, -1).contiguous()
        pre = torch.zeros(hidden.shape[0], g.hc, device=hidden.device, dtype=torch.float32)
        pre[:, 0] = 1
        shared = {"topk": None}
        if self.rotor_cache is not None:
            from .rotor_cache import live_rows

            shared["rotor_valid"] = live_rows(step, self.slots - 1)
            if (9 <= tokens < 1024 and step.image is None and step.num_reqs == 1
                    and step.prefill == [(0, 0, tokens, tokens - 1)]):
                from .kernels.kv_pack import C_DIM
                from .kernels.mu_packed import RING

                block_size = self.pool.layout.block_size
                blocks = -(-tokens // block_size)
                shared["rotor_cold_legacy"] = {
                    "table": torch.arange(blocks, device=hidden.device, dtype=torch.int32).view(1, blocks),
                    # Starts at position zero, so historical ring operands are always masked.
                    "ring": torch.empty((1, RING, C_DIM), device=hidden.device, dtype=torch.float16),
                }
        return {"step": step, "hashes": hashes, "hidden": hidden, "pre": pre, "shared": shared,
                "mix": None, "half": None, "next": 0, "result": None, "valid": valid,
                "image": step.image, "shard": shard}

    @property
    def units(self) -> int:
        """Pause points of a forward: two units per layer (attention, then experts)."""
        return 2 * len(self.w.layers)

    def layers(self, st: dict, stop: int, marks: list | None = None) -> None:
        """Advance ``st`` through units [st["next"], stop): unit 2i is layer i's attention (with its Engram and
        HC transition), unit 2i+1 its experts (with the transition into layer i+1). A pause at either changes no
        kernel or row, so a paused prompt forward computes exactly what an unpaused one does. ``marks``: append
        (first unit, CUDA events at the start and after each unit) for the caller's per-unit cost estimate."""
        from .kernels.mhc import collapse_hc, expand_hc, hc_mixes
        from .kernels.mhc_fused import hc_transition
        w = self.w
        layers = w.layers
        step, hashes, shared, shard = st["step"], st["hashes"], st["shared"], st["shard"]
        # with a shard, HC state (hidden, pre, mixes) holds this rank's rows; sublayer inputs are gathered whole
        full = (lambda rows: rows) if shard is None else (lambda rows: self.c.gather_rows(rows, shard))
        hidden, pre, state, half = st["hidden"], st["pre"], st["mix"], st["half"]
        events = None
        if marks is not None:
            events = [torch.cuda.Event(enable_timing=True)]
            events[0].record()
            marks.append((st["next"], events))
        for u in range(st["next"], stop):
            i, second = divmod(u, 2)
            ly = layers[i]
            if not second:
                if ly.engram_wkv is not None:
                    hidden = self.engram_layer(ly, hidden, hashes, st["valid"], shard)
                    state = None
                if state is None:
                    attn_mix = hc_mixes(hidden, ly.hc["attn_fn"], ly.hc["attn_scale"], ly.hc["attn_base"])
                    inputs = full(collapse_hc(hidden, pre, ly.attn_norm))
                else:
                    inputs, attn_mix = state
                table = self.compressed_rope if ly.ratio else self.window_rope
                y = self.attention(ly, inputs, step, table, shared, shard)
                rows, ffn_mix = hc_transition(hidden, y, attn_mix[1], attn_mix[2], attn_mix[0], ly.ffn_norm,
                                              ly.hc["ffn_fn"], ly.hc["ffn_scale"], ly.hc["ffn_base"])
                half = (full(rows), ffn_mix, rows)
            else:
                inputs, ffn_mix, rows = half
                half = None
                y = self.moe(ly, inputs, st["image"], shard, rows)
                pre = ffn_mix[0]
                following = layers[i + 1] if i + 1 < len(layers) else None
                if following is not None and following.engram_wkv is None:
                    inputs, mix = hc_transition(hidden, y, ffn_mix[1], ffn_mix[2], pre, following.attn_norm,
                                                following.hc["attn_fn"], following.hc["attn_scale"],
                                                following.hc["attn_base"])
                    state = (full(inputs), mix)
                elif following is not None:
                    hidden = expand_hc(y, hidden, ffn_mix[1], ffn_mix[2], inplace=True)
                    state = None
                else:
                    st["result"], _ = hc_transition(hidden, y, ffn_mix[1], ffn_mix[2], pre, w.norm)
            if events is not None:
                events.append(torch.cuda.Event(enable_timing=True))
                events[-1].record()
        st.update(hidden=hidden, pre=pre, mix=state, half=half, next=max(st["next"], stop))

    def end(self, st: dict, logits_rows: torch.Tensor | None):
        if st["next"] != self.units:
            raise RuntimeError("a DeepSeek-V4.1 forward ended before its last layer")
        if logits_rows is None:
            return None
        result = st["result"] if st["shard"] is None else self.c.gather_rows(st["result"], st["shard"])
        rows = result.index_select(0, logits_rows)
        return self._linear(rows, self.w.head).float()

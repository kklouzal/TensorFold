"""DeepSeek-V4.1-Flash forward on one SM70 rank: the 1Cat-vLLM DS41 forward with the friend's multi-request
packed attention (overlay patches 85/87), on TensorFold-owned KV pages, slots and collectives, in the TP8 x (ETP4 x EP2) layout of the Flash Next and Gemma 4 lanes.

Order and arithmetic follow 1Cat-vLLM 4d57473d ``vllm/models/deepseek_v41/model.py`` (``_forward_fused``:
VLLM_V41_HC_FUSED=1, VLLM_V41_ROUTER_FUSED=1) with patch 87's ``attention_forward`` (packed FP4 KV rows on the
GPU, per-slot window rings, row-wise decode indexer, single-request prefill indexer). Kernels are the vendored
1Cat/overlay Triton kernels (``.kernels``) and the 1Cat SM70 GGUF expert ops (``tensorfold.cuda.sm70.onecat_ds41``).

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

from .weights import EP, WORLD, Geometry, Weights

DECODE_MAX = 8                 # patch 85: requests with more tokens take the single-request prefill indexer
BLOCK = 64                     # KV page: tokens per page (the friend's production --block-size 64)
# Fewest step tokens for the fused sparse attention (ds41_sparse_attention), from single-GPU timings at 180 W
# (attn-1 eager, attn-2 inside CUDA graphs): compressed layers keep patch 87's gather + bmm for graph-replayed decode
# buckets (<= 32 rows: 19-140 us vs the fused 43-147 us split / 108-168 us unsplit) and take the fused kernel above
# (eager 8192-row chunks 36 ms -> 8 ms; 48 graph rows 224 -> 199 us); window-only layers take it from 2 rows.
FUSED_MIN_COMPRESSED = 33
FUSED_MIN_WINDOW = 2


# ------------------------------------------------------------------------------------------- collectives
class Collectives:
    """FP32 sums and gathers over the 8-rank communicator (TP8 attention/dense and the ETP4 x EP2 experts)."""

    def __init__(self, world):
        self.world = world

    def _all_reduce(self, tensor):
        """In-place NCCL sum of an FP32 or FP16 tensor."""
        import ctypes
        kind = {torch.float32: 7, torch.float16: 6}[tensor.dtype]
        comm = self.world
        fn = comm.lib.ncclAllReduce
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                       ctypes.c_void_p, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        stream = torch.cuda.current_stream().cuda_stream
        comm._check(fn(tensor.data_ptr(), tensor.data_ptr(), tensor.numel(), kind, 0, comm.comm, stream))

    def sum(self, partial: torch.Tensor, dtype=None) -> torch.Tensor:
        """8-rank sum of partials, accumulated in FP32 and rounded once to ``dtype`` (default: the partial's)."""
        total = partial.float().contiguous()
        self._all_reduce(total)
        return total.to(dtype or partial.dtype)

    def gather(self, part: torch.Tensor) -> torch.Tensor:
        """Concatenate the eight ranks' parts along the last dimension, rank order."""
        part = part.contiguous()
        recv = torch.empty((WORLD,) + tuple(part.shape), dtype=part.dtype, device=part.device)
        self.world.all_gather(part, recv)
        return torch.cat(list(recv.unbind(0)), dim=-1)


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
    prefill: list                # [(r, s0, s1)]: requests with > DECODE_MAX tokens
    decode_index: torch.Tensor | None   # token rows of the other requests (None: all rows)


class Pool:
    """Packed KV pages (patch 87 PackedLayout: per source, B/r compressed rows of 288 B then B/r index rows of
    68 B) and per-slot window rings [slots, 128, 512] FP16 for every layer. Identical on every rank."""

    def __init__(self, geometry: Geometry, pages: int, slots: int, device):
        from .kernels.mu_packed import PackedLayout, RING
        sources = {layer: geometry.compress_ratios[layer] for layer in geometry.kv_sources}
        self.layout = PackedLayout(sources, BLOCK)
        self.pages = pages
        self.cache = torch.zeros(pages, self.layout.page_bytes, dtype=torch.uint8, device=device)
        self.rings = [torch.zeros(slots, RING, 512, dtype=torch.float16, device=device)
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
                 engram, device):
        from .kernels import quant_rope, mu_state
        self.w, self.c, self.g = w, comm, w.geometry
        self.device = device
        self.slots = slots                    # the last slot is scratch (padding rows)
        self.context = context
        self.max_tokens = max_tokens
        self.pool = Pool(self.g, pages, slots, device)
        self.engram = engram
        self.window_rope = quant_rope.rope_table().to(device)
        self.compressed_rope = quant_rope.rope_table(compressed=True).to(device)
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
        from .kernels.norm import short_rms_norm
        if (x.dtype == torch.float16 and x.ndim == 2 and 1 <= x.shape[0] <= 6 and x.shape[1] in (128, 512, 1280, 5120)
                and x.is_contiguous()):
            return short_rms_norm(x, weight, self.g.eps)
        values = x.float()
        return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + self.g.eps) * weight).to(x.dtype)

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
        self.c._all_reduce(rows)
        return rows

    # ---- attention (patch 87 attention_forward)
    def attention(self, ly, x, step: Step, table, shared):
        from .kernels import quant_rope as qr_ops
        from .kernels import mu_packed as pk
        tokens = x.shape[0]
        positions = step.positions
        q_low, kv_low = ly.qa_kv(x)
        qr = self._norm(q_low, ly.q_norm)
        query = qr_ops.apply_rope(self._linear(qr, ly.wq_b).view(tokens, -1, 512), positions, table)
        window = qr_ops.attention_qdq(qr_ops.apply_rope(self._norm(kv_low, ly.kv_norm), positions, table))
        pool = self.pool
        extra = {}
        if ly.ratio:
            if ly.owns_kv:
                latent = self._compress(ly, x, step)
                key = self._norm(self._linear(latent, ly.idx_wk), ly.idx_k_norm)
                key = qr_ops.apply_rope(key, (positions - (ly.ratio - 1)).clamp_min(0), table)
                pk.insert_rows(pool.cache, key, positions, step.valid, step.token_req, step.block_table,
                               layout=pool.layout, layer=ly.index, compressed=False)
                compact = qr_ops.apply_rope(latent, (positions - ly.ratio + 1).clamp_min(0), table)
                pk.insert_rows(pool.cache, compact, positions, step.valid, step.token_req, step.block_table,
                               layout=pool.layout, layer=ly.index, compressed=True)
            if ly.has_indexer:
                shared["topk"] = self._select(ly, x, qr, step, table)
            extra = dict(cache=pool.cache, block_table=step.block_table, layout=pool.layout, layer=ly.source,
                         topk=shared["topk"])
        output = self._attend(query, step, pool.rings[ly.index], window, ly.sink, **extra)
        pk.insert_ring(pool.rings[ly.index], window, positions, step.valid, step.token_req, step.qsl, step.req_slot)
        output = qr_ops.apply_rope(output, positions, table, inverse=True)
        output = qr_ops.attention_qdq(output.reshape(tokens, 1, -1))      # TP8: one output group per rank
        projected = self._linear(output.reshape(tokens, -1), ly.wo_a)
        return self.c.sum(self._linear(projected, ly.wo_b))

    def _compress(self, ly, x, step):
        if ly.ratio == 1:
            return self._norm(self._linear(x, ly.comp_wkv), ly.comp_norm)
        xf = x.float()
        values, scores = self._linear(xf, ly.comp_wkv), self._linear(xf, ly.comp_wgate)
        state = self.c2[ly.index]
        latent, _ = state(values, scores, step.positions, step.token_req, step.qsl, step.req_slot)
        state.commit_all(values, scores, step.positions, step.qsl, step.req_slot, step.num_reqs, step.token_req)
        return latent

    def _select(self, ly, x, qr, step, table):
        """top-512 compressed positions per token (patch 87 ``_select``: Full layers, segmented top-k)."""
        from .kernels import quant_rope as qr_ops
        from .kernels import mu_packed as pk
        tokens = x.shape[0]
        query = self._linear(qr, ly.idx_wq_b).view(tokens, -1, 128)
        query = qr_ops.attention_qdq(qr_ops.apply_rope(query, step.positions, table), fp4=True, group=32)
        weights = self._linear(x, ly.idx_proj) * (1.0 / 64.0)
        return pk.select_positions(query, weights, step.positions, step, self.pool.cache, step.block_table,
                                   layout=self.pool.layout, source=ly.source, ratio=ly.ratio,
                                   width=self.context // ly.ratio, all_reduce=self.c.sum,
                                   score_dtype=torch.float32)

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
    def moe(self, ly, x):
        from tensorfold.cuda.sm70 import ds41_prefill_moe as prefill_moe, onecat_ds41 as native
        from .kernels.mhc import fp32_projection
        from .kernels.router import route_topk
        g = self.g
        logits = fp32_projection(x, ly.gate)
        probabilities, ids = route_topk(logits.float(), ly.gate_bias, g.top_k, g.routed_scale)
        first = self.local.start
        local = ids.to(torch.int32) - first
        mine = (local >= 0) & (local < len(self.local))
        local = torch.where(mine, local, torch.zeros_like(local)).contiguous()
        weights = torch.where(mine, probabilities.float(), torch.zeros_like(probabilities, dtype=torch.float32)).contiguous()
        if x.shape[0] >= prefill_moe.MIN_TOKENS:  # grouped prefill experts (one host sync; not captured)
            out = prefill_moe.experts(x.contiguous(), local, weights, ly.exp_gate, ly.exp_up, ly.exp_down,
                                      ly.gate_up_type, ly.down_type, ly.down_left)
        else:
            out = torch.empty_like(x)
            for at in range(0, x.shape[0], 6):
                end = min(at + 6, x.shape[0])
                out[at:end] = native.v41_experts(x[at:end].contiguous(), local[at:end], weights[at:end],
                                                 ly.exp_gate, ly.exp_up, ly.exp_down, ly.gate_up_type,
                                                 ly.down_type, ly.down_left)
        hidden = native.shared_swiglu(*ly.sh_gu(x))
        total = out.float() + self._linear(hidden, ly.sh_down).float()
        self.c._all_reduce(total)
        return total.half()

    # ---- Engram
    def engram_layer(self, ly, hidden, hashes, step):
        from .kernels.engram import apply_engram_gate
        tokens = hidden.shape[0]
        channel = self.g.engram_layers.index(ly.index)
        rows = self.engram.lookup(hashes, channel, self.engram_rows[:tokens])
        projected = self.c.gather(self._linear(rows, ly.engram_wkv))
        return apply_engram_gate(hidden, projected, ly.engram_q, ly.engram_k, step.valid, inplace=True)

    # ---- the whole forward (begin / layers / end: a prompt chunk may pause between layers while decode rounds run)
    def forward(self, step: Step, logits_rows: torch.Tensor | None):
        """Run every layer for ``step``; return this rank's vocabulary-shard logits [rows, vocab / 8] (FP32) of
        ``logits_rows`` (None: no logits)."""
        state = self.begin(step)
        self.layers(state, len(self.w.layers))
        return self.end(state, logits_rows)

    def begin(self, step: Step) -> dict:
        """Hash and embed ``step``'s tokens; the returned state runs through ``layers`` then ``end``. A paused
        state holds only its own tensors: per-slot KV, rings, Engram history and C2 partials are written in layer
        order, so other forwards may run between its layers (they touch other slots)."""
        g = self.g
        tokens = step.ids.shape[0]
        hashes = self.hash_state(step.ids, step.positions, step.valid, step.token_req, step.qsl, step.req_slot)
        self.hash_state.commit_all(step.ids, step.positions, step.valid, step.qsl, step.req_slot, step.num_reqs,
                                   step.token_req)
        if self.engram_trace is not None:
            self.engram_trace.write(hashes[step.valid].cpu().numpy().tobytes())
        hidden = self.embed(step.ids)
        # FP16 cannot carry the HC residual range of valid special tokens: the stream stays FP32 (upstream)
        hidden = hidden.float()[:, None].expand(-1, g.hc, -1).contiguous()
        pre = torch.zeros(tokens, g.hc, device=hidden.device, dtype=torch.float32)
        pre[:, 0] = 1
        return {"step": step, "hashes": hashes, "hidden": hidden, "pre": pre, "shared": {"topk": None},
                "mix": None, "next": 0, "result": None}

    def layers(self, st: dict, stop: int, marks: list | None = None) -> None:
        """Advance ``st`` through layers [st["next"], stop). ``marks``: append (first layer, CUDA events at the
        start and after each layer) for the caller's per-layer cost estimate."""
        from .kernels.mhc import collapse_hc, expand_hc, hc_mixes
        from .kernels.mhc_fused import hc_transition
        w = self.w
        layers = w.layers
        step, hashes, shared = st["step"], st["hashes"], st["shared"]
        hidden, pre, state = st["hidden"], st["pre"], st["mix"]
        events = None
        if marks is not None:
            events = [torch.cuda.Event(enable_timing=True)]
            events[0].record()
            marks.append((st["next"], events))
        for i in range(st["next"], stop):
            ly = layers[i]
            if ly.engram_wkv is not None:
                hidden = self.engram_layer(ly, hidden, hashes, step)
                state = None
            if state is None:
                attn_mix = hc_mixes(hidden, ly.hc["attn_fn"], ly.hc["attn_scale"], ly.hc["attn_base"])
                inputs = collapse_hc(hidden, pre, ly.attn_norm)
            else:
                inputs, attn_mix = state
            table = self.compressed_rope if ly.ratio else self.window_rope
            y = self.attention(ly, inputs, step, table, shared)
            inputs, ffn_mix = hc_transition(hidden, y, attn_mix[1], attn_mix[2], attn_mix[0], ly.ffn_norm,
                                            ly.hc["ffn_fn"], ly.hc["ffn_scale"], ly.hc["ffn_base"])
            y = self.moe(ly, inputs)
            pre = ffn_mix[0]
            following = layers[i + 1] if i + 1 < len(layers) else None
            if following is not None and following.engram_wkv is None:
                inputs, mix = hc_transition(hidden, y, ffn_mix[1], ffn_mix[2], pre, following.attn_norm,
                                            following.hc["attn_fn"], following.hc["attn_scale"],
                                            following.hc["attn_base"])
                state = (inputs, mix)
            elif following is not None:
                hidden = expand_hc(y, hidden, ffn_mix[1], ffn_mix[2], inplace=True)
                state = None
            else:
                st["result"], _ = hc_transition(hidden, y, ffn_mix[1], ffn_mix[2], pre, w.norm)
            if events is not None:
                events.append(torch.cuda.Event(enable_timing=True))
                events[-1].record()
        st.update(hidden=hidden, pre=pre, mix=state, next=max(st["next"], stop))

    def end(self, st: dict, logits_rows: torch.Tensor | None):
        if st["next"] != len(self.w.layers):
            raise RuntimeError("a DeepSeek-V4.1 forward ended before its last layer")
        if logits_rows is None:
            return None
        rows = st["result"].index_select(0, logits_rows)
        return self._linear(rows, self.w.head).float()

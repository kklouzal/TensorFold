"""DeepSeek-V4.1-Flash-1017 DSpark head (3 stages, block 5) on one SM70 rank of the TP8 x (ETP4 x EP2) lane.

Port of 1Cat-vLLM 4d57473d ``vllm/models/deepseek_v41/dspark.py`` (V41DraftModel / V41DraftAttention /
V41DraftMoE / DSparkDeepseekV41ForCausalLM), ``markov_fused.py`` and the DSpark parts of
``vllm/v1/spec_decode/{dspark,dflash}.py`` + ``utils.copy_and_expand_dflash_inputs_kernel`` (SAMPLE_FROM_ANCHOR).
Kernels are the vendored 1Cat Triton kernels (``.kernels``: mHC, RoPE/QDQ, router, sink softmax, SwiGLU, Markov)
and the TensorFold MXFP4 expert op (``tensorfold.cuda.sm70.mxfp4_ds41``).

Equations (upstream):
- context rows (target-verified tokens at positions t): ``states = main_norm(main_proj(taps))`` with
  ``taps = cat(mean_hc(h_37), mean_hc(h_38), mean_hc(h_39)) * 2^-6`` in FP16, h_i the FP32 HC residual entering
  target layer i (pre-block, ``DeepseekV41TextConfig.eagle_aux_hidden_state_layer_ids == dspark_target_layer_ids``);
  per stage ``kv = qdq(rope(kv_norm(wkv(states)), t))`` is stored in that stage's window at t. The context rows are
  NOT run through the draft stages.
- proposal for anchor a (the target's newly sampled token) at position p: query ids [a, noise x4] at p..p+4,
  embedded with the TARGET embedding, HC-expanded (4 FP32 streams, pre=[1,0,0,0]); each stage is V41Block's
  unfused order (attention collapse uses the previous sub-layer's pre-mix); attention is non-causal over the 128
  window rows p-128..p-1 plus the 5 query KVs (own sinks, MQA latent 512, scale 512^-0.5); MoE = 128 experts top-3
  (sqrtsoftplus, scale 1.5) + shared expert, partials summed in FP32. Output ``h = collapse_hc(hidden, pre)``
  (FP32, unnormalized); logits = TARGET head(fp16(norm(h))); all 5 rows are sampled: row j proposes the token at
  p+1+j. Markov: ``logits_j + W_head @ embed(prev_j)`` with prev_0 = a, prev_j = draft_{j-1}; greedy = argmax
  (lowest index on ties). Confidence logit_j = ``w_conf . cat(h_j, embed(prev_j))``.
TensorFold changes, recorded:
- TP8 dense/attention (8 heads, 1 output group per rank) and ETP4 x EP2 experts instead of TP4; every cross-rank sum
  in FP32 (``Comm.sum``). On stage 0 with FP16 expert activations, the routed expert slots and shared branch
  use separate planes in one all-reduce so their complete-output FP16 rounding boundaries remain independent.
- Window: a per-slot ring [slots, 3, 256, 512] FP16 indexed by position mod 256 instead of the paged draft cache
  (same rows; reads p-128..p-1 only). Rows of rejected positions may stay in the ring: they lie at >= p and are
  never read before being overwritten. Positions past 16K rotate on the fly (upstream VLLM_V41_PACKED_KV=1 mode).
- Requests are batched: B requests x 5 rows in one pass; row-sensitive reductions (RMS norms, mHC and gate FP32
  projections) launch upstream's M<=6 row kernels over all rows, so every row has upstream's M<=6 reduction
  boundaries in stages 1 and 2. Stage 0 attention and HC retain the per-request FP32 arithmetic and the
  original FP16 cut points. All stages use the native batched MoE.
- Markov embedding in its exact BF16 (upstream rounds it to FP16: 1 flushed, ~350 rounded values, see the audit).
- TP8 vocabulary: the target head is vocabulary-sharded (rank r: rows [V/8 r, +V/8)), so the Markov head is too;
  each greedy step takes every rank's local maximum (lowest index on ties) and one small all-gather picks the
  first rank holding the global maximum: exactly upstream's argmax over the concatenated row. ``main_proj`` is
  row-sharded (640 output features per rank, all-gathered): 137 MiB less replicated memory per rank.
- MXFP4 experts: exact weight decode, FP32 accumulation (see mxfp4_ds41; the dp4a Q8 mode is an A/B option).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from . import dspark_stage0 as stage0
from . import dspark_weights as dw

RING = dw.RING
WINDOW = 128
TAP_SCALE = 2.0 ** -6
EPS = 1e-20


class LocalComm:
    """world == 1: the FP32 'sum' is the identity (the partial is already the total), the gather a copy."""

    def sum(self, partial, dtype=None):
        return partial.float().to(dtype or partial.dtype)

    def gather(self, part):
        return part.contiguous()


@dataclass
class Proposal:
    tokens: torch.Tensor         # int64 [B, 5] drafts (positions p+1 .. p+5)
    logits: torch.Tensor         # FP32 [B, 5, V] Markov-corrected logits (None on the fused greedy path)
    confidence: torch.Tensor     # FP32 [B, 5] confidence logits (sigmoid = conditional acceptance estimate)
    hidden: torch.Tensor         # FP32 [B, 5, 5120] collapsed draft hidden (diagnostics/tests)


class DSparkHead:
    """One rank's DSpark head: the per-slot draft window ring and the batched proposal forward.

    ``comm``: the lane's ``Collectives`` (FP32 sums rounded once, rank-order gathers) for TP8, ``LocalComm`` for
    world 1. ``vocab_first``: this rank's first vocabulary row (the target head's shard).
    ``target_embed`` optionally prepares one NOISE row from the installed target before graph capture."""

    def __init__(self, w: dw.DSparkWeights, *, slots: int, comm=None, device="cuda", q8_experts=False,
                 vocab_first: int = 0, target_embed=None, window_rope=None):
        from .kernels import quant_rope
        self.w, self.comm = w, comm or LocalComm()
        if w.layout.world > 1 and comm is None:
            raise ValueError("a TP8 DSpark head needs the lane's 8-rank FP32 communicator")
        self.device = torch.device(device)
        self.q8 = bool(q8_experts)
        self._use_stage0_exact_loop = False
        if (w.layout.world == 8 and not self.q8 and self.device.type == "cuda"
                and torch.version.git_version == stage0.SINKHORN_TORCH_GIT):
            from .kernels.triton_utils import INTERPRET, triton

            self._use_stage0_exact_loop = (
                not INTERPRET and triton.__version__ == stage0.SINKHORN_TRITON_VERSION
                and torch.cuda.get_device_capability(self.device) == (7, 0)
            )
        if window_rope is not None and (window_rope.ndim != 2 or window_rope.shape[1] != 64
                                        or window_rope.dtype != torch.float32 or window_rope.device != self.device
                                        or not window_rope.is_contiguous()):
            raise ValueError("DSpark requires the installed target's contiguous FP32 window RoPE table")
        self.rope = quant_rope.rope_table(device=self.device) if window_rope is None else window_rope
        self.stage0_frequencies = stage0.frequencies(self.device)
        self.ring = torch.zeros(slots, dw.STAGES, RING, 512, dtype=torch.float16, device=self.device)
        self._proposal_embed = None
        if target_embed is not None:
            from .dspark_embedding import ProposalEmbedding
            self._proposal_embed = ProposalEmbedding(target_embed, self.ring.device)
        self.vocab_first = vocab_first
        self.local_vocab = w.markov_head.shape[0]
        self.first = w.first_expert
        self.local = len(w.layout.experts)

    # ------------------------------------------------------------------ small pieces (upstream V41Norm etc.)
    # Row-sensitive reductions (RMS norms, the mHC and gate FP32 projections) launch the upstream M <= 6 row kernels
    # over every row of the batch at once: each program reduces one row with fixed boundaries, so a row's bits are
    # those of a one-request (5-row) launch, with one launch for all B requests.
    @staticmethod
    def _norm1(x, weight):
        from .kernels.norm import _short_rms
        from .kernels.triton_utils import triton
        if (x.dtype == torch.float16 and x.ndim == 2 and x.shape[0] >= 1 and x.shape[1] in (128, 512, 1280, 5120)
                and x.is_contiguous() and x.is_cuda):
            out = torch.empty_like(x)
            _short_rms[(x.shape[0],)](x, weight, out, H=x.shape[1], EPS=EPS, BLOCK=triton.next_power_of_2(x.shape[1]),
                                      num_warps=4, enable_fp_fusion=False)
            return out
        values = x.float()
        return (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + EPS) * weight).to(x.dtype)

    def _norm(self, x, weight):
        return self._norm1(x.contiguous(), weight)

    @staticmethod
    def _projection(x, weight, normalized=False, eps=1e-20):
        """``fp32_projection``'s M <= 6 kernels for any M (row programs)."""
        from .kernels.mhc import _hc_projection_finish, _hc_projection_parts, _row_projection
        from .kernels.triton_utils import triton
        m = x.shape[0]
        result = torch.empty(m, weight.shape[0], dtype=torch.float32, device=x.device)
        if normalized and weight.shape == (24, 20480):
            dots = torch.empty(20, m, 24, dtype=torch.float32, device=x.device)
            squares = torch.empty_like(dots)
            _hc_projection_parts[(m, 24, 8)](x, weight, dots, squares, m, 8, num_warps=4, enable_fp_fusion=False)
            _hc_projection_finish[(triton.cdiv(result.numel(), 128),)](dots, squares, result, result.numel(), eps,
                                                                       num_warps=4, enable_fp_fusion=False)
            return result
        bits = weight.dtype == torch.bfloat16
        _row_projection[(m, weight.shape[0])](x, weight.view(torch.int16) if bits else weight, result, K=x.shape[1],
                                              N=weight.shape[0], NORMALIZE=normalized, EPS=eps, BLOCK=1024,
                                              BF16_BITS=bits, enable_fp_fusion=False)
        return result

    @classmethod
    def _mixes(cls, hidden, fn, scale, base, hc_eps=1e-6, iters=20):
        """``hc_mixes`` with the row-kernel projection."""
        from .kernels.mhc import _split_sinkhorn
        m = hidden.shape[0]
        mixes = cls._projection(hidden.flatten(1), fn, normalized=True)
        pre = torch.empty(m, 4, device=hidden.device, dtype=torch.float32)
        post, comb = torch.empty_like(pre), torch.empty(m, 4, 4, device=hidden.device, dtype=torch.float32)
        _split_sinkhorn[(m,)](mixes, scale, base, pre, post, comb, EPS=hc_eps, ITERS=iters, enable_fp_fusion=False)
        return pre, post, comb

    @staticmethod
    def _linear(x, weight):
        return torch.matmul(x, weight.t())

    # ------------------------------------------------------------------ context (precompute_and_store_context_kv)
    def combine(self, taps):
        """FP16 [T, 15360] pre-scaled taps (2^-6) -> FP16 [T, 5120] draft context states."""
        if taps.dtype != torch.float16 or taps.ndim != 2 or taps.shape[1] != 3 * dw.HIDDEN:
            raise ValueError("DSpark taps are FP16 [T, 3 * 5120] (layers 37, 38, 39 HC means times 2^-6)")
        # this rank's output rows, gathered in rank order (each element is one rank's dot product)
        states = self.comm.gather(self._linear(taps.contiguous(), self.w.main_proj))
        # FP32 RMS per row (the row kernel for every row count; upstream: FP32 torch above 6 rows)
        return self._norm1(states, self.w.main_norm)

    def context_kv(self, states, positions):
        """Per stage FP16 [T, 512] window rows for context states at positions."""
        from .kernels import quant_rope as qr
        return [qr.attention_qdq(qr.apply_rope(self._norm1(self._linear(states, s.wkv), s.kv_norm), positions,
                                               self.rope), group=128) for s in self.w.stages]

    def insert_context(self, slots, taps, positions, *, padding=None):
        """Store context rows: slots int64 [T] (one slot per row), FP16 taps [T, 15360], positions int64 [T].
        The positions for each live slot must be distinct modulo RING. Verify blocks and the final WINDOW
        prompt positions have this property. A graph may supply (scratch slot, distinct scratch rows) for the
        padded lanes. Those destinations do not change the rotary positions. Graph-capturable."""
        if not (slots.shape == positions.shape == (taps.shape[0],)):
            raise ValueError("DSpark context rows need one slot and position per tap row")
        if taps.shape[0] == 0:
            return
        rows = self.context_kv(self.combine(taps), positions)
        at = positions % RING
        if padding is not None:
            slot, scratch_rows = padding
            at = torch.where(slots == slot, scratch_rows, at)
        for i, kv in enumerate(rows):
            self.ring[slots, i, at] = kv

    def reset_slot(self, slot):
        self.ring[slot].zero_()

    # ------------------------------------------------------------------ one stage
    def _requestwise_attention(self, s, inputs, positions, slots, anchor_pos):
        at = anchor_pos[:, None] - WINDOW + torch.arange(WINDOW, device=self.device)
        windows = self.ring[slots[:, None], s.index, at % RING]
        partial = stage0.attention_partials(s, inputs, positions.split(dw.BLOCK), windows,
                                            at >= 0, self.stage0_frequencies)
        return self.comm.sum(partial, torch.float16)

    def _attention(self, s, x, positions, slots, anchor_pos):
        # The TP8 trace finds same-input attention drift in both later stages.
        if s.index in (1, 2) and self.w.layout.world == 8:
            return self._requestwise_attention(s, x.split(dw.BLOCK), positions, slots, anchor_pos)

        from .kernels import quant_rope as qr
        from .kernels.attention_ops import _attend_fused
        n = x.shape[0]
        heads = s.sink.shape[0]
        qrow = self._norm(self._linear(x, s.wq_a), s.q_norm)
        q = qr.apply_rope(self._linear(qrow, s.wq_b).view(n, heads, 512).contiguous(), positions, self.rope)
        kv = qr.attention_qdq(qr.apply_rope(self._norm(self._linear(x, s.wkv), s.kv_norm), positions, self.rope),
                              group=128)
        b = slots.shape[0]
        at = anchor_pos[:, None] - WINDOW + torch.arange(WINDOW, device=x.device)        # [B, 128]
        window = self.ring[slots[:, None], s.index, at % RING]                            # [B, 128, 512]
        window = torch.where((at >= 0)[..., None], window, torch.zeros((), dtype=window.dtype, device=x.device))
        queries = kv.view(b, dw.BLOCK, 512)
        keys = torch.cat((window, queries), 1)                                            # [B, 133, 512]
        keys = keys[:, None].expand(b, dw.BLOCK, WINDOW + dw.BLOCK, 512).reshape(n, WINDOW + dw.BLOCK, 512)
        unused = (at < 0)[:, None].expand(b, dw.BLOCK, WINDOW)
        unused = torch.cat((unused, torch.zeros(b, dw.BLOCK, dw.BLOCK, dtype=torch.bool, device=x.device)), -1)
        unused = unused.reshape(n, WINDOW + dw.BLOCK).contiguous()
        scores = torch.empty(n, heads, WINDOW + dw.BLOCK, dtype=torch.float32, device=x.device)
        out = torch.empty_like(q)
        _attend_fused(q, keys.contiguous(), unused, 512 ** -0.5, s.sink, out, scores)
        out = qr.apply_rope(out, positions, self.rope, inverse=True)
        out = qr.attention_qdq(out.reshape(n, s.wo_a.shape[0], -1).contiguous(), group=128)
        projected = torch.bmm(out.transpose(0, 1), s.wo_a.transpose(1, 2)).transpose(0, 1).reshape(n, -1)
        return self.comm.sum(self._linear(projected, s.wo_b))

    def _moe(self, s, x):
        from tensorfold.cuda.sm70 import mxfp4_ds41 as mx
        from .kernels.activations import swiglu
        from .kernels.router import route_topk
        logits = self._projection(x.contiguous(), s.gate)
        probs, ids = route_topk(logits, s.gate_bias, dw.TOP_K, 1.5)
        local = ids - self.first
        mine = (local >= 0) & (local < self.local)
        local = torch.where(mine, local, torch.full_like(local, -1)).to(torch.int32).contiguous()
        if s.index == 0 and self.w.layout.world == 8 and not self.q8:
            partials = torch.empty((dw.TOP_K + 1, x.shape[0], s.exp_down.shape[1]), dtype=torch.float32,
                                   device=x.device)
            mx.experts_partials(x.contiguous(), local, s.exp_gate, s.exp_gate_scale, s.exp_up, s.exp_up_scale,
                                s.exp_down, s.exp_down_scale, out=partials[:-1])
            shared = swiglu(self._linear(x, s.sh_gate), self._linear(x, s.sh_up))
            torch.mm(shared, s.sh_down.t(), out_dtype=torch.float32, out=partials[-1])
            return mx.finish_partials(self.comm.sum(partials), ids, probs, experts=s.gate.shape[0])
        routed = mx.experts(x.contiguous(), local, probs.contiguous(), s.exp_gate, s.exp_gate_scale, s.exp_up,
                            s.exp_up_scale, s.exp_down, s.exp_down_scale, q8=self.q8)
        shared = self._linear(swiglu(self._linear(x, s.sh_gate), self._linear(x, s.sh_up)), s.sh_down)
        return self.comm.sum(routed.float() + shared.float())

    def _first_stage(self, s, hidden, positions, previous_pre, slots, anchor_pos):
        hc = s.hc
        blocks = hidden.split(dw.BLOCK)
        pre, post, comb = zip(*(
            stage0.mixes(block, hc["attn_fn"], hc["attn_scale"], hc["attn_base"],
                         fuse_normalization=self._use_stage0_exact_loop) for block in blocks
        ), strict=True)
        inputs = [stage0.collapse(block, prev, s.attn_norm)
                  for block, prev in zip(blocks, previous_pre.split(dw.BLOCK), strict=True)]
        outputs = self._requestwise_attention(s, inputs, positions, slots, anchor_pos).split(dw.BLOCK)
        after_attention = [
            stage0.expand(out, block, mix, matrix)
            for out, block, mix, matrix in zip(outputs, blocks, post, comb, strict=True)
        ]

        next_pre, post, comb = zip(*(
            stage0.mixes(block, hc["ffn_fn"], hc["ffn_scale"], hc["ffn_base"],
                         fuse_normalization=self._use_stage0_exact_loop) for block in after_attention
        ), strict=True)
        inputs = torch.cat([stage0.collapse(block, mix, s.ffn_norm)
                            for block, mix in zip(after_attention, pre, strict=True)])
        outputs = self._moe(s, inputs).split(dw.BLOCK)
        updated = [
            stage0.expand(out, block, mix, matrix)
            for out, block, mix, matrix in zip(outputs, after_attention, post, comb, strict=True)
        ]
        return torch.cat(updated), torch.cat(next_pre)

    def _stage(self, s, hidden, positions, previous_pre, slots, anchor_pos):
        if s.index == 0:
            return self._first_stage(s, hidden, positions, previous_pre, slots, anchor_pos)
        from .kernels.mhc import collapse_hc, expand_hc
        hc = s.hc
        pre, post, comb = self._mixes(hidden, hc["attn_fn"], hc["attn_scale"], hc["attn_base"])
        inputs = collapse_hc(hidden, previous_pre, s.attn_norm)
        hidden = expand_hc(self._attention(s, inputs, positions, slots, anchor_pos), hidden, post, comb)
        next_pre, post, comb = self._mixes(hidden, hc["ffn_fn"], hc["ffn_scale"], hc["ffn_base"])
        inputs = collapse_hc(hidden, pre, s.ffn_norm)
        return expand_hc(self._moe(s, inputs), hidden, post, comb), next_pre

    # ------------------------------------------------------------------ proposal
    def _proposal_rows(self, anchors, embed):
        if embed is None:
            if self._proposal_embed is None:
                raise ValueError("DSpark needs an installed target embedding or a standalone embedding callback")
            return self._proposal_embed(anchors)
        b = anchors.shape[0]
        ids = torch.full((b, dw.BLOCK), dw.NOISE, dtype=torch.int64, device=self.device)
        ids[:, 0] = anchors
        x = embed(ids.reshape(-1))
        if x.dtype != torch.float16 or x.shape != (b * dw.BLOCK, dw.HIDDEN):
            raise ValueError("the target embedding callback returns FP16 [B*5, 5120]")
        return x

    def forward(self, anchors, anchor_pos, slots, embed):
        """Draft hidden for B requests with anchors, anchor positions and slots as int64 [B].

        Pass ``embed(ids [B*5]) -> FP16 [B*5, 5120]`` for standalone use or ``None`` for the installed target.
        The installed path takes contiguous anchor IDs from the proposal input buffer.
        """
        from .kernels.mhc import collapse_hc
        b = anchors.shape[0]
        positions = (anchor_pos[:, None] + torch.arange(dw.BLOCK, device=self.device)).reshape(-1)
        x = self._proposal_rows(anchors, embed)
        hidden = x.float()[:, None].expand(-1, 4, -1).contiguous()
        pre = torch.zeros(b * dw.BLOCK, 4, dtype=torch.float32, device=self.device)
        pre[:, 0] = 1
        for s in self.w.stages:
            hidden, pre = self._stage(s, hidden, positions, pre, slots, anchor_pos)
        return collapse_hc(hidden, pre)                           # FP32 [B*5, 5120], unnormalized

    def head_input(self, h):
        """fp16(norm(h)): the target head's input (upstream compute_logits)."""
        return self._norm(h, self.w.norm).half()

    def _sharded(self):
        from .dspark_markov import ShardedMarkov
        if getattr(self, "_markov_sharded", None) is None:
            gather = self.comm.gather_rank_major if self.w.layout.world == 8 else None
            self._markov_sharded = ShardedMarkov(self.w.markov_head, self.w.markov_embed, self.vocab_first,
                                                 gather, self.w.layout.world)
        return self._markov_sharded

    def draft(self, anchors, anchor_pos, slots, embed, head, *, depth: int = dw.BLOCK):
        """Greedy drafts int64 [B, depth] (positions p+1 .. p+depth), graph-capturable.

        Depth is fixed for the graph and shared by every rank. The bidirectional forward and target head retain
        all five rows per request. Only the sequential Markov steps are limited to the requested prefix.
        ``head(fp16 [B*5, 5120]) -> FP32 [B*5, V_local]`` is this rank's vocabulary shard of the target head.
        """
        if type(depth) is not int or not 1 <= depth <= dw.BLOCK:
            raise ValueError(f"DSpark draft depth must be an integer in [1, {dw.BLOCK}]")
        b = anchors.shape[0]
        h = self.forward(anchors, anchor_pos, slots, embed)
        base = head(self.head_input(h)).view(b, dw.BLOCK, -1)
        markov = self._sharded()
        tokens = torch.empty(b, depth, dtype=torch.int64, device=self.device)
        prev = anchors
        for step in range(depth):
            prev = markov.step(base[:, step], prev)
            tokens[:, step] = prev
        return tokens

    def propose(self, anchors, anchor_pos, slots, embed, head, *, sampler=None):
        """Draft 5 tokens per request (world 1: full-vocabulary ``head``; tests and single-GPU checks).
        ``sampler(logits FP32 [B, V], step) -> int64 [B]`` (None: greedy, upstream MarkovGreedy)."""
        b = anchors.shape[0]
        h = self.forward(anchors, anchor_pos, slots, embed)
        base = head(self.head_input(h))
        if base.shape != (b * dw.BLOCK, dw.VOCAB):
            raise ValueError("the target head callback returns full-vocabulary logits [B*5, V]")
        base = base.float().view(b, dw.BLOCK, dw.VOCAB)
        tokens = torch.empty(b, dw.BLOCK, dtype=torch.int64, device=self.device)
        corrected = None
        if sampler is None:
            markov = self._sharded()
            prev = anchors
            for step in range(dw.BLOCK):
                prev = markov.step(base[:, step].contiguous(), prev)
                tokens[:, step] = prev
        else:
            corrected = torch.empty_like(base)
            prev = anchors
            markov = self._sharded()
            for step in range(dw.BLOCK):
                corrected[:, step] = base[:, step] + markov.bias(prev.clamp(0, dw.VOCAB - 1))
                prev = sampler(corrected[:, step], step)
                tokens[:, step] = prev
        previous = torch.cat((anchors[:, None], tokens[:, :-1]), 1)
        embeds = self.w.markov_embed[previous].float()
        hidden = h.view(b, dw.BLOCK, dw.HIDDEN)
        confidence = (torch.cat((hidden, embeds), -1) @ self.w.confidence.t()).squeeze(-1)
        return Proposal(tokens, corrected, confidence, hidden)


def greedy_accept(drafts, target_argmax):
    """Exact greedy verification. drafts int64 [B, 5] (positions p+1..p+5); target_argmax int64 [B, 6] = the
    target's argmax over its verify rows [anchor, d1..d5] at positions p..p+5. Returns accepted counts [B] (0..5):
    the request emits d_1..d_k then target_argmax[:, k] (the bonus / next anchor)."""
    match = (drafts == target_argmax[:, :-1]).to(torch.int64)
    return match.cumprod(1).sum(1)

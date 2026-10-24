"""Owned Gemma DFlash proposals over persistent, target-accepted context.

The companion has causal sliding layers and a noncausal full layer by default.
Its noise block never enters committed context: a proposal writes the block's
keys and values into slots beyond the committed length, which the next
accepted context overwrites. The official Torch generator uses raw target
embedding rows; the borrowed target head owns softcap30 once.

The target verifies every proposal (probability-ratio rejection for sampled
requests), preserving its output law, so
the companion's arithmetic (BF16 row-reuse projections, or at bits 8 private
FP8 row-scaled copies of the projections and the borrowed head; kernel
geometry) must be deterministic for the same prefix, including when its prompt came from a partial cache hit.
Committed prompt projections therefore keep prompt arithmetic even for a tiny suffix. One DraftState belongs to
one request lane; its storage is allocated once for the admitted context.
"""

from __future__ import annotations

from functools import partial
from pathlib import Path

import torch
import triton
import triton.language as tl

from tensorfold.cleanup import rollback
from tensorfold.cuda import speculative
from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.cuda.requant import Head, prefix, requantize
from tensorfold.cuda.tensor_file import checkpoint_path
from tensorfold.families.qwen3_5.cuda import b16

from ..dflash_metadata import checked_headers, positive
from . import attention
from .forward import rms_norm
from .state import PREFILL_ROWS, companion_rings, sliding_slots

FULL_CHUNK = 1024        # companion full-layer key chunk (no invariance contract: proposals are verified)


@triton.jit
def _draft_prep(SRC, STRIDE, QW, KW, FREQ, POS, QO, KC, VC, RING,
                H: tl.constexpr, HK: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    """Qwen3 companion head norms and BF16 RoPE for one (row, head) of fused [k | v | q] rows.

    Programs j < HK publish normalized, rotated keys and HK <= j < 2HK raw
    values into slot position % RING; later programs rotate query heads into QO.
    The arithmetic mirrors the Torch companion: BF16 normalization rounding,
    BF16 weight product, BF16 cos/sin and BF16 rounding of each RoPE product.
    """
    row = tl.program_id(0)
    j = tl.program_id(1)
    half: tl.constexpr = D // 2
    d = tl.arange(0, half)
    pos = tl.load(POS) + row
    src = SRC + row * STRIDE + j * D
    x1 = tl.load(src + d)
    x2 = tl.load(src + half + d)
    if (j >= HK) & (j < 2 * HK):
        dst = VC + ((pos % RING) * HK + j - HK) * D
        tl.store(dst + d, x1)
        tl.store(dst + half + d, x2)
    else:
        f1, f2 = x1.to(tl.float32), x2.to(tl.float32)
        inv = tl.rsqrt((tl.sum(f1 * f1, 0) + tl.sum(f2 * f2, 0)) / D + EPS)
        if j < HK:
            weight = KW
        else:
            weight = QW
        y1 = ((f1 * inv).to(tl.bfloat16).to(tl.float32) * tl.load(weight + d).to(tl.float32)).to(tl.bfloat16)
        y2 = ((f2 * inv).to(tl.bfloat16).to(tl.float32) * tl.load(weight + half + d).to(tl.float32)).to(tl.bfloat16)
        angle = pos.to(tl.float32) * tl.load(FREQ + d)
        c = tl.cos(angle).to(tl.bfloat16).to(tl.float32)
        s = tl.sin(angle).to(tl.bfloat16).to(tl.float32)
        y1, y2 = y1.to(tl.float32), y2.to(tl.float32)
        a = ((y1 * c).to(tl.bfloat16).to(tl.float32) + (-(y2 * s)).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        b = ((y2 * c).to(tl.bfloat16).to(tl.float32) + (y1 * s).to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
        if j < HK:
            dst = KC + ((pos % RING) * HK + j) * D
        else:
            dst = QO + (row * H + j - 2 * HK) * D
        tl.store(dst + d, a)
        tl.store(dst + half + d, b)


def _project(x: torch.Tensor, weight: torch.Tensor, *, prompt: bool = False) -> torch.Tensor:
    """BF16 projection; prompt rows retain the prompt GEMM regardless of a partial cache hit's row count."""
    m = x.shape[0]
    if prompt or m > 16:
        return b16.prompt(x, weight)
    return b16.matmul_rows(x, weight, max(2, 1 << (m - 1).bit_length()))


def _project8(x: torch.Tensor, linear, *, prompt: bool = False) -> torch.Tensor:
    """FP8-copy projection: explicit prompt rows retain prompt arithmetic even for a one-row cache suffix."""
    return linear.prefill(x) if prompt or x.shape[0] > 16 else linear(x)


class DraftState:
    """One lane's committed companion context: persistent per-layer K/V and its length.

    Sliding layers keep a ring of max(window + block, rows) slots (rounded to
    64): the window plus a proposal block, and one append's rows on distinct
    slots. The full layer keeps every position of the admitted window.
    ``length`` counts committed target positions; ``pos`` is its device copy.
    Proposal IDs and query scratch belong to this lane and are not checkpoint
    authority. Callers serialize this state's operations on one CUDA stream,
    as required for its K/V writes. A proposal never returns scratch storage.
    """

    def __init__(self, cfg, device, window: int, rows: int, block: int):
        self.rings = companion_rings(cfg.layer_types, cfg.sliding_window, block, rows, window)
        self.keys = [torch.empty((n, cfg.kv_heads, cfg.head_dim), dtype=torch.bfloat16, device=device)
                     for n in self.rings]
        self.values = [torch.empty_like(k) for k in self.keys]
        self.rows, self.length = rows, 0
        self.pos = torch.zeros((1,), dtype=torch.int32, device=device)
        # Embedding only reads IDs: all non-anchor rows remain the mask token.
        self.ids = torch.full((block,), cfg.mask_id, dtype=torch.int32, device=device)
        self.query = torch.empty((block, cfg.heads, cfg.head_dim), dtype=torch.bfloat16, device=device)

    def reset(self) -> None:
        self.length = 0

    def _index(self, layer: int, length: int):
        count = min(length, self.rings[layer])
        return torch.arange(length - count, length, device=self.pos.device) % self.rings[layer]

    def snapshot(self):
        """Detached copies of every retained committed row, in position order."""
        held = []
        for i, (k, v) in enumerate(zip(self.keys, self.values)):
            index = self._index(i, self.length)
            held.append((k.index_select(0, index), v.index_select(0, index)))
        return self.length, tuple(held)

    def restore(self, snapshot) -> None:
        length, held = snapshot
        for i, (k, v) in enumerate(held):
            index = self._index(i, length)
            self.keys[i].index_copy_(0, index, k)
            self.values[i].index_copy_(0, index, v)
        self.length = length


class DFlash:
    """One shared companion; each lane owns a DraftState of accepted K/V only."""

    def __init__(self, draft_dir, target, *, context_limit, block=None, bits=16, rows=PREFILL_ROWS):
        if type(bits) is not int or bits not in (8, 16):
            raise ValueError("Gemma DFlash proposal weights are BF16 (16) or private FP8 row-scaled copies (8)")
        head = getattr(getattr(target, "weights", None), "embed", None)
        if bits == 8 and getattr(getattr(head, "weight", None), "dtype", None) != torch.bfloat16:
            raise ValueError("Gemma DFlash FP8 proposals need the target's BF16 tied head")
        cfg, _ = checked_headers(draft_dir)
        if ((target.config.hidden, target.config.layers, target.config.vocab)
                != (cfg.hidden, cfg.target_layers, cfg.vocab)):
            raise ValueError("Gemma DFlash hidden width, target layer count and vocabulary must match the target")
        if cfg.tap_layers != target.tap_layers:
            raise ValueError("Gemma DFlash target tap outputs must match the companion's trained layer IDs")
        self.window = positive(context_limit, "context limit")
        if self.window > cfg.rope.context_limit:
            raise ValueError("Gemma DFlash context exceeds the declared rotary context")
        self.block = cfg.block if block is None else positive(block, "block")
        if not 2 <= self.block <= 128:
            raise ValueError("Gemma DFlash blocks require 2 through128 rows, matching target tree verification")
        if self.block > cfg.sliding_window:
            raise ValueError("Gemma DFlash sliding blocks must fit the declared sliding window")
        if (cfg.heads % cfg.kv_heads or attention.QUERY_TILE % (cfg.heads // cfg.kv_heads)
                or cfg.head_dim not in (64, 128, 256)):
            raise ValueError("Gemma DFlash attention needs a query group dividing 16 and head width 64/128/256")
        self.rows = positive(rows, "context rows")
        # Kernel offsets are int32: cached key rows and full-layer chunk partials must index below 2**31.
        slots = max(companion_rings(cfg.layer_types, cfg.sliding_window, self.block, self.rows, self.window))
        if max(slots * cfg.kv_heads, self.block * cfg.heads * -(-self.window // FULL_CHUNK)) * cfg.head_dim >= 2**31:
            raise ValueError("Gemma DFlash attention offsets exceed int32 at this context")
        self.config, self.device, self.bits = cfg, target.device, bits
        self.hidden = cfg.hidden
        self.tap_layers, self.mask_id = cfg.tap_layers, cfg.mask_id
        self.raw_embed_tokens, self.project_logits = target.raw_embed_tokens, target.project_logits
        reader = SafeTensors([checkpoint_path(Path(draft_dir), "model.safetensors")])
        try:
            loaded = {name: reader.get(name).to(self.device).contiguous() for name in cfg.shapes()}
        except BaseException as primary:  # noqa: BLE001 - preserve primary and reader cleanup failures
            try:
                rollback(reader, primary, reader.close)
            except BaseException as transport:
                if transport is primary:
                    raise
                raise primary from transport
        else:
            reader.close()
        # Fuse projections that share an input: [k | v | q] (the [k | v] prefix
        # projects committed context) and [gate | up]. Fusion keeps every
        # output column's own dot product.
        for layer in range(cfg.layers):
            base = f"layers.{layer}."
            attn = base + "self_attn."
            loaded[base + "kvq"] = torch.cat([loaded.pop(attn + f"{p}_proj.weight") for p in "kvq"])
            loaded[base + "gate_up"] = torch.cat([loaded.pop(base + f"mlp.{p}_proj.weight") for p in ("gate", "up")])
        # Projections: BF16 on the row-reuse kernel, or (bits 8) private FP8 copies (tensorfold.cuda.requant: E4M3
        # codes, one FP32 scale an output row); with FP8 the borrowed head is an FP8 copy of the target's BF16 head
        # as well (the target keeps its own head). Each BF16 source is freed once its copy exists.
        names = list(cfg.fused_shapes())
        kv_width = 2 * cfg.kv_heads * cfg.head_dim
        self.linears = {}
        for name in names:
            weight = loaded.pop(name) if bits == 8 else loaded[name]
            copy = requantize(weight, "fp8") if bits == 8 else None
            self.linears[name] = partial(_project8, linear=copy) if bits == 8 else partial(_project, weight=weight)
            if name.endswith(".kvq"):
                kv = name[:-3] + "kv"
                self.linears[kv] = (partial(_project8, linear=prefix(copy, kv_width)) if bits == 8
                                    else partial(_project, weight=weight[:kv_width]))
            del weight, copy
        if bits == 8:
            self.project_logits = Head(head, "fp8", softcap=target.config.softcap)
        self.weights = loaded
        # Transformers5.6 Qwen3 default RoPE constructs the exponent in FP32.
        # Reusing Flash Next's FP64 default construction would change frequencies.
        exponent = torch.arange(0, cfg.head_dim, 2, dtype=torch.int64).float() / cfg.head_dim
        self.inv_freq = (1.0 / (cfg.rope.theta ** exponent)).to(self.device)
        if bits == 8:
            # Prepare only the providers selected by these validated private
            # copies. First-use compilation belongs to loading, before ready.
            from tensorfold.cuda.nvfp4.linear import _ext, _prompt_ext

            _prompt_ext()
            _ext()

    def make_state(self) -> DraftState:
        return DraftState(self.config, self.device, self.window, self.rows, self.block)

    def _publish(self, fused, state, layer, query: torch.Tensor | None):
        cfg = self.config
        base = f"layers.{layer}.self_attn."
        programs = 2 * cfg.kv_heads + (cfg.heads if query is not None else 0)
        _draft_prep[(fused.shape[0], programs)](
            fused, fused.stride(0), self.weights[base + "q_norm.weight"], self.weights[base + "k_norm.weight"],
            self.inv_freq, state.pos, query if query is not None else fused, state.keys[layer], state.values[layer],
            state.rings[layer], H=cfg.heads, HK=cfg.kv_heads, D=cfg.head_dim, EPS=cfg.eps, num_warps=1,
            enable_fp_fusion=False)

    @torch.no_grad()
    def append(self, state: DraftState, taps: torch.Tensor, *, prompt: bool = False) -> None:
        """Commit accepted taps. Prompt rows always use prompt arithmetic, including a tiny cache suffix.

        Cached prompt K/V must equal fresh prefill K/V: sampled rejection depends on the companion's actual q.
        The lane projection and prompt GEMM have different bits, so row count cannot determine a prompt's role.
        Decode appends retain the existing fast lane projection for at most16 rows.
        """
        cfg = self.config
        if (not isinstance(taps, torch.Tensor) or taps.ndim != 2 or taps.shape[1] != len(self.tap_layers) * self.hidden
                or taps.dtype != torch.bfloat16 or taps.device != self.device
                or state.length + taps.shape[0] > self.window or taps.shape[0] > state.rows):
            raise ValueError("Gemma DFlash accepted taps must match the target and admitted positions")
        if not taps.shape[0]:
            return
        context = rms_norm(self.linears["fc.weight"](taps.contiguous(), prompt=prompt),
                           self.weights["hidden_norm.weight"],
                           cfg.eps)
        state.pos.fill_(state.length)
        for layer in range(cfg.layers):
            self._publish(self.linears[f"layers.{layer}.kv"](context, prompt=prompt), state, layer, None)
        state.length += taps.shape[0]

    @torch.no_grad()
    def propose(self, state: DraftState, anchor: int, count: int, history=None, *, sampled=False, sampling=None
                ) -> speculative.Chains | None:
        """Up to ``count`` proposals after the anchor from committed context only: the shared chain utility's
        argmax for greedy requests or actual sampled q candidates. Both use the preceding proposal path's
        penalties; sampled packets retain their real distribution for target rejection verification."""
        logits = self.logits(state, anchor, count)
        return None if logits is None else speculative.Chains(logits, [history], logits.shape[0], sampled=sampled,
                                                             sampling=sampling)

    @torch.no_grad()
    def logits(self, state: DraftState, anchor: int, count: int) -> torch.Tensor | None:
        """Softcapped FP32 companion logits [n, vocab] for block rows 1..n, n <= ``count``."""
        cfg = self.config
        if type(count) is not int or not 0 <= count <= 127:
            raise ValueError("Gemma DFlash proposal count must be an integer from zero through127")
        length = min(self.block, count + 1, self.window - state.length)
        if length < 2 or state.length == 0:
            return None
        if type(anchor) is not int or not 0 <= anchor < cfg.vocab:
            raise ValueError("Gemma DFlash anchors must be native target-vocabulary token IDs")
        ids = state.ids[:length]
        ids[0] = anchor
        x = self.raw_embed_tokens(ids)
        start = state.length
        state.pos.fill_(start)
        # One Q buffer is safe across layers: attention consumes it before the
        # next layer's publish on this stream. Attention returns fresh storage.
        q = state.query[:length]
        scale = cfg.head_dim ** -0.5
        group = cfg.heads // cfg.kv_heads
        for layer, kind in enumerate(cfg.layer_types):
            base = f"layers.{layer}."
            h = rms_norm(x, self.weights[base + "input_layernorm.weight"], cfg.eps)
            fused = self.linears[base + "kvq"](h)
            self._publish(fused, state, layer, q)
            noncausal = not cfg.causal_layers[layer]
            if kind == "sliding_attention":
                window, chunk = cfg.sliding_window, attention.SLIDING_CHUNK
                slots = sliding_slots(window, length, group, length if noncausal else 0)
            else:
                window, chunk = 0, FULL_CHUNK
                slots = (start + length - 1) // chunk + 1
            # Only the noncausal case reads bounds; its uniform block is
            # derived in registers from pos/length, without materialized rows.
            out = attention.sliding(q, state.keys[layer], state.values[layer], state.pos, state.pos, state.pos,
                                    window=window, slots=slots, blocks=noncausal, scale=scale, chunk=chunk,
                                    uniform_block=noncausal)
            x = x + self.linears[base + "self_attn.o_proj.weight"](out.reshape(length, -1))
            h = rms_norm(x, self.weights[base + "post_attention_layernorm.weight"], cfg.eps)
            gate, up = self.linears[base + "gate_up"](h).split(cfg.width, dim=-1)
            x = x + self.linears[base + "mlp.down_proj.weight"](torch.nn.functional.silu(gate) * up)
        h = rms_norm(x[1:], self.weights["norm.weight"], cfg.eps)
        return self.project_logits(h).float()

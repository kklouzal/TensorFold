"""Owned Qwen DFlash v1 block proposals; every committed token is target-verified.

No checkpoint Python is imported. BF16 storage is retained by default. Explicit FP8
(8: one scale an output row) or NVFP4 (4: an E4M3 scale per 16 inputs and one global
scale) requantizes the projections and a private copy of a floating target head
(tensorfold.cuda.requant); a checkpoint's quantized target head is borrowed as it
is. These change proposals only; target weights, sampling and commits are separate.
Snapshots share immutable K/V tensors and own their per-layer lists.

A round's proposals stop where the drafter's own confidence runs out (``stop_score``): every draft row is verified
by the target at a cost that grows with the window (each row brings its own experts), while a draft rarely holds
once the drafter's top-path probability has fallen below a few percent. Proposals only; outputs are unchanged.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from triton.language.core import TRITON_MAX_TENSOR_NUMEL

from tensorfold.cleanup import rollback
from tensorfold.cuda import speculative
from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.cuda.requant import Head, requantize
from tensorfold.cuda.tensor_file import checkpoint_path
from tensorfold.families.qwen3_5.cuda.draft_attention import block_attention
from tensorfold.families.qwen3_5.cuda.forward import _mm
from tensorfold.families.qwen3_5.cuda.glue import embedding
from tensorfold.families.qwen3_5.cuda.weights import Plain

from ..dflash_metadata import KINDS, checked_headers, floating_head, positive

# Proposal cut: a round keeps its drafts while sum(logsumexp - max) over their rows (the drafter's -log top-path
# probability) stays at or under this many nats, and always the first. Selected on the GB10 (Qwen3.6 heretic + 42d3b34,
# one stream, 64/4096-token contexts, greedy and card sampling, thinking off; runs qwen36-f2s1): 3 gave +27..31%
# decode tok/s over all 15 drafts, 2 within 3% of it, 5 within 5% (0.5% ahead at 4096 card), 1 and fixed 4..12-row
# windows below; a partial thinking-on run agrees (+27..29%). Longer contexts and several streams are unmeasured.
STOP_SCORE = 3.0


def _norm(x, weight, eps):
    # Qwen3 RMSNorm casts the normalized activation before its BF16 weight multiply.
    value = x.float()
    return weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


class DFlash:
    """Full-context, noncausal block diffusion behind the shared draft-tree contract.

    ``stop_score``: a round's drafts end at the first row whose cumulative drafter score exceeds it (``STOP_SCORE``;
    None proposes every row of the block).
    """

    world = 1
    fast = True

    def __init__(self, draft_dir, target, *, context_limit, block=None, bits=16, stop_score=STOP_SCORE):
        if type(bits) is not int or bits not in (4, 8, 16):
            raise ValueError("DFlash bits must be 16, or explicit proposal-only FP8 (8) or NVFP4 (4)")
        cfg, _ = checked_headers(draft_dir)
        if (target.config.hidden, target.config.layers, target.config.vocab) != (cfg.hidden, cfg.target_layers, cfg.vocab):
            raise ValueError("DFlash hidden width, target layer count and vocabulary must match the target")
        target.bind_draft_taps(cfg.tap_layers)
        self.window = positive(context_limit, "context limit")
        if self.window > cfg.rope.context_limit:
            raise ValueError("DFlash context limit exceeds the declared rotary context")
        self.block = cfg.block if block is None else positive(block, "block")
        if not 2 <= self.block <= 128:
            raise ValueError("DFlash blocks require 2 through128 rows, matching the target tree verifier")
        padded = max(16, 1 << (self.block - 1).bit_length())
        if max(cfg.heads // cfg.kv_heads * padded * cfg.head_dim, 64 * cfg.head_dim) > TRITON_MAX_TENSOR_NUMEL:
            raise ValueError("DFlash grouped attention tile exceeds installed Triton tensor capacity")
        if stop_score is not None and (isinstance(stop_score, bool) or not isinstance(stop_score, (int, float))
                                       or not 0 <= stop_score < float("inf")):
            raise ValueError("DFlash proposal stop score must be a finite nonnegative number of nats, or None")
        self.stop_score = stop_score
        self.config, self.device, self.bits = cfg, target.norm.device, bits
        self.hidden, self.layers, self.trained = cfg.hidden, cfg.layers, cfg.block
        self.tap_layers, self.mask_id = cfg.tap_layers, cfg.mask_id
        self.target_embed, self.target_head = target.embed, target.head
        self.weights, self.linears = {}, {}
        reader = SafeTensors([checkpoint_path(Path(draft_dir), "model.safetensors")])
        try:
            for name, shape in cfg.shapes().items():
                value = reader.get(name).to(self.device).contiguous()
                if len(shape) == 2:
                    self.linears[name] = requantize(value, KINDS[bits]) if bits in KINDS else Plain(value)
                else:
                    self.weights[name] = value
        except BaseException as primary:  # noqa: BLE001 - preserve the primary if reader rollback transport fails
            try:
                rollback(reader, primary, reader.close)
            except BaseException as transport:
                if transport is primary:
                    raise
                raise primary from transport
        else:
            reader.close()
        if bits in KINDS and floating_head(self.target_head):    # proposals read a requantized copy of a BF16 head;
            self.target_head = Head(self.target_head, KINDS[bits])  # a head stored quantized is borrowed as it is
        self.inv_freq = cfg.rope.inverse_frequencies(torch).to(self.device)
        self._context = (None,) * cfg.layers, (None,) * cfg.layers, 0, 0

    @property
    def kc(self):
        return self._context[0]

    @property
    def vc(self):
        return self._context[1]

    @property
    def context_len(self):
        return self._context[2]

    @property
    def context_end(self):
        return self._context[3]

    def _lin(self, x, name):
        return _mm(x.contiguous(), self.linears[name])

    def _rotary(self, positions):
        phase = positions.float()[:, None] * self.inv_freq[None]
        phase = torch.cat((phase, phase), -1)
        amplitude = self.config.rope.attention_factor
        return (phase.cos() * amplitude).to(torch.bfloat16)[None], (phase.sin() * amplitude).to(torch.bfloat16)[None]

    def _qk(self, x, layer, kind, cos, sin):
        cfg = self.config
        heads = cfg.heads if kind == "q" else cfg.kv_heads
        base = f"layers.{layer}.self_attn."
        out = self._lin(x, base + f"{kind}_proj.weight").view(-1, heads, cfg.head_dim)
        out = _norm(out, self.weights[base + f"{kind}_norm.weight"], cfg.eps).transpose(0, 1)
        return _rotate(out, cos, sin).contiguous()

    def snapshot(self):
        keys, values, length, end = self._context
        return list(keys), list(values), length, end

    def restore(self, snap):
        keys, values, length, end = snap
        # Sequence copies and tuple allocation may fail. Publish one coherent
        # context only after both copies succeed; previous snapshots remain live.
        keys, values = tuple(keys), tuple(values)
        self._context = keys, values, length, end

    def skip(self, count):
        if type(count) is not int or count < 0 or self.context_end + count > self.window:
            raise ValueError("DFlash skipped context must remain within the admitted window")
        self._context = (None,) * self.layers, (None,) * self.layers, 0, self.context_end + count

    @torch.no_grad()
    def add_taps(self, taps):
        self.restore(self.add_taps_streams([self.snapshot()], [taps])[0])

    @torch.no_grad()
    def add_taps_streams(self, snaps, taps):
        if len(snaps) != len(taps):
            raise ValueError("DFlash requires one target tap tensor per owned context")
        if not snaps:
            return []
        cfg = self.config
        for snap, value in zip(snaps, taps):
            if (value.ndim != 2 or value.shape[1] != len(self.tap_layers) * self.hidden
                    or value.dtype != torch.bfloat16 or value.device != self.device
                    or snap[3] + value.shape[0] > self.window):
                raise ValueError("DFlash committed taps must match the shared target and admitted positions")
        sizes = [value.shape[0] for value in taps]
        if not sum(sizes):
            return [(list(k), list(v), n, end) for k, v, n, end in snaps]
        context = _norm(self._lin(torch.cat(taps), "fc.weight"), self.weights["hidden_norm.weight"], cfg.eps)
        positions = torch.tensor([p for snap, n in zip(snaps, sizes) for p in range(snap[3], snap[3] + n)],
                                 dtype=torch.int64, device=self.device)
        cos, sin = self._rotary(positions)
        keys, values = [list(s[0]) for s in snaps], [list(s[1]) for s in snaps]
        for layer in range(self.layers):
            k = self._qk(context, layer, "k", cos, sin)
            v = self._lin(context, f"layers.{layer}.self_attn.v_proj.weight")
            v = v.view(-1, cfg.kv_heads, cfg.head_dim).transpose(0, 1)
            start = 0
            for stream, n in enumerate(sizes):
                for cache, fresh in ((keys[stream], k), (values[stream], v)):
                    part = fresh[:, start:start + n]
                    cache[layer] = (part if cache[layer] is None else torch.cat((cache[layer], part), 1)).contiguous()
                start += n
        return [(k, v, snap[2] + n, snap[3] + n) for k, v, snap, n in zip(keys, values, snaps, sizes)]

    def in_vocab(self, token):
        return 0 <= token < self.config.vocab

    def _layer(self, layer, x, snaps, length, cos, sin):
        cfg, weights = self.config, self.weights
        base = f"layers.{layer}."
        h = _norm(x, weights[base + "input_layernorm.weight"], cfg.eps)
        q, k = (self._qk(h, layer, kind, cos, sin) for kind in ("q", "k"))
        v = self._lin(h, base + "self_attn.v_proj.weight").view(-1, cfg.kv_heads, cfg.head_dim)
        out = block_attention(q, k, v.transpose(0, 1).contiguous(), [s[0][layer] for s in snaps],
                              [s[1][layer] for s in snaps], length, self.window, cfg.head_dim ** -0.5, False)
        x = x + self._lin(out, base + "self_attn.o_proj.weight")
        h = _norm(x, weights[base + "post_attention_layernorm.weight"], cfg.eps)
        h = F.silu(self._lin(h, base + "mlp.gate_proj.weight")) * self._lin(h, base + "mlp.up_proj.weight")
        return x + self._lin(h, base + "mlp.down_proj.weight")

    @torch.no_grad()
    def launch_blocks(self, snaps, pendings, max_nodes, block=None, histories=None):
        if len(snaps) != len(pendings) or (histories is not None and len(histories) != len(snaps)):
            raise ValueError("DFlash requires one anchor token (and penalty history, if any) per context")
        length = min(self.block if block is None else positive(block, "block"), max_nodes + 1)
        out = [None] * len(snaps)
        if length < 2:
            return out
        live = [i for i, snap in enumerate(snaps) if snap[2] > 0]
        if not live:
            return out
        length = min(length, *(self.window - snaps[i][3] for i in live))
        if length < 2:
            return out
        if any(type(pendings[i]) is not int or not self.in_vocab(pendings[i]) for i in live):
            raise ValueError("DFlash anchors must be native target-vocabulary token IDs")
        cfg = self.config
        active = [snaps[i] for i in live]
        ids = torch.tensor([t for i in live for t in [pendings[i]] + [self.mask_id] * (length - 1)],
                           dtype=torch.int32, device=self.device)
        x = embedding(ids, self.target_embed)
        positions = torch.tensor([p for snap in active for p in range(snap[3], snap[3] + length)],
                                 dtype=torch.int64, device=self.device)
        cos, sin = self._rotary(positions)
        for layer in range(self.layers):
            x = self._layer(layer, x, active, length, cos, sin)
        h = _norm(x.view(len(live), length, self.hidden)[:, 1:].reshape(-1, self.hidden),
                  self.weights["norm.weight"], cfg.eps)
        logits = _mm(h, self.target_head).float()
        # The argmax proposals; a penalized stream's rows are penalized as the target's will be.
        chains = speculative.Chains(logits, [None if histories is None else histories[i] for i in live], length - 1)
        for row, i in enumerate(live):
            out[i] = chains, row
        return out

    def launch_block(self, pending, max_nodes, block=None, history=None):
        return self.launch_blocks([self.snapshot()], [pending], max_nodes, block, [history])[0]

    def finish_tree(self, launched, context_length, max_nodes, sampling=None, history=None):
        """A launched chain, cut at ``stop_score``. ``context_length`` (its first proposal's absolute position, the
        anchor's + 1) and ``sampling`` complete the drafter interface; argmax proposals use neither."""
        if launched is None:
            return [], [], []
        chains, row = launched
        return chains.finish(row, max_nodes, history, stop=self.stop_score)

    def propose_tree(self, pending, context_length, max_nodes, sampling=None, block=None, history=None):
        return self.finish_tree(self.launch_block(pending, max_nodes, block, history), context_length, max_nodes,
                                sampling, history)[:2]

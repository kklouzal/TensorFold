"""Owned BF16 Gemma DFlash proposals over immutable, target-accepted context.

The companion has causal sliding layers and a noncausal full layer by default.
Its noise block never enters committed context. The official Torch generator
uses raw target embedding rows; the borrowed target head owns softcap30 once.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from triton.language.core import TRITON_MAX_TENSOR_NUMEL

from tensorfold.cleanup import rollback
from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.cuda.tensor_file import checkpoint_path
from tensorfold.families.qwen3_5.cuda.draft_attention import block_attention
from tensorfold.families.qwen3_5.cuda.weights import Plain

from ..dflash_metadata import checked_headers, positive


def _norm(x, weight, eps):
    # Qwen3 norm, unlike the target's Gemma norm: BF16 normalization rounding
    # precedes multiplication by the BF16 learned weight, without a +1 offset.
    value = x.float()
    return weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


class DFlash:
    """One shared companion; per-stream snapshots contain accepted K/V only."""

    world = 1
    fast = True

    def __init__(self, draft_dir, target, *, context_limit, block=None, bits=16):
        if type(bits) is not int or bits != 16:
            raise ValueError("Gemma DFlash proposal weights must retain BF16 storage")
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
        padded = max(16, 1 << (self.block - 1).bit_length())
        if max(cfg.heads // cfg.kv_heads * padded * cfg.head_dim, 64 * cfg.head_dim) > TRITON_MAX_TENSOR_NUMEL:
            raise ValueError("Gemma DFlash grouped attention tile exceeds installed Triton tensor capacity")
        self.config, self.device, self.bits = cfg, target.device, bits
        self.hidden, self.layers, self.trained = cfg.hidden, cfg.layers, cfg.block
        self.tap_layers, self.mask_id = cfg.tap_layers, cfg.mask_id
        self.raw_embed_tokens, self.project_logits = target.raw_embed_tokens, target.project_logits
        self.weights, self.linears = {}, {}
        reader = SafeTensors([checkpoint_path(Path(draft_dir), "model.safetensors")])
        try:
            for name, shape in cfg.shapes().items():
                value = reader.get(name).to(self.device).contiguous()
                if len(shape) == 2:
                    self.linears[name] = Plain(value)
                else:
                    self.weights[name] = value
        except BaseException as primary:  # noqa: BLE001 - preserve primary and reader cleanup failures
            try:
                rollback(reader, primary, reader.close)
            except BaseException as transport:
                if transport is primary:
                    raise
                raise primary from transport
        else:
            reader.close()
        # Transformers5.6 Qwen3 default RoPE constructs the exponent in FP32.
        # Reusing Flash Next's FP64 default construction would change frequencies.
        exponent = torch.arange(0, cfg.head_dim, 2, dtype=torch.int64).float() / cfg.head_dim
        self.inv_freq = (1.0 / (cfg.rope.theta ** exponent)).to(self.device)
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
        return self.linears[name](x.contiguous())

    def _rotary(self, positions):
        phase = positions.float()[:, None] * self.inv_freq[None]
        phase = torch.cat((phase, phase), -1)
        return phase.cos().to(torch.bfloat16)[None], phase.sin().to(torch.bfloat16)[None]

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
        # Allocation failure must not publish mismatched K/V lists. Tensors are
        # immutable after publication, so snapshots own lists and share tensors.
        keys, values = tuple(keys), tuple(values)
        self._context = keys, values, length, end

    def skip(self, count):
        if type(count) is not int or count < 0 or self.context_end + count > self.window:
            raise ValueError("Gemma DFlash skipped context must remain within the admitted window")
        self._context = (None,) * self.layers, (None,) * self.layers, 0, self.context_end + count

    @torch.no_grad()
    def add_taps(self, taps):
        """Append only target-verified rows; the caller owns acceptance selection."""
        self.restore(self.add_taps_streams([self.snapshot()], [taps])[0])

    @torch.no_grad()
    def add_taps_streams(self, snaps, taps):
        if len(snaps) != len(taps):
            raise ValueError("Gemma DFlash requires one accepted target tap tensor per context")
        if not snaps:
            return []
        cfg = self.config
        for snap, value in zip(snaps, taps):
            if (not isinstance(value, torch.Tensor) or value.ndim != 2
                    or value.shape[1] != len(self.tap_layers) * self.hidden
                    or value.dtype != torch.bfloat16 or value.device != self.device
                    or snap[3] + value.shape[0] > self.window):
                raise ValueError("Gemma DFlash accepted taps must match the target and admitted positions")
        sizes = [value.shape[0] for value in taps]
        rows = sum(sizes)
        if not rows:
            return [(list(k), list(v), n, end) for k, v, n, end in snaps]
        context = _norm(self._lin(torch.cat(taps), "fc.weight"), self.weights["hidden_norm.weight"], cfg.eps)
        positions = torch.tensor([p for snap, n in zip(snaps, sizes) for p in range(snap[3], snap[3] + n)],
                                 dtype=torch.int64, device=self.device)
        cos, sin = self._rotary(positions)
        keys, values = [list(s[0]) for s in snaps], [list(s[1]) for s in snaps]
        for layer, kind in enumerate(cfg.layer_types):
            k = self._qk(context, layer, "k", cos, sin)
            v = self._lin(context, f"layers.{layer}.self_attn.v_proj.weight")
            v = v.view(-1, cfg.kv_heads, cfg.head_dim).transpose(0, 1)
            keep = cfg.sliding_window - 1 if kind == "sliding_attention" else self.window
            start = 0
            for stream, n in enumerate(sizes):
                if n:
                    for cache, fresh in ((keys[stream], k), (values[stream], v)):
                        part = fresh[:, start:start + n]
                        previous = cache[layer]
                        joined = part if previous is None else torch.cat((previous, part), 1)
                        tail = joined[:, -keep:]
                        # Contiguity alone does not compact a one-row/head view.
                        # Retire discarded rows and other streams' backing too.
                        sliced = joined.shape[1] > keep or (previous is None and n != rows)
                        cache[layer] = (tail.clone(memory_format=torch.contiguous_format)
                                        if sliced else tail.contiguous())
                start += n
        return [(k, v, snap[2] + n, snap[3] + n) for k, v, snap, n in zip(keys, values, snaps, sizes)]

    def in_vocab(self, token):
        return type(token) is int and 0 <= token < self.config.vocab

    def _layer(self, layer, x, snaps, length, cos, sin):
        cfg, weights = self.config, self.weights
        base = f"layers.{layer}."
        h = _norm(x, weights[base + "input_layernorm.weight"], cfg.eps)
        q, k = (self._qk(h, layer, kind, cos, sin) for kind in ("q", "k"))
        v = self._lin(h, base + "self_attn.v_proj.weight").view(-1, cfg.kv_heads, cfg.head_dim)
        # Shared attention uses distance < window+1; the checkpoint's SWA is
        # strict distance < sliding_window. Full context uses the admitted limit.
        window = cfg.sliding_window - 1 if cfg.layer_types[layer] == "sliding_attention" else self.window
        out = block_attention(q, k, v.transpose(0, 1).contiguous(), [s[0][layer] for s in snaps],
                              [s[1][layer] for s in snaps], length, window, cfg.head_dim ** -0.5,
                              cfg.causal_layers[layer])
        x = x + self._lin(out, base + "self_attn.o_proj.weight")
        h = _norm(x, weights[base + "post_attention_layernorm.weight"], cfg.eps)
        h = F.silu(self._lin(h, base + "mlp.gate_proj.weight")) * self._lin(h, base + "mlp.up_proj.weight")
        return x + self._lin(h, base + "mlp.down_proj.weight")

    @torch.no_grad()
    def launch_blocks(self, snaps, pendings, max_nodes, block=None):
        if len(snaps) != len(pendings):
            raise ValueError("Gemma DFlash requires one anchor token per context")
        if type(max_nodes) is not int or not 0 <= max_nodes <= 127:
            raise ValueError("Gemma DFlash proposal count must be an integer from zero through127")
        length = self.block if block is None else positive(block, "block")
        if not 2 <= length <= self.block:
            raise ValueError("Gemma DFlash requested block must fit the prepared block size")
        length = min(length, max_nodes + 1)
        out = [None] * len(snaps)
        if length < 2:
            return out
        live = [i for i, snap in enumerate(snaps) if snap[2] > 0]
        if not live:
            return out
        length = min(length, *(self.window - snaps[i][3] for i in live))
        if length < 2:
            return out
        if any(not self.in_vocab(pendings[i]) for i in live):
            raise ValueError("Gemma DFlash anchors must be native target-vocabulary token IDs")
        active = [snaps[i] for i in live]
        ids = torch.tensor([t for i in live for t in [pendings[i]] + [self.mask_id] * (length - 1)],
                           dtype=torch.int32, device=self.device)
        x = self.raw_embed_tokens(ids)
        positions = torch.tensor([p for snap in active for p in range(snap[3], snap[3] + length)],
                                 dtype=torch.int64, device=self.device)
        cos, sin = self._rotary(positions)
        for layer in range(self.layers):
            x = self._layer(layer, x, active, length, cos, sin)
        h = _norm(x.view(len(live), length, self.hidden)[:, 1:].reshape(-1, self.hidden),
                  self.weights["norm.weight"], self.config.eps)
        logits = self.project_logits(h).float()
        best, tokens = logits.max(-1)
        scores = (torch.logsumexp(logits, -1) - best).view(len(live), length - 1).cumsum(-1)
        # Attention receives context pointers through an indirect device table.
        # Keep the immutable owners live until result readback observes completion.
        shared = [tokens.view(len(live), length - 1), scores, None, active]
        for row, i in enumerate(live):
            out[i] = shared, row
        return out

    def launch_block(self, pending, max_nodes, block=None):
        return self.launch_blocks([self.snapshot()], [pending], max_nodes, block)[0]

    def finish_tree(self, launched, context_length, max_nodes, sampling=None):
        if launched is None:
            return [], [], []
        shared, row = launched
        if shared[2] is None:
            shared[2] = shared[0].cpu().tolist(), shared[1].cpu().tolist()
            shared[3] = None
        ids = shared[2][0][row][:max_nodes]
        return ids, list(range(-1, len(ids) - 1)), shared[2][1][row][:max_nodes]

    def propose_tree(self, pending, context_length, max_nodes, sampling=None, block=None):
        return self.finish_tree(self.launch_block(pending, max_nodes, block), context_length, max_nodes, sampling)[:2]

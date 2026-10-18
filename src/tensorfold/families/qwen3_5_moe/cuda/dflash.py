"""Owned Qwen DFlash v1 block proposals; every committed token is target-verified.

No checkpoint Python is imported. BF16 storage is retained by default. Explicit
affine4 changes proposals only; target weights, sampling and commits are separate.

Committed context K/V live in preallocated stores (``_Store``) appended in place, never
re-concatenated: a decode round writes only its accepted rows. Store capacities are
MIN_ROWS * 2**k rows capped at the admitted window, so a lineage allocates O(log window)
times from one size series and the caching allocator reuses freed stores across lineages
instead of every round allocating a longer copy.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
from triton.language.core import TRITON_MAX_TENSOR_NUMEL

from tensorfold.cleanup import rollback
from tensorfold.cuda import speculative
from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.cuda.tensor_file import checkpoint_path
from tensorfold.families.qwen3_5.cuda.dflash2 import quantize4
from tensorfold.families.qwen3_5.cuda.forward import _mm
from tensorfold.families.qwen3_5.cuda.glue import embedding
from tensorfold.families.qwen3_5.cuda.qmm_fast import tile
from tensorfold.families.qwen3_5.cuda.weights import Plain

from ..dflash_metadata import QUERY_TILE, checked_headers, padded_block, positive
from . import context_attention


def _norm(x, weight, eps):
    # Qwen3 RMSNorm casts the normalized activation before its BF16 weight multiply.
    value = x.float()
    return weight * (value * torch.rsqrt(value.square().mean(-1, keepdim=True) + eps)).to(x.dtype)


def _rotate(x, cos, sin):
    half = x.shape[-1] // 2
    return x * cos + torch.cat((-x[..., half:], x[..., :half]), -1) * sin


MIN_ROWS = 1024     # the smallest store capacity; capacities are MIN_ROWS * 2**k, capped at the admitted window


class _Store:
    """One context lineage's committed K/V rows, [layers, kv_heads, capacity, head_dim] each.

    Every snapshot of the lineage holds this object (as each entry of its key and value lists), so a growth that
    moves the rows reaches all of them, as the target State's shared KV list does. ``tail`` is the row count the
    lineage's latest append left written; rows past a snapshot's length belong to whichever append came later.
    """

    __slots__ = ("k", "v", "tail")

    def __init__(self, cfg, capacity, device):
        self.k = torch.empty((cfg.layers, cfg.kv_heads, capacity, cfg.head_dim), dtype=torch.bfloat16, device=device)
        self.v = torch.empty_like(self.k)
        self.tail = 0

    @property
    def capacity(self):
        return self.k.shape[2]

    def take(self, other, rows):
        """Copy ``other``'s first ``rows`` rows into this store's first rows."""
        if rows:
            self.k[:, :, :rows].copy_(other.k[:, :, :rows])
            self.v[:, :, :rows].copy_(other.v[:, :, :rows])


class DFlash:
    """Full-context, noncausal block diffusion behind the shared draft-tree contract.

    A context is the shared tuple (keys, values, length, end): ``length`` committed rows of positions
    [end - length, end), keys/values one entry a layer, each the lineage's ``_Store`` (None before any row).
    ``exclusive``: one live lineage head at a time (the one-stream engine, which drops cached extensions before it
    resumes a shorter prefix, as for its target State): an append past a snapshot overwrites later rows in place.
    Otherwise an append that does not extend its lineage's latest rows copies the snapshot's rows into a new store,
    so concurrent streams resumed from one cached context never write each other's rows.
    """

    world = 1

    def __init__(self, draft_dir, target, *, context_limit, block=None, bits=16, exclusive=False):
        if type(bits) is not int or bits not in (4, 16):
            raise ValueError("DFlash bits must be 16 or explicit proposal-only affine4")
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
        if max(QUERY_TILE, padded_block(self.block)) * cfg.head_dim > TRITON_MAX_TENSOR_NUMEL:
            raise ValueError("DFlash grouped attention tile exceeds installed Triton tensor capacity")
        if type(exclusive) is not bool:
            raise ValueError("DFlash context exclusivity must be a bool")
        self.config, self.device, self.bits, self.exclusive = cfg, target.norm.device, bits, exclusive
        self.hidden, self.layers, self.trained = cfg.hidden, cfg.layers, cfg.block
        self.tap_layers, self.mask_id = cfg.tap_layers, cfg.mask_id
        self.target_embed, self.target_head = target.embed, target.head
        self.weights, self.linears = {}, {}
        reader = SafeTensors([checkpoint_path(Path(draft_dir), "model.safetensors")])
        try:
            for name, shape in cfg.shapes().items():
                value = reader.get(name).to(self.device).contiguous()
                if len(shape) == 2:
                    self.linears[name] = tile(quantize4(value)) if bits == 4 else Plain(value)
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
        self.inv_freq = cfg.rope.inverse_frequencies(torch).to(self.device)
        self._context = (None,) * cfg.layers, (None,) * cfg.layers, 0, 0

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
        stores = [self._claim(snap, n) if n else None for snap, n in zip(snaps, sizes)]
        context = _norm(self._lin(torch.cat(taps), "fc.weight"), self.weights["hidden_norm.weight"], cfg.eps)
        positions = torch.tensor([p for snap, n in zip(snaps, sizes) for p in range(snap[3], snap[3] + n)],
                                 dtype=torch.int64, device=self.device)
        cos, sin = self._rotary(positions)
        dst, src = [], []
        for layer in range(self.layers):
            k = self._qk(context, layer, "k", cos, sin)
            v = self._lin(context, f"layers.{layer}.self_attn.v_proj.weight")
            v = v.view(-1, cfg.kv_heads, cfg.head_dim).transpose(0, 1)
            start = 0
            for store, snap, n in zip(stores, snaps, sizes):
                if n:
                    rows = slice(snap[2], snap[2] + n)
                    dst += [store.k[layer, :, rows], store.v[layer, :, rows]]
                    src += [k[:, start:start + n], v[:, start:start + n]]
                start += n
        torch._foreach_copy_(dst, src)
        return [(list(snap[0]), list(snap[1]), snap[2], snap[3]) if not n else
                ([store] * self.layers, [store] * self.layers, snap[2] + n, snap[3] + n)
                for store, snap, n in zip(stores, snaps, sizes)]

    def _claim(self, snap, rows):
        """The store a snapshot's next ``rows`` rows go into (grown or copied first when needed); claims its tail."""
        store, length = snap[0][0], snap[2]
        need = length + rows
        if store is not None and length > store.capacity:
            raise ValueError("DFlash context length exceeds its store")
        if store is None or not (self.exclusive or length == store.tail):
            fresh = _Store(self.config, self._capacity(need), self.device)
            if store is not None:
                fresh.take(store, length)
            store = fresh
        elif need > store.capacity:
            grown = _Store(self.config, self._capacity(need), self.device)
            grown.take(store, length)
            store.k, store.v = grown.k, grown.v
        store.tail = need
        return store

    def _capacity(self, need):
        """The next power-of-two multiple of MIN_ROWS holding ``need`` rows, capped at the admitted window: every
        lineage asks the allocator for sizes from one series, so stores freed by one are reused by the next."""
        capacity = MIN_ROWS
        while capacity < need:
            capacity *= 2
        return min(self.window, capacity)

    def in_vocab(self, token):
        return 0 <= token < self.config.vocab

    def _layer(self, layer, x, launch, cos, sin):
        cfg, weights = self.config, self.weights
        base = f"layers.{layer}."
        h = _norm(x, weights[base + "input_layernorm.weight"], cfg.eps)
        q, k = (self._qk(h, layer, kind, cos, sin) for kind in ("q", "k"))
        v = self._lin(h, base + "self_attn.v_proj.weight").view(-1, cfg.kv_heads, cfg.head_dim)
        out = context_attention.block_attention(q, k, v.transpose(0, 1).contiguous(), launch, layer,
                                                cfg.head_dim ** -0.5)
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
        if any(snap[0][0] is None or snap[2] > snap[0][0].capacity for snap in active):
            raise ValueError("DFlash contexts with committed rows must name their store")
        ids = torch.tensor([t for i in live for t in [pendings[i]] + [self.mask_id] * (length - 1)],
                           dtype=torch.int32, device=self.device)
        x = embedding(ids, self.target_embed)
        positions = torch.tensor([p for snap in active for p in range(snap[3], snap[3] + length)],
                                 dtype=torch.int64, device=self.device)
        cos, sin = self._rotary(positions)
        launch = context_attention.Round([snap[0][0] for snap in active], [snap[2] for snap in active],
                                         heads=cfg.heads, kv_heads=cfg.kv_heads, dim=cfg.head_dim, length=length,
                                         device=self.device)
        for layer in range(self.layers):
            x = self._layer(layer, x, launch, cos, sin)
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
        """A launched chain. ``context_length`` (its first proposal's absolute position, the anchor's + 1) and
        ``sampling`` complete the drafter interface; argmax proposals use neither."""
        if launched is None:
            return [], [], []
        chains, row = launched
        return chains.finish(row, max_nodes, history)

    def propose_tree(self, pending, context_length, max_nodes, sampling=None, block=None, history=None):
        return self.finish_tree(self.launch_block(pending, max_nodes, block, history), context_length, max_nodes,
                                sampling, history)[:2]

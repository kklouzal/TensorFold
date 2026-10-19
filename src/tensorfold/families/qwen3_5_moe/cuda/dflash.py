"""Owned Qwen DFlash v1 block proposals; every committed token is target-verified.

No checkpoint Python is imported. BF16 storage is retained by default. Explicit
affine4 changes proposals only; target weights, sampling and commits are separate.

Committed context K/V live in preallocated stores (``_Store``) appended in place, never
re-concatenated: a decode round writes only its accepted rows. Store capacities are
MIN_ROWS * 2**k rows capped at the admitted window, so a lineage allocates O(log window)
times from one size series and the caching allocator reuses freed stores across lineages
instead of every round allocating a longer copy.

A round's proposals stop where the drafter's own confidence runs out (``stop_score``): every draft row is verified
by the target at a cost that grows with the window (each row brings its own experts), while a draft rarely holds
once the drafter's top-path probability has fallen below a few percent. Proposals only; outputs are unchanged.
"""
from __future__ import annotations

from itertools import accumulate
from pathlib import Path

import torch
from triton.language.core import TRITON_MAX_TENSOR_NUMEL

from tensorfold.cleanup import rollback
from tensorfold.cuda import speculative
from tensorfold.cuda.direct_read import SafeTensors
from tensorfold.cuda.tensor_file import checkpoint_path
from tensorfold.families.qwen3_5.cuda.dflash2 import quantize4
from tensorfold.families.qwen3_5.cuda.forward import _mm
from tensorfold.families.qwen3_5.cuda.glue import embedding
from tensorfold.families.qwen3_5.cuda.qmm_fast import rows as tiled_rows, tile
from tensorfold.families.qwen3_5.cuda.weights import Plain

from ..dflash_metadata import QUERY_TILE, checked_headers, padded_block, positive, projection_groups
from . import context_attention, draft_glue

MIN_ROWS = 1024     # the smallest store capacity; capacities are MIN_ROWS * 2**k, capped at the admitted window
# Proposal cut: a round keeps its drafts while sum(logsumexp - max) over their rows (the drafter's -log top-path
# probability) stays at or under this many nats, and always the first. Selected on the GB10 (Qwen3.6 heretic + 42d3b34,
# one stream, 64/4096-token contexts, greedy and card sampling, thinking off; runs qwen36-f2s1): 3 gave +27..31%
# decode tok/s over all 15 drafts, 2 within 3% of it, 5 within 5% (0.5% ahead at 4096 card), 1 and fixed 4..12-row
# windows below; a partial thinking-on run agrees (+27..29%). Longer contexts and several streams are unmeasured.
STOP_SCORE = 3.0


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
    ``stop_score``: a round's drafts end at the first row whose cumulative drafter score exceeds it (``STOP_SCORE``;
    None proposes every row of the block).
    """

    world = 1

    def __init__(self, draft_dir, target, *, context_limit, block=None, bits=16, exclusive=False,
                 stop_score=STOP_SCORE):
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
        if stop_score is not None and (isinstance(stop_score, bool) or not isinstance(stop_score, (int, float))
                                       or not 0 <= stop_score < float("inf")):
            raise ValueError("DFlash proposal stop score must be a finite nonnegative number of nats, or None")
        self.stop_score = stop_score
        self.config, self.device, self.bits, self.exclusive = cfg, target.norm.device, bits, exclusive
        self.hidden, self.layers, self.trained = cfg.hidden, cfg.layers, cfg.block
        self.tap_layers, self.mask_id = cfg.tap_layers, cfg.mask_id
        self.target_embed, self.target_head = target.embed, target.head
        self.weights, self.linears = {}, {}
        reader = SafeTensors([checkpoint_path(Path(draft_dir), "model.safetensors")])
        try:
            for name, shape in cfg.shapes().items():
                if len(shape) == 1:
                    self.weights[name] = reader.get(name).to(self.device).contiguous()
            self.linears, self._context_kv = self._prepare(lambda name: reader.get(name).to(self.device))
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

    def _prepare(self, read):
        """``projection_groups``' projections (each group's parts read one at a time into its rows) and each layer's
        k|v rows for the context append: views of its q|k|v storage (BF16 rows or whole 4-bit tiles), not owned."""
        cfg, shapes, out = self.config, self.config.shapes(), {}
        for key, names in projection_groups(cfg):
            if len(names) == 1:
                weight = read(names[0]).contiguous()
            else:
                rows = [shapes[name][0] for name in names]
                weight = torch.empty((sum(rows), shapes[names[0]][1]), dtype=torch.bfloat16, device=self.device)
                for name, start, count in zip(names, accumulate([0] + rows), rows):
                    weight[start:start + count].copy_(read(name))
            out[key] = tile(quantize4(weight)) if self.bits == 4 else Plain(weight)
            del weight
        q, qkv = cfg.heads * cfg.head_dim, (cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim
        stacked = [out[f"layers.{i}.qkv"] for i in range(cfg.layers)]
        return out, [tiled_rows(w, q, qkv) if self.bits == 4 else Plain(w.weight[q:]) for w in stacked]

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
        return (phase.cos() * amplitude).to(torch.bfloat16), (phase.sin() * amplitude).to(torch.bfloat16)

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
        _, context = draft_glue.rmsnorm(self._lin(torch.cat(taps), "fc"), self.weights["hidden_norm.weight"], cfg.eps)
        positions = torch.tensor([p for snap, n in zip(snaps, sizes) for p in range(snap[3], snap[3] + n)],
                                 dtype=torch.int64, device=self.device)
        cos, sin = self._rotary(positions)
        dest = draft_glue.destinations(stores, [snap[2] for snap in snaps], sizes, self.device)
        for layer in range(self.layers):
            draft_glue.kv_append(_mm(context, self._context_kv[layer]),
                                 self.weights[f"layers.{layer}.self_attn.k_norm.weight"], cos, sin, dest, layer,
                                 cfg.eps, kv_heads=cfg.kv_heads, head_dim=cfg.head_dim)
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

    def _layer(self, layer, x, mlp, launch, cos, sin):
        """One drafter layer; ``mlp`` is the previous layer's MLP output, added here (None in the first layer)."""
        cfg, weights = self.config, self.weights
        base = f"layers.{layer}."
        x, h = draft_glue.rmsnorm(x, weights[base + "input_layernorm.weight"], cfg.eps, residual=mlp)
        q, k, v = draft_glue.qkv_heads(self._lin(h, base + "qkv"), weights[base + "self_attn.q_norm.weight"],
                                       weights[base + "self_attn.k_norm.weight"], cos, sin, cfg.eps,
                                       heads=cfg.heads, kv_heads=cfg.kv_heads, head_dim=cfg.head_dim)
        out = context_attention.block_attention(q, k, v, launch, layer, cfg.head_dim ** -0.5)
        x, h = draft_glue.rmsnorm(self._lin(out, base + "o"), weights[base + "post_attention_layernorm.weight"],
                                  cfg.eps, residual=x)
        return x, self._lin(draft_glue.silu_mul(self._lin(h, base + "gate_up")), base + "down")

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
        mlp = None
        for layer in range(self.layers):
            x, mlp = self._layer(layer, x, mlp, launch, cos, sin)
        # One launch adds the last MLP and norms every row, the anchors' too (a row's norm reads only that row).
        _, h = draft_glue.rmsnorm(x, self.weights["norm.weight"], cfg.eps, residual=mlp)
        logits = _mm(h.view(len(live), length, self.hidden)[:, 1:].reshape(-1, self.hidden), self.target_head).float()
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
        ids, parents, scores = chains.finish(row, max_nodes, history)
        if self.stop_score is not None and ids:
            keep = max(1, next((i for i, score in enumerate(scores) if score > self.stop_score), len(scores)))
            ids, parents, scores = ids[:keep], parents[:keep], scores[:keep]
        return ids, parents, scores

    def propose_tree(self, pending, context_length, max_nodes, sampling=None, block=None, history=None):
        return self.finish_tree(self.launch_block(pending, max_nodes, block, history), context_length, max_nodes,
                                sampling, history)[:2]

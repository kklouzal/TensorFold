"""Flash Next decode kernels give each row the same bits as serial decoding, including across streams."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from tensorfold.families.qwen3_5 import tensor_units
from tensorfold.kernels.qwen.dense.v1 import lane_qmm, simd_qmm
from tensorfold.kernels.qwen.flash_next.v1 import attention, base, embed, experts, gdn, hc, rows


class _Split:
    """Projections of one input kept in stacks of one bit width each; outputs come back in member order."""

    def __init__(self, parts: list[tuple[Any, list[int]]]) -> None:
        self.parts = parts

    def __call__(self, x: mx.array) -> mx.array:
        outs: dict[int, mx.array] = {}
        for linear, members in self.parts:
            y = project(x, linear)
            sizes = [int(s) for s in linear.__dict__["member_rows"]]
            cuts = list(np.cumsum(sizes)[:-1])
            for m, part in zip(members, mx.split(y, cuts, axis=-1) if cuts else [y]):
                outs[m] = part
        return mx.concatenate([outs[m] for m in sorted(outs)], axis=-1)


class _PreparedLinear(nn.QuantizedLinear):
    """An operation's exact packed stack; no random initialization or source edits."""

    def __init__(self, parts: list[Any], group: int) -> None:
        nn.Module.__init__(self)
        self.bits, self.group_size, self.mode = parts[0].bits, group, "affine"
        self.weight = mx.concatenate([p.weight for p in parts])
        self.scales = mx.concatenate([p.scales for p in parts])
        self.biases = mx.concatenate([p.biases for p in parts])
        self.__dict__["member_rows"] = [p.rows for p in parts]
        self.freeze()
        mx.eval(self.weight, self.scales, self.biases)


class _StackPlan:
    """Read the owning module's current projections at each prefill operation."""

    def __init__(self, owner: Any, names: tuple[str, ...]) -> None:
        self.owner, self.names = owner, names

    def current(self):
        linears = [getattr(self.owner, name) for name in self.names]
        if len({int(linear.bits) for linear in linears}) != 1:
            return None
        return _stacked(linears)[0]


def _stacked(linears: list[Any]) -> tuple[Any, list[int]]:
    """Stack current affine values, preserving bit-width groups and member order."""

    by_bits: dict[int, list[int]] = {}
    for i, linear in enumerate(linears):
        by_bits.setdefault(int(linear.bits), []).append(i)
    if len(by_bits) > 1:
        return _Split([(_stacked([linears[i] for i in members])[0], members)
                       for members in by_bits.values()]), []
    group = min(int(linear.group_size) for linear in linears)
    parts = [base.QWeights.of(linear).widened(int(linear.bits), group) for linear in linears]
    stacked = _PreparedLinear(parts, group)
    cuts, at = [], 0
    for part in parts:
        at += part.rows
        cuts.append(at)
    return stacked, cuts[:-1]


def _dense_rows(linear: Any) -> mx.array:
    """A router linear's rows in bf16 (a quantized one dequantized): one matvec takes them all."""

    if isinstance(linear, nn.QuantizedLinear):
        return mx.dequantize(linear.weight, linear.scales, linear.biases, group_size=linear.group_size,
                             bits=linear.bits).astype(mx.bfloat16)
    return linear.weight.astype(mx.bfloat16)


def first(a: mx.array) -> mx.array:
    """Return a view of the leading row; MLX's a[0] gathers and copies it."""

    return a.reshape(a.shape[1:])


# "lane": lane_qmm; "rows": per-row kernels; "simd": simd_qmm, selected by TF_FLASH_DENSE.
DENSE = os.environ.get("TF_FLASH_DENSE") or ("lane" if tensor_units() else "rows")


def _lane_project(x: mx.array, linear: Any) -> mx.array:
    """Project the current descriptor values; object identity is not a byte version."""

    weight = linear.weight
    n, bits, group = int(weight.shape[0]), int(linear.bits), int(linear.group_size)
    scales, biases = linear.scales, linear.biases
    if group == 128:
        scales, biases, group = mx.repeat(scales, 2, axis=1), mx.repeat(biases, 2, axis=1), 64
    nt = 64 if (bits == 4 and n % 64 == 0) else 32 if n % 32 == 0 else 0
    tiled = lane_qmm.tile_weight(weight, nt, group, bits=bits) if nt else weight
    sbt = lane_qmm.pack_scales(scales, biases)
    mx.eval(tiled, sbt)
    k = int(x.shape[-1])
    rows = x.size // k
    kwargs = {"tiled": bool(nt), "nt": nt or lane_qmm.NT, "group": group}
    if rows <= lane_qmm.MAX_ROWS:
        return lane_qmm.lane_matmul(x, tiled, sbt, **kwargs)
    flat = x.reshape(rows, k)
    parts = [lane_qmm.lane_matmul(flat[i:i + lane_qmm.MAX_ROWS], tiled, sbt, **kwargs)
             for i in range(0, rows, lane_qmm.MAX_ROWS)]
    return mx.concatenate(parts).reshape(*x.shape[:-1], -1)


def unreadable(*models: Any) -> dict[str, int]:
    """Quantized linears, by kind, whose width, group or mode the lane matmul does not read (shapes are not checked)."""

    counts: dict[str, int] = {}
    for model in (x for x in models if x is not None):
        for _, m in model.named_modules():
            mode = getattr(m, "mode", "affine")
            group = 64 if getattr(m, "group_size", 0) == 128 else getattr(m, "group_size", 0)   # split as 2 x 64
            if isinstance(m, nn.QuantizedLinear) and (mode != "affine" or not lane_qmm.reads(m.bits, group)):
                kind = f"{m.bits}-bit g{m.group_size}" + ("" if mode == "affine" else f" {mode}")
                counts[kind] = counts.get(kind, 0) + 1
    return counts


def project(x: mx.array, linear: Any) -> mx.array:
    """x [..., R, K] through an affine linear, a row's bits independent of R: per-row kernels before M5, lane_qmm on M5."""

    if not isinstance(linear, nn.QuantizedLinear):
        return linear(x)
    if DENSE == "lane":
        return _lane_project(x, linear)
    if (linear.bits, linear.group_size) != (4, 32):         # other widths before M5: every row alone, at any count
        return rows.qmv_rows(x, linear)
    if DENSE == "rows":
        return linear(x) if x.size // x.shape[-1] == 1 else rows.qmv_rows(x, linear)
    weight = linear.weight
    shape = (int(weight.shape[0]), int(weight.shape[1]) * 8, int(linear.group_size))
    if not simd_qmm.fits(linear):
        raise ValueError(f"simd_qmm does not take a {shape} linear ({linear.bits}-bit, group {linear.group_size})")
    # Public MLX arrays can change their descriptor without changing identity.
    # A current-weight check cannot authorize another matrix of the same shape.
    kind = None if simd_qmm.check(weight, linear.scales, linear.biases,
                                  group_size=linear.group_size) else "mma"
    return simd_qmm.qmm(x, weight, linear.scales, linear.biases, linear.group_size, kind=kind)


_NONE: tuple[str, tuple[mx.array, ...], Any] = ("none", (), None)


class _HC:
    """A hyper-connection's weights for hc_down / hc_up."""

    def __init__(self, conn: Any, *, inject: bool) -> None:
        parts = [conn.input_mix_weight_down] + ([conn.block_inject_weight] if inject else [])
        self.down = base.QWeights.of(*parts)
        mx.eval(self.down.weight, self.down.scales, self.down.biases)
        self.up = base.QWeights.of(conn.input_mix_weight_up)
        self.scale = (1.0 + conn.hc_norm.weight.astype(mx.float32))
        self.low = int(conn.input_mix_weight_down.weight.shape[0])
        mx.eval(self.scale)


class FusedDecode:
    """Decode current model parameters through fused kernels and reference caches.

    Parameter mutation/replacement is allowed between completed operations and
    must be serialized with forwards. Derived arrays are operation-owned; MLX
    object identity never serves as an immutable parameter-byte version.
    """

    def __init__(self, model: Any) -> None:
        self.model = model
        # Hints retain modules and resolve their current arrays, never derived bytes.
        for layer in model.layers:
            if layer.is_linear:
                g = layer.linear_attn
                g.__dict__["stacked"] = _StackPlan(g, ("in_proj_qkv", "in_proj_z", "in_proj_b", "in_proj_a"))
            if "ple" in layer:
                embedding = layer.ple.ple_embedding
                embedding.__dict__["fused_tables"] = embed.PleTables(embedding)
        # Keep per-row recurrent states by layer and stream for prefix rollback, plus each stream's first cache.
        self.row_states: dict[int, list[tuple[mx.array, mx.array, int]]] = {}
        self._last_heads: list[Any] = []
        self._pos: tuple[Any, Any] = (None, None)
        # Queue each layer as soon as Python finishes building it.
        self.eval_every = 1

    @property
    def cfg(self):
        return self.model.args

    @property
    def streams(self):
        return self.cfg.hc_count

    @property
    def eps(self):
        return mx.array([self.cfg.rms_norm_eps], dtype=mx.float32)

    @property
    def mixer(self):
        return _HC(self.model.model.hyper_connection_mixer, inject=False)

    def _entry(self, index, *, connections_only=False):
        layer = self.model.layers[index]
        entry = {"attn_hc": _HC(layer.attn_hyper_connection, inject=True),
                 "mlp_hc": _HC(layer.mlp_hyper_connection, inject=True)}
        if connections_only:
            return entry
        if layer.is_linear:
            g = layer.linear_attn
            proj, _ = _stacked([g.in_proj_qkv, g.in_proj_z, g.in_proj_b, g.in_proj_a])
            conv_w = mx.contiguous(g.conv1d.weight[:, :, 0])
            mx.eval(conv_w)
            entry["gdn"] = (proj, conv_w, g)
        else:
            a = layer.self_attn
            proj, _ = _stacked([a.q_proj, a.k_proj, a.v_proj, a.indexer.index_qk_proj])
            scales = [1.0 + n.weight.astype(mx.float32) for n in
                      (a.q_norm, a.k_norm, a.indexer.q_layernorm, a.indexer.k_layernorm)]
            mx.eval(*scales)
            entry["attn"] = (proj, *scales, a)
        moe = layer.mlp
        router_rows = mx.concatenate([_dense_rows(moe.gate), _dense_rows(moe.shared_expert_gate)])
        mx.eval(router_rows)
        entry["moe"] = (moe, router_rows)
        return entry

    @property
    def layers(self):
        """Current-value views for startup warmers; no authority survives mutation."""
        return [self._entry(i) for i in range(len(self.model.layers))]

    def _ple_parts(self, ple):
        if DENSE != "rows" or not all(isinstance(linear, nn.QuantizedLinear)
                                      for linear in (ple.key_proj, ple.value_proj)):
            return None
        kv, _ = _stacked([ple.key_proj, ple.value_proj])
        scales = [1.0 + n.weight.astype(mx.float32) for n in (ple.norm_key, ple.norm_query, ple.norm_conv)]
        conv_w = mx.contiguous(ple.conv1d.weight[:, :, 0]).astype(mx.float32)
        mx.eval(*scales, conv_w)
        return kv, *scales, conv_w

    # -- blocks ------------------------------------------------------------------
    def _gdn(self, index: int, x: mx.array, cache: Any, entry=None) -> mx.array:
        proj, conv_w, g = (self._entry(index) if entry is None else entry)["gdn"]
        cfg = self.cfg
        rows = x.shape[0]
        conv_state = first(cache.conv) if cache.conv is not None else mx.zeros(
            (cfg.linear_conv_kernel_dim - 1, conv_w.shape[0]), dtype=x.dtype)
        ssm_state = first(cache.ssm) if cache.ssm is not None else None
        out, conv_rows, ssm_rows = gdn.gdn_step(project(x, proj), conv_state, ssm_state, conv_w, g.A_log, g.dt_bias,
                                              g.norm.weight, self.eps, nk=cfg.linear_num_key_heads,
                                              nv=cfg.linear_num_value_heads, dk=cfg.linear_key_head_dim,
                                              dv=cfg.linear_value_head_dim)
        cache.conv, cache.ssm = conv_rows[rows - 1:rows], ssm_rows[rows - 1:rows]
        cache.offset += rows
        self.row_states[index] = [(conv_rows, ssm_rows, 0)]
        return project(out, g.out_proj)

    def _positions(self, past: int, rows: int) -> mx.array:
        """The rows' positions [R] int32, made once per forward (every attention layer is at the same offset)."""

        key = (past, rows)
        if self._pos[0] != key:
            self._pos = (key, mx.arange(past, past + rows, dtype=mx.int32))
        return self._pos[1]

    def _attention(self, index: int, x: mx.array, cache: Any, entry=None) -> mx.array:
        proj, q_scale, k_scale, iq_scale, pool_scale, a = (self._entry(index) if entry is None else entry)["attn"]
        cfg = self.cfg
        rows = x.shape[0]
        heads, kv_heads, dims = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        index_heads, index_dims = cfg.indexer_n_heads, cfg.indexer_head_dim
        past = cache.offset
        p = project(x, proj)
        positions = self._positions(past, rows)
        q, k, iq = attention.attn_prep(p, positions, q_scale, k_scale, iq_scale, self.eps, q_heads=heads,
                               kv_heads=kv_heads, head_dim=dims, index_heads=index_heads, index_dim=index_dims,
                               rotary_dim=cfg.rotary_dim, base=cfg.rope_theta)
        at = heads * 2 * dims
        v = p[:, at + kv_heads * dims: at + 2 * kv_heads * dims].reshape(rows, kv_heads, dims)
        raw_key = p[:, at + 2 * kv_heads * dims + index_heads * index_dims:]
        _, _, raw = cache.update(k.transpose(1, 0, 2)[None], v.transpose(1, 0, 2)[None], raw_key[None])
        ratio, top = cfg.indexer_compress_ratio, a.indexer.top_blocks
        ends = [past + r + 1 for r in range(rows)]
        complete = [e // ratio for e in ends]
        ids = self._select(iq, raw, cache, complete, ends, pool_scale, top) if complete[-1] > top else None
        # Attend to selected blocks and the tail past ``top`` complete blocks, otherwise all keys.
        sparse = [c > top for c in complete]
        counts = [ratio * top + e - ratio * c if sp else e for e, c, sp in zip(ends, complete, sparse)]
        side = getattr(cache, "side", None)
        if side is None:
            gated = attention.attention_rows(q, cache.keys, cache.values, counts, ids, sparse, a.scale, gate=p)
        else:                                           # a chained draft: keys from side_base on are the side's
            gated = attention.attention_rows_split(q, cache.keys, cache.values, side[0], side[1], cache.side_base,
                                                   counts, ids, sparse, a.scale, gate=p)
        return project(gated, a.o_proj)

    def _select(self, iq: mx.array, raw: mx.array, cache: Any, complete: list[int], ends: list[int],
                pool_scale: mx.array, top: int) -> mx.array:
        """The rows' key ids (``kernels.index_select``), after pooling the blocks the last row completes."""

        cfg = self.cfg
        done = 0 if cache.pooled is None else int(cache.pooled.shape[1])
        if complete[-1] > done:
            if raw is None:                             # a chained draft's block: raw keys read from its first row
                rows = cache.side_index_rows(cfg.indexer_compress_ratio * done)
                fresh = attention.index_pool(first(rows), done, complete[-1], pool_scale, self.eps,
                                             rotary_dim=cfg.rotary_dim, base=cfg.rope_theta, relative=True)[None]
            else:
                fresh = attention.index_pool(first(raw), done, complete[-1], pool_scale, self.eps,
                                             rotary_dim=cfg.rotary_dim, base=cfg.rope_theta)[None]
            cache.pooled = fresh if cache.pooled is None else mx.concatenate([cache.pooled, fresh], axis=1)
        return attention.index_select(iq, first(cache.pooled), complete, ends, top=top)

    def _moe(self, index: int, x: mx.array, h: mx.array, inject: mx.array, entry=None) -> tuple[mx.array, Any]:
        """The MoE: the streams and the grouped write-back the next hyper-connection's hc_norm applies."""

        moe, router_rows = (self._entry(index) if entry is None else entry)["moe"]
        cfg = self.cfg
        rows = int(x.shape[0])
        step = base.MAX_ROWS
        split = DENSE == "rows"
        logits = (experts.router(x, router_rows, split=split) if rows <= step else
                  mx.concatenate([experts.router(x[i:i + step], router_rows, split=split)
                                  for i in range(0, rows, step)]))
        sw, se = moe.switch_mlp, moe.shared_expert
        if "streamer" in moe.__dict__:            # routed experts from the slot pool (--ssd-experts)
            from tensorfold.families.qwen4_exp import stream

            return h, (*stream.moe_rows(moe, x, logits, (se.gate_proj, se.up_proj, se.down_proj)), inject)
        act, picks, weights = experts.expert_gateup(x, logits, cfg.num_experts_per_tok, cfg.num_experts, sw.gate_proj,
                                                    sw.up_proj, shared=(se.gate_proj, se.up_proj))
        return h, ("grouped", (experts.expert_down_y(act, picks, sw.down_proj, se.down_proj), weights, logits), inject)

    def _hc(self, h: mx.array, pending: Any, conn: Any) -> tuple[mx.array, mx.array, mx.array]:
        """Write the pending branch into the streams and mix them for the next block: (h_new, mixed, inject gates)."""

        kind, branch, inject = pending
        hn, ssp = hc.hc_norm(h, streams=self.streams, write_back=kind, branch=branch, inject=inject)
        project_hc = rows.hc_project if DENSE == "rows" else hc.hc_project     # before M5: each row its own threadgroups
        mixed, gates = project_hc(hn, ssp, conn.down, conn.up, conn.scale, eps=self.eps, streams=self.streams,
                                  low=conn.low)
        return hn, mixed, gates

    # -- a step --------------------------------------------------------------------
    def __call__(self, tokens: np.ndarray, cache: list[Any]) -> mx.array:
        """Mixed hidden states [1, R, D] after the last layer for R consecutive tokens (batch 1, host ids)."""

        h = embed.embed_rows(tokens.reshape(-1), self.model.model.embed_tokens, tile=self.streams)   # [R, S*D]
        return self.run(h, tokens, cache)

    def run(self, h: mx.array, tokens: np.ndarray | None, cache: list[Any]) -> mx.array:
        """The layers and the final mixer from residual streams h [R, S*D]: mixed hidden states [1, R, D]."""

        model = self.model
        self._last_heads = [cache[0]] if cache else []
        pending = _NONE
        for i, layer in enumerate(model.layers):
            c = cache[i]
            if "ple" in layer:
                h = self._write_back(h, pending)
                pending = _NONE
                host = np.asarray(tokens, dtype=np.int64).reshape(1, -1)      # a GPU window's ids, read after layer 0
                h = self._ple(layer.ple, h, host, c)
            entry = self._entry(i)
            h, mixed, inj = self._hc(h, pending, entry["attn_hc"])
            out = self._gdn(i, mixed, c, entry) if layer.is_linear else self._attention(i, mixed, c, entry)
            h, mixed, inj = self._hc(h, ("plain", (out,), inj), entry["mlp_hc"])
            h, pending = self._moe(i, mixed, h, inj, entry)
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.async_eval(h, *pending[1])
        h, mixed, _ = self._hc(h, pending, self.mixer)
        self.last_streams = h                                     # [R, S*D] before the final mixer (the MTP reads it)
        return mixed[None]

    def _ple(self, ple: Any, h: mx.array, tokens: np.ndarray, cache: Any) -> mx.array:
        """model.PLELayer on rows h [R, S*D], its projections through ``project`` (row-invariant): the new streams."""

        emb_mod = ple.ple_embedding
        history = cache.history
        if history is None:
            history = np.full((1, emb_mod.context), emb_mod.eos, dtype=np.int64)
        ids = emb_mod.ids(history, tokens)
        cache.history = np.concatenate([history, tokens.astype(np.int64)], axis=1)[:, -emb_mod.context:]
        emb = embed.ple_lookup(ids[0], embed.PleTables(emb_mod))                           # [R, E]
        gated, normed = self._ple_gate(ple, emb, h)
        tail = cache.ple_conv if cache.ple_conv is not None else mx.zeros((1, ple.tail, h.shape[-1]), h.dtype)
        conv_in = mx.concatenate([tail, normed[None]], axis=1)
        cache.ple_conv = conv_in[:, -ple.tail:]
        cache.ple_rollback = (history, tokens.astype(np.int64), conv_in)
        return self._ple_conv(ple, conv_in, gated, h)

    def _ple_gate(self, ple: Any, emb: mx.array, h: mx.array) -> tuple[mx.array, mx.array]:
        """(gated, conv-normed) rows [R, S*D]: before M5 two kernels, otherwise the reference ops."""

        parts = self._ple_parts(ple)
        if parts is not None:
            kv, key_scale, query_scale, conv_scale, _ = parts
            return embed.ple_gate(project(emb, kv), h, key_scale, query_scale, conv_scale, self.eps,
                                  streams=ple.streams)
        rows = h.shape[0]
        shape = (rows, ple.streams, ple.dims)
        keys = ple.norm_key(project(emb, ple.key_proj)).reshape(shape)
        values = project(emb, ple.value_proj)
        queries = ple.norm_query(h).reshape(shape)
        gate = mx.sum(keys * queries, axis=-1, keepdims=True) / float(np.sqrt(ple.dims))
        gate = mx.sign(gate) * mx.sqrt(mx.maximum(mx.abs(gate), 1e-6))
        gated = (mx.sigmoid(gate) * values[:, None, :]).reshape(h.shape)
        return gated, ple.norm_conv(gated)

    def _ple_conv(self, ple: Any, conv_in: mx.array, gated: mx.array, h: mx.array) -> mx.array:
        """h + gated + SiLU(conv) for the rows after conv_in's tail [1, T + R, S*D]."""

        parts = self._ple_parts(ple)
        if parts is not None:
            return embed.ple_conv(first(conv_in), parts[-1], gated, h, streams=ple.streams, dilation=ple.dilation)
        return h + (gated + first(nn.silu(ple.conv1d(conv_in))))

    def _write_back(self, h: mx.array, pending: Any) -> mx.array:
        kind, branch, inject = pending
        if kind == "none":
            return h
        # the PLE layer reads the streams themselves: apply the pending write-back first (M5's grouped one)
        return hc.hc_norm(h, streams=self.streams, write_back=kind, branch=branch, inject=inject)[0]

    # -- several streams' rows in one forward --------------------------------------------------------------------
    def run_multi(self, h: mx.array, tokens: list[np.ndarray] | None, caches: list[list[Any]], rows: list[int]
                  ) -> mx.array:
        """Return mixed states [1, N, D] for stream-ordered windows, with each row reading only its own stream's state."""

        if len(rows) == 1:
            return self.run(h, None if tokens is None else tokens[0], caches[0])
        model = self.model
        spans, at = [], 0
        for n in rows:
            spans.append((at, at + int(n)))
            at += int(n)
        self._last_heads = [c[0] for c in caches]
        pending = _NONE
        for i, layer in enumerate(model.layers):
            cs = [c[i] for c in caches]
            if "ple" in layer:
                h = self._write_back(h, pending)
                pending = _NONE
                h = self._ple_multi(layer.ple, h, tokens, cs, spans)
            entry = self._entry(i)
            h, mixed, inj = self._hc(h, pending, entry["attn_hc"])
            out = self._gdn_multi(i, mixed, cs, spans, entry) if layer.is_linear else self._attention_multi(i, mixed, cs, spans, entry)
            h, mixed, inj = self._hc(h, ("plain", (out,), inj), entry["mlp_hc"])
            h, pending = self._moe(i, mixed, h, inj, entry)
            if self.eval_every and (i + 1) % self.eval_every == 0:
                mx.async_eval(h, *pending[1])
        h, mixed, _ = self._hc(h, pending, self.mixer)
        self.last_streams = h
        return mixed[None]

    def _gdn_multi(self, index: int, x: mx.array, caches: list[Any], spans: list[tuple[int, int]], entry=None) -> mx.array:
        proj, conv_w, g = (self._entry(index) if entry is None else entry)["gdn"]
        cfg = self.cfg
        nv, dk, dv = cfg.linear_num_value_heads, cfg.linear_key_head_dim, cfg.linear_value_head_dim
        # the states go to the kernel as stored ([1, ...] row slices of the last call): it reads flat buffers
        convs = [c.conv if c.conv is not None else mx.zeros((cfg.linear_conv_kernel_dim - 1, conv_w.shape[0]),
                                                           dtype=x.dtype) for c in caches]
        ssms = [c.ssm if c.ssm is not None else mx.zeros((nv, dv, dk), dtype=mx.float32) for c in caches]
        projected = project(x, proj)
        outs, states = [], []
        for lo in range(0, len(spans), base.MAX_STREAMS):          # up to 8 streams a kernel call (Metal's buffers)
            group = spans[lo:lo + base.MAX_STREAMS]
            first = group[0][0]
            out, conv_rows, ssm_rows = gdn.gdn_step_multi(
                projected[first:group[-1][1]], convs[lo:lo + base.MAX_STREAMS], ssms[lo:lo + base.MAX_STREAMS],
                [e - s for s, e in group], conv_w, g.A_log, g.dt_bias, g.norm.weight, self.eps,
                nk=cfg.linear_num_key_heads, nv=nv, dk=dk, dv=dv)
            outs.append(out)
            # Keep per-stream views of group states to avoid evaluating a lazy concatenation inside the next forward.
            for c, (s, e) in zip(caches[lo:lo + base.MAX_STREAMS], group):
                c.conv, c.ssm = conv_rows[e - first - 1:e - first], ssm_rows[e - first - 1:e - first]
                c.offset += e - s
                states.append((conv_rows, ssm_rows, s - first))   # sliced only if the stream rolls back
        self.row_states[index] = states
        return project(outs[0] if len(outs) == 1 else mx.concatenate(outs), g.out_proj)

    def _attention_multi(self, index: int, x: mx.array, caches: list[Any], spans: list[tuple[int, int]], entry=None) -> mx.array:
        proj, q_scale, k_scale, iq_scale, pool_scale, a = (self._entry(index) if entry is None else entry)["attn"]
        cfg = self.cfg
        total = int(x.shape[0])
        heads, kv_heads, dims = cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim
        index_heads, index_dims = cfg.indexer_n_heads, cfg.indexer_head_dim
        p = project(x, proj)
        positions = mx.array([c.offset + r for c, (s, e) in zip(caches, spans) for r in range(e - s)], dtype=mx.int32)
        q, k, iq = attention.attn_prep(p, positions, q_scale, k_scale, iq_scale, self.eps, q_heads=heads,
                               kv_heads=kv_heads, head_dim=dims, index_heads=index_heads, index_dim=index_dims,
                               rotary_dim=cfg.rotary_dim, base=cfg.rope_theta)
        at = heads * 2 * dims
        v = p[:, at + kv_heads * dims: at + 2 * kv_heads * dims].reshape(total, kv_heads, dims)
        raw_key = p[:, at + 2 * kv_heads * dims + index_heads * index_dims:]
        ratio, top = cfg.indexer_compress_ratio, a.indexer.top_blocks
        counts, sparse, srow, ends_all, complete_all = [], [], [], [], []
        keys, values, pooled = [], [], []
        for b, (c, (s, e)) in enumerate(zip(caches, spans)):
            past = c.offset
            _, _, raw = c.update(k[s:e].transpose(1, 0, 2)[None], v[s:e].transpose(1, 0, 2)[None], raw_key[s:e][None])
            ends = [past + r + 1 for r in range(e - s)]
            complete = [end // ratio for end in ends]
            if complete[-1] > top:
                done = 0 if c.pooled is None else int(c.pooled.shape[1])
                if complete[-1] > done:
                    fresh = attention.index_pool(first(raw), done, complete[-1], pool_scale, self.eps,
                                         rotary_dim=cfg.rotary_dim, base=cfg.rope_theta)[None]
                    c.pooled = fresh if c.pooled is None else mx.concatenate([c.pooled, fresh], axis=1)
            row_sparse = [cc > top for cc in complete]
            counts += [ratio * top + end - ratio * cc if sp else end for end, cc, sp in zip(ends, complete, row_sparse)]
            sparse += row_sparse
            srow += [b] * (e - s)
            ends_all += ends
            complete_all += complete
            keys.append(c.keys)
            values.append(c.values)
            pooled.append(first(c.pooled) if c.pooled is not None else None)
        outs = []
        for lo in range(0, len(spans), base.MAX_STREAMS):          # up to 8 streams a kernel call (Metal's buffers)
            hi = min(len(spans), lo + base.MAX_STREAMS)
            r0, r1 = spans[lo][0], spans[hi - 1][1]
            rel = [b - lo for b in srow[r0:r1]]
            ids = None
            if any(sparse[r0:r1]):
                group_pooled = pooled[lo:hi]
                filler = next(pl for pl in group_pooled if pl is not None)
                ids = attention.index_select_multi(iq[r0:r1], [pl if pl is not None else filler[:1] for pl in group_pooled],
                                            rel, complete_all[r0:r1], ends_all[r0:r1], top=top)
            outs.append(attention.attention_rows_multi(q[r0:r1], keys[lo:hi], values[lo:hi], rel, counts[r0:r1], ids,
                                                sparse[r0:r1], a.scale, gate=p[r0:r1]))
        gated = outs[0] if len(outs) == 1 else mx.concatenate(outs)
        return project(gated, a.o_proj)

    def _ple_multi(self, ple: Any, h: mx.array, tokens: list[np.ndarray], caches: list[Any],
                   spans: list[tuple[int, int]]) -> mx.array:
        """Run PLE across streams using each stream's own token history and convolution tail: the new streams."""

        emb_mod = ple.ple_embedding
        histories, ids = [], []
        for c, t in zip(caches, tokens):
            history = c.history
            if history is None:
                history = np.full((1, emb_mod.context), emb_mod.eos, dtype=np.int64)
            histories.append(history)
            ids.append(emb_mod.ids(history, t)[0])
        emb = embed.ple_lookup(np.concatenate(ids), embed.PleTables(emb_mod))
        gated, normed = self._ple_gate(ple, emb, h)
        outs = []
        for c, t, history, (s, e) in zip(caches, tokens, histories, spans):
            c.history = np.concatenate([history, t.astype(np.int64)], axis=1)[:, -emb_mod.context:]
            tail = c.ple_conv if c.ple_conv is not None else mx.zeros((1, ple.tail, h.shape[-1]), h.dtype)
            conv_in = mx.concatenate([tail, normed[s:e][None]], axis=1)
            c.ple_conv = conv_in[:, -ple.tail:]
            c.ple_rollback = (history, t.astype(np.int64), conv_in)
            outs.append(self._ple_conv(ple, conv_in, gated[s:e], h[s:e]))
        return mx.concatenate(outs)

    def keep_rows(self, cache: list[Any], rows: int, keep: int) -> None:
        """After a call on ``rows`` rows, make ``cache`` hold only its first ``keep`` rows."""

        if keep == rows:
            return
        drop = rows - keep
        ratio = self.cfg.indexer_compress_ratio
        stream = next((b for b, head in enumerate(self._last_heads) if head is cache[0]), None)
        if stream is None:
            raise ValueError("keep_rows: this cache was not in the last forward")
        for i, layer in enumerate(self.model.layers):
            c = cache[i]
            if layer.is_linear:
                conv_rows, ssm_rows, at = self.row_states[i][stream]
                c.conv, c.ssm = conv_rows[at + keep - 1:at + keep], ssm_rows[at + keep - 1:at + keep]
                c.offset -= drop
                if "ple" in layer:
                    history, tokens, conv_in = c.ple_rollback
                    ple = layer.ple
                    c.history = np.concatenate([history, tokens[:, :keep]], axis=1)[:, -ple.ple_embedding.context:]
                    c.ple_conv = conv_in[:, keep:keep + ple.tail]
            else:
                c.trim(drop, ratio)

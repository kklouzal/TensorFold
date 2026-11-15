"""Gemma 4 26B-A4B target on one TP8/ETP4 rank (SM70), profile gemma4-sm70-w4a16-v1.

Every rank runs the identical operation sequence on identical inputs; the leader alone samples. Decode and
verification forwards (rows <= DECODE_ROWS) use the native W4A16 GEMVs with FP32 activations; prompt
forwards use exact FP16 dequantized weights on cuBLAS tensor cores (FP32 accumulation and output) with
FP16 activations. Both share the norm, RoPE/KV, routing and attention kernels. The residual stream is FP32.
Two cross-rank reductions per layer (attention output, [dense | routed] FFN partials): decode rows all-gather
the FP32 partials and the tail kernels sum them in rank order; prompt rows use an FP32 NCCL all-reduce.
"""

from __future__ import annotations

import ctypes
import math
from dataclasses import dataclass

import torch

from tensorfold.cuda.sm70.gemma4_native import extension
from ..cuda.weights import TAP_LAYERS

DECODE_ROWS = 16
PROMPT_ROWS = 1024          # rows of one prompt forward (sliding ring = window + PROMPT_ROWS)
CHUNK = 256                 # attention key chunk (native AC)


class AllReduce:
    """In-place FP32 sum over the cohort via the communicator's NCCL handle (no comm.py change)."""

    def __init__(self, comm):
        self.comm = comm
        fn = comm.lib.ncclAllReduce
        fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                       ctypes.c_void_p, ctypes.c_void_p]
        fn.restype = ctypes.c_int
        self.fn = fn

    def __call__(self, t: torch.Tensor):
        if t.dtype != torch.float32 or not t.is_contiguous():
            raise ValueError("all-reduce requires contiguous FP32")
        stream = torch.cuda.current_stream(t.device).cuda_stream
        code = self.fn(t.data_ptr(), t.data_ptr(), t.numel(), 7, 0, self.comm.comm, stream)
        if code != 0:
            self.comm._check(code)


@dataclass
class State:
    """One sequence's caches on this rank. Positions >= length are scratch; rollback only moves length."""
    context: int
    length: int
    pos: torch.Tensor                 # int32[1]: first position of the current forward
    sliding_k: list
    sliding_v: list
    sliding_pos: list
    full_k: list
    full_v: list

    def reset(self):
        self.length = 0
        for p in self.sliding_pos:
            p.fill_(-1)


class Target:
    def __init__(self, weights, comm, *, context: int):
        ext = extension()
        self.ext = ext
        self.w = weights
        self.c = c = weights.config
        self.comm = comm
        self.rank, self.world = weights.rank, weights.world
        self.device = weights.embed.device
        self.all_reduce = AllReduce(comm) if comm is not None else (lambda t: None)
        self.context = context
        self.ring = c.window + PROMPT_ROWS
        if self.ring % CHUNK:
            raise ValueError("sliding ring must be a multiple of the attention chunk")
        self.tap_layers = tuple(i for i in TAP_LAYERS if i < c.layers)
        self.embed_scale = float(torch.tensor(c.hidden ** 0.5, dtype=torch.bfloat16))
        self.router_root = float(torch.tensor(c.hidden ** -0.5, dtype=torch.bfloat16))
        self.inverse = {}
        for kind, d, base, rotated in (("sliding", c.head_dim, c.rope_theta, c.head_dim),
                                       ("full", c.global_head_dim, c.global_rope_theta,
                                        int(c.global_head_dim * c.global_rotary_fraction))):
            exponent = torch.arange(0, d, 2, dtype=torch.float32, device=self.device) / d
            inv = torch.pow(torch.tensor(base, dtype=torch.float32, device=self.device), -exponent)
            inv[rotated // 2:] = 0
            self.inverse[kind] = inv.contiguous()
        self.shard = c.vocab // self.world
        self._scratch = {}
        self.frozen = False        # set after graph capture: captured scratch must never be reallocated
        self.router_counter = torch.zeros(1, dtype=torch.int32, device=self.device)   # fused router's last-block count

    # ------------------------------------------------------------------------------------------ state
    def make_state(self, context: int | None = None) -> State:
        context = self.context if context is None else context
        alloc = -(-context // CHUNK) * CHUNK
        c, dev = self.c, self.device
        sk, sv, sp, fk, fv = [], [], [], [], []
        for layer in self.w.layers:
            if layer.sliding:
                sk.append(torch.zeros((self.ring, c.head_dim), dtype=torch.float16, device=dev))
                sv.append(torch.zeros((self.ring, c.head_dim), dtype=torch.float16, device=dev))
                sp.append(torch.full((self.ring,), -1, dtype=torch.int32, device=dev))
            else:
                fk.append(torch.zeros((alloc, c.global_head_dim), dtype=torch.float16, device=dev))
                fv.append(torch.zeros((alloc, c.global_head_dim), dtype=torch.float16, device=dev))
        return State(context, 0, torch.zeros(1, dtype=torch.int32, device=dev), sk, sv, sp, fk, fv)

    # ------------------------------------------------------------------------------------------ helpers
    def _buf(self, name, shape, dtype=torch.float32):
        key = (name, dtype)
        t = self._scratch.get(key)
        n = math.prod(shape)
        if t is None or t.numel() < n or t.dtype != dtype:
            if self.frozen and t is not None:
                raise RuntimeError(f"Gemma 4 scratch {name} would grow after CUDA graph capture")
            t = torch.empty(max(n, 1), dtype=dtype, device=self.device)
            self._scratch[key] = t
        return t[:n].view(shape)

    def _attention(self, q, rows, heads, d, k, v, key_pos, slots, window, out, *, splits=None, noncausal=False,
                   pos=None, scale=1.0):
        R = rows * heads
        rpad = -(-R // 16) * 16
        chunks = -(-slots // (64 if R <= 16 else CHUNK))    # the native kernel's keys per chunk for this tile count
        if splits is None:
            tiles = rpad // 16
            splits = max(1, min(chunks, -(-160 // tiles)))
        po = self._buf("attn_po", (splits, rpad, d))
        pm = self._buf("attn_pm", (splits, rpad))
        pl = self._buf("attn_pl", (splits, rpad))
        self.ext.attn(q, R, heads, d, k, v, key_pos, slots, self._state.pos if pos is None else pos, window,
                      1 if noncausal else 0, scale, chunks, splits, po, pm, pl, rpad, out)

    def _reduce(self, part, name, prompt):
        """Cross-rank sum source for a tail kernel: (tensor, world). Decode rows all-gather the FP32 partials
        (small-collective path when configured) and the tail sums them in rank order; prompt rows all-reduce."""
        if self.comm is None:
            return part, 1
        if prompt:
            self.all_reduce(part)
            return part, 1
        parts = self._buf(name, (self.world,) + tuple(part.shape))
        self.comm.all_gather(part, parts)
        return parts, self.world

    def _matmul_prompt(self, x16, packed, out=None):
        """FP16 rows x exact dequantized weights (tensor cores, FP32 out) x per-row multiplier."""
        n, k = packed.n, packed.k
        w16 = self._buf("dequant", (n, k), torch.float16)
        self.ext.w4_dequant(packed.codes, packed.scales, n, k, w16)
        y = torch.mm(x16, w16.t(), out_dtype=torch.float32)
        y.mul_(packed.rowmul)
        if out is not None:
            out.copy_(y)
            return out
        return y

    # ------------------------------------------------------------------------------------------ forward
    @torch.inference_mode()
    def forward(self, ids: torch.Tensor, state: State, *, logits: str = "all", taps: bool = False,
                embeddings: torch.Tensor | None = None, staged: bool = False):
        """Commit rows ids at state.length. logits: 'all' | 'last' | 'none' (this rank's vocab shard, softcapped).

        Returns (local_logits [n, shard] or None, taps [rows, 6, hidden] FP32 or None). Outputs are views of
        target-owned scratch, valid until the next forward. ``staged`` (CUDA graph capture/replay): the caller
        writes state.pos before each launch and full-attention grids cover the whole allocated context, so one
        graph per row count serves every position."""
        rows = int(ids.numel())
        if not 0 < rows <= PROMPT_ROWS or state.length + rows > state.context:
            raise ValueError("Gemma 4 forward exceeds its row or context admission")
        self._state = state
        if not staged:
            state.pos.fill_(state.length)
        c, ext = self.c, self.ext
        D = c.hidden
        h = self._buf("h", (rows, D))
        if embeddings is None:
            ext.embed(ids.to(torch.int64).contiguous(), rows, self.w.embed, D, self.embed_scale, h)
        else:
            h.copy_(embeddings)
        prompt = rows > DECODE_ROWS
        xdt = torch.float16 if prompt else torch.float32
        x = self._buf("x", (rows, D), xdt)
        ext.rms(h, rows, self.w.layers[0].norms["input"], D, c.eps, 1.0, x)
        tap_out = self._buf("taps", (rows, len(self.tap_layers), D)) if taps else None
        red = self._buf("red", (rows, 2, D))
        o_red = self._buf("o_red", (rows, D))
        dense_in = self._buf("dense_in", (rows, D), xdt)
        expert_in = self._buf("expert_in", (rows, D), xdt)
        router_in = self._buf("router_in", (rows, D))
        logits_e = self._buf("router_logits", (rows, c.experts))
        ids_sel = self._buf("route_ids", (rows, c.top_k), torch.int32)
        weights_sel = self._buf("route_w", (rows, c.top_k))
        local_sel = self._buf("route_local", (rows, c.top_k), torch.int32)
        si = fi = 0
        for i, layer in enumerate(self.w.layers):
            n = layer.norms
            if layer.sliding:
                d, kc, vc, kpos = c.head_dim, state.sliding_k[si], state.sliding_v[si], state.sliding_pos[si]
                si += 1
                inv, ring, window, slots = self.inverse["sliding"], self.ring, c.window, self.ring
            else:
                d, kc, vc, kpos = c.global_head_dim, state.full_k[fi], state.full_v[fi], None
                fi += 1
                inv, ring, window = self.inverse["full"], 0, 0
                # Decode/verify rows always span the allocated cache, as graph replays do: the split partition
                # (keys per split) then equals the staged one, so eager and graph forwards are bitwise equal;
                # splits wholly after every query exit at once.
                slots = kc.shape[0] if staged or not prompt else -(-(state.length + rows) // CHUNK) * CHUNK
            proj_n = layer.qkv.n
            if prompt:
                proj = self._matmul_prompt(x, layer.qkv)
            else:
                proj = self._buf("proj", (rows, proj_n))
                ext.w4_rows(x, rows, layer.qkv.codes, layer.qkv.scales, layer.qkv.rowmul, proj_n, layer.qkv.k,
                            proj, proj_n, False)
            q = self._buf("q", (rows, 2, d), torch.float16)
            ext.qkv_prep(proj, rows, 2, d, 1 if layer.sliding else 0, n["q"], n["k"], inv, state.pos, q, kc, vc,
                         kpos, ring, c.eps)
            att = self._buf("att", (rows, 2 * d), xdt)
            self._attention(q, rows, 2, d, kc, vc, kpos, slots, window, att)
            if prompt:
                self._matmul_prompt(att, layer.o, o_red)
            else:
                ext.w4_rows(att, rows, layer.o.codes, layer.o.scales, layer.o.rowmul, D, layer.o.k, o_red, D,
                            False)
            o_src, o_world = self._reduce(o_red, "o_parts", prompt)
            ext.attn_tail(o_src, rows, o_world, rows * D, h, n["post_attn"], n["pre_ff"], n["pre_ff2"], n["router_scale"],
                          self.router_root, c.eps, dense_in, expert_in, router_in)
            dense_out, routed_out = red[:, 0], red[:, 1]
            width = self.w.dense_width
            # routing (identical on every rank)
            if prompt:
                torch.mm(router_in, layer.router.float().t(), out=logits_e)
                ext.route(logits_e, rows, c.experts, c.top_k, n["expert_scale"], self.w.expert_first,
                          self.w.expert_count, ids_sel, weights_sel, local_sel)
            else:
                ext.router(router_in, rows, layer.router, c.experts, D, logits_e, self.router_counter, c.top_k,
                           n["expert_scale"], self.w.expert_first, self.w.expert_count, ids_sel, weights_sel,
                           local_sel)
            if prompt:
                gu = self._matmul_prompt(dense_in, layer.gate_up)
                act = (torch.nn.functional.gelu(gu[:, :width], approximate="tanh") * gu[:, width:]).half()
                dense_out.copy_(self._matmul_prompt(act, layer.down))
                self._experts_prompt(layer, expert_in, local_sel, weights_sel, routed_out, rows)
            else:
                act = self._buf("dense_act", (rows, width))
                ext.w4_rows(dense_in, rows, layer.gate_up.codes, layer.gate_up.scales, layer.gate_up.rowmul,
                            width, layer.gate_up.k, act, width, True)
                ext.w4_rows(act, rows, layer.down.codes, layer.down.scales, layer.down.rowmul, D, width,
                            dense_out, 2 * D, False)
                ew = self.w.expert_width
                eact = self._buf("expert_act", (rows * c.top_k, ew))
                eg = layer.experts_gate_up
                ext.w4_expert_gateup(expert_in, rows * c.top_k, c.top_k, local_sel, eg.codes, eg.scales, eg.rowmul,
                                     ew, D, eact)
                ed = layer.experts_down
                ext.w4_expert_down(eact, rows, c.top_k, local_sel, weights_sel, ed.codes, ed.scales, ed.rowmul, D,
                                   ew, routed_out, 2 * D)
            red_src, red_world = self._reduce(red, "red_parts", prompt)
            following = self.w.layers[i + 1].norms["input"] if i + 1 < len(self.w.layers) else self.w.norm
            tap = None
            if tap_out is not None and i in self.tap_layers:
                tap = tap_out[:, self.tap_layers.index(i)]
            if i + 1 == len(self.w.layers) and prompt:   # the head reads the final norm in FP32 on both paths
                x = self._buf("x_final", (rows, D))
            ext.ffn_tail(red_src, rows, red_world, rows * 2 * D, h, n["post_ff1"], n["post_ff2"], n["post_ff"],
                         n["scalar"], following, c.eps, x, tap)
        state.length += rows
        out = None
        if logits != "none":
            final = x if logits == "all" else x[rows - 1:]
            m = final.shape[0]
            out = self._buf("logits", (m, self.shard))
            for start in range(0, m, DECODE_ROWS):
                stop = min(m, start + DECODE_ROWS)
                ext.bf16_rows(final[start:stop].contiguous(), stop - start, self.w.head, self.shard, D,
                              out[start:], self.shard, c.softcap)
        return out, tap_out

    def _experts_prompt(self, layer, x16, local, weights, out, rows):
        """Prompt experts: one batch for short chunks, otherwise groups of sixteen.

        One host count read per layer prepares bounded group geometry. Original expert IDs and stable
        within-expert pair order survive packing; the final combine remains in original top-k slot order.
        """
        c = self.c
        ew = self.w.expert_width
        D = c.hidden
        E = self.w.expert_count
        pairs = rows * c.top_k
        flat = local.reshape(-1).long()
        valid = flat >= 0
        counts = torch.bincount(flat[valid], minlength=E)
        if rows < 64:
            # Below the observed 64-row GEMM tile, keep the original geometry
            # and avoid extra group launches and packing metadata transfers.
            width = max(1, int(counts.max().item()))
            groups, bases, packed_rows = [(0, E, width, 0)], None, E * width
        else:
            populations = counts.cpu().tolist()
            groups, bases, packed_rows = [], [], 0
            for first in range(0, E, 16):
                end = min(first + 16, E)
                width = max(populations[first:end])
                groups.append((first, end, width, packed_rows))
                bases.extend(packed_rows + expert * width for expert in range(end - first))
                packed_rows += (end - first) * width
        order = torch.argsort(torch.where(valid, flat, E), stable=True)        # local pairs grouped by expert
        starts = torch.cumsum(counts, 0) - counts
        sorted_e = torch.where(valid, flat, E)[order]
        keep = sorted_e < E
        order, sorted_e = order[keep], sorted_e[keep]
        slot = torch.arange(order.numel(), device=self.device) - starts[sorted_e]
        if bases is None:
            dest = sorted_e * width + slot
        else:
            expert_base = torch.tensor(bases, dtype=torch.int64, device=self.device)
            dest = expert_base[sorted_e] + slot
        index = torch.full((packed_rows,), rows, dtype=torch.long, device=self.device)   # pad -> zero row
        index[dest] = order // c.top_k
        padded = torch.cat([x16, x16.new_zeros((1, D))]).index_select(0, index)
        eg, ed = layer.experts_gate_up, layer.experts_down
        gu16 = self._buf("e_gu16", (E, 2 * ew, D), torch.float16)
        dn16 = self._buf("e_dn16", (E, D, ew), torch.float16)
        self.ext.w4_dequant(eg.codes, eg.scales, E * 2 * ew, D, gu16)
        self.ext.w4_dequant(ed.codes, ed.scales, E * D, ew, dn16)
        # Keep one output allocation, with no group-output concatenation. If this rank has no pairs,
        # a positive dummy extent satisfies the native boundary; every map entry is -1 so none is read.
        y = torch.empty((max(packed_rows, 1), D), dtype=torch.float32, device=self.device)
        for first, end, width, base in groups:
            if not width:
                continue
            stop = base + (end - first) * width
            x_group = padded[base:stop].view(end - first, width, D)
            gu = torch.bmm(x_group, gu16[first:end].transpose(1, 2), out_dtype=torch.float32)
            gu.mul_(eg.rowmul[first:end, None, :])
            act = (torch.nn.functional.gelu(gu[..., :ew], approximate="tanh") * gu[..., ew:]).half()
            torch.bmm(act, dn16[first:end].transpose(1, 2), out_dtype=torch.float32,
                      out=y[base:stop].view(end - first, width, D))
            del gu, act  # same-stream allocator reuse bounds intermediates to one group
        # Each retained original pair occurs once in order; dest is in [0, packed_rows).
        # Missing local pairs stay -1. Gather directly into the slot-ordered sum instead
        # of materializing [rows, top_k, D] and a separate tensor for every arithmetic step.
        pair_dest = torch.full((pairs,), -1, dtype=torch.int64, device=self.device)
        pair_dest[order] = dest
        self.ext.expert_prompt_combine(y, pair_dest.view(rows, c.top_k), weights,
                                      local, ed.rowmul, out)

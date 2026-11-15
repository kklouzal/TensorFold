"""Gemma 4 26B-A4B target on one TP8/ETP4 rank (SM70), profile gemma4-sm70-w4a16-v1.

Every rank runs the identical operation sequence on identical inputs; the leader alone samples. Decode and
verification forwards (rows <= DECODE_ROWS) use the native W4A16 GEMVs with FP32 activations; prompt
forwards use exact FP16 dequantized weights on cuBLAS tensor cores (FP32 accumulation and output) with
FP16 activations. Both share the norm, RoPE/KV, routing and attention kernels. The residual stream is FP32.
Two FP32 NCCL all-reduces per layer: attention output, and [dense | routed] FFN partials.
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
        t = self._scratch.get(name)
        n = math.prod(shape)
        if t is None or t.numel() < n or t.dtype != dtype:
            if self.frozen and t is not None:
                raise RuntimeError(f"Gemma 4 scratch {name} would grow after CUDA graph capture")
            t = torch.empty(max(n, 1), dtype=dtype, device=self.device)
            self._scratch[name] = t
        return t[:n].view(shape)

    def _attention(self, q, rows, heads, d, k, v, key_pos, slots, window, out, *, splits=None, noncausal=False,
                   pos=None):
        R = rows * heads
        rpad = -(-R // 16) * 16
        chunks = -(-slots // CHUNK)
        if splits is None:
            tiles = rpad // 16
            splits = max(1, min(chunks, -(-160 // tiles)))
        po = self._buf("attn_po", (splits, rpad, d))
        pm = self._buf("attn_pm", (splits, rpad))
        pl = self._buf("attn_pl", (splits, rpad))
        self.ext.attn(q, R, heads, d, k, v, key_pos, slots, self._state.pos if pos is None else pos, window,
                      1 if noncausal else 0, chunks, splits, po, pm, pl, rpad, out)

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
                slots = kc.shape[0] if staged else -(-(state.length + rows) // CHUNK) * CHUNK
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
            # routing (identical on every rank)
            if prompt:
                torch.mm(router_in, layer.router.float().t(), out=logits_e)
            else:
                ext.bf16_rows(router_in, rows, layer.router, c.experts, D, logits_e, c.experts, 0.0)
            ext.route(logits_e, rows, c.experts, c.top_k, n["expert_scale"], self.w.expert_first,
                      self.w.expert_count, ids_sel, weights_sel, local_sel)
            dense_out, routed_out = red[:, 0], red[:, 1]
            width = self.w.dense_width
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
            ext.ffn_tail(red_src, rows, red_world, rows * 2 * D, h, n["post_ff1"], n["post_ff2"], n["post_ff"], n["scalar"], following, c.eps,
                         x, tap)
        state.length += rows
        out = None
        if logits != "none":
            final = x if logits == "all" else x[rows - 1:]
            m = final.shape[0]
            if prompt:
                final = final.float()
            out = self._buf("logits", (m, self.shard))
            for start in range(0, m, DECODE_ROWS):
                stop = min(m, start + DECODE_ROWS)
                ext.bf16_rows(final[start:stop].contiguous(), stop - start, self.w.head, self.shard, D,
                              out[start:], self.shard, c.softcap)
        return out, tap_out

    def _experts_prompt(self, layer, x16, local, weights, out, rows):
        """Prompt experts: per local expert GEMMs over its (row, slot) pairs; slot-ordered combine."""
        c = self.c
        ew = self.w.expert_width
        D = c.hidden
        pairs = rows * c.top_k
        y = self._buf("pair_out", (pairs, D))
        y.zero_()
        flat = local.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat[flat >= 0], minlength=self.w.expert_count).cpu().tolist()
        eg, ed = layer.experts_gate_up, layer.experts_down
        gu16 = self._buf("e_gu16", (eg.codes.shape[0], 2 * ew, D), torch.float16)
        dn16 = self._buf("e_dn16", (ed.codes.shape[0], D, ew), torch.float16)
        self.ext.w4_dequant(eg.codes, eg.scales, eg.codes.shape[0] * 2 * ew, D, gu16)
        self.ext.w4_dequant(ed.codes, ed.scales, ed.codes.shape[0] * D, ew, dn16)
        start = int((flat < 0).sum().item())
        for e, n in enumerate(counts):
            if not n:
                continue
            members = order[start:start + n]
            start += n
            xe = x16.index_select(0, members // c.top_k)
            gu = torch.mm(xe, gu16[e].t(), out_dtype=torch.float32).mul_(eg.rowmul[e])
            act = (torch.nn.functional.gelu(gu[:, :ew], approximate="tanh") * gu[:, ew:]).half()
            ye = torch.mm(act, dn16[e].t(), out_dtype=torch.float32).mul_(ed.rowmul[e])
            y.index_copy_(0, members, ye)
        y = y.view(rows, c.top_k, D)
        acc = y[:, 0] * weights[:, 0:1]
        for s in range(1, c.top_k):
            acc = acc + y[:, s] * weights[:, s:s + 1]
        out.copy_(acc)

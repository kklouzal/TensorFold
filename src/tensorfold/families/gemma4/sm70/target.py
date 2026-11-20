"""Gemma 4 26B-A4B target on one TP8/ETP4 rank (SM70), profile gemma4-sm70-w4a16-v1.

Every rank runs the identical operation sequence on identical inputs; the leader alone samples. A forward's
rows are segments of KV slots (one stream each, see Layout/Pool). Decode and verification forwards, of any
number of streams, use the native W4A16 GEMVs with FP32 activations; prompt forwards (one stream's chunk)
use exact FP16 dequantized weights on cuBLAS tensor cores (FP32 accumulation and output) with
FP16 activations. Both share the norm, RoPE/KV, routing and attention kernels. The residual stream is FP32.
Two cross-rank reductions per layer (attention output, [dense | routed] FFN partials): decode rows all-gather
the FP32 partials and the tail kernels sum them in rank order; prompt rows use an FP32 NCCL all-reduce.
"""

from __future__ import annotations

import ctypes
import math
import os

import torch

from tensorfold.cuda.sm70.gemma4_native import extension
from ..cuda.weights import TAP_LAYERS

DECODE_ROWS = 16
PROMPT_ROWS = 1024          # rows of one prompt forward (sliding ring = window + PROMPT_ROWS)
CHUNK = 256                 # attention key chunk (native AC)
EXPERT_MMA_ROWS = 96        # wide rows from which expert-grouped tensor cores beat per-pair GEMVs (measured)
# Decode rows from which forwards take the tensor-core (wide) arithmetic. Off by default: its corpus comparison
# against the bitwise path is inconclusive so far (docs/gemma4-sm70-concurrency.md), so every decode forward keeps
# the C1 row-invariant arithmetic. TF_GEMMA_WIDE_ROWS=N enables it for forwards of at least N rows (experts
# included), e.g. 17 for batches beyond one row tile or 1 to teacher-force a corpus through it at C1.
WIDE_ROWS = 1 << 30
if os.environ.get("TF_GEMMA_WIDE_ROWS"):
    WIDE_ROWS = max(1, int(os.environ["TF_GEMMA_WIDE_ROWS"]))
    EXPERT_MMA_ROWS = max(WIDE_ROWS, EXPERT_MMA_ROWS) if WIDE_ROWS > DECODE_ROWS else WIDE_ROWS


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


class Layout:
    """Rows of one forward, grouped in segments: segment g is one stream's ``rows`` consecutive rows at positions
    ``pos .. pos + rows - 1`` of KV slot ``slot``. Device int32 tables (one buffer, one host-to-device copy):
    row_pos/row_slot per row, the rows' token ids, and per attention head count H the native ``seg`` table
    [6, width] (segment slot, pos, rows, first row; tile segment, tile first local row). A tile is 16 attention
    rows of one segment. Graph layouts keep their shape and restage slots/positions/ids before each replay."""

    def __init__(self, device, rows_per_segment, segments: int, heads=(2,)):
        import numpy as np
        self.n, self.count = int(rows_per_segment), int(segments)
        self.rows = rows = self.n * self.count
        self.heads = tuple(heads)
        self.tiles, self.small, self._seg = {}, {}, {}
        width = {}
        for H in self.heads:
            per = -(-(self.n * H) // 16)
            self.tiles[H] = per * self.count
            self.small[H] = self.n * H <= 16
            width[H] = max(self.count, self.tiles[H])
        total = 3 * rows + sum(6 * width[H] for H in self.heads)
        self.host = np.zeros(total, dtype=np.int32)
        self.dev = torch.zeros(total, dtype=torch.int32, device=device)
        self.row_pos = self.dev[:rows]
        self.row_slot = self.dev[rows:2 * rows]
        self.ids32 = self.dev[2 * rows:3 * rows]
        self._offsets = np.tile(np.arange(self.n, dtype=np.int32), self.count)
        at = 3 * rows
        for H in self.heads:
            w = width[H]
            self._seg[H] = (at, w)
            view = self.host[at:at + 6 * w].reshape(6, w)
            per = -(-(self.n * H) // 16)
            for g in range(self.count):
                view[2, g] = self.n
                view[3, g] = g * self.n
                for t in range(per):
                    view[4, g * per + t] = g
                    view[5, g * per + t] = 16 * t
            at += 6 * w
        self.slots = [0] * self.count
        self.positions = [0] * self.count

    def seg(self, H):
        at, w = self._seg[H]
        return self.dev[at:at + 6 * w].view(6, w)

    def stage(self, slots, positions, ids=None):
        """Host tables for these segments (and the rows' token ids); ``upload`` copies them."""
        import numpy as np
        self.slots, self.positions = list(slots), list(positions)
        n, h, rows = self.n, self.host, self.rows
        pos = np.asarray(positions, dtype=np.int32)
        slot = np.asarray(slots, dtype=np.int32)
        h[:rows] = np.repeat(pos, n) + self._offsets
        h[rows:2 * rows] = np.repeat(slot, n)
        if ids is not None:
            h[2 * rows:3 * rows] = ids
        for H in self.heads:
            at, w = self._seg[H]
            h[at:at + self.count] = slot
            h[at + w:at + w + self.count] = pos

    def upload(self):
        """Pageable source: the copy is staged before return, so the host buffer is reusable at once."""
        self.dev.copy_(torch.from_numpy(self.host), non_blocking=True)


# Full-attention KV formats (--kv-dtype): name -> native format (RotorQuant kernel codec, 0 = FP16) and bytes of one
# 512-wide K or V row (codes, then one four-byte metadata word per 128 values; gemma4.cu kv_row_bytes).
KV_FORMATS = {"fp16": 0, "rotorquant8-norm": 8, "rotorquant-iso64-norm4-h64": 11}


def kv_row_bytes(fmt: int, d: int) -> int:
    return {0: 2 * d, 8: d + 4 * (d // 128), 11: d // 2 + 4 * (d // 128)}[fmt]


def ring_slot_bytes(weights) -> int:
    """Device bytes one slot's target sliding-window rings take on this rank (FP16 K and V, int32 positions; ``Pool``)."""
    c, n = weights.config, sum(layer.sliding for layer in weights.layers)
    return n * (c.window + PROMPT_ROWS) * (2 * 2 * c.head_dim + 4)


class Pool:
    """Every KV slot's caches on this rank. Sliding-window rings exist for every slot from startup ([layers, slots,
    ring, head_dim], ring positions -1 when empty; every sliding layer of a slot holds the same positions);
    full-attention caches are attached per stream with the capacity it may reach. Native kernels address slot
    caches through per-layer int64 address tables; a detached slot's entries are zero and no layout names it.

    Prefix reuse: rows below a kept point stay valid in the full caches (later rows are rewritten before any read);
    the ring rows a point needs (its last ``window`` positions) are copied out by ``snapshot`` right after the prompt
    chunk containing the point (a chunk of PROMPT_ROWS rows never overwrites them: ring = window + PROMPT_ROWS) and
    written back by ``restore``.

    Full caches hold FP16 rows or, with a quantized ``target.kv_format``, uint8 rows of ``kv_row_bytes`` (codes and
    scales together, so attach/grow/copy/host-tier copies stay row slices); rings and snapshots stay FP16."""

    def __init__(self, target, slots: int):
        c, dev = target.c, target.device
        self.target, self.slots = target, slots
        self.fmt = KV_FORMATS[target.kv_format]
        self.row = kv_row_bytes(self.fmt, c.global_head_dim)
        sliding = [layer.sliding for layer in target.w.layers]
        n = sum(sliding)
        self.ring_k = torch.zeros((n, slots, target.ring, c.head_dim), dtype=torch.float16, device=dev)
        self.ring_v = torch.zeros_like(self.ring_k)
        self.ring_p = torch.full((n, slots, target.ring), -1, dtype=torch.int32, device=dev)
        self.sk, self.sv, self.sp = list(self.ring_k.unbind(0)), list(self.ring_v.unbind(0)), list(self.ring_p.unbind(0))
        self.sk_tab = self._table(self.sk)
        self.sv_tab = self._table(self.sv)
        self.sp_tab = self._table(self.sp)
        self.full_layers = sum(1 for s in sliding if not s)
        self.fk_tab = torch.zeros((self.full_layers, slots), dtype=torch.int64, device=dev)
        self.fv_tab = torch.zeros_like(self.fk_tab)
        self.full = [None] * slots
        self.capacity = [0] * slots

    def _table(self, tensors):
        return torch.tensor([[t[s].data_ptr() for s in range(self.slots)] for t in tensors], dtype=torch.int64,
                            device=self.target.device)

    def token_bytes(self) -> int:
        """Full-attention cache bytes one token position adds to a slot."""
        return self.full_layers * 2 * self.row

    def snapshot_bytes(self) -> int:
        """Device bytes of one kept point's ring rows (``snapshot``)."""
        return self.ring_k.shape[0] * (self.target.c.window + 1) * self.target.c.head_dim * 2 * 2

    def _full(self, capacity: int):
        """Zeroed full caches (masked keys of a visited chunk must be finite: zero codes and scales decode to 0)."""
        if self.fmt:
            return torch.zeros((self.full_layers, 2, capacity, self.row), dtype=torch.uint8, device=self.target.device)
        return torch.zeros((self.full_layers, 2, capacity, self.target.c.global_head_dim), dtype=torch.float16,
                           device=self.target.device)

    def _bind(self, slot, t):
        self.full[slot], self.capacity[slot] = t, t.shape[2]
        self.fk_tab[:, slot].copy_(torch.tensor([t[i, 0].data_ptr() for i in range(self.full_layers)]))
        self.fv_tab[:, slot].copy_(torch.tensor([t[i, 1].data_ptr() for i in range(self.full_layers)]))

    def attach(self, slot: int, capacity: int):
        """Allocate (zeroed: masked keys of a visited chunk must be finite) full caches for ``capacity`` rows and
        empty the slot's rings: a new sequence."""
        if capacity % CHUNK:
            raise ValueError("a KV slot's capacity must be a multiple of the attention chunk")
        self._bind(slot, self._full(capacity))
        self.ring_p[:, slot].fill_(-1)

    def grow(self, slot: int, capacity: int, rows: int):
        """Reallocate the slot's full caches for ``capacity`` rows keeping rows [0, ``rows``) (rings untouched)."""
        if capacity % CHUNK or rows > min(capacity, self.capacity[slot]):
            raise ValueError("a KV slot grows to a chunk multiple holding its kept rows")
        old = self.full[slot]
        t = self._full(capacity)
        t[:, :, :rows].copy_(old[:, :, :rows])
        self._bind(slot, t)

    def detach(self, slot: int):
        self.fk_tab[:, slot].zero_()
        self.fv_tab[:, slot].zero_()
        self.full[slot], self.capacity[slot] = None, 0

    def copy_rows(self, src: int, dst: int, rows: int):
        """Full-cache rows [0, ``rows``) of slot ``src`` into slot ``dst`` (a fork; rings come from a snapshot)."""
        if rows > min(self.capacity[src], self.capacity[dst]):
            raise ValueError("a prefix copy exceeds a slot's capacity")
        self.full[dst][:, :, :rows].copy_(self.full[src][:, :, :rows])

    def _rows(self, pos):
        lo = max(0, pos - self.target.c.window - 1)
        return lo, torch.arange(lo, pos, device=self.target.device)

    def snapshot(self, slot: int, pos: int):
        """The ring rows of positions [pos - window - 1, pos) (a point at ``pos`` or ``pos`` - 1 resumes from it)."""
        lo, at = self._rows(pos)
        index = at % self.target.ring
        return (lo, pos, self.ring_k[:, slot].index_select(1, index), self.ring_v[:, slot].index_select(1, index))

    def restore(self, slot: int, snap):
        """Write a snapshot's rows back into the slot's rings; every other ring row is emptied."""
        lo, pos, k, v = snap
        at = torch.arange(lo, pos, device=self.target.device)
        index = at % self.target.ring
        self.ring_k[:, slot].index_copy_(1, index, k)
        self.ring_v[:, slot].index_copy_(1, index, v)
        self.ring_p[:, slot].fill_(-1)
        self.ring_p[:, slot].index_copy_(1, index, at.to(torch.int32).expand(self.ring_p.shape[0], -1).contiguous())


class Target:
    def __init__(self, weights, comm, *, context: int, slots: int = 1, kv_format: str = "fp16"):
        if kv_format not in KV_FORMATS:
            raise ValueError(f"Gemma 4 SM70 full-attention KV formats: {', '.join(KV_FORMATS)}")
        self.kv_format = kv_format
        ext = extension()
        self.ext = ext
        self.w = weights
        self.c = c = weights.config
        self.comm = comm
        self.rank, self.world = weights.rank, weights.world
        self.device = weights.embed.device
        self.all_reduce = AllReduce(comm) if comm is not None else (lambda t: None)
        self.context = context
        self.alloc = -(-(context + DECODE_ROWS) // CHUNK) * CHUNK    # largest slot capacity (decode chunk bound)
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
        # fused router: one last-block counter per 2-row tile (zeroed; each tile's last block resets its own), for
        # up to 32 streams x 16-row blocks
        self.router_counter = torch.zeros(256, dtype=torch.int32, device=self.device)
        self.pool = Pool(self, slots)
        self._head16 = None

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

    def _attention(self, q, lay, heads, d, ktab, vtab, ptab, slots, window, out, *, noncausal=False, scale=1.0,
                   fmt=0):
        """Per-segment split-key attention. ``slots``: keys a segment's chunks may span (its ring, or the linear
        extent). Splits follow the one-stream rule (160 blocks over a segment's tiles), so a segment's arithmetic
        does not depend on the other segments of the launch."""
        small = lay.small[heads]
        tiles = lay.tiles[heads]
        chunks = -(-slots // (64 if small else CHUNK))    # the native kernel's keys per chunk for this tile count
        splits = max(1, min(chunks, -(-160 // (tiles // lay.count))))
        po = self._buf("attn_po", (splits, tiles * 16, d))
        pm = self._buf("attn_pm", (splits, tiles * 16))
        pl = self._buf("attn_pl", (splits, tiles * 16))
        self.ext.attn(q, heads, d, ktab, vtab, ptab, lay.seg(heads), tiles, small, slots, window,
                      1 if noncausal else 0, scale, chunks, splits, po, pm, pl, out, fmt)

    def _reduce(self, part, name, prompt):
        # prompt: also wide decode forwards (rank-order identity with the all-gather is not their contract)
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

    def _project(self, x, rows, packed, n, out, ldo, geglu, wide):
        """Decode-row projection: the row-invariant W4 GEMV (exact C1 arithmetic) or, for wide forwards, the
        tensor-core W4 kernel on hi/lo-split activations."""
        if wide:
            k = packed.k
            self.ext.w4_mma(x, rows, packed.codes, packed.scales, packed.rowmul, n, k, out, ldo, geglu,
                            *self._mma_scratch(rows))
        else:
            self.ext.w4_rows(x, rows, packed.codes, packed.scales, packed.rowmul, n, packed.k, out, ldo, geglu)

    def _mma_scratch(self, rows):
        """hi/lo FP16 activations and split-K partials (<= 8 splits x the widest projection: 8 expert slots x
        GEGLU 2 x 176 x 8 splits per row) for a wide forward of ``rows`` rows."""
        width = max(self.c.hidden, self.c.top_k * self.w.expert_width)
        return (self._buf("x_hi", (rows * width,), torch.float16), self._buf("x_lo", (rows * width,), torch.float16),
                self._buf("mma_part", (rows * 8 * self.c.top_k * 2 * self.w.expert_width,)))

    def head_wide(self, x, out, softcap):
        """Wide rows through this rank's vocabulary shard: cuBLAS FP16 tensor cores on hi/lo-split rows and an exact
        per-row power-of-two scaled FP16 copy of the BF16 head (built once), FP32 accumulation, then the softcap."""
        if self._head16 is None:
            w = self.w.head.float()
            exponent = torch.floor(torch.log2(w.abs().amax(1).clamp_min(2.0 ** -100)))
            scale = torch.pow(2.0, 14 - exponent)
            self._head16 = ((w * scale[:, None]).half().contiguous(), (1.0 / scale).contiguous())
            del w
        w16, inverse = self._head16
        m = x.shape[0]
        hi = x.half()
        both = torch.cat([hi, (x - hi.float()).half()])     # one GEMM reads the head once for hi and lo rows
        y = torch.mm(both, w16.t(), out_dtype=torch.float32)
        torch.mul(y[:m].add_(y[m:]), inverse, out=out)
        if softcap:
            out.div_(softcap).tanh_().mul_(softcap)
        return out

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
    def forward(self, ids: torch.Tensor, lay: Layout, *, prompt: bool, logits: str = "all", taps: bool = False,
                embeddings: torch.Tensor | None = None):
        """Rows ``ids`` at the layout's slots/positions (uploaded by the caller). logits: 'all' | 'last' | 'none'
        (this rank's vocab shard, softcapped).

        ``prompt``: one segment of up to PROMPT_ROWS rows on the prompt arithmetic (FP16 activations, dequantized
        tensor-core GEMMs, all-reduced partials). Otherwise every row takes the decode arithmetic (FP32-activation
        GEMVs, all-gathered partials summed in rank order), identical per row for any number of rows/segments.
        Full-attention decode chunks span the largest slot capacity (later keys are never visited), so one CUDA
        graph per layout shape serves every position. Returns (local_logits [n, shard] or None, taps
        [rows, 6, hidden] FP32 or None), views of target-owned scratch valid until the next forward."""
        rows = lay.rows
        if prompt and (lay.count != 1 or rows > PROMPT_ROWS):
            raise ValueError("Gemma 4 prompt forwards take one segment of at most PROMPT_ROWS rows")
        pool = self.pool
        c, ext = self.c, self.ext
        D = c.hidden
        h = self._buf("h", (rows, D))
        if embeddings is None:
            ext.embed(ids.to(torch.int64).contiguous(), rows, self.w.embed, D, self.embed_scale, h)
        else:
            h.copy_(embeddings)
        xdt = torch.float16 if prompt else torch.float32
        wide = not prompt and rows >= WIDE_ROWS         # tensor-core decode arithmetic (rows of one tile stay exact)
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
                d, kt, vt, pt, fmt = c.head_dim, pool.sk_tab[si], pool.sv_tab[si], pool.sp_tab[si], 0
                si += 1
                inv, ring, window, slots = self.inverse["sliding"], self.ring, c.window, self.ring
            else:
                d, kt, vt, pt, fmt = c.global_head_dim, pool.fk_tab[fi], pool.fv_tab[fi], None, pool.fmt
                fi += 1
                inv, ring, window = self.inverse["full"], 0, 0
                # Decode/verify rows span the largest capacity, eager or graph: the split partition (keys per
                # split) is then fixed, so every forward of a segment shape is bitwise equal; splits wholly after
                # every query exit at once. Prompt chunks span their own keys.
                slots = -(-(lay.positions[0] + rows) // CHUNK) * CHUNK if prompt else self.alloc
            proj_n = layer.qkv.n
            if prompt:
                proj = self._matmul_prompt(x, layer.qkv)
            else:
                proj = self._buf("proj", (rows, proj_n))
                self._project(x, rows, layer.qkv, proj_n, proj, proj_n, False, wide)
            q = self._buf("q", (rows, 2, d), torch.float16)
            ext.qkv_prep(proj, rows, 2, d, 1 if layer.sliding else 0, n["q"], n["k"], inv, lay.row_pos, lay.row_slot,
                         q, kt, vt, pt, ring, c.eps, fmt)
            att = self._buf("att", (rows, 2 * d), xdt)
            self._attention(q, lay, 2, d, kt, vt, pt, slots, window, att, fmt=fmt)
            if prompt:
                self._matmul_prompt(att, layer.o, o_red)
            else:
                self._project(att, rows, layer.o, D, o_red, D, False, wide)
            o_src, o_world = self._reduce(o_red, "o_parts", prompt or wide)
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
                self._project(dense_in, rows, layer.gate_up, width, act, width, True, wide)
                self._project(act, rows, layer.down, D, dense_out, 2 * D, False, wide)
                ew = self.w.expert_width
                eact = self._buf("expert_act", (rows * c.top_k, ew))
                eg, ed = layer.experts_gate_up, layer.experts_down
                if wide and rows >= EXPERT_MMA_ROWS:
                    # pairs grouped by local expert: each expert's weights decode once per 64 routed pairs, a K
                    # range per block
                    E = self.w.expert_count
                    ext.w4_experts_mma(expert_in, rows, c.top_k, local_sel, weights_sel, eg.codes, eg.scales,
                                       eg.rowmul, ed.codes, ed.scales, ed.rowmul, E, ew, D,
                                       self._buf("expert_counts", (E,), torch.int32),
                                       self._buf("expert_lists", (E, rows), torch.int32), eact,
                                       self._buf("expert_y", (rows * c.top_k, D)), routed_out, 2 * D,
                                       *self._mma_scratch(rows))
                else:
                    ext.w4_expert_gateup(expert_in, rows * c.top_k, c.top_k, local_sel, eg.codes, eg.scales,
                                         eg.rowmul, ew, D, eact)
                    ext.w4_expert_down(eact, rows, c.top_k, local_sel, weights_sel, ed.codes, ed.scales, ed.rowmul,
                                       D, ew, routed_out, 2 * D)
            red_src, red_world = self._reduce(red, "red_parts", prompt or wide)
            following = self.w.layers[i + 1].norms["input"] if i + 1 < len(self.w.layers) else self.w.norm
            tap = None
            if tap_out is not None and i in self.tap_layers:
                tap = tap_out[:, self.tap_layers.index(i)]
            if i + 1 == len(self.w.layers) and prompt:   # the head reads the final norm in FP32 on both paths
                x = self._buf("x_final", (rows, D))
            ext.ffn_tail(red_src, rows, red_world, rows * 2 * D, h, n["post_ff1"], n["post_ff2"], n["post_ff"],
                         n["scalar"], following, c.eps, x, tap)
        out = None
        if logits != "none":
            final = x if logits == "all" else x[rows - 1:]
            m = final.shape[0]
            out = self._buf("logits", (m, self.shard))
            if wide:
                self.head_wide(final, out, c.softcap)
            else:
                ext.bf16_rows(final.contiguous(), m, self.w.head, self.shard, D, out, self.shard, c.softcap)
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

"""z-lab Gemma 4 DFlash companion (Qwen3-style, 5 layers) on the TP8 ranks, with the target's head.

TP8 layout: fc output rows 352 per rank (all-gathered), 4 query heads + KV head r per layer, MLP 704 rows per
rank; o_proj/down row-parallel with FP32 all-reduce. Original BF16 weights (decode GEMV reads BF16 directly;
prompt rows use FP32 copies). Context K/V of target taps live in per-slot caches (sliding layers: rings of
RING rows with ring positions; full layers: linear, attached per admitted stream). A proposal block (anchor +
mask rows) writes its own K/V at the positions it covers; the verify's taps of those same positions then
replace them (every verified row, accepted or not: later blocks start inside that range and rewrite the rest
before any read), so no round carries pending state.
The companion receives raw (unscaled) target embeddings and the target's softcapped head (official contract).
Proposals are verified by the target, so this arithmetic never changes the reply law.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..dflash_metadata import checked_headers
from .target import DECODE_ROWS, PROMPT_ROWS

RING = 3072     # sliding drafter layers: window (2048) + PROMPT_ROWS, a multiple of the 256-key chunk: a prompt
#                 chunk containing a kept point leaves the window behind it in the ring (prefix snapshots)


class DFlash:
    def __init__(self, draft_dir, target, *, context: int, block: int, bits: int = 8, slots: int = 1):
        from tensorfold.cuda.direct_read import SafeTensors
        from tensorfold.tensor_file import checkpoint_path

        cfg, _ = checked_headers(Path(draft_dir))
        c = target.c
        if (cfg.hidden, cfg.target_layers, cfg.vocab) != (c.hidden, c.layers, c.vocab):
            raise ValueError("Gemma DFlash hidden width, target layer count and vocabulary must match the target")
        if cfg.tap_layers != target.tap_layers:
            raise ValueError("Gemma DFlash target taps differ from the target's tap layers")
        world, rank = target.world, target.rank
        if cfg.heads % world or cfg.kv_heads != world or cfg.width % world or cfg.hidden % world:
            raise ValueError("Gemma DFlash TP8 layout needs heads/KV heads/MLP/hidden divisible by 8")
        if bits not in (8, 16):
            raise ValueError("Gemma DFlash on SM70 runs BF16 (--drafter-bits 16) or private INT8 W8A16 copies (8)")
        self.bits = bits
        if not 2 <= block <= min(cfg.block, DECODE_ROWS):
            raise ValueError("Gemma DFlash block must be 2..16 rows")
        self.cfg, self.t, self.ext = cfg, target, target.ext
        self.block = block
        self.device = target.device
        self.heads = cfg.heads // world
        self.d = cfg.head_dim
        self.width = cfg.width // world
        self.fc_rows = cfg.hidden // world
        # Native small-row projections index output rows with signed32 integers.
        # Check the new combined extent before any bank allocation; no fixed layer
        # count is imposed beyond the existing metadata and executable size contract.
        if cfg.layers * 2 * cfg.head_dim > 2**31 - 1:
            raise ValueError("Gemma DFlash combined context KV projection exceeds signed32 output rows")
        reader = SafeTensors([checkpoint_path(Path(draft_dir), "model.safetensors")])
        get = lambda name: reader.get(name, "cpu")
        dev = self.device
        H, d = self.heads, self.d
        self.fc = get("fc.weight")[rank * self.fc_rows:(rank + 1) * self.fc_rows].to(dev).contiguous()
        self.hidden_norm = get("hidden_norm.weight").to(dev)
        self.norm = get("norm.weight").to(dev)
        self.layers = []
        for i in range(cfg.layers):
            p = f"layers.{i}."
            q = get(p + "self_attn.q_proj.weight")[rank * H * d:(rank + 1) * H * d]
            k = get(p + "self_attn.k_proj.weight")[rank * d:(rank + 1) * d]
            v = get(p + "self_attn.v_proj.weight")[rank * d:(rank + 1) * d]
            o = get(p + "self_attn.o_proj.weight")[:, rank * H * d:(rank + 1) * H * d]
            g = get(p + "mlp.gate_proj.weight")[rank * self.width:(rank + 1) * self.width]
            u = get(p + "mlp.up_proj.weight")[rank * self.width:(rank + 1) * self.width]
            dn = get(p + "mlp.down_proj.weight")[:, rank * self.width:(rank + 1) * self.width]
            self.layers.append({
                "qkv": torch.cat([q, k, v]).to(dev).contiguous(),
                "o": o.to(dev).contiguous(),
                "gate_up": torch.cat([g, u]).to(dev).contiguous(),
                "down": dn.to(dev).contiguous(),
                "in_norm": get(p + "input_layernorm.weight").to(dev),
                "post_norm": get(p + "post_attention_layernorm.weight").to(dev),
                "q_norm": get(p + "self_attn.q_norm.weight").to(dev),
                "k_norm": get(p + "self_attn.k_norm.weight").to(dev),
                "sliding": cfg.layer_types[i] == "sliding_attention",
                "causal": cfg.causal_layers[i],
            })
        reader.close()
        if bits == 8:
            # Private symmetric per-output-row INT8 copies (proposal arithmetic only; the target verifies).
            self.fc = self._int8(self.fc)
            for layer in self.layers:
                for key in ("qkv", "o", "gate_up", "down"):
                    layer[key] = self._int8(layer[key])
            # The proposal head reads a private per-row INT8 copy of this rank's BF16 vocabulary shard.
            self.head8 = self._int8(target.w.head)
        # Every layer projects the same normalized target context. Keep this bounded
        # immutable row bank alive with DFlash; original QKV weights still serve
        # proposal layers. The actual five-layer bank adds 3.61 MB/rank (bits8,
        # including FP32 scales), or 7.21 MB/rank (bits16).
        self.context_kv = self._pack_context_kv()
        exponent = torch.arange(0, d, 2, dtype=torch.int64).float() / d
        self.inv_freq = (1.0 / (cfg.rope.theta ** exponent)).to(dev).contiguous()
        if any(layer["sliding"] and cfg.sliding_window + max(block, PROMPT_ROWS) > RING for layer in self.layers):
            raise ValueError("Gemma DFlash sliding window exceeds its ring")
        self.alloc = target.alloc
        self.slots = slots
        # per layer: (k, v, positions) ring views [slots, RING, d] of stacked sliding tensors, or None (full layers,
        # attached per slot)
        sliding = [layer["sliding"] for layer in self.layers]
        self.ring_k = torch.zeros((sum(sliding), slots, RING, d), dtype=torch.float16, device=dev)
        self.ring_v = torch.zeros_like(self.ring_k)
        self.ring_p = torch.full((sum(sliding), slots, RING), -1, dtype=torch.int32, device=dev)
        self.window = cfg.sliding_window
        self.rings, j = [], 0
        for flag in sliding:
            self.rings.append((self.ring_k[j], self.ring_v[j], self.ring_p[j]) if flag else None)
            j += flag
        self.k_tab = torch.zeros((len(self.layers), slots), dtype=torch.int64, device=dev)
        self.v_tab = torch.zeros_like(self.k_tab)
        self.p_tab = torch.zeros_like(self.k_tab)
        for i, ring in enumerate(self.rings):
            if ring is not None:
                for s in range(slots):
                    self.k_tab[i, s], self.v_tab[i, s], self.p_tab[i, s] = (ring[0][s].data_ptr(),
                                                                         ring[1][s].data_ptr(), ring[2][s].data_ptr())
        self.full = [None] * slots
        self.mask_id = cfg.mask_id
        self._f32 = {}

    # ------------------------------------------------------------------------------------------- helpers
    @staticmethod
    def _int8(w):
        f = w.float()
        scale = (f.abs().amax(1) / 127).clamp_min(torch.finfo(torch.float32).tiny)
        q = torch.round(f / scale[:, None]).clamp_(-127, 127).to(torch.int8)
        return (q.contiguous(), scale.contiguous())

    @staticmethod
    def _rows(w, lo=0, hi=None):
        # Output-row slice of a BF16 weight or an (INT8, scale) copy.
        if isinstance(w, tuple):
            return (w[0][lo:hi], w[1][lo:hi])
        return w[lo:hi]

    def _pack_context_kv(self):
        """Immutable layer-major K/V rows, retaining the selected weight representation."""
        weights = [self._rows(layer["qkv"], self.heads * self.d) for layer in self.layers]
        if self.bits == 8:
            return (torch.cat([weight[0] for weight in weights]), torch.cat([weight[1] for weight in weights]))
        return torch.cat(weights)

    def _w32(self, key, w):
        t = self._f32.get(key)
        if t is None:
            # _int8 owns FP32 scales; promotion converts each exactly representable
            # INT8 value to FP32 before the same multiplication, without a q.float copy.
            t = self._f32[key] = (w[0] * w[1][:, None]) if isinstance(w, tuple) else w.float()
        return t

    def token_bytes(self) -> int:
        """Linear (full-attention) cache bytes one token position adds to a slot."""
        return sum(ring is None for ring in self.rings) * 2 * self.d * 2

    def snapshot_bytes(self) -> int:
        return self.ring_k.shape[0] * (self.window + 1) * self.d * 2 * 2

    def _bind(self, slot, t):
        self.full[slot] = t
        for j, i in enumerate(i for i, ring in enumerate(self.rings) if ring is None):
            self.k_tab[i, slot], self.v_tab[i, slot] = t[j, 0].data_ptr(), t[j, 1].data_ptr()

    def attach(self, slot: int, capacity: int):
        full = sum(ring is None for ring in self.rings)
        self._bind(slot, torch.zeros((full, 2, capacity, self.d), dtype=torch.float16, device=self.device))
        self.ring_p[:, slot].fill_(-1)

    def grow(self, slot: int, capacity: int, rows: int):
        old = self.full[slot]
        t = torch.zeros((old.shape[0], 2, capacity, self.d), dtype=torch.float16, device=self.device)
        t[:, :, :rows].copy_(old[:, :, :rows])
        self._bind(slot, t)

    def detach(self, slot: int):
        for i, ring in enumerate(self.rings):
            if ring is None:
                self.k_tab[i, slot] = self.v_tab[i, slot] = 0
        self.full[slot] = None

    def copy_rows(self, src: int, dst: int, rows: int):
        self.full[dst][:, :, :rows].copy_(self.full[src][:, :, :rows])

    def snapshot(self, slot: int, pos: int):
        lo = max(0, pos - self.window - 1)
        index = torch.arange(lo, pos, device=self.device) % RING
        return (lo, pos, self.ring_k[:, slot].index_select(1, index), self.ring_v[:, slot].index_select(1, index))

    def restore(self, slot: int, snap):
        lo, pos, k, v = snap
        at = torch.arange(lo, pos, device=self.device)
        index = at % RING
        self.ring_k[:, slot].index_copy_(1, index, k)
        self.ring_v[:, slot].index_copy_(1, index, v)
        self.ring_p[:, slot].fill_(-1)
        self.ring_p[:, slot].index_copy_(1, index, at.to(torch.int32).expand(self.ring_p.shape[0], -1).contiguous())

    def _linear(self, x, w, key, prompt=False):
        rows = x.shape[0]
        if not prompt and rows <= DECODE_ROWS:
            n, k = w[0].shape if isinstance(w, tuple) else w.shape
            out = torch.empty((rows, n), dtype=torch.float32, device=self.device)
            if isinstance(w, tuple):
                self.ext.i8_rows(x.contiguous(), rows, w[0], w[1], n, k, out, n, 0.0)
            else:
                self.ext.bf16_rows(x.contiguous(), rows, w, n, k, out, n, 0.0)
            return out
        return torch.mm(x, self._w32(key, w).t())

    def _reduce(self, part, name, prompt=False):
        """Cross-rank sum of a row-parallel partial: decode rows all-gather (small-collective path) and sum."""
        if prompt or part.shape[0] > DECODE_ROWS:
            self.t.all_reduce(part)
            return part
        parts = self.t._buf(name, (self.t.world,) + tuple(part.shape))
        self.t.comm.all_gather(part, parts)
        return parts.sum(0)

    def _rms(self, x, w):
        out = torch.empty_like(x)
        self.ext.rms(x, x.shape[0], w, x.shape[1], self.cfg.eps, 1.0, out)
        return out

    # ------------------------------------------------------------------------------------------- context
    @torch.inference_mode()
    def append(self, taps: torch.Tensor, lay, *, prompt: bool = False):
        """Commit target taps [rows, 6, hidden] (FP32) at the layout's slots/positions (``prompt`` or more than
        DECODE_ROWS rows: FP32 library GEMMs and an all-reduce; otherwise the row-invariant GEMVs)."""
        rows = taps.shape[0]
        flat = taps.reshape(rows, -1)
        local = self._linear(flat, self.fc, "fc", prompt)
        gathered = torch.empty((self.t.world, rows, self.fc_rows), dtype=torch.float32, device=self.device)
        self.t.comm.all_gather(local.contiguous(), gathered)
        ctx = self._rms(gathered.permute(1, 0, 2).reshape(rows, -1).contiguous(), self.hidden_norm)
        d = self.d
        projected = self._linear(ctx, self.context_kv, "kv", prompt)
        for i, layer in enumerate(self.layers):
            # qkv_prep reads the actual row stride. Hq0 writes only K/V caches,
            # so this slice can also serve as its unused qout without materializing it.
            kv = projected[:, i * 2 * d:(i + 1) * 2 * d]
            ring = self.rings[i]
            self.ext.qkv_prep(kv, rows, 0, d, 2, layer["q_norm"], layer["k_norm"], self.inv_freq, lay.row_pos,
                              lay.row_slot, kv, self.k_tab[i], self.v_tab[i],
                              None if ring is None else self.p_tab[i], 0 if ring is None else RING, self.cfg.eps)

    # ------------------------------------------------------------------------------------------- proposals
    @torch.inference_mode()
    def logits(self, ids, lay):
        """This rank's softcapped head shard [segments * (block - 1), shard] for block rows 1.. of every segment.

        ``ids``: device int64 [segments * block] (anchor, then mask ids, per segment); ``lay``: block-row segments
        at each stream's committed length (uploaded). Attention spans the largest capacity (later keys masked)."""
        cfg, ext, H, d = self.cfg, self.ext, self.heads, self.d
        rows, n = lay.rows, lay.n
        x = torch.empty((rows, cfg.hidden), dtype=torch.float32, device=self.device)
        ext.embed(ids, rows, self.t.w.embed, cfg.hidden, 1.0, x)
        for i, layer in enumerate(self.layers):
            ring = self.rings[i]
            h = self._rms(x, layer["in_norm"])
            qkv = self._linear(h, layer["qkv"], f"qkv{i}")
            q = torch.empty((rows, H, d), dtype=torch.float16, device=self.device)
            ext.qkv_prep(qkv, rows, H, d, 2, layer["q_norm"], layer["k_norm"], self.inv_freq, lay.row_pos,
                         lay.row_slot, q, self.k_tab[i], self.v_tab[i], None if ring is None else self.p_tab[i],
                         0 if ring is None else RING, cfg.eps)
            att = torch.empty((rows, H * d), dtype=torch.float32, device=self.device)
            self.t._attention(q, lay, H, d, self.k_tab[i], self.v_tab[i], None if ring is None else self.p_tab[i],
                              self.alloc if ring is None else RING, cfg.sliding_window if layer["sliding"] else 0,
                              att, noncausal=not layer["causal"], scale=d ** -0.5)
            x = x + self._reduce(self._linear(att, layer["o"], f"o{i}"), "dflash_o")
            h = self._rms(x, layer["post_norm"])
            gu = self._linear(h, layer["gate_up"], f"gu{i}")
            act = torch.nn.functional.silu(gu[:, :self.width]) * gu[:, self.width:]
            x = x + self._reduce(self._linear(act.contiguous(), layer["down"], f"down{i}"), "dflash_down")
        m = lay.count * (n - 1)
        h = self._rms(x.view(lay.count, n, -1)[:, 1:].reshape(m, -1), self.norm)
        out = torch.empty((m, self.t.shard), dtype=torch.float32, device=self.device)
        if m > DECODE_ROWS:             # wide proposal blocks: the target's tensor-core head (original BF16 values)
            self.t.head_wide(h, out, self.t.c.softcap)
        elif self.bits == 8:
            ext.i8_rows(h, m, self.head8[0], self.head8[1], self.t.shard, cfg.hidden, out, self.t.shard,
                        self.t.c.softcap)
        else:
            ext.bf16_rows(h, m, self.t.w.head, self.t.shard, cfg.hidden, out, self.t.shard, self.t.c.softcap)
        return out

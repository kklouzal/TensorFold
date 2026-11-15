"""z-lab Gemma 4 DFlash companion (Qwen3-style, 5 layers) on the TP8 ranks, with the target's head.

TP8 layout: fc output rows 352 per rank (all-gathered), 4 query heads + KV head r per layer, MLP 704 rows per
rank; o_proj/down row-parallel with FP32 all-reduce. Original BF16 weights (decode GEMV reads BF16 directly;
prompt rows use FP32 copies). Context K/V of accepted target taps live in linear per-layer caches; a proposal
block (anchor + mask rows) writes its own K/V beyond the committed length, which later commits overwrite.
The companion receives raw (unscaled) target embeddings and the target's softcapped head (official contract).
Proposals are verified by the target, so this arithmetic never changes the reply law.
"""

from __future__ import annotations

from pathlib import Path

import torch

from ..dflash_metadata import checked_headers
from .target import CHUNK, DECODE_ROWS


class DFlash:
    def __init__(self, draft_dir, target, *, context: int, block: int, bits: int = 8):
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
        exponent = torch.arange(0, d, 2, dtype=torch.int64).float() / d
        self.inv_freq = (1.0 / (cfg.rope.theta ** exponent)).to(dev).contiguous()
        alloc = -(-(context + DECODE_ROWS) // CHUNK) * CHUNK
        self.k = [torch.zeros((alloc, d), dtype=torch.float16, device=dev) for _ in self.layers]
        self.v = [torch.zeros((alloc, d), dtype=torch.float16, device=dev) for _ in self.layers]
        self.alloc = alloc
        self.length = 0
        self.pos = torch.zeros(1, dtype=torch.int32, device=dev)
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

    def _w32(self, key, w):
        t = self._f32.get(key)
        if t is None:
            t = self._f32[key] = (w[0].float() * w[1][:, None]) if isinstance(w, tuple) else w.float()
        return t

    def _linear(self, x, w, key):
        rows = x.shape[0]
        if rows <= DECODE_ROWS:
            n, k = w[0].shape if isinstance(w, tuple) else w.shape
            out = torch.empty((rows, n), dtype=torch.float32, device=self.device)
            if isinstance(w, tuple):
                self.ext.i8_rows(x.contiguous(), rows, w[0], w[1], n, k, out, n)
            else:
                self.ext.bf16_rows(x.contiguous(), rows, w, n, k, out, n, 0.0)
            return out
        return torch.mm(x, self._w32(key, w).t())

    def _rms(self, x, w):
        out = torch.empty_like(x)
        self.ext.rms(x, x.shape[0], w, x.shape[1], self.cfg.eps, 1.0, out)
        return out

    def reset(self):
        self.length = 0

    # ------------------------------------------------------------------------------------------- context
    @torch.inference_mode()
    def append(self, taps: torch.Tensor, start: int, *, pos: torch.Tensor | None = None):
        """Commit target taps [rows, 6, hidden] (FP32) for positions start..start+rows-1.

        ``pos`` (CUDA graphs): a device int32[1] the caller sets to ``start`` before each replay."""
        rows = taps.shape[0]
        if rows == 0:
            return
        if start + rows > self.alloc - DECODE_ROWS:
            raise ValueError("DFlash context exceeds its admitted cache")
        flat = taps.reshape(rows, -1)
        local = self._linear(flat, self.fc, "fc")
        gathered = torch.empty((self.t.world, rows, self.fc_rows), dtype=torch.float32, device=self.device)
        self.t.comm.all_gather(local.contiguous(), gathered)
        ctx = self._rms(gathered.permute(1, 0, 2).reshape(rows, -1).contiguous(), self.hidden_norm)
        if pos is None:
            pos = self.pos
            pos.fill_(start)
        H, d = self.heads, self.d
        for i, layer in enumerate(self.layers):
            kv = self._linear(ctx, self._rows(layer["qkv"], H * d), f"kv{i}")
            self.ext.qkv_prep(kv, rows, 0, d, 2, layer["q_norm"], layer["k_norm"], self.inv_freq, pos, kv,
                              self.k[i], self.v[i], None, 0, self.cfg.eps)
        self.length = start + rows

    # ------------------------------------------------------------------------------------------- proposals
    @torch.inference_mode()
    def logits(self, anchor, count: int, *, pos: torch.Tensor | None = None):
        """This rank's softcapped head shard [n, shard] for block rows 1..n (n = min(count, block - 1)).

        CUDA graphs: ``anchor`` is a device int64[1] and ``pos`` a device int32[1] (the committed length) set
        before each replay; attention then spans the whole allocated cache (later keys are masked)."""
        n = min(count, self.block - 1)
        staged = pos is not None
        if n < 1 or (not staged and self.length == 0):
            return None
        rows = n + 1
        cfg, ext, H, d = self.cfg, self.ext, self.heads, self.d
        ids = torch.full((rows,), self.mask_id, dtype=torch.int64, device=self.device)
        if staged:
            ids[0:1].copy_(anchor)
        else:
            ids[0] = anchor
            pos = self.pos
            pos.fill_(self.length)
        x = torch.empty((rows, cfg.hidden), dtype=torch.float32, device=self.device)
        ext.embed(ids, rows, self.t.w.embed, cfg.hidden, 1.0, x)
        slots = self.alloc if staged else -(-(self.length + rows) // CHUNK) * CHUNK
        for i, layer in enumerate(self.layers):
            h = self._rms(x, layer["in_norm"])
            qkv = self._linear(h, layer["qkv"], f"qkv{i}")
            q = torch.empty((rows, H, d), dtype=torch.float16, device=self.device)
            ext.qkv_prep(qkv, rows, H, d, 2, layer["q_norm"], layer["k_norm"], self.inv_freq, pos, q,
                         self.k[i], self.v[i], None, 0, cfg.eps)
            att = torch.empty((rows, H * d), dtype=torch.float32, device=self.device)
            self.t._attention(q, rows, H, d, self.k[i], self.v[i], None, slots,
                              cfg.sliding_window if layer["sliding"] else 0, att, noncausal=not layer["causal"],
                              pos=pos, scale=d ** -0.5)
            o = self._linear(att, layer["o"], f"o{i}")
            self.t.all_reduce(o)
            x = x + o
            h = self._rms(x, layer["post_norm"])
            gu = self._linear(h, layer["gate_up"], f"gu{i}")
            act = torch.nn.functional.silu(gu[:, :self.width]) * gu[:, self.width:]
            dn = self._linear(act.contiguous(), layer["down"], f"down{i}")
            self.t.all_reduce(dn)
            x = x + dn
        h = self._rms(x[1:].contiguous(), self.norm)
        out = torch.empty((n, self.t.shard), dtype=torch.float32, device=self.device)
        ext.bf16_rows(h, n, self.t.w.head, self.t.shard, cfg.hidden, out, self.t.shard, self.t.c.softcap)
        return out

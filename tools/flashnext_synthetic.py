"""A real-shape synthetic Flash Next (Qwen3.8-Flash-Next) for decode benchmarks on one CUDA GPU.

Random MLX affine 4-bit / group-32 weights at the production dimensions, assembled exactly as
``tensorfold.families.qwen4_exp.cuda.weights.load`` assembles a checkpoint's arrays (the same ``Config``,
``LayerW``/``HC``/``MoEW``/``AttnW``/``GDNW``/``PLEW``/``MTPW`` dataclasses, the same "tiled"/"frag" packings, the same
grouped expert tables with the shared expert last, the same draft head over the packaged draft vocabulary, the same
RoPE policy object) but with fewer layers. ``bench_decode_round.py`` documents what is and is not represented.

The generator follows ``tests/cuda/test_flashnext_forward.py``'s ``_Rand`` (random words; scales in
[s/64, s/64 + s/8]; biases -7.5 x scale, so values sit in about [-s, s]), copied here: tools never import tests.
Expert tables are packed a chunk of experts at a time into preallocated grouped tables (``experts.pack`` packs each
expert independently; ``check_chunked_pack`` proves the chunked table equals ``experts.make`` on a small table), so a
512-expert layer never holds its unpacked MLX arrays at once.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# The production checkpoint's text config (TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP snapshot 2b170fa6,
# copied from ~/tensorfold-perf-20261013/flashnext-config.json). ``--config`` may point at the real file instead.
REAL_CONFIG: dict = {
    "architectures": ["Qwen4ExpForConditionalGeneration"],
    "model_type": "qwen4_exp",
    "eos_token_id": [248046, 248044],
    "quantization": {"group_size": 32, "bits": 4, "mode": "affine"},
    "quantization_config": {"group_size": 32, "bits": 4, "mode": "affine"},
    "tie_word_embeddings": False,
    "text_config": {
        "attention_bias": False, "attention_dropout": 0.0, "bos_token_id": 248044, "dtype": "bfloat16",
        "eos_token_id": 248044, "initializer_range": 0.02, "output_router_logits": False,
        "router_aux_loss_coef": 0.001,
        "full_attention_interval": 4, "hc_count": 4, "hc_lowrank": 320, "head_dim": 256, "heads_per_ngram": 8,
        "hidden_act": "silu", "hidden_size": 2560, "indexer_budget": 2048, "indexer_compress_ratio": 4,
        "indexer_head_dim": 128, "indexer_kv_heads": 1, "indexer_n_heads": 4,
        "layer_types": [t for _ in range(12) for t in ("linear_attention",) * 3 + ("full_attention",)],
        "linear_conv_kernel_dim": 4, "linear_key_head_dim": 128, "linear_num_key_heads": 16,
        "linear_num_value_heads": 48, "linear_value_head_dim": 128, "make_ngram_vocab_size_divisible_by": 128,
        "mamba_ssm_dtype": "float32", "max_position_embeddings": 262144, "model_type": "qwen4_exp_text",
        "moe_intermediate_size": 640,
        "mtp": {"hybrid": True, "layer_types": ["full_attention"], "mtp_use_hidden_state_from_layer": None,
                "num_hidden_layers": 1, "rope_theta": 10000000},
        "mtp_num_hidden_layers": 1, "mtp_use_dedicated_embeddings": False, "ngram_size": 3,
        "ngram_vocab_size_base": 20000000, "num_attention_heads": 24, "num_experts": 512, "num_experts_per_tok": 10,
        "num_hidden_layers": 48, "num_key_value_heads": 2, "output_gate_type": "sigmoid",
        "partial_rotary_factor": 0.25, "ple_conv_kernel_size": 4, "ple_embed_dim": 2560, "ple_layer_ids": [2],
        "rms_norm_eps": 1e-06,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
                            "rope_theta": 10000000, "type": "default"},
        "shared_expert_intermediate_size": 640, "split_ngram_parts": 128, "tie_word_embeddings": False,
        "use_cache": True, "vocab_size": 248320,
    },
}

DEV = "cuda"
SCALE = 0.02            # most linears' value range (tests' _Rand)
EMBED_SCALE = 0.5
HEAD_SCALE = 0.2
NGRAM_PREFIX = "model.language_model.layers.{layer}.ple.ple_embedding.ngram_embedding"


def text(raw: dict) -> dict:
    return raw.get("text_config") or raw


def synthetic_config(real: dict, pattern: str, experts: int, ngram_base: int, ple: bool) -> dict:
    """The real config with ``pattern`` ("L" linear / "A" full attention per layer), ``experts`` and a smaller n-gram
    vocabulary base (``ngram_base``: the table's rows; nothing on the GPU depends on it). ``ple`` False drops the
    n-gram layers."""

    raw = copy.deepcopy(real)
    t = text(raw)
    kinds = {"L": "linear_attention", "A": "full_attention"}
    if not pattern or any(ch not in kinds for ch in pattern):
        raise ValueError(f"layer pattern {pattern!r}: one L (linear) or A (full attention) per layer")
    t["layer_types"] = [kinds[ch] for ch in pattern]
    t["num_hidden_layers"] = len(pattern)
    t["num_experts"] = int(experts)
    t["ngram_vocab_size_base"] = int(ngram_base)
    if not ple:
        t["ple_layer_ids"] = []
    else:   # ple_layer_ids are one-indexed (Config.read subtracts one): keep the ones inside the pattern
        t["ple_layer_ids"] = [i for i in t.get("ple_layer_ids") or [] if 1 <= int(i) <= len(pattern)]
    return raw


def read_config(model_dir: Path, yarn_factor: float | None):
    """``Config`` and the RoPE policy through the production readers (``rope_parameters``, ``Config.read``)."""

    from tensorfold.families.qwen4_exp import rope_parameters
    from tensorfold.families.qwen4_exp.cuda.weights import Config

    rope = rope_parameters(model_dir, yarn_factor)
    return Config.read(model_dir, rope=rope), rope


# -- byte plan (before any allocation) -----------------------------------------------------------------------------
Q4_BYTES = 0.625        # 4-bit words + bf16 scale + bf16 bias per 32 inputs, per weight


def plan_weight_bytes(c, *, mtp: bool, draft_rows: int, share_mtp_experts: bool) -> dict[str, int]:
    """Device bytes of each weight group as ``build`` allocates them (MLX 4-bit at 0.625 B/weight)."""

    D, S, LOW = c.hidden, c.streams, c.low

    def q4(n, k):
        return int(n * k * Q4_BYTES)

    def hc(inject):
        return 2 * (q4(LOW + (S if inject else 0), S * D) + q4(S * D, LOW)) + S * D * 4     # tiled + frag, scale

    moe_one = (c.experts + 1) * 3 * q4(c.moe_width, D) + (c.experts + 1) * D * 2               # experts + router
    attn_n = c.heads * 2 * c.head_dim + 2 * c.kv_heads * c.head_dim + (c.index_heads + 1) * c.index_dim
    attn = q4(attn_n, D) + q4(D, c.heads * c.head_dim) + (2 * c.head_dim + 2 * c.index_dim) * 4
    gdn = (q4(c.conv_dim + c.nv * c.dv + 2 * c.nv, D) + q4(D, c.nv * c.dv) + c.conv_dim * c.conv_kernel * 2
           + c.nv * 8 + c.dv * 2)
    ple = q4(S * D, c.ple_dim) + q4(D, c.ple_dim) + 3 * S * D * 4 + S * D * c.ple_kernel * 2
    lin = sum(1 for k in c.layer_types if k == "linear")
    att = len(c.layer_types) - lin
    out = {
        "experts_main": len(c.layer_types) * moe_one,
        "experts_mtp": moe_one if mtp and not share_mtp_experts else 0,
        "hyper_connections": len(c.layer_types) * 2 * hc(True) + hc(False) + ((2 * hc(True) + hc(False)) if mtp else 0),
        "attention": att * attn + (attn if mtp else 0),
        "deltanet": lin * gdn,
        "ple": len(c.ple_layers) * ple,
        "embed": q4(c.vocab, D),
        "head": q4(c.vocab, D),
        "draft_head": q4(draft_rows, D) + draft_rows * 8 if mtp else 0,
        "mtp_fc": (2 * q4(D, D) + D * 4 + S * D * 4) if mtp else 0,
    }
    out["total"] = sum(out.values())
    return out


def plan_state_bytes(c, kv_dtype: str, rows: int, *, mtp: bool) -> int:
    """One stream's committed state at ``rows`` cache rows (``State.cache_bytes`` plus its DeltaNet and conv state)."""

    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    pair = get_pair(kv_dtype)
    att = sum(1 for k in c.layer_types if k != "linear") + (1 if mtp else 0)
    lin = sum(1 for k in c.layer_types if k == "linear")
    row = pair.row_bytes(c.kv_heads, c.head_dim) + c.index_dim * 2
    caches = att * (rows * row + pair.dummy_bytes + -(-rows // c.index_ratio) * c.index_dim * 2)
    recurrent = 2 * lin * c.nv * c.dv * c.dk * 4 + lin * (c.conv_kernel - 1) * c.conv_dim * 2
    return caches + recurrent


# -- random MLX arrays ----------------------------------------------------------------------------------------------
class Rand:
    """tests' ``_Rand`` at any size: MLX (words, scales, biases) and the loader's packings of them."""

    def __init__(self, seed: int) -> None:
        self.g = torch.Generator(device=DEV).manual_seed(seed)

    def mlx(self, n: int, k: int, lead: tuple = (), scale: float = SCALE):
        words = torch.randint(-(2**31), 2**31 - 1, (*lead, n, k // 8), generator=self.g, device=DEV,
                              dtype=torch.int64).to(torch.int32)
        s = (torch.rand((*lead, n, k // 32), generator=self.g, device=DEV) * scale / 8 + scale / 64).to(torch.bfloat16)
        b = (-(s.float() * 7.5)).to(torch.bfloat16)
        return words, s, b

    def q4(self, n: int, k: int, scale: float = SCALE, layout: str = "frag"):
        from tensorfold.families.qwen4_exp.cuda import qmm

        return qmm.make_q4(*self.mlx(n, k, scale=scale), layout)

    def norm(self, n: int) -> torch.Tensor:
        return (1 + 0.05 * torch.randn((n,), generator=self.g, device=DEV)).float()

    def hc(self, c, inject: bool):
        """weights.load's ``hc``: [down (+ inject)] and up, each "tiled" (decode) and "frag" (prompt)."""

        from tensorfold.families.qwen4_exp.cuda import qmm
        from tensorfold.families.qwen4_exp.cuda.weights import HC

        S, D, LOW = c.streams, c.hidden, c.low
        parts = [self.mlx(LOW, S * D)] + ([self.mlx(S, S * D)] if inject else [])
        up = self.mlx(S * D, LOW)
        return HC(qmm.stack_q4(parts, "tiled"), qmm.make_q4(*up, "tiled"), self.norm(S * D), inject,
                  qmm.stack_q4(parts, "frag"), qmm.make_q4(*up, "frag"))

    def experts(self, c, chunk: int = 32):
        """weights.load's ``moe`` table (routed experts, then the shared one; gate, up and down), packed a chunk of
        experts at a time into the grouped layout ``experts.make`` builds."""

        from tensorfold.cuda import experts as grouped

        E1, W, D = c.experts + 1, c.moe_width, c.hidden
        if c.shared_width != W:
            raise ValueError("the grouped table holds one width: shared and routed experts must match")
        up = down = None
        for a in range(0, E1, chunk):
            n = min(E1, a + chunk) - a
            gate = grouped.pack(*self.mlx(W, D, (n,)), c.group_size)
            upp = grouped.pack(*self.mlx(W, D, (n,)), c.group_size)
            dn = grouped.pack(*self.mlx(D, W, (n,)), c.group_size)
            if up is None:
                up = torch.empty((E1, *gate.shape[1:3], 2, gate.shape[3]), dtype=torch.int32, device=DEV)
                down = torch.empty((E1, *dn.shape[1:3], 1, dn.shape[3]), dtype=torch.int32, device=DEV)
            up[a:a + n, :, :, 0] = gate
            up[a:a + n, :, :, 1] = upp
            down[a:a + n, :, :, 0] = dn
            del gate, upp, dn
        return grouped.Experts(up, down, c.group_size, W, D, 0.0)

    def moe(self, c, experts=None):
        from tensorfold.families.qwen4_exp.cuda.weights import MoEW

        # router rows (bf16) then the shared expert's gate row (dequantized MLX in the checkpoint: bf16 here)
        router = (torch.randn((c.experts + 1, c.hidden), generator=self.g, device=DEV) * 0.05).to(torch.bfloat16)
        return MoEW(router, self.experts(c) if experts is None else experts)

    def attention(self, c):
        from tensorfold.families.qwen4_exp.cuda import qmm
        from tensorfold.families.qwen4_exp.cuda.weights import AttnW

        D = c.hidden
        hd = c.head_dim
        # weights.load's ``attention``: stack_q4([q (with gates), k, v, indexer qk]) ("frag"), o
        parts = [self.mlx(c.heads * 2 * hd, D), self.mlx(c.kv_heads * hd, D), self.mlx(c.kv_heads * hd, D),
                 self.mlx((c.index_heads + 1) * c.index_dim, D)]
        return AttnW(qmm.stack_q4(parts), self.norm(hd), self.norm(hd), self.norm(c.index_dim),
                     self.norm(c.index_dim), self.q4(D, c.heads * hd))

    def gdn(self, c):
        from tensorfold.families.qwen4_exp.cuda import qmm
        from tensorfold.families.qwen4_exp.cuda.weights import GDNW

        D = c.hidden
        # weights.load's ``gdn``: stack_q4([qkv, z, b, a]), conv [conv_dim, taps] bf16, A_log/dt_bias fp32, norm bf16
        parts = [self.mlx(c.conv_dim, D), self.mlx(c.nv * c.dv, D), self.mlx(c.nv, D), self.mlx(c.nv, D)]
        conv = (torch.randn((c.conv_dim, c.conv_kernel), generator=self.g, device=DEV) * 0.3).to(torch.bfloat16)
        a_log = torch.randn((c.nv,), generator=self.g, device=DEV) * 0.5
        dt = torch.randn((c.nv,), generator=self.g, device=DEV) * 0.5
        return GDNW(qmm.stack_q4(parts), conv, a_log, dt, self.norm(c.dv).to(torch.bfloat16),
                    self.q4(D, c.nv * c.dv))

    def ple(self, c, table, ngram):
        from tensorfold.families.qwen4_exp.cuda.weights import PLEW

        S, D = c.streams, c.hidden
        conv = (torch.randn((S * D, c.ple_kernel), generator=self.g, device=DEV) * 0.3).to(torch.bfloat16)
        return PLEW(table, self.q4(S * D, c.ple_dim), self.q4(D, c.ple_dim), self.norm(S * D), self.norm(S * D),
                    self.norm(S * D), conv, ngram)


def check_chunked_pack(seed: int = 5) -> None:
    """``Rand.experts``'s chunked table equals ``experts.make`` over the same arrays (small shapes, chunks of 2)."""

    from tensorfold.cuda import experts as grouped

    @dataclass
    class Small:
        experts: int = 4
        moe_width: int = 64
        shared_width: int = 64
        hidden: int = 128
        group_size: int = 32

    c = Small()
    a = Rand(seed).experts(c, chunk=2)
    r = Rand(seed)
    E1, W, D = c.experts + 1, c.moe_width, c.hidden
    gates, ups, downs = [], [], []
    for at in range(0, E1, 2):
        n = min(E1, at + 2) - at
        gates.append(r.mlx(W, D, (n,)))
        ups.append(r.mlx(W, D, (n,)))
        downs.append(r.mlx(D, W, (n,)))

    def cat(parts):
        return tuple(torch.cat([p[i] for p in parts]) for i in range(3))

    b = grouped.make([cat(gates), cat(ups)], cat(downs), 32)
    same = (a.width, a.dims, a.gs) == (b.width, b.dims, b.gs)
    if not (same and torch.equal(a.up, b.up) and torch.equal(a.down, b.down)):
        raise AssertionError("the chunked expert table differs from experts.make's")


# -- the n-gram (PLE) table -----------------------------------------------------------------------------------------
def _bf16_bits(x: np.ndarray) -> np.ndarray:
    """float32 -> bf16 bit patterns (round to nearest even)."""

    u = x.astype(np.float32).view(np.uint32).astype(np.uint64)
    u = (u + 0x7FFF + ((u >> 16) & 1)) >> 16
    return u.astype(np.uint16)


def ngram_table(data_dir: Path, ngram, shards: int, layer: int, seed: int = 11) -> tuple[Path, str, list[str]]:
    """An MLX 4-bit n-gram table (``shards`` shards of equal rows, words/scales/biases, as the checkpoint ships it) in
    one safetensors file under ``data_dir``, made once and reused (keyed by its geometry and seed). Returns the
    directory, the file name and the shard keys."""

    rows, dims = int(ngram.rows), int(ngram.dims)
    if rows % shards or dims % 32:
        raise ValueError(f"n-gram table of {rows} rows x {dims} does not split into {shards} 4-bit shards")
    per = rows // shards
    prefix = NGRAM_PREFIX.format(layer=layer)
    keys = [f"{prefix}.shard_{i}" for i in range(shards)]
    ident = hashlib.sha256(json.dumps([rows, dims, shards, prefix, seed, 1]).encode()).hexdigest()[:16]
    where = data_dir / f"ngram-{ident}"
    name = "ngram.safetensors"
    path = where / name
    if path.is_file() and (where / "done").is_file():
        return where, name, keys
    where.mkdir(parents=True, exist_ok=True)
    k8, kg = dims // 8, dims // 32
    header, at = {}, 0
    for key in keys:
        for part, dtype, width in (("weight", "U32", k8 * 4), ("scales", "BF16", kg * 2), ("biases", "BF16", kg * 2)):
            size = per * width
            header[f"{key}.{part}"] = {"dtype": dtype, "shape": [per, width // (4 if dtype == "U32" else 2)],
                                       "data_offsets": [at, at + size]}
            at += size
    blob = json.dumps(header).encode()
    blob += b" " * ((8 - len(blob) % 8) % 8)
    rng = np.random.default_rng(seed)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        f.write(struct.pack("<Q", len(blob)))
        f.write(blob)
        for _ in keys:
            f.write(rng.integers(0, 2**32, (per, k8), dtype=np.uint32).tobytes())
            s = rng.random((per, kg), dtype=np.float32) * SCALE / 8 + SCALE / 64
            f.write(_bf16_bits(s).tobytes())
            f.write(_bf16_bits(-7.5 * _bf16_to_f32(_bf16_bits(s))).tobytes())
    os.replace(tmp, path)
    done = {"rows": rows, "dims": dims, "shards": shards, "bytes": path.stat().st_size}
    (where / "done").write_text(json.dumps(done))
    return where, name, keys


def _bf16_to_f32(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << 16).view(np.float32)


# -- the model ------------------------------------------------------------------------------------------------------
@dataclass
class Built:
    w: object               # weights.Weights
    cfg: object             # weights.Config
    rope: object
    ple_tables: object      # ple_lifetime.PLETables owning the n-gram table(s)
    table_path: Path | None
    seconds: float


def build(c, rope, *, seed: int, mtp: bool, draft_vocab: str | int | None, share_mtp_experts: bool,
          data_dir: Path | None, ple_on_ssd: bool, log=print) -> Built:
    """The model ``weights.load`` would return for a checkpoint of these dimensions (one GPU, MLX 4-bit)."""

    import time

    from tensorfold.families.qwen4_exp.cuda import qmm
    from tensorfold.families.qwen4_exp.cuda.weights import LayerW, MTPW, Weights, draft_token_ids
    from tensorfold.families.qwen4_exp.host_table import open_table
    from tensorfold.families.qwen4_exp.ple_lifetime import PLETables

    t0 = time.perf_counter()
    r = Rand(seed)
    D = c.hidden
    # the vocabulary-sized arrays first, while little else is resident (their unpacked copies are transient)
    embed = r.mlx(c.vocab, D, scale=EMBED_SCALE)                   # weights.load keeps the embedding's MLX arrays
    head_raw = r.mlx(c.vocab, D, scale=HEAD_SCALE)
    draft_head = draft_ids = None
    if mtp:
        ids = draft_token_ids(draft_vocab, c.vocab)
        if ids is not None:
            draft_ids = torch.from_numpy(ids).to(DEV)
            draft_head = qmm.make_q4(*(t.index_select(0, draft_ids) for t in head_raw))
    head = qmm.make_q4(*head_raw)
    del head_raw
    torch.cuda.empty_cache()
    log(f"[synthetic] embed/head/draft head built ({time.perf_counter() - t0:.1f}s)")
    ple_tables = PLETables()
    table_path = None
    layers = []
    for i, kind in enumerate(c.layer_types):
        linear = kind == "linear"
        layer = LayerW(i, linear, r.hc(c, True), r.hc(c, True), r.gdn(c) if linear else None,
                       None if linear else r.attention(c), r.moe(c))
        if i in c.ple_layers:
            if data_dir is None:
                raise ValueError("a PLE layer needs --data-dir for its n-gram table")
            ngram = c.ngram(c.ple_layers.index(i))
            where, name, keys = ngram_table(data_dir, ngram, c.ngram_shards, i)
            table = ple_tables.acquire(open_table(where, [(name, k) for k in keys], lambda n: 1.0, ssd=ple_on_ssd))
            if table.rows != ngram.rows:
                raise ValueError(f"n-gram tables hold {table.rows} rows, expected {ngram.rows}")
            if not ple_on_ssd:
                table.prefetch()
            table_path = where / name
            layer.ple = r.ple(c, table, ngram)
        layers.append(layer)
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        log(f"[synthetic] layer {i} ({kind}{', PLE' if layer.ple is not None else ''}) built "
            f"({time.perf_counter() - t0:.1f}s, {torch.cuda.memory_allocated() / 2**30:.2f} GiB allocated)")
    mixer = r.hc(c, False)
    inv = rope.inverse_frequencies(torch)
    w = Weights(c, embed, layers, mixer, head, inv.to(DEV), around_one=True)
    w.meta.update(rank=0, world=1, vocab_offset=0, full=c, ple_tables=ple_tables)
    w.meta["rope"] = c.rope.metadata()
    w.draft_head, w.draft_ids = draft_head, draft_ids
    if mtp:
        shared = layers[-1].moe.experts if share_mtp_experts else None
        mlayer = LayerW(-1, False, r.hc(c, True), r.hc(c, True), None, r.attention(c), r.moe(c, shared))
        w.mtp = MTPW(r.norm(D), r.norm(c.streams * D), r.q4(D, D), r.q4(D, D), mlayer, r.hc(c, False))
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    log(f"[synthetic] model built in {time.perf_counter() - t0:.1f}s: {w.nbytes() / 2**30:.3f} GiB of weights")
    return Built(w, c, rope, ple_tables, table_path, time.perf_counter() - t0)


def write_config(where: Path, raw: dict) -> Path:
    where.mkdir(parents=True, exist_ok=True)
    (where / "config.json").write_text(json.dumps(raw, indent=1))
    return where


__all__ = ["REAL_CONFIG", "synthetic_config", "read_config", "plan_weight_bytes", "plan_state_bytes", "Rand",
           "check_chunked_pack", "ngram_table", "build", "write_config"]

"""DeepSeek-V4.1-Flash GGUF weights for one rank of the eight-GPU SM70 engine.

TP8 attention/dense (the Flash Next / Gemma 4 lane layout): rank r owns slice r of every sharded projection (8
query heads, one output group, 4 indexer heads, 288 shared-expert rows, 1/8 of the Engram value projection and
the vocabulary). Routed experts use one, two or four tensor-parallel lanes per group. Expert group g = r // ETP owns
an equal, contiguous range of expert IDs, and lane r % ETP owns its contiguous intermediate slice.
Routed and shared partials meet in one eight-rank sum.

Names follow the ISTA-branch GGUF layout the 1Cat adapter (gguf_adapters/deepseek_v41.py at 1Cat-vLLM 4d57473d)
was qualified on; the uncensored AtomicChat backbone uses it unchanged. The Engram table is the llama.cpp file
(one ``per_layer_token_embd.weight``), split by the backbone's per-layer bucket sums (the friend's patch 221).

Projections are decoded once to FP16 on the GPU (the upstream "original dense storage" restore that its prefill
GEMMs use). Routed experts keep their original IQ1_M/IQ2_XXS/IQ2_XS bytes.
Each down row retains the complete 256-column blocks covering its assigned slice and its in-block offset.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from tensorfold.gguf_file import FORMATS, GGUFCheckpoint, PackedTensor, _shard

TP = 8                                   # attention/dense ranks
ETP = 4                                  # default expert tensor-parallel ranks per expert group
EXPERT_TP_SIZES = (1, 2, ETP)
WORLD = TP
EXPERT_TYPES = (16, 17, 29)              # IQ2_XXS, IQ2_XS, IQ1_M
DENSE_TYPES = (8, 13)                    # Q8_0, Q5_K
IQ4_NL = 20
ENGRAM_TABLE = "per_layer_token_embd.weight"
ENGRAM_ROW_BYTES = 144                   # 256 IQ4_NL values


def checked_etp(degree: int | None) -> int:
    if degree is None:
        return ETP
    if type(degree) is not int or degree not in EXPERT_TP_SIZES:
        raise ValueError("DeepSeek-V4.1 CUDA expert tensor parallelism requires degree 1, 2 or 4")
    return degree


@dataclass(frozen=True)
class RopeParameters:
    factor: float
    context_limit: int
    original_context: int
    beta_fast: float
    beta_slow: float
    window_theta: float
    compressed_theta: float
    rope_type = "yarn"


@dataclass(frozen=True)
class Geometry:
    """The checkpoint geometry the ported kernels admit (upstream contract.py and the adapter's checks)."""

    hidden: int = 5120
    layers: int = 40
    vocab: int = 129280
    heads: int = 64
    head_dim: int = 512
    rope_dim: int = 64
    q_lora: int = 1280
    o_groups: int = 8
    o_lora: int = 1024
    experts: int = 384
    top_k: int = 6
    moe_inter: int = 2304
    index_heads: int = 32
    index_dim: int = 128
    index_topk: int = 512
    window: int = 128
    hc: int = 4
    routed_scale: float = 1.5
    swiglu_limit: float = 10.0
    eps: float = 1e-20
    hc_eps: float = 1e-6
    hc_iters: int = 20
    compress_ratios: tuple = ()
    kv_sources: tuple = (2, 8, 14, 20)
    index_sources: tuple = (2, 8, 14, 20, 24, 28, 32, 36)
    engram_layers: tuple = (1, 14)
    engram_rows: tuple = ()
    engram_pad: int = 2
    candidate_source: int = 20
    candidate_blocks: int = 2048
    candidate_block: int = 8
    max_positions: int = 1048576
    rope_theta: float = 10000.0
    compress_rope_theta: float = 160000.0
    rope_scaling: dict = field(default_factory=dict)
    bos: int = 0
    eos: int = 1
    dspark_layers: tuple = (37, 38, 39)

    @classmethod
    def from_config(cls, raw: dict) -> "Geometry":
        c = raw.get("text_config") or raw
        if c.get("model_type") not in ("deepseek_v41_text", "deepseek_v41"):
            raise ValueError(f"DeepSeek-V4.1 CUDA lane reads model_type deepseek_v41, not {c.get('model_type')!r}")
        g = cls(hidden=c["hidden_size"], layers=c["num_hidden_layers"], vocab=c["vocab_size"],
                heads=c["num_attention_heads"], head_dim=c["head_dim"], rope_dim=c["qk_rope_head_dim"],
                q_lora=c["q_lora_rank"], o_groups=c["o_groups"], o_lora=c["o_lora_rank"],
                experts=c["n_routed_experts"], top_k=c["num_experts_per_tok"], moe_inter=c["moe_intermediate_size"],
                index_heads=c["index_n_heads"], index_dim=c["index_head_dim"], index_topk=c["index_topk"],
                window=c["sliding_window"], hc=c["hc_mult"], routed_scale=float(c["routed_scaling_factor"]),
                swiglu_limit=float(c["swiglu_limit"]), eps=float(c["rms_norm_eps"]), hc_eps=float(c["hc_eps"]),
                hc_iters=int(c["hc_sinkhorn_iters"]),
                compress_ratios=tuple(int(r) for r in c["compress_ratios"][:c["num_hidden_layers"]]),
                kv_sources=tuple(c["kv_source_layer_ids"]), index_sources=tuple(c["index_source_layer_ids"]),
                engram_layers=tuple(c["engram_layer_ids"]), engram_rows=tuple(c["engram_num_embeddings"]),
                engram_pad=int(c["engram_pad_token_id"]),
                candidate_source=int(c.get("candidate_source_layer_id", -1)),
                candidate_blocks=int(c.get("candidate_topk_blocks", 0)),
                candidate_block=int(c.get("candidate_block_size", 0)),
                max_positions=int(c["max_position_embeddings"]), rope_theta=float(c["rope_theta"]),
                compress_rope_theta=float(c["compress_rope_theta"]), rope_scaling=dict(c.get("rope_scaling") or {}),
                bos=int(raw.get("bos_token_id", c.get("bos_token_id", 0))),
                eos=int(raw.get("eos_token_id", c.get("eos_token_id", 1))),
                dspark_layers=tuple(c.get("dspark_target_layer_ids", ())))
        expected = (5120, 40, 384, 2304, 6, 4, 64, 512, 1280, 8, 1024, 32, 128, 512, 128)
        actual = (g.hidden, g.layers, g.experts, g.moe_inter, g.top_k, g.hc, g.heads, g.head_dim, g.q_lora,
                  g.o_groups, g.o_lora, g.index_heads, g.index_dim, g.index_topk, g.window)
        if (actual != expected or g.kv_sources != (2, 8, 14, 20) or g.index_sources != (2, 8, 14, 20, 24, 28, 32, 36)
                or g.engram_layers != (1, 14) or g.swiglu_limit != 10.0 or len(g.compress_ratios) != 40):
            raise ValueError("DeepSeek-V4.1 CUDA lane admits the V4.1 Flash geometry only (the ported 1Cat kernels' "
                             f"contract); this checkpoint has {actual}")
        return g

    def rope_parameters(self, yarn_factor: float | None = None) -> RopeParameters:
        """YaRN scales the compressed layers. The window-only layers keep plain RoPE."""
        scaling = self.rope_scaling
        kind = scaling.get("rope_type", scaling.get("type", "yarn"))
        if kind != "yarn" or scaling.get("type", kind) != kind:
            raise ValueError("DeepSeek-V4.1 compressed layers require static YaRN")
        original = scaling.get("original_max_position_embeddings", 65_536)
        if type(original) is not int or not 0 < original <= self.max_positions:
            raise ValueError("DeepSeek-V4.1 YaRN original context must be within the checkpoint's context")
        native = scaling.get("factor", 16.0)
        factor = native if yarn_factor is None else yarn_factor
        fast, slow = scaling.get("beta_fast", 32), scaling.get("beta_slow", 1)
        floats = np.finfo(np.float32)
        least, most = float(floats.smallest_subnormal), float(floats.max)
        for value, name in ((factor, "YaRN factor"), (fast, "beta_fast"), (slow, "beta_slow"),
                            (self.rope_theta, "rope_theta"), (self.compress_rope_theta, "compress_rope_theta")):
            if type(value) not in (int, float) or not least <= value <= most:
                raise ValueError(f"DeepSeek-V4.1 {name} must be positive in the finite FP32 range")
        if factor < 1:
            raise ValueError("DeepSeek-V4.1 requires YaRN factor >= 1")
        if np.float32(min(self.rope_theta, self.compress_rope_theta)) <= 1:
            raise ValueError("DeepSeek-V4.1 RoPE bases must exceed 1 in FP32")
        if slow > fast:
            raise ValueError("DeepSeek-V4.1 YaRN beta_slow cannot exceed beta_fast")
        context = self.max_positions if factor >= self.max_positions / original else math.floor(original * factor)
        return RopeParameters(float(factor), context, original, float(fast), float(slow), self.rope_theta,
                              self.compress_rope_theta)

    def source_of(self, layer: int) -> int:
        """The KV source layer a compressed layer reads (the last source at or below it)."""
        return max((i for i in self.kv_sources if i <= layer), default=-1)


# --------------------------------------------------------------------------------------------- GGUF reading
class Shards:
    """Read-only maps of the checkpoint files; slices are copied out (never retained views of the map)."""

    def __init__(self, entry: Path, engram: Path):
        self.ck = GGUFCheckpoint.open(entry)
        self.tensors = dict(self.ck.tensors)
        if ENGRAM_TABLE in self.tensors:
            raise ValueError("this backbone carries the Engram table as a shard; pass the backbone only")
        # the llama.cpp file is shard 6 of ISTA's 6-shard set (its split metadata counts that set's tensors), read
        # here as one standalone shard
        _, entries = _shard(Path(engram).resolve(strict=True))
        if [t.name for t in entries] != [ENGRAM_TABLE]:
            raise ValueError(f"{engram}: the llama.cpp Engram file holds exactly {ENGRAM_TABLE}")
        self.engram = entries[0]
        if self.engram.ggml_type != IQ4_NL or self.engram.shape[-1] != 256:
            raise ValueError("the Engram table must be IQ4_NL rows of 256 values")
        self._maps: dict[Path, np.memmap] = {}

    def _map(self, path: Path) -> np.memmap:
        found = self._maps.get(path)
        if found is None:
            found = self._maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        return found

    def raw(self, name: str) -> tuple[PackedTensor, np.ndarray]:
        t = self.tensors[name]
        return t, self._map(t.path)[t.offset:t.offset + t.nbytes]

    def rows(self, name: str) -> tuple[PackedTensor, np.ndarray]:
        """[rows, row_bytes] bytes of a 2-D (or flattened leading-dims) tensor."""
        t, data = self.raw(name)
        return t, data.reshape(t.rows, t.row_bytes)

    def close(self) -> None:
        self._maps.clear()


def _check_finite(name, tensor, limit=65504.0):
    import torch
    bad = ~torch.isfinite(tensor)
    if bool(bad.any()) or float(tensor.abs().max()) > limit:
        raise ValueError(f"{name}: decoded values exceed FP16")


class Loader:
    """Per-rank device tensors from the shards (every projection decoded once, in bounded row chunks)."""

    CHUNK_BYTES = 64 << 20

    def __init__(self, shards: Shards, rank: int, device, staging, *, expert_tensor_parallel_size: int | None = None):
        import torch
        self.s, self.rank, self.device = shards, rank, device
        self.etp = checked_etp(expert_tensor_parallel_size)
        self.t, self.e, self.g = rank, rank % self.etp, rank // self.etp
        self.torch = torch
        self.staging = staging          # staging.Staging: pinned-buffer uploads of the large byte ranges

    def _rows(self, t, first, count):
        """Original bytes uint8 [count, row_bytes] of rows [first, first + count) of ``t``, on the device."""
        out = self.torch.empty((count, t.row_bytes), dtype=self.torch.uint8, device=self.device)
        return self.staging.copy(t.path, [t.row_span(first, count)], out)

    # --- primitives
    def floating(self, name, dtype):
        torch = self.torch
        t, data = self.s.raw(name)
        if t.ggml_type == 0:
            value = torch.from_numpy(np.frombuffer(data, dtype=np.float32).copy())
        elif t.ggml_type == 30:
            value = torch.from_numpy(np.frombuffer(data, dtype=np.uint16).astype(np.int32) << 16).view(torch.float32)
        elif t.ggml_type == 1:
            value = torch.from_numpy(np.frombuffer(data, dtype=np.float16).copy()).float()
        else:
            raise ValueError(f"{name}: expected a floating tensor, found {FORMATS[t.ggml_type][0]}")
        value = value.reshape(t.shape).to(self.device)
        if dtype == torch.float16:
            _check_finite(name, value)
        return value.to(dtype).contiguous()

    def integers(self, name, dtype):
        t, data = self.s.raw(name)
        kind = {26: np.int32, 27: np.int64}.get(t.ggml_type)
        if kind is None:
            raise ValueError(f"{name}: expected an integer tensor")
        return self.torch.from_numpy(np.frombuffer(data, dtype=kind).copy()).reshape(t.shape).to(self.device)

    def dense(self, name, rows=None, cols=None, dtype=None):
        """FP16 (or ``dtype``) [rows, cols] of a 2-D projection; ``rows``/``cols`` select a (start, count) slice."""
        torch = self.torch
        dtype = dtype or torch.float16
        t = self.s.tensors[name]
        if t.ggml_type in (0, 1, 30):
            full = self.floating(name, torch.float32).reshape(t.rows, t.shape[-1])
            out = full
        else:
            from tensorfold.cuda.sm70 import onecat_ds41 as native
            first, count = rows if rows is not None else (0, t.rows)
            out = torch.empty(count, t.shape[-1], dtype=torch.float16, device=self.device)
            step = max(1, self.CHUNK_BYTES // t.row_bytes)
            for at in range(0, count, step):
                n = min(step, count - at)
                packed = self._rows(t, first + at, n)
                native.dequantize(packed, t.ggml_type, t.shape[-1], out=out[at:at + n])
            _check_finite(name, out)
            rows = None
        if rows is not None:
            out = out[rows[0]:rows[0] + rows[1]]
        if cols is not None:
            out = out[:, cols[0]:cols[0] + cols[1]]
        return out.to(dtype).contiguous()

    def planes(self, name, rows=None, cols=None):
        """dmv13 planes (``ds41_dense.Planes``) of a Q8_0 / Q5_K [rows, cols] slice, packed on the GPU from the
        original row bytes in bounded chunks; the only resident copy of the projection."""
        from tensorfold.cuda.sm70 import ds41_dense as dense
        torch = self.torch
        t = self.s.tensors[name]
        if t.ggml_type not in dense.FORMAT:
            raise ValueError(f"{name}: dmv13 planes take Q8_0 / Q5_K, found {FORMATS[t.ggml_type][0]}")
        first, count = rows if rows is not None else (0, t.rows)
        c0, width = cols if cols is not None else (0, t.shape[-1])
        if c0 % 32 or width % 32:
            raise ValueError(f"{name}: column window must align to the 32-value quantization steps")
        out = dense.Planes(dense.FORMAT[t.ggml_type], count, width, self.device)
        step = max(32, self.CHUNK_BYTES // t.row_bytes // 32 * 32)
        for at in range(0, count, step):
            n = min(step, count - at)
            out.pack(self._rows(t, first + at, n), t.ggml_type, c0 // 32, tile0=at // 32)
        return out

    def column_planes(self, name, parts=TP):
        t = self.s.tensors[name]
        n = t.rows // parts
        return self.planes(name, rows=(self.t * n, n))

    def row_planes(self, name, parts=TP):
        t = self.s.tensors[name]
        n = t.shape[-1] // parts
        return self.planes(name, cols=(self.t * n, n))

    def router_gate(self, name):
        """The router gate in BF16 when that is exact (the GGUF stores it as BF16, so FP32 values widen from
        it); otherwise FP32. fp32_projection widens BF16 to the same FP32 operands."""
        torch = self.torch
        value = self.floating(name, torch.float32)
        narrow = value.to(torch.bfloat16)
        if self.s.tensors[name].ggml_type == 30 or torch.equal(narrow.float(), value):
            return narrow.contiguous()
        return value

    def column(self, name, parts=TP):
        """This rank's ColumnParallel rows (output features)."""
        t = self.s.tensors[name]
        n = t.rows // parts
        return self.dense(name, rows=(self.t * n, n))

    def row(self, name, parts=TP):
        """This rank's RowParallel columns (input features)."""
        t = self.s.tensors[name]
        n = t.shape[-1] // parts
        return self.dense(name, cols=(self.t * n, n))

    # --- routed experts
    def expert_bank(self, name, geometry: Geometry, *, down: bool):
        """Original bytes of this rank's gate/up rows and the whole down blocks covering its input columns."""
        torch = self.torch
        t = self.s.tensors[name]
        kind = t.ggml_type
        if kind not in EXPERT_TYPES:
            raise ValueError(f"{name}: routed experts must be IQ2_XXS, IQ2_XS or IQ1_M, found {FORMATS[kind][0]}")
        _, block, size = FORMATS[kind]
        rows = (geometry.hidden, geometry.moe_inter) if down else (geometry.moe_inter, geometry.hidden)
        if t.shape != (geometry.experts, *rows):
            raise ValueError(f"{name}: expert tensor shape {t.shape} does not match {(geometry.experts, *rows)}")
        experts, out_rows, k = t.shape
        local = experts // (TP // self.etp)
        first = self.g * local
        row = k // block * size                     # bytes of one output row of one expert
        if down:
            inter = k
            share = inter // self.etp
            b0 = self.e * share // block
            left = self.e * share - b0 * block
            blocks = -(-(left + share) // block)
            if (b0 + blocks) * block > inter:
                raise ValueError("down carrier does not cover the rank's input columns")
            # The group's experts are contiguous. Cut the rank's complete blocks per row on the GPU.
            whole = torch.empty((local, out_rows, row), dtype=torch.uint8, device=self.device)
            self.staging.copy(t.path, [(t.offset + first * out_rows * row, local * out_rows * row)], whole)
            out = whole[:, :, b0 * size:(b0 + blocks) * size].contiguous()
            del whole
        else:
            share = out_rows // self.etp
            out = torch.empty((local, share, row), dtype=torch.uint8, device=self.device)
            if share == out_rows:
                spans = [(t.offset + first * out_rows * row, local * out_rows * row)]
            else:
                spans = [(t.offset + ((first + e) * out_rows + self.e * share) * row, share * row)
                         for e in range(local)]
            self.staging.copy(t.path, spans, out)
            left = 0
        return out, kind, left


@dataclass
class Layer:
    index: int
    ratio: int
    attn_norm: object = None
    ffn_norm: object = None
    wq_a: object = None
    wkv: object = None
    q_norm: object = None
    kv_norm: object = None
    wq_b: object = None
    wo_a: object = None            # [1 group, 1024, 4096]
    wo_b: object = None            # [5120, 1024]
    sink: object = None            # [8] fp32
    comp_wkv: object = None        # ratio 2: fp32 [512, 5120]; ratio 1: fp16
    comp_wgate: object = None
    comp_norm: object = None
    idx_wq_b: object = None        # [512, 1280]
    idx_proj: object = None        # [4, 5120]
    idx_wk: object = None          # [128, 512]
    idx_k_norm: object = None
    gate: object = None            # [384, 5120] fp32
    gate_bias: object = None
    gate_up_type: int = 0
    down_type: int = 0
    down_left: int = 0
    exp_gate: object = None
    exp_up: object = None
    exp_down: object = None
    sh_gate: object = None
    sh_up: object = None
    sh_down: object = None         # [5120, 288]
    hc: dict = field(default_factory=dict)
    engram_wkv: object = None      # [3200, 6144]
    qa_kv: object = None           # dmv13 group (wq_a, wkv): one launch for both same-input outputs
    sh_gu: object = None           # dmv13 group (sh_gate, sh_up)
    engram_q: object = None        # fp32 [4, 5120]
    engram_k: object = None
    source: int = -1
    owns_kv: bool = False
    has_indexer: bool = False


@dataclass
class Weights:
    geometry: Geometry
    rank: int
    layers: list
    embed: object                  # Q5_K bytes of this rank's vocabulary rows [vocab / 8, row_bytes]
    embed_type: int
    embed_first: int               # first vocabulary row of ``embed``
    head: object                   # dmv13 group of the head's [vocab / 8, 5120] rows
    norm: object
    engram_token_map: object
    engram_primes: object
    engram_offsets: object
    engram_multipliers: object
    dense_scratch: object = None   # ds41_dense.Scratch shared by every dmv13 group of this rank
    expert_tensor_parallel_size: int = ETP

    @property
    def group(self):
        return self.rank // self.expert_tensor_parallel_size

    @property
    def local_experts(self) -> range:
        count = self.geometry.experts // (TP // self.expert_tensor_parallel_size)
        first = self.group * count
        return range(first, first + count)


def load(shards: Shards, geometry: Geometry, rank: int, device, layers: range | None = None, *,
         expert_tensor_parallel_size: int | None = None) -> Weights:
    from .staging import Staging
    staging = Staging()
    try:
        loader = Loader(shards, rank, device, staging, expert_tensor_parallel_size=expert_tensor_parallel_size)
        return _load(loader, shards, geometry, rank, device, layers)
    finally:
        staging.close()


def _load(L, shards, geometry, rank, device, layers):
    import torch
    from tensorfold.cuda.sm70 import ds41_dense as dense
    g = geometry
    f32 = torch.float32
    scratch = dense.Scratch(device)
    group = lambda *members: dense.Group(members, scratch)   # noqa: E731
    out = []
    for i in (layers if layers is not None else range(g.layers)):
        p = f"blk.{i}."
        ly = Layer(index=i, ratio=g.compress_ratios[i])
        ly.attn_norm, ly.ffn_norm = L.floating(p + "attn_norm.weight", f32), L.floating(p + "ffn_norm.weight", f32)
        wq_a, wkv = L.planes(p + "attn_q_a.weight"), L.planes(p + "attn_kv.weight")
        ly.wq_a, ly.wkv, ly.qa_kv = group(wq_a), group(wkv), group(wq_a, wkv)
        ly.q_norm, ly.kv_norm = L.floating(p + "attn_q_a_norm.weight", f32), L.floating(p + "attn_kv_a_norm.weight", f32)
        ly.wq_b = group(L.column_planes(p + "attn_q_b.weight"))
        groups = g.o_groups // TP
        t = shards.tensors[p + "attn_output_a.weight"]
        if t.shape != (g.o_groups, g.o_lora, g.heads * g.head_dim // g.o_groups):
            raise ValueError(f"{p}attn_output_a: unexpected shape {t.shape}")
        if groups != 1:
            raise ValueError("TP8 holds exactly one output group per rank")
        ly.wo_a = group(L.planes(p + "attn_output_a.weight", rows=(L.t * g.o_lora, g.o_lora)))
        ly.wo_b = group(L.row_planes(p + "attn_output_b.weight"))
        heads = g.heads // TP
        ly.sink = L.floating(p + "attn_sinks.weight", f32)[L.t * heads:(L.t + 1) * heads].contiguous()
        ly.owns_kv = i in g.kv_sources
        ly.has_indexer = i in g.index_sources
        ly.source = g.source_of(i)
        if ly.owns_kv:
            wdt = f32 if ly.ratio == 2 else torch.float16
            ly.comp_wkv = L.dense(p + "attn_compressor_kv.weight", dtype=wdt)
            if ly.ratio == 2:
                ly.comp_wgate = L.dense(p + "attn_compressor_gate.weight", dtype=wdt)
            ly.comp_norm = L.floating(p + "attn_compressor_norm.weight", f32)
            ly.idx_wk = L.dense(p + "indexer.attn_k.weight")
            ly.idx_k_norm = L.floating(p + "indexer.k_norm.weight", f32)
        if ly.has_indexer:
            ly.idx_wq_b = L.column(p + "indexer.attn_q_b.weight")
            ly.idx_proj = L.column(p + "indexer.proj.weight")
        ly.gate = L.router_gate(p + "ffn_gate_inp.weight")
        ly.gate_bias = L.floating(p + "exp_probs_b.bias", f32)
        ly.exp_gate, ly.gate_up_type, _ = L.expert_bank(p + "ffn_gate_exps.weight", g, down=False)
        ly.exp_up, up_type, _ = L.expert_bank(p + "ffn_up_exps.weight", g, down=False)
        if up_type != ly.gate_up_type:
            raise ValueError(f"{p}: gate/up expert formats differ")
        ly.exp_down, ly.down_type, ly.down_left = L.expert_bank(p + "ffn_down_exps.weight", g, down=True)
        sh_gate, sh_up = L.column_planes(p + "ffn_gate_shexp.weight"), L.column_planes(p + "ffn_up_shexp.weight")
        ly.sh_gate, ly.sh_up, ly.sh_gu = group(sh_gate), group(sh_up), group(sh_gate, sh_up)
        ly.sh_down = group(L.row_planes(p + "ffn_down_shexp.weight"))
        for branch in ("attn", "ffn"):
            for part in ("fn", "base", "scale"):
                ly.hc[f"{branch}_{part}"] = L.floating(p + f"hc_{branch}_{part}.weight", f32)
        if i in g.engram_layers:
            ly.engram_wkv = group(L.column_planes(p + "engram_wkv.weight"))
            ly.engram_q = L.floating(p + "engram_q.weight", f32)
            ly.engram_k = L.floating(p + "engram_k.weight", f32)
        out.append(ly)
        torch.cuda.empty_cache()
    vocab = g.vocab // TP
    embed, embed_type = _embedding_shard(shards, L.t * vocab, vocab, device)
    head = group(L.planes("output.weight", rows=(L.t * vocab, vocab)))
    return Weights(geometry=g, rank=rank, layers=out, embed=embed, embed_type=embed_type,
                   expert_tensor_parallel_size=L.etp,
                   embed_first=L.t * vocab, head=head, dense_scratch=scratch,
                   norm=L.floating("output_norm.weight", f32),
                   engram_token_map=L.integers("engram_token_map.weight", torch.int32),
                   engram_primes=L.integers("engram_primes.weight", torch.int64),
                   engram_offsets=L.integers("engram_offsets.weight", torch.int64),
                   engram_multipliers=L.integers("engram_multipliers.weight", torch.int64))


def _embedding_shard(shards: Shards, first: int, count: int, device):
    """Q5_K bytes of vocabulary rows [first, first + count) (the TP8 vocabulary shard of the embedding)."""
    import torch
    t, data = shards.rows("token_embd.weight")
    if t.ggml_type != 13 or t.rows % TP:
        raise ValueError("token_embd must be a Q5_K table whose rows divide over the 8 ranks")
    return torch.from_numpy(np.array(data[first:first + count])).to(device), t.ggml_type


def engram_split(shards: Shards, geometry: Geometry) -> tuple[int, int]:
    """Rows of the two per-layer tables in the concatenated llama.cpp file (the backbone's bucket sums)."""
    t, data = shards.raw("engram_primes.weight")
    primes = np.frombuffer(data, dtype=np.int64).reshape(t.shape)
    rows = tuple(int(x) for x in primes.sum(axis=1))
    if geometry.engram_rows and rows != tuple(geometry.engram_rows):
        raise ValueError(f"Engram buckets {rows} differ from config engram_num_embeddings {geometry.engram_rows}")
    if sum(rows) != shards.engram.rows:
        raise ValueError(f"the Engram table has {shards.engram.rows} rows, the backbone's buckets sum to {sum(rows)}")
    return rows

"""DeepSeek-V4.1-Flash-1017 DSpark head (``mtp.0..2``) weights for one rank of the SM70 TP8 x (ETP4 x EP2) lane.

Sources: the official release (``deepseek-ai/DeepSeek-V4.1-Flash-1017`` rev 1bae0e10, shards 44-46) or the drop-in
fine-tune (``Avifenesh/DeepSeek-V4.1-Flash-DSpark-drafter-ft`` ``fp8/mtp-ft.safetensors``); both are read through
their ``model.safetensors.index.json`` and hold the same 2,401 ``mtp.*`` tensors in the same formats:

- 25 dense matmuls: FP8 E4M3FN with one UE8M0 (E8M0) scale per 32x32 block. Decoded ONCE to FP16 at load: the
  value is fp8 * 2^(s - 127) (s == 0 is 2^-127, as upstream's bitcast), rounded to FP16. 1Cat-vLLM's
  ``fp8_block32_linear`` (``vllm/models/deepseek_v41/fp8_linear.py`` @4d57473d) dequantizes each tile with exactly
  this product and ``.to(float16)`` before its FP16 MMA, so the FP16 bank is the same operand his GEMM sees.
  Each rank audits its owned dense values (all values in audit-only mode): FP16 overflow, flush to zero and inexact
  rounding are counted with their worst cases. Nonfinite decoded values and FP16 overflow are refused.
- Routed experts (128 per stage, top-3): MXFP4 -- int8 bytes holding two E2M1 codes (low nibble = even element)
  and one E8M0 scale per 32 inputs. Kept as the original bytes (``tensorfold.cuda.sm70.mxfp4_ds41``).
- BF16 norms, the router gate and the confidence head: widened to FP32 (exact; upstream keeps them FP32).
  The Markov embedding and head stay in their exact BF16 (upstream rounds the embedding to FP16, which the audit
  shows is lossy; ``MarkovGreedy`` reads the head's BF16 bits). HC (mHC) coefficients, sinks and gate biases are FP32.
- Unused: ``ffn.gate.bias_vl`` (vision routing bias; upstream ``draft_weight_name`` drops it).

Layout (TP8 attention/dense like ``weights.py``): rank r holds 8 query heads (wq_b rows [4096 r, +4096), sinks
[8 r, +8)), output group r (wo_a rows [1024 r, +1024)), wo_b input columns [1024 r, +1024), shared-expert rows
[288 r, +288), main_proj output rows [640 r, +640) and Markov head vocabulary rows [16160 r, +16160) (the target
head's vocabulary shard); wq_a, wkv, norms, gate, HC, the Markov embedding and the confidence head are replicated. Experts are
ETP4 x EP2: group g = r // ETP holds experts [64 g, +64), slice t = r % ETP holds gate/up rows [576 t, +576) and
down input columns [576 t, +576). ``world == 1`` (rank 0) holds everything (unit tests, single-GPU checks).
"""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

STAGES = 3
HIDDEN = 5120
EXPERTS = 128
TOP_K = 3
INTER = 2304
HEADS = 64
HEAD_DIM = 512
Q_LORA = 1280
O_GROUPS = 8
O_LORA = 1024
VOCAB = 129280
MARKOV_RANK = 256
BLOCK = 5
NOISE = 128799
TAPS = (37, 38, 39)
_DENSE_SHAPES = {
    "attn.wq_a": (Q_LORA, HIDDEN),
    "attn.wq_b": (HEADS * HEAD_DIM, Q_LORA),
    "attn.wkv": (HEAD_DIM, HIDDEN),
    "attn.wo_a": (O_GROUPS * O_LORA, HEADS * HEAD_DIM // O_GROUPS),
    "attn.wo_b": (HIDDEN, O_GROUPS * O_LORA),
    "ffn.shared_experts.w1": (INTER, HIDDEN),
    "ffn.shared_experts.w2": (HIDDEN, INTER),
    "ffn.shared_experts.w3": (INTER, HIDDEN),
    "main_proj": (HIDDEN, len(TAPS) * HIDDEN),
}
FP16_MAX = 65504.0
FP16_MIN_NORMAL = 2.0 ** -14


# ------------------------------------------------------------------------------------------ safetensors
_DTYPES = {"F32": (np.float32, 4), "BF16": (np.uint16, 2), "F16": (np.float16, 2), "F8_E4M3": (np.uint8, 1),
           "F8_E8M0": (np.uint8, 1), "I8": (np.uint8, 1), "U8": (np.uint8, 1)}


class Checkpoint:
    """``mtp.*`` tensors of a safetensors index; read-only maps, slices copied out on read."""

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        index = json.loads((self.directory / "model.safetensors.index.json").read_text())["weight_map"]
        self.entries: dict[str, tuple[Path, str, tuple, int]] = {}
        files = sorted({f for name, f in index.items() if name.startswith("mtp.")})
        self._maps: dict[Path, np.memmap] = {}
        for name in files:
            path = (self.directory / name).resolve(strict=True)
            with open(path, "rb") as stream:
                size = struct.unpack("<Q", stream.read(8))[0]
                if not 2 <= size <= 64 << 20:
                    raise ValueError(f"{path}: implausible safetensors header size {size}")
                header = json.loads(stream.read(size))
            header.pop("__metadata__", None)
            total = path.stat().st_size
            for tensor, meta in header.items():
                if not tensor.startswith("mtp.") or index.get(tensor) != name:
                    continue
                kind = meta["dtype"]
                if kind not in _DTYPES:
                    raise ValueError(f"{tensor}: unsupported dtype {kind}")
                start, end = meta["data_offsets"]
                shape = tuple(int(v) for v in meta["shape"])
                if not 0 <= start <= end or 8 + size + end > total or end - start != math.prod(shape) * _DTYPES[kind][1]:
                    raise ValueError(f"{tensor}: data range does not match its shape/dtype")
                self.entries[tensor] = (path, kind, shape, 8 + size + start)
        missing = sorted(n for n, f in index.items() if n.startswith("mtp.") and n not in self.entries)
        if missing:
            raise ValueError(f"{self.directory}: index tensors missing from their shards: {missing[:4]}")

    def kind(self, name):
        return self.entries[name][1]

    def shape(self, name):
        return self.entries[name][2]

    def array(self, name) -> np.ndarray:
        """A read-only view of the mapped tensor (numpy dtype of ``_DTYPES``)."""
        path, kind, shape, offset = self.entries[name]
        if path not in self._maps:
            self._maps[path] = np.memmap(path, dtype=np.uint8, mode="r")
        dtype, width = _DTYPES[kind]
        raw = self._maps[path][offset:offset + math.prod(shape) * width]
        return raw.view(dtype).reshape(shape)


def bf16_to_fp32(bits: np.ndarray) -> torch.Tensor:
    return torch.from_numpy((bits.astype(np.uint32) << 16).view(np.float32).copy())


def fp8_block_decode(codes: np.ndarray, scales: np.ndarray, device="cpu") -> torch.Tensor:
    """FP8 E4M3FN [N, K] x E8M0 [N/32, K/32] -> exact FP32 values (2^(s-127); fp8 NaN codes are refused), computed
    on ``device`` (the same exact operations on any device; the loader decodes on the GPU)."""
    n, k = codes.shape
    if scales.shape != (-(-n // 32), -(-k // 32)):
        raise ValueError("FP8 block scales do not cover the weight in 32x32 blocks")
    c = torch.from_numpy(np.array(codes)).to(device)
    if bool(((c & 0x7F) == 0x7F).any()):
        raise ValueError("FP8 weight holds NaN codes")
    values = c.view(torch.float8_e4m3fn).float()
    exponent = torch.from_numpy(np.array(scales)).to(device).to(torch.int32)
    scale = torch.ldexp(torch.ones((), dtype=torch.float64, device=exponent.device), exponent.double() - 127).float()
    scale = scale.repeat_interleave(32, 0)[:n].repeat_interleave(32, 1)[:, :k]
    return values * scale                       # power-of-two scaling: exact in FP32 (2^-127 is FP32 subnormal)


def _dense_fp8_arrays(ck: Checkpoint, name: str, shape: tuple[int, int], rows: slice | None, cols: slice | None):
    weight, scale = name + ".weight", name + ".scale"
    if ck.kind(weight) != "F8_E4M3" or ck.kind(scale) != "F8_E8M0":
        raise ValueError(f"{name}: expected FP8 E4M3 weight with E8M0 block scales")
    if ck.shape(weight) != shape:
        raise ValueError(f"{name}: FP8 weight shape {ck.shape(weight)} differs from {shape}")
    if ck.shape(scale) != tuple(-(-size // 32) for size in shape):
        raise ValueError(f"{name}: FP8 block scales do not cover the weight in 32x32 blocks")

    code_slices, scale_slices = [], []
    for size, owned in zip(shape, (rows, cols)):
        if owned is None:
            code_slices.append(slice(None))
            scale_slices.append(slice(None))
            continue
        start, stop = owned.start, owned.stop
        if owned.step is not None or not 0 <= start < stop <= size or start % 32 or (stop % 32 and stop != size):
            raise ValueError(f"{name}: invalid FP8 block ownership {owned} for extent {size}")
        code_slices.append(owned)
        scale_slices.append(slice(start // 32, -(-stop // 32)))
    return ck.array(weight)[tuple(code_slices)], ck.array(scale)[tuple(scale_slices)]


# ---------------------------------------------------------------------------------------------- audit
@dataclass
class TensorAudit:
    name: str
    source: str            # checkpoint dtype
    target: str            # kept dtype
    count: int
    nonzero: int = 0
    overflow: int = 0      # finite values beyond FP16 (become inf)
    flushed: int = 0       # nonzero values that become 0 in FP16
    subnormal: int = 0     # nonzero values kept as FP16 subnormals
    inexact: int = 0       # values whose FP16 differs from the exact value (includes flushed/subnormal rounding)
    max_abs: float = 0.0
    min_abs_nonzero: float = 0.0
    worst_rel: float = 0.0  # largest |fp16 - exact| / |exact| among nonzero values
    worst_value: float = 0.0

    def line(self):
        return (f"{self.name:48s} {self.source:>7s}->{self.target:<5s} n={self.count:>11d} max={self.max_abs:.4g} "
                f"min_nz={self.min_abs_nonzero:.4g} overflow={self.overflow} flushed={self.flushed} "
                f"subnormal={self.subnormal} inexact={self.inexact} worst_rel={self.worst_rel:.3g}"
                f"@{self.worst_value:.4g}")


def audit_fp16(name: str, source: str, exact: torch.Tensor, *,
               keep_widened: bool = False) -> tuple[torch.Tensor, TensorAudit]:
    """Audit FP16 rounding and optionally retain the exact widening used by the audit."""
    half = exact.half()
    a = TensorAudit(name, source, "F16", exact.numel())
    magnitude = exact.abs()
    nz = magnitude != 0
    a.nonzero = int(nz.sum())
    a.max_abs = float(magnitude.max()) if exact.numel() else 0.0
    a.min_abs_nonzero = float(magnitude[nz].min()) if a.nonzero else 0.0
    back = half.float()
    a.overflow = int((torch.isinf(back) & torch.isfinite(exact)).sum())
    a.flushed = int((nz & (back == 0)).sum())
    a.subnormal = int((nz & (back != 0) & (back.abs() < FP16_MIN_NORMAL)).sum())
    differ = back != exact
    a.inexact = int(differ.sum())
    if a.inexact:
        rel = torch.where(nz & differ, (back - exact).abs() / magnitude.clamp_min(1e-45), torch.zeros_like(exact))
        rel = torch.where(torch.isinf(back), torch.full_like(rel, float("inf")), rel)
        at = int(rel.flatten().argmax())
        a.worst_rel, a.worst_value = float(rel.flatten()[at]), float(exact.flatten()[at])
    return (back if keep_widened else half), a


@dataclass
class Audit:
    tensors: list = field(default_factory=list)
    expert_scale_min: int = 255
    expert_scale_max: int = 0
    expert_scale_hist: dict = field(default_factory=dict)

    def add(self, a: TensorAudit):
        self.tensors.append(a)

    def problems(self):
        return [a for a in self.tensors if a.overflow or a.flushed or a.inexact]

    def report(self):
        lines = [a.line() for a in self.tensors]
        lines.append(f"expert E8M0 scale exponents s-127 in [{self.expert_scale_min - 127}, "
                     f"{self.expert_scale_max - 127}]")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------- layout
@dataclass(frozen=True)
class Layout:
    rank: int = 0
    world: int = 1
    etp: int = 1

    def __post_init__(self):
        if self.world not in (1, 8) or not 0 <= self.rank < self.world or self.etp not in (1, 4) \
                or (self.world == 1) != (self.etp == 1):
            raise ValueError("DSpark head layouts: world 1 (etp 1) or TP8 x (ETP4 x EP2)")

    @property
    def ep(self):
        return self.world // self.etp

    @property
    def heads(self):
        return HEADS // self.world

    @property
    def experts(self):
        return range((self.rank // self.etp) * (EXPERTS // self.ep), (self.rank // self.etp + 1) * (EXPERTS // self.ep))

    @property
    def expert_rows(self):
        part = INTER // self.etp
        return range((self.rank % self.etp) * part, (self.rank % self.etp + 1) * part)

    def rows(self, total):
        part = total // self.world
        return slice(self.rank * part, (self.rank + 1) * part)


@dataclass
class Stage:
    # TP8 serving may retain widened FP16 values for the four requestwise attention banks.
    index: int
    wq_a: torch.Tensor          # [1280, 5120]
    q_norm: torch.Tensor        # FP32 [1280]
    wq_b: torch.Tensor          # [heads*512, 1280]
    wkv: torch.Tensor           # FP16 [512, 5120]
    kv_norm: torch.Tensor       # FP32 [512]
    wo_a: torch.Tensor          # [groups, 1024, 4096]
    wo_b: torch.Tensor          # [5120, groups*1024]
    sink: torch.Tensor          # FP32 [heads]
    attn_norm: torch.Tensor     # FP32 [5120]
    ffn_norm: torch.Tensor      # FP32 [5120]
    hc: dict                    # FP32 attn_fn [24,20480], attn_scale [3], attn_base [24], ffn_*
    gate: torch.Tensor          # FP32 [128, 5120]
    gate_bias: torch.Tensor     # FP32 [128]
    sh_gate: torch.Tensor       # FP16 [rows, 5120]
    sh_up: torch.Tensor         # FP16 [rows, 5120]
    sh_down: torch.Tensor       # FP16 [5120, rows]
    exp_gate: torch.Tensor      # uint8 [E_local, R, 2560] MXFP4 codes
    exp_gate_scale: torch.Tensor  # uint8 [E_local, R, 160]
    exp_up: torch.Tensor
    exp_up_scale: torch.Tensor
    exp_down: torch.Tensor      # uint8 [E_local, 5120, R/2]
    exp_down_scale: torch.Tensor  # uint8 [E_local, 5120, R/32]


@dataclass
class DSparkWeights:
    layout: Layout
    identity: str
    main_proj: torch.Tensor     # FP16 [5120 / world, 15360] (this rank's output rows)
    main_norm: torch.Tensor     # FP32 [5120]
    norm: torch.Tensor          # FP32 [5120]
    markov_embed: torch.Tensor  # BF16 [129280, 256] (exact; upstream rounds to FP16, see load)
    markov_head: torch.Tensor   # BF16 [129280 / world, 256] (this rank's vocabulary rows)
    confidence: torch.Tensor    # FP32 [1, 5376]
    stages: list
    first_expert: int
    audit: Audit

    def nbytes(self):
        tensors = [self.main_proj, self.main_norm, self.norm, self.markov_embed, self.markov_head, self.confidence]
        for s in self.stages:
            tensors += [v for v in vars(s).values() if isinstance(v, torch.Tensor)] + list(s.hc.values())
        return sum(t.numel() * t.element_size() for t in tensors)


def check_config(directory: str | Path) -> dict:
    raw = json.loads((Path(directory) / "config.json").read_text())
    c = raw.get("text_config") or raw
    got = (c["hidden_size"], c["moe_intermediate_size"], c.get("dspark_n_routed_experts"),
           c.get("dspark_num_experts_per_tok"), tuple(c.get("dspark_target_layer_ids", ())), c.get("dspark_block_size"),
           c.get("dspark_markov_rank"), c.get("dspark_noise_token_id"), c["num_attention_heads"], c["head_dim"],
           c["q_lora_rank"], c["o_groups"], c["o_lora_rank"], c["vocab_size"], c["sliding_window"], c["hc_mult"],
           c.get("num_nextn_predict_layers"), c["swiglu_limit"], c["routed_scaling_factor"])
    want = (HIDDEN, INTER, EXPERTS, TOP_K, TAPS, BLOCK, MARKOV_RANK, NOISE, HEADS, HEAD_DIM, Q_LORA, O_GROUPS, O_LORA,
            VOCAB, 128, 4, STAGES, 10.0, 1.5)
    if got != want:
        raise ValueError(f"DSpark head geometry {got} differs from the admitted V4.1-1017 contract {want}")
    return c


def load(directory: str | Path, *, layout: Layout = Layout(), device="cuda", config_dir: str | Path | None = None,
         audit_only: bool = False, keep_requestwise_fp32: bool = False) -> DSparkWeights | Audit:
    """Load one rank's DSpark head, or audit the full dense banks on CPU with ``audit_only``."""
    check_config(config_dir or directory)
    ck = Checkpoint(directory)
    audit = Audit()
    device = torch.device(device)

    def put(t):
        if audit_only:
            return None
        t = t.contiguous().to(device)
        # Do not retain unrelated rows when a parameter is a view into a larger allocation.
        return t.clone() if t.untyped_storage().nbytes() > t.numel() * t.element_size() else t

    def fp32(name):
        kind = ck.kind(name)
        arr = ck.array(name)
        if kind == "F32":
            value = torch.from_numpy(np.array(arr))
        elif kind == "BF16":
            value = bf16_to_fp32(arr)
        else:
            raise ValueError(f"{name}: expected F32/BF16, found {kind}")
        audit.add(TensorAudit(name, kind, "F32", value.numel(), int((value != 0).sum()),
                              max_abs=float(value.abs().max())))
        return value

    def dense(name, rows=None, cols=None, *, keep_widened=False):
        shape = _DENSE_SHAPES[name.split(".", 2)[2]]
        codes, scales = _dense_fp8_arrays(ck, name, shape, None if audit_only else rows, None if audit_only else cols)
        bank, a = audit_fp16(name, "F8_E4M3", fp8_block_decode(codes, scales, "cpu" if audit_only else device),
                             keep_widened=keep_widened)
        audit.add(a)
        if not math.isfinite(a.max_abs):
            raise ValueError(f"{name}: nonfinite values after FP8 decode; FP16 bank refused")
        if a.overflow:
            raise ValueError(f"{name}: {a.overflow} values overflow FP16 (max {a.max_abs}); FP16 bank refused")
        return bank

    lay = layout
    stages = []
    first = lay.experts.start
    for i in range(STAGES):
        p = f"mtp.{i}."
        keep_widened = keep_requestwise_fp32 and not audit_only and i < 2
        hc = {f"{b}_{part}": put(fp32(p + f"hc_{b}_{part}")) for b in ("attn", "ffn") for part in ("fn", "scale", "base")}
        heads = lay.rows(HEADS)
        wq_b = dense(p + "attn.wq_b", rows=slice(heads.start * HEAD_DIM, heads.stop * HEAD_DIM),
                     keep_widened=keep_widened)
        groups = lay.rows(O_GROUPS)
        wo_a = put(dense(p + "attn.wo_a", rows=slice(groups.start * O_LORA, groups.stop * O_LORA),
                         keep_widened=keep_widened))
        if wo_a is not None:
            wo_a = wo_a.view(groups.stop - groups.start, O_LORA, -1)
        wo_b = dense(p + "attn.wo_b", cols=slice(groups.start * O_LORA, groups.stop * O_LORA),
                     keep_widened=keep_widened)
        shared = lay.rows(INTER)
        sink = fp32(p + "attn.attn_sink")[heads]
        expert = {}
        rows = lay.expert_rows
        for w, key in (("w1", "gate"), ("w3", "up"), ("w2", "down")):
            codes, scales = [], []
            for e in lay.experts:
                n = p + f"ffn.experts.{e}.{w}"
                if ck.kind(n + ".weight") != "I8" or ck.kind(n + ".scale") != "F8_E8M0":
                    raise ValueError(f"{n}: expected MXFP4 int8 pairs with E8M0 scales")
                c, s = ck.array(n + ".weight"), ck.array(n + ".scale")
                want = ((INTER, HIDDEN // 2), (INTER, HIDDEN // 32)) if w != "w2" else \
                    ((HIDDEN, INTER // 2), (HIDDEN, INTER // 32))
                if (c.shape, s.shape) != want:
                    raise ValueError(f"{n}: MXFP4 shapes {c.shape}/{s.shape} differ from {want}")
                if w == "w2":
                    c, s = c[:, rows.start // 2:rows.stop // 2], s[:, rows.start // 32:rows.stop // 32]
                else:
                    c, s = c[rows.start:rows.stop], s[rows.start:rows.stop]
                hist = np.bincount(np.asarray(s).ravel(), minlength=256)
                for v in np.nonzero(hist)[0]:
                    audit.expert_scale_hist[int(v)] = audit.expert_scale_hist.get(int(v), 0) + int(hist[v])
                if not audit_only:
                    codes.append(torch.from_numpy(np.array(c)))
                    scales.append(torch.from_numpy(np.array(s)))
            if not audit_only:
                expert[key] = torch.stack(codes).to(device)
                expert[key + "_scale"] = torch.stack(scales).to(device)
        if audit.expert_scale_hist:
            audit.expert_scale_min = min(audit.expert_scale_hist)
            audit.expert_scale_max = max(audit.expert_scale_hist)
        stage = Stage(
            index=i,
            wq_a=put(dense(p + "attn.wq_a", keep_widened=keep_widened)), q_norm=put(fp32(p + "attn.q_norm.weight")),
            wq_b=put(wq_b), wkv=put(dense(p + "attn.wkv")), kv_norm=put(fp32(p + "attn.kv_norm.weight")),
            wo_a=wo_a, wo_b=put(wo_b), sink=put(sink),
            attn_norm=put(fp32(p + "attn_norm.weight")), ffn_norm=put(fp32(p + "ffn_norm.weight")), hc=hc,
            gate=put(fp32(p + "ffn.gate.weight")), gate_bias=put(fp32(p + "ffn.gate.bias")),
            sh_gate=put(dense(p + "ffn.shared_experts.w1", rows=shared)),
            sh_up=put(dense(p + "ffn.shared_experts.w3", rows=shared)),
            sh_down=put(dense(p + "ffn.shared_experts.w2", cols=shared)),
            exp_gate=expert.get("gate"), exp_gate_scale=expert.get("gate_scale"),
            exp_up=expert.get("up"), exp_up_scale=expert.get("up_scale"),
            exp_down=expert.get("down"), exp_down_scale=expert.get("down_scale"))
        stages.append(stage)
    main_proj = put(dense("mtp.0.main_proj", rows=lay.rows(HIDDEN)))
    main_norm = put(fp32("mtp.0.main_norm.weight"))
    norm = put(fp32("mtp.2.norm.weight"))
    # Upstream loads this BF16 table into an FP16 nn.Embedding; the audit shows that rounding flushes/overflows
    # values (see the report), so it is kept in its exact checkpoint BF16 (same bytes as FP16; every consumer
    # widens the gathered row to FP32).
    embed_bits = ck.array("mtp.2.markov_head.embed.weight")
    _, a = audit_fp16("mtp.2.markov_head.embed.weight", "BF16", bf16_to_fp32(embed_bits))
    a.target = "BF16"
    audit.add(a)
    embed = torch.from_numpy(np.array(embed_bits).view(np.int16)).view(torch.bfloat16)
    head_bits = ck.array("mtp.2.markov_head.head.weight")
    head = torch.from_numpy(np.array(head_bits[lay.rows(VOCAB)]).view(np.int16)).view(torch.bfloat16)
    confidence = fp32("mtp.2.confidence_head.proj.weight")
    if audit_only:
        return audit
    identity = f"{Path(directory).resolve()}:{sorted(set(e[0].name for e in ck.entries.values()))}"
    return DSparkWeights(lay, identity, main_proj, main_norm, norm, put(embed), put(head), put(confidence), stages,
                         first, audit)

"""Exact header plans and bounded CPU weight loading for optional DSpark.

The reference execution is deliberately unsharded. TP plans describe the future
shared-engine adapter, including replicated Markov input embeddings/projector.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path

from .dspark_config import DSparkConfig

SOURCE_BYTES = {"BF16": 2, "F16": 2, "F32": 4}
LOAD_CHUNK = 8 << 20


def tensor_shapes(c):
    shapes = {
        "fc.weight": (c.hidden, len(c.taps) * c.hidden),
        "hidden_norm.weight": (c.hidden,),
        "norm.weight": (c.hidden,),
        "markov_head.markov_w1.weight": (c.vocab, c.rank),
        "markov_head.markov_w2.weight": (c.vocab, c.rank),
    }
    if c.confidence:
        shapes.update({"confidence_head.proj.weight": (1, c.hidden + c.rank), "confidence_head.proj.bias": (1,)})
    for layer in range(c.layers):
        base = f"layers.{layer}."
        shapes.update(
            {
                base + "input_layernorm.weight": (c.hidden,),
                base + "post_attention_layernorm.weight": (c.hidden,),
                base + "self_attn.q_norm.weight": (c.head_dim,),
                base + "self_attn.k_norm.weight": (c.head_dim,),
                base + "self_attn.q_proj.weight": (c.heads * c.head_dim, c.hidden),
                base + "self_attn.k_proj.weight": (c.kv_heads * c.head_dim, c.hidden),
                base + "self_attn.v_proj.weight": (c.kv_heads * c.head_dim, c.hidden),
                base + "self_attn.o_proj.weight": (c.hidden, c.heads * c.head_dim),
                base + "mlp.gate_proj.weight": (c.intermediate, c.hidden),
                base + "mlp.up_proj.weight": (c.intermediate, c.hidden),
                base + "mlp.down_proj.weight": (c.hidden, c.intermediate),
            }
        )
    return shapes


def _f32(name):
    return "norm" in name or name.startswith("confidence_head.")


@dataclass(frozen=True)
class TensorPlan:
    name: str
    shape: tuple[int, ...]
    source_shape: tuple[int, ...]
    rows: tuple[int, int] | None = None
    columns: tuple[int, int] | None = None
    f32: bool = False

    @property
    def nbytes(self):
        return math.prod(self.shape) * (4 if self.f32 else 2)


def plan_weights(config, header, *, tp=(0, 1)):
    rank, world = tp
    local = config.partition(rank, world)
    expected = tensor_shapes(config)
    entries = {name: info for name, info in header.items() if name != "__metadata__"}
    if entries.keys() != expected.keys():
        raise ValueError(
            f"DSpark tensor namespace differs: missing={sorted(expected.keys() - entries.keys())}, extra={sorted(entries.keys() - expected.keys())}"
        )
    spans = []
    plans = []
    for name, shape in expected.items():
        info = entries[name]
        if not isinstance(info, dict) or info.get("shape") != list(shape) or info.get("dtype") not in SOURCE_BYTES:
            raise ValueError(f"DSpark tensor shape/dtype differs: {name}")
        offsets = info.get("data_offsets")
        if (
            not isinstance(offsets, list)
            or len(offsets) != 2
            or any(type(x) is not int or x < 0 for x in offsets)
            or offsets[1] - offsets[0] != math.prod(shape) * SOURCE_BYTES[info["dtype"]]
        ):
            raise ValueError(f"DSpark tensor byte interval differs: {name}")
        spans.append((*offsets, name))
        rows = columns = None
        if name.startswith("layers."):
            if name.endswith("self_attn.q_proj.weight"):
                width = local["heads"] * config.head_dim
                rows = (rank * width, (rank + 1) * width)
            elif name.endswith(("self_attn.k_proj.weight", "self_attn.v_proj.weight")):
                begin = local["first_kv"] * config.head_dim
                rows = (begin, begin + local["kv_heads"] * config.head_dim)
            elif name.endswith("self_attn.o_proj.weight"):
                width = local["heads"] * config.head_dim
                columns = (rank * width, (rank + 1) * width)
            elif name.endswith(("mlp.gate_proj.weight", "mlp.up_proj.weight")):
                width = local["intermediate"]
                rows = (rank * width, (rank + 1) * width)
            elif name.endswith("mlp.down_proj.weight"):
                width = local["intermediate"]
                columns = (rank * width, (rank + 1) * width)
        elif name == "markov_head.markov_w2.weight":
            q, r = divmod(config.vocab, world)
            begin = rank * q + min(rank, r)
            rows = (begin, begin + q + int(rank < r))
        selected = list(shape)
        if rows:
            selected[0] = rows[1] - rows[0]
        if columns:
            selected[1] = columns[1] - columns[0]
        plans.append(TensorPlan(name, tuple(selected), shape, rows, columns, _f32(name)))
    offset = 0
    for lo, hi, name in sorted(spans):
        if lo != offset:
            raise ValueError(f"DSpark payload intervals contain a gap/overlap: {name}")
        offset = hi
    return tuple(plans)


def weight_byte_plan(
    config, header, *, tp=(0, 1), capacity, slots=1, prefill_rows=256, block=None, chunk_bytes=LOAD_CHUNK
):
    plans = plan_weights(config, header, tp=tp)
    local = config.partition(*tp)
    block = config.block if block is None else block
    for name, value in [
        ("capacity", capacity),
        ("slots", slots),
        ("prefill_rows", prefill_rows),
        ("block", block),
        ("chunk_bytes", chunk_bytes),
    ]:
        if type(value) is not int or value <= 0:
            raise ValueError(f"DSpark {name} must be a positive integer")
    if capacity > config.context or block > config.block:
        raise ValueError("DSpark capacity/block exceeds its configured limit")
    if chunk_bytes < max(math.prod(p.source_shape[1:]) * 4 for p in plans):
        raise ValueError("DSpark load chunk must fit one complete source row")
    # K and V: one pair per layer for committed target context. Proposal K/V
    # never persist. One staged accepted-prefix update is permitted per slot.
    kv_row = local["kv_heads"] * config.head_dim
    state = slots * config.layers * capacity * kv_row * 4
    stage = config.layers * prefill_rows * kv_row * 4
    features = prefill_rows * (len(config.taps) * config.hidden * 2 + config.hidden * 12)
    # Conservative pure-Torch reference envelope: FP32 scores/probabilities,
    # concatenated half K/V and FP32 cache operands; base/corrected/sampled
    # proposal distributions and matrix/residual/MLP temporaries coexist.
    attention = 8 * local["heads"] * block * (capacity + block) + 12 * (capacity + block) * kv_row
    proposals = 12 * block * config.vocab + 16 * block * (
        config.hidden + local["intermediate"] + local["heads"] * config.head_dim
    )
    reference_cast = max(math.prod(p.shape) * 4 for p in plans if not p.f32)
    matrix_scratch = 32 * max(p.shape[0] + p.shape[1] for p in plans if len(p.shape) == 2) * 4
    workspace = slots * stage + max(features, attention + proposals) + reference_cast + matrix_scratch
    return {
        "resident_bytes": sum(p.nbytes for p in plans),
        "staging_bytes": 2 * chunk_bytes,
        "host_staging_bytes": 2 * chunk_bytes,
        "state_bytes": state,
        "context_staging_bytes": stage,
        "reference_weight_cast_bytes": reference_cast,
        "matrix_scratch_bytes": matrix_scratch,
        "workspace_bytes": workspace,
        "serving_bytes": sum(p.nbytes for p in plans) + state + workspace,
        "target_shared_weight_bytes": 0,
        "reference_execution_worlds": (1,),
        "planned_world": tp[1],
        "planned_rank": tp[0],
    }


@dataclass
class DSparkWeights:
    config: DSparkConfig
    tensors: dict
    identity: str

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in self.tensors.values())

    def device_tensors(self):
        return tuple(self.tensors.values())


def load_reference(directory, *, target, chunk_bytes=LOAD_CHUNK):
    """Explicit CPU-only load; no remote access or automatic provider selection."""
    import torch
    from safetensors import safe_open
    from tensorfold.cuda.tensor_file import read_header

    c = DSparkConfig.read(directory)
    c.check_target(target)
    file = Path(directory) / "model.safetensors"
    _, header = read_header(file, SOURCE_BYTES)
    plans = plan_weights(c, header)
    if type(chunk_bytes) is not int or chunk_bytes < max(
        math.prod(p.source_shape[1:]) * SOURCE_BYTES[header[p.name]["dtype"]] for p in plans
    ):
        raise ValueError("DSpark load chunk must fit one complete source row")
    tensors = {}
    digest = hashlib.sha256((c.identity + "|torch-f16-f32acc-m32-reference-v1").encode())
    with safe_open(file, framework="pt", device="cpu") as source:
        for plan in plans:
            dtype = torch.float32 if plan.f32 else torch.float16
            result = torch.empty(plan.shape, dtype=dtype)
            row_bytes = math.prod(plan.shape[1:]) * SOURCE_BYTES[header[plan.name]["dtype"]]
            step = max(1, chunk_bytes // row_bytes)
            value = source.get_slice(plan.name)
            digest.update(plan.name.encode() + b"\0" + header[plan.name]["dtype"].encode())
            for start in range(0, plan.shape[0], step):
                raw = value[start : min(start + step, plan.shape[0])]
                if not torch.isfinite(raw).all() or (
                    dtype == torch.float16 and raw.abs().max().float() > torch.finfo(torch.float16).max
                ):
                    raise ValueError(f"DSpark nonfinite or FP16-overflow source tensor: {plan.name}")
                digest.update(raw.contiguous().view(torch.uint8).numpy().tobytes())
                result[start : start + raw.shape[0]].copy_(raw.to(dtype))
            tensors[plan.name] = result
    return DSparkWeights(c, tensors, digest.hexdigest())

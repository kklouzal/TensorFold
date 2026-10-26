"""Bounded GGUF vision import into the pinned Transformers Qwen4Exp layout.

The enclosing coordinated loader owns the content-verified source cohort and
Reader lifetime. This module validates the complete named tensor map before
parameter allocation. It neither constructs another encoder nor chooses a GPU.
"""

from dataclasses import dataclass
import math
from pathlib import Path
from types import MappingProxyType

import numpy as np
import torch

from ..gguf import _shard, decode_rows


@dataclass(frozen=True)
class VisionLayout:
    depth: int
    hidden: int
    heads: int
    intermediate: int
    output: int
    patch: int
    temporal: int
    merge: int
    channels: int
    positions: int
    theta: float

    def __post_init__(self):
        sizes = (self.depth, self.hidden, self.heads, self.intermediate, self.output,
                 self.patch, self.temporal, self.merge, self.channels, self.positions)
        if (any(type(v) is not int or v <= 0 for v in sizes) or self.hidden % self.heads
                or (self.hidden // self.heads) % 4 or self.temporal != 2 or self.channels != 3
                or math.isqrt(self.positions) ** 2 != self.positions
                or type(self.theta) not in (int, float) or not math.isfinite(self.theta) or self.theta <= 1):
            raise ValueError("invalid declared Qwen4Exp vision geometry/rotary policy")

    def config_fields(self):
        return {"depth": self.depth, "hidden_size": self.hidden, "num_heads": self.heads,
                "intermediate_size": self.intermediate, "out_hidden_size": self.output,
                "patch_size": self.patch, "temporal_patch_size": self.temporal,
                "spatial_merge_size": self.merge, "in_channels": self.channels,
                "num_position_embeddings": self.positions, "hidden_act": "gelu_pytorch_tanh",
                "rope_parameters": {"rope_type": "axial", "rope_theta": self.theta}}


@dataclass(frozen=True)
class VisionWeight:
    sources: tuple
    shape: tuple
    temporal_stack: bool = False

    @property
    def resident_bytes(self):
        return math.prod(self.shape) * 2


@dataclass(frozen=True)
class VisionPlan:
    path: Path
    layout: VisionLayout
    weights: object

    @property
    def resident_bytes(self):
        return sum(weight.resident_bytes for weight in self.weights.values())


def plan_mmproj(path, layout):
    """Bind one declared vision manifest; reject unknown/missing tensors before allocation."""
    if not isinstance(layout, VisionLayout):
        raise ValueError("vision import requires a resolved VisionLayout")
    path = Path(path).resolve(strict=True)
    metadata, tensors = _shard(path)
    expected = {"general.architecture": "clip", "general.type": "mmproj", "clip.has_vision_encoder": True,
                "clip.projector_type": "qwen3vl_merger", "clip.use_gelu": True,
                "clip.vision.projection_dim": layout.output, "clip.vision.patch_size": layout.patch,
                "clip.vision.embedding_length": layout.hidden, "clip.vision.feed_forward_length": layout.intermediate,
                "clip.vision.block_count": layout.depth, "clip.vision.attention.head_count": layout.heads,
                "clip.vision.spatial_merge_size": layout.merge,
                "clip.vision.image_mean": (.5, .5, .5), "clip.vision.image_std": (.5, .5, .5)}
    if any(type(metadata.get(key)) is not type(value) or metadata[key] != value for key, value in expected.items()):
        raise ValueError("mmproj metadata disagrees with the declared Qwen4Exp vision profile")
    deepstack = metadata.get("clip.vision.is_deepstack_layers")
    if (type(deepstack) is not tuple or len(deepstack) != layout.depth
            or any(type(v) is not bool or v for v in deepstack)):
        raise ValueError("this vision manifest requires the single final merger without deepstack")
    epsilon = metadata.get("clip.vision.attention.layer_norm_epsilon")
    if (type(epsilon) not in (int, float) or not math.isfinite(epsilon)
            or not 0 < epsilon <= float(np.finfo(np.float32).max)
            or np.float32(epsilon) != np.float32(1e-6)):
        raise ValueError("mmproj LayerNorm must match the pinned vision epsilon")
    sources = {tensor.name: tensor for tensor in tensors}
    if len(sources) != len(tensors):
        raise ValueError("duplicate mmproj tensor name")
    used, weights = set(), {}

    def add(target, names, shapes, *, temporal_stack=False):
        selected = []
        for name, shape in zip(names, shapes, strict=True):
            if name not in sources or sources[name].shape != shape:
                raise ValueError("missing or incompatible mmproj tensor: " + name)
            source = sources[name]
            if source.encoding not in ("BF16", "F32", "F16"):
                raise ValueError("vision source encoding requires its own importer qualification: " + source.encoding)
            selected.append(source)
            used.add(name)
        shape = (layout.hidden, layout.channels, layout.temporal, layout.patch, layout.patch) if temporal_stack else shapes[0]
        weights[target] = VisionWeight(tuple(selected), shape, temporal_stack)

    for index in range(layout.depth):
        base, target = f"v.blk.{index}.", f"blocks.{index}."
        for source, name, shape in (("attn_qkv", "attn.qkv", (3 * layout.hidden, layout.hidden)),
                                    ("attn_out", "attn.proj", (layout.hidden, layout.hidden)),
                                    ("ffn_up", "mlp.linear_fc1", (layout.intermediate, layout.hidden)),
                                    ("ffn_down", "mlp.linear_fc2", (layout.hidden, layout.intermediate)),
                                    ("ln1", "norm1", (layout.hidden,)), ("ln2", "norm2", (layout.hidden,))):
            for suffix, dimensions in (("weight", shape), ("bias", (shape[0],))):
                add(target + name + "." + suffix, (base + source + "." + suffix,), (dimensions,))
    patch_shape = (layout.hidden, layout.channels, layout.patch, layout.patch)
    add("patch_embed.proj.weight", ("v.patch_embd.weight", "v.patch_embd.weight.1"),
        (patch_shape, patch_shape), temporal_stack=True)
    add("patch_embed.proj.bias", ("v.patch_embd.bias",), ((layout.hidden,),))
    add("pos_embed.weight", ("v.position_embd.weight",), ((layout.positions, layout.hidden),))
    merged = layout.hidden * layout.merge ** 2
    for source, target, shape in (("v.post_ln", "merger.norm", (layout.hidden,)),
                                  ("mm.0", "merger.linear_fc1", (merged, merged)),
                                  ("mm.2", "merger.linear_fc2", (layout.output, merged))):
        for suffix, dimensions in (("weight", shape), ("bias", (shape[0],))):
            add(target + "." + suffix, (source + "." + suffix,), (dimensions,))
    if set(sources) != used:
        raise ValueError("unmapped mmproj tensors: " + ", ".join(sorted(set(sources) - used)))
    return VisionPlan(path, layout, MappingProxyType(weights))


def _parameter(source, reader, staging_bytes):
    if source.row_bytes > staging_bytes:
        raise ValueError("one vision source row exceeds the admitted import staging")
    result = torch.empty(source.shape, dtype=torch.float16, device="cpu")
    rows = result.reshape(source.rows, source.shape[-1])
    for start in range(0, source.rows, max(1, staging_bytes // source.row_bytes)):
        count = min(source.rows - start, staging_bytes // source.row_bytes)
        raw = source.read_rows(reader, start, count, max_bytes=staging_bytes)
        values = decode_rows(source, raw.numpy())
        if not np.isfinite(values).all():
            raise ValueError("nonfinite vision parameter: " + source.name)
        with np.errstate(over="ignore", invalid="ignore"):
            half = values.astype(np.float16)
        if not np.isfinite(half).all():
            raise ValueError("vision parameter overflows FP16: " + source.name)
        rows[start:start + count].copy_(torch.from_numpy(half))
    return result


def read_vision_weights(plan, reader, *, staging_bytes):
    """Return one CPU parameter set for strict assignment to a meta model.

    Source reads are bounded by staging_bytes. Budget overlapping Reader/raw,
    FP32 decode, FP16 conversion and final retained parameters separately; the
    source-read bound alone is not a complete peak-memory claim. Delete this
    dictionary after assignment, before moving the model to its fixed GPU.
    """
    if type(staging_bytes) is not int or staging_bytes <= 0:
        raise ValueError("vision import requires a positive explicit staging bound")
    result = {}
    for name, weight in plan.weights.items():
        parts = [_parameter(source, reader, staging_bytes) for source in weight.sources]
        result[name] = torch.stack(parts, dim=2) if weight.temporal_stack else parts[0]
    return result


def load_cpu_vision_model(plan, reader, *, staging_bytes):
    """Build the pinned model with one FP16 parameter set and FP32 axial buffers.

    The meta constructor retains no initialized parameter copy. Exact library
    rotary construction materializes its two nonpersistent buffers; no target
    YaRN policy enters the independent vision transform. The caller assigns the
    resulting module to its journal before any fixed-device transfer.
    """
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

    config = Qwen4ExpVisionConfig(**plan.layout.config_fields())
    config._attn_implementation = "sdpa"
    with torch.device("meta"):
        model = Qwen4ExpVisionModel(config)
    model.requires_grad_(False)
    parameters = read_vision_weights(plan, reader, staging_bytes=staging_bytes)
    model.load_state_dict(parameters, strict=True, assign=True)
    del parameters
    rotary = model.rotary_pos_emb
    # The pinned implementation's intermediate arange follows the default
    # device even though its final vector accepts a device argument.
    with torch.device("cpu"):
        frequencies, scaling = rotary.compute_axial_rope_parameters(config, device="cpu")
    if frequencies.dtype != torch.float32 or not torch.isfinite(frequencies).all():
        raise ValueError("vision library produced an invalid axial frequency vector")
    rotary.register_buffer("inv_freq", frequencies, persistent=False)
    rotary.register_buffer("original_inv_freq", frequencies.clone(), persistent=False)
    rotary.attention_scaling = scaling
    if any(parameter.is_meta for parameter in model.parameters()) or any(buffer.is_meta for buffer in model.buffers()):
        raise ValueError("vision model retains an unmaterialized parameter or buffer")
    return model.eval()

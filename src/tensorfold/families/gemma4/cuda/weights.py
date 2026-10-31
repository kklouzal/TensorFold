"""Gemma 4 MLX affine weights: validate the complete text boundary before uploads.

The on-disk checkpoint is immutable for this reader's lifetime. Packed codes,
scales and affine offsets stay authoritative; only lossless native layouts are
permitted by the linear primitives. Vision tensors remain the vision owner's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PREFIX = "language_model.model."
TAP_LAYERS = (1, 6, 11, 17, 22, 27)


def _integer(raw: dict, name: str, *, minimum: int = 1, maximum: int = 1_048_576) -> int:
    value = raw.get(name)
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"Gemma4 {name} must be an integer in {minimum}..{maximum}")
    return value


@dataclass(frozen=True)
class Config:
    hidden: int
    layers: int
    vocab: int
    heads: int
    kv_heads: int
    global_kv_heads: int
    head_dim: int
    global_head_dim: int
    intermediate: int
    moe_width: int
    experts: int
    top_k: int
    window: int
    max_position: int
    eps: float
    softcap: float
    layer_types: tuple[str, ...]
    rope_theta: float
    global_rope_theta: float
    global_rotary_fraction: float
    vision_bidirectional: bool

    @classmethod
    def from_dict(cls, raw: dict) -> "Config":
        if not isinstance(raw, dict):
            raise ValueError("Gemma4 config must be an object")
        text = raw.get("text_config") or raw
        if not isinstance(text, dict) or text.get("model_type") not in ("gemma4_text", "gemma4"):
            raise ValueError("Gemma4 requires gemma4_text metadata")
        for key in ("hidden_size_per_layer_input", "num_kv_shared_layers"):
            if text.get(key, 0) != 0:
                raise ValueError(f"Gemma4 CUDA does not implement {key}")
        if text.get("enable_moe_block") is not True or text.get("attention_k_eq_v") is not True:
            raise ValueError("Gemma4 CUDA requires MoE layers and full-attention K=V")
        if text.get("tie_word_embeddings", raw.get("tie_word_embeddings")) is not True:
            raise ValueError("Gemma4 CUDA requires tied word embeddings")
        if text.get("attention_bias", False) is not False or text.get("attention_dropout", 0.0) != 0.0:
            raise ValueError("Gemma4 CUDA requires bias-free, dropout-free inference attention")
        if text.get("use_double_wide_mlp", False) is not False:
            raise ValueError("Gemma4 CUDA requires one fixed dense MLP width")
        if text.get("hidden_activation") != "gelu_pytorch_tanh":
            raise ValueError("Gemma4 CUDA requires the tanh GeGLU activation")
        if text.get("use_bidirectional_attention") not in (None, False, "vision"):
            raise ValueError("Gemma4 CUDA requires causal text or visual-block attention")
        if text.get("rope_traditional", False) is not False:
            raise ValueError("Gemma4 CUDA uses split-half RoPE")
        count = _integer(text, "num_hidden_layers", maximum=256)
        kinds = text.get("layer_types")
        if not isinstance(kinds, list) or len(kinds) != count or any(
                x not in ("sliding_attention", "full_attention") for x in kinds):
            raise ValueError("Gemma4 layer_types must describe every attention layer")
        rope = text.get("rope_parameters")
        if not isinstance(rope, dict):
            raise ValueError("Gemma4 requires explicit per-kind RoPE parameters")
        sliding, full = rope.get("sliding_attention", {}), rope.get("full_attention", {})
        if not isinstance(sliding, dict) or not isinstance(full, dict):
            raise ValueError("Gemma4 RoPE parameters must be objects")
        if (sliding.get("rope_type") != "default" or full.get("rope_type") != "proportional"
                or sliding.get("partial_rotary_factor", 1.0) != 1.0
                or sliding.get("factor", 1.0) != 1.0 or full.get("factor", 1.0) != 1.0):
            raise ValueError("Gemma4 CUDA requires default sliding and unscaled proportional full RoPE")
        floats = dict(eps=text.get("rms_norm_eps"), softcap=text.get("final_logit_softcapping"),
                      rope_theta=sliding.get("rope_theta"), global_rope_theta=full.get("rope_theta"),
                      global_rotary_fraction=full.get("partial_rotary_factor"))
        for name, value in floats.items():
            if type(value) not in (float, int) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"Gemma4 {name} must be finite and positive")
        c = cls(hidden=_integer(text, "hidden_size"), layers=count, vocab=_integer(text, "vocab_size"),
                heads=_integer(text, "num_attention_heads", maximum=256),
                kv_heads=_integer(text, "num_key_value_heads", maximum=256),
                global_kv_heads=_integer(text, "num_global_key_value_heads", maximum=256),
                head_dim=_integer(text, "head_dim", maximum=512),
                global_head_dim=_integer(text, "global_head_dim", maximum=512),
                intermediate=_integer(text, "intermediate_size"), moe_width=_integer(text, "moe_intermediate_size"),
                experts=_integer(text, "num_experts", maximum=1024), top_k=_integer(text, "top_k_experts", maximum=128),
                window=_integer(text, "sliding_window"),
                max_position=_integer(text, "max_position_embeddings", maximum=2**31 - 1),
                layer_types=tuple(kinds), vision_bidirectional=text.get("use_bidirectional_attention") == "vision",
                **{name: float(value) for name, value in floats.items()})
        if (c.hidden % 32 or c.intermediate % 32 or c.moe_width % 32
                or c.head_dim not in (64, 128, 256, 512) or c.global_head_dim not in (64, 128, 256, 512)
                or c.heads % c.kv_heads or c.heads % c.global_kv_heads
                or c.heads // c.kv_heads > 16 or c.heads // c.global_kv_heads > 16 or c.top_k > c.experts):
            raise ValueError("Gemma4 dimensions must respect quant groups, head groups and expert bounds")
        rotated = c.global_head_dim * c.global_rotary_fraction
        if not 0 < rotated <= c.global_head_dim or rotated != int(rotated) or int(rotated) % 2:
            raise ValueError("Gemma4 proportional rotary dimension must be positive, even and within its head")
        return c

    def sliding(self, layer: int) -> bool:
        return self.layer_types[layer] == "sliding_attention"

    def attention_geometry(self, layer: int) -> tuple[int, int]:
        return (self.kv_heads, self.head_dim) if self.sliding(layer) else (self.global_kv_heads, self.global_head_dim)


def quant_spec(raw: dict, name: str) -> tuple[int, int]:
    """Resolve each projection's explicit MLX override before consulting defaults."""
    quant = raw.get("quantization") or raw.get("quantization_config")
    if not isinstance(quant, dict) or quant.get("mode", "affine") != "affine":
        raise ValueError("Gemma4 CUDA requires MLX affine quantization metadata")
    spec = quant.get(name, {})
    if not isinstance(spec, dict) or spec.get("mode", "affine") != "affine":
        raise ValueError(f"{name}: invalid affine quantization override")
    bits, group = spec.get("bits", quant.get("bits")), spec.get("group_size", quant.get("group_size"))
    if type(bits) is not int or bits not in (4, 6) or type(group) is not int or group != 64:
        raise ValueError(f"{name}: Gemma4 CUDA reads original affine4/6 in group64")
    return bits, group


def tensor_contract(c: Config) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    """Logical affine matrix shapes and plain tensor shapes, without runtime imports."""
    matrices = {PREFIX + "embed_tokens": (c.vocab, c.hidden)}
    plain = {PREFIX + "norm.weight": (c.hidden,)}
    norms = ("input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
             "pre_feedforward_layernorm_2", "post_feedforward_layernorm", "post_feedforward_layernorm_1",
             "post_feedforward_layernorm_2")
    for i in range(c.layers):
        p = f"{PREFIX}layers.{i}."
        hk, d = c.attention_geometry(i)
        for name, shape in (("self_attn.q_proj", (c.heads * d, c.hidden)),
                            ("self_attn.k_proj", (hk * d, c.hidden)),
                            ("self_attn.o_proj", (c.hidden, c.heads * d)),
                            ("mlp.gate_proj", (c.intermediate, c.hidden)),
                            ("mlp.up_proj", (c.intermediate, c.hidden)),
                            ("mlp.down_proj", (c.hidden, c.intermediate)),
                            ("router.proj", (c.experts, c.hidden)),
                            ("experts.switch_glu.gate_proj", (c.experts, c.moe_width, c.hidden)),
                            ("experts.switch_glu.up_proj", (c.experts, c.moe_width, c.hidden)),
                            ("experts.switch_glu.down_proj", (c.experts, c.hidden, c.moe_width))):
            matrices[p + name] = shape
        if c.sliding(i):
            matrices[p + "self_attn.v_proj"] = (hk * d, c.hidden)
        plain.update({p + name + ".weight": (c.hidden,) for name in norms})
        plain.update({p + "self_attn.q_norm.weight": (d,), p + "self_attn.k_norm.weight": (d,),
                      p + "layer_scalar": (1,), p + "router.scale": (c.hidden,),
                      p + "router.per_expert_scale": (c.experts,)})
    return matrices, plain


def validate_tensor_metadata(c: Config, raw: dict, where: dict) -> None:
    """SafeTensors.where inventory is checked completely before the first payload read."""
    matrices, plain = tensor_contract(c)
    required = set(plain)
    for name, shape in matrices.items():
        bits, group = quant_spec(raw, name)
        if shape[-1] % group or shape[-1] * bits % 32:
            raise ValueError(f"{name}: input width does not align with the original quantization groups")
        for suffix, expected, dtypes in (
                (".weight", (*shape[:-1], shape[-1] * bits // 32), ("U32", "I32")),
                (".scales", (*shape[:-1], shape[-1] // group), ("BF16",)),
                (".biases", (*shape[:-1], shape[-1] // group), ("BF16",))):
            key = name + suffix
            required.add(key)
            if key not in where or tuple(where[key][4]) != expected or where[key][3] not in dtypes:
                raise ValueError(f"{key}: expected {dtypes} shape{expected} in the Gemma4 checkpoint")
    for key, shape in plain.items():
        if key not in where or tuple(where[key][4]) != shape or where[key][3] != "BF16":
            raise ValueError(f"{key}: expected BF16 shape{shape} in the Gemma4 checkpoint")
    extra = set(where) - required
    unexpected = sorted(name for name in extra if name.startswith("language_model."))
    if unexpected:
        raise ValueError(f"Gemma4 checkpoint has unconsumed text tensors: {unexpected[:8]}")


@dataclass
class Layer:
    q: Any
    k: Any
    v: Any | None
    o: Any
    gate: Any
    up: Any
    down: Any
    router: Any
    experts: Any
    input_norm: Any
    attention_norm: Any
    mlp_norm: Any
    expert_norm: Any
    dense_output_norm: Any
    expert_output_norm: Any
    output_norm: Any
    q_norm: Any
    k_norm: Any
    router_scale: Any
    expert_scale: Any
    scalar: Any


@dataclass
class Gemma4Weights:
    config: Config
    embed: Any
    norm: Any
    layers: list[Layer]
    device: Any
    raw_config: dict
    vision: dict | None = None
    projector: Any = None
    numerics: str = "mlx"

    @classmethod
    def load(cls, model_dir: str | Path, device: str = "cuda", *, vision: bool = False) -> "Gemma4Weights":
        import torch
        from tensorfold.checkpoint_metadata import read_config, read_json
        from tensorfold.cleanup import rollback
        from tensorfold.cuda.direct_read import SafeTensors
        from tensorfold.cuda.tensor_file import checkpoint_path
        from .experts import ExpertLinear, Experts
        from .linear import Linear

        root = Path(model_dir)
        raw = read_config(root)
        c = Config.from_dict(raw)
        index = read_json(root, "model.safetensors.index.json")
        if index is None:
            files = [checkpoint_path(root, "model.safetensors")]
        else:
            mapping = index.get("weight_map")
            if not isinstance(mapping, dict) or not mapping or any(
                    not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
                raise ValueError("Gemma4 safetensors index requires a nonempty string weight_map")
            files = [checkpoint_path(root, name) for name in sorted(set(mapping.values()))]
        reader = SafeTensors([])
        try:
            for file in files:
                shard = SafeTensors([file], reader=reader.reader)
                if set(shard.where).intersection(reader.where):
                    raise ValueError("Gemma4 checkpoint tensor names must be unique across shards")
                reader.where.update(shard.where)
            if index is not None:
                if set(reader.where) != set(mapping) or any(
                        reader.where[name][0] != checkpoint_path(root, file) for name, file in mapping.items()):
                    raise ValueError("Gemma4 safetensors index disagrees with the actual shard inventory")
            validate_tensor_metadata(c, raw, reader.where)
            vision_names = ()
            if vision:
                from .vision import projector_spec, vision_shapes, vision_weight_bytes
                headers = {name: {"dtype": entry[3], "shape": entry[4]} for name, entry in reader.where.items()}
                vision_weight_bytes(raw, headers)
                projector_spec(raw, headers)
                vision_names = tuple(vision_shapes(raw))
            dev = torch.device(device)
            def take(name):
                return reader.get(name, dev)
            def matrix(name):
                bits, group = quant_spec(raw, name)
                cls_ = ExpertLinear if len(reader.where[name + ".weight"][4]) == 3 else Linear
                return cls_(take(name + ".weight"), take(name + ".scales"), take(name + ".biases"),
                            bits=bits, group=group)
            embed = matrix(PREFIX + "embed_tokens")
            norm = take(PREFIX + "norm.weight")
            layers = []
            for i in range(c.layers):
                p = f"{PREFIX}layers.{i}."
                layers.append(Layer(
                    q=matrix(p + "self_attn.q_proj"), k=matrix(p + "self_attn.k_proj"),
                    v=matrix(p + "self_attn.v_proj") if c.sliding(i) else None,
                    o=matrix(p + "self_attn.o_proj"), gate=matrix(p + "mlp.gate_proj"),
                    up=matrix(p + "mlp.up_proj"), down=matrix(p + "mlp.down_proj"),
                    router=matrix(p + "router.proj"),
                    experts=Experts(*(matrix(p + "experts.switch_glu." + name + "_proj")
                                      for name in ("gate", "up", "down"))),
                    input_norm=take(p + "input_layernorm.weight"),
                    attention_norm=take(p + "post_attention_layernorm.weight"),
                    mlp_norm=take(p + "pre_feedforward_layernorm.weight"),
                    expert_norm=take(p + "pre_feedforward_layernorm_2.weight"),
                    dense_output_norm=take(p + "post_feedforward_layernorm_1.weight"),
                    expert_output_norm=take(p + "post_feedforward_layernorm_2.weight"),
                    output_norm=take(p + "post_feedforward_layernorm.weight"),
                    q_norm=take(p + "self_attn.q_norm.weight"), k_norm=take(p + "self_attn.k_norm.weight"),
                    router_scale=take(p + "router.scale"), expert_scale=take(p + "router.per_expert_scale"),
                    scalar=take(p + "layer_scalar")))
            tower = {name: take(name) for name in vision_names} if vision else None
            projector = matrix("embed_vision.embedding_projection") if vision else None
            result = cls(c, embed, norm, layers, norm.device, raw, tower, projector)
        except BaseException as primary:
            rollback(reader, primary, reader.close)
            raise
        reader.close()
        return result

    @property
    def nbytes(self) -> int:
        """Resident authoritative tensors, counting tied embedding/head once."""
        matrices = [self.embed]
        tensors = [self.norm]
        if self.vision is not None:
            tensors.extend(self.vision.values())
        if self.projector is not None:
            matrices.append(self.projector)
        for layer in self.layers:
            matrices.extend(getattr(layer, name) for name in ("q", "k", "v", "o", "gate", "up", "down", "router")
                            if getattr(layer, name) is not None)
            matrices.append(layer.experts)
            tensors.extend(getattr(layer, name) for name in (
                "input_norm", "attention_norm", "mlp_norm", "expert_norm", "dense_output_norm", "expert_output_norm",
                "output_norm", "q_norm", "k_norm", "router_scale", "expert_scale", "scalar"))
        return sum(m.nbytes() for m in matrices) + sum(t.numel() * t.element_size() for t in tensors)

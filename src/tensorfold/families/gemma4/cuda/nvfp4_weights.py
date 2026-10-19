"""Publisher Gemma 4 compressed-tensors NVFP4 with native calibrated A4 math.

This format has one original BF16 tied embedding/head, a custom BF16 GateLinear
router with FP32 outputs, individually calibrated native NVFP4 projections and
optional BF16 vision. It is a separate arithmetic contract from MLX affine.
Checkpoint files and prepared parameter storage remain immutable while loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .weights import Config, Gemma4Weights, Layer, PREFIX, tensor_contract

CT_PREFIX = "model.language_model."
CT_PARTS = ("weight_packed", "weight_scale", "weight_global_scale", "input_global_scale")


def ct_tensor_contract(c: Config) -> tuple[dict[str, tuple[int, int]], dict[str, tuple[int, ...]]]:
    """The publisher splits its experts into individually calibrated linears."""
    mlx_matrices, mlx_plain = tensor_contract(c)
    native = {}
    plain = {name.replace(PREFIX, CT_PREFIX, 1): shape for name, shape in mlx_plain.items()}
    for name, shape in mlx_matrices.items():
        name = name.replace(PREFIX, CT_PREFIX, 1)
        if name == CT_PREFIX + "embed_tokens" or name.endswith("router.proj"):
            # The publisher's custom GateLinear receives no quant_config;
            # the broad 'Linear' quantization target does not cover this class.
            plain[name + ".weight"] = shape
        elif len(shape) == 3:
            for expert in range(shape[0]):
                native[name.replace(".experts.switch_glu.", f".experts.{expert}.")] = shape[1:]
        else:
            native[name] = shape
    return native, plain


def _vision_headers(headers: dict) -> dict:
    return {name.removeprefix("model."): value for name, value in headers.items()
            if name.startswith(("model.vision_tower.", "model.embed_vision."))}


def validate_headers(raw: dict, headers: dict, *, vision: bool = False) -> Config:
    """Validate all selected schemas/declarations before the first tensor allocation.

    The shared safetensors reader separately validates byte spans, confinement,
    file types, shape products, duplicates and index identity. This validator
    consumes ordinary header objects and never imports an accelerator runtime.
    """
    from tensorfold.cuda.nvfp4.format import LinearMetadata, config_block

    c = Config.from_dict(raw)
    block = config_block(raw)
    if not block or block.get("quant_method") != "compressed-tensors" or block.get("format") != "nvfp4-pack-quantized":
        raise ValueError("Gemma4 NVFP4 requires the publisher's compressed-tensors nvfp4-pack-quantized declaration")
    if block.get("kv_cache_scheme") is not None or block.get("transform_config") or block.get("sparsity_config"):
        raise ValueError("Gemma4 NVFP4 requires untransformed dense weights and BF16 KV")
    metadata = LinearMetadata(block)
    native, plain = ct_tensor_contract(c)
    required = set(plain)
    for name, shape in plain.items():
        info = headers.get(name)
        if not info or info.get("dtype") != "BF16" or tuple(info.get("shape", ())) != shape:
            raise ValueError(f"{name}: expected original BF16 shape{shape}")
    for name, (n, k) in native.items():
        if k % 64:
            raise ValueError(f"{name}: native NVFP4 input width must be a multiple of64")
        expected = {"weight_packed": ("U8", [n, k // 2]), "weight_scale": ("F8_E4M3", [n, k // 16]),
                    "weight_global_scale": ("F32", [1]), "input_global_scale": ("F32", [1])}
        parts = {}
        for suffix, (dtype, shape) in expected.items():
            key = name + "." + suffix
            required.add(key)
            info = headers.get(key)
            if not info or (info.get("dtype"), info.get("shape")) != (dtype, shape):
                raise ValueError(f"{key}: expected original {dtype} shape{shape}")
            parts[suffix] = dtype, shape
        if metadata.validate(name, parts, checkpoint=True) != ("nvfp4", 4):
            raise ValueError(f"{name}: requires declared native four-bit activation math")
    extras = set(headers) - required - {"__metadata__"}
    if any(not name.startswith(("model.vision_tower.", "model.embed_vision.")) for name in extras):
        raise ValueError(f"Gemma4 NVFP4 has unconsumed tensors: {sorted(extras)[:8]}")
    if vision:
        from .vision import vision_weight_bytes
        normalized = _vision_headers(headers)
        vision_weight_bytes(raw, normalized)
        key = "model.embed_vision.embedding_projection.weight"
        info = headers.get(key)
        expected = (c.hidden, raw["vision_config"]["hidden_size"])
        if not info or info.get("dtype") != "BF16" or tuple(info.get("shape", ())) != expected:
            raise ValueError("Gemma4 NVFP4 vision requires its original BF16 projector")
        if {name for name in normalized if name.startswith("embed_vision.")} != {
                "embed_vision.embedding_projection.weight"}:
            raise ValueError("Gemma4 NVFP4 has an unexpected vision projector representation")
        if not metadata.policy.excluded("model.embed_vision.embedding_projection"):
            raise ValueError("Gemma4 NVFP4 vision projector must retain its declared BF16 exclusion")
    return c


@dataclass
class _Router:
    """Publisher BF16 GateLinear rows; the target's fused CT router (cuda.moe.route_logits) consumes ``weight``."""
    weight: Any

    @property
    def n(self):
        return self.weight.shape[0]

    @property
    def k(self):
        return self.weight.shape[1]

    def nbytes(self):
        return self.weight.numel() * self.weight.element_size()


def _calibrations(reader, native: dict) -> dict[str, tuple[float, float, float]]:
    """Read original scalar multipliers on CPU, once, before native packing.

    The publisher stores the complete small scalar table contiguously. Coalesce
    adjacent scalar ranges into bounded reads; retain neither the full checkpoint
    nor per-projection CUDA scalar tensors. Result is (weight dequant scale,
    activation dequant scale, original activation quantization multiplier).
    """
    import torch
    from tensorfold.cuda.nvfp4.format import calibration, native_calibration

    entries = []
    for name in native:
        for suffix in ("weight_global_scale", "input_global_scale"):
            file, begin, size, _, _ = reader.where[name + "." + suffix]
            entries.append((file, begin, begin + size, name + "." + suffix))
    entries.sort(key=lambda item: (str(item[0]), item[1]))
    values = {}
    batch = []
    def consume(items):
        first, end = items[0][1], items[-1][2]
        raw = reader.reader.read(items[0][0], first, end - first, "cpu")
        for _, begin, stop, key in items:
            value = raw[begin - first:stop - first].view(torch.float32).item()
            values[key] = calibration(value, name=key)
    for entry in entries:
        if batch and (entry[0] != batch[0][0] or entry[2] - batch[0][1] > 1 << 20
                      or entry[1] - batch[-1][2] > 4096):
            consume(batch)
            batch = []
        batch.append(entry)
    if batch:
        consume(batch)
    result = {}
    for name in native:
        weight = calibration(values[name + ".weight_global_scale"], name=name + ".weight_global_scale", reciprocal=True)
        inverse = values[name + ".input_global_scale"]
        act = calibration(inverse, name=name + ".input_global_scale", reciprocal=True)
        native_calibration(act, weight, name=name, act_inverse=inverse)
        result[name] = weight, act, inverse
    return result


def load(model_dir: str | Path, device: str = "cuda", *, vision: bool = False) -> Gemma4Weights:
    """Load original CT payloads once, selecting calibrated FP4xFP4 on SM12.x."""
    import torch
    from tensorfold.checkpoint_metadata import read_config, read_json
    from tensorfold.cleanup import rollback
    from tensorfold.cuda.direct_read import SafeTensors
    from tensorfold.cuda.nvfp4.checkpoint import available
    from tensorfold.cuda.nvfp4.linear import Fp4Linear
    from tensorfold.cuda.tensor_file import checkpoint_path
    from tensorfold.families.qwen3_5.cuda.weights import Plain
    from .nvfp4_experts import Workspace, load_experts
    from .head import select_head

    root = Path(model_dir)
    raw = read_config(root)
    index = read_json(root, "model.safetensors.index.json")
    if index is None:
        files = [checkpoint_path(root, "model.safetensors")]
    else:
        mapping = index.get("weight_map")
        if not isinstance(mapping, dict) or not mapping or any(
                not isinstance(k, str) or not isinstance(v, str) for k, v in mapping.items()):
            raise ValueError("Gemma4 NVFP4 requires a string safetensors weight_map")
        files = [checkpoint_path(root, name) for name in sorted(set(mapping.values()))]
    reader = SafeTensors([])
    try:
        for file in files:
            shard = SafeTensors([file], reader=reader.reader)
            if set(shard.where).intersection(reader.where):
                raise ValueError("Gemma4 NVFP4 tensor names must be unique across shards")
            reader.where.update(shard.where)
        if index is not None and (set(reader.where) != set(mapping) or any(
                reader.where[name][0] != checkpoint_path(root, file) for name, file in mapping.items())):
            raise ValueError("Gemma4 NVFP4 index differs from actual shard inventory")
        headers = {name: {"dtype": value[3], "shape": value[4]} for name, value in reader.where.items()}
        c = validate_headers(raw, headers, vision=vision)
        dev = torch.device(device)
        if dev.type != "cuda" or not available():
            raise ValueError("Gemma4 NVFP4 requires native calibrated activation math on SM12.x CUDA")
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        if torch.cuda.get_device_capability(dev)[0] != 12:
            raise ValueError("Gemma4 NVFP4 requires native calibrated activation math on the selected SM12.x device")
        native, _ = ct_tensor_contract(c)
        calibrated = _calibrations(reader, native)
        def read(name, device=dev):
            return reader.get(name, device)
        def linear(name):
            weight, act, inverse = calibrated[name]
            return Fp4Linear.from_checkpoint(read(name + ".weight_packed"), read(name + ".weight_scale"),
                                             weight, act=act, act_inverse=inverse)
        embed = select_head(Plain(read(CT_PREFIX + "embed_tokens.weight")))
        norm = read(CT_PREFIX + "norm.weight")
        layers = []
        workspace = Workspace()
        for i in range(c.layers):
            p = f"{CT_PREFIX}layers.{i}."
            experts = load_experts(read, p + "experts", count=c.experts, width=c.moe_width, dims=c.hidden, device=dev,
                                   calibrated=calibrated, workspace=workspace)
            layers.append(Layer(
                q=linear(p + "self_attn.q_proj"), k=linear(p + "self_attn.k_proj"),
                v=linear(p + "self_attn.v_proj") if c.sliding(i) else None,
                o=linear(p + "self_attn.o_proj"), gate=linear(p + "mlp.gate_proj"), up=linear(p + "mlp.up_proj"),
                down=linear(p + "mlp.down_proj"), router=_Router(read(p + "router.proj.weight")), experts=experts,
                input_norm=read(p + "input_layernorm.weight"), attention_norm=read(p + "post_attention_layernorm.weight"),
                mlp_norm=read(p + "pre_feedforward_layernorm.weight"), expert_norm=read(p + "pre_feedforward_layernorm_2.weight"),
                dense_output_norm=read(p + "post_feedforward_layernorm_1.weight"),
                expert_output_norm=read(p + "post_feedforward_layernorm_2.weight"),
                output_norm=read(p + "post_feedforward_layernorm.weight"), q_norm=read(p + "self_attn.q_norm.weight"),
                k_norm=read(p + "self_attn.k_norm.weight"), router_scale=read(p + "router.scale"),
                expert_scale=read(p + "router.per_expert_scale"), scalar=read(p + "layer_scalar")))
        tower = {name.removeprefix("model."): read(name) for name in headers
                 if name.startswith("model.vision_tower.")} if vision else None
        projector = Plain(read("model.embed_vision.embedding_projection.weight")) if vision else None
        result = Gemma4Weights(c, embed, norm, layers, norm.device, raw, tower, projector, numerics="ct_nvfp4")
        result.precision = "checkpoint"
    except BaseException as primary:
        rollback(reader, primary, reader.close)
        raise
    reader.close()
    return result

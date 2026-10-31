"""Header-only mixed-precision NVFP4 plans for dense E4B and its assistant.

Planning borrows immutable checkpoint metadata. It neither reads weight payloads
nor imports a calibration provider, allocates accelerator tensors or publishes a
checkpoint. A plan is not an export manifest or evidence of quality/capacity.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from tensorfold.checkpoint_metadata import read_config, read_text
from tensorfold.cuda.capacity import headers, itemsize
from tensorfold.families.gemma4.assistant_metadata import checked_headers
from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig

NVFP4_RECIPE = "gemma4-nvfp4-w4a4-local-hessian-v1"
PRODUCER_COMMIT = "e352caca3c36236fc15adea2627a9d42788380c4"
TEXT = "model.language_model."


def _identity(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(data).hexdigest()


def _target_shapes(cfg):
    """Exact text checkpoint storage, including unused shared-tail K/V tensors."""
    shapes = {TEXT+"embed_tokens.weight": (cfg.vocab, cfg.hidden),
              TEXT+"embed_tokens_per_layer.weight": (cfg.vocab, cfg.layers*cfg.per_layer_input),
              TEXT+"norm.weight": (cfg.hidden,),
              TEXT+"per_layer_model_projection.weight": (cfg.layers*cfg.per_layer_input, cfg.hidden),
              TEXT+"per_layer_projection_norm.weight": (cfg.per_layer_input,)}
    active, unused = set(), set()
    for index, kind in enumerate(cfg.layer_types):
        prefix = f"{TEXT}layers.{index}."
        dim = cfg.global_head_dim if kind == "full_attention" else cfg.head_dim
        matrices = {"self_attn.q_proj": (cfg.heads*dim, cfg.hidden),
                    "self_attn.k_proj": (cfg.kv_heads*dim, cfg.hidden),
                    "self_attn.v_proj": (cfg.kv_heads*dim, cfg.hidden),
                    "self_attn.o_proj": (cfg.hidden, cfg.heads*dim),
                    "mlp.gate_proj": (cfg.intermediate, cfg.hidden),
                    "mlp.up_proj": (cfg.intermediate, cfg.hidden),
                    "mlp.down_proj": (cfg.hidden, cfg.intermediate)}
        for suffix, shape in matrices.items():
            name = prefix+suffix+".weight"
            shapes[name] = shape
            if index >= cfg.layers-cfg.shared_layers and suffix in ("self_attn.k_proj", "self_attn.v_proj"):
                unused.add(name)
            else:
                active.add(name)
        shapes[prefix+"per_layer_input_gate.weight"] = (cfg.per_layer_input, cfg.hidden)
        shapes[prefix+"per_layer_projection.weight"] = (cfg.hidden, cfg.per_layer_input)
        for suffix in ("input_layernorm", "post_attention_layernorm", "post_feedforward_layernorm",
                       "pre_feedforward_layernorm", "post_per_layer_input_norm"):
            shapes[prefix+suffix+".weight"] = (cfg.hidden,)
        for suffix in ("q_norm", "k_norm"):
            name = prefix+"self_attn."+suffix+".weight"
            shapes[name] = (dim,)
            if suffix == "k_norm" and index >= cfg.layers-cfg.shared_layers:
                unused.add(name)
        shapes[prefix+"layer_scalar"] = (1,)
    return shapes, active, unused


def _exclusion(name, role):
    if role == "assistant":
        if name == "masked_embedding.token_ordering":
            return "preserve original INT64 ordered-vocabulary permutation"
        if name.startswith("masked_embedding.") or name == "model.embed_tokens.weight":
            return "BF16 tied/ordered vocabulary and centroid path; packed adapter not implemented"
        return "BF16 norms and layer scalars"
    if name.startswith(("model.vision_tower.", "model.audio_tower.", "model.embed_vision.", "model.embed_audio.")):
        return "BF16 vision/audio encoders and multimodal projectors"
    if name in (TEXT+"embed_tokens.weight", "lm_head.weight"):
        return "BF16 tied token embedding/head; packed adapter not implemented"
    if "per_layer" in name:
        return "BF16 PLE embedding, gates, projections and normalization"
    return "BF16 norms and layer scalars"


def plan_nvfp4(source: str | Path, *, recipe: str = NVFP4_RECIPE) -> dict:
    """Return a JSON-safe plan without modifying SOURCE or claiming conversion.

    Only the original BF16 E4B text architecture or native Google assistant
    metadata is admitted. Target media inspection verifies prefix presence,
    not complete encoder geometry or native-model tensor consumption.
    Source files must remain immutable during this metadata read;
    execution must later verify payload hashes and bind calibration provenance.
    """
    if recipe != NVFP4_RECIPE:
        raise ValueError(f"unsupported offline NVFP4 recipe; expected {NVFP4_RECIPE}")
    root = Path(source)
    config_text = read_text(root, "config.json", required=True)
    raw = read_config(root)
    if raw.get("architectures") == ["Gemma4AssistantForCausalLM"]:
        # Canonical read_config also reconciles quantization sidecars; do not
        # let the original-assistant reader ignore an external declaration.
        for block in (raw, raw.get("text_config") or {}):
            if block.get("quantization_config") not in (None, {}) or block.get("quantization") not in (None, {}):
                raise ValueError("offline NVFP4 requires an original BF16 source")
        cfg, inventory = checked_headers(root)
        role = "assistant"
        active = {name for name, shape in cfg.shapes().items() if len(shape) == 2
                  and name not in ("model.embed_tokens.weight", "masked_embedding.centroids.weight")}
        unused = set()
        if len(active) != 22:
            raise ValueError("native assistant requires exactly22 active core projections")
    else:
        cfg = E2BConfig.from_dict(raw)
        if cfg.profile != "E4B":
            raise ValueError("offline NVFP4 planning currently requires dense E4B or its native assistant")
        role = "target"
        inventory = headers(root)
        expected, active, unused = _target_shapes(cfg)
        actual = {name for name in inventory if name.startswith(TEXT)}
        if actual != set(expected):
            raise ValueError("E4B text tensor inventory differs from the original BF16 contract")
        for name, shape in expected.items():
            if tuple(inventory[name]["shape"]) != shape:
                raise ValueError(f"{name}: E4B source tensor geometry differs")
        media_prefixes = ("model.vision_tower.", "model.audio_tower.", "model.embed_vision.", "model.embed_audio.")
        if any(not any(name.startswith(prefix) for name in inventory) for prefix in media_prefixes):
            raise ValueError("E4B source requires declared modality tensor prefixes")
        if any(not name.startswith((TEXT, *media_prefixes)) for name in inventory):
            raise ValueError("E4B source has unexpected tensors outside native model groups")
    tensors = []
    for name, info in sorted(inventory.items()):
        expected_dtype = "I64" if role == "assistant" and name == "masked_embedding.token_ordering" else "BF16"
        if info["dtype"] != expected_dtype:
            raise ValueError(f"{name}: offline NVFP4 needs original {expected_dtype} source storage")
        source_bytes = math.prod(info["shape"])*itemsize(info, name)
        entry = {"name": name, "shape": list(info["shape"]), "source_dtype": info["dtype"],
                 "source_bytes": source_bytes}
        if name in active:
            n, k = info["shape"]
            if k % 64:
                raise ValueError(f"{name}: native NVFP4 requires K multiple64")
            # Canonical export payload only; runtime tile-padding/layouts and
            # provider calibration/search peaks are deliberately not inferred.
            codes, scales = n*k//2, n*k//16
            entry.update(action="quantize", module=name.removesuffix(".weight"),
                         codes_bytes=codes, block_scale_bytes=scales,
                         weight_global_bytes=4, calibrated_input_global_bytes=4,
                         estimated_export_bytes=codes+scales+8,
                         input_global_status="requires representative static calibration")
        elif name in unused:
            entry.update(action="expected-unused", reason="native shared-KV tail ignores K/V projections and K norm",
                         estimated_export_bytes=source_bytes)
        else:
            entry.update(action="exclude", reason=_exclusion(name, role), estimated_export_bytes=source_bytes)
        tensors.append(entry)
    if read_text(root, "config.json", required=True) != config_text:
        raise ValueError("source config changed during offline planning")
    selected = [item for item in tensors if item["action"] == "quantize"]
    return {
        "schema": "tensorfold.offline-quantization-plan", "schema_version": 1,
        "status": "plan-only", "role": role, "execution_implemented": False,
        "source": {"directory": str(root.absolute()),
                   "config_sha256": hashlib.sha256(config_text.encode()).hexdigest(),
                   "effective_config_sha256": _identity(raw), "validated_header_inventory_sha256": _identity(inventory),
                   "header_identity_scope": "canonical validated names/dtypes/shapes/offsets; not weight payloads",
                   "weight_payload_verification": "pending", "immutability": "required throughout execution"},
        "recipe": {"name": recipe, "version": 1, "format": "NVFP4", "numerics": "W4A4",
                   "weight_scale_algorithm": "local_hessian", "fp8_scale_sweep": True,
                   "block_size": 16, "weight_codes": "E2M1", "block_scales": "E4M3",
                   "global_scales": "FP32", "activation_global": "static calibrated per projection",
                   "activation_block_scales": "dynamic per16 values under fixed calibrated global",
                   "batch_amax_fallback": False, "kv_dtype": "BF16", "transforms": None,
                   "producer": {"repository": "https://github.com/NVIDIA/Model-Optimizer", "commit": PRODUCER_COMMIT,
                                "transformers_requirement": ">=5.5,<5.19", "torch_requirement": ">=2.10",
                                "install_status": "not checked; planner never imports or installs producer"}},
        "calibration": {"status": "required-not-run", "provenance_required": [
            "dataset revisions and content hashes", "tokenizer/processor/template identities", "rendered input hashes",
            "seed, sequence lengths and modality coverage", "static activation globals and layer coverage",
            "producer/source/container/runtime identities", "held-out quality evaluation"]},
        "coverage": {"selected_projections": len(selected),
                     "stored_core_projections": len(active)+sum(len(inventory[name]["shape"]) == 2 for name in unused),
                     "expected_unused_tensors": len(unused),
                     "source_tensors": len(tensors), "scope": "active text attention/MLP core only; explicit mixed precision",
                     "packed_embedding_and_ordered_head_adapter": "not implemented; tables remain BF16",
                     "ropes_norms_kv": "BF16/native arithmetic unchanged",
                     "media_tensor_validation": ("prefix-only; full native-model consumption pending"
                                                if role == "target" else "not applicable") },
        "estimates": {"source_payload_bytes": sum(item["source_bytes"] for item in tensors),
                      "selected_source_bytes": sum(item["source_bytes"] for item in selected),
                      "selected_packed_payload_bytes": sum(item["estimated_export_bytes"] for item in selected),
                      "mixed_export_payload_bytes": sum(item["estimated_export_bytes"] for item in tensors),
                      "expected_unused_storage_policy": "conservatively retained BF16 pending exporter contract",
                      "scope": "raw payload estimates, not measured runtime or memory admission",
                      "not_included": ["runtime padding/layout copies", "calibration/search/workspace peaks",
                                       "allocator reserve", "export metadata and file headers"]},
        "required_gates": (["complete native vision/audio/projector tensor consumption before calibration or export"]
                           if role == "target" else []) + [
                           "producer container/dependency and host-resource admission", "source payload verification",
                           "calibration and complete selected-layer coverage", "packed export and full tensor consumption",
                           "independent format/arithmetic and target row-law1..8", "multimodal/cache/native-context quality",
                           "BF16-reference held-out quality and assistant acceptance", "installed serving/resource verification"],
        "tensors": tensors,
    }

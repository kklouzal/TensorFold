"""Unconnected packed dense E4B loader with verified, consumed payload receipts.

Caller admits memory/host capacity and owns exclusive model construction and
inference. Native no_init_weights creates CPU empty parameter placeholders and
REAL native buffers. Only verified checkpoint tensors are placed on the device;
there is no full GPU BF16 mirror, meta-to_empty or fabricated buffer constants.
No conversion/provider code is imported. Actual GPU/model quality is a separate
qualification gate, not established by successful materialization.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading

import torch
from torch import nn

from tensorfold.checkpoint_metadata import read_config, read_json
from tensorfold.cleanup import rollback, raise_failures
from tensorfold.cuda.capacity import headers
from tensorfold.cuda.direct_read import SafeTensors, DTYPES
from tensorfold.cuda.nvfp4.format import LinearMetadata, config_block
from tensorfold.cuda.tensor_file import checkpoint_path, _regular_stream, MAX_HEADER_BYTES
from ..dense_tensor_contract import text_tensor_contract
from .dense_packed import PackedProjection, PackedProjectionSpec
from .e2b_metadata import E2BConfig

MANIFEST = "tensorfold_quantization_manifest.json"
PIECE = 64 << 20
# HF no_init_weights and Torch's default dtype are process-global. This lock
# serializes THIS loader; caller must exclude unrelated model construction too.
_CONSTRUCTION = threading.Lock()


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _file_hash(path):
    digest = hashlib.sha256()
    with _regular_stream(path) as stream:
        expected = os.fstat(stream.fileno()).st_size
        if not 0 <= expected <= MAX_HEADER_BYTES:
            raise ValueError("quantized checkpoint metadata exceeds the owned metadata size bound")
        count = 0
        while chunk := stream.read(min(PIECE, expected-count+1)):
            count += len(chunk)
            if count > expected:
                raise ValueError("checkpoint metadata changed during hashing")
            digest.update(chunk)
        if count != expected:
            raise ValueError("checkpoint metadata changed during hashing")
    return digest.hexdigest()


def _manifest(root, inventory):
    value = read_json(root, MANIFEST, required=True)
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("dense quantized checkpoint requires manifest schema_version1")
    hashes, metadata = value.get("tensor_sha256"), value.get("metadata_sha256")
    if (not isinstance(hashes, dict) or set(hashes) != set(inventory)
            or not isinstance(metadata, dict) or "config.json" not in metadata):
        raise ValueError("quantized manifest tensor/metadata inventory differs")
    if any(not isinstance(key, str) or not isinstance(sha, str) or re.fullmatch(r"[a-f0-9]{64}", sha) is None
           for mapping in (hashes, metadata) for key, sha in mapping.items()):
        raise ValueError("quantized manifest requires canonical SHA256 identities")
    for filename in ("hf_quant_config.json", "model.safetensors.index.json"):
        if checkpoint_path(root, filename).exists() and filename not in metadata:
            raise ValueError(f"quantized manifest must bind {filename}")
    policy = value.get("unused_shared_kv_policy")
    if policy not in ("omitted", "retained-original-bf16"):
        raise ValueError("explicit unused shared-KV storage policy required")
    _check_metadata(root, metadata)
    return value


def _check_metadata(root, mapping):
    for filename, expected in mapping.items():
        if _file_hash(checkpoint_path(root, filename)) != expected:
            raise ValueError(f"checkpoint metadata hash mismatch: {filename}")


class _VerifiedTensors:
    """Bounded host chunks; hash EXACT bytes copied to each temporary owner.

    Native synchronous copies complete before host chunks retire. A temporary
    device tensor is published to the model only after its complete hash agrees.
    Checkpoint files must be immutable throughout loading. Close drains the
    existing reader; no GPU allocations occur for ignored-source validation.
    """
    def __init__(self, root, inventory, hashes):
        index = read_json(root, "model.safetensors.index.json")
        files = ([checkpoint_path(root, file) for file in sorted(set(index["weight_map"].values()))]
                 if index else sorted(Path(root).glob("*.safetensors")))
        self.reader = SafeTensors(files)
        self.hashes, self.consumed = hashes, {}
        self.device_touched = None
        if (set(self.reader.where) != set(inventory) or any(
                (entry[3], entry[4]) != (inventory[name]["dtype"], inventory[name]["shape"])
                for name, entry in self.reader.where.items())):
            primary = ValueError("checkpoint headers changed before materialization")
            rollback(self.reader, primary, self.reader.close)
            raise primary

    def tensor(self, name, device, *, discard=False):
        path, begin, size, dtype, shape = self.reader.where[name]
        output = chunk = None
        try:
            if name in self.consumed:
                raise ValueError(f"checkpoint tensor consumed twice: {name}")
            output = None if discard else torch.empty((size,), dtype=torch.uint8, device=device)
            if output is not None and output.device.type == "cuda":
                self.device_touched = output.device
            digest = hashlib.sha256()
            for start in range(0, size, PIECE):
                count = min(PIECE, size-start)
                chunk = self.reader.reader.read(path, begin+start, count, "cpu")
                digest.update(memoryview(chunk.numpy()))
                if output is not None:
                    output[start:start+count].copy_(chunk, non_blocking=False)
                chunk = None
            actual = digest.hexdigest()
            if actual != self.hashes[name]:
                raise ValueError(f"checkpoint payload hash mismatch: {name}")
            self.consumed[name] = actual
            return None if output is None else output.view(DTYPES[dtype]).reshape(shape)
        finally:
            output = chunk = None

    def close(self):
        self.reader.close()


def _projection_specs(cfg, inventory, metadata):
    contract, active, _ = text_tensor_contract(cfg)
    specs = {}
    for name in sorted(active):
        module = name.removesuffix(".weight")
        parts = {key[len(module)+1:]: (value["dtype"], value["shape"])
                 for key, value in inventory.items() if key.startswith(module+".")}
        if "weight" in parts and parts["weight"][0] == "BF16":
            if metadata.validate(module, parts, checkpoint=True) != ("bf16", None):
                raise ValueError(f"{module}: BF16 source conflicts with quantized compute")
            continue
        shape = contract[name]
        specs[module] = PackedProjectionSpec.from_header(module, parts, metadata, expected_shape=shape)
    if not specs:
        raise ValueError("quantized dense checkpoint contains no packed active projections")
    return specs


def _module_slot(model, name):
    parent, _, leaf = name.rpartition(".")
    return model.get_submodule(parent) if parent else model, leaf


def _bind_native(model, cfg, inventory, specs, source, device, policy, unused_policy,
                 *, metadata=None, packed_factory=PackedProjection.from_checkpoint):
    """Native-model schema check BEFORE payload reads; CPU tiny tests may inject operators."""
    native = {}
    ephemeral = []
    parts = value = original = replacement = parent = owner = None
    try:
        contract, _, unused = text_tensor_contract(cfg)
        native = model.state_dict(keep_vars=True)
        alias = "lm_head.weight"
        canonical = "model.language_model.embed_tokens.weight"
        if alias not in native or canonical not in native or native[alias].shape != native[canonical].shape:
            raise ValueError("native dense model lacks its expected tied head")
        expected = set(native)-{alias}
        for module, spec in specs.items():
            owner = model.get_submodule(module)
            if (not isinstance(owner, nn.Linear) or owner.bias is not None
                    or (owner.out_features, owner.in_features) != spec.shape):
                raise ValueError(f"{module}: native projection geometry differs")
            expected.remove(module+".weight")
            expected.update(module+"."+suffix for suffix, _, _ in spec.parts)
        if unused_policy == "retained-original-bf16":
            expected.update(unused)
        if set(inventory) != expected:
            raise ValueError("quantized checkpoint differs from complete native parameter/buffer inventory")
        for name in expected:
            if name in unused:
                shape = contract[name]
            elif name in native:
                shape = tuple(native[name].shape)
            else:
                continue  # packed parts were validated against their precise native shape
            if inventory[name]["dtype"] != "BF16" or tuple(inventory[name]["shape"]) != tuple(shape):
                raise ValueError(f"{name}: retained native tensor must match original BF16 geometry")
        if metadata is not None:
            for name, value in native.items():
                if name == alias or name.removesuffix(".weight") in specs or not name.endswith(".weight"):
                    continue
                parent, _ = _module_slot(model, name)
                if isinstance(parent, nn.Linear):
                    metadata.validate(name.removesuffix(".weight"), {"weight": ("BF16", list(value.shape))}, checkpoint=True)
        # Preserve native buffer values/dtypes generated by the supported constructor.
        # state_dict intentionally excludes nonpersistent buffers, not their owners.
        ephemeral = [(name, value) for name, value in model.named_buffers(remove_duplicate=False) if name not in native]
        for module, spec in specs.items():
            scalar_suffixes = {"weight_scale_2", "weight_global_scale", "input_scale", "input_global_scale"}
            parts = {suffix: source.tensor(module+"."+suffix,
                                          "cpu" if suffix in scalar_suffixes or (suffix == "weight_scale" and math.prod(shape) == 1)
                                          else device)
                     for suffix, _, shape in spec.parts}
            replacement = packed_factory(spec, parts, policy)
            parent, leaf = _module_slot(model, module)
            setattr(parent, leaf, replacement)
            parts = replacement = None
        for name, original in native.items():
            if name == alias or name.removesuffix(".weight") in specs:
                continue
            value = source.tensor(name, device)
            parent, leaf = _module_slot(model, name)
            if leaf in parent._parameters:
                setattr(parent, leaf, nn.Parameter(value, requires_grad=False))
            else:
                setattr(parent, leaf, value)
            value = None
        model.lm_head.weight = model.get_input_embeddings().weight
        for name, original in ephemeral:
            if original.device.type == "meta":
                raise ValueError(f"native nonpersistent buffer has no initialized value: {name}")
            parent, leaf = _module_slot(model, name)
            setattr(parent, leaf, original.to(device=device))
        if unused_policy == "retained-original-bf16":
            for name in sorted(unused):
                source.tensor(name, "cpu", discard=True)
        if set(source.consumed) != set(inventory):
            raise ValueError("checkpoint payload consumption is incomplete")
        return {"native_nonpersistent_buffers": [{"name": name, "dtype": str(value.dtype), "shape": list(value.shape)}
                                                  for name, value in ephemeral],
                "packed_projections": len(specs), "expected_unused_tensors": len(unused) if unused_policy != "omitted" else 0,
                "tied_head_identity": model.lm_head.weight is model.get_input_embeddings().weight}
    finally:
        native.clear()
        ephemeral.clear()
        parts = value = original = replacement = parent = owner = model = None


@dataclass(frozen=True)
class DenseMaterialization:
    model: nn.Module
    config: E2BConfig
    receipt: dict
    raw_config: dict


def load_dense_quantized(model_dir, *, device, prompt_policy) -> DenseMaterialization:
    """Load one admitted checkpoint; unconnected until GPU/full-model qualification.

    This API requires external loading capacity admission and a drained exclusive
    process initialization region. It does not install a BLAS lease or claim an
    inference implementation is qualified merely because all tensors were bound.
    """
    from importlib.metadata import version
    import transformers
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    from .e2b_runtime import require_runtime
    from transformers.initialization import no_init_weights
    root, device = Path(model_dir), torch.device(device)
    if device != torch.device("cuda:0"):
        raise ValueError("dense packed materialization requires the admitted GB10 CUDA0 device")
    initial_metadata = {filename: _file_hash(checkpoint_path(root, filename))
                        for filename in ("config.json", "hf_quant_config.json", "model.safetensors.index.json")
                        if checkpoint_path(root, filename).exists()}
    manifest_identity = _file_hash(checkpoint_path(root, MANIFEST))
    raw = read_config(root)
    cfg = E2BConfig.from_architecture(raw)
    if cfg.profile != "E4B":
        raise ValueError("dense packed materialization currently requires E4B")
    block = config_block(raw)
    if (block is None or block.get("kv_cache_scheme") is not None or block.get("transform_config")
            or block.get("sparsity_config")):
        raise ValueError("dense packed loading requires declared untransformed NVFP4/FP8 with BF16 KV")
    inventory = headers(root)
    manifest = _manifest(root, inventory)
    if any(manifest["metadata_sha256"].get(name) != sha for name, sha in initial_metadata.items()):
        raise ValueError("source metadata changed before manifest validation")
    if _file_hash(checkpoint_path(root, MANIFEST)) != manifest_identity:
        raise ValueError("quantized manifest changed during validation")
    metadata = LinearMetadata(block)
    specs = _projection_specs(cfg, inventory, metadata)
    require_runtime(torch, transformers.__version__, version("triton"))
    model = source = result = None
    try:
        with _CONSTRUCTION:
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                with torch.device("cpu"), no_init_weights():
                    native_config = Gemma4Config.from_dict(raw)
                    native_config._attn_implementation = "sdpa"
                    model = Gemma4ForConditionalGeneration(native_config)
            finally:
                torch.set_default_dtype(previous)
        source = _VerifiedTensors(root, inventory, manifest["tensor_sha256"])
        details = _bind_native(model, cfg, inventory, specs, source, device, prompt_policy,
                               manifest["unused_shared_kv_policy"], metadata=metadata)
        _check_metadata(root, manifest["metadata_sha256"])
        if _file_hash(checkpoint_path(root, MANIFEST)) != manifest_identity:
            raise ValueError("quantized manifest changed during materialization")
        receipt = {"schema_version": 1, "fully_consumed": True,
                   "quantization_manifest_sha256": manifest_identity,
                   "calibration_payloads_sha256": _digest(source.consumed),
                   "consumed_tensor_sha256": dict(source.consumed), "config_sha256": _digest(raw),
                   "consumed_metadata_sha256": dict(manifest["metadata_sha256"]),
                   "storage": "actual verified NVFP4/FP8 plus explicit BF16 tensors", **details,
                   "qualification": "materialized only; GPU numerical/quality/resource qualification required"}
        result = DenseMaterialization(model.eval().requires_grad_(False), cfg, receipt, raw)
        source.close()
        return result
    except BaseException as primary:
        if source is not None:
            def retire():
                failures = []
                if source.device_touched is not None:
                    try:
                        torch.cuda.synchronize(source.device_touched)
                    except BaseException as error:
                        failures.append(error)
                try:
                    source.close()
                except BaseException as error:
                    failures.append(error)
                raise_failures(None, failures)
            rollback((model, source), primary, retire)
        raise
    finally:
        source = model = result = None

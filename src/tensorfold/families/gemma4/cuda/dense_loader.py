"""Packed dense E4B/31B loader with verified, consumed payload receipts.

Caller admits memory/host capacity and owns exclusive model construction and
inference. Native META construction allocates parameter shapes only; documented
native component constructors/formulas initialize REAL nonpersistent buffers. Only verified checkpoint tensors are placed on the device;
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
import struct
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
BINDING_METADATA = ("config.json", "hf_quant_config.json", "quantization_config.json",
                    "quant_config.json", "model.safetensors.index.json")
PIECE = 64 << 20
# HF no_init_weights and Torch's default dtype are process-global. This lock
# serializes THIS loader; caller must exclude unrelated model construction too.
_CONSTRUCTION = threading.Lock()


# Authorized publisher objects: metadata SHA pins bind native config and producer
# recipe; full LFS SHA binds every header/payload byte before derived tensor SHA
# checks during actual upload. These are not fabricated Local-Hessian manifests.
_PUBLISHERS = (
    ("AEON-7/Gemma-4-31B-it-DECKARD-HERETIC-Uncensored-NVFP4", "c9f11f3cbeb8efc93d5afcb5dc1b33abee1f09ba",
     "f4cf6df224883b55932c921091b032cdd10af944b111417fcd50f3cee77b1ce6", "ff1caf9529bd3efe50c6fa562b00e94f6346a278e2ea98afe8d5d31b604012b8",
     20450563656, "f8f9ccecc635da40a9d8c3787ca6a524cd1c26950dd8f1687c61fbda7a72c07f"),
    ("AEON-7/Gemma-4-E4B-DECKARD-HERETIC-Uncensored-NVFP4", "cd6d303e2de35fe6b8c3701191dba25d96aae26a",
     "e8e8ea47417292e5c5a58262c64281ffe79f7cc6c90de5918e6728fe496ca00c", "241b5d52e8bc21bf6ef2b20445aee8c5550f2e37a8750a77cce853634c860e76",
     10173136620, "98d0341b447d9b798bb4fcde20224e2798f8ec06fbe9e94aab8e41a9ee2667e5"),
)


def _publisher_manifest(root):
    config_sha = _file_hash(checkpoint_path(root, "config.json"))
    quant_sha = _file_hash(checkpoint_path(root, "hf_quant_config.json"))
    match = next((p for p in _PUBLISHERS if (p[2], p[3]) == (config_sha, quant_sha)), None)
    if match is None:
        raise ValueError("checkpoint without owned manifest requires an authorized publisher metadata pin")
    repo, revision, _, _, size, sha = match
    files = sorted(Path(root).glob("*.safetensors"))
    if [p.name for p in files] != ["model.safetensors"] or files[0].stat().st_size != size:
        raise ValueError("pinned publisher checkpoint file inventory/size differs")
    metadata = {name: _file_hash(checkpoint_path(root, name)) for name in BINDING_METADATA
                if checkpoint_path(root, name).exists()}
    return {"schema_version": 1, "source_kind": "publisher-pinned",
            "publisher": {"repo": repo, "revision": revision},
            "file_sha256": {"model.safetensors": sha}, "file_sizes": {"model.safetensors": size},
            "metadata_sha256": metadata, "unused_shared_kv_policy": "omitted"}


def _native_skeleton(raw):
    """META parameters plus only real constructor-owned nonpersistent buffers.

    No to_empty, BF16 parameter placeholder, fake zero, or media-model duplicate.
    The pinned HF5.17 small buffer-only constructors own their algorithms.
    Unknown nonpersistent buffers fail closed before payload/device acquisition.
    """
    from transformers import Gemma4Config, Gemma4ForConditionalGeneration
    from transformers.initialization import no_init_weights
    from transformers.models.gemma4.modeling_gemma4 import (
        Gemma4TextRotaryEmbedding, Gemma4VisionRotaryEmbedding,
        Gemma4AudioRelPositionalEncoding, Gemma4TextScaledWordEmbedding,
    )
    model = module = replacement = value = None
    try:
        with _CONSTRUCTION:
            previous = torch.get_default_dtype()
            try:
                torch.set_default_dtype(torch.bfloat16)
                config = Gemma4Config.from_dict(raw)
                config._attn_implementation = "sdpa"
                with torch.device("meta"), no_init_weights():
                    model = Gemma4ForConditionalGeneration(config)
                if any(value.device.type != "meta" for value in model.parameters()):
                    raise ValueError("native parameter skeleton must remain entirely META")
                with torch.device("cpu"):
                    for module in model.modules():
                        names = module._non_persistent_buffers_set
                        if not names:
                            continue
                        if isinstance(module, Gemma4TextRotaryEmbedding):
                            replacement = Gemma4TextRotaryEmbedding(module.config, device="cpu")
                        elif isinstance(module, Gemma4VisionRotaryEmbedding):
                            replacement = Gemma4VisionRotaryEmbedding(module.config, device="cpu")
                        elif isinstance(module, Gemma4AudioRelPositionalEncoding):
                            replacement = Gemma4AudioRelPositionalEncoding(config.audio_config)
                        else:
                            replacement = None
                        for name in names:
                            if replacement is not None:
                                value = replacement._buffers.get(name)
                            elif isinstance(module, Gemma4TextScaledWordEmbedding) and name == "embed_scale":
                                # Exact native constructor scalar expression, no embedding table.
                                value = torch.tensor(module.scalar_embed_scale)
                            elif name == "softcap" and type(module).__name__ == "Gemma4AudioAttention":
                                value = torch.tensor(module.attention_logits_soft_cap)
                            else:
                                raise ValueError(f"unsupported native nonpersistent buffer: {type(module).__name__}.{name}")
                            original = module._buffers.get(name)
                            if (value is None or original is None or value.device.type != "cpu"
                                    or value.shape != original.shape or value.dtype != original.dtype):
                                raise ValueError("native CPU buffer initializer changed geometry/dtype")
                            setattr(module, name, value)
                            value = None
                        replacement = None
            finally:
                torch.set_default_dtype(previous)
        result, model = model, None
        return result
    finally:
        model = module = replacement = value = None


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
    value = read_json(root, MANIFEST)
    if value is None:
        return _publisher_manifest(root)
    if type(value.get("schema_version")) is not int or value["schema_version"] != 1:
        raise ValueError("dense quantized checkpoint requires manifest schema_version1")
    hashes, metadata = value.get("tensor_sha256"), value.get("metadata_sha256")
    if (not isinstance(hashes, dict) or set(hashes) != set(inventory)
            or not isinstance(metadata, dict) or "config.json" not in metadata):
        raise ValueError("quantized manifest tensor/metadata inventory differs")
    if any(not isinstance(key, str) or not isinstance(sha, str) or re.fullmatch(r"[a-f0-9]{64}", sha) is None
           for mapping in (hashes, metadata) for key, sha in mapping.items()):
        raise ValueError("quantized manifest requires canonical SHA256 identities")
    for filename in BINDING_METADATA:
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
    def __init__(self, root, inventory, hashes, *, file_hashes=None, file_sizes=None):
        index = read_json(root, "model.safetensors.index.json")
        files = ([checkpoint_path(root, file) for file in sorted(set(index["weight_map"].values()))]
                 if index else sorted(Path(root).glob("*.safetensors")))
        self.reader = SafeTensors(files)
        self.hashes, self.consumed = hashes, {}
        self.verified_files = {}
        self.device_touched = None
        if (set(self.reader.where) != set(inventory) or any(
                (entry[3], entry[4]) != (inventory[name]["dtype"], inventory[name]["shape"])
                for name, entry in self.reader.where.items())):
            primary = ValueError("checkpoint headers changed before materialization")
            rollback(self.reader, primary, self.reader.close)
            raise primary
        # Native binding validates complete model/header topology before the
        # first tensor call triggers publisher's bounded full-file verification.
        self.file_hashes, self.file_sizes = file_hashes, file_sizes

    def _publisher_hashes(self, file_hashes, file_sizes):
        """Bind actual hashed header+payload to every subsequently consumed name.

        Read a single owned regular stream per file. The header parsed by the
        existing authority is EXACTLY the prefix fed into the full-file hash;
        stale names/offsets from an earlier header cannot relabel payload bytes.
        """
        from tensorfold.cuda.capacity import SIZES
        from tensorfold.cuda.tensor_file import parse_header_bytes
        chunk = raw = prefix = None
        result = {}
        try:
            for path in sorted({entry[0] for entry in self.reader.where.values()}):
                filename = Path(path).name
                if filename not in file_hashes:
                    raise ValueError("publisher source file inventory differs")
                digest = hashlib.sha256()
                with _regular_stream(path) as stream:
                    size = os.fstat(stream.fileno()).st_size
                    if size != file_sizes[filename]:
                        raise ValueError("publisher source file size differs")
                    prefix = stream.read(8)
                    if len(prefix) != 8:
                        raise ValueError("publisher header prefix truncated")
                    count = struct.unpack("<Q", prefix)[0]
                    if not 0 < count <= min(MAX_HEADER_BYTES, size-8):
                        raise ValueError("publisher header length invalid")
                    raw = stream.read(count)
                    if len(raw) != count:
                        raise ValueError("publisher JSON header truncated")
                    base, header = parse_header_bytes(raw, size, {d: SIZES[d] for d in DTYPES}, label=path)
                    digest.update(prefix)
                    digest.update(raw)
                    actual_where = {name: (Path(path), base+entry["data_offsets"][0],
                                          entry["data_offsets"][1]-entry["data_offsets"][0],
                                          entry["dtype"], list(entry["shape"]))
                                    for name, entry in header.items() if name != "__metadata__"}
                    previous_where = {name: entry for name, entry in self.reader.where.items() if Path(entry[0]) == Path(path)}
                    if actual_where != previous_where:
                        raise ValueError("publisher hashed header differs from prepared tensor descriptors")
                    prefix = raw = None
                    cursor = base
                    entries = sorted((entry[1], name, entry[2]) for name, entry in actual_where.items())
                    for begin, name, length in [*entries, (size, None, 0)]:
                        if begin < cursor:
                            raise ValueError("publisher tensor byte ranges overlap")
                        tensor_digest = hashlib.sha256()
                        for span, payload in ((begin-cursor, False), (length, True)):
                            for offset in range(0, span, PIECE):
                                wanted = min(PIECE, span-offset)
                                chunk = stream.read(wanted)
                                if len(chunk) != wanted:
                                    raise ValueError("publisher source truncated during verification")
                                digest.update(chunk)
                                if payload:
                                    tensor_digest.update(chunk)
                                chunk = None
                        if name is not None:
                            result[name] = tensor_digest.hexdigest()
                        cursor = begin+length
                    if stream.read(1) or os.fstat(stream.fileno()).st_size != size:
                        raise ValueError("publisher source size changed during verification")
                if digest.hexdigest() != file_hashes[filename]:
                    raise ValueError("publisher whole-file SHA256 mismatch")
                self.verified_files[filename] = digest.hexdigest()
            if set(self.verified_files) != set(file_hashes) or set(result) != set(self.reader.where):
                raise ValueError("publisher file/tensor verification is incomplete")
            return result
        finally:
            chunk = raw = prefix = None

    def tensor(self, name, device, *, discard=False):
        path, begin, size, dtype, shape = self.reader.where[name]
        output = chunk = None
        try:
            if self.hashes is None:
                self.hashes = self._publisher_hashes(self.file_hashes, self.file_sizes)
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


def _verified_source(root, inventory, manifest, owned_manifest_identity):
    """Physical owned-manifest authority wins over arbitrary extra JSON fields."""
    if owned_manifest_identity is not None:
        return _VerifiedTensors(root, inventory, manifest["tensor_sha256"])
    return _VerifiedTensors(root, inventory, None,
                            file_hashes=manifest["file_sha256"], file_sizes=manifest["file_sizes"])


def _projection_specs(cfg, inventory, metadata):
    contract, active, unused = text_tensor_contract(cfg)
    specs = {}
    eligible = active | {name for name, shape in contract.items() if len(shape) == 2
                         and "embed_tokens" not in name and name not in unused}
    for name in sorted(eligible):
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


def _validate_native_inventory(model, cfg, inventory, specs, unused_policy, metadata):
    """One authority for allocation-free preparation and actual binding."""
    native = {}
    owner = parent = value = None
    try:
        contract, _, unused = text_tensor_contract(cfg)
        native = model.state_dict(keep_vars=True)
        alias = "lm_head.weight"
        canonical = "model.language_model.embed_tokens.weight"
        if alias not in native or canonical not in native or native[alias].shape != native[canonical].shape:
            raise ValueError("native dense model lacks its expected tied head")
        expected = set(native)-{alias}
        packed_names = set()
        for module, spec in specs.items():
            owner = model.get_submodule(module)
            if (not isinstance(owner, nn.Linear) or owner.bias is not None
                    or (owner.out_features, owner.in_features) != spec.shape):
                raise ValueError(f"{module}: native projection geometry differs")
            expected.remove(module+".weight")
            packed_names.update(module+"."+suffix for suffix, _, _ in spec.parts)
        expected.update(packed_names)
        if unused_policy == "retained-original-bf16":
            expected.update(unused)
        if set(inventory) != expected:
            raise ValueError("quantized checkpoint differs from complete native parameter/buffer inventory")
        for name in expected:
            if name in packed_names:
                continue  # ModelOpt retains .weight as the packed payload name.
            shape = contract[name] if name in unused else tuple(native[name].shape)
            if inventory[name]["dtype"] != "BF16" or tuple(inventory[name]["shape"]) != tuple(shape):
                raise ValueError(f"{name}: retained native tensor must match original BF16 geometry")
        if metadata is not None:
            for name, value in native.items():
                if name == alias or name.removesuffix(".weight") in specs or not name.endswith(".weight"):
                    continue
                parent, _ = _module_slot(model, name)
                if isinstance(parent, nn.Linear):
                    metadata.validate(name.removesuffix(".weight"), {"weight": ("BF16", list(value.shape))}, checkpoint=True)
        return native, unused
    except BaseException:
        native.clear()
        raise
    finally:
        model = owner = parent = value = None


def _bind_native(model, cfg, inventory, specs, source, device, policy, unused_policy,
                 *, metadata=None, packed_factory=PackedProjection.from_checkpoint):
    """Native-model schema check BEFORE payload reads; CPU tiny tests may inject operators."""
    native = {}
    ephemeral = []
    parts = value = original = replacement = parent = None
    try:
        native, unused = _validate_native_inventory(model, cfg, inventory, specs, unused_policy, metadata)
        alias = "lm_head.weight"
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
        parts = value = original = replacement = parent = model = None


@dataclass(slots=True)
class DenseLoadStatus:
    """Caller-owned publication status, allocated before device binding.

    Only the exclusive loader mutates this record. Failure with unsafe release
    is terminal even if exception annotation itself exhausts; keep its resources
    and the caller's BLAS lease. Success transfers the model to the return owner.
    """
    safe_to_release: bool = True
    transferred: bool = False
    model: object = None
    source: object = None


@dataclass(frozen=True)
class DenseMaterialization:
    model: nn.Module
    config: E2BConfig
    receipt: dict
    raw_config: dict


def load_dense_quantized(model_dir, *, device, prompt_policy, status=None) -> DenseMaterialization:
    """Load one admitted checkpoint; inference qualification remains separate.

    This API requires external loading capacity admission and a drained exclusive
    process initialization region. It does not install a BLAS lease or claim an
    inference implementation is qualified merely because all tensors were bound.
    """
    from importlib.metadata import version
    import transformers
    from .e2b_runtime import require_runtime
    if status is None:
        status = DenseLoadStatus()
    if (type(status) is not DenseLoadStatus or not status.safe_to_release or status.transferred
            or status.model is not None or status.source is not None):
        raise ValueError("dense loader requires a fresh caller-owned load status")
    root, device = Path(model_dir), torch.device(device)
    if device != torch.device("cuda:0"):
        raise ValueError("dense packed materialization requires the admitted GB10 CUDA0 device")
    initial_metadata = {filename: _file_hash(checkpoint_path(root, filename))
                        for filename in BINDING_METADATA
                        if checkpoint_path(root, filename).exists()}
    manifest_path = checkpoint_path(root, MANIFEST)
    manifest_identity = _file_hash(manifest_path) if manifest_path.exists() else None
    raw = read_config(root)
    cfg = E2BConfig.from_architecture(raw)
    if cfg.profile not in ("E4B", "31B"):
        raise ValueError("dense packed materialization requires E4B or31B")
    block = config_block(raw)
    if (block is None or block.get("kv_cache_scheme") is not None or block.get("transform_config")
            or block.get("sparsity_config")):
        raise ValueError("dense packed loading requires declared untransformed NVFP4/FP8 with BF16 KV")
    inventory = headers(root)
    # Select protocol from the initial physical observation, not a later JSON
    # file that may appear while headers/configuration are being prepared.
    manifest = (_publisher_manifest(root) if manifest_identity is None
                else _manifest(root, inventory))
    if any(manifest["metadata_sha256"].get(name) != sha for name, sha in initial_metadata.items()):
        raise ValueError("source metadata changed before manifest validation")
    if manifest_identity is not None and _file_hash(manifest_path) != manifest_identity:
        raise ValueError("quantized manifest changed during validation")
    metadata = LinearMetadata(block)
    specs = _projection_specs(cfg, inventory, metadata)
    require_runtime(torch, transformers.__version__, version("triton"))
    model = source = result = None
    try:
        model = _native_skeleton(raw)
        source = _verified_source(root, inventory, manifest, manifest_identity)
        status.model = model
        status.source = source
        status.safe_to_release = False
        details = _bind_native(model, cfg, inventory, specs, source, device, prompt_policy,
                               manifest["unused_shared_kv_policy"], metadata=metadata)
        _check_metadata(root, manifest["metadata_sha256"])
        if manifest_identity is not None and _file_hash(manifest_path) != manifest_identity:
            raise ValueError("quantized manifest changed during materialization")
        receipt = {"schema_version": 1, "fully_consumed": True,
                   "quantization_manifest_sha256": manifest_identity,
                   "checkpoint_source_kind": "owned-manifest" if manifest_identity is not None else "publisher-pinned",
                   "publisher": manifest.get("publisher") if manifest_identity is None else None,
                   "verified_file_sha256": dict(source.verified_files),
                   "calibration_payloads_sha256": _digest(source.consumed),
                   "consumed_tensor_sha256": dict(source.consumed), "config_sha256": _digest(raw),
                   "consumed_metadata_sha256": dict(manifest["metadata_sha256"]),
                   "storage": "actual verified NVFP4/FP8 plus explicit BF16 tensors", **details,
                   "qualification": "materialized only; GPU numerical/quality/resource qualification required"}
        result = DenseMaterialization(model.eval().requires_grad_(False), cfg, receipt, raw)
        source.close()
        status.transferred = True
        status.safe_to_release = True
        status.model = status.source = None
        return result
    except BaseException as primary:
        _native_cause = BaseException.__cause__.__get__(primary)
        _native_context = BaseException.__context__.__get__(primary)
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
                status.safe_to_release = True
                status.model = status.source = None
            try:
                rollback((model, source), primary, retire)
            except BaseException as transport:
                # Strong status owners and native roots survive even if tuple
                # construction/transport exhausts before rollback can enter.
                if transport is not primary:
                    raise primary from transport
                raise
        raise
    finally:
        source = model = result = None

"""Explicit source and numerical contract for SM70 software NVFP4 imports.

Preparation streams file hashes; serving verifies this declaration once per
profile load, before device allocation. No conversion occurs on admission.
The manifest never upgrades an unqualified numerical profile into a release.
"""

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path

from tensorfold.checkpoint_metadata import read_config
from tensorfold.cuda.capacity import headers
from tensorfold.cuda.tensor_file import _regular_stream, checkpoint_path, read_metadata_json
from tensorfold.cuda.nvfp4.format import LinearMetadata
from tensorfold.cuda.sm70.nvfp4_layout import (EXPERT_FAMILIES, EXPERT_REFERENCE, LINEAR_FAMILIES, LINEAR_REFERENCE,
                                             PROFILE)

SCHEMA = 1
HASH_CHUNK = 32 << 20
_NUMERICS = {"profile": PROFILE, "activation_policy": "full", "weight_operands": "float16",
             "scale_arithmetic": "float32", "accumulation": "float32",
             "scale_order": "(e2m1*e4m3)*global", "source_activation_departure": "unqualified",
             "excluded_projection": "float32-before-output-cast",
             "expert_activation": "fp32-gate-up-silu-multiply-to-fp16",
             "ple_reconstruction": "fp32-source-decode-direct-fp16",
             "recurrent_source": "HF-A_log-canonical-key-group3-no-fold",
             "gdn_output_norm": "stored-direct-gamma"}


def numerical_profile(kernel_family, expert_family=EXPERT_REFERENCE):
    if kernel_family not in LINEAR_FAMILIES or expert_family not in EXPERT_FAMILIES:
        raise ValueError("unsupported SM70 NVFP4 linear/expert kernel family")
    return dict(_NUMERICS, linear_kernel_family=kernel_family, expert_kernel_family=expert_family)


def tokenizer_inputs(directory):
    """Actual default tokenizer/template closure, without compiling templates."""
    from tensorfold.chat_metadata import resolve_chat_templates

    directory = Path(directory)
    result = {"tokenizer.json", "tokenizer_config.json"}
    for name in ("special_tokens_map.json", "chat_template.json", "chat_template.jinja", "added_tokens.json"):
        if (directory / name).exists():
            result.add(name)
    for source in resolve_chat_templates(directory):
        if source.origin.endswith(".jinja"):
            result.add(source.origin)
    for name in result:
        checkpoint_path(directory, name)
    return result


def _version(value):
    return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns


def _digest(path):
    digest = hashlib.sha256()
    buffer = bytearray(HASH_CHUNK)
    view = memoryview(buffer)
    with _regular_stream(path) as source:
        before = os.fstat(source.fileno())
        total = 0
        while total < before.st_size:
            count = source.readinto(view[:min(HASH_CHUNK, before.st_size - total)])
            if not count:
                break
            digest.update(view[:count])
            total += count
        extra = source.read(1)
        opened = os.fstat(source.fileno())
        after = path.stat()
    initial = _version(before)
    if total != before.st_size or extra or initial != _version(opened) or initial != _version(after):
        raise ValueError(f"{path}: source changed during numerical-manifest hashing")
    return digest.hexdigest(), _version(after)


def _inventory(directory):
    from .reader import checkpoint_inventory

    infos, where = headers(directory), checkpoint_inventory(directory)
    if set(infos) != set(where):
        raise ValueError("SM70 family manifest requires one complete checkpoint tensor inventory")
    return infos, where


def _quantization(config, infos):
    block = config.get("quantization_config") or {}
    if block.get("quant_method") != "modelopt":
        raise ValueError("this SM70 recipe requires ModelOpt direct weight_scale_2; other global-scale conventions need a declared recipe")
    allowed = {"quant_method", "quant_algo", "group_size", "config_groups", "ignore", "quantized_layers",
               "producer", "format", "kv_cache_scheme", "quantization_status"}
    if set(block) - allowed:
        raise ValueError("unsupported source recipe metadata for SM70 NVFP4: " + ", ".join(sorted(set(block) - allowed)))
    pending = [block]
    while pending:
        value = pending.pop()
        if isinstance(value, dict):
            for key, item in value.items():
                if any(word in key.lower() for word in ("transform", "hadamard", "rotation", "pre_quant", "smoothquant")):
                    if item not in (None, False, "none", [], {}):
                        raise ValueError(f"SM70 NVFP4 recipe does not implement source transform {key}")
                pending.append(item)
        elif isinstance(value, list):
            pending.extend(value)
    metadata = LinearMetadata(block)
    for name in infos:
        if name.startswith("mtp.") or ".ngram_embedding." in name:
            continue  # Explicitly inventoried inactive embedded-draft source.
        if name.endswith((".weight", ".weight_packed")) and len(infos[name]["shape"]) == 2:
            base = name.rsplit(".", 1)[0]
            fields = {key: (infos[base + "." + key]["dtype"], infos[base + "." + key]["shape"])
                      for key in ("weight", "weight_packed", "weight_scale", "weight_scale_2", "input_scale",
                                  "weight_global_scale", "weight_scale_inv", "input_global_scale")
                      if base + "." + key in infos}
            kind, _ = metadata.validate(base, fields, checkpoint=False)
            if kind not in ("nvfp4", "bf16"):
                raise ValueError(f"{base}: unsupported {kind} projection in SM70 recipe {PROFILE}; packed FP8 needs its own implemented numerical profile")
            declaration = metadata.policy.resolve(base)
            if declaration is not None:
                if declaration.get("output_activations") is not None:
                    raise ValueError(f"{base}: this SM70 recipe does not implement source output activation quantization")
                activation = declaration.get("input_activations")
                if activation is not None:
                    if (kind != "nvfp4" or activation.get("type", "float") != "float"
                            or activation.get("num_bits") != 4 or activation.get("group_size", 16) != 16
                            or activation.get("dynamic", False) not in (False, "local")
                            or activation.get("symmetric", True) is not True or activation.get("zp_dtype") is not None
                            or activation.get("strategy") not in (None, "group", "tensor_group")
                            or activation.get("scale_dtype") not in (None, "torch.float8_e4m3fn")):
                        raise ValueError(f"{base}: unsupported source activation recipe; only declared W4A16/W4A4 departures are implemented")
                    metadata._scalar(base, fields, "input_scale")
    return block


def _check_model_config(config):
    text = config.get("text_config") or config
    if config.get("model_type") not in ("qwen4_exp", "qwen3_8_flash_next"):
        raise ValueError("SM70 NVFP4 manifest requires its implemented Flash Next model operators")
    required = ("hidden_size", "num_hidden_layers", "vocab_size", "max_position_embeddings", "num_attention_heads", "num_key_value_heads",
                "linear_num_key_heads", "linear_num_value_heads", "linear_key_head_dim", "linear_value_head_dim",
                "linear_conv_kernel_dim", "num_experts", "num_experts_per_tok", "moe_intermediate_size",
                "shared_expert_intermediate_size")
    if any(type(text.get(name)) is not int or text[name] <= 0 for name in required):
        raise ValueError("SM70 manifest requires positive integral family geometry, without numeric coercion")
    optional = ("head_dim", "hc_count", "hc_lowrank", "indexer_n_heads", "indexer_head_dim", "indexer_budget",
                "indexer_compress_ratio", "ple_embed_dim", "ple_conv_kernel_size", "ngram_size", "heads_per_ngram",
                "ngram_vocab_size_base", "make_ngram_vocab_size_divisible_by", "split_ngram_parts")
    if any(name in text and (type(text[name]) is not int or text[name] <= 0) for name in optional):
        raise ValueError("SM70 manifest optional geometry must use positive integral dimensions")
    if type(text.get("rms_norm_eps")) not in (int, float):
        raise ValueError("SM70 RMS epsilon must be an explicit positive FP32 number")
    from tensorfold.cuda.nvfp4.format import calibration

    calibration(text["rms_norm_eps"], name="rms_norm_eps")
    if (text.get("hidden_act", "silu") not in ("silu", "swish")
            or not isinstance(text.get("layer_types"), list)
            or any(kind not in ("linear_attention", "full_attention") for kind in text["layer_types"])
            or len(text["layer_types"]) != text.get("num_hidden_layers")):
        raise ValueError("SM70 manifest requires implemented SwiGLU and explicit recurrent/full-attention operators")


def prepare(directory, *, repository, revision, norm_offset, tokenizer_files=None,
            linear_kernel_family=LINEAR_REFERENCE, expert_kernel_family=EXPERT_REFERENCE):
    """Build a reviewable declaration, without altering checkpoint/config files.

    The caller supplies the source's actual normalization convention and
    tokenizer/template files. Their meaning is explicit, never inferred from
    a publisher name or a mean-of-weights heuristic.
    """
    directory = Path(directory)
    if type(norm_offset) is not int or norm_offset not in (0, 1):
        raise ValueError("SM70 norm_offset must explicitly select stored gamma0 or centered1")
    if not isinstance(repository, str) or not repository or not isinstance(revision, str) or not revision:
        raise ValueError("SM70 source needs a repository/source identity and immutable revision")
    required_tokens = tokenizer_inputs(directory)
    if tokenizer_files is None:
        tokenizer_files = sorted(required_tokens)
    if (not isinstance(tokenizer_files, (list, tuple))
            or any(not isinstance(name, str) or not name for name in tokenizer_files)
            or set(tokenizer_files) != required_tokens
            or len(tokenizer_files) != len(required_tokens)):
        raise ValueError("SM70 tokenizer files must exactly bind the actual selected default tokenizer/template")
    config = read_config(directory)
    _check_model_config(config)
    infos, where = _inventory(directory)
    _quantization(config, infos)
    files = {"config.json", *where.values(), *tokenizer_files}
    for name in ("model.safetensors.index.json", "hf_quant_config.json", "generation_config.json"):
        if (directory / name).exists():
            files.add(name)
    hashed = {name: _digest(checkpoint_path(directory, name)) for name in sorted(files)}
    identities = {name: value[0] for name, value in hashed.items()}
    document = {"schema_version": SCHEMA, "family": "qwen4_exp", "source": {"repository": repository,
            "revision": revision, "files": identities, "tokenizer_files": list(tokenizer_files)},
            "numerical": numerical_profile(linear_kernel_family, expert_kernel_family),
            "norm_offset": norm_offset, "transforms": [],
            "inactive_source_tensors": sorted(name for name in infos if name.startswith("mtp.")),
            "qualification": "unqualified", "tensor_inventory": {name: {"dtype": info["dtype"],
            "shape": list(info["shape"])} for name, info in sorted(infos.items())}}
    # The generator owns these just-hashed sources; reuse that validation
    # instead of streaming the complete model for a second verification pass.
    from .sm70_nvfp4_weights import _model_plan

    identity = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    bound = Manifest(document, identity, infos, where, tuple((name, value[1]) for name, value in hashed.items()))
    _model_plan(directory, bound)
    bound.check_versions(directory)
    return document


def required_source_files(path, directory):
    """Paths the structural-profile owner must freeze/hash with this manifest.

    This is an input-closure declaration, not validation of the weight recipe.
    `load` additionally proves hashes/layouts before any device allocation.
    """
    body = read_metadata_json(path)
    source = body.get("source")
    files = source.get("files") if isinstance(source, dict) else None
    if not isinstance(files, dict) or not files or any(not isinstance(name, str) or not name for name in files):
        raise ValueError("family manifest lacks a valid immutable source-file closure")
    return {Path(path).resolve(strict=True), *(checkpoint_path(directory, name) for name in files)}


@dataclass(frozen=True)
class Manifest:
    """Internal prepared record: frozen fields, shallow read-only metadata.

    Owning preparation/import callers preserve document/infos/where unchanged.
    The private _plan handoff accepts this owned record, never a user-decoded
    dictionary or public proof token; external paths still pass load/hash checks.
    """
    document: dict
    identity: str
    infos: dict
    where: dict
    versions: tuple = ()

    @property
    def norm_offset(self):
        return self.document["norm_offset"]

    def check_versions(self, directory):
        """Observe cached-input invalidation under the reader immutability contract."""
        if not self.versions:
            raise ValueError("SM70 cached family plan lacks verified source versions")
        for name, expected in self.versions:
            value = checkpoint_path(directory, name).stat()
            if _version(value) != expected:
                raise ValueError(f"{name}: prepared SM70 source changed; prepare the profile again")

    def check_cached(self, path, directory):
        body = read_metadata_json(path)
        identity = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if identity != self.identity:
            raise ValueError("cached SM70 numerical manifest differs from the selected profile")
        self.check_versions(directory)


def load(path, directory):
    """Validate names/layout/recipes and exact pinned files before device use.

    Source files must remain immutable through loading, as required by the
    shared checkpoint reader. Changed namespace/content requires preparation
    of a new manifest and invalidates all affected runtime proofs.
    """
    directory = Path(directory)
    body = read_metadata_json(path)
    numerical = body.get("numerical")
    if (not isinstance(numerical, dict) or numerical.get("linear_kernel_family") not in LINEAR_FAMILIES
            or numerical.get("expert_kernel_family") not in EXPERT_FAMILIES):
        raise ValueError("SM70 family manifest must name its implemented reference or WMMA candidate")
    if (set(body) != {"schema_version", "family", "source", "numerical", "norm_offset", "transforms",
                      "qualification", "tensor_inventory", "inactive_source_tensors"}
            or type(body["schema_version"]) is not int or body["schema_version"] != SCHEMA
            or body["family"] != "qwen4_exp" or numerical != numerical_profile(
                numerical["linear_kernel_family"], numerical["expert_kernel_family"])
            or body["transforms"] != [] or body["qualification"] != "unqualified"
            or type(body["norm_offset"]) is not int or body["norm_offset"] not in (0, 1)):
        raise ValueError("unsupported SM70 family/numerical manifest, transformation or qualification claim")
    source = body["source"]
    if (not isinstance(source, dict) or set(source) != {"repository", "revision", "files", "tokenizer_files"}
            or any(not isinstance(source[field], str) or not source[field] for field in ("repository", "revision"))
            or not isinstance(source["tokenizer_files"], list) or not source["tokenizer_files"]
            or any(not isinstance(name, str) or not name for name in source["tokenizer_files"])
            or not isinstance(source["files"], dict)):
        raise ValueError("SM70 manifest needs complete source and tokenizer/template identities")
    config = read_config(directory)
    _check_model_config(config)
    infos, where = _inventory(directory)
    expected = {name: {"dtype": info["dtype"], "shape": list(info["shape"])} for name, info in infos.items()}
    inventory = body["tensor_inventory"]
    if (not isinstance(inventory, dict) or any(not isinstance(info, dict)
            or set(info) != {"dtype", "shape"} or not isinstance(info["dtype"], str)
            or not isinstance(info["shape"], list) or any(type(n) is not int or n < 0 for n in info["shape"])
            for info in inventory.values()) or inventory != expected):
        raise ValueError("SM70 manifest tensor names, shapes or encodings differ from the selected checkpoint")
    if body["inactive_source_tensors"] != sorted(name for name in infos if name.startswith("mtp.")):
        raise ValueError("SM70 manifest must explicitly inventory its unused embedded draft tensors")
    if set(source["tokenizer_files"]) != tokenizer_inputs(directory):
        raise ValueError("SM70 manifest tokenizer/template closure differs from the actual default selection")
    required = {"config.json", *where.values(), *source["tokenizer_files"]}
    for name in ("model.safetensors.index.json", "hf_quant_config.json", "generation_config.json"):
        if (directory / name).exists():
            required.add(name)
    if set(source["files"]) != required:
        raise ValueError("SM70 manifest does not bind exactly the selected checkpoint and tokenizer files")
    _quantization(config, infos)
    versions = []
    for name, claimed in source["files"].items():
        if not isinstance(claimed, str) or len(claimed) != 64 or any(c not in "0123456789abcdef" for c in claimed):
            raise ValueError(f"{name}: SM70 source requires an exact lowercase SHA256 identity")
        actual, version = _digest(checkpoint_path(directory, name))
        if actual != claimed:
            raise ValueError(f"{name}: SM70 source hash differs from its prepared identity")
        versions.append((name, version))
    identity = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = Manifest(body, identity, infos, where, tuple(versions))
    result.check_versions(directory)
    return result

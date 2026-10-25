"""Prepared structural profiles for the real Flash Next load pipeline.

Runtime selection is a local relative manifest name, never arbitrary HTTP
constructor kwargs or a serving-time model download. The coordinator verifies
artifact content; peer ranks independently validate the same manifest/header
and immutable file identities before allocation or retirement of the old model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import inspect
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from types import MappingProxyType

from tensorfold.cuda.tensor_file import _metadata_object, _regular_stream, checkpoint_path

SCHEMA = "tensorfold.serving-profile"
MAX_MANIFEST_BYTES = 1 << 20
HEX = re.compile(r"[0-9a-f]{64}\Z")
FIXED = frozenset({"tp", "rank", "master", "port", "expert_session", "vram_experts", "vram_experts_pin",
                   "vision", "vision_urls", "indexer_history", "adaptive_sessions"})
PATH_OPTIONS = frozenset({"mtp_model_dir", "dspark_model_dir", "dflash2_model_dir", "family_manifest", "chat_template_file",
                          "execution_profile_manifest"})
API_FIELDS = frozenset({"served", "aliases", "default_thinking", "sampling", "max_tokens", "reasoning_effort",
                        "thinking_budget"})


def profile_root(value):
    try:
        root = Path(value).resolve(strict=True)
        if not root.is_dir():
            raise ValueError("serving profile root must be an existing local directory")
        return root
    except OSError as error:
        raise ValueError("serving profile root must be an existing accessible local directory") from error


def relative_file(root, name):
    if (not isinstance(name, str) or not 1 <= len(name) <= 1024 or "\\" in name or "\x00" in name
            or PurePosixPath(name).is_absolute() or any(part in ("", ".", "..") for part in name.split("/"))):
        raise ValueError("serving_profile must name a relative file inside the startup profile root")
    root = profile_root(root)
    target = (root / name).resolve(strict=True)
    if not target.is_relative_to(root):
        raise ValueError("serving profile file escapes the startup profile root")
    return target


def read_manifest(path):
    with _regular_stream(path) as stream:
        initial = os.fstat(stream.fileno())
        if not 0 < initial.st_size <= MAX_MANIFEST_BYTES:
            raise ValueError("serving profile manifest must fit the one-MiB metadata bound")
        raw = stream.read(MAX_MANIFEST_BYTES + 1)
        final = os.fstat(stream.fileno())
    if len(raw) != initial.st_size or (initial.st_size, initial.st_mtime_ns) != (final.st_size, final.st_mtime_ns):
        raise ValueError("serving profile manifest changed while reading")
    return _metadata_object(raw, path), hashlib.sha256(raw).hexdigest()


def stamp(path):
    return _file_stamp(Path(path).stat())


def _file_stamp(info):
    return info.st_dev, info.st_ino, info.st_size, info.st_mtime_ns, info.st_ctime_ns


def _path(base, value):
    if not isinstance(value, str) or not 1 <= len(value) <= 4096 or "\x00" in value:
        raise ValueError("serving profile paths require bounded local path strings")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve(strict=True)


def artifact_identity(path, *, expected_size=None, expected_sha256=None, verify_content=True):
    """Hash at most the opened extent plus one EOF byte; bind that exact file."""
    path = Path(path)
    with _regular_stream(path) as source:
        before = os.fstat(source.fileno())
        if expected_size is not None and before.st_size != expected_size:
            raise ValueError("serving artifact size differs from its prepared manifest")
        if verify_content:
            digest = hashlib.sha256()
            remaining = before.st_size
            while remaining:
                block = source.read(min(2 << 20, remaining))
                if not block:
                    raise ValueError("serving artifact was truncated during hashing")
                digest.update(block)
                remaining -= len(block)
            if source.read(1):
                raise ValueError("serving artifact grew during hashing")
            actual = digest.hexdigest()
            if expected_sha256 is not None and actual != expected_sha256:
                raise ValueError("serving artifact content differs from its prepared manifest")
        else:
            actual = expected_sha256
        after = os.fstat(source.fileno())
    initial, final = _file_stamp(before), _file_stamp(after)
    if initial != final:
        raise ValueError("serving artifact changed during preparation")
    return {"path": str(path), "size": before.st_size, "sha256": actual, "stat": list(initial)}


def _artifact(entry, base, *, verify_content):
    if (not isinstance(entry, dict) or set(entry) != {"path", "size", "sha256"}
            or type(entry["size"]) is not int or entry["size"] < 0
            or not isinstance(entry["sha256"], str) or not HEX.fullmatch(entry["sha256"])):
        raise ValueError("serving artifacts require exact local path, nonnegative size and SHA256")
    return artifact_identity(_path(base, entry["path"]), expected_size=entry["size"],
                             expected_sha256=entry["sha256"], verify_content=verify_content)


def _required_files(model_dir, options):
    from ..gguf import GGUFCheckpoint, resolve_gguf

    from tensorfold.checkpoint_metadata import _path as metadata_path
    from tensorfold.chat_metadata import resolve_chat_templates

    required = [checkpoint_path(model_dir, name) for name in ("config.json", "tokenizer.json", "tokenizer_config.json")]
    for name in ("generation_config.json", "special_tokens_map.json", "added_tokens.json", "chat_template.json", "chat_template.jinja",
                 "hf_quant_config.json", "quantization_config.json", "quant_config.json"):
        selected = metadata_path(model_dir, name, False)
        if selected is not None:
            required.append(selected)
    sources = resolve_chat_templates(model_dir, template_file=options.get("chat_template_file"),
                                     template_name=options.get("chat_template_name"))
    for source in sources:
        if source.origin.endswith(".jinja"):
            required.append(Path(source.origin) if Path(source.origin).is_absolute() else checkpoint_path(model_dir, source.origin))
    entrypoint = resolve_gguf(model_dir)
    family_manifest = options.get("family_manifest")
    if family_manifest is not None:
        from .sm70_weight_manifest import required_source_files

        required.extend(required_source_files(family_manifest, model_dir))
    elif entrypoint is None:
        raise ValueError("structural profiles require GGUF or an explicit SM70 family manifest")
    if entrypoint is not None:
        required.extend(GGUFCheckpoint.open(entrypoint).paths)
    drafts = [(method, options.get(name)) for method, name in (
        ("mtp", "mtp_model_dir"), ("dspark", "dspark_model_dir"), ("dflash2", "dflash2_model_dir"))
        if options.get(name) is not None]
    if len(drafts) > 1:
        raise ValueError("a structural serving profile selects exactly one speculative source owner")
    if drafts:
        from .execution_identity import draft_artifact_files

        method, directory = drafts[0]
        required.extend(draft_artifact_files(directory, method).values())
        if method == "dflash2":
            from tensorfold.token_metadata import tokenizer_override

            override = tokenizer_override()
            if override is not None:
                required.append(override.path.resolve(strict=True))
    for name in PATH_OPTIONS:
        value = options.get(name)
        if value is None or name.endswith("_model_dir"):
            continue
        required.append(Path(value))
    return {str(Path(path).resolve(strict=True)) for path in required}


@dataclass(frozen=True)
class ServingProfile:
    name: str
    identifier: str
    manifest_sha256: str
    identity: str
    model_dir: Path
    options: object
    api: object
    artifacts: tuple

    def check_sources(self):
        if _required_files(self.model_dir, self.options) - {item["path"] for item in self.artifacts}:
            raise ValueError("a new selected model input is absent from the prepared serving inventory")
        for artifact in self.artifacts:
            if list(stamp(artifact["path"])) != artifact["stat"]:
                raise ValueError("a prepared serving artifact changed before loading/publication")

    def wire(self):
        return {"name": self.name, "manifest_sha256": self.manifest_sha256,
                "identity": self.identity, "artifacts": list(self.artifacts)}

    def record(self):
        return {"id": self.identifier, "manifest": self.name, "manifest_sha256": self.manifest_sha256,
                "identity": self.identity, "model_dir": str(self.model_dir),
                "context_tokens": self.options.get("max_len"), "kv_dtype": self.options.get("kv_dtype"),
                "maximum_proposals": self.options.get("depth"),
                "artifact_count": len(self.artifacts), "qualification": "prepared-identity-checked"}


def prepare(root, name, *, initializer, fixed_options, verify_content=True, expected=None):
    path = relative_file(root, name)
    raw, digest = read_manifest(path)
    allowed = {"schema", "version", "id", "model_dir", "options", "api", "artifacts"}
    if (set(raw) != allowed or raw.get("schema") != SCHEMA or type(raw.get("version")) is not int or raw["version"] != 1
            or not isinstance(raw.get("id"), str) or not 1 <= len(raw["id"]) <= 256
            or not isinstance(raw.get("options"), dict) or not isinstance(raw.get("api"), dict)
            or not isinstance(raw.get("artifacts"), list) or not raw["artifacts"]):
        raise ValueError("a prepared serving profile requires its complete version1 schema")
    signature = inspect.signature(initializer)
    supported = set(signature.parameters) - FIXED - {"self", "model_dir"}
    options = dict(raw["options"])
    if any(key.startswith("_") for key in options) or set(options) - supported:
        raise ValueError("serving profile contains unsupported or process-fixed engine options")
    explicit_context = options.get("max_len")
    options = {**{key: parameter.default for key, parameter in signature.parameters.items()
                  if key in supported and parameter.default is not inspect.Parameter.empty}, **options}
    if set(raw["api"]) - API_FIELDS:
        raise ValueError("serving profile contains unsupported API binding settings")
    if type(explicit_context) is not int or explicit_context < 2:
        raise ValueError("serving profiles require an explicit positive prompt-plus-reply context")
    for key in ("depth", "streams", "session_ceiling", "execution_rows", "kv_full_windows"):
        value = options.get(key)
        if value is not None and (type(value) is not int or value < (0 if key == "depth" else 1)):
            raise ValueError(f"serving profile {key} requires an exact supported integer")
    if "context_explicit" in raw["options"] and raw["options"]["context_explicit"] is not True:
        raise ValueError("a serving profile's explicit context cannot disable full-context admission")
    options["context_explicit"] = True
    for key in ("prefetch", "graphs", "ple_on_ssd"):
        if key in options and type(options[key]) is not bool:
            raise ValueError(f"serving profile {key} requires a boolean")
    for key in ("confidence", "share", "yarn_factor"):
        value = options.get(key)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
            raise ValueError(f"serving profile {key} requires a finite number")
    for key in PATH_OPTIONS & options.keys():
        if options[key] is not None:
            options[key] = _path(path.parent, options[key])
    api = raw["api"]
    if "served" in api and (not isinstance(api["served"], str) or not 1 <= len(api["served"]) <= 256):
        raise ValueError("serving id must be a bounded nonempty string")
    if "aliases" in api and (not isinstance(api["aliases"], list) or len(api["aliases"]) > 1024
                            or any(not isinstance(alias, str) or not 1 <= len(alias) <= 256 for alias in api["aliases"])):
        raise ValueError("serving aliases must be a bounded list of model ids")
    if "default_thinking" in api and type(api["default_thinking"]) is not bool:
        raise ValueError("default_thinking requires a boolean")
    for key in ("max_tokens", "thinking_budget"):
        if key in api and (type(api[key]) is not int or api[key] < (1 if key == "max_tokens" else 0)):
            raise ValueError(f"serving {key} requires a supported exact integer")
    if "reasoning_effort" in api and api["reasoning_effort"] not in (None, "none", "minimal", "low", "medium", "high", "xhigh"):
        raise ValueError("serving reasoning_effort is unsupported")
    if "sampling" in api:
        from tensorfold.server.request_options import parse_numbers

        if not isinstance(api["sampling"], dict) or set(api["sampling"]) - {"temperature", "top_k", "top_p", "min_p"}:
            raise ValueError("serving sampling requires supported explicit defaults")
        parse_numbers(api["sampling"])
    model = _path(path.parent, raw["model_dir"])
    from tensorfold import hub
    from tensorfold.cuda.capacity import config
    from ..kv_formats import get_pair

    model_dir = hub.resolve(str(model), download=False)
    text = config(model_dir)
    if text.get("model_type") not in ("qwen4_exp_text", "qwen4_exp", "qwen3_8_flash_next"):
        raise ValueError("serving profile must bind a Flash Next target")
    maximum = text.get("max_position_embeddings")
    if type(maximum) is not int or options["max_len"] > maximum:
        raise ValueError("serving context exceeds the selected target position contract")
    pair = get_pair(options.get("kv_dtype", "bf16"), options.get("kv_key_dtype"), options.get("kv_value_dtype"))
    from .sm70_kv import check_pair
    check_pair(pair, head_dim=int(text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"]))
    options.update(fixed_options)
    options["context_explicit"] = True
    required = _required_files(model_dir, options)
    artifacts = tuple(_artifact(item, path.parent, verify_content=verify_content) for item in raw["artifacts"])
    if len({item["path"] for item in artifacts}) != len(artifacts) or required - {item["path"] for item in artifacts}:
        raise ValueError("serving artifacts must cover every target/draft/tokenizer/template load input exactly once")
    identity_options = {key: str(value) if isinstance(value, Path) else value for key, value in options.items() if key != "rank"}
    identity = hashlib.sha256(json.dumps({"model_dir": str(model_dir), "options": identity_options,
        "api": raw["api"], "artifacts": [{key: value for key, value in item.items() if key != "stat"}
                                          for item in artifacts]}, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    result = ServingProfile(name, raw["id"], digest, identity, model_dir, MappingProxyType(options),
                            MappingProxyType(dict(raw["api"])), artifacts)
    if expected is not None and (digest != expected["manifest_sha256"] or identity != expected["identity"]
                                 or list(artifacts) != expected["artifacts"]):
        raise ValueError("rank serving profile/artifacts differ from the coordinator's prepared binding")
    result.check_sources()
    return result

"""Immutable positional preparation for the SM70 family, without accelerator IO.

This authority resolves implementation bounds and identities. It issues no
quality, execution or performance qualification. Source and algorithm inputs
must stay immutable through load; the serving profile owns their complete
artifact identity and the drain/retirement transition.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct
from typing import Mapping

from tensorfold.checkpoint_metadata import read_config
from tensorfold.cuda.tensor_file import checkpoint_path, read_metadata_json

from .rope import RopeParameters

SCHEMA_VERSION = 1
MAX_POSITION = 2**31 - 1  # Existing family position/launch ABI, exclusive bound.
MAX_EPOCH = 2**63 - 1
HF_REVISION = "50e35cc0ca18ec19b80653690de553e6decf74f5"
INDEXER_REVISION = "dba22e9b68dda90345ac17f5146e7ac812a09ed6"
DFLASH_REVISION = "3c560089e1818a5217a73f2168273485a5350341"
NATIVE_ALGORITHM = "tensorfold-native-f64-power-f32-frequency-v1"
YARN_ALGORITHM = "transformers-static-yarn-50e35cc0-f32-v1"
DFLASH_ALGORITHM = "tensorfold-dflash-scalar-f32-power-v1"
APPLICATION = "sm70-f32-phase-trig-rotary-product-f16-output-v1"
_ROPE_KEYS = frozenset({"rope_type", "type", "rope_theta", "partial_rotary_factor", "mrope_section",
    "mrope_interleaved", "factor", "original_max_position_embeddings", "beta_fast", "beta_slow",
    "truncate", "attention_factor", "mscale", "mscale_all_dim"})
_YARN_ONLY_KEYS = frozenset({"factor", "original_max_position_embeddings", "beta_fast", "beta_slow",
                            "truncate", "attention_factor", "mscale", "mscale_all_dim"})


def _integer(value, name, *, maximum=MAX_POSITION, minimum=1):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError(f"position {name} must be an integer in [{minimum}, {maximum}]")
    return value


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode()).hexdigest()


def _source(source):
    """Actual configuration plus loader inputs, without rehashing whole weights."""
    source = Path(source)
    if source.is_file() and source.suffix.lower() == ".json":
        raw, files = read_metadata_json(source), {source.resolve()}
        root = source.parent
    else:
        raw, files = read_config(source), set()
        root = source if source.is_dir() else source.parent
        if (root / "config.json").exists():
            files.add(checkpoint_path(root, "config.json"))
    # A GGUF source's sidecar cannot silently replace its positional geometry.
    from .gguf import GGUFCheckpoint, metadata_config, resolve_gguf
    from tensorfold.families import quant_method

    if source.suffix.lower() == ".gguf" or quant_method(raw) == "gguf":
        entry = resolve_gguf(root if source.suffix.lower() == ".json" else source)
        checkpoint = GGUFCheckpoint.open(entry)
        files.update(path.resolve() for path in checkpoint.paths)
        declared = RopeParameters.from_config(metadata_config(entry))
        selected = RopeParameters.from_config(raw)
        if (declared.theta, declared.rotary_dim, declared.native_context, declared.mrope_section) != (
                selected.theta, selected.rotary_dim, selected.native_context, selected.mrope_section):
            raise ValueError("GGUF/external positional geometry or native reference conflicts")
        scaling = {key: value for key, value in checkpoint.metadata.items() if key.startswith("qwen4exp.rope.scaling.")}
        if scaling and scaling != {"qwen4exp.rope.scaling.type": "none"}:
            raise ValueError("GGUF positional scaling metadata requires an explicitly implemented source recipe")
    if not files:
        raise ValueError("position source has no actual configuration input")
    return raw, tuple(sorted(files)), _digest(raw)


def _native_mtp(raw):
    """Declare native defaults independently; a target CLI override never enters."""
    source = RopeParameters.from_config(raw)
    if source.native_context <= 0:
        raise ValueError("MTP native positional reference must be positive")
    return RopeParameters("default", source.theta, source.rotary_dim, source.native_context,
                          source.native_context, mrope_section=source.mrope_section)


def _resolve_rope(raw, declaration, *, yarn_factor=None, native_mtp=False):
    baseline = _native_mtp(raw) if native_mtp else RopeParameters.from_config(raw, yarn_factor)
    if declaration is None:
        return baseline
    if not isinstance(declaration, dict) or set(declaration) - _ROPE_KEYS:
        raise ValueError("position role has unknown RoPE parameters")
    kind = declaration.get("rope_type", declaration.get("type", "default"))
    if kind == "default" and set(declaration) & _YARN_ONLY_KEYS:
        raise ValueError("native/default positional declaration contains unused YaRN parameters")
    text = dict(raw.get("text_config", raw))
    original = text.get("rope_parameters", text.get("rope_scaling")) or {}
    # The external declaration is the single complete requested policy; source
    # geometry/reference is checked below, rather than scaling twice.
    text.pop("rope_scaling", None)
    text["rope_parameters"] = dict(declaration)
    resolved = RopeParameters.from_config({"text_config": text})
    source = RopeParameters.from_config(raw)
    if (resolved.theta, resolved.rotary_dim, resolved.mrope_section, resolved.native_context) != (
            source.theta, source.rotary_dim, source.mrope_section, source.native_context):
        raise ValueError("external positional policy conflicts with source theta/layout/native reference")
    if original.get("rope_type", original.get("type", "default")) != "default" and resolved != source:
        raise ValueError("external positional policy conflicts with already scaled source")
    if yarn_factor is not None and resolved != baseline:
        raise ValueError("CLI YaRN and explicit positional policy disagree")
    return resolved


@dataclass(frozen=True)
class PositionRole:
    """One attention/indexer transform; the two operators share its vector."""
    name: str
    rope: RopeParameters
    max_rows: int
    coordinate_limits: tuple[int, int, int]
    source_config_sha256: str
    origin: str

    @property
    def algorithm(self):
        return NATIVE_ALGORITHM if self.rope.rope_type == "default" else YARN_ALGORITHM

    def receipt(self):
        return {"name": self.name, "algorithm": self.algorithm, "rope": self.rope.metadata(),
                "correction_range": self.rope.correction_range, "max_rows": self.max_rows,
                "coordinate_limits_exclusive": list(self.coordinate_limits), "axes": ["t", "h", "w"],
                "amplitude": "rotary-cos-sin-only", "application": APPLICATION,
                "phase_dtype": "float32", "frequency_storage_dtype": "float32",
                "output_dtype": "float16", "pooling": "fp32-group-first-coordinate",
                "source_config_sha256": self.source_config_sha256, "origin": self.origin}

    def validate_span(self, row_start, row_count, *, coordinate_min, coordinate_max, lookahead_rows=0):
        """Validate actual live coordinates; padded rows are not semantic input."""
        _integer(row_start, "row start", minimum=0)
        _integer(row_count, "row count", minimum=0)
        _integer(lookahead_rows, "lookahead rows", minimum=0)
        if row_start + row_count + lookahead_rows > self.max_rows:
            raise ValueError(f"{self.name} live rows/lookahead exceed selected window")
        if row_count + lookahead_rows == 0:
            return
        self.validate_coordinates(coordinate_min, coordinate_max)

    def validate_coordinates(self, coordinate_min, coordinate_max):
        """Validate an actual producer's inclusive THW box, without row inference."""
        if (not isinstance(coordinate_min, (list, tuple)) or not isinstance(coordinate_max, (list, tuple))
                or len(coordinate_min) != 3 or len(coordinate_max) != 3):
            raise ValueError("position bounds require explicit THW minima and maxima")
        for low, high, limit in zip(coordinate_min, coordinate_max, self.coordinate_limits):
            _integer(low, "coordinate minimum", minimum=0)
            _integer(high, "coordinate maximum", minimum=0)
            if low > high or high >= limit:
                raise ValueError(f"{self.name} actual coordinate exceeds exclusive positional limit")


@dataclass(frozen=True)
class DFlashPositionRole:
    theta: float
    rotary_dim: int
    native_context: int
    max_rows: int
    coordinate_limit: int
    window_rows: int
    source_config_sha256: str

    def receipt(self):
        return {"name": "dflash2", "algorithm": DFLASH_ALGORITHM, "theta": self.theta,
                "rotary_dim": self.rotary_dim, "native_context": self.native_context,
                "max_rows": self.max_rows, "coordinate_limit_exclusive": self.coordinate_limit,
                "window_rows": self.window_rows, "layout": "full-width-neox",
                "phase_dtype": "float32", "frequency_dtype": "float32", "output_dtype": "float16",
                "amplitude": 1.0, "ring_modulo_changes_coordinates": False,
                "source_config_sha256": self.source_config_sha256}

    def validate_span(self, row_start, row_count, *, absolute_start, lookahead_rows=0):
        for name, value in (("row start", row_start), ("row count", row_count),
                            ("lookahead rows", lookahead_rows)):
            _integer(value, name, minimum=0)
        if row_start + row_count + lookahead_rows > self.max_rows:
            raise ValueError("DFlash live rows/lookahead exceed selected window")
        if row_count + lookahead_rows == 0:
            return
        _integer(absolute_start, "absolute start", minimum=0)
        if absolute_start + row_count + lookahead_rows > self.coordinate_limit:
            raise ValueError("DFlash absolute positions/lookahead exceed declared support; ring modulo is storage only")


@dataclass(frozen=True)
class ResolvedPositionProfile:
    target: PositionRole
    mtp: PositionRole
    dflash2: DFlashPositionRole | None
    epoch: int
    _inputs: tuple[Path, ...]
    _receipt_json: str
    _identity: str

    @property
    def target_attention(self):
        return self.target

    @property
    def target_indexer(self):
        return self.target

    @property
    def mtp_attention(self):
        return self.mtp

    @property
    def mtp_indexer(self):
        return self.mtp

    @property
    def identity(self):
        return self._identity

    @property
    def qualification(self):
        return "unqualified-candidate"

    def receipt(self):
        return json.loads(self._receipt_json)

    def required_input_files(self):
        return self._inputs

    def validate_method(self, method, kv_identity, lookahead_rows, media_bounds=None):
        """Bind method inputs; callers must also validate each actual live span.

        No factor, file boolean or this implementation-limit check authorizes
        quality/performance release. Actual lookahead shrinks at the boundary;
        guard cache rows never extend a coordinate or selected-row limit.
        """
        if method not in ("none", "mtp", "dflash2"):
            raise ValueError("position profile requires an explicit implemented speculative method")
        if not isinstance(kv_identity, str) or not kv_identity:
            raise ValueError("position pairing requires the actual KV schema identity")
        _integer(lookahead_rows, "lookahead rows", minimum=0)
        if method == "mtp" and self.target.max_rows > self.mtp.max_rows:
            raise ValueError("target extension lacks a matching explicit MTP row/coordinate policy")
        if method == "dflash2" and (self.dflash2 is None or self.target.max_rows > self.dflash2.max_rows):
            raise ValueError("DFlash requires its actual separate positional source and supported absolute range")
        roles = (self.target, self.mtp) if method == "mtp" else (self.target,)
        if any(lookahead_rows >= min(role.max_rows, *role.coordinate_limits) for role in roles):
            raise ValueError("lookahead block cannot fit the declared positional domain")
        if method == "dflash2" and lookahead_rows >= min(self.dflash2.max_rows, self.dflash2.coordinate_limit):
            raise ValueError("DFlash lookahead block cannot fit its absolute positional domain")
        if media_bounds is not None:
            # Inclusive maxima are from the actual bounded processor envelope.
            for role in roles:
                role.validate_coordinates((0, 0, 0), media_bounds)
        binding = {"position_identity": self.identity, "epoch": self.epoch, "method": method,
                   "kv_identity": kv_identity, "lookahead_rows": lookahead_rows, "media_coordinate_max": media_bounds,
                   "qualification": self.qualification, "actual_live_span_guard_required": True}
        return {**binding, "identity": _digest(binding)}


def _role(name, raw, digest, requested, declaration, *, yarn_factor=None, native_mtp=False, origin="actual-source"):
    if declaration is not None and (not isinstance(declaration, dict) or set(declaration) != {
            "source_config_sha256", "rope", "coordinate_limits"}):
        raise ValueError("position role declaration requires exact source/rope/coordinate fields")
    if declaration is not None and declaration["source_config_sha256"] != digest:
        raise ValueError("position role source configuration identity differs")
    policy = _resolve_rope(raw, None if declaration is None else declaration["rope"],
                           yarn_factor=yarn_factor, native_mtp=native_mtp)
    _integer(policy.native_context, "native reference")
    _integer(policy.context_limit, "transform context limit")
    if sum(policy.mrope_section) != policy.rotary_dim // 2:
        raise ValueError("position interleaved sections must exactly cover the rotary pairs")
    if policy.rope_type == "yarn":
        low, high = policy.correction_range
        if not math.isfinite(low) or not math.isfinite(high) or low >= high:
            raise ValueError("position YaRN correction ramp must have ordered finite bounds")
    else:
        # Native construction is monotone: the last pair is its smallest
        # positive frequency. Do not admit a policy whose stored FP32 vector
        # would silently underflow, while preserving its existing F64 formula.
        half = policy.rotary_dim // 2
        last = policy.theta ** (-(half - 1) / half)
        stored = struct.unpack("f", struct.pack("f", last))[0]
        if not math.isfinite(stored) or stored <= 0:
            raise ValueError("native position frequency cannot be stored as positive finite FP32")
    limits = (policy.context_limit,) * 3 if declaration is None else declaration["coordinate_limits"]
    if not isinstance(limits, (tuple, list)) or len(limits) != 3:
        raise ValueError("position role requires three exclusive coordinate limits")
    limits = tuple(_integer(value, "coordinate limit") for value in limits)
    if any(value > policy.context_limit for value in limits):
        raise ValueError("position coordinates exceed the declared transform implementation limit")
    return PositionRole(name, policy, min(requested, policy.context_limit, *limits), limits, digest, origin)


def resolve_position_profile(model_dir, *, requested_context, yarn_factor=None, manifest_path=None,
                             companion_configs: Mapping[str, Path] | None = None):
    """Resolve actual source roles before allocation; preserve legacy other paths.

    Companion paths are the selected configuration sources, not publisher/name
    dispatch. The enclosing serving profile hashes all required inputs, including
    GGUF shards, together with its actual model/operator/codec identities.
    """
    requested = _integer(requested_context, "requested context")
    raw, files, source_digest = _source(model_dir)
    inputs = set(files)
    if companion_configs is not None and not isinstance(companion_configs, Mapping):
        raise ValueError("positional companion configurations must be a mapping of actual sources")
    companions = {} if companion_configs is None else dict(companion_configs)
    if set(companions) - {"mtp", "dflash2"}:
        raise ValueError("unknown positional companion source")
    manifest = None
    if manifest_path is not None:
        path = Path(manifest_path).resolve()
        manifest = read_metadata_json(path)
        inputs.add(path)
        if (set(manifest) != {"schema_version", "family", "epoch", "requested_context", "qualification",
                             "target", "mtp", "dflash2"}
                or type(manifest["schema_version"]) is not int or manifest["schema_version"] != SCHEMA_VERSION
                or manifest["family"] != "qwen4_exp" or manifest["qualification"] != "unqualified-candidate"
                or type(manifest["requested_context"]) is not int or manifest["requested_context"] != requested):
            raise ValueError("unknown/incomplete positional schema, requested rows or qualification claim")
    epoch = 0 if manifest is None else _integer(manifest["epoch"], "epoch", maximum=MAX_EPOCH, minimum=0)
    target = _role("target", raw, source_digest, requested, None if manifest is None else manifest["target"],
                   yarn_factor=yarn_factor)
    if requested > target.max_rows:
        raise ValueError("requested rows exceed the resolved target positional implementation limit")
    mtp_raw, mtp_digest, mtp_origin = raw, source_digest, "explicit-family-native-mtp-defaults"
    if "mtp" in companions:
        mtp_raw, mtp_files, mtp_digest = _source(companions["mtp"])
        inputs.update(mtp_files)
        mtp_origin = "actual-companion-source"
    mtp_declaration = None if manifest is None else manifest["mtp"]
    if mtp_declaration is not None and "mtp" not in companions:
        raise ValueError("explicit MTP policy must bind its actual selected companion configuration")
    mtp = _role("mtp", mtp_raw, mtp_digest, requested, mtp_declaration,
                native_mtp=mtp_declaration is None and "mtp" not in companions,
                origin=mtp_origin)
    dflash = None
    declaration = None if manifest is None else manifest["dflash2"]
    if "dflash2" in companions:
        from .cuda.dflash2_config import DFlash2Config

        df_raw, df_files, df_digest = _source(companions["dflash2"])
        inputs.update(df_files)
        config = DFlash2Config.from_dict(df_raw)
        # The existing DF operator constructs its base/exponents/frequencies in
        # FP32. Check their representability without changing that algorithm.
        try:
            base = struct.unpack("f", struct.pack("f", config.theta))[0]
            if not math.isfinite(base) or base <= 0:
                raise ValueError("nonpositive/nonfinite FP32 base")
            last = struct.unpack("f", struct.pack("f", -(config.head_dim // 2 - 1) * (2.0 / config.head_dim)))[0]
            frequency = struct.unpack("f", struct.pack("f", base ** last))[0]
            if not math.isfinite(frequency) or frequency <= 0:
                raise ValueError("nonpositive/nonfinite FP32 frequency")
        except (OverflowError, ValueError) as error:
            raise ValueError("DFlash positional source cannot represent its FP32 frequency algorithm") from error
        if declaration is not None and (not isinstance(declaration, dict) or set(declaration) != {
                "source_config_sha256", "algorithm", "coordinate_limit"}
                or declaration["source_config_sha256"] != df_digest or declaration["algorithm"] != DFLASH_ALGORITHM):
            raise ValueError("DFlash positional role differs from its actual source/implemented numerical algorithm")
        limit = config.context if declaration is None else declaration["coordinate_limit"]
        _integer(limit, "DFlash coordinate limit")
        _integer(config.context, "DFlash native context")
        if limit > config.context:
            raise ValueError("DFlash extension requires a separately implemented/qualified draft positional recipe")
        dflash = DFlashPositionRole(config.theta, config.head_dim, config.context, min(requested, config.context, limit),
                                   limit, config.window, df_digest)
    elif declaration is not None:
        raise ValueError("DFlash positional declaration lacks the actual selected draft configuration")
    # Source file hashes are computed only at preparation. No hidden mutable
    # cache or serving-time IO participates in frequency/graph dispatch.
    algorithms = {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                  for name in ("rope.py", "position_profile.py")}
    receipt = {"schema_version": SCHEMA_VERSION, "family": "qwen4_exp", "epoch": epoch,
               "requested_context": requested, "qualification": "unqualified-candidate",
               "target_attention": target.receipt(), "target_indexer": target.receipt(),
               "mtp_attention": mtp.receipt(), "mtp_indexer": mtp.receipt(),
               "dflash2": None if dflash is None else dflash.receipt(),
               "reference_revisions": {"transformers": HF_REVISION, "indexer": INDEXER_REVISION,
                                       "dflash": DFLASH_REVISION}, "local_algorithm_sha256": algorithms,
               "source_files": [str(path) for path in sorted(inputs)]}
    encoded = _json(receipt)
    return ResolvedPositionProfile(target, mtp, dflash, epoch, tuple(sorted(inputs)), encoded,
                                   hashlib.sha256(encoded.encode()).hexdigest())


def _frequency_key(policy):
    # These are precisely the inputs read by inverse_frequencies. Row/axis
    # bounds, source provenance and rotary amplitude do not change that vector.
    return (policy.rope_type, policy.theta, policy.rotary_dim,
            *((policy.factor, policy.correction_range) if policy.rope_type == "yarn" else ()))


def position_vector_bytes(profile, *, mtp_active):
    """Logical persistent bytes; existing target vector is already budgeted.

    Call before capacity and use mtp_extra_vector_bytes exactly once. Equal
    target/MTP transforms share the target tensor, even with separate source
    provenance. No state/session phase arrays are allocated by this binding.
    """
    if type(mtp_active) is not bool:
        raise ValueError("positional MTP residency must be explicit")
    target = profile.target.rope.rotary_dim // 2 * 4
    extra = (profile.mtp.rope.rotary_dim // 2 * 4
             if mtp_active and _frequency_key(profile.mtp.rope) != _frequency_key(profile.target.rope) else 0)
    return {"target_vector_bytes": target, "mtp_extra_vector_bytes": extra, "total_vector_bytes": target + extra}


@dataclass(frozen=True)
class PositionOperands:
    inv_freq: object
    amplitude: float
    sections: tuple[int, int, int]
    domain: PositionRole | None
    identity: str | None
    epoch: int | None


@dataclass(frozen=True)
class BoundPositionProfile:
    profile: ResolvedPositionProfile
    target: PositionOperands
    mtp: PositionOperands | None


def bind_position_profile(weights, profile, *, mtp_active):
    """Publish once after load, before states/graphs can borrow the vectors."""
    import torch

    position_vector_bytes(profile, mtp_active=mtp_active)
    if weights.positions is not None:
        prior = weights.positions
        if prior.profile.identity != profile.identity or (prior.mtp is not None) != mtp_active:
            raise ValueError("positional rebinding requires retiring the resident model/state/graphs")
        return prior
    if (weights.cfg.rope != profile.target.rope or weights.inv_freq.dtype != torch.float32
            or weights.inv_freq.shape != (profile.target.rope.rotary_dim // 2,)
            or not weights.inv_freq.is_contiguous()):
        raise ValueError("loaded target positional operands differ from the resolved immutable profile")
    if mtp_active and (weights.mtp is None or profile.target.max_rows > profile.mtp.max_rows):
        raise ValueError("MTP positional binding lacks its admitted source and complete row domain")
    target = PositionOperands(weights.inv_freq, profile.target.rope.attention_factor,
                              profile.target.rope.mrope_section, profile.target, profile.identity, profile.epoch)
    mtp = None
    if mtp_active:
        policy = profile.mtp.rope
        inv = (weights.inv_freq if _frequency_key(policy) == _frequency_key(profile.target.rope)
               else policy.inverse_frequencies(torch).to(weights.device))
        mtp = PositionOperands(inv, policy.attention_factor, policy.mrope_section,
                               profile.mtp, profile.identity, profile.epoch)
    binding = BoundPositionProfile(profile, target, mtp)
    weights.positions = binding
    return binding


def position_for(weights, *, mtp=False):
    """Read prepared operands; preserve the existing unselected platform path."""
    binding = weights.positions
    if binding is None:
        return PositionOperands(weights.inv_freq, weights.cfg.rope_attention_factor,
                                weights.cfg.mrope_section, None, None, None)
    if mtp and binding.mtp is None:
        raise ValueError("MTP positional operands were not admitted/prepared for this model")
    return binding.mtp if mtp else binding.target

"""Host-only Qwen3.5 architecture and supported text-RoPE metadata (default, or static YaRN).

The CUDA kernels implement the named linear/full attention primitives, the
unscaled inverse-frequency formula and YaRN's frequencies with its rotary
amplitude. Metadata is resolved once before payload loading; accepting another
RoPE algorithm would change checkpoint math.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from typing import Any

# Decoded JSON field types are invalid metadata values. The boundary consistently
# reports ValueError, rather than treating file contents as Python API misuse.
_LAYER_KINDS = frozenset(("linear_attention", "full_attention"))


def _count(value: Any, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _config_int(value: Any, name: str) -> int:
    """Preserve Config.read's integer normalization with one metadata error type."""
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must be convertible to an integer") from error


def resolve_layer_types(text: dict[str, Any], *, layers: int) -> tuple[str, ...]:
    """Resolve an explicit main layout; empty means arithmetic interval dispatch.

    Transformers5.2 uses ``full_attention_interval`` only when ``layer_types``
    is absent/null. A declared list therefore takes precedence over that hint.
    The absent/null case validates the interval without materializing a tuple
    from an untrusted scalar count. ``layers`` is the normalized main count.
    Some supported
    exports append MTP entries to per-layer lists: only an exact tail declared
    by ``num_nextn_predict_layers`` may be removed. This does not enable MTP.
    """
    _count(layers, "num_hidden_layers")
    kinds = text.get("layer_types")
    if kinds is None:
        _count(text.get("full_attention_interval", 4), "full_attention_interval", minimum=1)
        return ()
    if not isinstance(kinds, list):
        raise ValueError("layer_types must be a JSON array")  # noqa: TRY004
    if len(kinds) != layers:
        auxiliary = _count(text.get("num_nextn_predict_layers", 0), "num_nextn_predict_layers")
        if not auxiliary or len(kinds) != layers + auxiliary:
            raise ValueError("layer_types must contain num_hidden_layers entries, optionally followed by exactly "
                             "num_nextn_predict_layers declared MTP entries")
    if any(not isinstance(kind, str) or kind not in _LAYER_KINDS for kind in kinds):
        raise ValueError("Qwen3.5 CUDA layer_types supports linear_attention and full_attention only")
    return tuple(kinds[:layers])


@dataclass(frozen=True)
class DefaultRope:
    theta: float
    rotary_dim: int
    mrope_section: tuple[int, int, int]


def _number(value: Any, name: str, *, zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")  # noqa: TRY004
    try:
        number = float(value)
    except OverflowError as error:
        raise ValueError(f"{name} must be a finite number") from error
    if not math.isfinite(number) or (number < 0 if zero else number <= 0):
        raise ValueError(f"{name} must be finite and {'nonnegative' if zero else 'positive'}")
    return number


def _rope_kind(parameters: dict[str, Any]) -> str:
    """The declared algorithm, refusing per-layer dictionaries and anything but default or static YaRN."""
    if any(kind in parameters for kind in _LAYER_KINDS):
        raise ValueError("Qwen3.5 CUDA does not support per-layer RoPE parameter dictionaries")
    kind = parameters.get("rope_type", parameters.get("type", "default"))
    if "rope_type" in parameters and "type" in parameters and parameters["rope_type"] != parameters["type"]:
        raise ValueError("rope_type and type disagree")
    if kind not in ("default", "yarn"):
        raise ValueError(f"Qwen3.5 CUDA supports default or static YaRN text RoPE, not {kind!r}")
    return kind


def declared_yarn(raw: dict[str, Any]):
    """The config's own static YaRN as the shared ``qwen4_exp.rope.RopeParameters`` (validated there), or None when
    it declares default RoPE only (``--yarn-factor`` can also select YaRN). When both ``rope_parameters`` and
    ``rope_scaling`` are declared, each must resolve to the same YaRN policy (as ``resolve_default_rope`` requires of
    default ones)."""
    text = raw.get("text_config") or raw
    sets = _rope_parameter_sets(text)
    kinds = [_rope_kind(parameters) for parameters in sets]
    if "yarn" not in kinds:
        return None
    if "default" in kinds:
        raise ValueError("rope_parameters and rope_scaling disagree: one declares YaRN, the other default RoPE")
    from tensorfold.families.qwen4_exp.rope import RopeParameters

    rest = {k: v for k, v in text.items() if k not in ("rope_parameters", "rope_scaling")}
    policies = [RopeParameters.from_config({**rest, "rope_parameters": parameters}) for parameters in sets]
    if len(policies) == 2 and policies[0] != policies[1]:
        raise ValueError("rope_parameters and rope_scaling disagree")
    return policies[0]


def _default_rope(text: dict[str, Any], parameters: dict[str, Any], head_dim: int) -> DefaultRope:
    if _rope_kind(parameters) != "default":
        raise ValueError("a YaRN checkpoint's rotary policy is resolved by its RopeParameters")
    theta = _number(parameters.get("rope_theta", text.get("rope_theta", 10000000.0)), "rope_theta")
    partial = _number(parameters.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25)),
                      "partial_rotary_factor", zero=True)
    if partial > 1:
        raise ValueError("partial_rotary_factor must not exceed one")
    # Match the existing loader and pinned upstream's truncation, not a newly
    # invented exact-product rule. The active attention geometry admits >=2.
    rotary = int(head_dim * partial)
    if rotary % 2:
        raise ValueError("partial_rotary_factor must produce an even rotary dimension")
    sections = parameters.get("mrope_section", (11, 11, 10))
    if not isinstance(sections, (list, tuple)) or len(sections) != 3:
        raise ValueError("mrope_section must contain three nonnegative integer pair counts")
    sections = tuple(_count(value, "mrope_section") for value in sections)
    # Section coverage belongs to the optional vision frontend, which knows
    # whether multimodal rotary is active. Text-only rotation ignores sections.
    return DefaultRope(theta, rotary, sections)


def _rope_parameter_sets(text: dict[str, Any]) -> list[dict[str, Any]]:
    parameters = []
    for field in ("rope_parameters", "rope_scaling"):
        value = text.get(field)
        if value is None:
            continue
        if not isinstance(value, dict):
            raise ValueError(f"{field} must be a JSON object or null")  # noqa: TRY004
        parameters.append(value)
    return parameters or [{}]


def resolve_default_rope(text: dict[str, Any], *, head_dim: int) -> DefaultRope:
    """Return the current unscaled policy; reject unsupported declared math.

    Legacy ``rope_scaling``/``type`` spellings are recognized. When both
    parameter objects are declared, their effective default policies must agree.
    Unrelated default-mode fields do not select a scaling algorithm. No input
    dictionaries are mutated, and no accelerator runtime is imported.
    """
    _count(head_dim, "head_dim")
    if head_dim > 2**31 - 1:
        raise ValueError("head_dim exceeds signed32-bit kernel dimensions")
    policies = [_default_rope(text, parameters, head_dim) for parameters in _rope_parameter_sets(text)]
    if len(policies) == 2 and policies[0] != policies[1]:
        raise ValueError("rope_parameters and rope_scaling disagree")
    return policies[0]


def _finite_frequencies(rope: DefaultRope) -> None:
    """Admit the largest unscaled frequency for the loader's FP32 storage.

    For theta>=1 every frequency lies in [0,1]. For smaller positive theta,
    frequencies increase with the pair index, so the final pair bounds them.
    This checks storage range, not cross-library bit identity of exponentiation.
    """
    half = rope.rotary_dim // 2
    if rope.theta >= 1 or half <= 1:
        return
    try:
        largest = math.pow(rope.theta, -(half - 1) / half)
        finite = math.isfinite(struct.unpack("f", struct.pack("f", largest))[0])
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("default RoPE inverse frequencies exceed finite float32 range")


def validate_cuda_metadata(raw: dict[str, Any]) -> None:
    """Check CUDA-only layout/RoPE declarations before format dispatch/load.

    Format-capability probes may omit architecture dimensions. Such probes
    still reject unsupported declared RoPE algorithms; complete configs also
    receive layout and frequency-geometry checks. Required model fields and
    installed-kernel geometry remain the loading Config's responsibility.
    Call this only for CUDA; other providers own their supported RoPE policies.
    """
    if not isinstance(raw, dict):
        raise ValueError("config must be a JSON object")  # noqa: TRY004
    selected = raw.get("text_config")
    if selected is not None and not isinstance(selected, dict):
        raise ValueError("text_config must be a JSON object or null")
    text = selected or raw
    if text.get("hidden_act", "silu") != "silu":
        raise ValueError("Qwen3.5 CUDA supports hidden_act='silu' only")
    attention_bias = text.get("attention_bias", False)
    if type(attention_bias) is not bool:
        raise ValueError("attention_bias must be a JSON boolean")
    if attention_bias:
        raise ValueError("Qwen3.5 CUDA does not support attention_bias=true")
    parameters = _rope_parameter_sets(text)
    for item in parameters:
        _rope_kind(item)
    attention = None
    if "num_hidden_layers" in text:
        layers = _count(_config_int(text["num_hidden_layers"], "num_hidden_layers"), "num_hidden_layers")
        if text.get("layer_types") is None:
            interval = _count(text.get("full_attention_interval", 4), "full_attention_interval", minimum=1)
            # Admission runs before resource fitting: do not expand a scalar
            # layer count into unbounded metadata merely to count its pattern.
            attention = layers >= interval
        else:
            attention = "full_attention" in resolve_layer_types(text, layers=layers)
    elif text.get("layer_types") is not None:
        raise ValueError("layer_types requires num_hidden_layers")
    elif "full_attention_interval" in text:
        _count(text["full_attention_interval"], "full_attention_interval", minimum=1)
    if declared_yarn(raw) is not None:             # validated by its policy; its frequencies are finite by build
        return
    head_dim = text.get("head_dim")
    if head_dim:
        rope = resolve_default_rope(text, head_dim=_config_int(head_dim, "head_dim"))
    elif "hidden_size" in text and "num_attention_heads" in text:
        heads = _config_int(text["num_attention_heads"], "num_attention_heads")
        if heads <= 0:
            raise ValueError("num_attention_heads must be positive to derive head_dim")
        rope = resolve_default_rope(text, head_dim=_config_int(text["hidden_size"], "hidden_size") // heads)
    else:
        return
    if attention is not False:
        _finite_frequencies(rope)


def check_metadata(raw: dict[str, Any], backend: str) -> None:
    """Family preflight hook; other backends retain their own math capabilities."""
    if backend == "cuda":
        validate_cuda_metadata(raw)

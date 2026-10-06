"""Validated static text RoPE policy shared by Flash Next startup and CUDA weight loaders.

YaRN follows Transformers' ``_compute_yarn_parameters``: interpolate low-frequency
pairs, retain high-frequency pairs, and scale the rotary cos/sin amplitudes. The
vision tower retains its own axial RoPE. Policy is immutable for a loaded engine;
caches must never be reused across policies.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

FLOAT32_MAX = float.fromhex("0x1.fffffep+127")
FLOAT32_MIN_POSITIVE = float.fromhex("0x1p-149")


def _number(value: Any, name: str, *, minimum: float = 0, inclusive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    try:
        number = float(value)
    except OverflowError as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(number) or (number < minimum if inclusive else number <= minimum):
        relation = "at least" if inclusive else "greater than"
        raise ValueError(f"{name} must be finite and {relation} {minimum}")
    return number


def _integer(value: Any, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return value


def _correction_range(dim: int, theta: float, original: int, fast: float, slow: float,
                      truncate: bool) -> tuple[float, float]:
    try:
        low = dim * math.log(original / (fast * 2 * math.pi)) / (2 * math.log(theta))
        high = dim * math.log(original / (slow * 2 * math.pi)) / (2 * math.log(theta))
        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("nonfinite correction range")
        if truncate:
            low, high = math.floor(low), math.ceil(high)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("YaRN beta parameters produce an unrepresentable correction range") from exc
    low, high = max(low, 0), min(high, dim - 1)
    if low == high:
        high += 0.001
    return low, high


@dataclass(frozen=True)
class RopeParameters:
    rope_type: str
    theta: float
    rotary_dim: int
    native_context: int
    context_limit: int
    factor: float = 1.0
    attention_factor: float = 1.0
    beta_fast: float = 32.0
    beta_slow: float = 1.0
    truncate: bool = True
    mrope_section: tuple[int, int, int] = (11, 11, 10)
    correction_range: tuple[float, float] | None = None

    @classmethod
    def from_config(cls, raw: dict[str, Any], yarn_factor: float | None = None) -> "RopeParameters":
        text = raw.get("text_config")
        if text is None:
            text = raw
        if not isinstance(text, dict):
            raise ValueError("text_config must be an object")
        params = text.get("rope_parameters")
        if params is None:
            params = text.get("rope_scaling")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("rope_parameters must be an object")
        params = dict(params)
        current = params.get("rope_type", params.get("type", "default"))
        if "rope_type" in params and "type" in params and params["rope_type"] != params["type"]:
            raise ValueError("rope_type and type disagree")
        if current not in ("default", "yarn"):
            raise ValueError(f"Flash Next CUDA supports default or yarn text RoPE, not {current!r}")
        if yarn_factor is not None:
            params["factor"] = _number(yarn_factor, "--yarn-factor", minimum=1, inclusive=True)
            current = "yarn"
        theta = _number(params.get("rope_theta", 10_000_000), "rope_theta", minimum=1)
        head_dim = text.get("head_dim")
        if head_dim is None:
            hidden = _integer(text.get("hidden_size"), "hidden_size")
            heads = _integer(text.get("num_attention_heads"), "num_attention_heads")
            if hidden % heads:
                raise ValueError("hidden_size must be divisible by num_attention_heads")
            head_dim = hidden // heads
        head_dim = _integer(head_dim, "head_dim")
        if head_dim > 2**31 - 1:
            raise ValueError("head_dim exceeds signed 32-bit kernel dimensions")
        partial = _number(params.get("partial_rotary_factor", text.get("partial_rotary_factor", 0.25)),
                          "partial_rotary_factor")
        rotary = head_dim * partial
        if partial > 1 or not rotary.is_integer() or int(rotary) <= 0 or int(rotary) % 2:
            raise ValueError("partial_rotary_factor must produce a positive even rotary dimension within head_dim")
        sections = params.get("mrope_section", (11, 11, 10))
        if not isinstance(sections, (list, tuple)) or len(sections) != 3:
            raise ValueError("mrope_section must contain three nonnegative integer pair counts")
        sections = tuple(_integer(value, "mrope_section", minimum=0) for value in sections)
        if "mrope_section" in params and sum(sections) != int(rotary) // 2:
            raise ValueError("mrope_section must sum to half the rotary dimension")
        if params.get("mrope_interleaved", True) is not True:
            raise ValueError("Flash Next CUDA supports interleaved text M-RoPE only")
        declared = _integer(text.get("max_position_embeddings", raw.get("max_position_embeddings", 0)),
                            "max_position_embeddings", minimum=0)
        if declared > 2**31 - 1:
            raise ValueError("max_position_embeddings exceeds signed 32-bit positions")
        if current == "default":
            return cls(current, theta, int(rotary), declared, declared, mrope_section=sections)
        original = params.get("original_max_position_embeddings")
        if original is None and yarn_factor is not None:
            original = declared
        original = _integer(original, "YaRN original_max_position_embeddings")
        if original > 2**31 - 1:
            raise ValueError("YaRN original context exceeds signed 32-bit positions")
        factor = _number(params.get("factor"), "YaRN factor", minimum=1, inclusive=True)
        if theta > FLOAT32_MAX:
            raise ValueError("YaRN rope_theta must fit the float32 frequency calculation")
        extension = original * factor
        # Cache and rotary position tensors use signed 32-bit positions.
        if not math.isfinite(extension) or extension > 2**31 - 1:
            raise ValueError("YaRN context limit exceeds signed 32-bit positions")
        limit = math.floor(extension)
        if declared > limit:
            raise ValueError(f"max_position_embeddings {declared} exceeds YaRN's {limit}-token limit")
        fast = _number(params.get("beta_fast", 32.0), "YaRN beta_fast")
        slow = _number(params.get("beta_slow", 1.0), "YaRN beta_slow")
        if fast <= slow:
            raise ValueError("YaRN beta_fast must exceed beta_slow")
        truncate = params.get("truncate", True)
        if not isinstance(truncate, bool):
            raise ValueError("YaRN truncate must be a boolean")
        attention = params.get("attention_factor")
        if attention is None:
            mscale, all_dim = params.get("mscale"), params.get("mscale_all_dim")
            if (mscale is None) != (all_dim is None):
                raise ValueError("YaRN mscale and mscale_all_dim must be provided together")
            if mscale is not None:
                mscale = _number(mscale, "YaRN mscale", inclusive=True)
                all_dim = _number(all_dim, "YaRN mscale_all_dim", inclusive=True)
            attention = 1.0 + 0.1 * math.log(factor)
            if mscale and all_dim:
                attention = (1.0 + 0.1 * mscale * math.log(factor)) / (1.0 + 0.1 * all_dim * math.log(factor))
        attention = _number(attention, "YaRN attention_factor")
        if not FLOAT32_MIN_POSITIVE <= attention <= FLOAT32_MAX:
            raise ValueError("YaRN attention_factor must be representable as positive finite float32")
        correction = _correction_range(int(rotary), theta, original, fast, slow, truncate)
        return cls(current, theta, int(rotary), original, limit, factor, attention, fast, slow, truncate, sections,
                   correction)

    def inverse_frequencies(self, torch: Any):
        """Build one small CPU tensor at load time; native frequencies keep the pre-fork float64 arithmetic."""

        half = self.rotary_dim // 2
        if self.rope_type == "default":
            inv = torch.tensor(self.theta, dtype=torch.float64) ** (
                -torch.arange(0, half, dtype=torch.float64) / half)
            return inv.to(torch.float32)
        dim = self.rotary_dim
        pos_freqs = self.theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
        extrapolation = 1.0 / pos_freqs
        interpolation = 1.0 / (self.factor * pos_freqs)
        if self.correction_range is None:
            raise ValueError("YaRN policy must be initialized through from_config")
        low, high = self.correction_range
        ramp = ((torch.arange(half, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
        retain = 1 - ramp
        inv = interpolation * (1 - retain) + extrapolation * retain
        if not bool(torch.isfinite(inv).all()) or not bool((inv > 0).all()):
            raise ValueError("YaRN parameters produce nonfinite or zero float32 inverse frequencies")
        return inv

    def metadata(self) -> dict[str, Any]:
        return {"rope_type": self.rope_type, "factor": self.factor, "attention_factor": self.attention_factor,
                "original_max_position_embeddings": self.native_context, "context_limit": self.context_limit,
                "rope_theta": self.theta, "rotary_dim": self.rotary_dim,
                "beta_fast": self.beta_fast, "beta_slow": self.beta_slow, "truncate": self.truncate,
                "mrope_section": list(self.mrope_section), "mrope_interleaved": True}

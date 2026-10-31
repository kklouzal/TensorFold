"""Strict, opt-in metadata for the Qwen3 dense DSpark reference adapter.

No engine or CLI selects this experimental adapter. Other DSpark architectures
and checkpoints with different projector/attention conventions are refused.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path


def _integer(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"DSpark {name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class DSparkConfig:
    hidden: int
    vocab: int
    layers: int
    heads: int
    kv_heads: int
    head_dim: int
    intermediate: int
    target_layers: int
    taps: tuple[int, ...]
    mask: int
    rank: int
    block: int
    context: int
    eps: float
    theta: float
    confidence: bool = True

    @classmethod
    def from_dict(cls, raw):
        if (
            not isinstance(raw, dict)
            or raw.get("architectures") != ["DSparkDraftModel"]
            or raw.get("model_type") != "qwen3"
        ):
            raise ValueError("DSpark reference requires the dense Qwen3 DSparkDraftModel architecture")
        extra = raw.get("dflash_config")
        if not isinstance(extra, dict):
            raise ValueError("DSpark requires dflash_config")
        required = {
            "attention_mode": "gqa",
            "projector_type": "dspark",
            "markov_head_type": "vanilla",
            "confidence_head_with_markov": True,
        }
        for key, expected in required.items():
            actual = extra.get(key, raw.get(key))
            if actual != expected:
                raise ValueError(f"DSpark {key}={actual!r} is unsupported; requires {expected!r}")
        if extra.get("confidence_head_alpha", raw.get("confidence_head_alpha", 1.0)) != 1.0:
            raise ValueError("DSpark confidence alpha differs from the pinned reference")
        for key in ("conv_kernel_size", "selector_rank", "selector_top_k"):
            if extra.get(key, 0):
                raise ValueError(f"DSpark {key} is unsupported")
        if extra.get("attention_sink_bias", False) or extra.get("attention_value_scale", 1.0) != 1.0:
            raise ValueError("DSpark attention sink/value scaling is unsupported")
        if raw.get("partial_rotary_factor", 1.0) != 1.0 or raw.get("rope_is_neox_style", True) is not True:
            raise ValueError("DSpark requires full NeoX RoPE")
        for key in ("attention_bias", "use_sliding_window", "is_causal", "fp32_residual"):
            if raw.get(key, False) is not False:
                raise ValueError(f"DSpark {key} is unsupported")
        for key in ("rope_scaling", "sliding_window"):
            if raw.get(key) is not None:
                raise ValueError(f"DSpark {key} is unsupported")
        if raw.get("sample_from_anchor", True) is not True or raw.get("hidden_act", "silu") != "silu":
            raise ValueError("DSpark requires sample_from_anchor=True and silu")
        if raw.get("target_hidden_size") not in (None, raw.get("hidden_size")):
            raise ValueError("DSpark target and draft hidden dimensions must match")
        fields = {
            "hidden": "hidden_size",
            "vocab": "vocab_size",
            "layers": "num_hidden_layers",
            "heads": "num_attention_heads",
            "kv_heads": "num_key_value_heads",
            "head_dim": "head_dim",
            "intermediate": "intermediate_size",
            "target_layers": "num_target_layers",
            "block": "block_size",
            "context": "max_position_embeddings",
        }
        values = {key: _integer(raw.get(name), name) for key, name in fields.items()}
        taps = extra.get("target_layer_ids")
        if (
            not isinstance(taps, list)
            or not taps
            or any(type(t) is not int or t < 0 or t >= values["target_layers"] - 1 for t in taps)
            or sorted(set(taps)) != taps
        ):
            raise ValueError("DSpark taps must be ordered unique target output-layer IDs before the final layer")
        for key in ("markov_rank", "enable_confidence_head", "confidence_head_with_markov", "markov_head_type"):
            if key in raw and key in extra and raw[key] != extra[key]:
                raise ValueError(f"DSpark conflicting {key}")
        rank = _integer(extra.get("markov_rank", raw.get("markov_rank")), "markov_rank")
        confidence = extra.get("enable_confidence_head", raw.get("enable_confidence_head", True))
        if type(confidence) is not bool:
            raise ValueError("DSpark enable_confidence_head must be boolean")
        mask = _integer(extra.get("mask_token_id"), "mask_token_id", 0)
        if mask >= values["vocab"] or values["head_dim"] % 2 or values["heads"] % values["kv_heads"]:
            raise ValueError("DSpark mask/head geometry is invalid")
        if raw.get("layer_types") != ["full_attention"] * values["layers"]:
            raise ValueError("DSpark requires one full_attention declaration per draft layer")
        eps, theta = raw.get("rms_norm_eps"), raw.get("rope_theta")
        if any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or x <= 0
            for x in (eps, theta)
        ):
            raise ValueError("DSpark RMS epsilon and RoPE theta must be finite and positive")
        return cls(
            **values, taps=tuple(taps), mask=mask, rank=rank, eps=float(eps), theta=float(theta), confidence=confidence
        )

    @classmethod
    def read(cls, directory):
        from tensorfold.cuda.tensor_file import read_metadata_json

        return cls.from_dict(read_metadata_json(Path(directory) / "config.json"))

    @property
    def capture_before(self):
        # SGLang0.5.21 Qwen3VL wrapper adds1, then the HF add-on captures
        # BEFORE the marked layer. IDs describe preceding layer outputs.
        return tuple(t + 1 for t in self.taps)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()

    def check_target(self, target):
        cfg = getattr(target, "cfg", target)
        for name, expected in (("hidden", self.hidden), ("vocab", self.vocab), ("layers", self.target_layers)):
            if getattr(cfg, name, None) != expected:
                raise ValueError(f"DSpark target {name} differs from the trained geometry")
        if getattr(cfg, "streams", None) != 4:
            raise ValueError("DSpark Qwen4 target capture requires four HC branches")

    def partition(self, rank=0, world=1):
        if type(world) is not int or world not in (1, 2, 4, 8) or type(rank) is not int or not 0 <= rank < world:
            raise ValueError("DSpark TP requires rank within world1/2/4/8")
        if (
            self.heads % world
            or self.intermediate % world
            or (self.kv_heads % world if self.kv_heads >= world else world % self.kv_heads)
        ):
            raise ValueError("DSpark attention and MLP dimensions do not partition this TP world")
        kv = max(1, self.kv_heads // world)
        first_kv = rank * kv if self.kv_heads >= world else rank // (world // self.kv_heads)
        return {
            "heads": self.heads // world,
            "kv_heads": kv,
            "first_kv": first_kv,
            "intermediate": self.intermediate // world,
        }

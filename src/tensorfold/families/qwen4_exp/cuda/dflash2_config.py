"""Validated DFlash2 family contract, independent of publisher or file names.

The first serving profile is FP16 TP8, a four-token non-causal block and the
checkpoint's physically bounded sliding window. Other dimensions are derived
from the supplied manifest; unsupported operators fail before device loading.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math


def count(value, name, minimum=1):
    if type(value) is not int or value < minimum:
        raise ValueError(f"DFlash2 {name} must be an integer >= {minimum}")
    return value


@dataclass(frozen=True)
class DFlash2Config:
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
    selector_rank: int
    top_k: int
    window: int
    context: int
    eps: float
    theta: float
    source_block: int
    block: int = 4
    group_size: int = 16
    conv_taps: int = 2
    hc_streams: int = 4
    output_multiplier: float = 1.0
    logit_softcap: float | None = None

    @classmethod
    def from_dict(cls, raw, *, block=4):
        if not isinstance(raw, dict) or raw.get("architectures") != ["DFlash2DraftModel"]:
            raise ValueError("DFlash2DraftModel architecture required")
        extra = raw.get("dflash_config")
        if not isinstance(extra, dict):
            raise ValueError("DFlash2 dflash_config is required")
        if block != 4:
            raise ValueError("only the four-row DFlash2 serving block is implemented")
        for key, expected in (("attention_bias", False), ("is_causal", False), ("hidden_act", "silu")):
            if raw.get(key, expected) != expected:
                raise ValueError(f"DFlash2 operator {key} is unsupported")
        if raw.get("use_sliding_window") is not True:
            raise ValueError("DFlash2 requires physically bounded sliding-window attention")
        if raw.get("partial_rotary_factor", 1.0) != 1.0 or raw.get("rope_is_neox_style", True) is not True:
            raise ValueError("DFlash2 requires full-width NeoX RoPE")
        rope = raw.get("rope_parameters", raw.get("rope_scaling")) or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("DFlash2 RoPE scaling needs a separately implemented profile")
        if set(rope) - {"rope_type", "rope_theta"}:
            raise ValueError("DFlash2 extended/partial rotary operators need a separately implemented profile")
        if extra.get("attention_value_scale") is not None or extra.get("attention_sink_bias", False):
            raise ValueError("DFlash2 value scaling/attention sinks are not implemented")
        fields = {"hidden": "hidden_size", "vocab": "vocab_size", "layers": "num_hidden_layers",
                  "heads": "num_attention_heads", "kv_heads": "num_key_value_heads", "head_dim": "head_dim",
                  "intermediate": "intermediate_size", "target_layers": "num_target_layers",
                  "window": "sliding_window", "context": "max_position_embeddings"}
        values = {key: count(raw.get(source), source) for key, source in fields.items()}
        if raw.get("layer_types") != ["sliding_attention"] * values["layers"]:
            raise ValueError("DFlash2 every declared draft layer must have its bounded window")
        taps = extra.get("target_layer_ids")
        if (not isinstance(taps, list) or not taps or any(type(i) is not int or not 0 <= i < values["target_layers"] for i in taps)
                or taps != sorted(set(taps))):
            raise ValueError("DFlash2 completed target taps must be ordered unique valid block IDs")
        mask = count(extra.get("mask_token_id"), "mask token", 0)
        if mask >= values["vocab"] or values["head_dim"] % 2 or values["heads"] % values["kv_heads"]:
            raise ValueError("DFlash2 vocabulary/head geometry is invalid")
        group = count(extra.get("conv_group_size"), "convolution group size")
        conv = count(extra.get("conv_kernel_size"), "convolution width")
        if conv != 2 or values["hidden"] % group:
            raise ValueError("DFlash2 requires two-tap grouped block-local convolutions")
        if any(isinstance(value, bool) for value in (raw.get("rms_norm_eps"), rope.get("rope_theta"),
                                                     extra.get("output_multiplier"), extra.get("final_logit_softcapping"))):
            raise ValueError("DFlash2 numerical parameters cannot be booleans")
        eps, theta = float(raw.get("rms_norm_eps", 1e-6)), float(rope.get("rope_theta", raw.get("rope_theta", 10000)))
        multiplier = float(extra.get("output_multiplier", 1.0))
        softcap = extra.get("final_logit_softcapping")
        if softcap is not None:
            softcap = float(softcap)
        if any(not math.isfinite(v) or v <= 0 for v in (eps, theta, multiplier, *(() if softcap is None else (softcap,)))):
            raise ValueError("DFlash2 normalization/position/logit parameters must be finite positive")
        top_k = count(extra.get("selector_top_k"), "selector top-k")
        if top_k > values["vocab"]:
            raise ValueError("DFlash2 selector support exceeds actual vocabulary")
        source_block = count(extra.get("block_size"), "training block")
        if source_block < block:
            raise ValueError("DFlash2 fixed execution block exceeds the source checkpoint block")
        return cls(**values, taps=tuple(taps), mask=mask,
                   selector_rank=count(extra.get("selector_rank"), "selector rank"),
                   top_k=top_k,
                   source_block=source_block, block=block,
                   group_size=group, conv_taps=conv, eps=eps, theta=theta,
                   output_multiplier=multiplier, logit_softcap=softcap)

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True, separators=(",", ":")).encode()).hexdigest()

    @classmethod
    def read(cls, directory, *, block=4):
        from tensorfold.cuda.tensor_file import checkpoint_path, read_metadata_json
        return cls.from_dict(read_metadata_json(checkpoint_path(directory, "config.json")), block=block)

    def partition(self, rank, world):
        if (type(rank) is not int or type(world) is not int or world < 1 or not 0 <= rank < world
                or any(n % world for n in (self.heads, self.kv_heads, self.intermediate, self.vocab))):
            raise ValueError("DFlash2 TP must divide actual Q/KV heads and intermediate width")
        return {"heads": self.heads // world, "kv_heads": self.kv_heads // world,
                "intermediate": self.intermediate // world}

    def check_target(self, target):
        cfg = target.meta.get("full", target.cfg) if hasattr(target, "meta") else target
        for key, actual in (("hidden", self.hidden), ("vocab", self.vocab), ("layers", self.target_layers),
                            ("streams", self.hc_streams)):
            if getattr(cfg, key, None) != actual:
                raise ValueError("DFlash2 target feature/embedding contract differs: " + key)

    def state_bytes(self, *, tp=8, context=None):
        local = self.partition(0, tp)
        window = min(self.window, self.context if context is None else count(context, "context"))
        return 2 * self.layers * window * local["kv_heads"] * self.head_dim * 2


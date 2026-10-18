"""Header-only contracts for the BF16 Gemma 4 Qwen3-style DFlash companion."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tensorfold.checkpoint_metadata import read_json
from tensorfold.families.qwen4_exp.rope import FLOAT32_MAX, FLOAT32_MIN_POSITIVE, RopeParameters


def positive(value, name):
    if type(value) is not int or not 1 <= value <= 2**31 - 1:
        raise ValueError(f"Gemma DFlash {name} must be a positive signed32 integer")
    return value


@dataclass(frozen=True)
class DraftConfig:
    hidden: int
    width: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    target_layers: int
    vocab: int
    tap_layers: tuple[int, ...]
    mask_id: int
    block: int
    eps: float
    layer_types: tuple[str, ...]
    causal_layers: tuple[bool, ...]
    sliding_window: int
    rope: RopeParameters

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("Gemma DFlash config must be an object")  # noqa: TRY004 - external JSON schema
        if raw.get("architectures") != ["DFlashDraftModel"] or raw.get("model_type") != "qwen3":
            raise ValueError("Gemma DFlash requires the Qwen3-style DFlash v1 companion")
        if any(raw.get(key) not in (None, "bfloat16") for key in ("dtype", "torch_dtype")):
            raise ValueError("Gemma DFlash declared execution dtype must be bfloat16")
        if raw.get("quantization_config") not in (None, {}) or raw.get("quantization") not in (None, {}):
            raise ValueError("Gemma DFlash requires original BF16 proposal weights")
        names = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                 "num_hidden_layers", "num_target_layers", "vocab_size")
        hidden, width, heads, kv, dim, layers, target, vocab = [positive(raw.get(k), k) for k in names]
        if heads % kv or dim < 16 or dim & (dim - 1):
            raise ValueError("Gemma DFlash grouped attention needs divisible heads and a power-of-two head dimension")
        if raw.get("attention_bias", False) is not False or raw.get("hidden_act", "silu") != "silu":
            raise ValueError("Gemma DFlash requires unbiased projections and SiLU")
        if raw.get("attention_dropout", 0.0) != 0.0:
            raise ValueError("Gemma DFlash inference requires zero attention dropout")
        kinds = raw.get("layer_types")
        if (not isinstance(kinds, list) or len(kinds) != layers
                or any(k not in ("full_attention", "sliding_attention") for k in kinds)):
            raise ValueError("Gemma DFlash requires one full/sliding attention declaration per layer")
        sliding = positive(raw.get("sliding_window"), "sliding window")
        if sliding < 2 or "sliding_attention" not in kinds or "full_attention" not in kinds:
            raise ValueError("Gemma DFlash requires mixed full and sliding attention with a window of at least two")
        if raw.get("use_sliding_window", True) is not True:
            raise ValueError("Gemma DFlash sliding attention cannot be disabled")
        causal = raw.get("is_causal")
        if causal is not None and type(causal) is not bool:
            raise ValueError("Gemma DFlash is_causal must be boolean or null")
        policies = tuple(k == "sliding_attention" if causal is None else causal for k in kinds)
        dc = raw.get("dflash_config")
        if not isinstance(dc, dict):
            raise ValueError("Gemma DFlash needs dflash_config")  # noqa: TRY004 - external JSON schema
        taps, mask = dc.get("target_layer_ids"), dc.get("mask_token_id")
        if (not isinstance(taps, list) or not taps
                or any(type(i) is not int or not 0 <= i < target for i in taps)
                or taps != sorted(set(taps))):
            raise ValueError("Gemma DFlash target taps must be unique increasing zero-based layer outputs")
        if type(mask) is not int or not 0 <= mask < vocab:
            raise ValueError("Gemma DFlash mask token must lie in the shared target vocabulary")
        block = positive(dc.get("block_size", raw.get("block_size")), "block size")
        if not 2 <= block <= 128:
            raise ValueError("Gemma DFlash blocks require 2 through128 rows, matching target tree verification")
        if block > sliding:
            raise ValueError("Gemma DFlash sliding blocks must fit the declared sliding window")
        # The borrowed Gemma projection already owns softcap30; accepting another
        # cap or a pre-cap multiplier would silently change the companion model.
        if dc.get("final_logit_softcapping", raw.get("final_logit_softcapping")) != 30.0:
            raise ValueError("Gemma DFlash requires the target's final logit softcap30")
        for key in ("input_embedding_scale", "output_multiplier"):
            value = dc.get(key, raw.get(key, 1.0))
            if isinstance(value, bool) or value != 1.0:
                raise ValueError(f"Gemma DFlash {key} must be one with the raw embedding and capped head boundary")
        eps = raw.get("rms_norm_eps")
        try:
            valid_eps = (not isinstance(eps, bool) and isinstance(eps, (int, float))
                         and FLOAT32_MIN_POSITIVE <= eps <= FLOAT32_MAX and math.isfinite(eps))
        except OverflowError:
            valid_eps = False
        if not valid_eps:
            raise ValueError("Gemma DFlash RMS epsilon must be representable as positive finite float32")
        params = raw.get("rope_parameters")
        if params is None:
            params = raw.get("rope_scaling")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("Gemma DFlash rotary parameters must be an object")  # noqa: TRY004 - external JSON schema
        params = dict(params)
        if params.get("rope_type", params.get("type", "default")) != "default":
            raise ValueError("Gemma DFlash companion requires its declared default full-head RoPE")
        if any(isinstance(v, bool) or v != 1 for v in
               (raw.get("partial_rotary_factor", 1), params.get("partial_rotary_factor", 1))):
            raise ValueError("Gemma DFlash uses full-head rotary dimensions")
        if "rope_theta" in params and "rope_theta" in raw and params["rope_theta"] != raw["rope_theta"]:
            raise ValueError("Gemma DFlash rotary theta declarations disagree")
        params.update(rope_theta=params.get("rope_theta", raw.get("rope_theta", 10000)), partial_rotary_factor=1)
        rope = RopeParameters.from_config({**raw, "rope_parameters": params})
        if rope.context_limit <= 0:
            raise ValueError("Gemma DFlash rotary context limit must be positive")
        if rope.theta > FLOAT32_MAX:
            raise ValueError("Gemma DFlash rotary theta must fit the FP32 inverse-frequency calculation")
        return cls(hidden, width, heads, kv, dim, layers, target, vocab, tuple(taps), mask, block, float(eps),
                   tuple(kinds), policies, sliding, rope)

    @classmethod
    def read(cls, directory):
        return cls.from_dict(read_json(directory, "config.json", required=True))

    def shapes(self):
        out = {"fc.weight": (self.hidden, len(self.tap_layers) * self.hidden),
               "hidden_norm.weight": (self.hidden,), "norm.weight": (self.hidden,)}
        for i in range(self.layers):
            base = f"layers.{i}."
            out.update({base + key: shape for key, shape in {
                "input_layernorm.weight": (self.hidden,), "post_attention_layernorm.weight": (self.hidden,),
                "self_attn.q_norm.weight": (self.head_dim,), "self_attn.k_norm.weight": (self.head_dim,),
                "self_attn.q_proj.weight": (self.heads * self.head_dim, self.hidden),
                "self_attn.k_proj.weight": (self.kv_heads * self.head_dim, self.hidden),
                "self_attn.v_proj.weight": (self.kv_heads * self.head_dim, self.hidden),
                "self_attn.o_proj.weight": (self.hidden, self.heads * self.head_dim),
                "mlp.gate_proj.weight": (self.width, self.hidden), "mlp.up_proj.weight": (self.width, self.hidden),
                "mlp.down_proj.weight": (self.hidden, self.width)}.items()})
        return out


def checked_headers(directory):
    from tensorfold.cuda.capacity import headers

    cfg = DraftConfig.read(directory)
    found = headers(directory)
    if len(found) != 11 * cfg.layers + 3:
        raise ValueError("Gemma DFlash tensor count differs from its declared architecture")
    expected = cfg.shapes()
    if set(found) != set(expected):
        raise ValueError("Gemma DFlash tensor inventory differs from its declared architecture")
    if any(tuple(found[k]["shape"]) != shape or found[k]["dtype"] != "BF16" for k, shape in expected.items()):
        raise ValueError("Gemma DFlash weights require declared shapes and BF16 storage")
    return cfg, found


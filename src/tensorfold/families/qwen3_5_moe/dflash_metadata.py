"""The published Qwen DFlash v1 architecture and startup memory contract."""
from __future__ import annotations

import math
from dataclasses import dataclass

from tensorfold.checkpoint_metadata import read_json
from tensorfold.families.qwen4_exp.rope import RopeParameters


def positive(value, name):
    if type(value) is not int or not 1 <= value <= 2**31 - 1:
        raise ValueError(f"DFlash {name} must be a positive signed32 integer")
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
    rope: RopeParameters

    @classmethod
    def from_dict(cls, raw):
        if raw.get("architectures") != ["DFlashDraftModel"] or raw.get("model_type") != "qwen3":
            raise ValueError("this adapter requires Qwen3 DFlash v1, not native MTP or DFlash2")
        if any(raw.get(key) not in (None, "bfloat16") for key in ("dtype", "torch_dtype")):
            raise ValueError("DFlash declared execution dtype must be bfloat16")
        if raw.get("quantization_config") not in (None, {}):
            raise ValueError("DFlash checkpoint quantization declarations are unsupported; supply BF16 draft weights")
        names = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads", "head_dim",
                 "num_hidden_layers", "num_target_layers", "vocab_size", "block_size")
        d, width, heads, kv, hd, layers, target, vocab, block = [positive(raw.get(k), k) for k in names]
        if heads % kv or hd < 16 or hd & (hd - 1) or any(n % 64 for n in (d, width, heads * hd, kv * hd)):
            raise ValueError("DFlash grouped attention and linear dimensions must satisfy native tile geometry")
        if raw.get("attention_bias", False) is not False or raw.get("hidden_act", "silu") != "silu":
            raise ValueError("DFlash requires unbiased projections and SiLU")
        if raw.get("sliding_window") is not None or raw.get("use_sliding_window", False) is not False:
            raise ValueError("this DFlash adapter uses the declared full-attention context")
        kinds = raw.get("layer_types")
        if kinds is not None and (not isinstance(kinds, list) or len(kinds) != layers
                                  or any(k != "full_attention" for k in kinds)):
            raise ValueError("DFlash layers must all use full attention")
        dc = raw.get("dflash_config")
        if not isinstance(dc, dict):
            raise ValueError("DFlash needs dflash_config")  # noqa: TRY004 - external checkpoint JSON schema
        taps, mask = dc.get("target_layer_ids"), dc.get("mask_token_id")
        if (not isinstance(taps, list) or not taps or any(type(i) is not int or not 0 <= i < target for i in taps)
                or taps != sorted(set(taps))):
            raise ValueError("DFlash target taps must be unique increasing zero-based layer outputs")
        if type(mask) is not int or not 0 <= mask < vocab:
            raise ValueError("DFlash mask token must lie in the shared target vocabulary")
        eps = raw.get("rms_norm_eps")
        if isinstance(eps, bool) or not isinstance(eps, (int, float)) or not math.isfinite(eps) or eps <= 0:
            raise ValueError("DFlash RMS epsilon must be finite and positive")
        # Reuse the owned YaRN interpolation/amplitude implementation, with
        # Qwen3's full-dimensional text RoPE instead of Flash Next's partial default.
        params = raw.get("rope_parameters")
        if params is None:
            params = raw.get("rope_scaling")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            raise ValueError("DFlash rotary parameters must be an object")  # noqa: TRY004 - external JSON schema
        params = dict(params)
        partials = (raw.get("partial_rotary_factor", 1), params.get("partial_rotary_factor", 1))
        if any(isinstance(value, bool) or value != 1 for value in partials):
            raise ValueError("DFlash Qwen3 uses full-head rotary dimensions")
        if "rope_theta" in params and "rope_theta" in raw and params["rope_theta"] != raw["rope_theta"]:
            raise ValueError("DFlash rotary theta declarations disagree")
        params.update(rope_theta=params.get("rope_theta", raw.get("rope_theta", 10000)), partial_rotary_factor=1)
        rope = RopeParameters.from_config({**raw, "rope_parameters": params})
        return cls(d, width, heads, kv, hd, layers, target, vocab, tuple(taps), mask, block, float(eps), rope)

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
        raise ValueError("DFlash checkpoint tensor count differs from its declared architecture")
    expected = cfg.shapes()
    if set(found) != set(expected):
        raise ValueError("DFlash checkpoint tensor inventory differs from its declared architecture")
    if any(tuple(found[k]["shape"]) != shape or found[k]["dtype"] != "BF16" for k, shape in expected.items()):
        raise ValueError("DFlash weights require declared shapes and BF16 storage")
    return cfg, found


def draft_weights(directory, *, bits=16):
    """One projection is read/converted at a time; target embedding is borrowed, the head too unless FP8 (8) copies
    it privately (counted here: proposals only, the target keeps its BF16 head)."""
    from tensorfold.cuda.capacity import Weights
    from tensorfold.cuda.requant import resident_bytes, staging_bytes

    if type(bits) is not int or bits not in (4, 8, 16):
        raise ValueError("DFlash draft bits must be 16, or explicit proposal-only FP8 (8) or affine4 (4)")
    cfg, _ = checked_headers(directory)
    resident, staging = cfg.head_dim // 2 * 4, 0  # retained FP32 inverse frequencies
    for shape in cfg.shapes().values():
        count = math.prod(shape)
        if bits == 4 and len(shape) == 2:
            rows, columns = shape
            rows = -(-rows // 128) * 128  # shared Q4 tile layout pads output rows to128
            resident += rows * columns // 2 + rows * (columns // 64) * 4
        elif bits == 8 and len(shape) == 2:
            resident += resident_bytes("fp8", *shape)
            staging = max(staging, staging_bytes("fp8", *shape))
        else:
            resident += count * 2
        # affine4 conversion owns FP32 reductions/codes alongside one BF16 input.
        staging = max(staging, count * (24 if bits == 4 and len(shape) == 2 else 2))
    if bits == 8:
        resident += resident_bytes("fp8", cfg.vocab, cfg.hidden)
        staging = max(staging, staging_bytes("fp8", cfg.vocab, cfg.hidden))
    return Weights(resident, staging, 0)


def draft_geometry(raw, *, streams=1, kept=0, block=None, prompt_rows=4096):
    """Bound live/retained contexts, replacement overlap and bounded proposal/prompt scratch."""
    from tensorfold.cuda.capacity import Geometry

    cfg = DraftConfig.from_dict(raw)
    streams, prompt_rows = positive(streams, "streams"), positive(prompt_rows, "prompt rows")
    if type(kept) is not int or kept < 0:
        raise ValueError("DFlash retained context count must be nonnegative")
    block = cfg.block if block is None else positive(block, "block")
    rows = max(prompt_rows, streams * block)
    # Captured target taps, FC/norm intermediates, QKV/attention/MLP live tensors,
    # full target-head proposal logits and CPU readback's device source.
    scratch = rows * (len(cfg.tap_layers) * cfg.hidden * 2 + cfg.hidden * 24 +
                      (cfg.heads + 2 * cfg.kv_heads) * cfg.head_dim * 8 + cfg.width * 16)
    scratch += streams * block * cfg.vocab * 8
    per_token = 2 * cfg.layers * cfg.kv_heads * cfg.head_dim * 2
    return Geometry(lambda slots: scratch + (2 * streams + kept + 1) * (slots + block) * per_token, block)

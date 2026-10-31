"""Metadata-only boundary for Speculators Gemma E2B DFlash2 checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import math

from tensorfold.checkpoint_metadata import read_json
from tensorfold.families.qwen4_exp.rope import FLOAT32_MAX, FLOAT32_MIN_POSITIVE
from .dflash_metadata import positive


@dataclass(frozen=True)
class E2BDraftConfig:
    hidden: int
    width: int
    heads: int
    kv_heads: int
    head_dim: int
    layers: int
    vocab: int
    tap_hidden_indices: tuple[int, ...]
    mask_id: int
    block: int
    window: int
    max_position: int
    group_size: int
    selector_rank: int
    selector_top_k: int
    eps: float
    theta: float
    layer_types: tuple[str, ...]

    @property
    def tap_layers(self) -> tuple[int, ...]:
        """Decoder indices: Speculators indices include the embedding state at zero."""
        return tuple(i - 1 for i in self.tap_hidden_indices)

    @classmethod
    def from_dict(cls, raw: dict) -> E2BDraftConfig:
        if not isinstance(raw, dict) or raw.get("architectures") != ["DFlash2DraftModel"]:
            raise ValueError("Gemma E2B companion requires DFlash2DraftModel metadata")
        if raw.get("speculators_model_type") != "dflash2":
            raise ValueError("Gemma E2B companion requires Speculators dflash2 metadata")
        if (raw.get("dtype", raw.get("torch_dtype")) != "bfloat16"
                or any(raw.get(k) not in (None, "bfloat16") for k in ("dtype", "torch_dtype"))):
            raise ValueError("Gemma E2B DFlash2 requires original BF16 weights")
        if raw.get("quantization_config") not in (None, {}) or raw.get("quantization") not in (None, {}):
            raise ValueError("Gemma E2B DFlash2 requires unquantized proposal weights")
        if raw.get("sample_from_anchor") is not False or raw.get("sliding_window_non_causal") is not True:
            raise ValueError("Gemma E2B DFlash2 requires an anchor followed by noncausal mask rows")
        if raw.get("conv_kernel_size") != 2:
            raise ValueError("Gemma E2B DFlash2 implements the declared two-tap convolution")
        text = raw.get("transformer_layer_config")
        if not isinstance(text, dict) or text.get("model_type") != "qwen3":
            raise ValueError("Gemma E2B DFlash2 needs its Qwen3 transformer declaration")
        fields = ("hidden_size", "intermediate_size", "num_attention_heads", "num_key_value_heads",
                  "head_dim", "num_hidden_layers", "vocab_size", "sliding_window", "max_position_embeddings")
        hidden, width, heads, kv, dim, layers, vocab, window, maximum = [
            positive(text.get(k), k) for k in fields]
        if heads % kv or hidden % positive(raw.get("conv_group_size"), "conv_group_size"):
            raise ValueError("Gemma E2B DFlash2 head/group dimensions do not divide")
        if heads * dim != hidden:
            raise ValueError("Gemma E2B DFlash2 query heads must span its hidden width")
        if (text.get("attention_bias", False) is not False or text.get("attention_dropout", 0.0) != 0.0
                or text.get("hidden_act") != "silu" or text.get("use_sliding_window") is not True):
            raise ValueError("Gemma E2B DFlash2 requires bias/dropout-free SiLU sliding attention")
        kinds = text.get("layer_types")
        if (not isinstance(kinds, list) or len(kinds) != layers
                or any(k not in ("sliding_attention", "full_attention") for k in kinds)):
            raise ValueError("Gemma E2B DFlash2 needs one valid attention kind per draft layer")
        taps = raw.get("aux_hidden_state_layer_ids")
        if (not isinstance(taps, list) or not taps or any(type(i) is not int or i < 1 for i in taps)
                or taps != sorted(set(taps))):
            raise ValueError("Gemma E2B DFlash2 needs unique increasing positive hidden-state indices")
        block = positive(raw.get("block_size"), "block_size")
        if not 2 <= block <= window:
            raise ValueError("Gemma E2B DFlash2 block must fit its sliding window")
        if positive(raw.get("draft_vocab_size"), "draft_vocab_size") != vocab:
            raise ValueError("Gemma E2B DFlash2 requires the full verifier vocabulary")
        mask = raw.get("mask_token_id")
        if type(mask) is not int or not 0 <= mask < vocab:
            raise ValueError("Gemma E2B DFlash2 mask token is outside its vocabulary")
        rank, top_k = [positive(raw.get(k), k) for k in ("selector_rank", "selector_top_k")]
        if top_k > vocab:
            raise ValueError("Gemma E2B DFlash2 selector exceeds the full vocabulary")
        rope = text.get("rope_parameters")
        if not isinstance(rope, dict) or rope.get("rope_type") != "default":
            raise ValueError("Gemma E2B DFlash2 requires its unscaled full-head rotary contract")
        if any(isinstance(v, bool) or v != 1 for v in (
                text.get("partial_rotary_factor", 1), rope.get("partial_rotary_factor", 1))):
            raise ValueError("Gemma E2B DFlash2 requires full-head rotary dimensions")
        eps, theta = text.get("rms_norm_eps"), rope.get("rope_theta")
        if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
               or not FLOAT32_MIN_POSITIVE <= v <= FLOAT32_MAX
               for v in (eps, theta)):
            raise ValueError("Gemma E2B DFlash2 epsilon/theta must be positive finite numbers")
        if dim % 2:
            raise ValueError("Gemma E2B DFlash2 rotary heads require an even dimension")
        if theta < 1:
            raise ValueError("Gemma E2B DFlash2 theta must keep FP32 inverse frequencies bounded")
        declared = raw.get("speculators_config")
        if not isinstance(declared, dict) or declared.get("algorithm") != "dflash2":
            raise ValueError("Gemma E2B DFlash2 algorithm declaration differs")
        return cls(hidden, width, heads, kv, dim, layers, vocab, tuple(taps), mask, block, window,
                   maximum, raw["conv_group_size"], rank, top_k, float(eps), float(theta), tuple(kinds))

    @classmethod
    def read(cls, directory):
        return cls.from_dict(read_json(directory, "config.json", required=True))

    def shapes(self) -> dict[str, tuple[int, ...]]:
        out = {"fc.weight": (self.hidden, len(self.tap_layers) * self.hidden),
               "hidden_norm.weight": (self.hidden,), "norm.weight": (self.hidden,),
               "candidate_selector.hidden_projection.weight": (self.selector_rank, self.hidden),
               "candidate_selector.predecessor_codebook": (self.vocab, self.selector_rank),
               "candidate_selector.successor_codebook": (self.vocab, self.selector_rank)}
        for i in range(self.layers):
            base = f"layers.{i}."
            out.update({base + k: v for k, v in {
                "input_layernorm.weight": (self.hidden,), "post_attention_layernorm.weight": (self.hidden,),
                "self_attn.q_norm.weight": (self.head_dim,), "self_attn.k_norm.weight": (self.head_dim,),
                "self_attn.q_proj.weight": (self.heads * self.head_dim, self.hidden),
                "self_attn.k_proj.weight": (self.kv_heads * self.head_dim, self.hidden),
                "self_attn.v_proj.weight": (self.kv_heads * self.head_dim, self.hidden),
                "self_attn.o_proj.weight": (self.hidden, self.heads * self.head_dim),
                "mlp.gate_proj.weight": (self.width, self.hidden), "mlp.up_proj.weight": (self.width, self.hidden),
                "mlp.down_proj.weight": (self.hidden, self.width),
                "attention_conv.base_kernel": (2, 2, self.hidden),
                "mlp_conv.base_kernel": (2, 2, self.hidden),
                "attention_conv.kernel_projection.weight": (4 * (self.hidden // self.group_size), self.hidden),
                "mlp_conv.kernel_projection.weight": (4 * (self.hidden // self.group_size), self.hidden),
            }.items()})
        return out

    @property
    def parameter_bytes(self) -> int:
        return 2 * sum(math.prod(s) for s in self.shapes().values())

    def projection_shapes(self) -> dict[str, tuple[int, int]]:
        """Only ordinary draft projections; convolution and selector stay BF16."""
        names = {"fc.weight"}
        for i in range(self.layers):
            names.update(f"layers.{i}.self_attn.{p}_proj.weight" for p in ("q", "k", "v", "o"))
            names.update(f"layers.{i}.mlp.{p}_proj.weight" for p in ("gate", "up", "down"))
        shapes = self.shapes()
        return {name: shapes[name] for name in sorted(names)}

    def memory_bytes(self, bits: int = 16) -> tuple[int, int]:
        """Private resident/load-peak bytes, including the optional private head.

        Borrowed authoritative BF16 target embedding/head are charged by the
        target. Existing requant helpers own padded codes/scales and staging.
        """
        if type(bits) is not int or bits not in (8, 16):
            raise ValueError("Gemma E2B proposal precision is BF16(16) or private FP8(8)")
        raw = self.parameter_bytes
        if bits == 16:
            return raw, raw
        from tensorfold.cuda.requant import resident_bytes, staging_bytes

        projections = list(self.projection_shapes().values())
        if any(k % 64 for _, k in projections) or self.hidden % 64:
            raise ValueError("Gemma E2B FP8 proposals require projection inputs divisible by64")
        projected = sum(resident_bytes("fp8", n, k) for n, k in projections)
        remaining = raw - sum(n * k * 2 for n, k in projections)
        private_head = resident_bytes("fp8", self.vocab, self.hidden)
        resident = remaining + projected + private_head
        peak = max(raw + projected + max(staging_bytes("fp8", n, k) for n, k in projections),
                   resident + staging_bytes("fp8", self.vocab, self.hidden))
        return resident, peak

    def state_bytes(self, context_limit: int) -> int:
        context_limit = positive(context_limit, "context_limit")
        if context_limit > self.max_position:
            raise ValueError("Gemma E2B DFlash2 context exceeds its declared position limit")
        # Full caches hold a disposable mask block after the committed prefix.
        # Sliding caches also own chronological gather buffers for SDPA.
        slots = sum((min(self.window, context_limit) * 2 + self.block)
                    if kind == "sliding_attention" else context_limit + self.block
                    for kind in self.layer_types)
        return slots * self.kv_heads * self.head_dim * 2 * 2


def checked_headers(directory):
    from tensorfold.cuda.capacity import headers

    cfg = E2BDraftConfig.read(directory)
    found, expected = headers(directory), cfg.shapes()
    if set(found) != set(expected):
        raise ValueError("Gemma E2B DFlash2 tensor inventory differs from its declared architecture")
    if any(tuple(found[k]["shape"]) != shape or found[k]["dtype"] != "BF16" for k, shape in expected.items()):
        raise ValueError("Gemma E2B DFlash2 tensors require declared BF16 shapes")
    return cfg, found

"""CPU-only exact profiles for original BF16 Gemma 4 E2B/E4B checkpoints."""
from __future__ import annotations

from dataclasses import dataclass
import math


def is_e2b(raw: dict) -> bool:
    text = raw.get("text_config") or raw
    return (isinstance(text, dict) and text.get("enable_moe_block") is False
            and text.get("hidden_size") == 1536 and text.get("num_hidden_layers") == 35)


def is_e4b(raw: dict) -> bool:
    text = raw.get("text_config") or raw
    return (isinstance(text, dict) and text.get("enable_moe_block") is False
            and text.get("hidden_size") == 2560 and text.get("num_hidden_layers") == 42)


def is_dense_gemma(raw: dict) -> bool:
    return is_e2b(raw) or is_e4b(raw)


def _positive(raw: dict, key: str) -> int:
    value = raw.get(key)
    if type(value) is not int or not 0 < value <= 2**31 - 1:
        raise ValueError(f"Gemma E2B {key} must be a positive signed32 integer")
    return value


@dataclass(frozen=True)
class E2BConfig:
    hidden: int
    layers: int
    vocab: int
    heads: int
    kv_heads: int
    head_dim: int
    global_head_dim: int
    intermediate: int
    per_layer_input: int
    shared_layers: int
    window: int
    max_position: int
    softcap: float
    layer_types: tuple[str, ...]
    profile: str = "E2B"

    @classmethod
    def from_dict(cls, raw: dict) -> "E2BConfig":
        if not isinstance(raw, dict) or not is_dense_gemma(raw):
            raise ValueError("Gemma CUDA requires the complete dense E2B or E4B architecture")
        profile = "E2B" if is_e2b(raw) else "E4B"
        expected_kv, expected_mlp, expected_shared, cadence, double_wide = (
            (1, 6144, 20, 5, True) if profile == "E2B" else (2, 10240, 18, 6, False))
        if raw.get("model_type") != "gemma4" or raw.get("architectures") != ["Gemma4ForConditionalGeneration"]:
            raise ValueError("Gemma E2B requires the complete conditional-generation checkpoint")
        if not isinstance(raw.get("vision_config"), dict) or not isinstance(raw.get("audio_config"), dict):
            raise ValueError("Gemma E2B requires its declared image/video and audio tower configurations")
        text = raw.get("text_config") or raw
        if text.get("model_type") != "gemma4_text":
            raise ValueError("Gemma E2B requires gemma4_text decoder metadata")
        if (raw["vision_config"].get("model_type") != "gemma4_vision"
                or raw["audio_config"].get("model_type") != "gemma4_audio"):
            raise ValueError("Gemma E2B requires its native vision and audio tower types")
        for block in (raw, text):
            if block.get("quantization_config") not in (None, {}) or block.get("quantization") not in (None, {}):
                raise ValueError("Gemma E2B requires its original unquantized BF16 checkpoint")
            if block.get("dtype", block.get("torch_dtype", "bfloat16")) != "bfloat16":
                raise ValueError("Gemma E2B execution requires the declared BF16 checkpoint")
        keys = ("hidden_size", "num_hidden_layers", "vocab_size", "num_attention_heads",
                "num_key_value_heads", "head_dim", "global_head_dim", "intermediate_size",
                "hidden_size_per_layer_input", "num_kv_shared_layers", "sliding_window",
                "max_position_embeddings")
        values = [_positive(text, key) for key in keys]
        hidden, layers, vocab, heads, kv, dim, global_dim, intermediate, ple, shared, window, native = values
        ple_vocab = text.get("vocab_size_per_layer_input")
        if vocab != 262144 or native != 131072 or type(ple_vocab) is not int or ple_vocab != 262144:
            raise ValueError("Gemma E2B requires its original vocabulary, PLE table and native context geometry")
        global_kv = text.get("num_global_key_value_heads")
        if global_kv is not None and (type(global_kv) is not int or global_kv != expected_kv):
            raise ValueError(f"Gemma {profile} global attention requires {expected_kv} KV heads")
        media_ids = [raw.get(k) for k in ("image_token_id", "video_token_id", "audio_token_id")]
        if (any(type(t) is not int or not 0 <= t < vocab for t in media_ids)
                or len(set(media_ids)) != len(media_ids)):
            raise ValueError("Gemma E2B media token IDs must be distinct integers inside the target vocabulary")
        if ((heads, kv, dim, global_dim, intermediate, ple, shared, window) !=
                (8, expected_kv, 256, 512, expected_mlp, 256, expected_shared, 512)):
            raise ValueError("Gemma E2B attention, shared-KV, PLE or MLP dimensions differ from the supported architecture")
        if text.get("use_double_wide_mlp") is not double_wide or text.get("attention_k_eq_v") is not False:
            raise ValueError(f"Gemma {profile} MLP width or independent key/value projection contract differs")
        if text.get("tie_word_embeddings", raw.get("tie_word_embeddings")) is not True:
            raise ValueError("Gemma E2B requires the tied embedding/head")
        if text.get("attention_bias", False) is not False or text.get("attention_dropout", 0.0) != 0.0:
            raise ValueError("Gemma E2B requires bias-free, dropout-free inference attention")
        if text.get("use_bidirectional_attention") not in (None, False):
            raise ValueError("Gemma E2B uses causal text and media attention")
        if text.get("hidden_activation") != "gelu_pytorch_tanh":
            raise ValueError("Gemma E2B requires its declared tanh GeGLU")
        kinds = tuple(text.get("layer_types") or ())
        expected = tuple("full_attention" if (i + 1) % cadence == 0 else "sliding_attention" for i in range(layers))
        if kinds != expected:
            raise ValueError(f"Gemma {profile} requires its declared {cadence}-layer sliding/full attention cadence")
        softcap = text.get("final_logit_softcapping")
        if type(softcap) not in (int, float) or not math.isfinite(softcap) or softcap != 30.0:
            raise ValueError("Gemma E2B requires the original final logit softcap30")
        return cls(hidden, layers, vocab, heads, kv, dim, global_dim, intermediate, ple, shared,
                   window, native, float(softcap), kinds, profile)

    @property
    def tap_layers(self) -> tuple[int, ...]:
        # Only E2B has an admitted companion with an authoritative tap contract.
        return (2, 12, 22, 32) if self.profile == "E2B" else ()

    @property
    def donor_layers(self) -> tuple[str, ...]:
        return self.layer_types[:self.layers - self.shared_layers]

    def cache_bytes(self, length: int, rows: int) -> int:
        """Actual donor KV owners; recording retains one uncommitted forward."""
        return sum((length if kind == "full_attention" else min(length, self.window - 1) + rows)
                   * self.kv_heads * (self.global_head_dim if kind == "full_attention" else self.head_dim) * 4
                   for kind in self.donor_layers)

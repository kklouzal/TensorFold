"""CPU-only native geometry and original BF16 storage checks for Gemma dense profiles."""
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


def is_31b(raw: dict) -> bool:
    text = raw.get("text_config") or raw
    return (isinstance(text, dict) and text.get("enable_moe_block") is False
            and text.get("hidden_size") == 5376 and text.get("num_hidden_layers") == 60)


def is_dense_gemma(raw: dict) -> bool:
    return is_e2b(raw) or is_e4b(raw) or is_31b(raw)


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
    global_kv_heads: int | None = None
    attention_k_eq_v: bool = False
    has_audio: bool = True
    bidirectional_vision: bool = False

    @classmethod
    def from_dict(cls, raw: dict) -> "E2BConfig":
        """Original BF16 checkpoint boundary; quantized storage is separate."""
        result = cls.from_architecture(raw)
        for block in (raw, raw.get("text_config") or raw):
            if block.get("quantization_config") not in (None, {}) or block.get("quantization") not in (None, {}):
                raise ValueError("Gemma E2B requires its original unquantized BF16 checkpoint")
        return result

    @classmethod
    def from_checkpoint(cls, raw: dict) -> "E2BConfig":
        """Storage-aware dispatch; payload schemas are validated by the loader.

        ModelOpt NVFP4/AWQ dense profiles retain declared checkpoint arithmetic.
        Architecture validation alone never authorizes arbitrary quantization.
        """
        from tensorfold.cuda.nvfp4.format import config_block
        block = config_block(raw)
        if block is None:
            return cls.from_dict(raw)
        result = cls.from_architecture(raw)
        if (result.profile not in ("E4B", "31B") or block.get("quant_method") != "modelopt"
                or block.get("kv_cache_scheme") is not None or block.get("transform_config")
                or block.get("sparsity_config") or block.get("quant_algo") not in ("NVFP4", "NVFP4_AWQ")):
            raise ValueError("dense packed inference requires declared ModelOpt NVFP4 with BF16 KV")
        return result

    @classmethod
    def from_architecture(cls, raw: dict) -> "E2BConfig":
        """Native geometry/compute contract, without claiming any storage format.

        Quantized loaders must separately validate every declared tensor schema;
        this method observes the original config unchanged, including quantizers.
        """
        if not isinstance(raw, dict) or not is_dense_gemma(raw):
            raise ValueError("Gemma CUDA requires the complete dense E2B or E4B architecture")
        if is_31b(raw):
            return cls._from_31b(raw)
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

    @classmethod
    def _from_31b(cls, raw: dict) -> "E2BConfig":
        """Exact native AEON/Google31B architecture; no invented PLE/shared KV."""
        text = raw.get('text_config')
        if (raw.get('model_type') != 'gemma4' or raw.get('architectures') != ['Gemma4ForConditionalGeneration']
                or not isinstance(text, dict) or text.get('model_type') != 'gemma4_text'
                or not isinstance(raw.get('vision_config'), dict)
                or raw['vision_config'].get('model_type') != 'gemma4_vision' or raw.get('audio_config') is not None):
            raise ValueError('Gemma31B requires its native vision-only conditional-generation architecture')
        for block in (raw, text):
            if block.get('dtype', block.get('torch_dtype', 'bfloat16')) != 'bfloat16':
                raise ValueError('Gemma31B requires BF16 native activation/norm arithmetic')
        dimensions = {'hidden_size':5376, 'num_hidden_layers':60, 'vocab_size':262144,
                      'num_attention_heads':32, 'num_key_value_heads':16, 'num_global_key_value_heads':4,
                      'head_dim':256, 'global_head_dim':512, 'intermediate_size':21504,
                      'hidden_size_per_layer_input':0, 'num_kv_shared_layers':0,
                      'sliding_window':1024, 'max_position_embeddings':262144,
                      'vocab_size_per_layer_input':262144}
        if any(type(text.get(key)) is not int or text[key] != value for key,value in dimensions.items()):
            raise ValueError('Gemma31B native text geometry differs from the admitted profile')
        if (text.get('enable_moe_block') is not False or text.get('attention_k_eq_v') is not True
                or text.get('use_double_wide_mlp') is not False
                or text.get('tie_word_embeddings', raw.get('tie_word_embeddings')) is not True
                or text.get('attention_bias', False) is not False
                or type(text.get('attention_dropout', 0.0)) not in (int,float)
                or text.get('attention_dropout',0.0) != 0
                or text.get('hidden_activation') != 'gelu_pytorch_tanh'
                or text.get('use_bidirectional_attention') != 'vision'):
            raise ValueError('Gemma31B K=V/MLP/tied-head/bidirectional vision contract differs')
        media = [raw.get(key) for key in ('image_token_id','video_token_id','audio_token_id')]
        if any(type(value) is not int or not 0 <= value < 262144 for value in media) or len(set(media)) != 3:
            raise ValueError('Gemma31B requires native distinct in-vocabulary media token IDs')
        kinds = tuple(text.get('layer_types') or ())
        if kinds != tuple('full_attention' if (i+1)%6==0 else 'sliding_attention' for i in range(60)):
            raise ValueError('Gemma31B requires its native six-layer attention cadence')
        softcap = text.get('final_logit_softcapping')
        if type(softcap) not in (int,float) or softcap != 30.0:
            raise ValueError('Gemma31B requires native final softcap30')
        return cls(5376,60,262144,32,16,256,512,21504,0,0,1024,262144,30.0,kinds,
                   '31B',global_kv_heads=4,attention_k_eq_v=True,has_audio=False,bidirectional_vision=True)

    def kv_heads_for(self, kind):
        return (self.global_kv_heads or self.kv_heads) if kind == 'full_attention' else self.kv_heads

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
                   * self.kv_heads_for(kind) * (self.global_head_dim if kind == "full_attention" else self.head_dim) * 4
                   for kind in self.donor_layers)

    def prompt_kv_repeat_bytes(self, length: int, rows: int) -> int:
        """Largest native HF SDPA repeated K+V owner, one layer at a time.

        Native head512 SDPA materializes grouped heads. Charge the complete
        repeated tensors, even if a selected provider may avoid them later.
        """
        return max((length if kind == 'full_attention' else min(length,self.window-1+rows))
                   * self.heads * (self.global_head_dim if kind == 'full_attention' else self.head_dim) * 4
                   for kind in self.donor_layers)

    def prompt_math_accumulator_bytes(self, length: int, rows: int) -> int:
        """Pinned Torch MATH-SDPA FP32 owners, beyond score/probability planes.

        ef46daf attention.cpp898–966 promotes Q/K/V, scales query, and may
        expand GQA AFTER promotion. Charge original grouped FP32 KV too even
        when default HF has already repeated BF16 heads. For R>=D the scaled
        key temporary fits the existing second FP32 score plane; smaller R
        explicitly charges the excess. Also include causal/additive masks and
        the safe-softmax row predicate. This bound does not select a backend.
        """
        peaks = []
        for kind in self.donor_layers:
            held = length if kind == 'full_attention' else min(length,self.window-1+rows)
            dim = self.global_head_dim if kind == 'full_attention' else self.head_dim
            query = 2*rows*self.heads*dim*4  # FP32 query_acc + scaled query
            expanded = 2*held*self.heads*dim*4  # FP32 K/V (or GQA expansion)
            original = 2*held*self.kv_heads_for(kind)*dim*4  # GQA input owners
            scaled_key_extra = max(0,held*self.heads*(dim-rows)*4)
            masks = 5*rows*held+self.heads*rows+4
            peaks.append(query+expanded+original+scaled_key_extra+masks)
        return max(peaks)

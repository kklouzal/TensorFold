"""Metadata boundary for the original BF16 Gemma4 E4B official assistant."""
from __future__ import annotations

from dataclasses import dataclass

from tensorfold.checkpoint_metadata import read_json
from .dflash_metadata import positive


@dataclass(frozen=True)
class AssistantConfig:
    hidden: int = 256
    backbone_hidden: int = 2560
    vocab: int = 262144
    max_position: int = 131072
    block: int = 8

    @classmethod
    def from_dict(cls, raw):
        if not isinstance(raw, dict):
            raise ValueError("Gemma assistant config must be an object")
        text = raw.get("text_config")
        if not isinstance(text, dict):
            raise ValueError("Gemma assistant requires its text configuration")
        root_fields = {"architectures": ["Gemma4AssistantForCausalLM"], "model_type": "gemma4_assistant",
                       "backbone_hidden_size": 2560, "use_ordered_embeddings": True,
                       "num_centroids": 2048, "centroid_intermediate_top_k": 32,
                       "tie_word_embeddings": True}
        text_fields = {"model_type": "gemma4_text", "hidden_size": 256, "num_hidden_layers": 4,
                       "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 256,
                       "global_head_dim": 512, "intermediate_size": 2048, "num_kv_shared_layers": 4,
                       "sliding_window": 512, "max_position_embeddings": 131072, "vocab_size": 262144,
                       "hidden_size_per_layer_input": 0, "vocab_size_per_layer_input": 0,
                       "hidden_activation": "gelu_pytorch_tanh", "final_logit_softcapping": None,
                       "enable_moe_block": False, "use_double_wide_mlp": False, "attention_k_eq_v": False,
                       "use_bidirectional_attention": None,
                       "attention_bias": False, "rms_norm_eps": 1e-6,
                       "layer_types": ["sliding_attention"] * 3 + ["full_attention"]}
        for data, expected in ((raw, root_fields), (text, text_fields)):
            for key, value in expected.items():
                actual = data.get(key)
                if actual != value or (isinstance(value, (int, bool)) and type(actual) is not type(value)):
                    raise ValueError(f"Gemma E4B assistant {key} differs from the native checkpoint contract")
            if data.get("quantization_config") not in (None, {}) or data.get("quantization") not in (None, {}):
                raise ValueError("Gemma E4B assistant requires original BF16 weights")
            if data.get("dtype", data.get("torch_dtype", "bfloat16")) != "bfloat16":
                raise ValueError("Gemma E4B assistant requires BF16 execution")
        rope = {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                                   "partial_rotary_factor": 0.25}}
        if text.get("rope_parameters") != rope:
            raise ValueError("Gemma E4B assistant requires native local/global RoPE")
        return cls()

    @classmethod
    def read(cls, directory):
        return cls.from_dict(read_json(directory, "config.json", required=True))

    def shapes(self):
        shapes = {"masked_embedding.token_ordering": (self.vocab,),
                  "masked_embedding.centroids.weight": (2048, self.hidden),
                  "model.embed_tokens.weight": (self.vocab, self.hidden),
                  "model.norm.weight": (self.hidden,),
                  "pre_projection.weight": (self.hidden, 2 * self.backbone_hidden),
                  "post_projection.weight": (self.backbone_hidden, self.hidden)}
        for i in range(4):
            prefix = f"model.layers.{i}."
            dim = 256 if i < 3 else 512
            for name in ("input_layernorm", "post_attention_layernorm", "post_feedforward_layernorm",
                         "pre_feedforward_layernorm"):
                shapes[prefix + name + ".weight"] = (self.hidden,)
            shapes[prefix + "layer_scalar"] = (1,)
            shapes[prefix + "self_attn.q_norm.weight"] = (dim,)
            shapes[prefix + "self_attn.q_proj.weight"] = (4 * dim, self.hidden)
            shapes[prefix + "self_attn.o_proj.weight"] = (self.hidden, 4 * dim)
            for name in ("gate_proj", "up_proj"):
                shapes[prefix + "mlp." + name + ".weight"] = (2048, self.hidden)
            shapes[prefix + "mlp.down_proj.weight"] = (self.hidden, 2048)
        return shapes

    @property
    def parameter_bytes(self):
        import math

        return sum(math.prod(shape) * (8 if name == "masked_embedding.token_ordering" else 2)
                   for name, shape in self.shapes().items())

    def memory_bytes(self, bits=16):
        if type(bits) is not int or bits != 16:
            raise ValueError("Gemma E4B native assistant supports original BF16(16) only")
        return self.parameter_bytes, 2 * self.parameter_bytes

    def state_bytes(self, context_limit):
        if positive(context_limit, "context_limit") > self.max_position:
            raise ValueError("Gemma assistant context exceeds native131072 positions")
        # All donor KV and accepted hidden ownership belongs to the target state.
        return 0


def checked_headers(directory):
    from tensorfold.cuda.capacity import headers

    cfg = AssistantConfig.read(directory)
    actual, expected = headers(directory), cfg.shapes()
    if set(actual) != set(expected):
        raise ValueError("Gemma assistant checkpoint tensor inventory differs")
    for name, shape in expected.items():
        dtype = "I64" if name == "masked_embedding.token_ordering" else "BF16"
        if tuple(actual[name]["shape"]) != shape or actual[name]["dtype"] != dtype:
            raise ValueError(f"Gemma assistant tensor geometry/dtype differs: {name}")
    return cfg, actual

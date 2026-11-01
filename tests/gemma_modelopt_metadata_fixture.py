"""Tiny CPU-only ModelOpt expert/native-header fixture; no checkpoint payloads."""

from tensorfold.checkpoint_quantization import effective_config
from tensorfold.families.gemma4.cuda.weights import Config
from tensorfold.families.gemma4.cuda.nvfp4_weights import ct_tensor_contract


def fixture():
    raw = {
        "model_type": "gemma4_text",
        "hidden_size": 64,
        "num_hidden_layers": 1,
        "vocab_size": 128,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "num_global_key_value_heads": 1,
        "head_dim": 64,
        "global_head_dim": 64,
        "intermediate_size": 128,
        "moe_intermediate_size": 64,
        "num_experts": 32,
        "top_k_experts": 2,
        "sliding_window": 8,
        "max_position_embeddings": 64,
        "rms_norm_eps": 1e-6,
        "final_logit_softcapping": 30.0,
        "enable_moe_block": True,
        "attention_k_eq_v": True,
        "tie_word_embeddings": True,
        "hidden_activation": "gelu_pytorch_tanh",
        "layer_types": ["sliding_attention"],
        "rope_parameters": {
            "sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0, "partial_rotary_factor": 0.25},
        },
        "quantization_config": {
            "quant_method": "modelopt",
            "quant_algo": "NVFP4",
            "ignore": [
                "lm_head",
                "model.language_model.layers.*.self_attn*",
                "model.language_model.layers.*.mlp*",
                "model.language_model.layers.*.router*",
            ],
        },
    }
    raw = effective_config(raw, None, where="tinyModelOpt")
    c = Config.from_dict(raw)
    matrices, plain = ct_tensor_contract(c)
    headers = {name: {"dtype": "BF16", "shape": list(shape)} for name, shape in plain.items()}
    for name, (n, k) in matrices.items():
        if ".experts." not in name:
            headers[name + ".weight"] = {"dtype": "BF16", "shape": [n, k]}
        else:
            for suffix, dtype, shape in [
                ("weight", "U8", [n, k // 2]),
                ("weight_scale", "F8_E4M3", [n, k // 16]),
                ("input_scale", "F32", []),
                ("weight_scale_2", "F32", []),
            ]:
                headers[name + "." + suffix] = {"dtype": dtype, "shape": shape}
    return raw, headers

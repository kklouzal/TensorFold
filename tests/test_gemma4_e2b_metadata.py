"""Actual E2B metadata boundary; no private checkpoint or GPU is needed."""
import copy
import json

import pytest

from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig, is_e2b


def e2b_config():
    return {
        "model_type": "gemma4", "architectures": ["Gemma4ForConditionalGeneration"],
        "dtype": "bfloat16", "tie_word_embeddings": True,
        "image_token_id": 258880, "video_token_id": 258884, "audio_token_id": 258881,
        "vision_config": {"model_type": "gemma4_vision"},
        "audio_config": {"model_type": "gemma4_audio"},
        "text_config": {
            "model_type": "gemma4_text", "enable_moe_block": False, "hidden_size": 1536,
            "num_hidden_layers": 35, "vocab_size": 262144, "num_attention_heads": 8,
            "vocab_size_per_layer_input": 262144,
            "num_key_value_heads": 1, "head_dim": 256, "global_head_dim": 512,
            "intermediate_size": 6144, "hidden_size_per_layer_input": 256,
            "num_kv_shared_layers": 20, "sliding_window": 512,
            "max_position_embeddings": 131072, "use_double_wide_mlp": True,
            "attention_k_eq_v": False, "tie_word_embeddings": True,
            "attention_bias": False, "attention_dropout": 0.0,
            "hidden_activation": "gelu_pytorch_tanh", "use_bidirectional_attention": None,
            "final_logit_softcapping": 30.0,
            "layer_types": ["full_attention" if (i + 1) % 5 == 0 else "sliding_attention" for i in range(35)],
        },
    }


def test_original_e2b_architecture_is_admitted_without_erasing_features():
    raw = e2b_config()
    before = copy.deepcopy(raw)
    cfg = E2BConfig.from_dict(raw)
    assert raw == before
    assert cfg.layers == 35 and cfg.shared_layers == 20 and cfg.per_layer_input == 256
    assert cfg.donor_layers.count("full_attention") == 3
    assert cfg.donor_layers.count("sliding_attention") == 12
    assert cfg.cache_bytes(131072, 8) == 3 * 131072 * 512 * 4 + 12 * (511 + 8) * 256 * 4


def test_family_dispatch_accepts_only_e2b_for_the_new_unquantized_format(tmp_path):
    from tensorfold import families
    from tensorfold.families import gemma4
    raw = e2b_config()
    family = families.Family("gemma4", "Gemma 4", "tensorfold.families.gemma4", True)
    families.require_readable(family, raw, "cuda")
    (tmp_path / "config.json").write_text(json.dumps(raw))
    gemma4.check(tmp_path)
    assert gemma4.cuda_drafter(tmp_path) == "rasyosef/gemma-4-E2B-it-dflash2"
    raw["text_config"]["enable_moe_block"] = True
    with pytest.raises(ValueError, match="Unquantized"):
        families.require_readable(family, raw, "cuda")


def test_dense_e2b_is_not_silently_routed_to_existing_mlx_moe_kernels():
    from tensorfold import families
    family = families.Family("gemma4", "Gemma 4", "tensorfold.families.gemma4", True)
    with pytest.raises(ValueError):
        families.require_readable(family, e2b_config(), "mlx")


@pytest.mark.parametrize("key,value", [
    ("hidden_size_per_layer_input", 0), ("num_kv_shared_layers", 0),
    ("attention_k_eq_v", True), ("use_double_wide_mlp", False),
    ("num_attention_heads", 16), ("num_key_value_heads", True),
    ("attention_dropout", 0.1), ("use_bidirectional_attention", "vision"),
    ("final_logit_softcapping", 0.0),
    ("vocab_size", 2**30), ("vocab_size_per_layer_input", 2**30),
    ("num_global_key_value_heads", 8), ("model_type", "wrong"),
    ("num_global_key_value_heads", True), ("vocab_size_per_layer_input", 262144.0),
])
def test_incompatible_e2b_metadata_is_rejected(key, value):
    raw = e2b_config()
    raw["text_config"][key] = value
    with pytest.raises(ValueError):
        E2BConfig.from_dict(raw)


def test_moe_config_is_not_routed_through_the_dense_adapter():
    raw = e2b_config()
    raw["text_config"]["enable_moe_block"] = True
    assert not is_e2b(raw)


@pytest.mark.parametrize("key", ["vision_config", "audio_config"])
def test_missing_tower_cannot_be_silently_text_only(key):
    raw = e2b_config()
    raw[key] = None
    with pytest.raises(ValueError, match="tower"):
        E2BConfig.from_dict(raw)


def test_quantization_or_execution_dtype_is_not_silently_changed():
    raw = e2b_config()
    raw["quantization_config"] = {"quant_method": "compressed-tensors"}
    with pytest.raises(ValueError, match="unquantized"):
        E2BConfig.from_dict(raw)
    raw = e2b_config()
    raw["dtype"] = "float16"
    with pytest.raises(ValueError, match="BF16"):
        E2BConfig.from_dict(raw)

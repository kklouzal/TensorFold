"""Gemma target metadata checks run without importing Torch or Triton."""

from copy import deepcopy

import pytest

from tensorfold.families.gemma4.cuda.weights import Config, PREFIX, quant_spec, tensor_contract, validate_tensor_metadata


def tiny_config():
    return {
        "model_type": "gemma4", "tie_word_embeddings": True,
        "text_config": {
            "model_type": "gemma4_text", "hidden_size": 64, "num_hidden_layers": 2, "vocab_size": 128,
            "num_attention_heads": 2, "num_key_value_heads": 1, "num_global_key_value_heads": 1,
            "head_dim": 64, "global_head_dim": 128, "intermediate_size": 128, "moe_intermediate_size": 64,
            "num_experts": 4, "top_k_experts": 2, "sliding_window": 8, "max_position_embeddings": 256,
            "rms_norm_eps": 1e-6, "final_logit_softcapping": 30.0, "enable_moe_block": True,
            "attention_k_eq_v": True, "hidden_activation": "gelu_pytorch_tanh",
            "layer_types": ["sliding_attention", "full_attention"], "use_bidirectional_attention": "vision",
            "rope_parameters": {"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                                "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                                                   "partial_rotary_factor": .25}},
        },
        "quantization": {"bits": 4, "group_size": 64, "mode": "affine",
                         PREFIX + "embed_tokens": {"bits": 6, "group_size": 64}},
    }


def inventory(raw):
    c = Config.from_dict(raw)
    matrices, plain = tensor_contract(c)
    where = {name: (None, 0, 0, "BF16", list(shape)) for name, shape in plain.items()}
    for name, shape in matrices.items():
        bits, group = quant_spec(raw, name)
        for suffix, dtype, width in (("weight", "U32", shape[-1] * bits // 32),
                                     ("scales", "BF16", shape[-1] // group),
                                     ("biases", "BF16", shape[-1] // group)):
            where[name + "." + suffix] = (None, 0, 0, dtype, [*shape[:-1], width])
    return c, where


def test_mixed_width_and_four_bit_router_are_native():
    raw = tiny_config()
    c, where = inventory(raw)
    validate_tensor_metadata(c, raw, where)
    assert quant_spec(raw, PREFIX + "embed_tokens") == (6, 64)
    assert quant_spec(raw, PREFIX + "layers.0.router.proj") == (4, 64)
    assert c.attention_geometry(0) == (1, 64)
    assert c.attention_geometry(1) == (1, 128)
    assert PREFIX + "layers.1.self_attn.v_proj.weight" not in where


@pytest.mark.parametrize("field,value", [("enable_moe_block", False), ("attention_k_eq_v", False),
    ("num_kv_shared_layers", 1), ("hidden_size_per_layer_input", 1), ("tie_word_embeddings", False),
    ("hidden_activation", "silu"), ("rms_norm_eps", float("nan")), ("head_dim", 192),
    ("num_hidden_layers", True), ("top_k_experts", 5), ("num_key_value_heads", 3)])
def test_unsupported_metadata_refused_before_runtime(field, value):
    raw = tiny_config()
    raw["text_config"][field] = value
    with pytest.raises(ValueError):
        Config.from_dict(raw)


@pytest.mark.parametrize("kind", ["missing", "shape", "dtype", "extra"])
def test_complete_inventory_is_validated(kind):
    raw = tiny_config()
    c, where = inventory(raw)
    key = PREFIX + "layers.1.experts.switch_glu.down_proj.weight"
    if kind == "missing":
        del where[key]
    elif kind == "shape":
        where[key] = (*where[key][:4], [4, 64, 1])
    elif kind == "dtype":
        where[key] = (*where[key][:3], "F32", where[key][4])
    else:
        where[PREFIX + "layers.1.self_attn.v_proj.weight"] = (None, 0, 0, "U32", [128, 8])
    with pytest.raises(ValueError):
        validate_tensor_metadata(c, raw, where)


def test_group_override_resolves_before_global_default():
    raw = tiny_config()
    raw["quantization"]["group_size"] = 32
    for name in tensor_contract(Config.from_dict(raw))[0]:
        raw["quantization"][name] = {"bits": 6 if name.endswith("down_proj") else 4, "group_size": 64}
    c, where = inventory(raw)
    validate_tensor_metadata(c, raw, where)
    raw["quantization"][PREFIX + "layers.0.router.proj"]["bits"] = 8
    with pytest.raises(ValueError, match="affine4/6"):
        validate_tensor_metadata(c, raw, where)


def test_nonproportional_rope_is_not_silently_reinterpreted():
    raw = deepcopy(tiny_config())
    raw["text_config"]["rope_parameters"]["full_attention"]["partial_rotary_factor"] = .333
    with pytest.raises(ValueError, match="rotary dimension"):
        Config.from_dict(raw)

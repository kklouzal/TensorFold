"""Gemma companion startup contracts without loading Torch or checkpoint Python."""

import json
import math
import struct

import pytest

from tensorfold.families.gemma4.dflash_metadata import DraftConfig, checked_headers


def config():
    return {"architectures": ["DFlashDraftModel"], "model_type": "qwen3", "hidden_size": 64,
            "intermediate_size": 128, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 32,
            "num_hidden_layers": 2, "num_target_layers": 4, "vocab_size": 32, "block_size": 4,
            "rms_norm_eps": 1e-6, "max_position_embeddings": 8192, "rope_theta": 1000000,
            "layer_types": ["sliding_attention", "full_attention"], "sliding_window": 8,
            "use_sliding_window": True, "final_logit_softcapping": 30.0,
            "dflash_config": {"target_layer_ids": [0, 2], "mask_token_id": 4}}


def write_checkpoint(directory, raw, change=None):
    cfg = DraftConfig.from_dict(raw)
    header, offset = {}, 0
    for name, shape in cfg.shapes().items():
        size = math.prod(shape) * 2
        header[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    if change is not None:
        change(header)
        offset = 0
        for info in header.values():
            size = math.prod(info["shape"]) * 2
            info["data_offsets"] = [offset, offset + size]
            offset += size
    encoded = json.dumps(header).encode()
    (directory / "config.json").write_text(json.dumps(raw))
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))
    return cfg, header


def test_published_gemma_companion_geometry_and_layer_policy():
    raw = {**config(), "hidden_size": 2816, "intermediate_size": 5632, "num_attention_heads": 32,
           "num_key_value_heads": 8, "head_dim": 128, "num_hidden_layers": 5, "num_target_layers": 30,
           "vocab_size": 262144, "block_size": 16, "sliding_window": 2048,
           "max_position_embeddings": 262144,
           "layer_types": ["sliding_attention"] * 4 + ["full_attention"],
           "dflash_config": {"target_layer_ids": [1, 6, 11, 17, 22, 27], "mask_token_id": 4}}
    cfg = DraftConfig.from_dict(raw)
    assert (cfg.hidden, cfg.heads, cfg.kv_heads, cfg.head_dim, cfg.block) == (2816, 32, 8, 128, 16)
    assert cfg.tap_layers == (1, 6, 11, 17, 22, 27)
    assert cfg.causal_layers == (True, True, True, True, False)
    assert len(cfg.shapes()) == 58 and cfg.shapes()["fc.weight"] == (2816, 16896)
    assert cfg.rope.theta == 1000000 and cfg.rope.context_limit == 262144


@pytest.mark.parametrize("change", [
    {"architectures": ["DFlash2DraftModel"]}, {"model_type": "gemma4"},
    {"attention_bias": True}, {"hidden_act": "gelu"}, {"attention_dropout": 0.1},
    {"dtype": "float16"}, {"torch_dtype": "float32"}, {"quantization_config": {"bits": 4}},
    {"quantization": {"bits": 4}}, {"num_attention_heads": 3}, {"num_key_value_heads": True},
    {"head_dim": 30}, {"num_hidden_layers": 2**31}, {"vocab_size": 0},
    {"layer_types": ["sliding_attention"]}, {"layer_types": ["full_attention"] * 2},
    {"layer_types": ["sliding_attention"] * 2}, {"layer_types": ["sliding_attention", "linear_attention"]},
    {"sliding_window": 1}, {"sliding_window": False}, {"use_sliding_window": False}, {"is_causal": 1},
    {"sliding_window": 3}, {"sliding_window": 3, "is_causal": True},
    {"block_size": 1}, {"block_size": 129}, {"final_logit_softcapping": 20},
    {"input_embedding_scale": 2}, {"output_multiplier": 2}, {"output_multiplier": True},
    {"rms_norm_eps": float("nan")}, {"rms_norm_eps": float("inf")}, {"rms_norm_eps": True},
    {"rms_norm_eps": 0}, {"rms_norm_eps": 10**1000}, {"rms_norm_eps": 1e-50},
    {"rope_theta": float("inf")}, {"rope_theta": 1e100}, {"rope_theta": True},
    {"max_position_embeddings": 0}, {"max_position_embeddings": 2**31},
    {"rope_scaling": {"rope_type": "yarn", "factor": 2}}, {"partial_rotary_factor": True},
    {"rope_parameters": {"partial_rotary_factor": 0.5}},
    {"dflash_config": {"target_layer_ids": [2, 0], "mask_token_id": 4}},
    {"dflash_config": {"target_layer_ids": [0, 0], "mask_token_id": 4}},
    {"dflash_config": {"target_layer_ids": [0, 4], "mask_token_id": 4}},
    {"dflash_config": {"target_layer_ids": [0, True], "mask_token_id": 4}},
    {"dflash_config": {"target_layer_ids": [0, 2], "mask_token_id": 32}},
    {"dflash_config": {"target_layer_ids": [0, 2], "mask_token_id": False}},
])
def test_invalid_or_unsupported_metadata_refuses_before_payload(change):
    with pytest.raises(ValueError):
        DraftConfig.from_dict({**config(), **change})


@pytest.mark.parametrize("causal", [True, False])
def test_explicit_authoritative_causal_override(causal):
    cfg = DraftConfig.from_dict({**config(), "is_causal": causal})
    assert cfg.causal_layers == (causal, causal)
    with pytest.raises(ValueError, match="fit the declared sliding window"):
        DraftConfig.from_dict({**config(), "is_causal": causal, "sliding_window": 3})


def test_nested_draft_values_have_the_official_precedence():
    raw = config()
    raw["dflash_config"] = {**raw["dflash_config"], "block_size": 6, "final_logit_softcapping": 30.0}
    raw["block_size"], raw["final_logit_softcapping"] = 2, 20
    assert DraftConfig.from_dict(raw).block == 6


def test_conflicting_rotary_theta_and_partial_boolean_refuse():
    for params in ({"rope_theta": 10000}, {"partial_rotary_factor": True}):
        with pytest.raises(ValueError):
            DraftConfig.from_dict({**config(), "rope_parameters": params})


def test_checked_headers_accept_the_declared_bf16_inventory(tmp_path):
    cfg, _ = write_checkpoint(tmp_path, config())
    assert checked_headers(tmp_path)[0] == cfg


@pytest.mark.parametrize("change", [
    lambda h: h.pop("norm.weight"),
    lambda h: h.__setitem__("unknown.weight", h.pop("norm.weight")),
    lambda h: h["norm.weight"].__setitem__("shape", [32, 2]),
    lambda h: h["norm.weight"].__setitem__("dtype", "F16"),
])
def test_header_inventory_shape_and_dtype_are_independently_checked(tmp_path, change):
    write_checkpoint(tmp_path, config(), change)
    with pytest.raises(ValueError, match="tensor count|tensor inventory|declared shapes"):
        checked_headers(tmp_path)


def test_excessive_layer_declaration_fails_before_shape_materialization(tmp_path, monkeypatch):
    # The count check precedes cfg.shapes, including enormous but signed32-valid
    # declarations. Avoid allocating a multi-billion-entry fixture list here.
    cfg, _ = write_checkpoint(tmp_path, config())
    from dataclasses import replace

    monkeypatch.setattr(DraftConfig, "read", classmethod(lambda cls, directory: replace(cfg, layers=2**31 - 1)))
    monkeypatch.setattr(DraftConfig, "shapes", lambda self: pytest.fail("unsafe shape materialization"))
    with pytest.raises(ValueError, match="tensor count"):
        checked_headers(tmp_path)

"""Speculators metadata/header boundaries without importing accelerator code."""
import json
import math
import struct

import pytest

from tensorfold.families.gemma4.e2b_dflash_metadata import E2BDraftConfig, checked_headers


def config():
    return {
        "architectures": ["DFlash2DraftModel"], "speculators_model_type": "dflash2",
        "dtype": "bfloat16", "aux_hidden_state_layer_ids": [1, 3],
        "block_size": 4, "conv_group_size": 4, "conv_kernel_size": 2,
        "draft_vocab_size": 16, "mask_token_id": 4, "sample_from_anchor": False,
        "selector_rank": 4, "selector_top_k": 3, "sliding_window_non_causal": True,
        "speculators_config": {"algorithm": "dflash2"},
        "transformer_layer_config": {
            "model_type": "qwen3", "attention_bias": False, "attention_dropout": 0.0,
            "head_dim": 8, "hidden_act": "silu", "hidden_size": 16, "intermediate_size": 32,
            "layer_types": ["sliding_attention", "full_attention"], "max_position_embeddings": 32,
            "num_attention_heads": 2, "num_hidden_layers": 2, "num_key_value_heads": 1,
            "rms_norm_eps": 1e-6, "rope_parameters": {"rope_theta": 10000.0, "rope_type": "default"},
            "sliding_window": 4, "use_sliding_window": True, "vocab_size": 16,
        },
    }


def test_hidden_indices_are_not_decoder_indices():
    cfg = E2BDraftConfig.from_dict(config())
    assert cfg.tap_hidden_indices == (1, 3)
    assert cfg.tap_layers == (0, 2)
    assert cfg.shapes()["fc.weight"] == (16, 32)
    assert cfg.shapes()["candidate_selector.predecessor_codebook"] == (16, 4)
    assert cfg.parameter_bytes == 2 * sum(math.prod(s) for s in cfg.shapes().values())


def test_declared_published_geometry():
    raw = config()
    raw.update(aux_hidden_state_layer_ids=[3, 13, 23, 33], block_size=8, conv_group_size=16,
               draft_vocab_size=262144, selector_rank=256, selector_top_k=16)
    raw["transformer_layer_config"].update(
        hidden_size=1536, intermediate_size=6144, num_attention_heads=12, num_key_value_heads=4,
        head_dim=128, num_hidden_layers=5, vocab_size=262144, sliding_window=2048,
        max_position_embeddings=131072, layer_types=["sliding_attention"] * 4 + ["full_attention"])
    cfg = E2BDraftConfig.from_dict(raw)
    assert cfg.tap_layers == (2, 12, 22, 32)
    assert len(cfg.shapes()) == 81
    assert cfg.shapes()["fc.weight"] == (1536, 6144)
    # Four prefix rings + chronological SDPA buffers, and one full prefix+mask tail.
    assert cfg.state_bytes(131072) == (4 * (2048 * 2 + 8) + 131072 + 8) * 4 * 128 * 4


@pytest.mark.parametrize("change", [
    {"architectures": ["DFlashDraftModel"]}, {"dtype": "float16"},
    {"sample_from_anchor": True}, {"sliding_window_non_causal": False},
    {"conv_kernel_size": 3}, {"draft_vocab_size": 8}, {"mask_token_id": 16},
    {"aux_hidden_state_layer_ids": [0, 3]}, {"aux_hidden_state_layer_ids": [3, 1]},
    {"aux_hidden_state_layer_ids": [1, 1]}, {"selector_top_k": 17},
    {"conv_group_size": 3}, {"quantization_config": {"bits": 4}},
    {"torch_dtype": "float32"}, {"draft_vocab_size": 16.0},
])
def test_incompatible_draft_is_rejected(change):
    raw = config()
    raw.update(change)
    with pytest.raises(ValueError):
        E2BDraftConfig.from_dict(raw)


@pytest.mark.parametrize("name,value", [
    ("num_attention_heads", 3), ("head_dim", 7), ("attention_bias", True),
    ("attention_dropout", 0.1), ("hidden_act", "gelu"), ("use_sliding_window", False),
    ("rms_norm_eps", float("nan")), ("layer_types", ["full_attention"]),
    ("rms_norm_eps", 1e-100), ("partial_rotary_factor", 0.5),
])
def test_nested_transformer_boundary(name, value):
    raw = config()
    raw["transformer_layer_config"][name] = value
    with pytest.raises(ValueError):
        E2BDraftConfig.from_dict(raw)


def write_headers(directory, raw, mutate=None):
    cfg = E2BDraftConfig.from_dict(raw)
    header = {k: {"shape": list(v), "dtype": "BF16"} for k, v in cfg.shapes().items()}
    if mutate:
        mutate(header)
    offset = 0
    for entry in header.values():
        size = math.prod(entry["shape"]) * 2
        entry["data_offsets"] = [offset, offset + size]
        offset += size
    encoded = json.dumps(header).encode()
    (directory / "config.json").write_text(json.dumps(raw))
    (directory / "model.safetensors").write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(offset))


def test_complete_header_contract(tmp_path):
    write_headers(tmp_path, config())
    cfg, found = checked_headers(tmp_path)
    assert set(found) == set(cfg.shapes())


@pytest.mark.parametrize("mutation", [
    lambda h: h.pop("candidate_selector.successor_codebook"),
    lambda h: h["fc.weight"].update(shape=[16, 16]),
    lambda h: h["norm.weight"].update(dtype="F16"),
])
def test_wrong_tensor_header_is_refused(tmp_path, mutation):
    write_headers(tmp_path, config(), mutation)
    with pytest.raises(ValueError):
        checked_headers(tmp_path)


def test_context_bound_is_checkpoint_owned():
    cfg = E2BDraftConfig.from_dict(config())
    with pytest.raises(ValueError, match="position limit"):
        cfg.state_bytes(33)


def test_private_fp8_only_selects_proposal_projections_and_accounts_for_head():
    raw = config()
    raw.update(conv_group_size=16, draft_vocab_size=128)
    raw["transformer_layer_config"].update(hidden_size=64, intermediate_size=128,
                                          num_attention_heads=4, num_key_value_heads=2,
                                          head_dim=16, vocab_size=128)
    cfg = E2BDraftConfig.from_dict(raw)
    assert len(cfg.projection_shapes()) == 1 + 7 * cfg.layers
    assert all("conv" not in n and "norm" not in n and "selector" not in n
               for n in cfg.projection_shapes())
    assert cfg.memory_bytes(16) == (cfg.parameter_bytes, cfg.parameter_bytes)
    resident, peak = cfg.memory_bytes(8)
    assert resident > 0 and peak >= resident
    with pytest.raises(ValueError, match="precision"):
        cfg.memory_bytes(4)


@pytest.mark.parametrize("theta", [0.5, 1e300])
def test_inverse_rotary_frequencies_must_fit_float32(theta):
    raw = config()
    raw["transformer_layer_config"]["rope_parameters"]["rope_theta"] = theta
    with pytest.raises(ValueError):
        E2BDraftConfig.from_dict(raw)

"""CPU-only metadata/load boundary checks; no external checkpoint required."""

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from tensorfold.families.qwen4_exp.cuda.dspark_config import DSparkConfig
from tensorfold.families.qwen4_exp.cuda.dspark_weights import (
    tensor_shapes,
    plan_weights,
    weight_byte_plan,
    load_reference,
)


def config_dict():
    return {
        "architectures": ["DSparkDraftModel"],
        "model_type": "qwen3",
        "hidden_size": 8,
        "vocab_size": 17,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 4,
        "intermediate_size": 16,
        "num_target_layers": 4,
        "block_size": 3,
        "max_position_embeddings": 32,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "layer_types": ["full_attention"] * 2,
        "dflash_config": {
            "attention_mode": "gqa",
            "projector_type": "dspark",
            "markov_head_type": "vanilla",
            "confidence_head_with_markov": True,
            "markov_rank": 3,
            "mask_token_id": 16,
            "target_layer_ids": [0, 2],
        },
    }


def target(config):
    return SimpleNamespace(hidden=config.hidden, vocab=config.vocab, layers=config.target_layers, streams=4)


def header(config):
    offset = 0
    result = {}
    for name, shape in tensor_shapes(config).items():
        size = 2 * __import__("math").prod(shape)
        result[name] = {"dtype": "BF16", "shape": list(shape), "data_offsets": [offset, offset + size]}
        offset += size
    return result


def write_tiny(directory, *, overflow=False, seed=17):
    directory.mkdir(exist_ok=True)
    raw = config_dict()
    config = DSparkConfig.from_dict(raw)
    torch.manual_seed(seed)
    tensors = {
        name: (torch.randn(shape) * 0.07 + (1 if "norm" in name else 0)).bfloat16()
        for name, shape in tensor_shapes(config).items()
    }
    if overflow:
        tensors["fc.weight"][0, 0] = float("inf")
    save_file(tensors, directory / "model.safetensors")
    (directory / "config.json").write_text(json.dumps(raw))
    return config, tensors


def test_config_pinned_capture_translation_and_target_contract():
    raw = config_dict()
    c = DSparkConfig.from_dict(raw)
    assert c.taps == (0, 2) and c.capture_before == (1, 3)
    assert c.identity == DSparkConfig.from_dict(deepcopy(raw)).identity
    c.check_target(target(c))
    for field in ("hidden", "vocab", "layers", "streams"):
        t = target(c)
        setattr(t, field, 99)
        with pytest.raises(ValueError, match="target|capture"):
            c.check_target(t)


@pytest.mark.parametrize(
    "path,value",
    [
        (("model_type",), "qwen4_exp"),
        (("sample_from_anchor",), False),
        (("num_attention_heads",), True),
        (("dflash_config", "target_layer_ids"), [0]),
        (("dflash_config", "markov_head_type"), "rnn"),
        (("dflash_config", "conv_kernel_size"), 3),
        (("dflash_config", "attention_mode"), "mla"),
        (("is_causal",), True),
        (("rope_scaling",), {"type": "yarn"}),
        (("dflash_config", "confidence_head_alpha"), 0.5),
        (("partial_rotary_factor",), 0.25),
    ],
)
def test_unsupported_config_fails(path, value):
    raw = config_dict()
    raw["dflash_config"]["target_layer_ids"] = [0, 2]
    cursor = raw
    for item in path[:-1]:
        cursor = cursor[item]
    cursor[path[-1]] = value
    if path == ("dflash_config", "target_layer_ids"):
        cursor[path[-1]] = [2, 0]
    with pytest.raises(ValueError):
        DSparkConfig.from_dict(raw)


def test_header_exact_names_shapes_offsets_and_tp():
    raw = config_dict()
    raw["dflash_config"]["target_layer_ids"] = [0, 2]
    c = DSparkConfig.from_dict(raw)
    h = header(c)
    for world in [1, 2, 4]:
        for rank in range(world):
            plans = plan_weights(c, h, tp=(rank, world))
            q = next(p for p in plans if p.name == "layers.0.self_attn.q_proj.weight")
            assert q.shape == (16 // world, 8)
            kv = next(p for p in plans if p.name == "layers.0.self_attn.k_proj.weight")
            assert kv.shape == (max(1, 2 // world) * 4, 8)
        totals = [
            next(p for p in plan_weights(c, h, tp=(rank, world)) if p.name == "markov_head.markov_w2.weight").shape[0]
            for rank in range(world)
        ]
        assert sum(totals) == 17 and max(totals) - min(totals) <= 1
    for mutate in [
        lambda x: x.pop("fc.weight"),
        lambda x: x.update(extra=h["fc.weight"]),
        lambda x: x["fc.weight"].update(shape=[1, 1]),
        lambda x: x["fc.weight"].update(dtype="I8"),
        lambda x: x["fc.weight"].update(data_offsets=[1, 513]),
    ]:
        broken = deepcopy(h)
        mutate(broken)
        with pytest.raises(ValueError):
            plan_weights(c, broken)


def test_reference_load_bounded_conversion_identity_and_plan(tmp_path):
    c, original = write_tiny(tmp_path)
    w = load_reference(tmp_path, target=target(c), chunk_bytes=128)
    assert w.nbytes == sum(p.nbytes for p in plan_weights(c, header(c)))
    for name, tensor in w.tensors.items():
        assert torch.equal(tensor, original[name].to(tensor.dtype))
        assert tensor.device.type == "cpu"
    plan = weight_byte_plan(c, header(c), capacity=16, slots=6, prefill_rows=4, chunk_bytes=256)
    assert plan["resident_bytes"] == w.nbytes and plan["target_shared_weight_bytes"] == 0
    assert plan["state_bytes"] == 6 * 2 * 16 * 2 * 4 * 4
    assert plan["staging_bytes"] == 512 and plan["reference_execution_worlds"] == (1,)
    old = w.identity
    original["fc.weight"][0, 0] += 0.25
    save_file(original, tmp_path / "model.safetensors")
    assert load_reference(tmp_path, target=target(c), chunk_bytes=128).identity != old
    with pytest.raises(ValueError, match="row"):
        load_reference(tmp_path, target=target(c), chunk_bytes=1)


def test_nonfinite_conversion_rejected(tmp_path):
    c, _ = write_tiny(tmp_path, overflow=True)
    with pytest.raises(ValueError, match="nonfinite|overflow"):
        load_reference(tmp_path, target=target(c))


def test_invalid_budget_dimensions():
    raw = config_dict()
    raw["dflash_config"]["target_layer_ids"] = [0, 2]
    c = DSparkConfig.from_dict(raw)
    for kwargs in [{"capacity": 0}, {"capacity": 33}, {"capacity": 16, "block": 4}, {"capacity": 16, "slots": True}]:
        with pytest.raises(ValueError):
            weight_byte_plan(c, header(c), **kwargs)


def test_real_pinned_header_matches_schema_and_exact_tp_allocations():
    root = Path(__file__).parent / "fixtures/dspark_qwen38"
    c = DSparkConfig.from_dict(json.loads((root / "dspark_config.json").read_text()))
    h = json.loads((root / "dspark_header.json").read_text())
    assert c.capture_before == (2, 13, 24, 35, 46)
    assert len(tensor_shapes(c)) == 62
    # Independently counted: five dense blocks shard their 1,009,254,400
    # matrix bytes, while projector/Markov-input/norm/confidence replicate.
    for world, expected in [(1, 1329209348), (4, 476913668), (8, 334864388)]:
        for rank in [0, world - 1]:
            plan = weight_byte_plan(c, h, tp=(rank, world), capacity=32768, slots=6, prefill_rows=256)
            assert plan["resident_bytes"] == expected
            assert plan["state_bytes"] == 6 * 5 * 32768 * (8 // world) * 128 * 4
            assert plan["serving_bytes"] == plan["resident_bytes"] + plan["state_bytes"] + plan["workspace_bytes"]
    assert (
        sum(info["data_offsets"][1] - info["data_offsets"][0] for name, info in h.items() if name != "__metadata__")
        == 1329139714
    )


@pytest.mark.parametrize("value", [65536.0, float("nan")])
def test_finite_bf16_overflow_is_rejected_before_half_rounding(tmp_path, value):
    c, tensors = write_tiny(tmp_path)
    tensors["fc.weight"][0, 0] = value
    save_file(tensors, tmp_path / "model.safetensors")
    with pytest.raises(ValueError, match="nonfinite|overflow"):
        load_reference(tmp_path, target=target(c))

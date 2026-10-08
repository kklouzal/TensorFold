"""Independent header oracles for compact EXL3 expert spillover admission."""

import json
import math
import struct
import importlib
from types import SimpleNamespace

import pytest

from tensorfold.cuda import capacity
from tensorfold.cuda.geometry import indexed_weights
from tensorfold.families.qwen4_exp import ram_experts
from test_cuda_geometry import allocations, bytes_in  # noqa: F401 (pytest fixture registration)


def checkpoint(path, *, mtp=True, signs=False, missing=None, shared_bits=4, routed_bits=2):
    text = {"hidden_size": 256, "moe_intermediate_size": 128, "shared_expert_intermediate_size": 128,
            "num_experts": 3, "num_experts_per_tok": 2, "num_hidden_layers": 2,
            "max_position_embeddings": 128}
    (path / "config.json").write_text(json.dumps({"model_type": "qwen4_exp", "text_config": text,
        "quantization_config": {"quant_method": "exl3", "version": "1.4.2", "bits": 2.05, "codebook": "mul1"}}))
    entries, offset = {}, 0

    def add(name, dtype, shape):
        nonlocal offset
        if name == missing:
            return
        size = math.prod(shape) * (4 if dtype == "I32" else 2)
        entries[name] = {"dtype": dtype, "shape": shape, "data_offsets": [offset, offset + size]}
        offset += size

    bases = [f"model.language_model.layers.{i}.mlp" for i in range(2)]
    if mtp:
        bases.append("mtp.layers.0.mlp")
    for base in bases:
        add(base + ".gate.weight", "BF16", [3, 256])
        add(base + ".shared_expert_gate.weight", "BF16", [1, 256])
        for expert, bits in [(f"experts.{i}", routed_bits) for i in range(3)] + [("shared_expert", shared_bits)]:
            for projection, k, n in (("gate_proj", 256, 128), ("up_proj", 256, 128), ("down_proj", 128, 256)):
                prefix = f"{base}.{expert}.{projection}"
                add(prefix + ".trellis", "I16", [k // 16, n // 16, int(bits * 16)])
                add(prefix + ".su" if signs else prefix + ".suh", "I16" if signs else "F16",
                    [k // 16 if signs else k])
                add(prefix + ".sv" if signs else prefix + ".svh", "I16" if signs else "F16",
                    [n // 16 if signs else n])
                add(prefix + ".mul1", "I32", [])
    raw = json.dumps(entries).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw)
    return text, entries


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("signs", [False, True])
def test_compact_authority_maximum_cell_and_resident_logical_scales(tmp_path, mtp, signs):
    _, entries = checkpoint(tmp_path, signs=signs)
    # Each projection has 256*128 weights; routed streams are 2 bit and
    # shared streams 4 bit. The host stores each original stream once.
    small, large = 3 * 256 * 128 * 2 // 8, 3 * 256 * 128 * 4 // 8
    layers = 2 + int(mtp)
    plan = ram_experts.layout(tmp_path, (2 * large + 13) / 2**30, mtp=mtp)
    assert plan.format == "exl3" and plan.entry_bytes == large
    assert plan.slots == 2 and plan.gpu_bytes == 2 * large
    assert plan.host_bytes == layers * (3 * small + large)
    assert plan.host_bytes < layers * 4 * large
    assert plan.staging_bytes == 2 * large
    assert plan.metadata_device_bytes == layers * 4 * (3 * 8 + 3 * 4)
    assert plan.metadata_host_bytes == layers * 4 * 28
    assert plan.control_device_bytes == plan.slots * 32
    assert plan.control_host_bytes == plan.slots * 64
    transform = plan.transform(indexed_weights(1, mtp))
    resident = capacity.estimate_weights(tmp_path, transform).resident
    # Gate/up input scales D, output scales I; down scales I,D.
    scales = layers * 4 * 3 * (256 + 128) * 2
    routers = layers * (3 + 1) * 256 * 2
    assert resident == scales + routers
    for name, info in entries.items():
        if name.endswith((".trellis", ".mul1")):
            assert transform(name, info) == (0, 0)


@pytest.mark.parametrize("routed_bits,shared_bits", [(1, 2), (1.5, 3.5), (3, 8)])
def test_supported_original_widths_remain_distinct(tmp_path, routed_bits, shared_bits):
    checkpoint(tmp_path, routed_bits=routed_bits, shared_bits=shared_bits)
    plan = ram_experts.layout(tmp_path, 0.001)
    assert plan.entry_bytes == 3 * 256 * 128 * shared_bits // 8
    assert plan.host_bytes == 3 * (3 * 3 * 256 * 128 * routed_bits // 8 + plan.entry_bytes)


def test_incomplete_expert_group_and_subcell_budget_fail_before_payload(tmp_path):
    checkpoint(tmp_path, missing="model.language_model.layers.1.mlp.experts.2.down_proj.svh")
    with pytest.raises(ValueError, match="no output scales"):
        ram_experts.layout(tmp_path, 1)
    checkpoint(tmp_path)
    with pytest.raises(ValueError, match="at least one packed EXL3 expert"):
        ram_experts.layout(tmp_path, 1 / 2**30)


def test_bad_configured_layer_width_and_mixed_codebooks_fail_before_payload(tmp_path):
    text, entries = checkpoint(tmp_path)
    text["num_hidden_layers"] = 3
    raw = json.loads((tmp_path / "config.json").read_text())
    raw["text_config"] = text
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="every configured main"):
        ram_experts.layout(tmp_path, 1)
    _, entries = checkpoint(tmp_path)
    name = "model.language_model.layers.0.mlp.experts.0.gate_proj.trellis"
    entries[name]["shape"][0] = 8
    entries[name]["data_offsets"][1] = entries[name]["data_offsets"][0] + math.prod(entries[name]["shape"]) * 2
    scale = name.removesuffix(".trellis") + ".suh"
    entries[scale]["shape"] = [128]
    entries[scale]["data_offsets"][1] = entries[scale]["data_offsets"][0] + 128 * 2
    blob = json.dumps(entries).encode()
    (tmp_path / "model.safetensors").write_bytes(struct.pack("<Q", len(blob)) + blob)
    with pytest.raises(ValueError, match="incompatible EXL3 projection"):
        ram_experts.layout(tmp_path, 1)
    checkpoint(tmp_path, missing="model.language_model.layers.0.mlp.experts.0.gate_proj.mul1")
    with pytest.raises(ValueError, match="mixed EXL3 codebooks"):
        ram_experts.layout(tmp_path, 1)


@pytest.mark.torch
@pytest.mark.parametrize("dims,width,rows,slots,count", [(2560, 640, 1024, 9, 513), (384, 128, 17, 3, 8)])
def test_native_exl3_member_storage_and_count_are_in_the_memory_budget(
        monkeypatch, allocations, dims, width, rows, slots, count):  # noqa: F811 (pytest fixture)
    from tensorfold.cuda import geometry

    arrays, fake = allocations
    experts = importlib.import_module("tensorfold.cuda.exl3.experts")
    mm = importlib.import_module("tensorfold.families.qwen4_exp.cuda.exl3_mm")
    monkeypatch.setattr(experts, "torch", fake)
    ex = SimpleNamespace(dims=dims, width=width, count=count)
    scratch = experts.Scratch(ex, rows, slots, device="cpu")
    actual = bytes_in(arrays)
    owner = mm.Scratch.__new__(mm.Scratch)
    owner.xh = owner.z = owner.tmp = owner.part = owner.ple_dev = owner.ple_emb = None
    owner.moe, owner.prefill = scratch, SimpleNamespace(nbytes=lambda: 0)
    assert owner.nbytes() == actual  # includes the four-byte unique count
    if dims == 2560:
        assert actual == 391_387_144
    text = {"hidden_size": dims, "hc_count": 4, "num_experts_per_tok": slots - 1,
            "num_experts": count - 1, "moe_intermediate_size": width, "num_attention_heads": 8,
            "head_dim": 64, "linear_num_key_heads": 2, "linear_key_head_dim": 128,
            "linear_num_value_heads": 4, "linear_value_head_dim": 128, "heads_per_ngram": 8,
            "ple_embed_dim": dims}
    budget = geometry.exl3_indexed_scratch(text, rows, 2048)
    conv = 2 * 2 * 128 + 4 * 128
    workspace = geometry.exl3_workspace(dims * max(2 * 8 * 64, conv, 4 * dims), 2048 * 4,
                                        max(dims, 8 * 64))
    ple = 2048 * (4 * 81 * 2 * 8 + 2 * dims)
    assert budget - workspace - ple >= actual

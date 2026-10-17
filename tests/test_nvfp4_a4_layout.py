"""Native A4 header/field accounting independent of device execution."""

import math

import pytest

from tensorfold.cuda.nvfp4.host_layout import authority_fields_a4, bank_bases_a4, routed_tensor_a4


def headers():
    cfg = {"num_hidden_layers": 2, "num_experts": 3, "moe_intermediate_size": 64, "hidden_size": 128}
    infos = {}
    for layer in range(2):
        for expert in range(3):
            for projection, n, k in (("gate_proj", 64, 128), ("up_proj", 64, 128), ("down_proj", 128, 64)):
                for field, dtype, shape in (("weight_packed", "U8", [n, k // 2]),
                    ("weight_scale", "F8_E4M3", [n, k // 16]),
                    ("weight_global_scale", "F32", [1]), ("input_global_scale", "F32", [1])):
                    infos[f"model.language_model.layers.{layer}.mlp.experts.{expert}.{projection}.{field}"] = {
                        "dtype": dtype, "shape": shape}
    return cfg, infos


def test_six_fields_match_original_payload_and_padding_oracle():
    fields = authority_fields_a4(256, 512, 2048)
    sizes = [math.prod(shape) * (1 if dtype == "U8" else 4) for shape, dtype in fields]
    assert sum(sizes) == 256 * (3 * 512 * 2048 * 9 // 16 + 24)
    gpu = sum(-(-(n // 256) // 16) * 16 for n in sizes)
    assert gpu == 1_769_536
    assert [dtype for _, dtype in fields] == ["U8", "U8", "F32", "F32", "F32", "F32"]


def test_complete_main_banks_and_exact_field_selection():
    cfg, infos = headers()
    assert bank_bases_a4(infos, cfg) == tuple(f"model.language_model.layers.{i}.mlp" for i in range(2))
    assert all(routed_tensor_a4(name) for name in infos)
    assert not routed_tensor_a4("model.language_model.layers.0.mlp.shared_expert.gate_proj.weight_packed")
    assert not routed_tensor_a4("model.language_model.layers.0.mlp.experts.0.gate_proj.weight")


@pytest.mark.parametrize("damage", ["missing", "dtype", "shape", "expert", "layer", "suffix"])
def test_incomplete_or_conflicting_storage_refuses_before_allocations(damage):
    cfg, infos = headers(); name = next(iter(infos))
    if damage == "missing": del infos[name]
    elif damage == "dtype": infos[name]["dtype"] = "I8"
    elif damage == "shape": infos[name]["shape"] = [1]
    elif damage == "expert": infos[name.replace("experts.0.", "experts.3.")] = infos.pop(name)
    elif damage == "layer": cfg["num_hidden_layers"] = 2**62
    else: infos[name.replace("weight_packed", "weight")] = infos.pop(name)
    with pytest.raises(ValueError): bank_bases_a4(infos, cfg)


@pytest.mark.parametrize("args", [(True, 512, 2048), (0, 512, 2048), (1024, 512, 2048),
                                  (256, 63, 2048), (256, 512, 2**31), (1023, 2**30, 2**30)])
def test_invalid_or_oversize_geometry_is_rejected(args):
    with pytest.raises(ValueError): authority_fields_a4(*args)

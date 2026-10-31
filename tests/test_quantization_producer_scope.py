"""Producer selectors bind exact planned modules; no ModelOpt or GPU needed."""

import copy
from fnmatch import fnmatchcase

import pytest

from tensorfold.offline_quantization import NVFP4_RECIPE
from tensorfold.offline_quantization.producer_scope import scoped_local_hessian_config, selected_projections


def plan():
    return {
        "schema": "tensorfold.offline-quantization-plan",
        "schema_version": 1,
        "recipe": {
            "name": NVFP4_RECIPE,
            "numerics": "W4A4",
            "block_size": 16,
            "weight_scale_algorithm": "local_hessian",
            "fp8_scale_sweep": True,
            "batch_amax_fallback": False,
        },
        "coverage": {"selected_projections": 1},
        "tensors": [
            {
                "name": "model.language_model.layers.0.mlp.gate_proj.weight",
                "shape": [10240, 2560],
                "source_dtype": "BF16",
                "action": "quantize",
            },
            {
                "name": "model.audio_tower.layers.0.linear.weight",
                "shape": [256, 256],
                "source_dtype": "BF16",
                "action": "exclude",
            },
        ],
    }


def preset():
    return {
        "quant_cfg": [
            {"quantizer_name": "*", "enable": False},
            {
                "quantizer_name": "*weight_quantizer",
                "cfg": {"num_bits": "e2m1", "block_sizes": {-1: 16, "type": "static", "scale_bits": "e4m3"}},
            },
            {
                "quantizer_name": "*input_quantizer",
                "cfg": {"num_bits": "e2m1", "block_sizes": {-1: 16, "type": "dynamic", "scale_bits": "e4m3"}},
            },
        ],
        "algorithm": {"method": "local_hessian", "fp8_scale_sweep": True},
    }


def test_exact_scope_keeps_audio_head_tables_and_unselected_text_disabled():
    p, native = plan(), preset()
    before = copy.deepcopy((p, native))
    result = scoped_local_hessian_config(p, native)
    assert (p, native) == before
    selected = "model.language_model.layers.0.mlp.gate_proj"
    for module in (
        selected,
        "model.language_model.layers.0.mlp.up_proj",
        "model.audio_tower.layers.0.linear",
        "model.vision_tower.linear",
        "model.embed_tokens",
        "lm_head",
        "masked_embedding.centroids",
    ):
        for suffix in ("weight_quantizer", "input_quantizer", "output_quantizer"):
            key = module + "." + suffix
            enabled = False
            for entry in result["quant_cfg"]:
                if fnmatchcase(key, entry["quantizer_name"]):
                    enabled = entry["enable"]
            assert enabled is (module == selected and suffix != "output_quantizer")
    assert result["algorithm"]["debug"] is True
    assert result["algorithm"]["hessian_input_dtype"] == "float32"


def test_scope_copies_each_config_and_keeps_actual_producer_numeric_attributes():
    p, native = plan(), preset()
    result = scoped_local_hessian_config(p, native)
    assert result["quant_cfg"][1]["cfg"] == dict(native["quant_cfg"][1]["cfg"], type="static")
    assert result["quant_cfg"][2]["cfg"] == dict(native["quant_cfg"][2]["cfg"], type="static")
    result["quant_cfg"][1]["cfg"]["block_sizes"][-1] = 32
    assert native["quant_cfg"][1]["cfg"]["block_sizes"][-1] == 16
    assert result["quant_cfg"][2]["cfg"]["block_sizes"][-1] == 16


@pytest.mark.parametrize("name", ["*.weight", "model.[0].weight", "model..layer.weight", "model.layer.bias"])
def test_selector_injection_and_nonweight_names_rejected(name):
    p = plan()
    p["tensors"][0]["name"] = name
    with pytest.raises(ValueError):
        selected_projections(p)


def test_duplicate_or_count_mismatch_and_batch_amax_fallback_refused():
    p = plan()
    p["tensors"].append(copy.deepcopy(p["tensors"][0]))
    with pytest.raises(ValueError):
        selected_projections(p)
    p = plan()
    p["coverage"]["selected_projections"] = 2
    with pytest.raises(ValueError):
        selected_projections(p)
    p = plan()
    p["recipe"]["batch_amax_fallback"] = True
    with pytest.raises(ValueError):
        selected_projections(p)


@pytest.mark.parametrize("mutation", ["algorithm", "weight_format", "block_size", "scale_format", "old_mapping"])
def test_changed_producer_recipe_refused_instead_of_guessing_compatibility(mutation):
    native = preset()
    if mutation == "algorithm":
        native["algorithm"]["method"] = "max"
    elif mutation == "old_mapping":
        native["quant_cfg"] = {}
    elif mutation == "weight_format":
        native["quant_cfg"][1]["cfg"]["num_bits"] = 8
    elif mutation == "block_size":
        native["quant_cfg"][1]["cfg"]["block_sizes"][-1] = 32
    else:
        native["quant_cfg"][1]["cfg"]["block_sizes"]["scale_bits"] = "e8m0"
    with pytest.raises(ValueError):
        scoped_local_hessian_config(plan(), native)


@pytest.mark.parametrize(
    "mutation",
    ["legacy_dynamic", "global_dynamic", "axis", "static_input", "dynamic_weight", "scale_axis", "bias", "real_quant"],
)
def test_calibration_modes_cannot_silently_become_batch_dependent(mutation):
    native = preset()
    weight = native["quant_cfg"][1]["cfg"]
    activation = native["quant_cfg"][2]["cfg"]
    if mutation == "legacy_dynamic":
        activation["dynamic"] = True
    elif mutation == "global_dynamic":
        activation["type"] = "dynamic"
    elif mutation == "axis":
        activation["axis"] = 0
    elif mutation == "static_input":
        activation["block_sizes"]["type"] = "static"
    elif mutation == "dynamic_weight":
        weight["block_sizes"]["type"] = "dynamic"
    elif mutation == "scale_axis":
        activation["block_sizes"]["scale_block_sizes"] = {-1: 2}
    elif mutation == "bias":
        activation["bias"] = {"type": "static"}
    else:
        activation["fake_quant"] = False
    with pytest.raises(ValueError):
        scoped_local_hessian_config(plan(), native)

"""Publisher NVFP4 schema boundaries without importing an accelerator runtime."""

from copy import deepcopy

import pytest

from gemma_cuda_target_metadata_test import tiny_config
from tensorfold.families.gemma4.cuda.nvfp4_weights import CT_PREFIX, ct_tensor_contract, validate_headers
from tensorfold.families.gemma4.cuda.weights import Config


def fixture():
    raw = tiny_config()
    del raw["quantization"]
    raw["quantization_config"] = {
        "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized", "kv_cache_scheme": None,
        "ignore": ["lm_head", "re:.*embed_vision.*", "re:.*vision_tower.*"],
        "config_groups": {"group_0": {"targets": ["Linear"], "format": "nvfp4-pack-quantized",
            "weights": {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group",
                        "symmetric": True, "dynamic": False, "scale_dtype": "torch.float8_e4m3fn"},
            "input_activations": {"num_bits": 4, "type": "float", "group_size": 16, "strategy": "tensor_group",
                        "symmetric": True, "dynamic": "local", "scale_dtype": "torch.float8_e4m3fn"},
            "output_activations": None}}}
    native, plain = ct_tensor_contract(Config.from_dict(raw))
    headers = {name: {"dtype": "BF16", "shape": list(shape)} for name, shape in plain.items()}
    for name, (n,k) in native.items():
        for part,dtype,shape in (("weight_packed","U8",[n,k//2]),("weight_scale","F8_E4M3",[n,k//16]),
                                 ("weight_global_scale","F32",[1]),("input_global_scale","F32",[1])):
            headers[name+"."+part] = {"dtype":dtype,"shape":shape}
    return raw,headers


def test_publisher_per_expert_original_storage_and_plain_router():
    raw,headers = fixture()
    c = validate_headers(raw,headers)
    assert c.hidden == 64 and c.layers == 2
    assert headers[CT_PREFIX+"embed_tokens.weight"] == {"dtype":"BF16","shape":[128,64]}
    assert headers[CT_PREFIX+"layers.0.router.proj.weight"] == {"dtype":"BF16","shape":[4,64]}
    assert headers[CT_PREFIX+"layers.0.experts.3.down_proj.weight_packed"]["shape"] == [64,32]


@pytest.mark.parametrize("part,dtype,shape",[("weight_packed","U8",[64,1]),("weight_scale","F32",[64,4]),
    ("input_global_scale","F32",[2]),("weight_global_scale","BF16",[1])])
def test_native_layout_refuses_malformed_parts(part,dtype,shape):
    raw,headers = fixture()
    headers[CT_PREFIX+"layers.0.experts.0.down_proj."+part] = {"dtype":dtype,"shape":shape}
    with pytest.raises(ValueError):
        validate_headers(raw,headers)


@pytest.mark.parametrize("field,value",[("num_bits",8),("group_size",32),("dynamic",True),("symmetric",False)])
def test_declared_activation_math_cannot_silently_degrade(field,value):
    raw,headers = fixture()
    raw["quantization_config"]["config_groups"]["group_0"]["input_activations"][field] = value
    with pytest.raises(ValueError):
        validate_headers(raw,headers)


@pytest.mark.parametrize("extra", ["model.language_model.layers.0.self_attn.v_norm.weight",
    "model.language_model.layers.0.experts.4.gate_proj.weight_packed","lm_head.weight"])
def test_unknown_or_duplicate_text_authority_refused_even_without_vision(extra):
    raw,headers = fixture()
    headers[extra] = {"dtype":"BF16","shape":[64]}
    with pytest.raises(ValueError,match="unconsumed"):
        validate_headers(raw,headers,vision=False)


def test_metadata_validation_does_not_mutate_config_or_inventory():
    raw,headers = fixture()
    before = deepcopy((raw,headers))
    validate_headers(raw,headers)
    assert (raw,headers) == before

"""Host-only checkpoint layout and default-RoPE contracts, with no model SDK."""

import json
import subprocess
import sys
from copy import deepcopy
from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from tensorfold.families.qwen3_5.config_metadata import (
    DefaultRope,
    resolve_default_rope,
    resolve_layer_types,
    validate_cuda_metadata,
)


def test_explicit_layout_is_authoritative_over_the_interval_hint():
    kinds = ["full_attention", "linear_attention", "full_attention", "full_attention"]
    text = {"layer_types": kinds, "full_attention_interval": 0}
    assert resolve_layer_types(text, layers=4) == tuple(kinds)
    assert text["layer_types"] is kinds


@pytest.mark.parametrize("text", [{}, {"layer_types": None}, {"full_attention_interval": 4}])
def test_missing_layout_retains_arithmetic_interval_dispatch(text):
    assert resolve_layer_types(text, layers=6) == ()
    assert resolve_layer_types(text, layers=0) == ()
    assert resolve_layer_types(text, layers=10**100) == ()


def test_declared_mtp_tail_is_separate_from_the_main_decoder():
    main = ["linear_attention", "full_attention"]
    text = {"layer_types": main + ["full_attention"], "num_nextn_predict_layers": 1}
    original = deepcopy(text)
    assert resolve_layer_types(text, layers=2) == tuple(main)
    assert text == original
    text["layer_types"] = main
    assert resolve_layer_types(text, layers=2) == tuple(main)


@pytest.mark.parametrize("text,layers", [
    ({"layer_types": "linear_attention"}, 1),
    ({"layer_types": ["full_attention"]}, 2),
    ({"layer_types": ["full_attention"] * 3}, 2),
    ({"layer_types": ["full_attention"] * 4, "num_nextn_predict_layers": 1}, 2),
    ({"layer_types": ["full_attention"] * 3, "num_nextn_predict_layers": -1}, 2),
    ({"layer_types": ["full_attention"] * 3, "num_nextn_predict_layers": True}, 2),
    ({"layer_types": ["linear_attention", "sparse_attention"]}, 2),
    ({"layer_types": ["linear_attention", "full_attention", None], "num_nextn_predict_layers": 1}, 2),
    ({"full_attention_interval": 0}, 2),
    ({"full_attention_interval": 1.5}, 2),
    ({"full_attention_interval": True}, 2),
    ({}, -1),
    ({}, 2.0),
])
def test_invalid_layout_declarations_fail(text, layers):
    with pytest.raises(ValueError):
        resolve_layer_types(text, layers=layers)


def test_current_checkpoint_default_metadata_preserves_its_math():
    # Relevant fields of surogate/Qwen3.5-9B-NVFP4, revision3410fbc4f407...
    # Its declared Transformers5.2.0 selects this same unscaled formula.
    text = {
        "num_hidden_layers": 32,
        "head_dim": 256,
        "layer_types": ["linear_attention"] * 3 + ["full_attention"],
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000,
                            "partial_rotary_factor": .25, "mrope_section": [11, 11, 10],
                            "mrope_interleaved": True},
    }
    text["layer_types"] *= 8
    before = deepcopy(text)
    assert resolve_layer_types(text, layers=32) == tuple(text["layer_types"])
    assert resolve_default_rope(text, head_dim=256) == DefaultRope(10000000.0, 64, (11, 11, 10))
    validate_cuda_metadata({"text_config": text})
    assert text == before


@pytest.mark.parametrize("text", [{}, {"rope_parameters": None}, {"rope_parameters": {}},
                                   {"rope_scaling": {"type": "default"}},
                                   {"rope_parameters": {"rope_type": "default", "factor": 4.0}}])
def test_default_policy_and_legacy_spelling_do_not_enable_scaling(text):
    assert resolve_default_rope(text, head_dim=256) == DefaultRope(10000000.0, 64, (11, 11, 10))


def test_nested_rope_values_precede_top_level_legacy_values():
    text = {"rope_theta": 12345, "partial_rotary_factor": 1,
            "rope_parameters": {"rope_theta": 10000, "partial_rotary_factor": .5,
                                "mrope_section": [3, 5, 8]}}
    assert resolve_default_rope(text, head_dim=64) == DefaultRope(10000.0, 32, (3, 5, 8))


def test_rotary_dimension_keeps_the_pinned_integer_truncation():
    # int(128 * .3) is38, and the default initializer uses that dimension.
    rope = resolve_default_rope({"partial_rotary_factor": .3}, head_dim=128)
    assert rope.rotary_dim == 38
    assert resolve_default_rope({"partial_rotary_factor": 0}, head_dim=128).rotary_dim == 0
    with pytest.raises(FrozenInstanceError):
        rope.theta = 5


def test_text_metadata_does_not_enable_or_require_multimodal_rotary():
    text = {"rope_parameters": {"partial_rotary_factor": .5, "mrope_section": [3, 2, 3]}}
    assert resolve_default_rope(text, head_dim=64) == DefaultRope(10000000.0, 32, (3, 2, 3))


def test_redundant_modern_legacy_policies_must_have_identical_effective_math():
    text = {"rope_parameters": {"rope_type": "default", "rope_theta": 10000},
            "rope_scaling": {"type": "default", "rope_theta": 10000.0}}
    assert resolve_default_rope(text, head_dim=256).theta == 10000
    text["rope_scaling"]["rope_theta"] = 500000
    with pytest.raises(ValueError, match="rope_parameters and rope_scaling disagree"):
        resolve_default_rope(text, head_dim=256)


@pytest.mark.parametrize("kind", ["linear", "dynamic", "yarn", "longrope", "llama3", None, False])
@pytest.mark.parametrize("field", ["rope_parameters", "rope_scaling"])
def test_nondefault_algorithms_are_refused_before_any_model_execution(kind, field):
    text = {field: {"rope_type": kind}}
    with pytest.raises(ValueError, match="default text RoPE only"):
        resolve_default_rope(text, head_dim=256)
    # Format probes without dimensions must still reject unsupported math.
    with pytest.raises(ValueError, match="default text RoPE only"):
        validate_cuda_metadata({"text_config": text})


@pytest.mark.parametrize("text", [
    {"rope_parameters": []},
    {"rope_scaling": "default"},
    {"rope_parameters": {"rope_type": "default", "type": "linear"}},
    {"rope_parameters": {"full_attention": {"rope_type": "default"}}},
    {"rope_parameters": {"linear_attention": {"rope_type": "default"}}},
    {"rope_theta": 0},
    {"rope_theta": True},
    {"rope_theta": float("inf")},
    {"rope_theta": float("nan")},
    {"rope_theta": 10**400},
    {"partial_rotary_factor": -0.25},
    {"partial_rotary_factor": 1.01},
    {"partial_rotary_factor": True},
    {"partial_rotary_factor": "0.25"},
    {"partial_rotary_factor": 3 / 256},
    {"rope_parameters": {"mrope_section": [11, 11]}},
    {"rope_parameters": {"mrope_section": [12, 21, -1]}},
    {"rope_parameters": {"mrope_section": [True, 15, 16]}},
])
def test_invalid_default_parameters_are_refused(text):
    with pytest.raises(ValueError):
        resolve_default_rope(text, head_dim=256)


@pytest.mark.parametrize("head_dim", [-1, True, 256.0, 2**31])
def test_rope_dimension_boundary_is_validated(head_dim):
    with pytest.raises(ValueError):
        resolve_default_rope({}, head_dim=head_dim)


def test_early_check_accepts_format_only_probes_and_complete_normalized_configs():
    for raw in ({"quantization_config": {"quant_method": "modelopt"}},
                {"text_config": {"quantization_config": {"quant_method": "exl3"}}},
                {"num_hidden_layers": 2.0, "hidden_size": 256, "num_attention_heads": 4},
                {"head_dim": 0, "hidden_size": 256, "num_attention_heads": 4},
                {"rope_parameters": {"rope_type": "default"}}):
        before = deepcopy(raw)
        validate_cuda_metadata(raw)
        assert raw == before


@pytest.mark.parametrize("raw", [None, [], {"text_config": []},
                                 {"layer_types": ["full_attention"]},
                                 {"full_attention_interval": 0},
                                 {"head_dim": 256, "partial_rotary_factor": 3 / 256},
                                 {"num_hidden_layers": None},
                                 {"head_dim": [256]},
                                 {"hidden_size": None, "num_attention_heads": 4},
                                 {"hidden_size": 256, "num_attention_heads": 0}])
def test_early_check_rejects_bad_declared_shapes(raw):
    with pytest.raises(ValueError):
        validate_cuda_metadata(raw)


@pytest.mark.parametrize("raw", [{"hidden_act": "gelu"}, {"hidden_act": None},
                                 {"attention_bias": True}, {"attention_bias": 1},
                                 {"attention_bias": "false"}, {"attention_bias": None}])
def test_early_check_refuses_recognized_math_the_cuda_kernels_do_not_implement(raw):
    with pytest.raises(ValueError):
        validate_cuda_metadata({"text_config": raw})


def test_supported_activation_and_attention_bias_defaults_are_explicit():
    validate_cuda_metadata({"hidden_act": "silu", "attention_bias": False})


@pytest.mark.parametrize("nested", [None, {}])
def test_empty_or_null_text_config_keeps_existing_outer_config_semantics(nested):
    validate_cuda_metadata({"text_config": nested, "num_hidden_layers": 4, "head_dim": 256})
    with pytest.raises(ValueError, match="hidden_act"):
        validate_cuda_metadata({"text_config": nested, "hidden_act": "gelu"})


def test_cuda_preflight_does_not_impose_its_math_limits_on_other_backends():
    from tensorfold.families.qwen3_5.config_metadata import check_metadata

    raw = {"hidden_act": "gelu", "rope_parameters": {"rope_type": "linear", "factor": 2}}
    check_metadata(raw, "mlx")
    with pytest.raises(ValueError, match="hidden_act"):
        check_metadata(raw, "cuda")


def test_early_interval_validation_does_not_expand_an_untrusted_scalar_count():
    # Resource fitting owns admission; this pure preflight stays constant-space.
    validate_cuda_metadata({"num_hidden_layers": 10**100, "head_dim": 256})


def test_active_default_rope_requires_finite_fp32_inverse_frequencies():
    text = {"head_dim": 256, "num_hidden_layers": 4, "rope_theta": 1e-100}
    with pytest.raises(ValueError, match="inverse frequencies exceed finite float32"):
        validate_cuda_metadata(text)
    # A base below one is supported when the actual stored frequencies fit.
    validate_cuda_metadata(dict(text, rope_theta=.01))
    validate_cuda_metadata(dict(text, partial_rotary_factor=0))
    validate_cuda_metadata(dict(text, partial_rotary_factor=2 / 256))
    # Main decoder layers without attention never consume these frequencies.
    validate_cuda_metadata(dict(text, num_hidden_layers=3))
    with pytest.raises(ValueError, match="inverse frequencies exceed finite float32"):
        validate_cuda_metadata(dict(text, num_hidden_layers=1, layer_types=["full_attention"]))


def test_import_and_early_probe_never_load_an_accelerator_runtime():
    source = Path(__file__).resolve().parents[1] / "src"
    program = f"""
import importlib.abc
import sys
sys.path.insert(0, {str(source)!r})
class RefuseRuntime(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {{'torch', 'mlx', 'triton', 'numpy', 'cuda', 'ctypes'}}:
            raise RuntimeError('runtime import: ' + fullname)
sys.meta_path.insert(0, RefuseRuntime())
from tensorfold.families.qwen3_5.config_metadata import validate_cuda_metadata
validate_cuda_metadata({json.dumps({'head_dim': 256, 'num_hidden_layers': 4})})
"""
    completed = subprocess.run([sys.executable, "-I", "-c", program], check=False, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

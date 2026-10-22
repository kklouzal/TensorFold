"""The NVFP4 / FP8 reader against an independent dequantizer (torch's float8 types and the e2m1 value list)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.cuda.nvfp4 import format as fmt

E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def _codes(rng, n, k):
    return rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)


@pytest.mark.torch
def test_nvfp4_matches_torch_float8_and_the_e2m1_list():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(1)
    n, k = 5, 64
    packed = _codes(rng, n, k)
    scale = rng.integers(0, 0x7F, size=(n, k // 16), dtype=np.uint8)          # finite positive e4m3
    got = fmt.dequant("nvfp4", packed, scale, 0.0625)
    bs = torch.from_numpy(scale).view(torch.float8_e4m3fn).float().numpy()
    want = np.empty((n, k), dtype=np.float32)
    for r in range(n):
        for c in range(k):
            byte = int(packed[r, c // 2])
            code = byte & 0xF if c % 2 == 0 else byte >> 4
            value = E2M1_VALUES[code & 7] * (-1 if code & 8 else 1)
            want[r, c] = value * bs[r, c // 16] * 0.0625
    assert np.array_equal(got, want)


@pytest.mark.torch
def test_fp8_and_mxfp8_match_torch_float8():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(2)
    w = rng.integers(0, 256, size=(4, 64), dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] = 0x10                                                 # no NaN bytes
    e4 = torch.from_numpy(w).view(torch.float8_e4m3fn).float().numpy()
    assert np.array_equal(fmt.dequant("fp8", w, np.array([0.25], dtype=np.float32)), e4 * np.float32(0.25))
    s = rng.integers(100, 140, size=(4, 2), dtype=np.uint8)
    e8 = torch.from_numpy(s).view(torch.float8_e8m0fnu).float().numpy()
    assert np.array_equal(fmt.dequant("mxfp8", w, s), e4 * np.repeat(e8, 32, axis=1))


def test_compressed_tensors_global_scale_is_a_reciprocal():
    packed = np.array([[0x21, 0x07]], dtype=np.uint8)                          # codes 1, 2, 7, 0
    scale = np.full((1, 1), 0x38, dtype=np.uint8)                              # e4m3 1.0
    a = fmt.dequant("nvfp4", np.pad(packed, ((0, 0), (0, 6))), scale, 4.0, reciprocal_global=True)
    assert list(a[0, :4]) == [0.125, 0.25, 1.5, 0.0]


def test_schemes_from_tensor_storage():
    assert fmt.scheme({"weight": ("U8", [8, 32]), "weight_scale": ("F8_E4M3", [8, 4]),
                       "weight_scale_2": ("F32", [])}) == "nvfp4"
    assert fmt.scheme({"weight_packed": ("U8", [8, 32]), "weight_scale": ("F8_E4M3", [8, 4]),
                       "weight_global_scale": ("F32", [1])}) == "nvfp4"
    assert fmt.scheme({"weight": ("F8_E4M3", [8, 64]), "weight_scale": ("F32", [])}) == "fp8"
    assert fmt.scheme({"weight": ("F8_E4M3", [8, 64]), "weight_scale": ("U8", [8, 2])}) == "mxfp8"
    assert fmt.scheme({"weight": ("BF16", [8, 64])}) == "bf16"


def test_config_gate_takes_nvfp4_and_fp8_and_refuses_integer_weights():
    kw = dict(where="CUDA", tested="x", help="")
    ok = {"quantization_config": {"quant_method": "modelopt", "config_groups": {
        "a": {"weights": {"num_bits": 8, "type": "float"}}, "b": {"weights": {"num_bits": 4, "type": "float",
                                                                               "group_size": 16}}}}}
    fmt.require_config(ok, **kw)
    assert fmt.config_block(ok)["quant_method"] == "modelopt"
    ct = {"quantization_config": {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
                                  "config_groups": {"g": {"weights": {"num_bits": 4, "type": "float"}}}}}
    fmt.require_config(ct, **kw)
    awq = {"quantization_config": {"quant_method": "compressed-tensors", "format": "pack-quantized",
                                   "config_groups": {"g": {"weights": {"num_bits": 4, "type": "int"}}}}}
    with pytest.raises(ValueError, match="4-bit int"):
        fmt.require_config(awq, **kw)


@pytest.mark.torch
def test_block_fp8_matches_torch_float8_and_its_block_scales():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(3)
    w = rng.integers(0, 256, size=(200, 256), dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] = 0x10
    s = rng.random((2, 2)).astype(np.float32) + 0.5                                # [ceil(200/128), 256/128]
    e4 = torch.from_numpy(w).view(torch.float8_e4m3fn).float().numpy()
    want = e4 * np.repeat(np.repeat(s, 128, axis=0)[:200], 128, axis=1)
    assert np.array_equal(fmt.dequant("fp8block", w, s), want)


def test_block_fp8_scheme_from_tensor_storage():
    assert fmt.scheme({"weight": ("F8_E4M3", [200, 256]), "weight_scale_inv": ("F32", [2, 2])}) == "fp8block"
    assert fmt.scheme({"weight": ("F8_E4M3", [200, 256]), "weight_scale": ("F32", [])}) == "fp8"


def _projection(*, reciprocal=False, activation=True):
    value = {"weight": ("U8", [8, 32]), "weight_scale": ("F8_E4M3", [8, 4]),
             "weight_global_scale" if reciprocal else "weight_scale_2": ("F32", [])}
    if activation:
        value["input_global_scale" if reciprocal else "input_scale"] = ("F32", [1])
    return value


def _metadata(*, activation=True, method="modelopt", dynamic=False):
    return fmt.LinearMetadata({"quant_method": method, "config_groups": {"g": {
        "targets": ["Linear"], "weights": {"type": "float", "num_bits": 4, "group_size": 16},
        "input_activations": {"type": "float", "num_bits": 4, "group_size": 16, "dynamic": dynamic}
        if activation else None}}})


@pytest.mark.parametrize("method,dynamic", [("modelopt", False), ("compressed-tensors", "local")])
def test_activation_math_requires_metadata_not_presence_of_static_scales(method, dynamic):
    headers = _projection(reciprocal=method == "compressed-tensors")
    assert _metadata(method=method, dynamic=dynamic).validate("mlp", headers, checkpoint=True) == ("nvfp4", 4)
    assert _metadata(method=method, activation=False).validate("mlp", headers, checkpoint=True) == ("nvfp4", None)
    assert _metadata(method=method, dynamic=dynamic).validate("mlp", headers, checkpoint=False) == ("nvfp4", None)


def test_borrowed_validation_observes_current_default_mutation_and_policy_replacement():
    policy = _metadata()
    headers = _projection()
    assert policy.validate("mlp", headers, checkpoint=True) == ("nvfp4", 4)
    policy.policy.defaults[0]["input_activations"] = None
    assert policy.validate("mlp", headers, checkpoint=True) == ("nvfp4", None)
    replacement = _metadata().policy
    policy.policy = replacement
    assert policy.validate("mlp", headers, checkpoint=True) == ("nvfp4", 4)
    replacement.defaults[0]["weights"]["num_bits"] = 8
    with pytest.raises(ValueError, match="stored nvfp4 weights disagree"):
        policy.validate("mlp", headers, checkpoint=True)


@pytest.mark.parametrize("checkpoint", [False, True])
@pytest.mark.parametrize("damage", [None, "activation", "weight", "output", "missing_scale", "shape"])
def test_borrowed_validation_matches_independent_public_policy_result_and_error(checkpoint, damage):
    class PublicPolicy(fmt.LinearQuantization):
        """The unchanged public resolver is the project-owned declaration oracle."""

    candidate, reference = _metadata(), _metadata()
    reference.policy = PublicPolicy({"quant_method": "modelopt", "config_groups": {"g": {
        "targets": ["Linear"], "weights": {"type": "float", "num_bits": 4, "group_size": 16},
        "input_activations": {"type": "float", "num_bits": 4, "group_size": 16, "dynamic": False}}}})
    headers = _projection()
    for metadata in (candidate, reference):
        group = metadata.policy.defaults[0]
        if damage == "activation":
            group["input_activations"]["dynamic"] = True
        elif damage == "weight":
            group["weights"]["group_size"] = 32
        elif damage == "output":
            group["output_activations"] = {"num_bits": 8}
    if damage == "missing_scale":
        del headers["input_scale"]
    elif damage == "shape":
        headers["weight"] = ("U8", [0, 32])
    try:
        expected = reference.validate("mlp", headers, checkpoint=checkpoint)
    except ValueError as error:
        with pytest.raises(ValueError) as actual:
            candidate.validate("mlp", headers, checkpoint=checkpoint)
        assert actual.value.args == error.args
    else:
        actual = candidate.validate("mlp", headers, checkpoint=checkpoint)
        assert actual == expected
        assert isinstance(actual[0], str) and (actual[1] is None or type(actual[1]) is int)


def test_metadata_validation_keeps_replaced_public_resolver_authoritative(monkeypatch):
    metadata, calls = _metadata(), []

    def resolve(name):
        calls.append(name)
        return {"weights": {"type": "float", "num_bits": 4, "group_size": 16}, "input_activations": None}

    monkeypatch.setattr(metadata.policy, "resolve", resolve)
    assert metadata.validate("mlp", _projection(), checkpoint=True) == ("nvfp4", None)
    assert calls == ["mlp"]


@pytest.mark.parametrize("field,value", [
    ("input_scale", ("F32", [2])), ("input_scale", ("I32", [])),
    ("weight_scale_2", ("F32", [2])), ("weight_scale", ("F8_E4M3", [8, 3])),
    ("weight", ("U8", [8, 32, 1])), ("weight", ("U8", [True, 32])),
    ("weight_packed", ("U8", [8, 32]))])
def test_declared_storage_and_calibrations_are_cross_checked(field, value):
    headers = _projection(); headers[field] = value
    with pytest.raises(ValueError):
        _metadata().validate("mlp", headers, checkpoint=True)


def test_missing_or_conflicting_static_activation_calibration_is_explicit():
    headers = _projection(activation=False)
    with pytest.raises(ValueError, match="static input calibration"):
        _metadata().validate("mlp", headers, checkpoint=True)
    assert _metadata().validate("mlp", headers, checkpoint=False) == ("nvfp4", None)
    headers = _projection(); headers["input_global_scale"] = ("F32", [])
    with pytest.raises(ValueError, match="one static"):
        _metadata().validate("mlp", headers, checkpoint=True)


def test_weight_storage_disagreement_and_excluded_quantized_weights_are_refused():
    policy = _metadata()
    with pytest.raises(ValueError, match="disagree"):
        policy.validate("mlp", {"weight": ("BF16", [8, 64])}, checkpoint=True)
    excluded = fmt.LinearMetadata({"quant_method": "modelopt", "ignore": ["mlp"]})
    for full in (False, True):
        with pytest.raises(ValueError, match="no matching"):
            excluded.validate("mlp", _projection(), checkpoint=full)


@pytest.mark.parametrize("dynamic", [True, "local"])
def test_fp8_native_math_refuses_dynamic_nonstatic_scheme(dynamic):
    metadata = fmt.LinearMetadata({"quant_method": "compressed-tensors", "config_groups": {"g": {
        "weights": {"num_bits": 8, "type": "float"},
        "input_activations": {"num_bits": 8, "type": "float", "dynamic": dynamic}}}})
    header = {"weight": ("F8_E4M3", [8, 64]), "weight_scale": ("F32", []), "input_scale": ("F32", [])}
    with pytest.raises(ValueError, match="unsupported checkpoint activation"):
        metadata.validate("mlp", header, checkpoint=True)
    assert metadata.validate("mlp", header, checkpoint=False) == ("fp8", None)


@pytest.mark.parametrize("value", [0., -0., -1., float("inf"), float("nan"), 1e-100, 1e100])
@pytest.mark.parametrize("reciprocal", [False, True])
def test_calibration_rejects_invalid_or_nonrepresentable_fp32_values(value, reciprocal):
    with pytest.raises(ValueError, match="calibration"):
        fmt.calibration(value, name="projection.input", reciprocal=reciprocal)


def test_calibration_preserves_original_scaling_and_reciprocal_convention():
    assert fmt.calibration(.25, name="input") == .25
    assert fmt.calibration(4., name="input_global_scale", reciprocal=True) == .25


def test_output_quantization_is_not_lost_when_input_activation_is_absent():
    policy = fmt.LinearMetadata({"quant_method": "modelopt", "config_groups": {"g": {
        "weights": {"num_bits": 4, "type": "float", "group_size": 16},
        "input_activations": None, "output_activations": {"num_bits": 8, "type": "float"}}}})
    with pytest.raises(ValueError, match="output quantization"):
        policy.validate("mlp", _projection(), checkpoint=True)
    assert policy.validate("mlp", _projection(), checkpoint=False) == ("nvfp4", None)


@pytest.mark.parametrize("key,value", [("scale_dtype", "torch.float32"), ("strategy", "channel"),
                                       ("symmetric", False), ("dynamic", True)])
def test_weight_scale_representation_metadata_is_not_silently_ignored(key, value):
    weights = {"num_bits": 4, "type": "float", "group_size": 16, key: value}
    policy = fmt.LinearMetadata({"quant_method": "modelopt", "config_groups": {"g": {
        "weights": weights, "input_activations": None}}})
    with pytest.raises(ValueError):
        policy.validate("mlp", _projection(), checkpoint=False)


@pytest.mark.parametrize("additional", [{"dynamic": True}, {"group_size": 32},
                                        {"scale_dtype": "torch.float32"}, {"strategy": "channel"}])
@pytest.mark.parametrize("reverse", [False, True])
def test_constraints_in_later_matching_groups_reach_native_refusal_before_payload(additional, reverse):
    first = {"targets": ["Linear"], "weights": {"num_bits": 4, "type": "float"}}
    second = {"targets": ["Linear"], "weights": additional}
    records = [second, first] if reverse else [first, second]
    policy = fmt.LinearMetadata({"quant_method": "compressed-tensors",
                                "config_groups": dict(zip(("a", "b"), records))})
    # Only immutable storage headers are supplied; no payload exists to read.
    with pytest.raises(ValueError):
        policy.validate("mlp", _projection(reciprocal=True), checkpoint=False)


@pytest.mark.torch
@pytest.mark.parametrize("act,weight", [(1e-40, 1.), (1e30, 1e30), (1e-30, 1e-30)])
def test_native_calibration_refuses_actual_fp32_inverse_or_product_exhaustion(act, weight):
    torch = pytest.importorskip("torch")
    inverse = torch.tensor(1., dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)
    product = torch.tensor(act, dtype=torch.float32) * torch.tensor(weight, dtype=torch.float32)
    assert not (torch.isfinite(inverse) and inverse > 0 and torch.isfinite(product) and product > 0)
    with pytest.raises(ValueError, match="native FP32"):
        fmt.native_calibration(act, weight, name="mlp")


@pytest.mark.torch
@pytest.mark.parametrize("act,weight", [(1., .5), (.037, .13), (1e-30, 1e-8)])
def test_native_calibration_keeps_admissible_scale_values_without_reassociation(act, weight):
    torch = pytest.importorskip("torch")
    inverse = torch.tensor(1., dtype=torch.float32) / torch.tensor(act, dtype=torch.float32)
    product = torch.tensor(act, dtype=torch.float32) * torch.tensor(weight, dtype=torch.float32)
    assert torch.isfinite(inverse) and inverse > 0 and torch.isfinite(product) and product > 0
    assert fmt.native_calibration(act, weight, name="mlp") == act


# Per-channel FP8 weights with dynamic per-token activations (ModelOpt FP8_PER_CHANNEL_PER_TOKEN, as the AEON MIXED
# Qwen3.8-27B export stores them): an F32 scale an output channel, no input calibration.
def _channel(n=8, k=64, *, scale_shape=None, input_scale=False):
    value = {"weight": ("F8_E4M3", [n, k]), "weight_scale": ("F32", [n] if scale_shape is None else scale_shape)}
    if input_scale:
        value["input_scale"] = ("F32", [])
    return value


def _per_token_policy(activation=None, *, strategy="channel", algo="FP8_PER_CHANNEL_PER_TOKEN"):
    from tensorfold.checkpoint_quantization import effective_config

    group = {"targets": ["mlp"], "weights": {"dynamic": False, "num_bits": 8, "type": "float", "strategy": strategy},
             "input_activations": activation or {"dynamic": False, "num_bits": 8, "type": "float"}}
    raw = {"quantization_config": {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION",
                                   "quantized_layers": {"mlp": {"quant_algo": algo}}, "config_groups": {"g": group}}}
    return fmt.LinearMetadata(effective_config(raw, None, where="test")["quantization_config"])


def test_channel_scales_are_their_own_storage_scheme():
    assert fmt.scheme(_channel()) == "fp8channel"
    assert fmt.scheme(_channel(scale_shape=[8, 1])) == "fp8channel"
    assert fmt.scheme({"weight": ("F8_E4M3", [8, 64]), "weight_scale": ("BF16", [8])}) == "fp8channel"
    assert fmt.scheme(_channel(scale_shape=[])) == "fp8"
    assert fmt.scheme(_channel(scale_shape=[1])) == "fp8"
    assert fmt.scheme(_channel(scale_shape=[7])) == "fp8"                          # not one a channel: refused below
    assert fmt.scheme(_channel(n=1, scale_shape=[1])) == "fp8"                     # one channel is one tensor scale


def test_per_token_activations_are_declared_math_without_calibration():
    policy = _per_token_policy()                                                  # ModelOpt 0.46.0's record
    assert policy.validate("mlp", _channel(), checkpoint=True) == ("fp8channel", 8)
    assert policy.validate("mlp", _channel(scale_shape=[8, 1]), checkpoint=True) == ("fp8channel", 8)
    assert policy.validate("mlp", _channel(), checkpoint=False) == ("fp8channel", None)
    later = _per_token_policy({"dynamic": True, "num_bits": 8, "type": "float", "strategy": "token"})
    assert later.validate("mlp", _channel(), checkpoint=True) == ("fp8channel", 8)  # later ModelOpt's record
    with pytest.raises(ValueError, match="no static input calibration"):
        policy.validate("mlp", _channel(input_scale=True), checkpoint=True)


@pytest.mark.parametrize("headers", [_channel(scale_shape=[]), _channel(scale_shape=[7]),
                                     {"weight": ("F8_E4M3", [8, 64]), "weight_scale": ("F32", [8, 2])}])
def test_channel_declarations_need_one_scale_a_channel(headers):
    with pytest.raises(ValueError):
        _per_token_policy().validate("mlp", headers, checkpoint=False)


def test_tensor_scale_declarations_refuse_channel_storage():
    policy = fmt.LinearMetadata({"quant_method": "modelopt", "quant_algo": "FP8", "config_groups": {"g": {
        "targets": ["Linear"], "weights": {"num_bits": 8, "type": "float"},
        "input_activations": {"num_bits": 8, "type": "float"}}}})
    with pytest.raises(ValueError, match="strategy"):
        policy.validate("mlp", _channel(input_scale=True), checkpoint=False)


@pytest.mark.parametrize("activation", [{"dynamic": True, "num_bits": 8, "type": "float"},
                                        {"dynamic": True, "num_bits": 8, "type": "float", "strategy": "tensor"},
                                        {"dynamic": True, "num_bits": 8, "type": "float", "strategy": "group",
                                         "group_size": 128},
                                        {"dynamic": True, "num_bits": 8, "type": "float", "strategy": "token",
                                         "symmetric": False}])
def test_other_dynamic_fp8_activation_records_are_refused_in_checkpoint_math(activation):
    policy = fmt.LinearMetadata({"quant_method": "compressed-tensors", "config_groups": {"g": {
        "targets": ["Linear"], "weights": {"num_bits": 8, "type": "float", "strategy": "channel"},
        "input_activations": activation}}})
    with pytest.raises(ValueError, match="unsupported checkpoint activation"):
        policy.validate("mlp", _channel(), checkpoint=True)
    assert policy.validate("mlp", _channel(), checkpoint=False) == ("fp8channel", None)


@pytest.mark.torch
def test_channel_fp8_matches_torch_float8_times_each_row_scale():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    w = rng.integers(0, 256, size=(6, 64), dtype=np.uint8)
    w[(w & 0x7F) == 0x7F] = 0x10
    s = (rng.random(6) + 0.25).astype(np.float32)
    e4 = torch.from_numpy(w).view(torch.float8_e4m3fn).float().numpy()
    assert np.array_equal(fmt.dequant("fp8channel", w, s), e4 * s[:, None])
    assert np.array_equal(fmt.dequant("fp8channel", w, s[:, None]), e4 * s[:, None])

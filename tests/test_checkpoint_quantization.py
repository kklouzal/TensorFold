"""Independent export-schema, boundary and activation-intent controls."""

import itertools
import re
from copy import deepcopy

import pytest

from tensorfold.checkpoint_quantization import LinearQuantization, effective_config


def side(algo="NVFP4", **kw):
    return {"producer": {"name": "modelopt", "version": "0.46.0"},
            "quantization": dict(quant_algo=algo, **kw)}


def test_sidecar_only_and_aliases_are_owned_without_changing_architecture():
    original = {"model_type": "qwen3_5", "text_config": {"hidden_size": 4096}, "eos_token_id": [1, 7]}
    ancillary = side(exclude_modules=["model.visual*", "lm_head"], kv_cache_quant_algo=None)
    before = deepcopy((original, ancillary))
    got = effective_config(original, ancillary, where="model")
    assert (original, ancillary) == before
    assert got["text_config"] == original["text_config"]
    assert got["eos_token_id"] == [1, 7]
    q = got["quantization_config"]
    assert q["quant_method"] == "modelopt" and q["quant_algo"] == "NVFP4"
    assert q["config_groups"]["group_0"]["input_activations"]["num_bits"] == 4
    assert q["ignore"] == ["model.visual*", "lm_head"] and q["kv_cache_scheme"] is None
    q["ignore"].clear()
    assert ancillary == before[1]


@pytest.mark.parametrize("location", ["quantization", "quantization_config", "text"])
def test_weight_only_algorithm_never_infers_activation_quantization(location):
    value = {"quant_method": "modelopt", "quant_algo": "W4A16_NVFP4", "group_size": 16}
    raw = {"text_config": {"quantization": value}} if location == "text" else {location: value}
    got = effective_config(raw, None, where="model")
    assert "quantization" not in got and "quantization" not in got.get("text_config", {})
    spec = LinearQuantization(got["quantization_config"]).resolve("model.layers.0.mlp.gate_proj")
    assert spec["weights"]["num_bits"] == 4 and spec["input_activations"] is None


def test_mixed_sidecar_keeps_exact_names_bank_descendants_and_exclusions():
    layers = {"model.layers.0.mlp.experts": {"quant_algo": "W4A16_NVFP4", "group_size": 16},
              "model.layers.0.self_attn.q_proj": {"quant_algo": "FP8"}}
    got = effective_config({}, side("MIXED_PRECISION", quantized_layers=layers,
                                   exclude_modules=["model.visual*"]), where="mixed")
    q = got["quantization_config"]
    assert q["quantized_layers"] == layers
    policy = LinearQuantization(q)
    assert policy.resolve("model.layers.0.mlp.experts.3.gate_proj")["input_activations"] is None
    assert policy.resolve("model.layers.0.self_attn.q_proj")["input_activations"]["num_bits"] == 8
    assert policy.resolve("model.visual.layers.0.self_attn.q_proj") is None
    assert policy.resolve("model.language_model.layers.0.self_attn.q_proj") is None


@pytest.mark.parametrize("raw,ancillary", [
    ({"quantization": {"bits": 4, "group_size": 32}}, side()),
    ({"quantization_config": {"quant_method": "compressed-tensors"}}, side()),
    ({"quantization_config": {"quant_method": "modelopt", "quant_algo": "W4A16_NVFP4"}}, side()),
    ({"quantization_config": {"quant_method": "modelopt", "ignore": ["a"]}}, side(exclude_modules=["b"])),
    ({"quantization_config": {"quant_method": "modelopt", "kv_cache_scheme": "FP8"}},
     side(kv_cache_quant_algo=None)),
    ({"quantization": {"bits": 4}, "text_config": {"quantization_config": {"bits": 8}}}, None),
])
def test_conflicts_are_errors_instead_of_metadata_precedence(raw, ancillary):
    with pytest.raises(ValueError, match="conflict"):
        effective_config(raw, ancillary, where="checkpoint")


@pytest.mark.parametrize("raw,ancillary", [([], None), ({"text_config": []}, None),
    ({"quantization": False}, None), ({}, {}), ({}, side("MYSTERY")),
    ({}, side("MIXED_PRECISION", quantized_layers=[])),
    ({}, side(exclude_modules="lm_head")), ({}, side(group_size=True))])
def test_malformed_unknown_and_boolean_counts_are_rejected(raw, ancillary):
    with pytest.raises(ValueError):
        effective_config(raw, ancillary, where="checkpoint")


def test_existing_affine_per_layer_aliases_and_unrelated_fields_are_preserved():
    q = {"bits": 4, "group_size": 64, "model.layers.0.mlp": {"bits": 8}, "lm_head": False}
    original = {"quantization": q, "custom": {"keep": [None, False, 0]}}
    got = effective_config(original, None, where="affine")
    assert got["custom"] == original["custom"]
    assert got["quantization_config"] == dict(q, quant_method="mlx")


def test_declared_modelopt_activation_conflict_is_not_filled_from_a_sidecar():
    group = {"weights": {"num_bits": 4, "type": "float", "group_size": 16},
             "input_activations": None, "targets": ["Linear"]}
    with pytest.raises(ValueError, match="conflict"):
        effective_config({"quantization_config": {"quant_method": "modelopt", "quant_algo": "NVFP4",
                                                   "config_groups": {"group_0": group}}}, None, where="x")


@pytest.mark.parametrize("pattern", ["re:(a+)+$", "re:(a|aa)+$", "re:(?=a)a", "re:(a)\\1", "re:(" * 5000])
def test_hostile_regex_is_rejected_before_matching(pattern):
    with pytest.raises(ValueError, match="pattern"):
        LinearQuantization({"ignore": [pattern]})


def test_safe_regex_subset_matches_independent_regex_for_all_small_names():
    from tensorfold.checkpoint_quantization import _Pattern

    patterns = ["re:a", "re:a$", "re:^a.*b$", "re:.*\\.mlp\\..*", "re:..*a.*b", "re:.*$", "re:."]
    names = ["".join(v) for n in range(5) for v in itertools.product("ab.\n", repeat=n)]
    names += ["model.layers.0.mlp.gate_proj", "vision.mlp.down_proj"]
    for pattern in patterns:
        match = _Pattern(pattern)
        for name in names:
            assert bool(match.match(name)) == bool(re.match(pattern[3:], name)), (pattern, name)


def test_partial_modelopt_groups_are_completed_from_declared_algorithm_only():
    got = effective_config({"quantization": {"quant_method": "modelopt", "quant_algo": "NVFP4",
                           "config_groups": {"group_0": {"weights": {"group_size": 16}}}}},
                           None, where="partial")
    group = got["quantization_config"]["config_groups"]["group_0"]
    assert group["weights"]["num_bits"] == 4
    assert group["input_activations"]["num_bits"] == 4
    ct = effective_config({"quantization_config": {"quant_method": "compressed-tensors",
                          "config_groups": {"g": {"weights": {"num_bits": 4}}}}}, None, where="ct")
    assert LinearQuantization(ct["quantization_config"]).resolve("mlp").get("input_activations") is None


def test_mixed_untargeted_inventory_never_assigns_all_formats_to_every_linear():
    layers = {"model.layers.0.mlp.experts": {"quant_algo": "NVFP4", "group_size": 16},
              "model.layers.0.attn": {"quant_algo": "MXFP8", "group_size": 32}}
    q = {"quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "quantized_layers": layers,
         "config_groups": {"mx": {"weights": {"num_bits": 8, "group_size": 32}},
                           "fp4": {"weights": {"num_bits": 4, "group_size": 16}}}}
    got = effective_config({"quantization_config": q}, None, where="inventory")
    assert got["quantization_config"]["config_groups"] == q["config_groups"]
    policy = LinearQuantization(got["quantization_config"])
    assert policy.resolve("model.layers.0.mlp.experts.7.gate_proj")["weights"]["num_bits"] == 4
    assert policy.resolve("model.layers.0.attn")["weights"]["group_size"] == 32
    assert policy.resolve("model.other") is None


def test_legacy_modelopt_kv_alias_is_reconciled_identically_to_the_sidecar():
    raw = {"quantization": {"quant_method": "modelopt", "quant_algo": "W4A16_NVFP4",
                            "kv_cache_quant_algo": "FP8"}}
    direct = effective_config(raw, None, where="direct")["quantization_config"]
    ancillary = effective_config({}, side("W4A16_NVFP4", kv_cache_quant_algo="FP8"),
                                 where="sidecar")["quantization_config"]
    assert direct["kv_cache_scheme"] == ancillary["kv_cache_scheme"]
    assert "kv_cache_quant_algo" not in direct


def test_bounded_glob_matcher_agrees_with_stdlib_over_all_small_patterns():
    import fnmatch

    from tensorfold.checkpoint_quantization import _Pattern

    names = ["".join(v) for n in range(5) for v in itertools.product("ab\n", repeat=n)]
    for size in range(5):
        for tokens in itertools.product("ab?*", repeat=size):
            pattern = "".join(tokens)
            matcher = _Pattern(pattern)
            for name in names:
                assert bool(matcher.match(name)) == fnmatch.fnmatchcase(name, pattern), (pattern, name)


@pytest.mark.parametrize("method,targets", [
    ("compressed-tensors", (["Linear"], ["Linear"])),
    ("compressed-tensors", (["model.mlp"], ["re:^model\\.mlp$"])),
    ("modelopt", (["model"], ["model.mlp"])),
])
@pytest.mark.parametrize("reverse", [False, True])
def test_every_matching_semantic_constraint_is_returned_as_an_independent_union(method, targets, reverse):
    first = {"targets": targets[0], "weights": {"num_bits": 4, "type": "float"},
             "input_activations": None}
    later = {"targets": targets[1], "weights": {"group_size": 32, "dynamic": True},
             "input_activations": None, "pre_quant_scale": True}
    records = [later, first] if reverse else [first, later]
    raw = {"quant_method": method, "config_groups": dict(zip(("a", "b"), records))}
    before = deepcopy(raw)
    policy = LinearQuantization(raw)
    got = policy.resolve("model.mlp")
    assert got == {"weights": {"num_bits": 4, "type": "float", "group_size": 32, "dynamic": True},
                   "input_activations": None, "output_activations": None, "pre_quant_scale": True}
    got["weights"].clear()
    assert raw == before
    assert policy.resolve("model.mlp")["weights"]["dynamic"] is True


def test_optional_group_format_inherits_global_before_matching_union():
    policy = LinearQuantization({"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
        "config_groups": {"a": {"weights": {"num_bits": 4}, "format": None},
                          "b": {"weights": {"num_bits": 4}, "format": "nvfp4-pack-quantized"}}})
    assert policy.resolve("mlp")["format"] == "nvfp4-pack-quantized"


def test_missing_activation_is_still_weight_only_and_conflicts_with_later_quantization():
    policy = LinearQuantization({"quant_method": "compressed-tensors", "config_groups": {
        "a": {"weights": {"num_bits": 4}},
        "b": {"weights": {"num_bits": 4}, "input_activations": {"num_bits": 4}}}})
    with pytest.raises(ValueError, match="conflicting"):
        policy.resolve("mlp")


def test_class_default_is_only_used_when_no_explicit_module_target_matches():
    policy = LinearQuantization({"quant_method": "compressed-tensors", "config_groups": {
        "default": {"targets": ["Linear"], "weights": {"num_bits": 8}},
        "specific": {"targets": ["model.mlp"], "weights": {"num_bits": 4}}}})
    assert policy.resolve("model.mlp")["weights"]["num_bits"] == 4
    assert policy.resolve("model.attn")["weights"]["num_bits"] == 8


def test_exclusions_cannot_silently_erase_explicit_layer_declarations():
    with pytest.raises(ValueError, match="exclusion"):
        LinearQuantization({"quant_method": "compressed-tensors", "ignore": ["model.*"],
                            "config_groups": {"g": {"targets": ["model.mlp"], "weights": {"num_bits": 4}}}})
    with pytest.raises(ValueError, match="exclusion"):
        effective_config({}, side("MIXED_PRECISION", exclude_modules=["model.*"], quantized_layers={
            "model.mlp": {"quant_algo": "NVFP4"}}), where="mixed")
    default = LinearQuantization({"quant_method": "compressed-tensors", "ignore": ["model.mlp"],
                                  "config_groups": {"g": {"targets": ["Linear"], "weights": {"num_bits": 4}}}})
    assert default.resolve("model.mlp") is None


PER_TOKEN = {"dynamic": True, "num_bits": 8, "type": "float", "strategy": "token"}
STATIC_046 = {"dynamic": False, "num_bits": 8, "type": "float"}             # ModelOpt 0.46.0's converter record
CHANNEL = {"dynamic": False, "num_bits": 8, "type": "float", "strategy": "channel"}


def _mixed(groups, layers):
    return effective_config({"quantization_config": {
        "quant_method": "modelopt", "quant_algo": "MIXED_PRECISION", "producer": {"name": "modelopt",
        "version": "0.46.0"}, "quantized_layers": layers, "config_groups": groups}}, None, where="mixed")


def test_per_channel_per_token_algorithm_declares_dynamic_token_activations():
    got = effective_config({}, side("FP8_PER_CHANNEL_PER_TOKEN"), where="sidecar")["quantization_config"]
    spec = LinearQuantization(got).resolve("model.layers.0.self_attn.q_proj")
    assert spec["weights"] == CHANNEL and spec["input_activations"] == PER_TOKEN
    recorded = effective_config({"quantization_config": {"quant_method": "modelopt",
                                 "quant_algo": "FP8_PER_CHANNEL_PER_TOKEN", "config_groups": {"g": {
                                     "targets": ["Linear"], "weights": CHANNEL,
                                     "input_activations": STATIC_046}}}}, None, where="single")
    assert LinearQuantization(recorded["quantization_config"]).resolve("x")["input_activations"] == PER_TOKEN


def test_mixed_0460_records_take_the_algorithms_per_token_activations_only_where_declared():
    layers = {"a.qkv": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"},
              "a.out": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}, "b.mlp": {"quant_algo": "NVFP4", "group_size": 16}}
    groups = {"g0": {"targets": ["a.qkv", "a.out"], "weights": CHANNEL, "input_activations": STATIC_046},
              "g1": {"targets": ["b.mlp"], "weights": {"dynamic": False, "num_bits": 4, "type": "float",
                                                       "group_size": 16},
                     "input_activations": {"dynamic": False, "num_bits": 4, "type": "float", "group_size": 16}}}
    q = _mixed(deepcopy(groups), layers)["quantization_config"]
    policy = LinearQuantization(q)
    assert policy.resolve("a.qkv")["input_activations"] == PER_TOKEN
    assert policy.resolve("a.out")["weights"] == CHANNEL
    assert policy.resolve("b.mlp")["input_activations"]["num_bits"] == 4
    current = deepcopy(groups)
    current["g0"]["input_activations"] = PER_TOKEN                             # later ModelOpt's record
    assert LinearQuantization(_mixed(current, layers)["quantization_config"]).resolve("a.qkv") == \
        policy.resolve("a.qkv")


def test_a_static_record_on_other_layers_stays_a_conflict():
    layers = {"a.qkv": {"quant_algo": "FP8_PER_CHANNEL_PER_TOKEN"}, "a.o": {"quant_algo": "FP8"}}
    groups = {"g0": {"targets": ["a.qkv", "a.o"], "weights": CHANNEL, "input_activations": STATIC_046}}
    with pytest.raises(ValueError, match="conflict"):
        LinearQuantization(_mixed(groups, layers)["quantization_config"])
    other = {"g0": {"targets": ["a.qkv"], "weights": CHANNEL,
                    "input_activations": {"dynamic": False, "num_bits": 8, "type": "float", "strategy": "tensor"}}}
    with pytest.raises(ValueError, match="conflict"):
        LinearQuantization(_mixed(other, {"a.qkv": layers["a.qkv"]})["quantization_config"])


@pytest.mark.parametrize("method", ["compressed-tensors", "modelopt"])
@pytest.mark.parametrize("targets", [["Linear"], ["model"], ["re:^model\\..*$"]])
@pytest.mark.parametrize("second", [None, {"group_size": 16}, {"num_bits": 8}])
def test_validation_resolution_preserves_public_union_errors_exclusions_and_names(method, targets, second):
    groups = {"first": {"targets": targets, "weights": {"type": "float", "num_bits": 4}, "format": None}}
    if second is not None:
        groups["second"] = {"targets": targets, "weights": second}
    policy = LinearQuantization({"quant_method": method, "format": "nvfp4-pack-quantized",
                                 "ignore": ["model.ignored", "vision*"], "config_groups": groups})
    for name in ("model", "model.mlp", "model.ignored", "vision.proj", "other"):
        try:
            expected = policy.resolve(name)
        except ValueError as error:
            with pytest.raises(ValueError) as actual:
                LinearQuantization._resolve_for_validation(policy, name)
            assert actual.value.args == error.args
        else:
            assert LinearQuantization._resolve_for_validation(policy, name) == expected


def test_validation_borrow_is_transient_while_public_results_remain_independent():
    policy = LinearQuantization({"quant_method": "compressed-tensors", "config_groups": {
        "g": {"weights": {"num_bits": 4}, "input_activations": {"num_bits": 4}}}})
    independent = policy.resolve("model.mlp")
    borrowed = LinearQuantization._resolve_for_validation(policy, "model.mlp")
    assert borrowed == independent
    assert borrowed["weights"] is policy.defaults[0]["weights"]
    del borrowed
    independent["weights"]["num_bits"] = 8
    assert LinearQuantization._resolve_for_validation(policy, "model.mlp")["weights"]["num_bits"] == 4
    policy.defaults[0]["input_activations"] = None
    assert LinearQuantization._resolve_for_validation(policy, "model.mlp")["input_activations"] is None


def test_validation_resolution_preserves_subclass_and_substituted_policy_dispatch():
    calls, result = [], {"weights": None}

    class Derived(LinearQuantization):
        def resolve(self, name):
            calls.append(("derived", name))
            return result

    class Other:
        def resolve(self, name):
            calls.append(("other", name))
            return result

    for policy, label in ((Derived({}), "derived"), (Other(), "other")):
        assert LinearQuantization._resolve_for_validation(policy, "mlp") is result
        assert calls[-1] == (label, "mlp")
    assert len(calls) == 2


def test_validation_resolution_preserves_class_instance_and_cross_bound_overrides(monkeypatch):
    policy = LinearQuantization({"config_groups": {"g": {"weights": {"num_bits": 4}}}})
    other = LinearQuantization({"config_groups": {"g": {"weights": {"num_bits": 8}}}})
    policy.resolve = other.resolve
    assert LinearQuantization._resolve_for_validation(policy, "mlp")["weights"]["num_bits"] == 8
    del policy.resolve

    class Resolver:
        def __getattribute__(self, name):
            if name == "__func__":
                raise AssertionError("a custom resolver's attribute hooks must not be inspected")
            return object.__getattribute__(self, name)

        def __call__(self, name):
            return {"name": name}

    policy.resolve = Resolver()
    assert LinearQuantization._resolve_for_validation(policy, "mlp") == {"name": "mlp"}
    del policy.resolve
    result = {"class_override": True}
    monkeypatch.setattr(LinearQuantization, "resolve", lambda self, name: result)
    assert LinearQuantization._resolve_for_validation(policy, "mlp") is result

"""Family positional identity/domain controls, without model/GPU qualification."""
from dataclasses import FrozenInstanceError
import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.position_profile import (
    DFLASH_ALGORITHM, MAX_EPOCH, NATIVE_ALGORITHM, YARN_ALGORITHM, resolve_position_profile,
    bind_position_profile, position_for, position_vector_bytes, prepare_position_vectors, mtp_source_parameters,
)


def config():
    return {"model_type": "qwen4_exp", "text_config": {
        "head_dim": 256, "hidden_size": 2560, "num_attention_heads": 24, "max_position_embeddings": 262144,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000000, "partial_rotary_factor": .25,
                            "mrope_interleaved": True, "mrope_section": [11, 11, 10]}}}


def put(root, raw):
    root.mkdir()
    path = root / "config.json"
    path.write_text(json.dumps(raw))
    return path


def role(profile, *, method="target", factor=1, limits=None):
    origin = profile.target if method == "target" else profile.mtp
    policy = dict(config()["text_config"]["rope_parameters"])
    if factor != 1:
        policy.update(rope_type="yarn", factor=factor, original_max_position_embeddings=262144)
    return {"source_config_sha256": origin.source_config_sha256, "rope": policy,
            "coordinate_limits": [math.floor(262144 * factor)] * 3 if limits is None else limits}


def manifest(root, base, *, rows=262144, factor=1, mtp=None, dflash=None, epoch=0):
    body = {"schema_version": 1, "family": "qwen4_exp", "epoch": epoch, "requested_context": rows,
            "qualification": "unqualified-candidate", "target": role(base, factor=factor),
            "mtp": mtp, "dflash2": dflash}
    path = root / "position.json"
    path.write_text(json.dumps(body))
    return path, body


def test_native_receipt_is_immutable_role_complete_and_preserves_exact_frequencies(tmp_path):
    source = put(tmp_path / "target", config())
    profile = resolve_position_profile(source.parent, requested_context=262144)
    assert profile.qualification == "unqualified-candidate"
    assert profile.target_attention is profile.target_indexer
    assert profile.mtp_attention is profile.mtp_indexer
    assert profile.target.algorithm == NATIVE_ALGORITHM
    assert profile.mtp.origin == "explicit-family-native-mtp-defaults"
    expected = (torch.tensor(10000000., dtype=torch.float64) ** (-torch.arange(32, dtype=torch.float64) / 32)).float()
    assert torch.equal(profile.target.rope.inverse_frequencies(torch), expected)
    assert profile.required_input_files() == (source.resolve(),)
    with pytest.raises(FrozenInstanceError):
        profile.target.rope.factor = 2
    receipt = profile.receipt()
    receipt["target_attention"]["rope"]["factor"] = 999
    assert profile.receipt()["target_attention"]["rope"]["factor"] == 1
    assert len(profile.receipt()["local_algorithm_sha256"]["rope.py"]) == 64


@pytest.mark.parametrize("factor,rows", [(2, 524288), (4, 1048576), (1.5, 393216), (2, 400001)])
def test_static_yarn_independent_formula_and_native_mtp_separation(tmp_path, factor, rows):
    source = put(tmp_path / "target", config())
    profile = resolve_position_profile(source.parent, requested_context=rows, yarn_factor=factor)
    policy = profile.target.rope
    # Literal pinned-HF formula, independent of RopeParameters' implementation.
    low = max(0, math.floor(64 * math.log(262144 / (32 * 2 * math.pi)) / (2 * math.log(10000000))))
    high = min(63, math.ceil(64 * math.log(262144 / (1 * 2 * math.pi)) / (2 * math.log(10000000))))
    if low == high:
        high += .001
    frequencies = 10000000 ** (torch.arange(0, 64, 2, dtype=torch.float32) / 64)
    ramp = ((torch.arange(32, dtype=torch.float32) - low) / (high - low)).clamp(0, 1)
    expected = (1 / (factor * frequencies)) * ramp + (1 / frequencies) * (1 - ramp)
    assert torch.equal(policy.inverse_frequencies(torch), expected)
    assert policy.attention_factor == 1 + .1 * math.log(factor)
    assert profile.target.max_rows == rows and policy.native_context == 262144
    assert profile.target.algorithm == YARN_ALGORITHM
    assert profile.mtp.rope.rope_type == "default" and profile.mtp.rope.attention_factor == 1
    with pytest.raises(ValueError, match="explicit MTP"):
        profile.validate_method("mtp", "actual-codec", 2)
    assert profile.validate_method("none", "actual-codec", 0)["qualification"] == "unqualified-candidate"


def test_explicit_companion_yarn_has_own_source_identity_without_target_override(tmp_path):
    target = put(tmp_path / "target", config())
    mtp_raw = config()
    mtp_raw["text_config"]["rope_parameters"].update(rope_type="yarn", factor=2,
                                                   original_max_position_embeddings=262144)
    companion = put(tmp_path / "mtp", mtp_raw)
    p = resolve_position_profile(target.parent, requested_context=524288, yarn_factor=2,
                                  companion_configs={"mtp": companion})
    assert p.mtp.rope.rope_type == "yarn" and p.mtp.origin == "actual-companion-with-explicit-source-native-defaults"
    assert p.mtp.source_config_sha256 != p.target.source_config_sha256
    assert set(p.required_input_files()) == {target.resolve(), companion.resolve()}
    p.validate_method("mtp", "q8-wire-v1", 2)


def test_manifest_extension_and_epoch_are_bound_to_actual_companion_inputs(tmp_path):
    target, companion = put(tmp_path / "target", config()), put(tmp_path / "mtp", config())
    base = resolve_position_profile(target.parent, requested_context=262144, companion_configs={"mtp": companion})
    path, body = manifest(tmp_path, base, rows=524288, factor=2, mtp=role(base, method="mtp", factor=2), epoch=MAX_EPOCH)
    p = resolve_position_profile(target.parent, requested_context=524288, manifest_path=path,
                                  companion_configs={"mtp": companion})
    assert p.epoch == MAX_EPOCH and p.target.rope == p.mtp.rope
    assert set(p.required_input_files()) == {target.resolve(), companion.resolve(), path.resolve()}
    p.validate_method("mtp", "iso4-wire-v1", 2)
    with pytest.raises(ValueError, match="actual selected companion"):
        resolve_position_profile(target.parent, requested_context=524288, manifest_path=path)
    with pytest.raises(ValueError, match="CLI YaRN"):
        resolve_position_profile(target.parent, requested_context=524288, manifest_path=path, yarn_factor=4,
                                 companion_configs={"mtp": companion})
    body["epoch"] = MAX_EPOCH + 1
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="epoch"):
        resolve_position_profile(target.parent, requested_context=524288, manifest_path=path,
                                 companion_configs={"mtp": companion})


@pytest.mark.parametrize("field,value", [("qualification", "qualified"), ("epoch", True),
                                        ("schema_version", True), ("requested_context", True)])
def test_manifest_cannot_issue_qualification_or_confuse_integer_domains(tmp_path, field, value):
    target = put(tmp_path / "target", config())
    base = resolve_position_profile(target.parent, requested_context=262144)
    path, body = manifest(tmp_path, base)
    body[field] = value
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        resolve_position_profile(target.parent, requested_context=262144, manifest_path=path)


def test_actual_rows_coordinates_and_lookahead_have_separate_exclusive_bounds(tmp_path):
    target = put(tmp_path / "target", config())
    p = resolve_position_profile(target.parent, requested_context=262144)
    p.target.validate_span(0, 0, coordinate_min=None, coordinate_max=None)
    p.target.validate_span(262140, 2, coordinate_min=(262140, 10, 20), coordinate_max=(262143, 12, 22), lookahead_rows=2)
    with pytest.raises(ValueError, match="live rows/lookahead"):
        p.target.validate_span(262140, 3, coordinate_min=(0, 0, 0), coordinate_max=(5, 5, 5), lookahead_rows=2)
    with pytest.raises(ValueError, match="coordinate"):
        p.target.validate_span(0, 1, coordinate_min=(0, 0, 0), coordinate_max=(262144, 2, 2))
    with pytest.raises(ValueError, match="coordinate minimum"):
        p.target.validate_span(0, 1, coordinate_min=(-1, 0, 0), coordinate_max=(1, 2, 2))
    with pytest.raises(ValueError, match="coordinate"):
        p.validate_method("mtp", "actual-codec", 2, media_bounds=(100, 262144, 100))


def test_dflash_uses_actual_scalar_algorithm_and_absolute_limit_not_ring_size(tmp_path):
    target = put(tmp_path / "target", config())
    df_raw = json.loads((Path(__file__).parent / "fixtures/dflash2/config.json").read_text())
    draft = put(tmp_path / "draft", df_raw)
    p = resolve_position_profile(target.parent, requested_context=262144, companion_configs={"dflash2": draft})
    assert p.dflash2.theta == 10000 and p.dflash2.rotary_dim == 128 and p.dflash2.window_rows == 2048
    assert p.dflash2.receipt()["algorithm"] == DFLASH_ALGORITHM
    assert p.dflash2.receipt()["ring_modulo_changes_coordinates"] is False
    p.dflash2.validate_span(0, 0, absolute_start=None)
    p.dflash2.validate_span(262140, 2, absolute_start=262140, lookahead_rows=2)
    with pytest.raises(ValueError, match="absolute positions"):
        p.dflash2.validate_span(0, 2, absolute_start=262143)
    p.validate_method("dflash2", "actual-codec", 4)
    assert draft.resolve() in p.required_input_files()
    native = resolve_position_profile(target.parent, requested_context=262144)
    with pytest.raises(ValueError, match="actual separate"):
        native.validate_method("dflash2", "actual-codec", 4)
    extended = resolve_position_profile(target.parent, requested_context=524288, yarn_factor=2,
                                        companion_configs={"dflash2": draft})
    with pytest.raises(ValueError, match="supported absolute range"):
        extended.validate_method("dflash2", "actual-codec", 4)


def test_source_mutation_and_explicit_geometry_conflicts_change_identity_or_refuse(tmp_path):
    target = put(tmp_path / "target", config())
    base = resolve_position_profile(target.parent, requested_context=262144)
    path, body = manifest(tmp_path, base)
    raw = config()
    raw["text_config"]["rope_parameters"]["rope_theta"] = 20000000
    target.write_text(json.dumps(raw))
    changed = resolve_position_profile(target.parent, requested_context=262144)
    assert base.identity != changed.identity
    with pytest.raises(ValueError, match="source configuration identity"):
        resolve_position_profile(target.parent, requested_context=262144, manifest_path=path)
    target.write_text(json.dumps(config()))
    body["target"]["rope"]["partial_rotary_factor"] = .5
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError):
        resolve_position_profile(target.parent, requested_context=262144, manifest_path=path)


def test_rotary_amplitude_preserves_nonrotary_score_contribution(tmp_path):
    from tensorfold.cuda.sm70.family_ops import rotary

    target = put(tmp_path / "target", config())
    p = resolve_position_profile(target.parent, requested_context=524288, yarn_factor=2)
    # Zero phase makes the independently expected FP16 rotary-prefix rounding
    # explicit; the much larger tail must not be uniformly amplitude-scaled.
    q, k = torch.ones((1, 256), dtype=torch.float16), torch.ones((1, 256), dtype=torch.float16)
    q[:, 64:] = 8
    scale = p.target.rope.attention_factor
    inv = p.target.rope.inverse_frequencies(torch)
    actual_q, actual_k = rotary(q, inv, torch.tensor([0]), scale=scale), rotary(k, inv, torch.tensor([0]), scale=scale)
    expected_rotary = torch.tensor(scale, dtype=torch.float16).float().square() * 64
    assert torch.equal(actual_q[:, 64:], q[:, 64:]) and torch.equal(actual_k[:, 64:], k[:, 64:])
    assert torch.equal((actual_q.float() * actual_k.float()).sum(), expected_rotary + 192 * 8)
    assert not torch.equal((actual_q.float() * actual_k.float()).sum(), (q.float() * k.float()).sum() * scale**2)


@pytest.mark.parametrize("factor", [None, 2])
def test_prepared_target_mtp_vectors_have_exact_alias_and_byte_ownership(tmp_path, factor):
    target = put(tmp_path / "target", config())
    p = resolve_position_profile(target.parent, requested_context=1024, yarn_factor=factor)
    weights = SimpleNamespace(cfg=SimpleNamespace(rope=p.target.rope), inv_freq=p.target.rope.inverse_frequencies(torch),
                              positions=None, mtp=object(), device=torch.device("cpu"))
    prepared = prepare_position_vectors(p, mtp_active=True)
    bound = bind_position_profile(weights, p, mtp_active=True, prepared=prepared)
    assert bind_position_profile(weights, p, mtp_active=True, prepared=prepared) is bound
    target_operand, mtp_operand = position_for(weights), position_for(weights, mtp=True)
    assert target_operand.inv_freq is weights.inv_freq
    assert (mtp_operand.inv_freq is weights.inv_freq) is (factor is None)
    assert mtp_operand.amplitude == 1 and mtp_operand.domain is p.mtp
    actual = sum(t.numel() * t.element_size() for t in
                 ({id(target_operand.inv_freq): target_operand.inv_freq, id(mtp_operand.inv_freq): mtp_operand.inv_freq}).values())
    assert position_vector_bytes(p, mtp_active=True) == {
        "target_vector_bytes": 128, "mtp_extra_vector_bytes": 0 if factor is None else 128, "total_vector_bytes": actual}
    changed = resolve_position_profile(target.parent, requested_context=1023, yarn_factor=factor)
    with pytest.raises(ValueError, match="retiring"):
        bind_position_profile(weights, changed, mtp_active=True,
                              prepared=prepare_position_vectors(changed, mtp_active=True))
    with pytest.raises(FrozenInstanceError):
        target_operand.amplitude = 2


def test_inactive_mtp_has_no_vector_or_implicit_runtime_substitution(tmp_path):
    source = put(tmp_path / "target", config())
    p = resolve_position_profile(source.parent, requested_context=524288, yarn_factor=2)
    weights = SimpleNamespace(cfg=SimpleNamespace(rope=p.target.rope), inv_freq=p.target.rope.inverse_frequencies(torch),
                              positions=None, mtp=None, device=torch.device("cpu"))
    assert position_vector_bytes(p, mtp_active=False)["mtp_extra_vector_bytes"] == 0
    bind_position_profile(weights, p, mtp_active=False, prepared=prepare_position_vectors(p, mtp_active=False))
    with pytest.raises(ValueError, match="not admitted/prepared"):
        position_for(weights, mtp=True)


@pytest.mark.parametrize("theta", [1e308, 1e-300])
def test_dflash_unrepresentable_fp32_frequency_source_refuses_before_binding(tmp_path, theta):
    target = put(tmp_path / "target", config())
    raw = json.loads((Path(__file__).parent / "fixtures/dflash2/config.json").read_text())
    raw["rope_parameters"]["rope_theta"] = theta
    draft = put(tmp_path / "draft", raw)
    with pytest.raises(ValueError, match="FP32 frequency"):
        resolve_position_profile(target.parent, requested_context=262144, companion_configs={"dflash2": draft})


def test_gguf_source_header_closure_and_external_position_conflict_are_explicit(tmp_path):
    from gguf_fixture import tiny_model
    from tensorfold.families.qwen4_exp.gguf import metadata_config

    source = tiny_model(tmp_path / "tiny.gguf")
    p = resolve_position_profile(source, requested_context=512, yarn_factor=2)
    assert source.resolve() in p.required_input_files()
    assert p.target.rope.native_context == 256 and p.target.rope.context_limit == 512
    raw = metadata_config(source)
    raw["gguf_file"] = source.name
    raw["text_config"]["rope_parameters"]["rope_theta"] *= 2
    (tmp_path / "config.json").write_text(json.dumps(raw))
    with pytest.raises(ValueError, match="GGUF/external positional"):
        resolve_position_profile(tmp_path, requested_context=256)


@pytest.mark.parametrize("factor", [2, 4])
def test_large_live_thw_coordinates_use_fp32_phases_and_preserve_tail(tmp_path, factor):
    from tensorfold.cuda.sm70.family_ops import rotary

    target = put(tmp_path / "target", config())
    p = resolve_position_profile(target.parent, requested_context=262144 * factor, yarn_factor=factor)
    end = p.target.max_rows - 1
    coords = torch.tensor([[end, end - 17, 101]], dtype=torch.int64)
    p.target.validate_span(end, 1, coordinate_min=tuple(coords[0].tolist()), coordinate_max=tuple(coords[0].tolist()))
    inv = p.target.rope.inverse_frequencies(torch)
    x = (torch.arange(256, dtype=torch.float32).remainder(17) / 8).half().view(1, 256)
    result = rotary(x, inv, torch.tensor([0]), rope=coords, length=1,
                    sections=p.target.rope.mrope_section, scale=p.target.rope.attention_factor)
    expected = x.clone()
    # Independent per-pair axis and FP32 phase/application oracle. No FP16
    # absolute coordinate conversion or cached phase table participates.
    for pair in range(32):
        axis = 1 if pair % 3 == 1 and pair < 33 else 2 if pair % 3 == 2 and pair < 30 else 0
        phase = coords[0, axis].float() * inv[pair]
        cosine, sine = phase.cos() * p.target.rope.attention_factor, phase.sin() * p.target.rope.attention_factor
        a, b = x[0, pair].float(), x[0, pair + 32].float()
        expected[0, pair] = (a * cosine - b * sine).half()
        expected[0, pair + 32] = (b * cosine + a * sine).half()
    assert torch.equal(result, expected)


@pytest.mark.parametrize("theta,factor,message", [(1e308, None, "positive finite FP32"),
                                                  (1.1, 2, "correction ramp")])
def test_unrepresentable_native_vector_or_reversed_yarn_ramp_refuses_before_binding(tmp_path, theta, factor, message):
    raw = config()
    raw["text_config"]["rope_parameters"]["rope_theta"] = theta
    target = put(tmp_path / "target", raw)
    with pytest.raises(ValueError, match=message):
        resolve_position_profile(target.parent, requested_context=262144, yarn_factor=factor)


def test_frequency_alias_ignores_distinct_bounds_and_separate_rotary_amplitudes(tmp_path):
    target = put(tmp_path / "target", config())
    raw = config()
    raw["text_config"]["max_position_embeddings"] = 524288
    companion = put(tmp_path / "mtp", raw)
    p = resolve_position_profile(target.parent, requested_context=1024, companion_configs={"mtp": companion})
    assert p.target.rope != p.mtp.rope  # Distinct reference bounds, identical native vector formula.
    assert position_vector_bytes(p, mtp_active=True)["mtp_extra_vector_bytes"] == 0
    companion.write_text(json.dumps(config()))
    base = resolve_position_profile(target.parent, requested_context=1024, companion_configs={"mtp": companion})
    mtp = role(base, method="mtp", factor=2)
    mtp["rope"]["attention_factor"] = 1.25
    path, _ = manifest(tmp_path, base, rows=1024, factor=2, mtp=mtp)
    p = resolve_position_profile(target.parent, requested_context=1024, manifest_path=path,
                                  companion_configs={"mtp": companion})
    weights = SimpleNamespace(cfg=SimpleNamespace(rope=p.target.rope), inv_freq=p.target.rope.inverse_frequencies(torch),
                              positions=None, mtp=object(), device=torch.device("cpu"))
    prepared = prepare_position_vectors(p, mtp_active=True)
    bound = bind_position_profile(weights, p, mtp_active=True, prepared=prepared)
    assert bound.target.inv_freq is bound.mtp.inv_freq
    assert bound.target.amplitude != bound.mtp.amplitude
    assert position_vector_bytes(p, mtp_active=True)["total_vector_bytes"] == 128


@pytest.mark.parametrize("key,value", [("factor", 2), ("factor", 1), ("beta_fast", 32), ("attention_factor", 1),
                                      ("original_max_position_embeddings", 262144), ("truncate", True), ("mscale", 1)])
def test_explicit_default_manifest_refuses_ignored_yarn_parameters(tmp_path, key, value):
    target = put(tmp_path / "target", config())
    base = resolve_position_profile(target.parent, requested_context=262144)
    path, body = manifest(tmp_path, base)
    body["target"]["rope"][key] = value
    path.write_text(json.dumps(body))
    with pytest.raises(ValueError, match="unused YaRN parameters"):
        resolve_position_profile(target.parent, requested_context=262144, manifest_path=path)


def test_unselected_source_native_defaults_keep_existing_rope_parameter_behavior(tmp_path):
    raw = config()
    raw["text_config"]["rope_parameters"]["factor"] = 2
    target = put(tmp_path / "target", raw)
    p = resolve_position_profile(target.parent, requested_context=262144)
    assert p.target.rope.rope_type == "default" and p.target.rope.factor == 1


def test_exact_cpu_preparation_refuses_combined_fp32_overflow_before_retention(tmp_path):
    raw = config()
    raw["text_config"]["rope_parameters"]["rope_theta"] = 3e38
    target = put(tmp_path / "target", raw)
    p = resolve_position_profile(target.parent, requested_context=262144, yarn_factor=32)
    # Metadata can describe this finite candidate, but its exact existing FP32
    # builder overflows factor*pos_freqs and produces forbidden zero inverses.
    assert p.target.rope.theta == 3e38 and p.target.rope.factor == 32
    with pytest.raises(ValueError, match="nonfinite or zero"):
        prepare_position_vectors(p, mtp_active=False)
    with pytest.raises(ValueError, match="exact CPU preparation"):
        bind_position_profile(SimpleNamespace(), p, mtp_active=False)


def test_preparation_forces_cpu_and_bind_refuses_stale_vectors(tmp_path):
    target = put(tmp_path / "target", config())
    p = resolve_position_profile(target.parent, requested_context=64, yarn_factor=2)
    with torch.inference_mode(), torch.device("meta"):
        prepared = prepare_position_vectors(p, mtp_active=True)
    assert prepared.target_cpu.device.type == prepared.mtp_cpu.device.type == "cpu"
    prepared.target_cpu.add_(1)
    with pytest.raises(ValueError, match="stale positional CPU preparation"):
        bind_position_profile(SimpleNamespace(), p, mtp_active=True, prepared=prepared)


def test_visual_row_allowance_and_lookahead_do_not_use_thw_units(tmp_path):
    target = put(tmp_path / "target", config())
    base = resolve_position_profile(target.parent, requested_context=64)
    path, body = manifest(tmp_path, base, rows=64)
    body["target"]["coordinate_limits"] = [16, 16, 16]
    path.write_text(json.dumps(body))
    p = resolve_position_profile(target.parent, requested_context=64, manifest_path=path)
    assert p.target.max_rows == 64 and p.target.coordinate_limits == (16, 16, 16)
    coordinates = torch.arange(64).remainder(16)[:, None].expand(-1, 3)
    assert coordinates.shape == (64, 3)
    p.target.validate_span(0, 64, coordinate_min=tuple(coordinates.amin(0).tolist()),
                           coordinate_max=tuple(coordinates.amax(0).tolist()))
    # More than16 proposal rows may still share coordinates below16. Row
    # lookahead is constrained by the64-row window, never a THW axis limit.
    p.validate_method("mtp", "actual-codec", 20, media_bounds=(15, 15, 15))
    with pytest.raises(ValueError, match="coordinate"):
        p.target.validate_span(0, 1, coordinate_min=(0, 0, 0), coordinate_max=(16, 0, 0))
    with pytest.raises(ValueError, match="lookahead block"):
        p.validate_method("mtp", "actual-codec", 64)


def test_legacy_mtp_undeclared_positions_use_explicit_target_source_native_defaults(tmp_path):
    target_raw = config()
    target_raw["text_config"]["rope_parameters"]["rope_theta"] = 20000000
    target = put(tmp_path / "target", target_raw)
    companion = put(tmp_path / "mtp", {"head_dim": 256, "hidden_size": 2560, "num_attention_heads": 24})
    p = resolve_position_profile(target.parent, requested_context=64, yarn_factor=2,
                                  companion_configs={"mtp": companion})
    assert p.target.rope.rope_type == "yarn" and p.mtp.rope.rope_type == "default"
    assert p.mtp.rope.theta == 20000000 and p.mtp.rope.native_context == 262144
    assert p.mtp.rope.attention_factor == 1
    assert companion.resolve() in p.required_input_files()


def test_public_mtp_source_defaults_never_inherit_target_runtime_scaling():
    from tensorfold.families.qwen4_exp.rope import RopeParameters

    reference = RopeParameters.from_config(config(), 2)
    actual = mtp_source_parameters({"head_dim": 256, "hidden_size": 2560, "num_attention_heads": 24}, reference)
    assert actual.rope_type == "default" and actual.factor == actual.attention_factor == 1
    assert (actual.theta, actual.rotary_dim, actual.native_context, actual.mrope_section) == (
        10000000, 64, 262144, (11, 11, 10))
    declared = config()
    declared["text_config"]["rope_parameters"].update(rope_type="yarn", factor=4,
        original_max_position_embeddings=262144, attention_factor=1.25)
    actual = mtp_source_parameters(declared, reference)
    assert actual.factor == 4 and actual.attention_factor == 1.25 and actual.context_limit == 1048576


@pytest.mark.parametrize("world,heads", [(8, 2), (1, 2), (8, 1), (4, 2), (8.0, 2), (8, 2.0)])
def test_shared_layout_predicate_matches_actual_history_constructor(world, heads):
    from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement

    supported = HistoryPlacement.supports(world, heads)
    if supported:
        placement = HistoryPlacement(world, heads, 2)
        assert placement.world == 8 and placement.global_kv_heads == 2
    else:
        with pytest.raises(ValueError, match="requires TP8"):
            HistoryPlacement(world, heads, 2)

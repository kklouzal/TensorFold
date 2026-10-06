"""Static YaRN policy, independent Transformers frequencies and startup context contracts (CPU only)."""

from dataclasses import FrozenInstanceError
import json
import math
import struct
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.rope import RopeParameters


def config(**parameters):
    return {"model_type": "qwen4_exp", "text_config": {
        "head_dim": 256, "hidden_size": 2560, "num_attention_heads": 24, "max_position_embeddings": 262144,
        "rope_parameters": {"type": "default", "rope_theta": 10000000, "partial_rotary_factor": 0.25,
                            "mrope_interleaved": True, "mrope_section": [11, 11, 10], **parameters}}}


def yarn(**parameters):
    raw = config()
    raw["text_config"]["rope_parameters"].pop("type")
    raw["text_config"]["rope_parameters"].update({"rope_type": "yarn", "factor": 2.0,
                                                 "original_max_position_embeddings": 262144, **parameters})
    return raw


def test_native_and_cli_override_leave_checkpoint_unchanged():
    raw = config()
    before = json.dumps(raw, sort_keys=True)
    default = RopeParameters.from_config(raw)
    extended = RopeParameters.from_config(raw, 2.0)
    assert default.rope_type == "default" and default.context_limit == default.native_context == 262144
    assert extended.context_limit == 524288 and extended.native_context == 262144
    assert extended.rotary_dim == 64 and extended.attention_factor == 1 + 0.1 * math.log(2)
    assert json.dumps(raw, sort_keys=True) == before
    with pytest.raises(FrozenInstanceError):
        extended.factor = 4


def test_official_checkpoint_policy_and_factor_override():
    raw = yarn()
    assert RopeParameters.from_config(raw) == RopeParameters.from_config(config(), 2.0)
    assert RopeParameters.from_config(raw, 4.0).context_limit == 1048576
    raw["text_config"]["max_position_embeddings"] = 524288
    assert RopeParameters.from_config(raw).native_context == 262144


@pytest.mark.parametrize("factor", [True, False, 0, -2, 0.99, math.inf, math.nan, "2", 10**400])
def test_invalid_cli_factors_fail_at_the_boundary(factor):
    with pytest.raises(ValueError, match="yarn-factor"):
        RopeParameters.from_config(config(), factor)


@pytest.mark.parametrize("parameters", [
    {"factor": 0}, {"factor": True}, {"factor": math.nan}, {"factor": 1e30},
    {"original_max_position_embeddings": 0}, {"original_max_position_embeddings": True},
    {"original_max_position_embeddings": 262144.5}, {"original_max_position_embeddings": 2**40},
    {"original_max_position_embeddings": 10**400},
    {"beta_fast": 0}, {"beta_slow": -1}, {"beta_fast": 1, "beta_slow": 2},
    {"attention_factor": 0}, {"attention_factor": math.inf}, {"attention_factor": True},
    {"attention_factor": 1e308}, {"attention_factor": 1e-300}, {"rope_theta": 1e308},
    {"beta_fast": 1e308},
    {"truncate": "false"}, {"mscale": 1}, {"mscale_all_dim": 1},
    {"mscale": -1, "mscale_all_dim": 1},
])
def test_invalid_checkpoint_yarn_parameters_fail(parameters):
    with pytest.raises(ValueError):
        RopeParameters.from_config(yarn(**parameters))


@pytest.mark.parametrize("params", [
    {"type": "linear"}, {"type": "dynamic"}, {"type": "default", "rope_type": "yarn"},
    {"partial_rotary_factor": 0}, {"partial_rotary_factor": 2}, {"partial_rotary_factor": 0.1},
    {"rope_theta": 0},
])
def test_unsupported_modes_and_invalid_rotary_dimensions_fail(params):
    with pytest.raises(ValueError):
        RopeParameters.from_config(config(**params))


@pytest.mark.parametrize("invalid", [[], "yarn", 2, False])
def test_rope_configuration_must_be_an_object(invalid):
    raw = config()
    raw["text_config"]["rope_parameters"] = invalid
    with pytest.raises(ValueError, match="object"):
        RopeParameters.from_config(raw)


def test_declared_limit_cannot_exceed_frequency_extension():
    raw = yarn()
    raw["text_config"]["max_position_embeddings"] = 524289
    with pytest.raises(ValueError, match="exceeds YaRN"):
        RopeParameters.from_config(raw)


def test_huge_kernel_dimension_fails_with_actionable_value_error():
    raw = config()
    raw["text_config"]["head_dim"] = 10**400
    with pytest.raises(ValueError, match="head_dim"):
        RopeParameters.from_config(raw)


def test_legacy_scaling_preserves_the_canonical_multimodal_sections():
    raw = yarn()
    params = raw["text_config"].pop("rope_parameters")
    params["mrope_section"] = [12, 10, 10]
    raw["text_config"]["rope_scaling"] = params
    policy = RopeParameters.from_config(raw)
    assert policy.mrope_section == (12, 10, 10)
    assert policy.metadata()["mrope_section"] == [12, 10, 10]


@pytest.mark.torch
def test_cuda_config_uses_the_canonical_policy_for_legacy_scaling(tmp_path):
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from test_flashnext_eos import TEXT
    from tensorfold.families.qwen4_exp.cuda.weight_types import Config

    raw = yarn()
    raw["text_config"] = {**TEXT, **raw["text_config"], "eos_token_id": 5}
    params = raw["text_config"].pop("rope_parameters")
    params["mrope_section"] = [12, 10, 10]
    raw["text_config"]["rope_scaling"] = params
    (tmp_path / "config.json").write_text(json.dumps(raw))
    cfg = Config.read(tmp_path)
    assert cfg.mrope_section == cfg.rope.mrope_section == (12, 10, 10)
    assert cfg.rope_theta == cfg.rope.theta == 1e7
    assert cfg.rotary_dim == cfg.rope.rotary_dim == 64
    assert cfg.rope_attention_factor == 1 + 0.1 * math.log(2)


@pytest.mark.torch
def test_native_frequencies_keep_pre_fork_bits():
    torch = pytest.importorskip("torch")
    policy = RopeParameters.from_config(config())
    old = (torch.tensor(1e7, dtype=torch.float64) ** (-torch.arange(32, dtype=torch.float64) / 32)).float()
    assert torch.equal(policy.inverse_frequencies(torch), old)


@pytest.mark.torch
def test_zero_frequency_from_float32_expansion_is_refused_before_use():
    torch = pytest.importorskip("torch")
    policy = RopeParameters.from_config(yarn(factor=1000.0, rope_theta=3e38))
    with pytest.raises(ValueError, match="inverse frequencies"):
        policy.inverse_frequencies(torch)


@pytest.mark.torch
@pytest.mark.parametrize("factor,extra", [(1.0, {}), (2.0, {}), (4.0, {}),
    (2.0, {"beta_fast": 16.0, "beta_slow": 2.0, "attention_factor": 1.05}),
    (4.0, {"truncate": False}), (2.0, {"mscale": 2.0, "mscale_all_dim": 1.0}),
    (2.0, {"beta_fast": 1e-300, "beta_slow": 1e-301})])
def test_frequencies_and_amplitude_match_independent_transformers_oracle(factor, extra):
    torch = pytest.importorskip("torch")
    oracle = pytest.importorskip("transformers.modeling_rope_utils")
    raw = yarn(**extra)
    raw["text_config"]["rope_parameters"]["factor"] = factor
    policy = RopeParameters.from_config(raw)
    text = raw["text_config"]
    hf = SimpleNamespace(**text, standardize_rope_params=lambda: None)
    expected, amplitude = oracle._compute_yarn_parameters(hf, device=torch.device("cpu"))
    assert torch.equal(policy.inverse_frequencies(torch), expected)
    assert policy.attention_factor == amplitude


def _checkpoint(path):
    (path / "config.json").write_text(json.dumps(config()))
    header = json.dumps({"test.weight": {"dtype": "BF16", "shape": [1], "data_offsets": [0, 2]}}).encode()
    (path / "model.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + bytes(2))


def _admission(path, monkeypatch, requested, explicit, *, budget=1024**3, limit=524288):
    from tensorfold.cuda import build, capacity

    _checkpoint(path)
    monkeypatch.setattr(build, "refuse_old_gpu", lambda *a: None)
    monkeypatch.setattr(capacity, "available_bytes", lambda *a: budget)
    monkeypatch.setattr(capacity, "unified", lambda *a: True)
    monkeypatch.setattr(capacity, "page_room", lambda *a: None)
    return capacity.admit(path, requested, explicit, None, capacity.Geometry(lambda slots: slots * 1024, 16),
                          lambda *a: (2, 0), context_limit=limit, original_context=262144)


def test_extended_admission_preserves_native_metadata(tmp_path, monkeypatch):
    plan = _admission(tmp_path, monkeypatch, 524288, True)
    assert plan["native_window"] == 262144 and plan["rope_context_limit"] == 524288
    assert plan["context_window"] == 524288 and plan["cache_slots"] == 524304


def test_context_above_yarn_limit_refuses(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="exceeds"):
        _admission(tmp_path, monkeypatch, 524289, True)


def test_explicit_yarn_context_refuses_insufficient_memory(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="memory budget cannot fit"):
        _admission(tmp_path, monkeypatch, 524288, True, budget=100 * 1024**2)


@pytest.mark.parametrize("requested_context", [None, 0])
def test_default_yarn_context_fits_memory_without_claiming_the_full_window(tmp_path, monkeypatch, requested_context):
    plan = _admission(tmp_path, monkeypatch, requested_context, requested_context is not None, budget=100 * 1024**2)
    assert 0 < plan["context_window"] < 524288
    assert plan["rope_context_limit"] == 524288 and plan["native_window"] == 262144


@pytest.mark.parametrize("backend,package", [("mlx", True), ("cuda", False)])
def test_cli_refuses_yarn_on_unsupported_backend_or_family_before_load(backend, package):
    from tensorfold import cli
    from tensorfold.serve_options import check

    args = cli.build_parser().parse_args(["serve", "placeholder", "--yarn-factor", "2"])
    family = SimpleNamespace(package=SimpleNamespace(rope_parameters=lambda *a: pytest.fail("unsupported load"))
                             if package else SimpleNamespace())
    with pytest.raises(ValueError, match="Flash Next on CUDA only"):
        check(args, family, backend)


def test_cli_validates_yarn_before_weight_resolution(tmp_path, monkeypatch):
    from tensorfold import cli, families, hub
    from tensorfold.families import qwen4_exp

    (tmp_path / "config.json").write_text(json.dumps(config()))
    family = SimpleNamespace(package=qwen4_exp, title=qwen4_exp.TITLE, model_type="qwen4_exp")
    monkeypatch.setattr(families, "detect", lambda *a: family)
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(cli, "_note_untested", lambda *a: None)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: pytest.fail("weight resolution before validation"))
    for tail in [["--yarn-factor", "nan"], ["--yarn-factor", "2", "--context", "524289"]]:
        args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", *tail])
        with pytest.raises(ValueError):
            cli.cmd_serve(args)


@pytest.mark.parametrize("use_flag", [False, True])
def test_cli_passes_the_extended_context_and_override_to_cuda(tmp_path, monkeypatch, use_flag):
    from tensorfold import cli, families, hub
    from tensorfold.families import qwen4_exp

    (tmp_path / "config.json").write_text(json.dumps(config() if use_flag else yarn()))
    package = SimpleNamespace(rope_parameters=qwen4_exp.rope_parameters, cuda_engine=lambda *a, **k: None)
    family = SimpleNamespace(package=package, title=qwen4_exp.TITLE, model_type="qwen4_exp")
    monkeypatch.setattr(families, "detect", lambda *a: family)
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(cli, "_note_untested", lambda *a: None)
    monkeypatch.setattr(hub, "resolve", lambda *a, **k: tmp_path)
    monkeypatch.setattr(cli.stacks, "start", lambda: None)
    seen = []
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, family, path, context:
                        seen.append((context, args.yarn_factor)) or 0)
    tail = ["--yarn-factor", "2"] if use_flag else []
    args = cli.build_parser().parse_args(["serve", str(tmp_path), "--backend", "cuda", *tail])
    assert cli.cmd_serve(args) == 0
    assert seen == [(524288, 2.0 if use_flag else None)]

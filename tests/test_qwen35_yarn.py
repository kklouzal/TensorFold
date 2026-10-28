"""Qwen3.8 dense static YaRN policy: default checkpoints keep their RoPE; ``--yarn-factor`` and declared YaRN extend
positions with Transformers' frequencies and rotary amplitude (no accelerator imports)."""

from __future__ import annotations

import json
import math

import pytest

from tensorfold.families import qwen3_5

TEXT = {"model_type": "qwen3_5_text", "hidden_size": 5120, "num_attention_heads": 24, "num_key_value_heads": 4,
        "head_dim": 256, "num_hidden_layers": 64, "full_attention_interval": 4, "max_position_embeddings": 262144,
        "vocab_size": 248320, "rope_parameters": {"mrope_interleaved": True, "mrope_section": [11, 11, 10],
                                                  "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                                                  "rope_type": "default"}}


def _model(tmp_path, text):
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5", "text_config": text}))
    return tmp_path


def test_default_checkpoint_keeps_its_rope_and_native_window(tmp_path):
    rope = qwen3_5.rope_parameters(_model(tmp_path, TEXT))
    assert (rope.rope_type, rope.rotary_dim, rope.theta, rope.context_limit) == ("default", 64, 1e7, 262144)


def test_unrelated_mrope_layouts_do_not_need_yarn_validation(tmp_path):
    text = json.loads(json.dumps(TEXT))
    text["rope_parameters"]["mrope_interleaved"] = False         # a layout only YaRN's policy would refuse
    assert qwen3_5.rope_parameters(_model(tmp_path, text)).rope_type == "default"


def test_yarn_factor_four_reaches_the_cards_window_with_transformers_amplitude(tmp_path):
    rope = qwen3_5.rope_parameters(_model(tmp_path, TEXT), 4.0)
    assert rope.rope_type == "yarn" and rope.native_context == 262144 and rope.context_limit == 4 * 262144
    assert rope.attention_factor == pytest.approx(1 + 0.1 * math.log(4.0))
    with pytest.raises(ValueError, match="yarn-factor"):
        qwen3_5.rope_parameters(_model(tmp_path, TEXT), 0.5)


def test_declared_yarn_is_read_from_the_checkpoint(tmp_path):
    text = json.loads(json.dumps(TEXT))
    text["max_position_embeddings"] = 1048576
    text["rope_parameters"].update(rope_type="yarn", factor=4.0, original_max_position_embeddings=262144)
    rope = qwen3_5.rope_parameters(_model(tmp_path, text))
    assert rope.rope_type == "yarn" and rope.factor == 4.0 and rope.context_limit == 1048576


def test_cuda_yarn_option_reaches_qwen_dense_and_mlx_refuses(tmp_path):
    from types import SimpleNamespace

    from tensorfold import serve_options

    family = SimpleNamespace(title=qwen3_5.TITLE, package=qwen3_5, model_type="qwen3_5")
    args = SimpleNamespace(yarn_factor=4.0)
    model = _model(tmp_path, TEXT)
    check = serve_options.check
    with pytest.raises(ValueError, match="Qwen3.8 dense and Qwen3.6 on CUDA only"):
        check(args, family, "mlx", model)
    check(args, family, "cuda", model)


def test_both_rope_fields_must_declare_the_same_yarn(tmp_path):
    yarn = dict(TEXT["rope_parameters"], rope_type="yarn", factor=4.0, original_max_position_embeddings=262144)
    for scaling, agree in ((dict(yarn), True), (dict(yarn, beta_fast=16.0), False), (TEXT["rope_parameters"], False)):
        text = dict(TEXT, max_position_embeddings=1048576, rope_parameters=yarn, rope_scaling=scaling)
        if agree:
            assert qwen3_5.rope_parameters(_model(tmp_path, text)).factor == 4.0
        else:
            with pytest.raises(ValueError, match="disagree"):
                qwen3_5.rope_parameters(_model(tmp_path, text))
    text = dict(TEXT, rope_scaling=yarn)                          # default rope_parameters, YaRN rope_scaling
    with pytest.raises(ValueError, match="disagree"):
        qwen3_5.rope_parameters(_model(tmp_path, text))


def test_the_mac_decoders_read_rope_themselves_and_qwen36_cuda_reads_yarn_like_the_dense_family():
    from tensorfold.families import qwen3_5_moe

    assert qwen3_5.ROPE_BACKENDS == qwen3_5_moe.ROPE_BACKENDS == ("cuda",)
    yarn = dict(TEXT["rope_parameters"], rope_type="yarn", factor=4.0, original_max_position_embeddings=262144)
    raw = {"model_type": "qwen3_5_moe", "text_config": dict(TEXT, rope_parameters=yarn)}
    qwen3_5_moe.check_metadata(raw, "mlx")
    qwen3_5_moe.check_metadata(raw, "cuda")                      # declared YaRN is read, not refused
    assert qwen3_5_moe.rope_parameters is qwen3_5.rope_parameters  # one policy: the dense family's tests cover it


@pytest.mark.parametrize("quantized", [True, False])
def test_qwen36_cuda_engines_receive_the_yarn_factor(tmp_path, monkeypatch, quantized):
    import tensorfold.cuda.nvfp4.format as fmt
    from tensorfold.families import qwen3_5_moe
    from tensorfold.families.qwen3_5_moe.cuda import engine, nvfp4_engine

    got = []
    monkeypatch.setattr(fmt, "is_quantized", lambda path: quantized)
    monkeypatch.setattr(qwen3_5_moe, "check_vram_experts", lambda *a, **k: None)
    monkeypatch.setattr(nvfp4_engine, "QwenNvfp4MoeEngine", lambda *a, **k: got.append(k) or "ct")
    monkeypatch.setattr(engine, "Qwen36Engine", lambda *a, **k: got.append(k) or "affine")
    assert qwen3_5_moe.cuda_engine(tmp_path, yarn_factor=4.0) == ("ct" if quantized else "affine")
    assert got[0]["yarn_factor"] == 4.0


@pytest.mark.parametrize("layout", ["set-without-theta", "no-set"])
def test_yarn_keeps_a_legacy_text_level_theta(tmp_path, layout):
    from tensorfold.families.qwen4_exp.rope import RopeParameters

    text = json.loads(json.dumps(TEXT))
    parameters = text.pop("rope_parameters")
    del parameters["rope_theta"]
    text["rope_theta"] = 5000000                                  # not the 1e7 fallback: a silent change would show
    if layout == "set-without-theta":
        text["rope_parameters"] = parameters
    default = qwen3_5.rope_parameters(_model(tmp_path, text))
    extended = qwen3_5.rope_parameters(_model(tmp_path, text), 4.0)
    assert default.theta == extended.theta == 5e6
    nested = dict(TEXT, rope_parameters=dict(TEXT["rope_parameters"], rope_theta=5000000))
    assert extended == RopeParameters.from_config({"text_config": nested}, 4.0)
    text.update(max_position_embeddings=1048576,
                rope_scaling={"rope_type": "yarn", "factor": 4.0, "original_max_position_embeddings": 262144})
    text.pop("rope_parameters", None)
    declared = qwen3_5.rope_parameters(_model(tmp_path, text))
    assert declared.rope_type == "yarn" and declared.theta == 5e6


def _serve(tmp_path, monkeypatch, text, *tail, resolve=None):
    from types import SimpleNamespace

    from tensorfold import cli, families, hub

    model = _model(tmp_path, text)
    family = SimpleNamespace(package=qwen3_5, title=qwen3_5.TITLE, model_type="qwen3_5")
    monkeypatch.setattr(families, "detect", lambda *a: family)
    monkeypatch.setattr(families, "require_readable", lambda *a: None)
    monkeypatch.setattr(cli, "_note_untested", lambda *a: None)
    monkeypatch.setattr(cli.stacks, "start", lambda: None)
    seen = []
    monkeypatch.setattr(hub, "resolve", resolve or (lambda *a, **k: seen.append("resolve") or model))
    monkeypatch.setattr(cli, "_serve_cuda", lambda args, family, path, context: seen.append(context) or 0)
    args = cli.build_parser().parse_args(["serve", str(model), "--no-update-check", *tail])
    return cli.cmd_serve(args), seen


@pytest.mark.parametrize("tail,window", [((), 262144), (("--yarn-factor", "4"), 1048576)])
def test_cuda_serve_admits_the_yarn_window_and_no_more(tmp_path, monkeypatch, tail, window):
    assert _serve(tmp_path, monkeypatch, TEXT, "--backend", "cuda", *tail) == (0, ["resolve", window])
    with pytest.raises(ValueError, match=f"exceeds this model's {window}-token window"):
        _serve(tmp_path, monkeypatch, TEXT, "--backend", "cuda", *tail, "--context", str(window + 1))


def test_mac_serve_does_not_consult_the_cuda_rope_policy(tmp_path, monkeypatch):
    class Resolved(Exception):
        pass

    def resolved(*args, **kwargs):
        raise Resolved

    text = dict(TEXT, rope_parameters={"rope_type": "linear", "factor": 2.0})       # CUDA's policy refuses this
    with pytest.raises(ValueError, match="default or static YaRN"):
        _serve(tmp_path, monkeypatch, text, "--backend", "cuda")
    with pytest.raises(Resolved):                              # the Mac path reaches weight resolution
        _serve(tmp_path, monkeypatch, text, "--backend", "mlx", resolve=resolved)

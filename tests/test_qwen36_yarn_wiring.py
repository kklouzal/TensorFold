"""Qwen3.6 static YaRN wiring without a GPU: the affine engine admits, loads and logs the policy's window, and its MTP
layer rotates with the policy's amplitude (the shared forward's rope_scale), as the target's attention does. The
loaders' forwarding is checked with their controls (test_qwen36_nvfp4_load_controls.py), the CT engine's admission,
loader and log in test_qwen_nvfp4_startup.py."""

import json
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen3_5_moe.cuda import mtp


def _model(tmp_path, *, declared=False):
    rope = {"mrope_interleaved": True, "mrope_section": [11, 11, 10], "partial_rotary_factor": 0.25,
            "rope_theta": 10000000, "rope_type": "default"}
    text = {"hidden_size": 2048, "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
            "max_position_embeddings": 262144, "rope_parameters": rope}
    if declared:
        text["max_position_embeddings"] = 1048576
        rope.update(rope_type="yarn", factor=4.0, original_max_position_embeddings=262144)
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_5_moe", "text_config": text}))
    return tmp_path


@pytest.mark.parametrize("factor,declared,expected", [(None, False, None), (4.0, False, 4.0),
                                                    (None, True, 4.0), (8.0, True, 8.0)])
def test_the_affine_engine_admits_loads_and_logs_the_yarn_window(tmp_path, monkeypatch, capsys,
                                                              factor, declared, expected):
    from tensorfold.cuda import capacity, chat_template, markers
    from tensorfold.families.qwen3_5_moe.cuda import engine, weights

    _model(tmp_path, declared=declared)
    admissions, loads = [], []
    monkeypatch.setattr(chat_template, "prepare_chat_template", lambda *a, **k: None)
    monkeypatch.setattr(markers, "resume_points", lambda *a, **k: ())
    monkeypatch.setattr(torch.cuda, "set_device", lambda device: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(capacity, "admit", lambda *a, **options: admissions.append(options) or {
        "context_window": 64, "cache_slots": 72})
    monkeypatch.setattr(weights, "load", lambda path, **options: loads.append(options) or NS(config=NS(eos=(1,))))
    engine.Qwen36Engine(tmp_path, depth=0, yarn_factor=factor)
    log = capsys.readouterr().out
    if expected is None:
        assert "context_limit" not in admissions[0] and loads[0]["rope"] is None and "static YaRN" not in log
    else:
        assert (admissions[0]["context_limit"], admissions[0]["original_context"]) == (int(262144 * expected), 262144)
        policy = loads[0]["rope"]
        assert (policy.rope_type, policy.factor, policy.rotary_dim, policy.mrope_section) == (
            "yarn", expected, 64, (11, 11, 10))
        assert f"static YaRN factor {expected:g}, rotary amplitude" in log


@pytest.mark.parametrize("factor", [True, False, 0.0, 0.5, -1.0, float("nan"), float("inf")])
def test_bad_affine_yarn_factor_is_refused_before_cuda_or_template_preparation(tmp_path, monkeypatch, factor):
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    _model(tmp_path)
    monkeypatch.setattr(chat_template, "prepare_chat_template", lambda *a, **k: pytest.fail("template prepared"))
    monkeypatch.setattr(torch.cuda, "set_device", lambda *a, **k: pytest.fail("CUDA prepared"))
    with pytest.raises(ValueError, match="yarn-factor"):
        Qwen36Engine(tmp_path, depth=0, yarn_factor=factor)


def test_an_override_cannot_silently_reduce_a_declared_checkpoint_position_limit(tmp_path, monkeypatch):
    from tensorfold.cuda import chat_template
    from tensorfold.families.qwen3_5_moe.cuda.engine import Qwen36Engine

    _model(tmp_path, declared=True)
    monkeypatch.setattr(chat_template, "prepare_chat_template", lambda *a, **k: pytest.fail("template prepared"))
    with pytest.raises(ValueError, match="max_position_embeddings 1048576 exceeds YaRN's 524288-token limit"):
        Qwen36Engine(tmp_path, depth=0, yarn_factor=2.0)


def test_the_mtp_layer_rotates_with_the_configured_amplitude(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def attn_prep(*args, **kwargs):
        seen.update(kwargs)
        raise Stop

    rows = torch.zeros(2, 4)
    glue = NS(embed=lambda *a: rows, add_rmsnorm=lambda x, r, w, eps: (x, x, x), attn_prep=attn_prep)
    monkeypatch.setattr(mtp, "glue", glue)
    monkeypatch.setattr(mtp, "matmul", lambda x, w, xs=None: torch.zeros(x.shape[0], 8))
    head = object.__new__(mtp.Head)
    attn = NS(q=None, k=None, v=None, o=None, q_norm=None, k_norm=None)
    head.w = NS(config=NS(hidden=4, eps=1e-6, heads=1, kv_heads=1, head_dim=8, rope_scale=1.25),
                embed=NS(weight=None, scales=None, biases=None), inv_freq=None)
    head.m = NS(norm_e=None, norm_h=None, fc_e=None, fc_h=None, input_norm=None, attn=attn)
    with pytest.raises(Stop):
        head._layer(rows, torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int64), lambda *a: None)
    assert seen["amplitude"] == 1.25

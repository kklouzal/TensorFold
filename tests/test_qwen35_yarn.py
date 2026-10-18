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
    with pytest.raises(ValueError, match="Qwen3.8 dense on CUDA only"):
        check(args, family, "mlx", model)
    check(args, family, "cuda", model)

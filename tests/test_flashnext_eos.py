"""Flash Next on CUDA stops where the checkpoint says a reply ends, reading generation_config.json too (CPU only)."""

import json

import pytest

pytest.importorskip("torch")
pytest.importorskip("triton")          # Flash Next's CUDA weights import its triton kernels

from tensorfold.families.qwen4_exp.cuda.weights import Config  # noqa: E402
from tensorfold.token_metadata import resolve_eos  # noqa: E402

IM_END, ENDOFTEXT = 248046, 248044
TEXT = {
    "hidden_size": 64, "num_hidden_layers": 2, "layer_types": ["linear_attention", "full_attention"],
    "vocab_size": 248320, "rms_norm_eps": 1e-6, "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16,
    "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_key_head_dim": 16, "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4, "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
    "shared_expert_intermediate_size": 32, "hidden_act": "silu", "output_gate_type": "sigmoid",
}
GATELESS = {k: v for k, v in TEXT.items() if k not in ("hidden_act", "output_gate_type")}


def checkpoint(tmp_path, config, generation=None):
    (tmp_path / "config.json").write_text(json.dumps(config))
    if generation is not None:
        (tmp_path / "generation_config.json").write_text(json.dumps(generation))
    return tmp_path


def test_exl3_pack_stops_at_im_end_from_generation_config(tmp_path):
    # the EXL3 packs' layout: no top-level eos_token_id, text_config names <|endoftext|> only
    d = checkpoint(tmp_path, {"model_type": "qwen4_exp", "text_config": {**TEXT, "eos_token_id": ENDOFTEXT}},
                   {"eos_token_id": [IM_END, ENDOFTEXT], "bos_token_id": ENDOFTEXT})
    c = Config.read(d)
    assert c.eos == (ENDOFTEXT, IM_END)
    assert c.ple_eos == ENDOFTEXT                  # the n-gram tables' end id still comes from text_config


def test_top_level_ids_keep_their_order(tmp_path):
    # the MLX checkpoint's layout: the top level already lists both
    d = checkpoint(tmp_path, {"model_type": "qwen4_exp", "eos_token_id": [IM_END, ENDOFTEXT],
                              "text_config": {**TEXT, "eos_token_id": ENDOFTEXT}},
                   {"eos_token_id": [IM_END, ENDOFTEXT]})
    assert Config.read(d).eos == (IM_END, ENDOFTEXT)


def test_without_generation_config_the_config_ids_are_used(tmp_path):
    d = checkpoint(tmp_path, {"model_type": "qwen4_exp", "text_config": {**TEXT, "eos_token_id": ENDOFTEXT}})
    assert Config.read(d).eos == (ENDOFTEXT,)


def test_scalar_and_missing_values(tmp_path):
    gen = tmp_path / "generation_config.json"
    gen.write_text(json.dumps({"eos_token_id": IM_END}))
    assert resolve_eos(tmp_path, ENDOFTEXT) == (ENDOFTEXT, IM_END)
    gen.write_text(json.dumps({"temperature": 1.0}))
    assert resolve_eos(tmp_path, [ENDOFTEXT]) == (ENDOFTEXT,)
    absent = tmp_path / "absent"
    absent.mkdir()
    with pytest.raises(ValueError, match="eos_token_id"):
        resolve_eos(absent)


def test_named_tokenizer_eos_is_reply_stop_but_does_not_change_training_ple_eos(tmp_path):
    d = checkpoint(tmp_path, {"model_type": "qwen4_exp", "text_config": {**TEXT, "eos_token_id": ENDOFTEXT}},
                   {"eos_token_id": ENDOFTEXT})
    (d / "tokenizer_config.json").write_text(json.dumps({"eos_token": {"content": "turn_end"},
        "added_tokens_decoder": {str(IM_END): {"content": "turn_end", "special": True}}}))
    assert Config.read(d).eos == (ENDOFTEXT, IM_END)
    overridden = Config.read(d, eos_token_ids=[7, 7, 0])
    assert overridden.eos == (7, 0) and overridden.ple_eos == ENDOFTEXT


# transformers' Qwen4ExpConfig: an unset output_gate_type is hidden_act, whose own default is "silu"
@pytest.mark.parametrize("gate", [{"hidden_act": "silu", "output_gate_type": "sigmoid"},
                                  {"output_gate_type": "sigmoid"}, {"hidden_act": "sigmoid"}],
                         ids=["checkpoint", "gate_only", "unset_gate_is_hidden_act"])
def test_the_sigmoid_output_gate_loads(tmp_path, gate):
    text = {**GATELESS, "eos_token_id": ENDOFTEXT, **gate}
    assert Config.read(checkpoint(tmp_path, {"model_type": "qwen4_exp", "text_config": text})).eos == (ENDOFTEXT,)


@pytest.mark.parametrize("gate", [{"hidden_act": "silu", "output_gate_type": "silu"}, {"hidden_act": "silu"}, {}],
                         ids=["silu", "unset_gate_is_hidden_act", "both_unset_are_silu"])
def test_an_output_gate_the_kernels_lack_is_refused(tmp_path, gate):
    # the MLX layers also apply SiLU; the CUDA DeltaNet kernels apply a sigmoid only, so loading would compute wrong
    text = {**GATELESS, "eos_token_id": ENDOFTEXT, **gate}
    d = checkpoint(tmp_path, {"model_type": "qwen4_exp", "text_config": text})
    with pytest.raises(ValueError, match="output_gate_type 'silu'"):
        Config.read(d)

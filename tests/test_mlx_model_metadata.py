"""Preparation contracts with CPU provider objects; these are not Apple/MLX execution tests."""
import json

import pytest

from tensorfold.families.glm5_next.prompts import GlmTokenizer
from tensorfold.families.tokenizer import apply_model_metadata


class Provider:
    """The public EOS/template configuration surface exposed by mlx-lm's wrapper."""
    def __init__(self, *, custom=None):
        self.eos_token_ids = {3}
        self.chat_template = "existing callable default" if custom else None
        self._chat_template = custom
        self.has_chat_template = False

    def get_vocab(self):
        return {"unknown": 0, "model_stop": 1, "turn_stop": 2, "recipe_stop": 3}


def checkpoint(directory):
    (directory / "config.json").write_text(json.dumps({"vocab_size": 4, "eos_token_id": 1}))
    (directory / "generation_config.json").write_text(json.dumps({"eos_token_id": [1, 2]}))
    (directory / "tokenizer_config.json").write_text(json.dumps({"eos_token": "turn_stop", "chat_template": {
        "default": "model default {{ messages }}", "tool_use": "model tools {{ tools }}"}}))


def test_public_provider_defaults_keep_declared_model_generation_and_recipe_stops(tmp_path):
    checkpoint(tmp_path)
    provider = Provider()
    apply_model_metadata(provider, tmp_path)
    assert provider.eos_token_ids == (1, 2, 3)
    assert provider.chat_template == {"default": "model default {{ messages }}", "tool_use": "model tools {{ tools }}"}
    assert provider.has_chat_template


def test_owned_glm_wrapper_updates_actual_inner_template_and_replaces_all_stop_roles(tmp_path):
    checkpoint(tmp_path)
    inner = Provider()
    tokenizer = GlmTokenizer(inner)
    apply_model_metadata(tokenizer, tmp_path, eos_token_ids=[0], chat_template_name="tool_use")
    assert tokenizer.eos_token_ids == inner.eos_token_ids == (0,)
    assert inner.chat_template == "model tools {{ tools }}"
    assert "chat_template" not in vars(tokenizer)  # no shadow source that rendering would ignore


def test_explicit_file_applies_captured_source_instead_of_a_path_lookup(tmp_path):
    checkpoint(tmp_path)
    path = tmp_path / "operator.jinja"
    path.write_text("operator source")
    provider = Provider()
    apply_model_metadata(provider, tmp_path, chat_template_file=path)
    path.write_text("later replacement")
    assert provider.chat_template == "operator source"


def test_custom_callable_default_remains_authoritative_and_explicit_override_is_refused(tmp_path):
    checkpoint(tmp_path)
    custom = lambda *args: "custom"
    provider = Provider(custom=custom)
    apply_model_metadata(provider, tmp_path)
    assert provider._chat_template is custom and provider.chat_template == "existing callable default"
    with pytest.raises(ValueError, match="does not support chat template overrides"):
        apply_model_metadata(provider, tmp_path, chat_template_name="tool_use")


@pytest.mark.parametrize("override", [[], [4], [False], [-1]])
def test_invalid_explicit_stops_fail_before_provider_mutation(tmp_path, override):
    checkpoint(tmp_path)
    provider = Provider()
    with pytest.raises(ValueError):
        apply_model_metadata(provider, tmp_path, eos_token_ids=override)
    assert provider.eos_token_ids == {3} and provider.chat_template is None

"""Pinned HF v5.17.0 _from_pretrained selects modern roles or legacy map replacement."""
import json

import pytest

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.token_metadata import resolve_eos, resolve_token_roles


def write(directory, name, value):
    (directory / name).write_text(json.dumps(value))


@pytest.mark.parametrize("role", ["bos_token", "eos_token", "pad_token", "unk_token"])
@pytest.mark.parametrize("config_value,modern,legacy,expected", [
    ("configured", True, {"value": "legacy"}, "configured"),
    ("configured", False, {"value": "legacy"}, "legacy"),
    (None, True, {"value": "legacy"}, None),
    ("configured", False, {"value": None}, None),
    (None, False, {"value": "legacy"}, "legacy"),
    ("configured", False, None, "configured"),
    (None, False, None, None),
])
def test_existing_roles_follow_modern_key_presence_or_legacy_replacement(
        tmp_path, role, config_value, modern, legacy, expected):
    config = {role: config_value}
    if modern:
        config["added_tokens_decoder"] = {}  # key presence, not decoder truthiness
    write(tmp_path, "tokenizer_config.json", config)
    if legacy is not None:
        write(tmp_path, "special_tokens_map.json", {role: legacy["value"]})
    assert resolve_token_roles(tmp_path) == ({} if expected is None else {role: expected})


def test_modern_absent_role_does_not_inherit_legacy_role(tmp_path):
    write(tmp_path, "tokenizer_config.json", {"added_tokens_decoder": {}})
    write(tmp_path, "special_tokens_map.json", {"eos_token": "legacy"})
    assert resolve_token_roles(tmp_path) == {}


@pytest.mark.parametrize("role", ["bos_token", "pad_token", "unk_token"])
@pytest.mark.parametrize("value", ["", {"content": ""}])
def test_rendering_only_roles_preserve_valid_empty_string_content(tmp_path, role, value):
    write(tmp_path, "tokenizer_config.json", {role: value, "added_tokens_decoder": {},
                                               "chat_template": f"[{{{{ {role}|default('missing') }}}}]"})
    assert resolve_token_roles(tmp_path) == {role: ""}
    template = ChatTemplate(tmp_path)
    assert template.specials == {role: ""}
    assert template.render([{"role": "user", "content": "hello"}], tools=None, enable_thinking=True) == "[]"


@pytest.mark.parametrize("role", ["bos_token", "pad_token", "unk_token"])
@pytest.mark.parametrize("value", [{}, {"content": None}, {"content": 2}])
def test_empty_string_support_does_not_admit_malformed_rendering_token_objects(tmp_path, role, value):
    write(tmp_path, "tokenizer_config.json", {role: value})
    with pytest.raises(ValueError, match=f"tokenizer {role}"):
        resolve_token_roles(tmp_path)


@pytest.mark.parametrize("decoder", [None, [], False, "decoder"])
def test_present_malformed_modern_decoder_is_not_legacy_absence(tmp_path, decoder):
    write(tmp_path, "tokenizer_config.json", {"added_tokens_decoder": decoder})
    write(tmp_path, "special_tokens_map.json", {"eos_token": "legacy"})
    with pytest.raises(ValueError, match="added_tokens_decoder must be an object"):
        resolve_token_roles(tmp_path)


@pytest.mark.parametrize("modern,config_value,legacy_value,expected_token,expected_ids", [
    (True, "configured", "legacy", "configured", (1, 2)),
    (False, "configured", "legacy", "legacy", (1, 3)),
    (True, None, "legacy", "absent", (1,)),
    (False, "configured", None, "absent", (1,)),
])
def test_chat_rendering_and_reply_stops_share_the_same_authoritative_eos_role(
        tmp_path, modern, config_value, legacy_value, expected_token, expected_ids):
    config = {"eos_token": config_value, "chat_template": "{{ eos_token|default('absent') }}"}
    if modern:
        config["added_tokens_decoder"] = {}
    write(tmp_path, "tokenizer_config.json", config)
    write(tmp_path, "special_tokens_map.json", {"eos_token": legacy_value})
    write(tmp_path, "tokenizer.json", {"model": {"vocab": {
        "unknown": 0, "model_stop": 1, "configured": 2, "legacy": 3}}})
    template = ChatTemplate(tmp_path)
    assert template.render([{"role": "user", "content": "hello"}], tools=None,
                           enable_thinking=True) == expected_token
    assert resolve_eos(tmp_path, 1, vocab_size=4) == expected_ids


def test_modern_roles_do_not_read_unused_legacy_file_and_override_bypasses_role_metadata(tmp_path):
    write(tmp_path, "tokenizer_config.json", {"eos_token": "configured", "added_tokens_decoder": {},
                                               "chat_template": "{{ eos_token }}"})
    write(tmp_path, "tokenizer.json", {"model": {"vocab": {"configured": 2}}})
    (tmp_path / "special_tokens_map.json").write_text("not JSON")
    assert resolve_token_roles(tmp_path) == {"eos_token": "configured"}
    assert resolve_eos(tmp_path, 1, vocab_size=4) == (1, 2)
    assert ChatTemplate(tmp_path).specials == {"eos_token": "configured"}
    write(tmp_path, "tokenizer_config.json", {"eos_token": {}, "added_tokens_decoder": None})
    assert resolve_eos(tmp_path, 1, vocab_size=4, override=[0]) == (0,)

"""Selected model chat source is deterministic, captured, and never synthesized."""
import json

import pytest

from tensorfold.chat_metadata import resolve_chat_templates


def config(directory, value):
    (directory / "tokenizer_config.json").write_text(json.dumps({"chat_template": value}))


@pytest.mark.parametrize("value", ["default source", {"default": "default source", "tool_use": "tool source"},
                                  [{"name": "default", "template": "default source"},
                                   {"name": "tool_use", "template": "tool source"}]])
def test_single_mapping_and_list_select_default_and_tool_sources(tmp_path, value):
    config(tmp_path, value)
    default, tools = resolve_chat_templates(tmp_path)
    assert default.source == "default source"
    assert tools.source == ("default source" if isinstance(value, str) else "tool source")


def test_standalone_default_and_hf_named_files_override_embedded_entries(tmp_path):
    config(tmp_path, {"default": "embedded default", "tool_use": "embedded tools"})
    (tmp_path / "chat_template.jinja").write_text("standalone default")
    named = tmp_path / "additional_chat_templates"
    named.mkdir()
    (named / "tool_use.jinja").write_text("standalone tools")
    default, tools = resolve_chat_templates(tmp_path)
    assert (default.source, default.origin) == ("standalone default", "chat_template.jinja")
    assert (tools.source, tools.origin) == ("standalone tools", "additional_chat_templates/tool_use.jinja")
    (named / "tool_use.jinja").write_text("changed later")
    assert tools.source == "standalone tools"  # active ownership is captured, not a path lookup


def test_explicit_named_template_selects_one_source_for_both_routes(tmp_path):
    config(tmp_path, {"default": "normal", "tool_use": "tools", "rag": "retrieval"})
    default, tools = resolve_chat_templates(tmp_path, template_name="rag")
    assert default is tools and default.source == "retrieval" and default.name == "rag"


def test_explicit_file_takes_precedence_without_reading_unused_embedded_metadata(tmp_path):
    (tmp_path / "tokenizer_config.json").write_text("not JSON")
    path = tmp_path / "operator.jinja"
    path.write_text("explicit source")
    default, tools = resolve_chat_templates(tmp_path, template_file=path)
    assert default is tools and default.source == "explicit source" and default.origin == str(path)


def test_legacy_processor_file_can_supply_an_absent_default(tmp_path):
    (tmp_path / "chat_template.json").write_text(json.dumps({"chat_template": "legacy source"}))
    assert resolve_chat_templates(tmp_path)[0].source == "legacy source"


@pytest.mark.parametrize("name", ["", "../outside", "/absolute", "..", "a\\b", "missing"])
def test_bad_or_missing_explicit_name_is_refused(tmp_path, name):
    config(tmp_path, "default")
    with pytest.raises(ValueError):
        resolve_chat_templates(tmp_path, template_name=name)


@pytest.mark.parametrize("value", [[], {}, {"only_named": "text"}, ["text"],
                                  [{"name": "default", "template": "a"}, {"name": "default", "template": "b"}]])
def test_missing_default_or_ambiguous_metadata_is_not_a_fallback_template(tmp_path, value):
    config(tmp_path, value)
    with pytest.raises(ValueError):
        resolve_chat_templates(tmp_path)

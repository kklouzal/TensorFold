"""Captured model-owned chat-template selection, shared by backend consumers.

The pinned HF storage contract uses chat_template.jinja for default and
additional_chat_templates/<name>.jinja for named sources. Standalone files
override legacy tokenizer_config string/mapping/name-template-list entries.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from tensorfold.checkpoint_metadata import read_json, read_text


class MissingChatTemplate(ValueError):
    """No template metadata exists; bare model execution may omit a chat frontend."""


@dataclass(frozen=True)
class TemplateSource:
    name: str
    source: str
    origin: str

    @property
    def sha256(self):
        return hashlib.sha256(self.source.encode("utf-8")).hexdigest()


def _named(value, origin):
    if value is None:
        return {}
    if isinstance(value, str):
        value = {"default": value}
    elif isinstance(value, list):
        result = {}
        for entry in value:
            if not isinstance(entry, dict) or "name" not in entry or "template" not in entry:
                raise ValueError("named chat templates require name/template objects")
            name = entry["name"]
            if type(name) is not str or not name or name in result:
                raise ValueError("chat template names must be unique nonempty strings")
            result[name] = entry["template"]
        value = result
    if not isinstance(value, dict) or any(type(name) is not str or not name or type(source) is not str
                                          for name, source in value.items()):
        raise ValueError("chat_template must be a string, named mapping or name/template list")
    return {name: TemplateSource(name, source, origin if name == "default" else f"{origin}[{name}]")
            for name, source in value.items()}


def resolve_chat_templates(model_dir, *, template_file=None, template_name=None):
    """Return captured default/tool route sources; explicit FILE/NAME fixes both routes.

    A missing default is an explicit error, as in HF named-template selection.
    Only selected routes are read/compiled; unrelated named files need not be
    enumerated or materialized. Explicit names are single file components.
    """
    if template_file is not None and template_name is not None:
        raise ValueError("chat_template_file and chat_template_name are mutually exclusive")
    if template_file is not None:
        path = Path(template_file)
        source = TemplateSource("override", read_text(path.parent, path.name, required=True), str(path))
        return source, source
    if template_name is not None and (type(template_name) is not str or not template_name
                                      or template_name in (".", "..")
                                      or any(char in template_name for char in ("/", "\\", "\x00"))):
        raise ValueError("chat_template_name must be a nonempty template name, not a path")
    config = read_json(model_dir, "tokenizer_config.json") or {}
    sources = _named(config.get("chat_template"), "tokenizer_config.chat_template")
    legacy = read_json(model_dir, "chat_template.json")
    if legacy is not None and not sources:
        sources = _named(legacy.get("chat_template"), "chat_template.json")
    default = read_text(model_dir, "chat_template.jinja")
    if default is not None:
        sources["default"] = TemplateSource("default", default, "chat_template.jinja")
    names = (template_name,) if template_name is not None else ("default", "tool_use")
    for name in names:
        filename = f"additional_chat_templates/{name}.jinja"
        source = read_text(model_dir, filename)
        if source is not None:
            sources[name] = TemplateSource(name, source, filename)
    selected = "default" if template_name is None else template_name
    if selected not in sources:
        if not sources and template_name is None:
            raise MissingChatTemplate("no model chat template metadata")
        raise ValueError("no selected chat template; provide a model default or explicit chat_template_name/file")
    default = sources[selected]
    return default, sources.get("tool_use", default) if template_name is None else default

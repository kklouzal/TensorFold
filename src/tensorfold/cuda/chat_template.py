"""The CUDA server's chat template: the checkpoint's own Jinja template, rendered as Hugging Face does."""
from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorfold.chat_metadata import MissingChatTemplate, TemplateSource, resolve_chat_templates
from tensorfold.checkpoint_metadata import read_json
from tensorfold.server.messages import _normalize_tool_call_arguments, late_system_role, normalize_messages
from tensorfold.server.request_options import effort_levels

from .tensor_file import _regular_stream, checkpoint_path


@dataclass(frozen=True)
class CompiledTemplate:
    source: TemplateSource
    template: Any
    efforts: frozenset[str]
    late_system: str | None
    sha256: str


class ChatTemplate:
    """The model's own Jinja chat template, rendered the way Hugging Face's apply_chat_template does."""

    def __init__(self, model_dir: Path, *, template_file=None, template_name=None):
        import jinja2
        import jinja2.ext
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        cfg = read_json(model_dir, "tokenizer_config.json") or {}
        self.options = (None if template_file is None else str(Path(template_file).expanduser().resolve()), template_name)
        special = read_json(model_dir, "special_tokens_map.json") or {}
        sources = resolve_chat_templates(model_dir, template_file=template_file, template_name=template_name)

        def tojson(x, ensure_ascii=False, indent=None, separators=None, sort_keys=False):
            return json.dumps(x, ensure_ascii=ensure_ascii, indent=indent, separators=separators, sort_keys=sort_keys)

        def raise_exception(message):
            raise jinja2.exceptions.TemplateError(message)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True,
                                            extensions=[jinja2.ext.loopcontrols])
        env.filters["tojson"] = tojson
        env.globals["raise_exception"] = raise_exception
        env.globals["strftime_now"] = lambda fmt: datetime.now().strftime(fmt)
        self.specials = {key: (value.get("content") if isinstance(value, dict) else value)
                         for key in ("bos_token", "eos_token", "pad_token", "unk_token")
                         if (value := cfg.get(key, special.get(key))) is not None}
        if any(type(value) is not str for value in self.specials.values()):
            raise ValueError("template special tokens must be strings or AddedToken content")
        compiled, routes = {}, []
        probe_tools = [{"type": "function", "function": {"name": "tf_template_probe",
                        "parameters": {"type": "object", "properties": {}}}}]
        for route, source in enumerate(sources):
            if source not in compiled:
                template = env.from_string(source.source)
                compiled[source] = template, effort_levels(source.source), source.sha256
            template, efforts, fingerprint = compiled[source]
            late = late_system_role(lambda messages, template=template, tools=probe_tools if route else None:
                template.render(**self.specials, messages=messages, tools=tools, add_generation_prompt=False))
            routes.append(CompiledTemplate(source, template, efforts, late, fingerprint))
        self._routes = tuple(routes)
        active = self._routes[0]
        self.template, self.efforts, self.late_system = active.template, active.efforts, active.late_system

    def selected(self, *, tools=False):
        """The captured compilation used by this request route; no metadata I/O or compilation."""
        return self._routes[int(bool(tools))]

    def provenance(self, *, tools=False):
        active = self.selected(tools=tools)
        return {"name": active.source.name, "source": active.source.origin, "sha256": active.sha256}

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
               enable_thinking: bool, extra: dict[str, Any] | None = None, allow_images: bool = False) -> str:
        active = self.selected(tools=tools)
        messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=active.late_system,
                                                                     allow_images=allow_images))
        kwargs = dict(self.specials, messages=messages, tools=tools or None, add_generation_prompt=True,
                      enable_thinking=enable_thinking)
        kwargs.update(extra or {})
        return active.template.render(**kwargs)


def prepare_chat_template(model_dir, *, template_file=None, template_name=None):
    """Capture before engine preparation; bare model APIs may have no chat metadata."""
    if template_file is None and template_name is None:
        root = Path(model_dir)
        path = checkpoint_path(root, "tokenizer.json")
        if not stat.S_ISDIR(root.stat().st_mode):
            raise NotADirectoryError(f"{root}: checkpoint metadata root must be a directory")
        try:
            (root / "tokenizer.json").lstat()
        except FileNotFoundError:
            return None
        # Establish presence/type without parsing the tokenizer twice. The
        # immutable checkpoint contract spans this admission and its later read.
        with _regular_stream(path):
            pass
    try:
        return ChatTemplate(model_dir, template_file=template_file, template_name=template_name)
    except MissingChatTemplate:
        if template_file is not None or template_name is not None:
            raise
        return None

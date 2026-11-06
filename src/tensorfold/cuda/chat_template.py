"""The CUDA server's chat template: the checkpoint's own Jinja template, rendered as Hugging Face does."""
from __future__ import annotations

import json
import stat
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from tensorfold.chat_metadata import MissingChatTemplate, TemplateSource, resolve_chat_templates
from tensorfold.server.messages import _normalize_tool_call_arguments, late_system_role, normalize_messages
from tensorfold.server.request_options import effort_levels
from tensorfold.server.prompt_literals import (MAX_RENDER_BYTES, PromptEncodingError, PromptLiterals,
                                              RenderedPrompt, bounded_render)
from tensorfold.server.prompt_origin import origin_environment, trust_message_roles, trust_policy_strings, trusted
from tensorfold.token_metadata import resolve_token_roles, tokenizer_override
from tensorfold.checkpoint_metadata import read_json

from tensorfold.tensor_file import _regular_stream, checkpoint_path


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

        self.options = (None if template_file is None else str(Path(template_file).expanduser().resolve()), template_name)
        self.specials = resolve_token_roles(model_dir)
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
        env = origin_environment(env)
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
        override = tokenizer_override()
        tokenizer_data = json.loads(override.text) if override is not None else (read_json(model_dir, "tokenizer.json") or {})
        self.literals = PromptLiterals(tokenizer_data, [route.source.source for route in self._routes],
                                       self.specials.values())

    def selected(self, *, tools=False):
        """The captured compilation used by this request route; no metadata I/O or compilation."""
        return self._routes[int(bool(tools))]

    def provenance(self, *, tools=False):
        active = self.selected(tools=tools)
        return {"name": active.source.name, "source": active.source.origin, "sha256": active.sha256}

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None,
               enable_thinking: bool, extra: dict[str, Any] | None = None, allow_images: bool = False,
               allow_audio: bool = False, max_render_bytes: int = MAX_RENDER_BYTES,
               add_generation_prompt: bool = True) -> str:
        active = self.selected(tools=tools)
        messages = _normalize_tool_call_arguments(normalize_messages(messages, late_system=active.late_system,
                                                                     allow_images=allow_images,
                                                                     allow_audio=allow_audio,
                                                                     audio_placeholders=allow_audio))
        try:
            if extra is not None and not isinstance(extra, dict):
                raise PromptEncodingError("chat template kwargs must be a mapping")
            if set(extra or {}) & {"messages", "tools", "add_generation_prompt"}:
                raise PromptEncodingError("chat template kwargs cannot replace prepared messages, tools or generation mode")
            if type(add_generation_prompt) is not bool:
                raise PromptEncodingError("generation prompt mode must be a boolean")
            values, _ = self.literals.protect({"messages": messages, "tools": tools or None,
                                              "extra": extra or {}}, max_bytes=max_render_bytes)
            kwargs = dict({key: trusted(value) for key, value in self.specials.items()},
                          messages=trust_message_roles(values["messages"]), tools=values["tools"],
                          add_generation_prompt=add_generation_prompt, enable_thinking=enable_thinking)
            kwargs.update(values["extra"])
            trust_policy_strings(kwargs)
            text = bounded_render(active.template, kwargs, max_render_bytes)
            return RenderedPrompt(text, self.literals.literals)
        except (PromptEncodingError, UnicodeError) as exc:
            from jinja2.exceptions import TemplateError

            raise TemplateError(str(exc)) from exc

    def encode(self, tokenizer, text, *, segments=None):
        """Count and generate from the same provenance-aware token sequence."""
        return self.literals.encode(tokenizer, text, segments=segments)


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

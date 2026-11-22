"""DeepSeek-V4.1-Flash on the CUDA OpenAI server: prompts and tool calls in DeepSeek's own V4.1 format.

``V41ChatTemplate`` stands where the server holds a checkpoint's Jinja template (``cuda.chat_template.ChatTemplate``,
same methods): it renders through DeepSeek's V4.1 encoder (``prompts_v41``; the checkpoint's Jinja export is not
that authority), and ``calls`` reads the replies' DSML blocks::

    <｜DSML｜ calls>
    <｜DSML｜ invoke name="NAME">
    <｜DSML｜ parameter name="KEY" string="true|false">VALUE</｜DSML｜ parameter>
    </｜DSML｜ invoke>
    </｜DSML｜ calls>

A block is read by the vendored encoder's ``parse_tool_calls`` (DeepSeek's exact framing; string="false" values
are JSON), whole or not at all: a block that does not parse, or names a tool the request did not offer, stays
reply text. Streams send a block's calls once it closes, so streamed and final calls are the same.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tensorfold.markup_lex import DSML41_CALLS, markup_regions

from .prompts_v41 import SOURCE, render_literals
from .vendor import encoding_dsv41 as native

OPENER, CLOSER = DSML41_CALLS
EFFORTS = frozenset(("low", "medium", "high", "xhigh", "max"))   # names the encoder hears (prompts_v41.EFFORTS)
_SOURCES = ("prompts_v41.py", "literal_prompts_v41.py", "vendor/encoding_dsv41.py")
# what the server's startup note and late-system probe read in place of Jinja source
_DESCRIPTION = (f"{SOURCE}: thinking by enable_thinking (chat_template_kwargs.thinking), reasoning_effort "
                "none/low/medium/high/xhigh/max or 1-100")


def _no_constant(name: str) -> Any:
    raise ValueError(f"{name} is not a JSON value")


def decode_block(block: str) -> list[tuple[str, dict[str, Any]]] | None:
    """A complete ``<｜DSML｜ calls>`` block's calls as (name, arguments), or None where DeepSeek's parser refuses it,
    it holds no call, or an argument object is not JSON (a namespaced call keeps its ``NAMESPACE::NAME``)."""

    if not block.startswith(OPENER):
        return None
    try:
        # the vendored parser starts after "<｜DSML｜ calls" and checks the ">\n" framing itself
        end, stop, found = native.parse_tool_calls(len(OPENER) - 1, block)
    except (ValueError, AssertionError):
        return None
    if stop != CLOSER or end != len(block) or not found:
        return None
    calls = []
    for call in found:
        name = call["name"] if call.get("namespace") is None else f"{call['namespace']}::{call['name']}"
        try:
            arguments = json.loads(call["arguments"], parse_constant=_no_constant)
        except ValueError:
            return None
        if not isinstance(arguments, dict):
            return None
        calls.append((name, arguments))
    return calls


def _known(tools: list[dict[str, Any]] | None) -> dict[str, str]:
    from tensorfold.server.tools import tool_spec_name

    return {tool_spec_name(tool).lower(): tool_spec_name(tool) for tool in tools or ()}


def _blocks(text: str, finished: bool):
    return [region for region in markup_regions(text, finished=finished)
            if region.kind == "call" and region.opener == OPENER]


def parse(text: str, tools: list[dict[str, Any]], *, max_calls: int | None = None,
          finished: bool = True) -> tuple[str, list[dict[str, Any]] | None]:
    """(content, OpenAI calls or None) of a finished reply's answer, as ``server.tools.parse_tool_calls_from_content``
    reads other formats: content is the text outside the parsed blocks; a one-call request keeps the first call."""

    from tensorfold.server.tools import _openai_tool_call

    if not tools:
        return text, None
    known = _known(tools)
    calls: list[dict[str, Any]] = []
    residue: list[str] = []
    cursor = 0
    for region in _blocks(text, finished):
        if not region.complete:
            continue
        residue.append(text[cursor:region.start])
        cursor = region.end
        if max_calls is not None and len(calls) >= max_calls:
            continue
        found = decode_block(text[region.start:region.end])
        if not found or any(name.lower() not in known for name, _ in found):
            if max_calls is None:           # a malformed or unoffered block is the reply's content
                residue.append(text[region.start:region.end])
            continue
        for name, arguments in found:
            if max_calls is None or len(calls) < max_calls:
                calls.append(_openai_tool_call(name, arguments, known))
    residue.append(text[cursor:])
    return "".join(residue).strip(), calls or None


def hide(text: str, *, finished: bool) -> str:
    """The answer without its DSML blocks; while streaming, a tail that may begin one is held back."""

    from tensorfold.server.text import hide_tool_calls

    return hide_tool_calls(text, finished=finished, openers=(OPENER, OPENER.lower()))


class CallStreamer:
    """A streamed reply's calls as OpenAI deltas, each block's once it closes and parses (the final parser's calls)."""

    complete = True              # calls are sent only from closed blocks

    def __init__(self, tools: list[dict[str, Any]] | None) -> None:
        self.known = _known(tools)
        self.pos = 0                 # the answer offset read so far
        self.index = -1              # the last call sent
        self.streamed = False

    def feed(self, text: str, *, finished: bool = False) -> list[dict[str, Any]]:
        from tensorfold.server.tools import _openai_tool_call, stream_tool_call_deltas

        out: list[dict[str, Any]] = []
        for region in _blocks(text, finished):
            if region.start < self.pos:
                continue
            if not region.complete:
                break
            self.pos = region.end
            found = decode_block(text[region.start:region.end])
            if not found or any(name.lower() not in self.known for name, _ in found):
                continue
            for name, arguments in found:
                self.index += 1
                self.streamed = True
                for delta in stream_tool_call_deltas([_openai_tool_call(name, arguments, self.known)]):
                    for part in delta["tool_calls"]:
                        part["index"] = self.index
                    out.append(delta)
        return out


class Calls:
    """The server's call grammar for DeepSeek-V4.1 replies (``App.run`` reads it from ``template.calls``)."""

    parse = staticmethod(parse)
    hide = staticmethod(hide)
    streamer = CallStreamer


@dataclass(frozen=True)
class _Source:
    name: str
    source: str
    origin: str
    sha256: str


class _Probe:
    """``route.template``: the Jinja-like ``render(**values)`` the CUDA server's marker probes call."""

    def __init__(self, owner: "V41ChatTemplate") -> None:
        self._owner = owner

    def render(self, *, messages, tools=None, add_generation_prompt=False, enable_thinking=False, **values) -> str:
        extra = {"reasoning_effort": values["reasoning_effort"]} if "reasoning_effort" in values else None
        return str(self._owner.render(messages, tools=tools, enable_thinking=enable_thinking, extra=extra,
                                      add_generation_prompt=add_generation_prompt))


@dataclass(frozen=True)
class _Route:
    source: _Source
    template: _Probe
    efforts: frozenset[str]
    late_system: str
    sha256: str
    numeric_efforts: bool = True     # reasoning_effort may be an integer 1-100


class V41ChatTemplate:
    """DeepSeek-V4.1 prompts for the CUDA server, by DeepSeek's V4.1 encoder (``cuda.chat_template.ChatTemplate``'s
    methods). Tools join the first system message, as DeepSeek's ``load_cases`` attaches them."""

    options = (None, None)           # no --chat-template selection: the encoder is the only renderer
    calls = Calls

    @staticmethod
    def strict_tool_plan(tools, *, required, parallel, offered):
        from .strict_tools_v41 import Plan

        return Plan(tools, required=required, parallel=parallel, offered=offered)

    def __init__(self, model_dir: str | Path) -> None:
        from tokenizers import Tokenizer

        from tensorfold.server.prompt_literals import PromptLiterals
        from tensorfold.token_metadata import tokenizer_json

        from .literal_prompts_v41 import reserved_tokens

        text = tokenizer_json(model_dir)
        self.specials: dict[str, str] = {}
        self.literals = PromptLiterals(json.loads(text), [SOURCE], reserved_tokens(Tokenizer.from_str(text)))
        digest = hashlib.sha256()
        here = Path(__file__).parent
        for name in _SOURCES:
            digest.update(name.encode())
            digest.update((here / name).read_bytes())
        sha = digest.hexdigest()
        self._route = _Route(_Source("deepseek_v41", _DESCRIPTION, SOURCE, sha), _Probe(self), EFFORTS, "system", sha)

    def selected(self, *, tools=False) -> _Route:
        return self._route

    def provenance(self, *, tools=False) -> dict[str, str]:
        return {"name": "deepseek_v41", "source": SOURCE, "sha256": self._route.sha256}

    def render(self, messages: list[dict[str, Any]], *, tools: list[dict[str, Any]] | None, enable_thinking: bool,
               extra: dict[str, Any] | None = None, allow_images: bool = False, allow_audio: bool = False,
               max_render_bytes: int | None = None, add_generation_prompt: bool = True, render_credit=None,
               render_time=None) -> Any:
        """The prompt (a ``RenderedPrompt``); ``extra`` (chat_template_kwargs) gives ``reasoning_effort``, its other
        keys are unread. Refusals raise jinja2's ``TemplateError``, as the Jinja renderer's do (HTTP 400)."""

        from jinja2.exceptions import TemplateError

        from tensorfold.server.messages import normalize_messages
        from tensorfold.server.prompt_literals import MAX_RENDER_BYTES, PromptEncodingError

        if allow_audio:
            raise TemplateError("DeepSeek-V4.1 takes no audio inputs")
        # leading instructions merged; later system turns kept; image parts only where the server's vision frontend
        # asks (``allow_images``: it expands each image's one placeholder)
        messages = normalize_messages(messages, allow_images=bool(allow_images))
        limit = MAX_RENDER_BYTES if max_render_bytes is None else max_render_bytes
        try:
            if extra is not None and not isinstance(extra, dict):
                raise PromptEncodingError("chat template kwargs must be a mapping")
            if type(add_generation_prompt) is not bool:
                raise PromptEncodingError("generation prompt mode must be a boolean")
            if render_credit is not None:
                render_credit(0, 0)
            text = render_literals(messages, self.literals, tools=tools or None, thinking=bool(enable_thinking),
                                   reasoning_effort=(extra or {}).get("reasoning_effort"),
                                   add_generation_prompt=add_generation_prompt, max_bytes=limit,
                                   images=bool(allow_images))
            if render_credit is not None:
                render_credit(len(text), 1)
            if len(str.__str__(text).encode("utf-8")) > limit:
                raise PromptEncodingError("rendered prompt exceeds the chat-render byte budget")
            return text
        except (ValueError, AssertionError, NotImplementedError, KeyError) as exc:
            # the encoder's own refusals (bad effort, role, tool name or namespace) and literal-encoding limits
            raise TemplateError(str(exc)) from exc

    def encode(self, tokenizer, text, *, segments=None, ids_credit=None, cache_credit=None):
        """The prompt's token ids, request text never becoming a control token."""
        options = {}
        if ids_credit is not None:
            options["ids_credit"] = ids_credit
        if cache_credit is not None:
            options["cache_credit"] = cache_credit
        return self.literals.encode(tokenizer, text, segments=segments, **options)


def chat_template(model_dir: str | Path, *, template_file=None, template_name=None) -> V41ChatTemplate:
    if template_file is not None or template_name is not None:
        raise ValueError("DeepSeek-V4.1 prompts are rendered by DeepSeek's own V4.1 encoder, whose DSML the server "
                         "reads tool calls in: drop --chat-template and --chat-template-name")
    return V41ChatTemplate(model_dir)

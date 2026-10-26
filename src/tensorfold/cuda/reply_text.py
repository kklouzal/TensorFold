"""The CUDA server's reply text: streamed decoding, think blocks and tool-call parsing (the Mac lane server's rules)."""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from tensorfold.server.stopping import StopPolicy
from tensorfold.server.text import _STRAY_QWEN_CLOSES_RE, _partial_tag
from tensorfold.server.tools import normalize_qwen_tool_markup, parse_glm_tool_call_block
from tensorfold.tool_parameters import decode_parameter, parameter_schemas

_CALL_OPEN, _CALL_CLOSE = "<tool_call>", "</tool_call>"
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.IGNORECASE | re.DOTALL)
_TOOL_FUNCTION_BLOCK_RE = re.compile(r"^\s*<function=([^>\s]+)>\s*(.*?)\s*</function>\s*$", re.IGNORECASE | re.DOTALL)
_TOOL_PARAMETER_BLOCK_RE = re.compile(r"<parameter=([^>\s]+)>\n?(.*?)\n?</parameter>", re.IGNORECASE | re.DOTALL)


class StreamDecoder:
    """Decode a shared token window to preserve leading spaces and byte boundaries, deferring incomplete characters."""

    def __init__(self, tok, skip: tuple[int, ...] = ()):
        self.tok = tok
        self.skip = (frozenset(skip) | getattr(tok, "control_token_ids", frozenset())
                     | getattr(tok, "unused_token_ids", frozenset()))
        self.ids: list[int] = []
        self.text = ""
        self.prefix = 0             # window start
        self.read = 0               # tokens already reflected in ``text``

    def _decode(self, ids: list[int]) -> str:
        return self.tok.decode(ids, skip_special_tokens=False)

    def add(self, new: list[int]) -> str:
        self.ids.extend(t for t in new if t not in self.skip)
        before = self._decode(self.ids[self.prefix:self.read])
        after = self._decode(self.ids[self.prefix:])
        if len(after) > len(before) and not after.endswith("\ufffd"):
            self.text += after[len(before):]
            self.prefix, self.read = self.read, len(self.ids)
        return self.text

    def final(self) -> str:
        """Everything, including a trailing partial character (as decoding it all at once gives)."""

        before = self._decode(self.ids[self.prefix:self.read])
        return self.text + self._decode(self.ids[self.prefix:])[len(before):]


class StopStrings:
    """A request's ``stop`` strings, decided token by token, so a reply cuts at the same token however its tokens arrive."""

    def __init__(self, strings: tuple[str, ...], tok, skip: tuple[int, ...] = ()):
        self.strings, self.tok = strings, tok
        self.skip = (frozenset(skip) | getattr(tok, "control_token_ids", frozenset())
                     | getattr(tok, "unused_token_ids", frozenset()))
        # a new match lies in the last (its UTF-8 length) tokens, plus eight as the Mac's ``StopPolicy`` keeps
        self.tail = max((len(s.encode()) for s in strings), default=0) + 8

    def hit(self, tokens: list[int]) -> bool:
        """Whether the text through the newest token holds a stop string (earlier tokens were checked already)."""

        if not self.strings:
            return False
        text = self.tok.decode(self.tail_ids(tokens), skip_special_tokens=False)
        return any(stop in text for stop in self.strings)

    def tail_ids(self, tokens, *, start=0, limit=None) -> list[int]:
        """Bound visible context in tokens[start:] without copying a retained history."""
        limit = self.tail if limit is None else limit
        if limit == 0:
            return []
        tail = []
        for index in range(len(tokens) - 1, start - 1, -1):
            token = tokens[index]
            if token not in self.skip:
                tail.append(token)
                if len(tail) == limit:
                    break
        tail.reverse()
        return tail

    def visible(self, text: str, *, partial: bool = False) -> str:
        """The text before the first match; ``partial`` also holds back an end that may begin one (the Mac's rule)."""

        return StopPolicy.visible(self, text, partial=partial)


def stop_prefix_limit(stops: StopStrings, baseline, prefix: list[int], new: list[int]) -> tuple[int, bool]:
    """Pure precommit cut, including a stop at the exact end of a verified block.

    ``baseline`` is generated text carried into a gated continuation; ``prefix``
    is this generation run's output. Keep only the tail used by StopStrings,
    rather than copying an arbitrarily long generated history on each token.
    The tokenizer and stop strings are shared immutable request inputs.
    """
    if not stops.strings or not any(token not in stops.skip for token in new):
        return len(new), False
    tail = stops.tail_ids(prefix)
    if len(tail) < stops.tail:
        tail = [*stops.tail_ids(baseline, limit=stops.tail - len(tail)), *tail]
    for count, token in enumerate(new, 1):
        if token in stops.skip:
            continue
        tail.append(token)
        if stops.hit(tail):
            return count, True
        if len(tail) > stops.tail:
            del tail[:-stops.tail]
    return len(new), False


def hide_tool_calls(text: str, *, finished: bool) -> str:
    out: list[str] = []
    pos = 0
    while True:
        start = text.find(_CALL_OPEN, pos)
        if start < 0:
            tail = text[pos:]
            out.append(tail[: len(tail) - (0 if finished else _partial_tag(tail, _CALL_OPEN))])
            return "".join(out)
        out.append(text[pos:start])
        end = text.find(_CALL_CLOSE, start)
        if end < 0:
            return "".join(out)
        # a doubled closer after a call is part of its markup (the end parser collapses it), not visible text
        pos = _STRAY_QWEN_CLOSES_RE.match(text, end + len(_CALL_CLOSE)).end()
        if not finished and _CALL_CLOSE.startswith(text[pos:].lstrip()):
            return "".join(out)          # a doubled closer may still be arriving: hold the whitespace before it too


def _tool_name(tool: dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool, dict) else None
    return str((fn or tool).get("name") or "").strip() if isinstance(tool, dict) else ""


def parse_tool_calls(text: str, tools: list[dict[str, Any]], *, max_calls: int | None = None) -> tuple[str, list[dict[str, Any]] | None]:
    """Qwen ``<function=name><parameter=k>v</parameter></function>``, GLM ``name<arg_key>..`` or JSON calls."""

    if not tools:
        return text, None
    known = {_tool_name(t).lower(): _tool_name(t) for t in tools}
    schemas = parameter_schemas(tools)
    text = normalize_qwen_tool_markup(text)
    calls: list[dict[str, Any]] = []
    residue: list[str] = []
    cursor = 0
    for match in _TOOL_CALL_BLOCK_RE.finditer(text):
        residue.append(text[cursor:match.start()])
        cursor = match.end()
        if max_calls is not None and len(calls) >= max_calls:
            continue                                  # a one-call request drops the calls past its first
        block = match.group(1).strip()
        name, args = None, {}
        try:
            payload = json.loads(block)
            if isinstance(payload, dict):
                fn = payload.get("function") if isinstance(payload.get("function"), dict) else payload
                name = fn.get("name")
                args = fn.get("arguments", fn.get("parameters", {}))
                if isinstance(args, str):
                    args = json.loads(args) if args.strip() else {}
        except (json.JSONDecodeError, AttributeError):
            m = _TOOL_FUNCTION_BLOCK_RE.match(block)
            if m:
                if max_calls is not None and _TOOL_PARAMETER_BLOCK_RE.sub("", m.group(2)).strip():
                    residue.append(match.group(0))
                    continue
                name = m.group(1).strip()
                # typed parameters (array, object, number...) decode per the tool's schema, as the Mac server does
                props = schemas.get(name.lower(), {})
                args = {key: decode_parameter(value, props.get(key, {}))
                        for key, value in ((p.group(1).strip(), p.group(2))
                                           for p in _TOOL_PARAMETER_BLOCK_RE.finditer(m.group(2)))}
            else:
                glm = parse_glm_tool_call_block(block, tools, complete=max_calls is not None)
                if glm is not None:
                    name, args = glm
        # a block that is not a call of an offered tool stays text, in either mode; a one-call request's content
        # filter hides tool markup, and the server shows this text only when nothing else would answer
        if not name or str(name).lower() not in known:
            residue.append(match.group(0))
            continue
        if max_calls is not None:
            try:
                if not isinstance(args, dict):
                    raise TypeError("arguments are not an object")
                json.dumps(args, allow_nan=False)
            except (ValueError, TypeError):
                residue.append(match.group(0))
                continue
        calls.append({"id": f"call_{uuid.uuid4().hex[:24]}", "type": "function",
                      "function": {"name": known[str(name).lower()],
                                   "arguments": json.dumps(args, ensure_ascii=False, separators=(",", ":"))}})
    residue.append(text[cursor:])
    return "".join(residue).strip(), calls or None


# -- chat template -------------------------------------------------------------------------

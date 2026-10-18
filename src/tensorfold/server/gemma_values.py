"""Gemma's native tool-value codec; strings are raw between checkpoint sentinel tokens."""
from __future__ import annotations

import json
import math
import re
from typing import Any

from tensorfold.server.text import _partial_tag

GEMMA_STRING = '<|"|>'
GEMMA_OPEN, GEMMA_CLOSE = "<|tool_call>", "<tool_call|>"
GEMMA_KEY = re.compile(r"[A-Za-z_][\w.-]*")
THOUGHT_OPEN, THOUGHT_CLOSE = "<|channel>thought", "<channel|>"


def gemma_value(value: Any) -> str:
    """The pinned template's recursive spelling; refuse values its delimiters cannot represent."""

    if isinstance(value, str):
        if GEMMA_STRING in value:
            raise ValueError("a Gemma string cannot contain its <|\"|> delimiter")
        return GEMMA_STRING + value + GEMMA_STRING
    if value is None:
        return "None"
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, dict):
        if any(not isinstance(key, str) or not GEMMA_KEY.fullmatch(key) for key in value):
            raise ValueError("a Gemma object key cannot be written in bare-key call markup")
        return "{" + ",".join(key + ":" + gemma_value(value[key]) for key in sorted(value, key=str.lower)) + "}"
    if isinstance(value, list):
        return "[" + ",".join(gemma_value(item) for item in value) + "]"
    if type(value) not in (int, float) or isinstance(value, float) and not math.isfinite(value):
        raise ValueError("a Gemma value must be finite JSON data")
    return str(value)


def gemma_json(text: str) -> str:
    """Transcode native Gemma values to JSON, leaving strings uninterpreted.

    The publisher template spells Python None recursively and writes object keys
    bare. A lexical pass avoids placeholder collisions and replacing None, keys
    or apparent call delimiters within a quoted value. The JSON decoder remains
    responsible for value syntax, nesting and complete-document validation.
    """

    parts, at = [], 0
    decoder = json.JSONDecoder()
    while at < len(text):
        if text.startswith(GEMMA_STRING, at):
            start = at + len(GEMMA_STRING)
            end = text.find(GEMMA_STRING, start)
            if end < 0:
                raise ValueError("unfinished Gemma string")
            parts.append(json.dumps(text[start:end], ensure_ascii=False))
            at = end + len(GEMMA_STRING)
        elif text[at] == '"':
            # Retain the JSON-string spelling already accepted by the lenient
            # parser, without mistaking its contents for bare Gemma tokens.
            value, end = decoder.raw_decode(text, at)
            if not isinstance(value, str):
                raise ValueError("invalid Gemma string")
            parts.append(text[at:end])
            at = end
        elif text[at].isspace() or text[at] in "{}[],:":
            parts.append(text[at])
            at += 1
        else:
            end = at + 1
            while end < len(text) and not text[end].isspace() and text[end] not in "{}[],:":
                end += 1
            word = text[at:end]
            after = end
            while after < len(text) and text[after].isspace():
                after += 1
            if after < len(text) and text[after] == ":" and GEMMA_KEY.fullmatch(word):
                word = json.dumps(word, ensure_ascii=False)
            elif word == "None":
                word = "null"
            parts.append(word)
            at = end
    return "".join(parts)


def gemma_envelopes(text: str) -> list[tuple[int, int, str]]:
    """Complete Gemma envelopes, ignoring delimiter text inside native strings."""

    found, cursor = [], 0
    while (start := text.find(GEMMA_OPEN, cursor)) >= 0:
        at, quoted = start + len(GEMMA_OPEN), False
        while at < len(text):
            if text.startswith(GEMMA_STRING, at):
                quoted = not quoted
                at += len(GEMMA_STRING)
            elif not quoted and text.startswith(GEMMA_CLOSE, at):
                cursor = at + len(GEMMA_CLOSE)
                found.append((start, cursor, text[start + len(GEMMA_OPEN):at].strip()))
                break
            else:
                at += 1
        else:
            break
    return found


def hide_gemma_calls(text: str, *, finished: bool) -> str:
    """Visible text outside complete calls; hold an unfinished envelope during SSE."""
    parts, at = [], 0
    for start, end, _ in gemma_envelopes(text):
        parts.append(text[at:start])
        at = end
    tail = text[at:]
    opening = tail.find(GEMMA_OPEN)
    if opening >= 0:
        tail = tail[:opening]
    elif not finished:
        tail = tail[:len(tail) - _partial_tag(tail, GEMMA_OPEN)]
    parts.append(tail)
    return "".join(parts)


def split_gemma_thinking(text: str, *, finished: bool) -> tuple[str, str]:
    """Structural thought channels; native tool values may contain marker text literally."""
    opens, closes, calls = [], [], []
    at, quoted, call = 0, False, None
    while (at := text.find("<", at)) >= 0:
        if call is not None and text.startswith(GEMMA_STRING, at):
            quoted = not quoted
            at += len(GEMMA_STRING)
        elif quoted:
            at += 1
        elif text.startswith(GEMMA_OPEN, at):
            if call is None:
                call = at
            at += len(GEMMA_OPEN)
        elif text.startswith(GEMMA_CLOSE, at):
            if call is not None:
                calls.append((call, at + len(GEMMA_CLOSE)))
                call = None
            at += len(GEMMA_CLOSE)
        elif call is None and text.startswith(THOUGHT_OPEN, at):
            opens.append(at)
            at += len(THOUGHT_OPEN)
        elif call is None and text.startswith(THOUGHT_CLOSE, at):
            closes.append(at)
            at += len(THOUGHT_CLOSE)
        else:
            at += 1
    if not opens:
        held = _partial_tag(text, THOUGHT_OPEN) if call is None else 0
        if finished and text[:len(text) - held].strip():
            held = 0
        # A normal token cutoff can occur after the channel token but before
        # the word thought. Only an initial proper opener prefix is held at
        # completion; arbitrary text and native call-string data remain answer.
        return "", text[:len(text) - held]
    opening = opens[0]
    start = opening + len(THOUGHT_OPEN)
    while start < len(text) and text[start] == "\n":
        start += 1
    close = next((position for position in closes if position >= start), None)
    if close is not None:
        return text[start:close], text[:opening] + text[close + len(THOUGHT_CLOSE):].lstrip("\n")
    first_call = next((position for position, _ in calls if position >= start), None)
    if first_call is None and call is not None and call >= start:
        first_call = call
    if first_call is not None:
        if finished and any(position == first_call for position, _ in calls):
            return text[start:first_call], text[:opening] + text[first_call:]
        return text[start:first_call] if not finished else text[start:], text[:opening]
    held = 0 if finished else max(_partial_tag(text, marker) for marker in (THOUGHT_CLOSE, GEMMA_OPEN))
    return text[start:len(text) - held], text[:opening]


class GemmaJSONStream:
    """An incremental lexical transcoder; buffers only an incomplete token/string delimiter."""
    def __init__(self):
        self.pending, self.quoted = "", False

    def feed(self, text: str, *, finished=False) -> str:
        self.pending += text
        out, at = [], 0
        while at < len(self.pending):
            if self.quoted:
                end = self.pending.find(GEMMA_STRING, at)
                upto = end if end >= 0 else len(self.pending) - _partial_tag(self.pending[at:], GEMMA_STRING)
                out.append(json.dumps(self.pending[at:upto], ensure_ascii=False)[1:-1])
                at = upto
                if end < 0:
                    break
                out.append('"')
                self.quoted = False
                at += len(GEMMA_STRING)
            elif self.pending.startswith(GEMMA_STRING, at):
                self.quoted = True
                out.append('"')
                at += len(GEMMA_STRING)
            elif GEMMA_STRING.startswith(self.pending[at:]):
                break
            elif self.pending[at].isspace() or self.pending[at] in "{}[],:":
                out.append(self.pending[at])
                at += 1
            else:
                end = at + 1
                while end < len(self.pending) and not self.pending[end].isspace() and self.pending[end] not in "{}[],:":
                    end += 1
                after = end
                while after < len(self.pending) and self.pending[after].isspace():
                    after += 1
                if after == len(self.pending) and not finished:
                    break
                word = self.pending[at:end]
                if after < len(self.pending) and self.pending[after] == ":" and GEMMA_KEY.fullmatch(word):
                    word = json.dumps(word, ensure_ascii=False)
                elif word == "None":
                    word = "null"
                elif word == "null":
                    raise ValueError("Gemma's captured template spells null as None")
                out.append(word)
                at = end
        self.pending = self.pending[at:]
        if finished and (self.pending or self.quoted):
            raise ValueError("unfinished Gemma value")
        return "".join(out)

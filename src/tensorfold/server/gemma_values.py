"""Gemma's native tool-value codec; strings are raw between checkpoint sentinel tokens."""
from __future__ import annotations

import json
from bisect import bisect_right
from dataclasses import replace
import math
import re
from typing import Any

from tensorfold.server.text import _partial_tag
from tensorfold.markup_lex import GEMMA_STRING, CALL_MARKS, markup_regions

GEMMA_OPEN, GEMMA_CLOSE = "<|tool_call>", "<tool_call|>"
GEMMA_KEY = re.compile(r"[A-Za-z_][\w.-]*")
THOUGHT_OPEN, THOUGHT_CLOSE = "<|channel>thought", "<channel|>"
# A template's spelling of null in call values: Google's launch template (47b6801b, as AEON ships it) falls through
# to Jinja's Python ``None``; its current template (35b4173c and later) writes JSON ``null``. Discovery selects one.
GEMMA_NULLS = ("None", "null")


def gemma_value(value: Any, *, null: str) -> str:
    """The template's recursive spelling with its ``null`` word; refuse values its delimiters cannot represent."""

    if isinstance(value, str):
        if GEMMA_STRING in value:
            raise ValueError("a Gemma string cannot contain its <|\"|> delimiter")
        return GEMMA_STRING + value + GEMMA_STRING
    if value is None:
        return null
    if type(value) is bool:
        return "true" if value else "false"
    if isinstance(value, dict):
        if any(not isinstance(key, str) or not GEMMA_KEY.fullmatch(key) for key in value):
            raise ValueError("a Gemma object key cannot be written in bare-key call markup")
        return "{" + ",".join(key + ":" + gemma_value(value[key], null=null)
                              for key in sorted(value, key=str.lower)) + "}"
    if isinstance(value, list):
        return "[" + ",".join(gemma_value(item, null=null) for item in value) + "]"
    if type(value) not in (int, float) or isinstance(value, float) and not math.isfinite(value):
        raise ValueError("a Gemma value must be finite JSON data")
    return str(value)


def gemma_json(text: str) -> str:
    """Transcode native Gemma values to JSON, leaving strings uninterpreted.

    Publisher templates spell null as Python None (launch) or JSON null (current,
    passed through) and write object keys bare; this lenient reader accepts either.
    A lexical pass avoids placeholder collisions and replacing None, keys
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
    """Complete native frames outside code/quoted data; repeated framing counted once."""
    return [(region.start, region.end, text[region.payload_start:region.payload_end].strip())
            for region in markup_regions(text, finished=True) if region.kind == 'call'
            and region.opener == GEMMA_OPEN and region.complete]


def hide_gemma_calls(text: str, *, finished: bool) -> str:
    """Hide only lexical native calls, never quoted examples or opaque strings."""
    parts, at = [], 0
    for region in markup_regions(text, finished=finished):
        if region.kind == 'call' and region.opener == GEMMA_OPEN:
            parts.append(text[at:region.start])
            at = region.end
    tail = text[at:]
    protected = markup_regions(tail)
    quoted_tail = bool(protected and protected[-1].end == len(tail) and protected[-1].kind != 'call')
    if not finished and not quoted_tail:
        if at and GEMMA_CLOSE.startswith(tail.lstrip()):
            tail = ''
        else:
            held = max(_partial_tag(tail, GEMMA_OPEN), _partial_tag(tail, GEMMA_CLOSE))
            tail = tail[:len(tail) - held]
    parts.append(tail)
    return ''.join(parts)


def prompt_opens_thought(prompt: str) -> bool:
    """Only a prompt's own literal thought suffix opens the response channel."""
    return prompt.rstrip().endswith(THOUGHT_OPEN)


_CONTROL = ('<|channel>', '<channel|>', '<|turn>', '<turn|>', '<|think|>')


def _safe_join(parts):
    """Removing reasoning/framing cannot manufacture a new executable call opener."""
    text = ''.join(part[0] for part in parts)
    boundaries, total = [], 0
    for previous, current in zip(parts, parts[1:]):
        total += len(previous[0])
        if previous[2] != current[1]:
            boundaries.append(total)
    split = set()
    for region in markup_regions(text):
        if region.kind != 'call':
            continue
        first = bisect_right(boundaries, region.start)
        for boundary in boundaries[first:]:
            if boundary >= region.start + len(region.opener):
                break
            split.add(boundary)
    for boundary in sorted(split, reverse=True):
        text = text[:boundary] + ' ' + text[boundary:]
    return text


def _clean_controls(text, *, finished):
    parts, at = [], 0
    for region in markup_regions(text):
        plain = text[at:region.start]
        parts.append(_clean_plain(plain))
        parts.append(text[region.start:region.end])
        at = region.end
    parts.append(_clean_plain(text[at:]))
    result = ''.join(parts)
    protected = markup_regions(result)
    opaque_tail = bool(protected and protected[-1].end == len(result))
    if not finished and not opaque_tail:
        held = max((_partial_tag(result, marker) for marker in (*_CONTROL, '</think>', *(o for o, _ in CALL_MARKS))), default=0)
        result = result[:len(result) - held]
    return result


def _clean_plain(text):
    while True:
        before = text
        for marker in _CONTROL:
            start = text.find(marker)
            while start >= 0:
                left, right = text[:start], text[start + len(marker):]
                # A framing boundary must not concatenate unrelated text into
                # a tool opener; separating adjacent non-whitespace is faithful
                # to that boundary rather than creating executable syntax.
                space = ' ' if left and right and not left[-1].isspace() and not right[0].isspace() else ''
                text = left + space + right
                start = text.find(marker)
        if text == before:
            return text


def split_gemma_thinking(text: str, *, finished: bool, opened: bool = False,
                         expect_thinking: bool = False, reasoning_only: bool = False) -> tuple[str, str]:
    """Partition presentation without touching quoted code or native tool data.

    Known unclosed reasoning remains reasoning, including native EOS. Unknown
    malformed channels and unframed expected-thinking prefixes are withheld
    while ambiguous, never emitted then retroactively moved between fields.
    ``reasoning_only`` is a caller-proved phase, not an invented prompt opener.
    """
    # A stopped initial thought-name prefix is framing, not an unknown answer.
    # This applies only before any non-whitespace data; opaque code/tool values
    # and noninitial invalid answers retain their independent lexical treatment.
    initial = text.lstrip()
    if (finished and not opened and not reasoning_only and initial.startswith('<|channel>')
            and THOUGHT_OPEN.startswith(initial)):
        return '', text[:len(text) - len(initial)]
    def call_payload(region):
        if region.kind != 'call':
            return None
        raw_body = text[region.payload_start:region.payload_end]
        body = raw_body.lstrip()
        body_start = region.payload_start + len(raw_body) - len(body)
        if not region.complete and not finished:
            repeat, repeat_prefix = body, region.opener
            if region.opener.lower() == '<tool_call>':
                repeat, repeat_prefix = repeat.lower(), repeat_prefix.lower()
            pending_header = not body or repeat_prefix.startswith(repeat)
            if region.opener == GEMMA_OPEN:
                pending_header = (pending_header or 'call:'.startswith(body) or ':'.startswith(body)
                                  or re.fullmatch(r'(?:call)?:[\w.-]*\s*', body) is not None)
            elif region.opener == '<｜DSML｜tool_calls>':
                lead = '<｜DSML｜invoke name="'
                pending_header = (pending_header or lead.startswith(body)
                                  or re.fullmatch(r'<｜DSML｜invoke name="[^"<>]*', body) is not None)
            else:
                pending_header = (pending_header or '<function='.startswith(body.lower())
                                  or re.fullmatch(r'<function=[^<>\s]*', body, re.IGNORECASE) is not None)
            if pending_header:
                # A bare opener can still acquire a wrapped/named payload.
                # Publishing it as thought now would retract on the next delta.
                return region.start, region.end
        if body.startswith(('{', '[')):
            try:
                value, end = json.JSONDecoder().raw_decode(body)
            except ValueError:
                # A partial wrapped call may still acquire its name/arguments;
                # hold it while ambiguous, never retroactively move sent data.
                return (region.start, region.end) if not region.complete and not finished else None
            pending = [value]
            while pending:
                item = pending.pop()
                if isinstance(item, list):
                    pending.extend(item)
                elif isinstance(item, dict):
                    function = item.get('function')
                    named = (function.get('name') or function.get('tool') or function.get('function')
                             if isinstance(function, dict) else
                             item.get('name') or item.get('tool') or function or item.get('call'))
                    if named:
                        return body_start, body_start + end
            return None
        if region.opener == GEMMA_OPEN:
            actual = re.match(r'(?:call)?:[\w.-]+\s*\{', body) is not None
        elif region.opener == '<｜DSML｜tool_calls>':
            actual = re.match(r'<｜DSML｜invoke name="[^"]+">', body) is not None
        else:
            actual = re.match(r'<function=[^>\s]+>', body, re.IGNORECASE) is not None
        return (region.start, region.end) if actual else None

    def scan(start):
        regions = [replace(region, start=region.start + start, end=region.end + start,
                           payload_start=region.payload_start + start, payload_end=region.payload_end + start)
                   for region in markup_regions(text[start:], finished=finished)]
        # Ordinary quotes/Markdown in a proved thought cannot consume native101.
        # Real call payloads retain their independent quoted-data boundaries.
        payloads = [payload for region in regions if (payload := call_payload(region)) is not None]
        closes = [match.start() for match in re.finditer(re.escape(THOUGHT_CLOSE), text[start:])]
        closes = [position + start for position in closes
                  if not any(left <= position + start < right for left, right in payloads)]
        return {region.start: region for region in regions}, closes

    starts, closes = scan(0)
    reason, answer, pending = [], [], []
    mode = 'reason' if opened or reasoning_only else 'answer'
    unframed = expect_thinking and mode == 'answer'
    at = 0
    def emit(piece):
        if reasoning_only or mode == 'reason':
            reason.append((piece, at, at + len(piece)))
        elif mode == 'unknown' or unframed:
            pending.append((piece, at, at + len(piece)))
        else:
            answer.append((piece, at, at + len(piece)))
    while at < len(text):
        region = starts.get(at)
        if region is not None:
            if mode == 'reason':
                payload = call_payload(region)
                close = text.find(THOUGHT_CLOSE, at, region.end)
                while close >= 0 and payload is not None and payload[0] <= close < payload[1]:
                    close = text.find(THOUGHT_CLOSE, close + len(THOUGHT_CLOSE), region.end)
                if close >= 0:
                    emit(text[at:close])
                    at = close
                    continue
                if payload is None:
                    held = _partial_tag(text[at:region.end], THOUGHT_CLOSE) if not finished else 0
                    emit(text[at:region.end - held])
                    at = region.end
                    if held:
                        break
                    continue
            if region.kind == 'call' and not reasoning_only and (mode != 'answer' or unframed):
                later_close = any(position >= region.end for position in closes)
                if not later_close and finished and region.complete:
                    if pending:
                        reason.extend(pending)
                        pending.clear()
                    mode, unframed = 'answer', False
                elif not later_close and not finished:
                    # A later real thought close can still make this a mention.
                    break
            emit(text[region.start:region.end])
            at = region.end
            continue
        if text.startswith('<|channel>', at):
            body = at + len('<|channel>')
            # A repeated header can still change its name as deltas arrive.
            # Never publish that name as reasoning and retract it at newline.
            if not finished and text.find('\n', body) < 0 and text.find(THOUGHT_CLOSE, body) < 0:
                break
            known = text.startswith('thought', body)
            if known:
                body += len('thought')
                while body < len(text) and text[body] in '\r\n':
                    body += 1
            else:
                newline = text.find('\n', body)
                closer = text.find(THOUGHT_CLOSE, body)
                if newline >= 0 and (closer < 0 or newline < closer):
                    body = newline + 1
            if mode == 'answer' and pending:
                answer.extend(pending)
                pending.clear()
            if mode == 'unknown' and known:
                reason.extend(pending)
                pending.clear()
            mode = 'reason' if reasoning_only or mode == 'reason' or known else 'unknown'
            unframed = False
            at = body
            continue
        if text.startswith(THOUGHT_CLOSE, at):
            if mode == 'unknown' or unframed:
                reason.extend(pending)
                pending.clear()
            mode, unframed = ('reason' if reasoning_only else 'answer'), False
            at += len(THOUGHT_CLOSE)
            while at < len(text) and text[at] in '\r\n':
                at += 1
            # Thought Markdown/quotes do not carry into the answer's own
            # independent lexical phase after a real native close.
            starts, closes = scan(at)
            continue
        if text.startswith('</think>', at):
            line = text.rfind('\n', 0, at) + 1
            end = text.find('\n', at)
            complete_line = end >= 0 or finished
            end = len(text) if end < 0 else end
            if complete_line and text[line:end].strip() == '</think>':
                if mode == 'unknown' or unframed:
                    reason.extend(pending)
                    pending.clear()
                mode, unframed = ('reason' if reasoning_only else 'answer'), False
                at = end + (1 if end < len(text) else 0)
                continue
            if not complete_line and text[line:end].strip() == '</think>':
                break
        if text.startswith('<|think|>', at):
            if mode == 'answer' and pending:
                answer.extend(pending)
                pending.clear()
            mode, unframed = 'reason', False
            at += len('<|think|>')
            continue
        marker = next((marker for marker in ('<|turn>', '<turn|>') if text.startswith(marker, at)), None)
        if marker is not None:
            at += len(marker)
            continue
        emit(text[at])
        at += 1
    if finished and pending:
        (reason if reasoning_only else answer).extend(pending)
    r = _clean_controls(_safe_join(reason), finished=finished)
    a = '' if reasoning_only else _clean_controls(_safe_join(answer), finished=finished)
    return r, a


class GemmaJSONStream:
    """An incremental lexical transcoder; buffers only an incomplete token/string delimiter.

    ``null`` is the captured template's null word (one of ``GEMMA_NULLS``); the other word is refused.
    """
    def __init__(self, *, null: str):
        if null not in GEMMA_NULLS:
            raise ValueError(f"a Gemma null word is one of {GEMMA_NULLS}")
        self.pending, self.quoted, self.null = "", False, null

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
                elif word == self.null:
                    word = "null"
                elif word in GEMMA_NULLS:
                    raise ValueError(f"Gemma's captured template spells null as {self.null}")
                out.append(word)
                at = end
        self.pending = self.pending[at:]
        if finished and (self.pending or self.quoted):
            raise ValueError("unfinished Gemma value")
        return "".join(out)

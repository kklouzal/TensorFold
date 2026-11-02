"""Lexical boundaries shared by output presentation, tools and grammar gates.

This identifies framing, not tool authorization or argument validity. Markdown
code and JSON/native quoted data stay opaque. Inside XML calls parameter text
is opaque until its parameter delimiter; Markdown in an argument is not reply
formatting. Callers retain schema/name/strict-completion policy.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

GEMMA_STRING = '<|"|>'
CALL_MARKS = (("<tool_call>", "</tool_call>"), ("<|tool_call>", "<tool_call|>"),
              ("<｜DSML｜tool_calls>", "</｜DSML｜tool_calls>"))
_PARAMETER_CLOSE = re.compile(r'</parameter>', re.IGNORECASE)
_PARAMETER_OPEN = re.compile(r'<parameter=([^>\s]+)>', re.IGNORECASE)
_DSML_PARAMETER_CLOSE = re.compile(r'</｜DSML｜parameter>')
_FUNCTION = re.compile(r'<function=[^>\s]+>', re.IGNORECASE)
_NAMESPACE = re.compile(r'<([A-Za-z_][\w.-]*):tool_call>', re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class MarkupRegion:
    kind: str
    start: int
    end: int
    opener: str = ''
    closer: str = ''
    payload_start: int = 0
    payload_end: int = 0
    complete: bool = True


def _matches(text, at, marker, insensitive=False):
    value = text[at:at + len(marker)]
    return value.lower() == marker.lower() if insensitive else value == marker


def _quote_end(text, at, quote='"', *, require_close=False):
    if text.startswith(GEMMA_STRING, at):
        end = text.find(GEMMA_STRING, at + len(GEMMA_STRING))
        return len(text) if end < 0 else end + len(GEMMA_STRING)
    delimiter = quote * 3 if text.startswith(quote * 3, at) else quote
    end = at + len(delimiter)
    while end < len(text):
        if text[end] == '\\':
            end += 2
        elif text.startswith(delimiter, end):
            return end + len(delimiter)
        else:
            end += 1
    return -1 if require_close else len(text)


def _run(text, at, char):
    end = at
    while end < len(text) and text[end] == char:
        end += 1
    return end


def _code_end(text, at, char):
    stop = _run(text, at, char)
    count = stop - at
    line = text.rfind('\n', 0, at) + 1
    fenced = count >= 3 and not text[line:at].strip(' \t\r')
    cursor = stop
    while (candidate := text.find(char, cursor)) >= 0:
        end = _run(text, candidate, char)
        width = end - candidate
        if fenced:
            begin = text.rfind('\n', 0, candidate) + 1
            tail = text.find('\n', end)
            tail = len(text) if tail < 0 else tail
            if width >= count and not text[begin:candidate].strip(' \t\r') and not text[end:tail].strip():
                return end
        elif width == count:
            return end
        cursor = end
    return len(text)


def _call_at(text, at):
    for opener, closer in CALL_MARKS:
        insensitive = opener == '<tool_call>'
        if _matches(text, at, opener, insensitive):
            return text[at:at + len(opener)], closer, insensitive
    match = _NAMESPACE.match(text, at)
    if match:
        return match[0], '</' + match[1] + ':tool_call>', True
    return None


def _parameter_end(text, start, close_pattern, *, finished=False):
    """Locate a raw XML delimiter without mistaking a JSON string's bytes for it.

    Only a JSON-shaped value enables JSON quoting; ordinary shell/prose quote
    characters remain raw XML parameter data and cannot swallow later frames.
    This locates boundaries, not a schema/argument conversion or JSON repair.
    """
    opening = text.find('>', start)
    if opening < 0:
        return None
    return _value_end(text, opening + 1, close_pattern, finished=finished)


def _valid_quoted_literal(value):
    """Use the actual accepted value grammars only for ambiguous delimiter spans."""
    import ast
    import json
    from tensorfold.tool_parameters import closed_json

    for candidate in (value, closed_json(value)):
        if candidate is not None:
            try:
                json.loads(candidate)
                return True
            except (ValueError, TypeError):
                pass
    try:
        ast.literal_eval(value.strip())
        return True
    except (ValueError, SyntaxError):
        return False


def _value_end(text, body, close_pattern, *, finished=False):
    while body < len(text) and text[body].isspace():
        body += 1
    first_close = close_pattern.search(text, body)
    if body == len(text) or text[body] not in "{[(\"'":
        return first_close
    cursor = body
    while cursor < len(text):
        if text[cursor] in ('"', "'"):
            quoted_end = _quote_end(text, cursor, text[cursor], require_close=True)
            if quoted_end < 0:
                # Only normal finality resolves malformed quoted data. Before
                # then a future quote may make delimiter bytes literal.
                return first_close if finished else None
            cursor = quoted_end
        elif (match := close_pattern.match(text, cursor)) is not None:
            if first_close is not None and first_close.start() < match.start():
                # A malformed first call must not borrow a closing quote from
                # an independent later call. Whole-literal grammar decides
                # only this ambiguous case, not every normal parameter.
                if not _valid_quoted_literal(text[body:match.start()]):
                    return first_close if finished else None
            return match
        else:
            cursor += 1
    return None


def xml_parameter_value_end(text: str, start: int = 0, *, finished=False) -> int:
    """Absolute closing-tag offset for a value; quoted JSON delimiters are data."""
    match = _value_end(text, start, _PARAMETER_CLOSE, finished=finished)
    return -1 if match is None else match.start()


def xml_parameter_fields(text: str, *, finished=True) -> tuple[list[tuple[str, str]], str]:
    """Complete XML parameters plus untouched residue, dropping one framing newline.

    Both lenient decoders and complete-call checks share this boundary. Unknown
    or incomplete framing remains residue, rather than silently repairing it.
    """
    fields, residue, at = [], [], 0
    while (opening := _PARAMETER_OPEN.search(text, at)) is not None:
        end = _value_end(text, opening.end(), _PARAMETER_CLOSE, finished=finished)
        if end is None:
            break
        residue.append(text[at:opening.start()])
        value = text[opening.end():end.start()]
        if value.startswith('\n'):
            value = value[1:]
        if value.endswith('\n'):
            value = value[:-1]
        fields.append((opening[1].strip(), value))
        at = end.end()
    residue.append(text[at:])
    return fields, ''.join(residue)


def _call_end(text, start, opener, closer, insensitive, *, repeated=True, finished=False):
    cursor = payload = start + len(opener)
    # Adjacent repeated openers are envelope slips, never value rewrites.
    while repeated:
        after = cursor
        while after < len(text) and text[after].isspace():
            after += 1
        if not _matches(text, after, opener, insensitive):
            break
        cursor = payload = after + len(opener)
    while cursor < len(text):
        if text.startswith(GEMMA_STRING, cursor) or text[cursor] == '"':
            cursor = _quote_end(text, cursor)
        elif text[cursor:cursor + 11].lower() == '<parameter=' or text.startswith('<｜DSML｜parameter ', cursor):
            parameter_end = '</parameter>' if text[cursor:cursor + 11].lower() == '<parameter=' else '</｜DSML｜parameter>'
            close_pattern = _PARAMETER_CLOSE if parameter_end == '</parameter>' else _DSML_PARAMETER_CLOSE
            matched = _parameter_end(text, cursor, close_pattern, finished=finished)
            if matched is None:
                return MarkupRegion('call', start, len(text), opener, closer, payload, len(text), False)
            cursor = matched.end()
        elif _matches(text, cursor, closer, insensitive):
            body_end, end = cursor, cursor + len(closer)
            while repeated:
                after = end
                while after < len(text) and text[after].isspace():
                    after += 1
                if not _matches(text, after, closer, insensitive):
                    break
                end = after + len(closer)
            return MarkupRegion('call', start, end, opener, closer, payload, body_end)
        else:
            cursor += 1
    return MarkupRegion('call', start, len(text), opener, closer, payload, len(text), False)


def markup_regions(text: str, *, finished=False, call_headers=None) -> list[MarkupRegion]:
    """Ordered nonoverlapping code/string/call spans; an unfinished span ends at EOF."""
    result, at = [], 0
    while at < len(text):
        char = text[at]
        if text.startswith(GEMMA_STRING, at) or char == '"':
            end = _quote_end(text, at)
            result.append(MarkupRegion('string', at, end))
        elif char == '`' or char == '~' and _run(text, at, char) - at >= 3 and not text[text.rfind('\n', 0, at) + 1:at].strip():
            end = _code_end(text, at, char)
            result.append(MarkupRegion('code', at, end))
        elif char == '<' and (call := _call_at(text, at)) is not None:
            if call_headers is not None and not any(text.startswith(header, at) for header in call_headers):
                # Strict free-reasoning mode: an opener alone is not a call.
                # Only the selected grammar's exact offered header owns data.
                at += 1
                continue
            region = _call_end(text, at, *call, finished=finished)
            result.append(region)
            end = region.end
        else:
            at += 1
            continue
        at = end
    return result


def structural_tool_opener(text: str, opener: str) -> bool:
    """Whether the suffix is an actual lexical opener, not quoted data/a nested argument."""
    start = len(text) - len(opener)
    return start >= 0 and text.endswith(opener) and any(
        region.kind == 'call' and region.start == start and region.opener == opener for region in markup_regions(text))


def partial_call_suffix(text: str, openers=None) -> int:
    """Length of an unfinished lexical opener, including XML namespaces/case.

    Callers check that the tail is outside an opaque region. Namespace names
    remain ambiguous until ':' or '>' resolves them; no already-emitted '<n'
    can later be retracted when a supported namespaced call completes.
    """
    start = text.rfind('<')
    if start < 0:
        return 0
    tail = text[start:]
    for opener, _ in CALL_MARKS:
        if openers is not None and opener not in openers:
            continue
        candidate, marker = (tail.lower(), opener.lower()) if opener == '<tool_call>' else (tail, opener)
        if len(candidate) < len(marker) and marker.startswith(candidate):
            return len(tail)
    if openers is None and re.fullmatch(r'<[A-Za-z_][\w.-]*(?::(?:t|to|too|tool|tool_|tool_c|tool_ca|tool_cal|tool_call)?)?',
                                       tail, re.IGNORECASE):
        return len(tail)
    return 0


def structural_marker(text: str, marker: str, *, call_headers=None) -> bool:
    """Whether a framing suffix begins outside code, strings and tool payloads."""
    start = len(text) - len(marker)
    return bool(marker) and start >= 0 and text.endswith(marker) and not any(
        region.start <= start < region.end for region in markup_regions(text, call_headers=call_headers))


def structural_call_header(text: str, opener: str, headers) -> tuple[int, int] | None:
    """Latest actual full offered header, as (opener offset, header end).

    Headers are complete native literals including opener/lead/name/tail from
    the selected strict plan. This establishes framing, not argument validity.
    Default lenient parsing has no dependency on this strict-only mode.
    """
    headers = tuple(header for header in headers if header.startswith(opener))
    found = None
    for region in markup_regions(text, call_headers=headers):
        if region.kind == 'call' and region.opener == opener:
            for header in headers:
                if text.startswith(header, region.start):
                    found = region.start, region.start + len(header)
                    break
    return found


def pending_call_header(text: str, opener: str, headers) -> int | None:
    """A trailing actual opener whose remaining bytes are a proper header prefix."""
    start = text.rfind(opener)
    if start < 0 or not any(header.startswith(text[start:]) and header != text[start:]
                            for header in headers if header.startswith(opener)):
        return None
    if any(region.start <= start < region.end for region in markup_regions(text, call_headers=headers)):
        return None
    return start


def unquoted_find(text: str, marker: str, start: int = 0) -> int:
    """Find framing outside code, string data and existing call payloads."""
    spans = markup_regions(text)
    at = text.find(marker, start)
    while at >= 0:
        region = next((r for r in spans if r.start <= at < r.end), None)
        if region is None:
            return at
        at = text.find(marker, region.end)
    return -1


def xml_function_frames(text: str, *, finished=True) -> list[str] | None:
    """Only complete top-level XML functions; parameter data cannot split functions."""
    result, at = [], 0
    while at < len(text):
        while at < len(text) and text[at].isspace():
            at += 1
        if at == len(text):
            break
        match = _FUNCTION.match(text, at)
        if match is None:
            return None
        region = _call_end(text, at, match[0], '</function>', True, repeated=False, finished=finished)
        if not region.complete:
            return None
        result.append(text[at:region.end])
        at = region.end
    return result or None

"""Bounded rank request data; limits follow the CUDA HTTP 96 MiB body contract.

Only the approved rank pair writes these messages. The native TCPStore returns
bytes before Python can inspect their size; senders therefore validate before
publication, and receivers bound decoding and expansion before use.
"""
from __future__ import annotations

import json
import io
import math

MAX_MESSAGE_BYTES = 96 << 20
MAX_GRAMMAR_ITEMS = MAX_MESSAGE_BYTES + 2


def bounded_json(body):
    """Encode validated flat request fields without a larger transient string."""
    size = 0
    with io.StringIO() as output:
        for chunk in json.JSONEncoder().iterencode(body):
            size += len(chunk)  # the default encoder emits ASCII
            if size > MAX_MESSAGE_BYTES:
                raise ValueError("rank request exceeds the 96 MiB transport envelope")
            output.write(chunk)
        return output.getvalue()


def ints(values, maximum, *, lower=0, upper=(1 << 31) - 1, empty=False):
    if (type(values) is not list or not (0 if empty else 1) <= len(values) <= maximum
            or any(type(value) is not int or not lower <= value <= upper for value in values)):
        raise ValueError("rank integer list exceeds its typed count/value contract")
    return values


def grammar(values, vocab):
    ints(values, MAX_GRAMMAR_ITEMS, empty=True)
    if values:
        if len(values) < 2 or values[0] not in range(5) or not 0 <= values[1] <= vocab:
            raise ValueError("rank grammar header is invalid")
        if any(values[index] > 255 for index in range(2, len(values))):
            raise ValueError("rank grammar payload must be UTF-8 bytes")
        bytes(values[index] for index in range(2, len(values))).decode("utf-8")
    return values


def packed_grammar(constraint, vocab):
    from tensorfold.engine.grammar import pack

    if constraint is not None:
        text = constraint.spec.text
        if (type(text) is not str or len(text) > MAX_MESSAGE_BYTES
                or len(text.encode("utf-8")) > MAX_MESSAGE_BYTES):
            raise ValueError("rank grammar exceeds the 96 MiB source envelope")
    return grammar(pack(constraint), vocab)


def finite(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("rank sampling values must be finite numbers")
    return value


def request_json(raw, *, counts):
    """Reject size/nesting/numeric expansion before the JSON decoder allocates.

    Rank requests contain an object of scalar values and flat lists. Strings
    are only field names; nested objects/lists have no supported schema.
    """
    if type(raw) not in (str, bytes) or len(raw) > MAX_MESSAGE_BYTES:
        raise ValueError("rank request exceeds the 96 MiB transport envelope")
    if type(raw) is bytes:
        raw = raw.decode("utf-8")
    elif len(raw.encode("utf-8")) > MAX_MESSAGE_BYTES:
        raise ValueError("rank request exceeds the UTF-8 transport envelope")
    depth, quoted, escaped, key_start = 0, False, False, 0
    last_key, array_maximum, array_count, next_item = None, 0, 0, False
    allowed = set(counts) | {"max_tokens", "draft", "cached", "stop_eos", "stop"}
    seen, numeric_length = set(), 0
    decoder = json.JSONDecoder()
    for index, char in enumerate(raw):
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
                if index - key_start > 64:
                    raise ValueError("rank JSON strings are bounded field names")
                last_key = decoder.raw_decode(raw[key_start:index + 1])[0]
                if last_key not in allowed or last_key in seen:
                    raise ValueError("rank request has unknown or duplicate fields")
                seen.add(last_key)
        elif char == '"':
            if depth != 1:
                raise ValueError("rank lists contain only scalar numbers")
            quoted = True
            key_start = index
        elif char in "[{":
            depth += 1
            if depth > 2:
                raise ValueError("rank request requires flat schema lists")
            if char == "[":
                if last_key not in counts:
                    raise ValueError("rank array field has no declared count bound")
                array_maximum, array_count, next_item = counts[last_key], 0, True
            elif depth == 2:
                raise ValueError("rank request has no nested objects")
        elif char in "]}":
            depth -= 1
        elif depth == 2:
            if char == ",":
                next_item = True
            elif not char.isspace() and next_item:
                array_count += 1
                if array_count > array_maximum:
                    raise ValueError("rank list exceeds its count before JSON expansion")
                next_item = False
        if not quoted and char in "0123456789+-.eE":
            numeric_length += 1
            if numeric_length > 24:  # maximum finite float64 repr, including sign/exponent
                raise ValueError("rank numeric token exceeds its scalar envelope")
        else:
            numeric_length = 0
    def integer(text):
        if len(text.lstrip("-")) > 20:
            raise ValueError("rank integer exceeds its 64-bit envelope")
        return int(text)
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate rank request field")
            result[key] = value
        return result
    def constant(text):
        raise ValueError("rank request cannot contain non-finite constants")
    body = json.loads(raw, parse_int=integer, parse_float=lambda text: finite(float(text)),
                      parse_constant=constant, object_pairs_hook=pairs)
    if type(body) is not dict:
        raise ValueError("rank request must be an object")
    return body

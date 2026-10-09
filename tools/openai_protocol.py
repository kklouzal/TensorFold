"""Bounded OpenAI benchmark response contracts, using only the standard library.

SSE data is dispatched at blank lines, including CR/LF/CRLF and multiline data.
The response and each line/event are bounded before materialization. Responses
must finish with a dispatched [DONE] and typed usage; a JSON error on HTTP200
is an operation failure. These readers impose size bounds, not a wall deadline.
"""
from dataclasses import dataclass
import json
import math
import sys

MAX_RESPONSE_BYTES = 16 << 20
MAX_LINE_BYTES = 65536
READ_BYTES = 4096


@dataclass(frozen=True)
class Limits:
    response_bytes: int = MAX_RESPONSE_BYTES
    line_bytes: int = MAX_LINE_BYTES

    def __post_init__(self):
        for value in (self.response_bytes, self.line_bytes):
            if type(value) is not int or not 0 < value <= sys.maxsize:
                raise ValueError('positive representable response size limits required')


def add_arguments(parser):
    parser.add_argument('--response-mib', type=int, default=16,
                        help='maximum decoded response bytes in MiB (default16; raise for large replies)')
    parser.add_argument('--sse-line-kib', type=int, default=64,
                        help='maximum SSE line bytes in KiB (default64; raise for batched replies)')


def from_arguments(args):
    return Limits(args.response_mib * (1 << 20), args.sse_line_kib * (1 << 10))


def _pairs(rows):
    result = {}
    for name, value in rows:
        if name in result:
            raise ValueError('duplicate response JSON key')
        result[name] = value
    return result


def object_json(raw):
    def constant(_value):
        raise ValueError('nonfinite response JSON value')
    def finite(value):
        parsed = float(value)
        if not math.isfinite(parsed):
            raise ValueError('nonfinite response JSON number')
        return parsed
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=constant, parse_float=finite)
    if type(value) is not dict or 'error' in value:
        raise ValueError('SSE error or invalid object in OpenAI response')
    return value


def response_type(response, expected):
    if response.status != 200 or response.getheader('Content-Type', '').split(';', 1)[0].strip().lower() != expected:
        raise ValueError('unexpected OpenAI HTTP status or response content type')


def sse_objects(response, *, limits=None):
    """Yield complete JSON data events; DONE and final usage are caller contracts."""
    limits = limits or Limits()
    line = bytearray()
    data = []
    event_size = consumed = 0
    previous_cr = False
    first = True

    def finish_line():
        nonlocal event_size, first
        raw = bytes(line)
        line.clear()
        if first:
            raw = raw.removeprefix(b'\xef\xbb\xbf')
            first = False
        if not raw:
            if not data:
                return None
            payload = '\n'.join(data)
            data.clear()
            event_size = 0
            return payload
        if raw.startswith(b':'):
            return None
        field, sep, value = raw.partition(b':')
        if field == b'data':
            if sep and value.startswith(b' '):
                value = value[1:]
            event_size += len(value) + 1
            if event_size > limits.response_bytes:
                raise ValueError('SSE event exceeded bounded size')
            data.append(value.decode('utf-8'))
        return None

    while True:
        raw = response.read1(READ_BYTES)
        if not raw:
            raise ValueError('SSE ended before [DONE]')
        consumed += len(raw)
        if consumed > limits.response_bytes:
            raise ValueError('SSE response exceeded bounded size')
        # bytes.splitlines uses C scanning for the three SSE line endings.
        # An ending split across reads is handled by previous_cr, while a
        # partial final line is bounded before the owned accumulator grows.
        if previous_cr:
            previous_cr = False
            if raw.startswith(b'\n'):
                raw = raw[1:]
        for fragment in raw.splitlines(keepends=True):
            ending = fragment.endswith((b'\n', b'\r'))
            content = fragment.rstrip(b'\r\n') if ending else fragment
            if len(content) > limits.line_bytes - len(line):
                raise ValueError('SSE line exceeded bounded size')
            line.extend(content)
            previous_cr = ending and fragment.endswith(b'\r')
            if ending:
                payload = finish_line()
                if payload is not None:
                    if payload == '[DONE]':
                        return
                    yield object_json(payload)


def chunk_fields(chunk):
    """Validate nested fields before callers read any text, usage, or identity."""
    usage, runtime = chunk.get('usage'), chunk.get('tensorfold')
    if usage is not None and type(usage) is not dict:
        raise ValueError('OpenAI usage must be an object or null')
    if runtime is not None and type(runtime) is not dict:
        raise ValueError('TensorFold runtime must be an object or null')
    if runtime and runtime.get('token_sha') is not None and type(runtime['token_sha']) is not str:
        raise ValueError('token hash must be text or null')
    choices = chunk.get('choices', [])
    if type(choices) is not list:
        raise ValueError('OpenAI choices must be a list')
    pieces = []
    for choice in choices:
        if type(choice) is not dict:
            raise ValueError('OpenAI choice must be an object')
        delta = choice.get('delta')
        if delta is not None and type(delta) is not dict:
            raise ValueError('OpenAI delta must be an object or null')
        delta = delta or {}
        fields = [choice.get('text'), *(delta.get(key) for key in ('content', 'reasoning_content', 'reasoning'))]
        if any(value is not None and type(value) is not str for value in fields):
            raise ValueError('OpenAI text and reasoning pieces must be text or null')
        pieces.append(next((value for value in fields if value), ''))
    return usage, runtime, pieces


def completion_usage(usage, maximum, *, prompt=False):
    value = usage.get('completion_tokens')
    if type(value) is not int or not 0 < value <= maximum:
        raise ValueError('OpenAI missing bounded positive completion usage')
    if prompt:
        value = usage.get('prompt_tokens')
        if type(value) is not int or value <= 0:
            raise ValueError('OpenAI missing positive integer prompt usage')
    return usage['completion_tokens']


def json_response(response, maximum, *, limits=None):
    limits = limits or Limits()
    response_type(response, 'application/json')
    chunks, consumed = [], 0
    while True:
        raw = response.read1(READ_BYTES)
        if not raw:
            break
        consumed += len(raw)
        if consumed > limits.response_bytes:
            raise ValueError('OpenAI JSON response exceeded bounded size')
        chunks.append(raw)
    value = object_json(b''.join(chunks))
    usage, _runtime, _pieces = chunk_fields(value)
    if not value.get('choices'):
        raise ValueError('OpenAI JSON response contains no completion choice')
    completion_usage(usage or {}, maximum)
    return value

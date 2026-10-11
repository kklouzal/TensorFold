"""One HTTP request's supported framing and exact bounded body consumption.

Parsed headers belong to the handler and stay unchanged during this operation.
Equal decimal Content-Length lists normalize to one value (RFC9112 section6.3).
Fixed-length-only callers reject Transfer-Encoding; read_body also supports chunked HTTP/1.1. Any ambiguous framing,
unread body failure or premature EOF closes the connection before another
request can consume its bytes. Completed valid bodies preserve keepalive.
"""
import re

from .errors import RequestError

LIMIT = 32 << 20
_METADATA_LIMIT = 65536
_BLOCK_SIZE = 64 << 10


def _refuse(handler, message):
    handler.close_connection = True
    raise RequestError(message)


def content_length(handler, limit: int, *, require_nonempty: bool = False) -> int:
    """Validate before allocation; identical duplicate/list lengths are valid."""
    if handler.headers.get_all('Transfer-Encoding'):
        _refuse(handler, 'Transfer-Encoding is unsupported; use Content-Length')
    fields = handler.headers.get_all('Content-Length') or []
    number = None
    for field in fields:
        for item in field.split(','):
            item = item.strip(' \t')
            if not item or not item.isascii() or not item.isdecimal():
                _refuse(handler, 'Content-Length must contain a decimal byte count')
            # Arbitrarily many leading zeroes do not require giant integer
            # conversion; header size itself is bounded by the HTTP parser.
            canonical = item.lstrip('0') or '0'
            if number is not None and canonical != number:
                _refuse(handler, 'conflicting Content-Length values')
            number = canonical
    number = number or '0'
    maximum = str(limit)
    if len(number) > len(maximum) or len(number) == len(maximum) and number > maximum:
        _refuse(handler, f'request body exceeds the {limit // (1 << 20)} MiB limit')
    length = int(number)
    if require_nonempty and not length:
        _refuse(handler, 'request body must be nonempty and include Content-Length')
    return length


def read(handler, limit: int) -> bytes:
    """Read exactly one supported frame; never accept truncated valid JSON."""
    length = content_length(handler, limit)
    try:
        payload = handler.rfile.read(length)
    except BaseException:
        handler.close_connection = True
        raise
    if len(payload) != length:
        _refuse(handler, 'request body ended before Content-Length')
    return payload


def _chunked(handler, limit: int, *, discard: bool) -> bytes:
    """Cap total payload and framing work, including chunk lines, delimiters and trailers."""
    if handler.headers.get_all('Content-Length'):
        _refuse(handler, 'Content-Length and Transfer-Encoding cannot be combined')
    codings = [item.strip().lower() for field in handler.headers.get_all('Transfer-Encoding')
               for item in field.split(',')]
    if codings != ['chunked'] or getattr(handler, 'request_version', 'HTTP/1.1') != 'HTTP/1.1':
        _refuse(handler, 'unsupported request Transfer-Encoding; expected chunked over HTTP/1.1')
    body = bytearray()
    metadata = total = 0

    def line():
        nonlocal metadata
        data = handler.rfile.readline(_METADATA_LIMIT - metadata + 1)
        metadata += len(data)
        if metadata > _METADATA_LIMIT or not data.endswith(b'\r\n'):
            _refuse(handler, 'invalid or oversized chunked request framing')
        return data[:-2]

    while True:
        header = line()
        size_text, separator, extension = header.partition(b';')
        if separator:
            size_text = size_text.rstrip(b' \t')
            if any(c < 32 and c != 9 or c == 127 for c in extension):
                _refuse(handler, 'invalid chunk extension')
        if not re.fullmatch(rb'[0-9a-fA-F]+', size_text):
            _refuse(handler, 'invalid chunk size')
        # Compare canonical digits before conversion: arbitrarily long sizes never allocate giant integers.
        canonical = size_text.lstrip(b'0') or b'0'
        maximum = format(limit - total, 'x').encode()
        lower = canonical.lower()
        if len(lower) > len(maximum) or len(lower) == len(maximum) and lower > maximum:
            _refuse(handler, f'request body exceeds the {limit // (1 << 20)} MiB limit')
        size = int(canonical, 16)
        if not size:
            while trailer := line():
                if not re.fullmatch(rb"[!#$%&'*+.^_`|~0-9A-Za-z-]+:[\t\x20-\x7e\x80-\xff]*", trailer):
                    _refuse(handler, 'invalid request trailers')
                if trailer.partition(b':')[0].lower() in (b'content-length', b'transfer-encoding', b'host'):
                    _refuse(handler, 'request framing fields are forbidden in trailers')
            return bytes(body)
        total += size
        if discard:
            _discard_exact(handler, size)
        else:
            block = handler.rfile.read(size)
            if len(block) != size:
                _refuse(handler, 'request body ended before chunk size')
            body.extend(block)
        metadata += 2
        if metadata > _METADATA_LIMIT:
            _refuse(handler, 'chunked request framing exceeds the limit')
        if handler.rfile.read(2) != b'\r\n':
            _refuse(handler, 'invalid chunk terminator')


def read_body(handler, *, limit: int = LIMIT) -> bytes:
    """Read exactly one bounded fixed-length or chunked frame; close on incomplete consumption."""
    try:
        if handler.headers.get_all('Transfer-Encoding'):
            return _chunked(handler, limit, discard=False)
        return read(handler, limit)
    except BaseException:
        handler.close_connection = True
        raise


def _discard_exact(handler, remaining: int) -> None:
    while remaining:
        block = handler.rfile.read(min(remaining, _BLOCK_SIZE))
        if not block:
            _refuse(handler, 'request body ended before Content-Length or chunk size')
        remaining -= len(block)


def discard(handler, limit: int = LIMIT) -> None:
    """Drain one refused/unused frame using at most 64 KiB transient payload storage."""
    try:
        if handler.headers.get_all('Transfer-Encoding'):
            _chunked(handler, limit, discard=True)
        else:
            _discard_exact(handler, content_length(handler, limit))
    except BaseException:
        handler.close_connection = True
        raise

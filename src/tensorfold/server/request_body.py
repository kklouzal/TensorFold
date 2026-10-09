"""One HTTP request's supported framing and exact bounded body consumption.

Parsed headers belong to the handler and stay unchanged during this operation.
Equal decimal Content-Length lists normalize to one value (RFC9112 section6.3).
Transfer-Encoding is unsupported by these JSON APIs. Any ambiguous framing,
unread body failure or premature EOF closes the connection before another
request can consume its bytes. Completed valid bodies preserve keepalive.
"""
from .errors import RequestError


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


def discard(handler, limit: int = 32 << 20) -> None:
    """Drain a refused/unused body with at most64KiB transient storage."""
    remaining = content_length(handler, limit)
    try:
        while remaining:
            block = handler.rfile.read(min(remaining, 64 << 10))
            if not block:
                _refuse(handler, 'request body ended before Content-Length')
            remaining -= len(block)
    except BaseException:
        handler.close_connection = True
        raise

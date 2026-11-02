"""One HTTP request's supported framing and exact bounded body consumption.

Parsed headers belong to the handler and stay unchanged during this operation.
Equal decimal Content-Length lists normalize to one value (RFC9112 section6.3).
Fixed-length-only callers reject Transfer-Encoding; read_body also supports chunked HTTP/1.1. Any ambiguous framing,
unread body failure or premature EOF closes the connection before another
request can consume its bytes. Completed valid bodies preserve keepalive.
"""
import math
import re
import time

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


class _DeadlineReader:
    """One HTTP body borrows the binary reader, with checks between single I/O.

    read1 performs at most one raw read (BufferedReader and BytesIO contract).
    Framing lines consume one byte per read: the existing aggregate64KiB framing
    cap bounds that work and avoids reading/persisting the next request's bytes.
    Payload reads are64KiB. Cancellation is checked before/after each actual
    read; idle reads are bounded by the remaining absolute deadline. A timed-out
    socket file is never retried (its buffering may no longer be usable).
    """
    def __init__(self, handler, deadline, cancelled):
        self.reader, self.connection = handler.rfile, handler.connection
        self.deadline, self.cancelled = deadline, cancelled
        if not callable(getattr(self.reader, 'read1', None)):
            raise ValueError('deadline body consumption requires a binary read1 stream')

    def _check(self):
        if self.cancelled is not None and self.cancelled():
            from .cancellation import RequestCancelled
            raise RequestCancelled('request body cancelled')
        remaining = None if self.deadline is None else self.deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
            raise TimeoutError('request body deadline exceeded')
        return remaining

    def _once(self, size):
        remaining = self._check()
        if remaining is not None:
            self.connection.settimeout(remaining)
        value = self.reader.read1(size)
        self._check()
        return value

    def read(self, size):
        result = bytearray()
        while len(result) < size:
            value = self._once(min(size - len(result), _BLOCK_SIZE))
            if not value:
                break
            result.extend(value)
        return bytes(result)

    def readline(self, size):
        result = bytearray()
        while len(result) < size:
            value = self._once(1)
            if not value:
                break
            result.extend(value)
            if value == b'\n':
                break
        return bytes(result)


def read_body(handler, *, limit: int = LIMIT, deadline=None, cancelled=None) -> bytes:
    """Consume one bounded frame. Optional deadline is absolute monotonic seconds.

    Default callers keep their original reader/timeout behavior. An opted-in
    caller owns exclusive HTTP body consumption and supplies a read1 reader and
    socket connection. Both bindings/timeouts restore before returning/raising.
    Any incomplete/ambiguous/deadline/cancel failure closes the connection.
    """
    original = previous_timeout = None
    primary = _native_cause = _native_context = None
    try:
        if deadline is not None or cancelled is not None:
            if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline)):
                raise ValueError('body deadline must be finite monotonic seconds')
            if cancelled is not None and not callable(cancelled):
                raise ValueError('body cancellation must be callable')
            reader = _DeadlineReader(handler, deadline, cancelled)
            reader._check()
            previous_timeout = handler.connection.gettimeout()
            original = handler.rfile
            handler.rfile = reader
        if handler.headers.get_all('Transfer-Encoding'):
            return _chunked(handler, limit, discard=False)
        return read(handler, limit)
    except BaseException as error:
        primary = error
        _native_cause = BaseException.__cause__.__get__(error)
        _native_context = BaseException.__context__.__get__(error)
        handler.close_connection = True
        raise
    finally:
        if original is not None:
            handler.rfile = original
            try:
                handler.connection.settimeout(previous_timeout)
            except BaseException as cleanup:
                handler.close_connection = True
                if primary is not None:
                    raise primary from cleanup
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

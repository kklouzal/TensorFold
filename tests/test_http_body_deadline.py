"""Absolute body deadlines/cancellation across fragmented fixed and chunked framing."""
from email.message import Message
from io import BytesIO
from types import SimpleNamespace

import pytest

from tensorfold.server import request_body
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import RequestError


class Connection:
    timeout = 7
    def gettimeout(self):
        return self.timeout
    def settimeout(self, timeout):
        self.timeout = timeout


def handler(data, headers, *, reader=None):
    parsed = Message()
    for name, value in headers:
        parsed[name] = value
    return SimpleNamespace(rfile=BytesIO(data) if reader is None else reader, headers=parsed,
                           connection=Connection(), close_connection=False, request_version='HTTP/1.1')


@pytest.mark.parametrize('chunked', [False, True])
def test_fragmented_frame_restores_reader_timeout_and_preserves_next_request(monkeypatch, chunked):
    class Fragmented(BytesIO):
        def read1(self, size=-1):
            return super().read1(min(size, 2))
    raw = b'2\r\n{}\r\n0\r\nX-Fact: valid\r\n\r\nNEXT' if chunked else b'{}NEXT'
    reader = Fragmented(raw)
    value = handler(raw, [('Transfer-Encoding', 'chunked')] if chunked else [('Content-Length', '2')], reader=reader)
    monkeypatch.setattr(request_body.time, 'monotonic', lambda: 1.)
    assert request_body.read_body(value, deadline=2., cancelled=lambda: False) == b'{}'
    assert value.rfile is reader and reader.read() == b'NEXT'
    assert value.connection.timeout == 7 and not value.close_connection


@pytest.mark.parametrize('raw,headers,start', [
    (b'{}', [('Content-Length', '2')], 0),
    (b'1;slow=yes\r\nx\r\n0\r\n\r\n', [('Transfer-Encoding', 'chunked')], 0),
    (b'0\r\nX-Slow: trailer\r\n\r\n', [('Transfer-Encoding', 'chunked')], 3),
])
def test_slow_progress_cannot_reset_absolute_deadline(monkeypatch, raw, headers, start):
    clock = [0.]
    class Trickle(BytesIO):
        def read1(self, size=-1):
            if self.tell() >= start:
                clock[0] += .4
            return super().read1(min(size, 1))
    reader = Trickle(raw)
    value = handler(raw, headers, reader=reader)
    monkeypatch.setattr(request_body.time, 'monotonic', lambda: clock[0])
    with pytest.raises(TimeoutError, match='deadline'):
        request_body.read_body(value, deadline=.7)
    assert value.close_connection and value.rfile is reader and value.connection.timeout == 7


def test_cancellation_checks_actual_reads_and_socket_timeout_is_not_retried(monkeypatch):
    calls = []
    failure = TimeoutError('actual socket timeout')
    class Broken(BytesIO):
        def read1(self, size=-1):
            calls.append(size)
            raise failure
    value = handler(b'', [('Content-Length', '2')], reader=Broken())
    monkeypatch.setattr(request_body.time, 'monotonic', lambda: 0.)
    with pytest.raises(TimeoutError) as caught:
        request_body.read_body(value, deadline=1.)
    assert caught.value is failure and calls == [2] and value.close_connection
    value = handler(b'{}', [('Content-Length', '2')])
    with pytest.raises(RequestCancelled):
        request_body.read_body(value, deadline=1., cancelled=lambda: True)
    assert value.close_connection and value.rfile.tell() == 0


@pytest.mark.parametrize('raw,headers,limit', [
    (b'{}', [('Content-Length', '2'), ('Transfer-Encoding', 'chunked')], 10),
    (b'3\r\nabc\r\n0\r\n\r\n', [('Transfer-Encoding', 'chunked')], 2),
    (b'x\r\n', [('Transfer-Encoding', 'chunked')], 10),
    (b'1\r\nx!\n0\r\n\r\n', [('Transfer-Encoding', 'chunked')], 10),
])
def test_optional_deadline_preserves_framing_refusals(monkeypatch, raw, headers, limit):
    value = handler(raw, headers)
    monkeypatch.setattr(request_body.time, 'monotonic', lambda: 0.)
    with pytest.raises(RequestError):
        request_body.read_body(value, limit=limit, deadline=1.)
    assert value.close_connection and value.connection.timeout == 7


def test_primary_transport_error_survives_timeout_restoration_failure(monkeypatch):
    primary, cleanup = OSError('body read'), OSError('restore timeout')
    class Broken(BytesIO):
        def read1(self, size=-1):
            raise primary
    value = handler(b'', [('Content-Length', '1')], reader=Broken())
    def settimeout(timeout):
        if timeout == 7:
            raise cleanup
    value.connection.settimeout = settimeout
    monkeypatch.setattr(request_body.time, 'monotonic', lambda: 0.)
    with pytest.raises(OSError) as caught:
        request_body.read_body(value, deadline=1.)
    assert caught.value is primary and primary.__cause__ is cleanup and value.close_connection


def test_deadline_mode_keeps_cumulative_chunk_metadata_bound(monkeypatch):
    value = handler(b'1\r\nx\r\n' * 14000 + b'0\r\n\r\n', [('Transfer-Encoding', 'chunked')])
    monkeypatch.setattr(request_body.time, 'monotonic', lambda: 0.)
    with pytest.raises(RequestError, match='framing'):
        request_body.read_body(value, deadline=1.)
    assert value.close_connection and value.connection.timeout == 7

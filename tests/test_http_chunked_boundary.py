"""Chunk-aware framing preserves the fork's bounded drain and failed-input containment."""
from email.message import Message
from io import BytesIO
from types import SimpleNamespace
import unittest

from tensorfold.server import request_body
from tensorfold.server.errors import RequestError


def request(wire, *, version='HTTP/1.1'):
    headers = Message()
    headers['Transfer-Encoding'] = 'chunked'
    return SimpleNamespace(headers=headers, rfile=BytesIO(wire), request_version=version,
                           close_connection=False)


class ChunkedBoundary(unittest.TestCase):
    def test_exact_payload_and_refused_body_drain_preserve_next_request(self):
        wire = b'2;name="value"\r\n{}\r\n0\r\nX-Test: ignored\r\n\r\nnext'
        for discard in (False, True):
            with self.subTest(discard=discard):
                value = request(wire)
                if discard:
                    request_body.discard(value)
                else:
                    self.assertEqual(request_body.read_body(value), b'{}')
                self.assertFalse(value.close_connection)
                self.assertEqual(value.rfile.read(), b'next')

    def test_drain_uses_bounded_blocks_and_handles_short_reads(self):
        class ShortReads(BytesIO):
            largest = 0

            def read(self, size=-1):
                self.largest = max(self.largest, size)
                return super().read(min(size, 17))

        size = 131072
        value = request(f'{size:x}\r\n'.encode() + b'x' * size + b'\r\n0\r\n\r\nnext')
        value.rfile = ShortReads(value.rfile.read())
        request_body.discard(value)
        self.assertLessEqual(value.rfile.largest, 64 << 10)
        self.assertEqual(value.rfile.read(4), b'next')
        self.assertFalse(value.close_connection)

    def test_total_metadata_is_bounded_even_without_extensions(self):
        value = request(b'1\r\nx\r\n' * 14000 + b'0\r\n\r\n')
        with self.assertRaises(RequestError):
            request_body.read_body(value)
        self.assertTrue(value.close_connection)
        self.assertLessEqual(value.rfile.tell(), 65537 + 14000)

    def test_extremely_large_hex_count_is_refused_before_payload(self):
        header = b'f' * 10000 + b'\r\n'
        value = request(header + b'next')
        with self.assertRaises(RequestError):
            request_body.read_body(value)
        self.assertEqual(value.rfile.tell(), len(header))
        self.assertTrue(value.close_connection)

    def test_framing_trailers_and_extension_controls_are_rejected(self):
        for suffix in (b'Content-Length: 2', b'Transfer-Encoding: chunked', b'Host: other',
                       b'X-Test: value\x00'):
            with self.subTest(suffix=suffix):
                value = request(b'0\r\n' + suffix + b'\r\n\r\nnext')
                with self.assertRaises(RequestError):
                    request_body.read_body(value)
                self.assertTrue(value.close_connection)
        value = request(b'1;bad=\x00\r\nx\r\n0\r\n\r\n')
        with self.assertRaises(RequestError):
            request_body.read_body(value)
        self.assertTrue(value.close_connection)

    def test_transport_failure_preserves_primary_and_closes(self):
        failure = OSError('controlled chunk transport failure')

        class Broken(BytesIO):
            def read(self, size=-1):
                raise failure

        for discard in (False, True):
            value = request(b'1\r\nx\r\n0\r\n\r\n')
            value.rfile = Broken(value.rfile.read())
            with self.assertRaises(OSError) as caught:
                if discard:
                    request_body.discard(value)
                else:
                    request_body.read_body(value)
            self.assertIs(caught.exception, failure)
            self.assertTrue(value.close_connection)

    def test_chunking_requires_http11(self):
        value = request(b'0\r\n\r\n', version='HTTP/1.0')
        with self.assertRaises(RequestError):
            request_body.read_body(value)
        self.assertEqual(value.rfile.tell(), 0)
        self.assertTrue(value.close_connection)

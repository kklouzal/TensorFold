"""Actual generic/CUDA/Responses framing; no accelerator/model imports."""
from contextlib import redirect_stdout
from http.client import HTTPMessage
from io import BytesIO, StringIO
import importlib.util
from pathlib import Path
import socket
import threading
from types import SimpleNamespace
import unittest

from http_import_control import assert_accelerator_free_imports

from tensorfold.server import request_body
from tensorfold.server.errors import RequestError
from tensorfold.server.http import Server, make_handler as generic
from tensorfold.cuda.http import make_handler as cuda


def handler(fields, payload=b''):
    headers = HTTPMessage()
    for name, value in fields:
        headers[name] = value
    return SimpleNamespace(headers=headers, rfile=BytesIO(payload), close_connection=False)


def exchange(factory, first, *, decisions=True):
    incoming = first + b'GET /v1/models HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n'
    connected, peer = socket.socketpair()

    class Connection:
        def __init__(self):
            self.output = bytearray()

        def makefile(self, *args):
            return BytesIO(incoming)

        def sendall(self, data):
            self.output.extend(data)

        def fileno(self):
            return connected.fileno()

        def recv(self, *args):
            return connected.recv(*args)

        def gettimeout(self):
            return connected.gettimeout()

        def settimeout(self, timeout):
            connected.settimeout(timeout)

    class App:
        served_name = 'framing-fixture'
        model_ids = ['framing-fixture']
        max_batch_size = 1
        exact_mode = {'mode': 'exact'}
        thinking_budget = 0
        vision = None
        effective_context_window = 2048
        engine = SimpleNamespace(generate=lambda *, constraint=None: None)

        def _check_fields(self, body):
            return None

        def _prepare(self, body, chat):
            return SimpleNamespace(prompt=[1, 2, 3])

    app = App()
    app.decisions = (lambda body: {'preserved': body}) if decisions else None
    connection = Connection()
    try:
        with redirect_stdout(StringIO()):
            factory(app)(connection, ('127.0.0.1', 0), None)
    finally:
        connected.close()
        peer.close()
    raw = bytes(connection.output)
    statuses = [int(part.split(b' ', 1)[0]) for part in raw.split(b'HTTP/1.1 ')[1:]]
    return statuses, raw


def post(route, fields, body=b''):
    return ('POST ' + route + ' HTTP/1.1\r\nHost: fixture\r\n' + fields + '\r\n').encode() + body


class Framing(unittest.TestCase):
    def test_valid_decimal_and_identical_lists_preserve_exact_frame_and_keepalive(self):
        for fields in ([('Content-Length', '4')], [('Content-Length', '04, 4')],
                       [('Content-Length', '4'), ('Content-Length', '0004')],
                       [('Content-Length', '0' * 5000 + '4')]):
            with self.subTest(fields=fields):
                value = handler(fields, b'datanext')
                self.assertEqual(request_body.read(value, 4), b'data')
                self.assertEqual(value.rfile.read(), b'next')
                self.assertFalse(value.close_connection)

    def test_invalid_ambiguous_unsupported_and_overlimit_frames_close_before_read(self):
        cases = [[('Content-Length', value)] for value in ('-1', '+4', '4.0', '', '4,', '4 0', '\u0664', '99999999999999999999999')]
        cases += [[('Content-Length', '4'), ('Content-Length', '5')],
                  [('Content-Length', '4, 5')], [('Transfer-Encoding', 'chunked')],
                  [('Content-Length', '4'), ('Transfer-Encoding', 'chunked')],
                  [('Transfer-Encoding', '')]]
        for fields in cases:
            with self.subTest(fields=fields):
                value = handler(fields, b'data')
                with self.assertRaises(RequestError):
                    request_body.read(value, 4)
                self.assertTrue(value.close_connection)
                self.assertEqual(value.rfile.tell(), 0)

    def test_missing_zero_nonempty_policy_and_short_body(self):
        for fields in ([], [('Content-Length', '0')]):
            value = handler(fields)
            self.assertEqual(request_body.read(value, 4), b'')
            self.assertFalse(value.close_connection)
            with self.assertRaises(RequestError):
                request_body.content_length(value, 4, require_nonempty=True)
            self.assertTrue(value.close_connection)
        value = handler([('Content-Length', '4')], b'{}')
        with self.assertRaises(RequestError):
            request_body.read(value, 4)
        self.assertTrue(value.close_connection)

    def test_discard_is_bounded_and_exact_even_with_short_transport_reads(self):
        class SmallChunks(BytesIO):
            maximum_requested = 0

            def read(self, size=-1):
                self.maximum_requested = max(self.maximum_requested, size)
                return super().read(min(size, 17))

        value = handler([('Content-Length', '131072')])
        value.rfile = SmallChunks(b'x' * 131072 + b'next')
        request_body.discard(value)
        self.assertEqual(value.rfile.maximum_requested, 64 << 10)
        self.assertEqual(value.rfile.read(4), b'next')
        self.assertFalse(value.close_connection)
        short = handler([('Content-Length', '4')], b'{}')
        with self.assertRaises(RequestError):
            request_body.discard(short)
        self.assertTrue(short.close_connection)

    def test_body_reader_transport_failure_preserves_primary_and_closes(self):
        primary = OSError('controlled body transport failure')

        class Broken:
            def read(self, _size):
                raise primary

        value = handler([('Content-Length', '4')])
        value.rfile = Broken()
        with self.assertRaises(OSError) as caught:
            request_body.read(value, 4)
        self.assertIs(caught.exception, primary)
        self.assertTrue(value.close_connection)

    def test_actual_all_post_paths_contain_invalid_framing_before_pipeline(self):
        for factory in (generic, cuda):
            for route in ('/v1/chat/completions', '/v1/decisions', '/v1/responses', '/unknown'):
                for fields in ('Content-Length: invalid\r\n', 'Content-Length: 0\r\nContent-Length: 4\r\n',
                               'Transfer-Encoding: chunked\r\nContent-Length: 0\r\n'):
                    with self.subTest(factory=factory, route=route, fields=fields):
                        statuses, output = exchange(factory, post(route, fields))
                        self.assertEqual(statuses, [400])
                        self.assertIn(b'Connection: close\r\n', output)

    def test_actual_supported_decisions_identical_length_valid_keepalive(self):
        for factory in (generic, cuda):
            for fields in ('Content-Length: 2\r\n', 'Content-Length: 02, 2\r\n',
                           'Content-Length: 2\r\nContent-Length: 0002\r\n'):
                statuses, output = exchange(factory, post('/v1/decisions', fields, b'{}'))
                self.assertEqual(statuses, [200, 200])
                self.assertIn(b'"preserved": {}', output)

    def test_unsupported_decisions_drains_complete_body_and_preserves_pipeline(self):
        for factory in (generic, cuda):
            statuses, _ = exchange(factory, post('/v1/decisions', 'Content-Length: 2\r\n', b'{}'), decisions=False)
            self.assertEqual(statuses, [404, 200])
            statuses, _ = exchange(factory, post('/v1/decisions', 'Content-Length: 33554433\r\n'), decisions=False)
            self.assertEqual(statuses, [400])

    def test_get_and_delete_unused_bodies_cannot_become_next_request(self):
        for factory in (generic, cuda):
            for method, route in (('GET', '/v1/models'), ('DELETE', '/v1/responses/unknown')):
                message = (method + ' ' + route + ' HTTP/1.1\r\nHost: fixture\r\nContent-Length: invalid\r\n\r\n').encode()
                statuses, _ = exchange(factory, message)
                self.assertEqual(statuses, [400])
            message = b'GET /v1/models HTTP/1.1\r\nHost: fixture\r\nContent-Length: 2\r\n\r\n{}'
            statuses, _ = exchange(factory, message)
            self.assertEqual(statuses, [200, 200])

    def test_token_counting_harness_uses_same_framing_and_nonempty_contract(self):
        path = Path(__file__).resolve().parents[1] / 'deploy/gb10/tensorfold_harness.py'
        spec = importlib.util.spec_from_file_location('owned_framing_harness_fixture', path)
        bridge = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(bridge)
        factory = bridge.make_handler_factory(cuda)
        for fields in ('Content-Length: invalid\r\n', 'Content-Length: 0\r\n',
                       'Content-Length: 2\r\nContent-Length: 3\r\n',
                       'Transfer-Encoding: chunked\r\nContent-Length: 2\r\n'):
            statuses, raw = exchange(factory, post('/v1/tokenize', fields))
            self.assertEqual(statuses, [400])
            self.assertIn(b'Connection: close\r\n', raw)
        body = b'{"model":"framing-fixture","messages":[]}'
        count = str(len(body))
        for fields in ('Content-Length: ' + count + '\r\n',
                       'Content-Length: ' + count + ', 0' + count + '\r\n'):
            statuses, raw = exchange(factory, post('/v1/tokenize', fields, body))
            self.assertEqual(statuses, [200, 200])
            self.assertIn(b'"count": 3', raw)

    def test_no_real_accelerator_runtime_imports(self):
        assert_accelerator_free_imports("tensorfold.server.request_body", "tensorfold.server.http", "tensorfold.cuda.http")

    def test_real_loopback_socket_contains_bad_frame_and_keeps_valid_pipeline(self):
        class App:
            served_name = 'socket-fixture'
            model_ids = [served_name]
            max_batch_size = 1

            def decisions(self, body):
                return {'preserved': body}

        for factory in (generic, cuda):
            workers = []
            finished = threading.Event()

            class Observed(Server):
                def process_request_thread(self, request, address):
                    workers.append(threading.current_thread())
                    try:
                        return super().process_request_thread(request, address)
                    finally:
                        finished.set()

            server = Observed(('127.0.0.1', 0), factory(App()))
            serving = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01}, daemon=True)
            try:
                with redirect_stdout(StringIO()):
                    serving.start()
                    for fields, body, expected in (('Content-Length: invalid\r\n', b'', [400]),
                        ('Content-Length: 02, 2\r\n', b'{}', [200, 200])):
                        finished.clear()
                        with socket.create_connection(('127.0.0.1', server.server_port), timeout=2) as connection:
                            connection.sendall(post('/v1/decisions', fields, body)
                                + b'GET /v1/models HTTP/1.1\r\nHost: fixture\r\nConnection: close\r\n\r\n')
                            output = bytearray()
                            while block := connection.recv(65536):
                                output.extend(block)
                                if len(output) > 65536:
                                    raise AssertionError('bounded socket fixture response exceeded limit')
                            statuses = [int(part.split(b' ', 1)[0]) for part in output.split(b'HTTP/1.1 ')[1:]]
                            self.assertEqual(statuses, expected)
                        self.assertTrue(finished.wait(2))
            finally:
                server.shutdown()
                serving.join(2)
                server.server_close()
                for worker in workers:
                    worker.join(2)
                    self.assertFalse(worker.is_alive())
                self.assertFalse(serving.is_alive())


if __name__ == '__main__':
    unittest.main()

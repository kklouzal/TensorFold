"""HTTP limits use real owned sockets and threads, without an accelerator SDK."""
from __future__ import annotations

from contextlib import contextmanager
import ast
import builtins
import http.client
import importlib.abc
import json
from pathlib import Path
import socket
import sys
import threading
from types import SimpleNamespace
import unittest


class _NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'numpy', 'mlx', 'triton', 'ctypes', 'xgrammar', 'tokenizers'}:
            raise RuntimeError('SDK/native import forbidden in HTTP source controls: ' + fullname)


def _sources():
    guard = _NoSDK()
    sys.meta_path.insert(0, guard)
    try:
        from tensorfold.cuda.http import make_handler as cuda_handler
        from tensorfold.server.http import Server, make_handler
        from tensorfold.server.errors import CapacityError
    finally:
        sys.meta_path.remove(guard)
    return Server, make_handler, cuda_handler, CapacityError


@contextmanager
def _running(handler, *, limit=None, poll=.01):
    from tensorfold.cleanup import raise_failures
    Server, _, _, _ = _sources()
    server = Server(('127.0.0.1', 0), handler, max_connections=limit)
    failures = []
    def serve():
        try:
            server.serve_forever(poll_interval=poll)
        except BaseException as error:
            failures.append(error)
    worker = threading.Thread(target=serve)
    primary = None
    try:
        worker.start()
        yield server, failures
    except BaseException as error:
        primary = error
    finally:
        cleanup = []
        for retire in (server.shutdown, server.server_close, lambda: worker.join(3)):
            try:
                retire()
            except BaseException as error:
                cleanup.append(error)
        if worker.is_alive():
            cleanup.append(AssertionError('owned server loop did not retire'))
        raise_failures(primary, [*failures, *cleanup])


class _App:
    """Explicit source fixture: admission and fixed output; no model execution."""
    model_ids = ['source-fixture']
    served_name = 'source-fixture'
    exact_mode = {'mode': 'source-fixture'}
    accepts_sampling = False
    accepts_cancellation = False
    effort_levels = frozenset()
    def __init__(self, *, limit, busy=False, empty=False):
        self.max_pending_requests, self.busy, self.empty = limit, busy, empty
        self.preparations = 0
    def prepare(self, body, chat):
        self.preparations += 1
        return body
    def reply_model(self, body):
        return self.served_name
    def _admit(self):
        if self.busy:
            raise _sources()[3]('configured request capacity is full; retry shortly')
    def run(self, body, chat, emit, **kwargs):
        self._admit()
        if kwargs.get('prepared') is None:
            self.prepare(body, chat)
        if not self.empty:
            emit({'content': 'source-fixture-output'})
        return {'final': {}, 'calls': [], 'stats': {}, 'finish': 'stop', 'content': 'source-fixture-output',
                'prompt_tokens': 1, 'completion_tokens': 0 if self.empty else 1, 'cached_tokens': 0}
    def chat(self, messages, **kwargs):
        self._admit()
        if not self.empty and kwargs.get('on_delta') is not None:
            kwargs['on_delta']('source-fixture-output')
        return {'content': '' if self.empty else 'source-fixture-output', 'finish_reason': 'stop',
                'prompt_tokens': 1, 'completion_tokens': 0 if self.empty else 1, 'cached_tokens': 0}


class HTTPServiceLimits(unittest.TestCase):
    def handlers(self, app):
        _, generic, cuda, _ = _sources()
        return generic(app), cuda(app)
    def request(self, server):
        client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        try:
            client.request('POST', '/v1/chat/completions', json.dumps(
                {'model': 'source-fixture', 'messages': [{'role': 'user', 'content': 'source fixture'}], 'stream': True}),
                {'Content-Type': 'application/json'})
            response = client.getresponse()
            return response.status, response.getheader('Content-Type'), response.read()
        finally:
            client.close()
    def test_configured_request_capacity_refuses_before_sse_for_both_codecs(self):
        app = _App(limit=1, busy=True)
        for handler in self.handlers(app):
            with self.subTest(handler=handler.__module__), _running(handler) as (server, failures):
                status, content_type, payload = self.request(server)
                self.assertEqual(status, 503)
                self.assertEqual(content_type, 'application/json')
                self.assertIn('capacity is full', json.loads(payload)['error']['message'])
                self.assertNotIn(b'data:', payload)
                self.assertEqual(failures, [])
        self.assertEqual(app.preparations, 0)
    def test_default_stream_headers_and_role_behavior_remain_for_both_codecs(self):
        for handler in self.handlers(_App(limit=None, busy=True)):
            with self.subTest(handler=handler.__module__), _running(handler) as (server, failures):
                status, content_type, payload = self.request(server)
                self.assertEqual(status, 200)
                self.assertEqual(content_type, 'text/event-stream')
                self.assertIn(b'"role": "assistant"', payload)
                self.assertIn(b'data: [DONE]', payload)
                self.assertEqual(failures, [])
    def test_configured_rpc_capacity_alone_refuses_before_generic_sse(self):
        app = _App(limit=None, busy=True)
        app.scheduler = SimpleNamespace(max_engine_calls=1)
        handler = _sources()[1](app)
        with _running(handler) as (server, failures):
            status, content_type, payload = self.request(server)
            self.assertEqual((status, content_type), (503, 'application/json'))
            self.assertIn('capacity is full', json.loads(payload)['error']['message'])
            self.assertNotIn(b'data:', payload)
            self.assertEqual(failures, [])
    def test_deferred_success_keeps_role_then_content_and_final_even_without_tokens(self):
        for empty in (False, True):
            for handler in self.handlers(_App(limit=1, empty=empty)):
                with self.subTest(empty=empty, handler=handler.__module__), _running(handler) as (server, failures):
                    status, content_type, payload = self.request(server)
                    self.assertEqual((status, content_type), (200, 'text/event-stream'))
                    self.assertEqual(payload.count(b'"role": "assistant"'), 1)
                    if not empty:
                        self.assertLess(payload.index(b'"role": "assistant"'), payload.index(b'source-fixture-output'))
                    self.assertIn(b'data: [DONE]', payload)
                    self.assertEqual(failures, [])
    def test_invalid_connection_limits_refuse_before_socket_creation(self):
        Server = _sources()[0]
        for value in (0, -1, True, 1.5, '2'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                Server(('127.0.0.1', 0), self.handlers(_App(limit=None))[0], max_connections=value)
    def test_tcp_nodelay_is_selected_only_on_real_keepalive_reuse(self):
        for handler in self.handlers(_App(limit=None)):
            observed = []
            lock = threading.Lock()
            original_get = handler.do_GET
            def get(owner, original=original_get):
                with lock:
                    observed.append((owner.connection.getsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY),
                                     'parse_request' in owner.__dict__))
                original(owner)
            handler.do_GET = get
            with self.subTest(handler=handler.__module__), _running(handler) as (server, failures):
                client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
                try:
                    for headers in ({}, {}, {'Connection': 'close'}):
                        client.request('GET', '/v1/models', headers=headers)
                        response = client.getresponse()
                        self.assertEqual(response.status, 200)
                        response.read()
                finally:
                    client.close()
                with lock:
                    enabled = int(sys.platform == 'linux')
                    self.assertEqual(observed, [(0, False), (enabled, False), (enabled, False)])
                close_client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
                try:
                    close_client.request('GET', '/v1/models', headers={'Connection': 'close'})
                    response = close_client.getresponse()
                    self.assertEqual(response.status, 200)
                    response.read()
                finally:
                    close_client.close()
                with lock:
                    self.assertEqual(observed[-1], (0, False))
                self.assertEqual(failures, [])
    def test_platform_selection_preserves_stock_parser_outside_linux(self):
        from http.server import BaseHTTPRequestHandler
        source = (Path(__file__).resolve().parents[1] / 'src/tensorfold/server/stacks.py').read_text()
        observed = {}
        for platform in ('linux', 'darwin', 'win32'):
            original_import = builtins.__import__
            def import_for_source(name, *args, selected=platform, **kwargs):
                return SimpleNamespace(platform=selected) if name == 'sys' else original_import(name, *args, **kwargs)
            namespace = {'__builtins__': {**vars(builtins), '__import__': import_for_source}}
            exec(compile(ast.parse(source), 'isolated-stacks-source', 'exec'), namespace)
            observed[platform] = namespace['Rearming'].__dict__.get('parse_request')
            if platform != 'linux':
                self.assertIs(namespace['Rearming'].parse_request, BaseHTTPRequestHandler.parse_request)
        self.assertIsNotNone(observed['linux'])
        self.assertIsNone(observed['darwin'])
        self.assertIsNone(observed['win32'])
    def test_configured_health_and_metrics_report_owned_policy_counts(self):
        from tensorfold.server.request_limits import RequestLimit
        app = _App(limit=3)
        app._request_limit = RequestLimit(3)
        app.max_batch_size = 4
        app.prompt_memory = SimpleNamespace(memory_snapshot=lambda reset: {})
        app.scheduler = SimpleNamespace(max_engine_calls=2, engine_calls=1)
        with app._request_limit.request():
            for handler in self.handlers(app):
                with self.subTest(handler=handler.__module__), _running(handler, limit=2) as (server, failures):
                    client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
                    try:
                        client.request('GET', '/health')
                        body = json.loads(client.getresponse().read())
                        self.assertEqual(body['admission'], {
                            'requests': {'in_use': 1, 'limit': 3}, 'engine_calls': {'in_use': 1, 'limit': 2},
                            'http_connections': {'in_use': 1, 'limit': 2}})
                        client.request('GET', '/metrics')
                        text = client.getresponse().read().decode()
                        for value in ('http_connections_in_use 1', 'http_connections_limit 2',
                                      'request_admission_in_use 1', 'request_admission_limit 3',
                                      'engine_calls_in_use 1', 'engine_calls_limit 2'):
                            self.assertIn('tensorfold:' + value, text)
                        self.assertEqual(failures, [])
                    finally:
                        client.close()
    def test_unconfigured_transport_does_not_install_capacity_dispatch_or_event(self):
        Server = _sources()[0]
        server = Server(('127.0.0.1', 0), self.handlers(_App(limit=None))[0])
        try:
            self.assertIsNone(server.max_connections)
            self.assertIsNone(server._capacity_changed)
            self.assertNotIn('_handle_request_noblock', server.__dict__)
            self.assertNotIn('service_actions', server.__dict__)
        finally:
            server.server_close()
    def _models_handler(self):
        return self.handlers(_App(limit=None))[1]
    def test_at_capacity_retirement_wakes_acceptance_without_poll_timeout(self):
        with _running(self._models_handler(), limit=1, poll=30) as (server, failures):
            first = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
            second = None
            try:
                first.request('GET', '/v1/models')
                self.assertEqual(first.getresponse().read(), b'{"object": "list", "data": [{"id": "source-fixture", "object": "model", "owned_by": "tensorfold"}]}')
                self.assertEqual(server.connection_snapshot(), {'in_use': 1, 'limit': 1})
                second = socket.create_connection(('127.0.0.1', server.server_port), timeout=3)
                second.sendall(b'GET /v1/models HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n')
                second.settimeout(.05)
                with self.assertRaises(TimeoutError):
                    second.recv(1)
                self.assertEqual(server.connection_snapshot()['in_use'], 1)
                first.close()
                second.settimeout(3)
                self.assertTrue(second.recv(4096).startswith(b'HTTP/1.1 200'))
                self.assertEqual(failures, [])
            finally:
                first.close()
                if second is not None:
                    second.close()
    def test_retirement_before_full_observation_is_not_discarded_by_event_clear(self):
        from tensorfold.cleanup import raise_failures
        Server = _sources()[0]
        server = Server(('127.0.0.1', 0), self._models_handler(), max_connections=1)
        server._capacity_poll = 30
        client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
        caller = None
        failures = []
        primary = None
        try:
            client.request('GET', '/v1/models', headers={'Connection': 'close'})
            server._accept_without_limit()
            response = client.getresponse()
            self.assertEqual(response.status, 200)
            response.read()
            with server._connection_lock:
                journal = next(iter(server._connections.values()))
            journal['thread'].join(3)
            self.assertFalse(journal['thread'].is_alive())
            self.assertTrue(journal['closed'])
            self.assertTrue(server._capacity_changed.is_set())
            completed = threading.Event()
            def dispatch():
                try:
                    server._handle_request_noblock()
                except BaseException as error:
                    failures.append(error)
                finally:
                    completed.set()
            caller = threading.Thread(target=dispatch)
            caller.start()
            self.assertTrue(completed.wait(3), 'already retired owner was made to wait for another notification')
            self.assertEqual(server.connection_snapshot(), {'in_use': 0, 'limit': 1})
        except BaseException as error:
            primary = error
        finally:
            cleanup = []
            for retire in (client.close, server.shutdown, server.server_close):
                try:
                    retire()
                except BaseException as error:
                    cleanup.append(error)
            if caller is not None:
                caller.join(3)
                if caller.is_alive():
                    cleanup.append(AssertionError('owned dispatch did not retire'))
            raise_failures(primary, [*failures, *cleanup])
    def test_shutdown_wakes_saturated_acceptance_without_poll_timeout(self):
        with _running(self._models_handler(), limit=1, poll=30) as (server, failures):
            first = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
            second = None
            try:
                first.request('GET', '/v1/models')
                first.getresponse().read()
                second = socket.create_connection(('127.0.0.1', server.server_port), timeout=3)
                second.sendall(b'GET /v1/models HTTP/1.1\r\nHost: localhost\r\n\r\n')
                second.settimeout(.05)
                with self.assertRaises(TimeoutError):
                    second.recv(1)
                completed = threading.Event()
                caller = threading.Thread(target=lambda: (server.shutdown(), completed.set()))
                caller.start()
                try:
                    self.assertTrue(completed.wait(3))
                finally:
                    caller.join(3)
                self.assertFalse(caller.is_alive())
                self.assertEqual(failures, [])
            finally:
                first.close()
                if second is not None:
                    second.close()


if __name__ == '__main__':
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
    unittest.main(verbosity=2)

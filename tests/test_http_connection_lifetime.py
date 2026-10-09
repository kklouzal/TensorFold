"""Owned HTTP shutdown using stdlib threads/sockets, without a model runtime."""
from contextlib import redirect_stderr, redirect_stdout
from http.client import HTTPConnection
from http.server import BaseHTTPRequestHandler
from io import StringIO
import inspect
import socket
import sys
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from tensorfold import cli
from tensorfold.server.cancellation import socket_cancellation
from tensorfold.server.http import Server, make_handler


def launch(function):
    results = []

    def run():
        try:
            results.append(function())
        except BaseException as error:
            results.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    return worker, results


def join(worker):
    worker.join(3)
    if worker.is_alive():
        raise AssertionError("owned fixture thread did not drain")


class Socket:
    def __init__(self):
        self.interrupted = threading.Event()
        self.closed = False

    def shutdown(self, how):
        self.interrupted.set()

    def close(self):
        self.closed = True


class Lifetime(unittest.TestCase):
    def test_real_idle_keepalive_is_interrupted_and_joined_before_return(self):
        app = SimpleNamespace(served_name="fixture", model_ids=["fixture"], max_batch_size=1)
        server = Server(("127.0.0.1", 0), make_handler(app))
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        serving = None
        try:
            with redirect_stdout(StringIO()):
                serving, result = launch(lambda: server.serve_forever(.01))
                for _ in range(2):
                    connection.request("GET", "/v1/models")
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    self.assertIn(b'"fixture"', response.read())
                    response.close()
                with server._connection_lock:
                    workers = [j["thread"] for j in server._connections.values()]
                self.assertTrue(workers)
                self.assertTrue(all(not t.daemon for t in workers))
                server.server_close()  # also stops the distinct serving loop
                join(serving)
                self.assertEqual(result, [None])
                self.assertTrue(server.handlers_drained)
                self.assertFalse(server._connections)
                self.assertTrue(all(not t.is_alive() for t in workers))
                server.server_close()
        finally:
            connection.close()
            if serving is not None and serving.is_alive():
                server.shutdown()
                join(serving)
            server.server_close()

    def test_real_partial_body_shutdown_unblocks_reader_without_client_help(self):
        from tensorfold.server import request_body
        entered = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                entered.set()
                request_body.read(self, 128)

        server = Server(("127.0.0.1", 0), Handler)
        serving, _ = launch(lambda: server.serve_forever(.01))
        peer = socket.create_connection(server.server_address, timeout=2)
        try:
            with redirect_stderr(StringIO()):
                peer.sendall(b"POST / HTTP/1.1\r\nHost: fixture\r\nContent-Length: 100\r\n\r\nx")
                self.assertTrue(entered.wait(2))
                server.server_close()
                join(serving)
                self.assertTrue(server.handlers_drained)
        finally:
            peer.close()
            if serving.is_alive():
                server.shutdown()
                join(serving)
            server.server_close()

    def test_active_handler_sees_explicit_cancellation_and_close_waits_for_retirement(self):
        entered, release = threading.Event(), threading.Event()
        observed = []
        cancelled = threading.Event()

        def handler(request, address, server):
            entered.set()
            self.assertTrue(server.stopping.wait(2))
            a, b = socket.socketpair()
            try:
                # Pending socket data need not be consumed for explicit stop.
                b.sendall(b"pending")
                observed.append(socket_cancellation(a, stopping=server.stopping).cancelled)
                cancelled.set()
            finally:
                a.close()
                b.close()
            self.assertTrue(release.wait(2))

        server, request = Server(("127.0.0.1", 0), handler), Socket()
        closing = None
        try:
            server.process_request(request, None)
            self.assertTrue(entered.wait(2))
            closing, result = launch(server.server_close)
            self.assertTrue(request.interrupted.wait(2))
            self.assertTrue(closing.is_alive())
            self.assertFalse(server.handlers_drained)
            self.assertFalse(request.closed)
            self.assertTrue(cancelled.wait(2))
            self.assertEqual(observed, [True])
        finally:
            release.set()
            if closing is not None:
                join(closing)
                self.assertEqual(result, [None])
            server.server_close()
        self.assertTrue(request.closed)
        self.assertTrue(server.handlers_drained)

    def test_new_admission_and_reentrant_handler_close_are_rejected(self):
        entered, release = threading.Event(), threading.Event()
        reentry = []

        def handler(request, address, server):
            try:
                server.server_close()
            except RuntimeError as error:
                reentry.append(error)
            entered.set()
            self.assertTrue(release.wait(2))

        server, request = Server(("127.0.0.1", 0), handler), Socket()
        try:
            server.process_request(request, None)
            self.assertTrue(entered.wait(2))
            self.assertEqual(len(reentry), 1)
            self.assertFalse(server.stopping.is_set())
            closing, result = launch(server.server_close)
            self.assertTrue(server.stopping.wait(2))
            refused = Socket()
            server.process_request(refused, None)
            self.assertTrue(refused.closed)
        finally:
            release.set()
            join(closing)
            server.server_close()
        self.assertEqual(result, [None])

    def test_concurrent_close_joins_all_workers_before_either_returns(self):
        entered, release = threading.Event(), threading.Event()

        def handler(*args):
            entered.set()
            self.assertTrue(release.wait(2))

        server, request = Server(("127.0.0.1", 0), handler), Socket()
        workers = []
        try:
            server.process_request(request, None)
            self.assertTrue(entered.wait(2))
            workers = [launch(server.server_close) for _ in range(2)]
            self.assertTrue(server.stopping.wait(2))
            self.assertTrue(all(w.is_alive() for w, _ in workers))
        finally:
            release.set()
            for w, r in workers:
                join(w)
                self.assertEqual(r, [None])
            server.server_close()

    def test_interrupted_join_retains_journal_for_explicit_retry(self):
        entered, release = threading.Event(), threading.Event()

        def handler(*args):
            entered.set()
            self.assertTrue(release.wait(2))

        server, request = Server(("127.0.0.1", 0), handler), Socket()
        try:
            server.process_request(request, None)
            self.assertTrue(entered.wait(2))
            worker = server._connections[request]["thread"]
            primary = KeyboardInterrupt()
            release.set()                 # join now follows retirement of the native handler/socket scope
            with patch.object(worker, "join", side_effect=primary):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    server.server_close()
                self.assertIs(caught.exception, primary)
            self.assertIn(request, server._connections)
            self.assertFalse(server.handlers_drained)
        finally:
            release.set()
            server.server_close()
        self.assertTrue(server.handlers_drained)

    def test_thread_construction_failure_keeps_the_accepted_socket_owned_for_drain(self):
        server, request = Server(("127.0.0.1", 0), lambda *args: None), Socket()
        primary = KeyboardInterrupt("actor construction interrupted")
        try:
            with patch.object(threading, "Thread", side_effect=primary):
                with self.assertRaises(KeyboardInterrupt) as caught:
                    server.process_request(request, None)
            self.assertIs(caught.exception, primary)
            self.assertIn(request, server._connections)
            self.assertFalse(server._connections[request]["start_attempted"])
            self.assertIsNone(server._connections[request]["thread"])
            self.assertFalse(request.closed)
        finally:
            server.server_close()
        self.assertTrue(request.closed)
        self.assertTrue(server.handlers_drained)
        self.assertFalse(server._connections)

    def test_ambiguous_start_failure_retains_socket_until_worker_settles(self):
        for after in (False, True):
            with self.subTest(after=after):
                server, request = Server(("127.0.0.1", 0), lambda *args: None), Socket()
                primary = KeyboardInterrupt()
                original = threading.Thread.start

                def start(worker):
                    if after:
                        original(worker)
                    raise primary

                try:
                    with patch.object(threading.Thread, "start", start):
                        with self.assertRaises(KeyboardInterrupt) as caught:
                            server.process_request(request, None)
                    self.assertIs(caught.exception, primary)
                    server.shutdown_request(request)  # BaseServer's caller cleanup
                    if not after:
                        self.assertFalse(request.closed)
                        with self.assertRaisesRegex(RuntimeError, "start did not settle"):
                            server.server_close()
                        self.assertFalse(server.handlers_drained)
                        original(server._connections[request]["thread"])  # fixture resolves ambiguous start
                    join(server._connections[request]["thread"])
                finally:
                    server.server_close()
                self.assertTrue(request.closed)
                self.assertTrue(server.handlers_drained)

    def test_socket_interrupt_failure_retains_model_owners_and_retry(self):
        entered, release = threading.Event(), threading.Event()

        def handler(*args):
            entered.set()
            self.assertTrue(release.wait(2))

        server, request = Server(("127.0.0.1", 0), handler), Socket()
        primary, order = RuntimeError("interrupt failed"), []
        owners = {"server": server, "app": SimpleNamespace(close=lambda: order.append("app")),
                  "engine": SimpleNamespace(close=lambda: order.append("engine"))}
        try:
            server.process_request(request, None)
            self.assertTrue(entered.wait(2))
            with patch.object(request, "shutdown", side_effect=primary):
                with self.assertRaises(RuntimeError) as caught:
                    cli._close_serving(owners, None, lambda: order.append("unwire"))
            self.assertIs(caught.exception, primary)
            self.assertEqual(order, [])
            self.assertFalse(server.handlers_drained)
        finally:
            release.set()
            cli._close_serving(owners, None, lambda: order.append("unwire"))
        self.assertEqual(order, ["app", "engine", "unwire"])

    def test_completed_worker_journal_is_reaped_at_control_point(self):
        server, request = Server(("127.0.0.1", 0), lambda *args: None), Socket()
        try:
            server.process_request(request, None)
            worker = server._connections[request]["thread"]
            join(worker)
            self.assertIn(request, server._connections)
            server.service_actions()
            self.assertFalse(server._connections)
        finally:
            server.server_close()

    def test_legal_single_request_api_reaps_completed_workers(self):
        app = SimpleNamespace(served_name="fixture", model_ids=["fixture"], max_batch_size=1)
        server = Server(("127.0.0.1", 0), make_handler(app))
        connection = HTTPConnection("127.0.0.1", server.server_port, timeout=2)
        try:
            with redirect_stdout(StringIO()):
                for _ in range(3):
                    serving, outcome = launch(server.handle_request)
                    connection.request("GET", "/v1/models", headers={"Connection": "close"})
                    response = connection.getresponse()
                    self.assertEqual(response.status, 200)
                    response.read()
                    response.close()
                    connection.close()
                    join(serving)
                    self.assertEqual(outcome, [None])
                    for journal in tuple(server._connections.values()):
                        join(journal["thread"])
                    server.timeout = 0
                    server.handle_request()  # no next arrival is needed to reap
                    server.timeout = None
                    self.assertFalse(server._connections)
        finally:
            connection.close()
            server.server_close()

    def test_shutdown_waits_for_owned_loop_even_before_selector_startup(self):
        from tensorfold.server import http
        entered, release = threading.Event(), threading.Event()
        real_selector = getattr(http.selectors, "PollSelector", http.selectors.SelectSelector)

        class Delayed:
            def __enter__(self):
                entered.set()
                if not release.wait(2):
                    raise AssertionError("selector startup release deadline")
                self.inner = real_selector()
                return self.inner.__enter__()

            def __exit__(self, *args):
                return self.inner.__exit__(*args)

        server = Server(("127.0.0.1", 0), lambda *args: None)
        serving = closing = None
        try:
            with patch.object(http.selectors, "PollSelector", Delayed, create=True):
                serving, outcome = launch(lambda: server.serve_forever(.01))
                self.assertTrue(entered.wait(2))
                closing, result = launch(server.server_close)
                self.assertTrue(server.stopping.wait(2))
                self.assertTrue(closing.is_alive())
                release.set()
                join(serving)
                join(closing)
                self.assertEqual(outcome, [None])
                self.assertEqual(result, [None])
        finally:
            release.set()
            if serving is not None:
                join(serving)
            if closing is not None:
                join(closing)
            server.server_close()

    def test_interrupt_before_python_loop_finalizer_cannot_strand_shutdown(self):
        lines, first = inspect.getsourcelines(Server.serve_forever)
        line = next(first + i + 1 for i, s in enumerate(lines) if s.strip() == "finally:")
        primary = KeyboardInterrupt()
        server = Server(("127.0.0.1", 0), lambda *args: None)

        def trace(frame, event, arg):
            if frame.f_code is Server.serve_forever.__code__ and event == "line" and frame.f_lineno == line:
                sys.settrace(None)
                raise primary
            return trace

        # End the loop at its first control point, then interrupt BEFORE its
        # Python finalizer. Native with unwinding must release the lease.
        server.service_actions = lambda: server._serve_stop.set()
        try:
            sys.settrace(trace)
            with self.assertRaises(KeyboardInterrupt) as caught:
                server.serve_forever(0)
            self.assertIs(caught.exception, primary)
        finally:
            sys.settrace(None)
        self.assertIsNotNone(server._serving_thread)
        self.assertFalse(server._serving_thread[1].locked())
        server.shutdown()
        server.server_close()
        self.assertTrue(server.handlers_drained)

    def test_reentrant_loop_and_self_shutdown_cannot_release_original_owner(self):
        observed = []
        server = Server(("127.0.0.1", 0), lambda *args: None)

        def control():
            owner = server._serving_thread
            for call in (lambda: server.serve_forever(0), server.shutdown, server.server_close):
                with self.assertRaises(RuntimeError):
                    call()
                self.assertIs(server._serving_thread, owner)
            observed.append(True)
            server._serve_stop.set()

        server.service_actions = control
        try:
            server.serve_forever(0)
            self.assertEqual(observed, [True])
        finally:
            server.server_close()

    def test_fatal_handler_failure_stops_loop_and_primary_survives_drained_cleanup(self):
        class Opaque(BaseException):
            def __str__(self):
                raise AssertionError("foreign failure formatted")

            def add_note(self, value):
                raise AssertionError("foreign failure note called")

        primary = Opaque()

        def handler(*args):
            raise primary

        server, request = Server(("127.0.0.1", 0), handler), Socket()
        order = []
        server.process_request(request, None)
        join(server._connections[request]["thread"])
        with self.assertRaises(Opaque) as caught:
            server.service_actions()
        self.assertIs(caught.exception, primary)
        owners = {"server": server, "app": SimpleNamespace(close=lambda: order.append("app"))}
        with self.assertRaises(Opaque) as caught:
            cli._close_serving(owners, primary, lambda: order.append("unwire"))
        self.assertIs(caught.exception, primary)
        self.assertTrue(server.handlers_drained)
        self.assertEqual(order, ["app", "unwire"])
        self.assertIsNone(primary.__cause__)

    def test_failed_worker_socket_close_is_joined_and_retried_before_return(self):
        server, request = Server(("127.0.0.1", 0), lambda *args: None), Socket()
        primary = RuntimeError("socket close failure")
        try:
            with patch.object(request, "close", side_effect=primary):
                server.process_request(request, None)
                worker = server._connections[request]["thread"]
                join(worker)
                self.assertIn(request, server._connections)
                self.assertIs(server._connections[request]["close_error"], primary)
                with self.assertRaises(RuntimeError) as caught:
                    server.server_close()
                self.assertIs(caught.exception, primary)
                self.assertFalse(server.handlers_drained)
        finally:
            server.server_close()
        self.assertTrue(request.closed)
        self.assertTrue(server.handlers_drained)

    def test_no_accelerator_imports(self):
        self.assertFalse(any(name in sys.modules for name in ("torch", "numpy", "mlx")))


if __name__ == "__main__":
    unittest.main()

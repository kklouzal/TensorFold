"""HTTP allocation containment controls; injected failures are not native OOM tests."""
from io import BytesIO
import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda.http import make_handler as cuda_handler
from tensorfold.server import json_output, responses, stats
from tensorfold.server.http import Server, make_handler as generic_handler


def _app(*, certified=True):
    failures = []
    envelope = {"http_owner_scope": {"schema": "fixture"}, "max_http_connections": 2} if certified else None
    return SimpleNamespace(
        engine=SimpleNamespace(retention_catalog=SimpleNamespace(host_resource_envelope=envelope)),
        max_pending_requests=1, _http_payload_failure=None,
        _fatal_http_payload=failures.append, failures=failures)


@pytest.fixture(params=[cuda_handler, generic_handler], ids=["actual-cuda", "generic"])
def endpoint(request):
    app = _app()
    handler_type = request.param(app)
    server = Server(("127.0.0.1", 0), handler_type, max_connections=2)
    handler = handler_type.__new__(handler_type)
    handler.server = server
    handler.close_connection = False
    handler.wfile = BytesIO()
    handler.send_response = lambda *args: None
    handler.send_header = lambda *args: None
    handler.end_headers = lambda: None
    try:
        yield app, server, handler
    finally:
        expected = server._handler_failure
        prior = None if expected is None else (
            expected.__cause__, expected.__context__, tuple(getattr(expected, "__notes__", ())))
        try:
            server.server_close()
        except BaseException as error:
            # Retirement intentionally reports the previously asserted fatal
            # status. A distinct cleanup failure must still fail this fixture.
            current = (error.__cause__, error.__context__, tuple(getattr(error, "__notes__", ())))
            if error is not expected or current != prior:
                raise


def test_wire_preserves_ascii_json_and_framing(endpoint):
    _, _, handler = endpoint
    assert json_output.dumps(handler, {"text": "é"}, prefix="data: ", suffix="\n\n", as_bytes=True) == (
        b'data: {"text": "\\u00e9"}\n\n')


@pytest.mark.parametrize("factory", [cuda_handler, generic_handler])
@pytest.mark.parametrize("actual", [None, 3])
def test_listener_cannot_exceed_admitted_handler_owners(factory, actual):
    with pytest.raises(ValueError, match="HTTP|admitted|connection"):
        Server(("127.0.0.1", 0), factory(_app()), max_connections=actual)


@pytest.mark.parametrize("factory", [cuda_handler, generic_handler])
@pytest.mark.parametrize("actual", [1, 2])
def test_listener_can_use_fewer_than_admitted_handler_owners(factory, actual):
    server = Server(("127.0.0.1", 0), factory(_app()), max_connections=actual)
    try:
        assert server.max_connections == actual
    finally:
        server.server_close()


def test_encoder_failure_poison_is_published_before_any_retry(endpoint, monkeypatch):
    app, server, handler = endpoint
    primary = MemoryError("injected encoder allocation failure")
    calls = []

    def failed(value):
        calls.append(value)
        raise primary

    monkeypatch.setattr(json_output.json, "dumps", failed)
    with pytest.raises(MemoryError) as caught:
        json_output.dumps(handler, {"first": True})
    assert caught.value is primary
    assert server.stopping.is_set() and handler.close_connection
    assert app.failures == [primary]
    with pytest.raises(MemoryError) as caught:
        json_output.dumps(handler, {"retry": True})
    assert caught.value is primary
    assert calls == [{"first": True}]
    with pytest.raises(MemoryError) as caught:
        server.service_actions()
    assert caught.value is primary


def test_actual_handler_terminal_serializer_uses_containment(endpoint, monkeypatch):
    app, _, handler = endpoint
    primary = MemoryError("injected terminal serialization failure")

    def failed(value):
        raise primary

    monkeypatch.setattr(json_output.json, "dumps", failed)
    with pytest.raises(MemoryError) as caught:
        if hasattr(handler, "_json"):
            handler._json(200, {"answer": "fixture"})
        else:
            handler._send_json({"answer": "fixture"})
    assert caught.value is primary and app.failures == [primary]
    assert handler.wfile.getvalue() == b""


def test_responses_serializer_uses_same_owner(endpoint, monkeypatch):
    app, _, handler = endpoint
    primary = MemoryError("injected Responses serialization failure")

    def failed(value):
        raise primary

    monkeypatch.setattr(json_output.json, "dumps", failed)
    with pytest.raises(MemoryError) as caught:
        responses._send(handler, 200, {"object": "response"})
    assert caught.value is primary and app.failures == [primary]


def test_notification_failure_preserves_allocation_primary(endpoint, monkeypatch):
    _, server, handler = endpoint
    primary = MemoryError("injected encoder failure")
    secondary = OSError("injected cohort notification failure")

    def failed(value):
        raise primary

    def notify(error):
        assert error is primary
        raise secondary

    handler.notify_json_failure = notify
    monkeypatch.setattr(json_output.json, "dumps", failed)
    with pytest.raises(MemoryError) as caught:
        json_output.dumps(handler, {})
    assert caught.value is primary and primary.__cause__ is secondary
    assert server.stopping.is_set() and handler.close_connection


def test_handler_diagnostic_failure_keeps_published_encoder_primary(endpoint, monkeypatch):
    _, server, handler = endpoint
    primary = MemoryError("injected encoder failure")
    secondary = OSError("injected diagnostic failure")

    class Socket:
        def shutdown(self, how):
            pass

        def close(self):
            pass

    request = Socket()
    journal = {"lease": threading.Lock(), "entry": threading.Event(), "closed": False}
    server._connections[request] = journal

    def failed(value):
        raise primary

    def finish(*args):
        json_output.dumps(handler, {})

    def diagnostic(*args):
        raise secondary

    monkeypatch.setattr(json_output.json, "dumps", failed)
    monkeypatch.setattr(server, "finish_request", finish)
    monkeypatch.setattr(server, "handle_error", diagnostic)
    try:
        server.process_request_thread(request, ("fixture", 0))
        assert server._handler_failure is primary
        assert primary.__cause__ is secondary
        assert journal["closed"] and server.stopping.is_set()
    finally:
        server._connections.pop(request)


def test_stats_serializer_uses_same_owner(endpoint, monkeypatch):
    app, _, handler = endpoint
    handler.path = "/stats"
    primary = MemoryError("injected stats serialization failure")
    monkeypatch.setattr(stats, "snapshot", lambda *args, **kwargs: {"fixture": True})

    def failed(value):
        raise primary

    monkeypatch.setattr(json_output.json, "dumps", failed)
    with pytest.raises(MemoryError) as caught:
        stats.send(handler, app)
    assert caught.value is primary and app.failures == [primary]


def test_authentication_refusal_uses_same_owner(endpoint, monkeypatch):
    from tensorfold.server import auth_http

    app, server, handler = endpoint
    app.auth = SimpleNamespace(enabled=True, metrics_open=False, match_identity=lambda headers: (None, None))
    wrapped = auth_http.handler(type(handler), app)
    request = wrapped.__new__(wrapped)
    request.server, request.headers = server, {}
    request.command, request.path = "GET", "/v1/models"
    request.close_connection = False
    primary = MemoryError("injected authentication refusal encoding failure")

    def failed(value):
        raise primary

    monkeypatch.setattr(json_output.json, "dumps", failed)
    with pytest.raises(MemoryError) as caught:
        request._authenticate()
    assert caught.value is primary and app.failures == [primary]
    assert request.close_connection and server.stopping.is_set()


def test_inflight_encode_may_finish_but_later_encodes_are_refused(endpoint, monkeypatch):
    app, server, handler = endpoint
    entered, release = threading.Event(), threading.Event()
    primary = MemoryError("injected second-handler failure")
    original = json_output.json.dumps
    calls, outcomes = [], []

    def controlled(value):
        calls.append(value)
        if value == "inflight":
            entered.set()
            if not release.wait(3):
                raise AssertionError("fixture did not release its owned encoder")
            return original(value)
        raise primary

    def run():
        try:
            outcomes.append(json_output.dumps(handler, "inflight"))
        except BaseException as error:
            outcomes.append(error)

    monkeypatch.setattr(json_output.json, "dumps", controlled)
    worker = threading.Thread(target=run)
    worker.start()
    try:
        assert entered.wait(3)
        other = type(handler).__new__(type(handler))
        other.server, other.close_connection = server, False
        with pytest.raises(MemoryError) as caught:
            json_output.dumps(other, "fails")
        assert caught.value is primary
        with pytest.raises(MemoryError) as caught:
            json_output.dumps(other, "late")
        assert caught.value is primary
    finally:
        release.set()
        worker.join(3)
    assert not worker.is_alive()
    assert outcomes == ['"inflight"']
    assert calls == ["inflight", "fails"] and app.failures == [primary]


def test_noncertified_callers_keep_request_local_failure(monkeypatch):
    handler = SimpleNamespace(certified_json_payloads=False, close_connection=False)
    primary = MemoryError("injected ordinary caller failure")

    def failed(value):
        raise primary

    monkeypatch.setattr(json_output.json, "dumps", failed)
    with pytest.raises(MemoryError) as caught:
        json_output.dumps(handler, {})
    assert caught.value is primary and handler.close_connection is False


@pytest.mark.parametrize("factory", [cuda_handler, generic_handler])
def test_poisoned_app_cannot_reopen_a_handler(factory):
    app = _app()
    primary = MemoryError("previous fatal allocation")
    app._http_payload_failure = primary
    with pytest.raises(MemoryError) as caught:
        factory(app)
    assert caught.value is primary


def test_actual_cuda_app_poison_publishes_abort_and_notifies_once(monkeypatch):
    from tensorfold.cuda import cohort
    from tensorfold.cuda.server import App

    app = App.__new__(App)
    app.engine = SimpleNamespace(_abort_comm=False, retention_catalog=SimpleNamespace(
        host_resource_envelope={"http_owner_scope": {"schema": "fixture"}, "max_http_connections": 2}))
    app.max_pending_requests = 1
    app._http_payload_failure = None
    primary = MemoryError("injected actual App allocation failure")
    observed = []

    def notify():
        observed.append((app._http_payload_failure, app.engine._abort_comm))
        return True

    monkeypatch.setattr(cohort, "notify_failure", notify)
    app._fatal_http_payload(primary)
    app._fatal_http_payload(MemoryError("later failure"))
    assert observed == [(primary, True)]
    assert app._http_payload_failure is primary and app.engine._abort_comm is True
    with pytest.raises(MemoryError) as caught:
        cuda_handler(app)
    assert caught.value is primary

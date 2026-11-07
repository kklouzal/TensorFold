"""CUDA serving cleanup keeps primary identity without importing a model SDK."""

import signal
from types import SimpleNamespace
from unittest.mock import patch

from tensorfold.cuda.http import serve


class Opaque(BaseException):
    def __str__(self):
        raise AssertionError("foreign failure formatted")

    def add_note(self, note):
        raise AssertionError("foreign failure note called")


def serving(primary=None, cleanup=None):
    calls = []

    def run():
        calls.append("serve")
        if primary is not None:
            raise primary

    def close():
        calls.append("close")
        if cleanup is not None:
            raise cleanup

    server = SimpleNamespace(serve_forever=run, server_close=close)
    with patch.object(signal, "signal"):
        try:
            serve(None, "127.0.0.1", 0, server=server)
        except BaseException as error:
            return calls, error
    return calls, None


def test_normal_serving_and_termination_drain_the_supplied_owner():
    for primary in (None, KeyboardInterrupt()):
        calls, error = serving(primary)
        assert calls == ["serve", "close"] and error is None


def test_primary_identity_and_cleanup_context_survive_cuda_serve():
    primary, cleanup = Opaque(), Opaque()
    calls, error = serving(primary, cleanup)
    assert calls == ["serve", "close"]
    assert error is primary and error.__cause__ is cleanup


def test_cleanup_failure_after_normal_serving_or_termination_is_reported():
    for primary in (None, KeyboardInterrupt()):
        cleanup = Opaque()
        calls, error = serving(primary, cleanup)
        assert calls == ["serve", "close"] and error is cleanup


def test_identical_primary_and_cleanup_failure_do_not_self_chain():
    primary = Opaque()
    calls, error = serving(primary, primary)
    assert calls == ["serve", "close"] and error is primary
    assert primary.__cause__ is None


def certified_app():
    return SimpleNamespace(engine=SimpleNamespace(retention_catalog=SimpleNamespace(
        host_resource_envelope={"http_owner_scope": {"schema": "fixture"}, "max_http_connections": 2})),
        max_pending_requests=1, max_http_connections=2, _http_payload_failure=None,
        _fatal_http_payload=lambda primary: None)


def test_default_serve_passes_the_actual_admitted_connection_cap(monkeypatch):
    from tensorfold.cuda import http
    from tensorfold.server.http import Server

    seen = []
    class ObservedServer(Server):
        def serve_forever(self):
            seen.append((self.max_connections, self.RequestHandlerClass.certified_http_owner))

    app = certified_app()
    monkeypatch.setattr(http, "Server", ObservedServer)
    with patch.object(signal, "signal"):
        http.serve(app, "127.0.0.1", 0)
    assert seen == [(2, app)]


def test_supplied_certified_server_uses_same_app_and_may_have_lower_cap(monkeypatch):
    from tensorfold.cuda import http
    from tensorfold.server.http import Server

    app = certified_app()
    owned = Server(("127.0.0.1", 0), http.make_handler(app), max_connections=1)
    calls = []
    monkeypatch.setattr(owned, "serve_forever", lambda: calls.append("serve"))
    with patch.object(signal, "signal"):
        http.serve(app, "127.0.0.1", 0, server=owned)
    assert calls == ["serve"] and owned.handlers_drained


def test_supplied_other_app_or_unrelated_handler_is_refused_before_serving(monkeypatch):
    from tensorfold.cuda import http
    from tensorfold.server.http import Server
    import pytest

    app = certified_app()
    for handler in (http.make_handler(certified_app()), type("Unrelated", (), {})):
        owned = Server(("127.0.0.1", 0), handler, max_connections=2)
        monkeypatch.setattr(owned, "serve_forever", lambda: pytest.fail("foreign handler served"))
        try:
            with patch.object(signal, "signal"), pytest.raises(ValueError, match="certified App handler"):
                http.serve(app, "127.0.0.1", 0, server=owned)
        finally:
            owned.server_close()  # A refused supplied owner stays with its caller.


def test_supplied_server_rechecks_cap_before_serving(monkeypatch):
    from tensorfold.cuda import http
    from tensorfold.server.http import Server
    import pytest

    app = certified_app()
    owned = Server(("127.0.0.1", 0), http.make_handler(app), max_connections=2)
    owned.max_connections = 3
    monkeypatch.setattr(owned, "serve_forever", lambda: pytest.fail("unpaid handlers served"))
    try:
        with patch.object(signal, "signal"), pytest.raises(ValueError, match="admitted payload envelope"):
            http.serve(app, "127.0.0.1", 0, server=owned)
    finally:
        owned.server_close()

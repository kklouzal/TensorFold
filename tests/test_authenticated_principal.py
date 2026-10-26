"""Trusted key domains reach normal HTTP handlers and whole logical admission."""
from contextlib import contextmanager
import http.client
import json
import threading

import pytest

from tensorfold.cuda import http as cuda_http
from tensorfold.server import auth_http, http as shared_http
from tensorfold.server.authentication import KeyStore, current_principal
from tensorfold.server.request_limits import RequestLimit
from tests.test_api_keys_http import request
from tests.test_cuda_thinking_controls import app_for


def key_file(tmp_path, text):
    path = tmp_path / "keys"
    path.write_text(text)
    path.chmod(0o600)
    return path


def test_key_identity_is_independent_of_labels_and_rotation_never_inherits_owner(tmp_path):
    path = key_file(tmp_path, "first: fixture-key-a\n")
    store = KeyStore(key_file=path)
    first_label, first = store.match_identity({"Authorization": "Bearer fixture-key-a"})
    assert first_label == "first" and first and "fixture-key-a" not in first
    path.write_text("renamed: fixture-key-a\n")
    store.request_reload()
    assert store.match_identity({"x-api-key": "fixture-key-a"}) == ("renamed", first)
    path.write_text("renamed: fixture-key-b\n")
    store.request_reload()
    second_label, second = store.match_identity({"Authorization": "Bearer fixture-key-b"})
    assert second_label == "renamed" and second != first
    assert store.match_identity({"Authorization": "Bearer fixture-key-a"}) == (None, None)
    duplicate = KeyStore(["fixture-key-b"], key_file=path)
    assert duplicate.match_identity({"x-api-key": "fixture-key-b"}) == ("cli-1", second)


@contextmanager
def serving(app, backend):
    handler = cuda_http.make_handler(app) if backend == "cuda" else shared_http.make_handler(app)
    server = shared_http.Server(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
    thread.start()
    try:
        yield server.server_port
    finally:
        server.shutdown()
        server.server_close()
        thread.join(3)
        assert not thread.is_alive()


@pytest.mark.parametrize("backend", ["mlx", "cuda"])
def test_real_handlers_reset_principal_on_connection_reuse_and_never_expose_it(tmp_path, monkeypatch, backend, capsys):
    from types import SimpleNamespace
    app = SimpleNamespace(served="fixture", served_name="fixture", model_ids=["fixture"], max_batch_size=1)
    app.auth = KeyStore(["fixture-key-a", "fixture-key-b"])
    app._request_limit = RequestLimit(4, per_principal=1)
    module = cuda_http if backend == "cuda" else shared_http
    original, observed = module.models_reply, []
    def models(*args, **kwargs):
        observed.append((current_principal(), app._request_limit._identity()))
        return original(*args, **kwargs)
    monkeypatch.setattr(module, "models_reply", models)
    expected = [app.auth.match_identity({"x-api-key": key})[1] for key in ("fixture-key-a", "fixture-key-b")]
    with serving(app, backend) as port:
        client = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            for key in ("fixture-key-a", "fixture-key-b"):
                client.request("GET", "/v1/models", headers={"Authorization": "Bearer " + key})
                response = client.getresponse()
                payload = response.read().decode()
                assert response.status == 200 and "key-sha256:" not in payload and key not in payload
            client.request("GET", "/v1/models")
            response = client.getresponse()
            assert response.status == 401
            response.read()
        finally:
            client.close()
    assert observed == [(identity, identity) for identity in expected]
    assert current_principal() is None
    logs = capsys.readouterr().out
    assert "key-sha256:" not in logs and "fixture-key-a" not in logs and "fixture-key-b" not in logs


def test_normal_chat_quota_is_bound_by_key_not_user_metadata_or_cache_hint(tmp_path):
    class Engine:
        eos, concurrent = (0,), True
        def __init__(self):
            self.entered, self.release = threading.Event(), threading.Event()
            self.first = True
            self.seen = []
        def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
            identity = current_principal()
            self.seen.append(identity)
            if self.first:
                self.first = False
                self.entered.set()
                assert self.release.wait(3)
            on_tokens([ord("x")])
            return {"rounds": 1}
    engine = Engine()
    app = app_for(tmp_path, engine, thinking=False)
    app.auth = KeyStore(["fixture-key-a", "fixture-key-b"])
    limit = app._request_limit = RequestLimit(4, per_principal=1)
    app.run = limit.wrap(app.run)
    body = json.dumps({"messages": [{"role": "user", "content": "hi"}], "max_tokens": 1,
                       "user": "client-selected", "metadata": {"principal": "forged"},
                       "prompt_cache_key": "forged"})
    answers, failures = [], []
    with serving(app, "cuda") as port:
        def first():
            try:
                answers.append(request(port, "/v1/chat/completions", {"Authorization": "Bearer fixture-key-a"},
                                       "POST", body)[0])
            except BaseException as error:
                failures.append(error)
        worker = threading.Thread(target=first)
        worker.start()
        try:
            assert engine.entered.wait(3)
            status, _, _ = request(port, "/v1/chat/completions", {"Authorization": "Bearer fixture-key-a"}, "POST", body)
            assert status == 503 and limit.used == 1
            status, _, _ = request(port, "/v1/chat/completions", {"Authorization": "Bearer fixture-key-b"}, "POST", body)
            assert status == 200 and limit.used == 1
        finally:
            engine.release.set()
            worker.join(3)
            assert not worker.is_alive()
    assert answers == [200] and failures == [] and limit.used == 0
    assert engine.seen == [app.auth.match_identity({"x-api-key": key})[1]
                           for key in ("fixture-key-a", "fixture-key-b")]
    assert current_principal() is None
    with pytest.raises(ValueError, match="authenticated request principal"):
        limit.prepare()


def test_auth_scope_resets_when_handler_operation_raises():
    from types import SimpleNamespace
    limit = RequestLimit(2, per_principal=1)
    identity = "trusted-digest-domain"
    handler = auth_http.Authenticated()
    handler.principal_id, handler.auth_app = identity, SimpleNamespace(_request_limit=limit)
    primary = RuntimeError("dispatch failed")
    def failed():
        assert current_principal() == identity and limit._identity() == identity
        raise primary
    with pytest.raises(RuntimeError) as raised:
        handler._principal_dispatch(failed)
    assert raised.value is primary and current_principal() is None
    with pytest.raises(ValueError, match="authenticated request principal"):
        limit.prepare()


def test_missing_auth_cannot_publish_a_per_principal_endpoint():
    from types import SimpleNamespace
    class Base:
        pass
    app = SimpleNamespace(_request_limit=RequestLimit(2, per_principal=1))
    with pytest.raises(ValueError, match="API-key authentication"):
        auth_http.handler(Base, app)


def test_standard_conversations_alias_is_auth_gated():
    assert auth_http.gated("/conversations")
    assert auth_http.gated("/conversations/conv_123/items")

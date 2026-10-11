"""Startup auth selection preserves the exact ordinary handler and enabled gate lifecycle."""
from contextlib import contextmanager
import http.client
import json
import threading
from types import SimpleNamespace

import pytest

from tensorfold.cuda.http import make_handler as cuda_handler
from tensorfold.server import auth_http, metrics
from tensorfold.server.authentication import KeyStore
from tensorfold.server.http import Server, make_handler as mlx_handler


def test_absent_and_disabled_stores_return_the_same_base_type_without_attribute_changes():
    class Base:
        marker = object()
    before = dict(vars(Base))
    for app in (SimpleNamespace(), SimpleNamespace(auth=None), SimpleNamespace(auth=KeyStore())):
        assert auth_http.handler(Base, app) is Base
        assert dict(vars(Base)) == before
        assert not issubclass(Base, auth_http.Authenticated)


@pytest.mark.parametrize('factory', [mlx_handler, cuda_handler])
@pytest.mark.parametrize('configured', [False, True], ids=['absent', 'disabled'])
def test_real_frontends_execute_base_requests_without_auth_state_or_key_lock(factory, configured, monkeypatch):
    store = KeyStore() if configured else None
    app = SimpleNamespace(served='controlled', served_name='controlled', model_ids=['controlled'], max_batch_size=1)
    if configured:
        app.auth = store
        def forbidden_labels():
            raise AssertionError('disabled authentication acquired the key-label scope')
        monkeypatch.setattr(store, 'metric_labels', forbidden_labels)
    bases = []
    original = auth_http.handler
    def selected(base, app):
        bases.append(base)
        return original(base, app)
    monkeypatch.setattr(auth_http, 'handler', selected)
    chosen = factory(app)
    assert chosen is bases[0]
    assert not issubclass(chosen, auth_http.Authenticated)
    states = []
    class Observed(chosen):
        def do_GET(self):
            states.append(('_auth_counted' in self.__dict__, 'key_label' in self.__dict__))
            return super().do_GET()
    server = Server(('127.0.0.1', 0), Observed)
    worker = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': 0.01})
    worker.start()
    client = http.client.HTTPConnection('127.0.0.1', server.server_port, timeout=3)
    try:
        for path in ('/v1/models', '/v1/models', '/metrics'):
            client.request('GET', path)
            reply = client.getresponse()
            payload = reply.read()
            assert reply.status == 200
            if path.endswith('models'):
                assert json.loads(payload)['data'][0]['id'] == 'controlled'
            else:
                assert b'requests_total{key=' not in payload
                assert b'http_replies_total{' not in payload
        assert states == [(False, False)] * 3
        counters = metrics.of(app)
        assert counters.http_requests == counters.http_replies == {}
        assert counters._http_labels is None
    finally:
        client.close()
        server.shutdown()
        server.server_close()
        worker.join(3)
    assert not worker.is_alive()


def test_enabled_file_store_never_switches_to_plain_handler_on_failed_reload(tmp_path):
    path = tmp_path / 'keys'
    path.write_text('controlled: controlled-key\n')
    path.chmod(0o600)
    store = KeyStore(key_file=path)
    class Base:
        pass
    chosen = auth_http.handler(Base, SimpleNamespace(auth=store))
    assert issubclass(chosen, auth_http.Authenticated)
    assert chosen.auth is store
    path.write_text('')
    store.request_reload()
    store.refresh()
    assert store.enabled
    assert store.match({'x-api-key': 'controlled-key'}) is None
    assert issubclass(auth_http.handler(Base, SimpleNamespace(auth=store)), auth_http.Authenticated)


def test_enabled_metric_scope_remains_required_during_fail_closed_configuration(tmp_path, monkeypatch):
    path = tmp_path / 'keys'
    path.write_text('controlled: controlled-key\n')
    path.chmod(0o600)
    store = KeyStore(key_file=path)
    app = SimpleNamespace(auth=store)
    metrics.http_request(app, 'controlled', 200)
    path.write_text('')
    store.request_reload()
    store.refresh()
    calls = []
    original = store.metric_labels
    @contextmanager
    def labels():
        calls.append(True)
        with original() as active:
            yield active
    monkeypatch.setattr(store, 'metric_labels', labels)
    body = metrics.render(app)
    assert calls == [True]
    assert 'key="controlled"' not in body
    assert 'http_replies_total{status="200"} 1' in body

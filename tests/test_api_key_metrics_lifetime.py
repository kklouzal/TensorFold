"""Key rotation bounds label state while every HTTP reply keeps its lifetime count."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from types import SimpleNamespace
import threading

import pytest

from tensorfold.server import authentication, metrics


@pytest.fixture
def configured(tmp_path, monkeypatch):
    path = tmp_path / "keys"
    path.write_text("controlled input")
    snapshot = [((authentication._digest("fixture-first"), "first"),)]
    monkeypatch.setattr(authentication.KeyStore, "_read", lambda self: (snapshot[0], ()))
    store = authentication.KeyStore(["fixture-static"], key_file=path)
    app = SimpleNamespace(auth=store)

    def rotate(label):
        snapshot[0] = ((authentication._digest("fixture-current"), label),)
        store.request_reload()
        store.refresh()

    return app, rotate


def test_rotating_labels_retire_series_but_preserve_lifetime_totals(configured):
    app, rotate = configured
    metrics.note(app, prompt=7, generation=3)
    metrics.http_request(app, "cli-1", 200)
    metrics.http_request(app, "unauthenticated", 401)
    metrics.http_request(app, "first", 200)
    for index in range(100):
        label = f"rotation-{index}"
        rotate(label)
        metrics.http_request(app, label, 200)
        assert metrics.of(app).http_requests == {
            ("cli-1", 200): 1, ("unauthenticated", 401): 1, (label, 200): 1,
        }
    counters = metrics.of(app)
    assert counters.http_replies == {200: 102, 401: 1}
    assert (counters.prompt, counters.generation) == (7, 3)
    body = metrics.render(app)
    assert 'requests_total{key="rotation-99",status="200"} 1' in body
    assert 'http_replies_total{status="200"} 102' in body
    assert 'key="first"' not in body


def test_retired_response_stays_only_in_lifetime_total_and_readd_starts_new_series(configured):
    app, rotate = configured
    metrics.http_request(app, "first", 200)
    rotate("second")
    metrics.http_request(app, "first", 200)
    assert metrics.of(app).http_requests == {}
    assert metrics.of(app).http_replies == {200: 2}
    rotate("first")
    metrics.http_request(app, "first", 200)
    assert metrics.of(app).http_requests == {("first", 200): 1}
    assert metrics.of(app).http_replies == {200: 3}


@pytest.mark.parametrize("stats_only", [False, True])
def test_scrape_retires_departed_series_without_a_new_reply(configured, stats_only):
    app, rotate = configured
    metrics.http_request(app, "first", 200)
    rotate("second")
    if stats_only:
        assert metrics.counters_now(app, stats_only=True).requests == 0
    else:
        body = metrics.render(app)
        assert 'key="first"' not in body
        assert 'http_replies_total{status="200"} 1' in body
    assert metrics.of(app).http_requests == {}


def test_invalid_key_configuration_retires_all_authenticated_series_and_stays_closed(configured, monkeypatch):
    app, _ = configured
    metrics.http_request(app, "cli-1", 200)
    metrics.http_request(app, "first", 200)

    def refuse():
        raise ValueError("controlled malformed replacement")

    monkeypatch.setattr(app.auth, "_read", refuse)
    app.auth.request_reload()
    app.auth.refresh()
    assert app.auth.match({"x-api-key": "fixture-static"}) is None
    metrics.http_request(app, "first", 200)
    metrics.http_request(app, "unauthenticated", 401)
    assert metrics.of(app).http_requests == {("unauthenticated", 401): 1}
    assert metrics.of(app).http_replies == {200: 3, 401: 1}


def test_key_lock_pins_configuration_through_counter_commit(configured, monkeypatch):
    app, rotate = configured
    entered = threading.Event()
    original = app.auth.metric_labels

    @contextmanager
    def pinned_labels():
        with original() as labels:
            entered.set()
            yield labels

    monkeypatch.setattr(app.auth, "metric_labels", pinned_labels)
    counters = metrics.of(app)
    # Hold the second owner lock: the reply must pin the first owner before waiting for it.
    with ThreadPoolExecutor(max_workers=2) as workers:
        with counters.lock:
            pending_reply = workers.submit(metrics.http_request, app, "first", 200)
            assert entered.wait(2)
            assert not pending_reply.done()
            pending_rotation = workers.submit(rotate, "second")
            assert not pending_rotation.done()
        pending_reply.result(timeout=2)
        pending_rotation.result(timeout=2)
    assert metrics.of(app).http_replies == {200: 1}
    assert 'key="first"' not in metrics.render(app)

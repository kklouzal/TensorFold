"""Prepared callers reserve/adopt the existing finite logical request owner."""

import gc
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
import weakref

import pytest

from tensorfold.server.errors import CapacityError
from tensorfold.server.request_limits import RequestLimit, require_finite_endpoint_limits


def test_prepared_owner_is_adopted_without_double_count_and_keeps_queued_work():
    limit = RequestLimit(1)
    prepared = limit.prepare()
    assert limit.used == 1
    with pytest.raises(CapacityError):
        limit.prepare()
    worker = object()
    with limit.request(prepared):
        assert limit.used == 1
        limit.borrow(worker)
        assert limit.used == 1
    assert limit.used == 1  # Caller retirement does not retire queued/running work.
    limit.release(worker)
    assert limit.used == 0
    prepared.close()
    with pytest.raises(CapacityError):
        with limit.request(prepared):
            pytest.fail("a consumed prepared owner was reused")


def test_standalone_prepare_is_independent_of_a_callback_current_owner():
    limit = RequestLimit(1)
    with limit.request():
        with pytest.raises(CapacityError):
            limit.prepare()
    assert limit.used == 0


def test_abandoned_reservation_does_not_retain_its_holder():
    limit = RequestLimit(1)
    prepared = limit.prepare()
    ref = weakref.ref(prepared)
    del prepared
    gc.collect()
    assert ref() is None and limit.used == 0


def test_interrupted_prepared_release_is_reclaimable_and_explicitly_retryable(monkeypatch):
    limit = RequestLimit(1)
    prepared = limit.prepare()
    original = limit.release
    monkeypatch.setattr(limit, "release", lambda token: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        prepared.close()
    assert limit.used == 0  # Native unlocked scope remains authoritative evidence.
    monkeypatch.setattr(limit, "release", original)
    prepared.close()
    assert limit.used == 0


def test_missing_finalizer_bookkeeping_cannot_pin_an_unreachable_preparation():
    limit = RequestLimit(1)
    prepared = limit.prepare()
    prepared._finalizer.detach()  # Model a lost callback: independent native lifetime is the oracle.
    del prepared
    gc.collect()
    assert limit.used == 0


@pytest.mark.parametrize("pending,connections", [(None, 2), (2, None), (None, None)])
def test_finite_endpoint_policy_rejects_missing_limits_before_socket_creation(monkeypatch, pending, connections):
    from tensorfold.server.http import Server

    sockets = []
    monkeypatch.setattr(ThreadingHTTPServer, "__init__", lambda *args, **kwargs: sockets.append("bound"))
    handler = SimpleNamespace(requires_finite_limits=True, pending_request_limit=pending)
    with pytest.raises(ValueError, match="explicit positive"):
        Server(("127.0.0.1", 0), handler, max_connections=connections)
    assert not sockets
    with pytest.raises(ValueError):
        require_finite_endpoint_limits(max_pending_requests=pending, max_http_connections=connections)


def test_structural_cli_requires_explicit_finite_limits_before_metadata_access(tmp_path):
    from tensorfold.serve_options import check

    family = SimpleNamespace(package=SimpleNamespace(), model_type="qwen4_exp")
    args = SimpleNamespace(serving_profile_root=tmp_path, max_pending_requests=None, max_http_connections=2)
    with pytest.raises(ValueError, match="max-pending-requests"):
        check(args, family, "cuda")

"""Authenticated logical-owner quotas reuse the actual request lifetime registry."""

import threading

import pytest

from tensorfold.server.errors import CapacityError
from tensorfold.server.request_limits import RequestLimit


def test_principal_quota_spans_sequential_jobs_and_caller_delivery():
    limit = RequestLimit(4, per_principal=1)
    with limit.principal("trusted-a"), limit.request():
        for _ in range(2):
            job = object()
            limit.borrow(job)
            limit.release(job)
            assert limit.used == 1
            with pytest.raises(CapacityError, match="principal request capacity"):
                limit.prepare()
        with limit.principal("trusted-b"):
            prepared = limit.prepare()
            assert limit.used == 2
            with limit.request(prepared):
                assert limit.used == 2
        assert limit.used == 1
    assert limit.used == 0


def test_prepared_principal_cannot_change_and_cancelled_caller_keeps_worker_charge():
    limit = RequestLimit(3, per_principal=1)
    prepared = limit.prepare(principal="trusted-a")
    with limit.principal("trusted-b"), pytest.raises(ValueError, match="another authenticated principal"):
        with limit.request(prepared):
            pytest.fail("foreign principal adopted preparation")
    assert limit.used == 1 and prepared._scope.locked()
    worker = object()
    with limit.request(prepared):  # The trusted prepared owner carries its identity across scopes.
        limit.borrow(worker)
        with limit.principal("trusted-b"), pytest.raises(ValueError, match="another authenticated principal"):
            with limit.inherited_request():
                pytest.fail("foreign scope inherited an existing logical request")
    with pytest.raises(CapacityError, match="principal request capacity"):
        limit.prepare(principal="trusted-a")
    other = limit.prepare(principal="trusted-b")
    assert limit.used == 2
    other.close()
    limit.release(worker)
    assert limit.used == 0
    final = limit.prepare(principal="trusted-a")
    final.close()


def test_missing_identity_is_refused_before_acceptance_and_scope_is_restored():
    limit = RequestLimit(2, per_principal=1)
    for accept in (limit.prepare, lambda: limit.borrow(object())):
        with pytest.raises(ValueError, match="authenticated request principal"):
            accept()
        assert limit.used == 0
    with pytest.raises(ValueError, match="authenticated request principal"):
        with limit.request():
            pytest.fail("unbound principal accepted")
    with pytest.raises(RuntimeError, match="caller failed"):
        with limit.principal("trusted-a"), limit.request():
            raise RuntimeError("caller failed")
    assert limit.used == 0
    with pytest.raises(ValueError, match="authenticated request principal"):
        limit.prepare()


def test_queue_adapter_reads_the_trusted_current_logical_owner_principal():
    limit = RequestLimit(2, per_principal=1)
    with limit.principal("trusted-a"):
        assert limit.current_principal() == "trusted-a"
        prepared = limit.prepare()
    with limit.request(prepared):
        assert limit.current_principal() == "trusted-a"  # Prepared handoff owns identity after HTTP scope ends.
        with limit.principal("trusted-b"), pytest.raises(ValueError, match="another authenticated principal"):
            limit.current_principal()
    with pytest.raises(ValueError, match="authenticated request principal"):
        limit.current_principal()


@pytest.mark.parametrize("value", [True, 0, -1, 1.5])
def test_principal_limit_is_explicit_positive_integer(value):
    with pytest.raises(ValueError, match="max_pending_requests_per_principal"):
        RequestLimit(2, per_principal=value)


def test_principal_cap_can_exceed_global_without_changing_global_refusal():
    limit = RequestLimit(1, per_principal=7)
    first = limit.prepare(principal="trusted-a")
    with pytest.raises(CapacityError, match="^request capacity reached"):
        limit.prepare(principal="trusted-b")
    first.close()


def test_parallel_principals_and_cancelled_callers_keep_exact_worker_ownership():
    limit = RequestLimit(2, per_principal=1)
    entered, callers_leave = threading.Barrier(3, timeout=3), threading.Event()
    workers, failures = [object(), object()], []
    def caller(index):
        try:
            with limit.principal(f"trusted-{index}"), limit.request():
                limit.borrow(workers[index])
                entered.wait()
                assert callers_leave.wait(3)
        except BaseException as error:
            failures.append(error)
    threads = [threading.Thread(target=caller, args=(index,)) for index in range(2)]
    for thread in threads:
        thread.start()
    try:
        entered.wait()
        with pytest.raises(CapacityError):
            limit.prepare(principal="trusted-0")
        callers_leave.set()
        for thread in threads:
            thread.join(3)
            assert not thread.is_alive()
        assert failures == [] and limit.used == 2
        limit.release(workers[0])
        assert limit.used == 1
        replacement = limit.prepare(principal="trusted-0")
        assert limit.used == 2
        replacement.close()
        limit.release(workers[1])
        assert limit.used == 0
    finally:
        callers_leave.set()
        for thread in threads:
            thread.join(3)
        for worker in workers:
            limit.release(worker)

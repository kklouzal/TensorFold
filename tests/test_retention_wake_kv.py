"""Real CPU Store controls for canonical KV wake semantics on both backends."""

from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
import threading
import time

import pytest

pytest.importorskip("torch")
from torch.distributed import DistStoreError, TCPStore

from tensorfold.cleanup import raise_failures
from retention_wake_queue_reference import RetentionWakeRelay as QueueRelay
from tensorfold.families.qwen4_exp.cuda.retention_wake import TRANSPORT_DOORBELL, TRANSPORT_LEASE, RetentionWakeRelay

pytestmark = pytest.mark.torch


@pytest.fixture(params=[True, False], ids=["libuv", "legacy"])
def backend(request):
    return request.param


def fence_all(relays):
    for relay in relays:
        relay.fence()


@contextmanager
def cohort(wake, backend, *, world=2, timeout=timedelta(seconds=2), namespace="activation-A"):
    server = TCPStore("127.0.0.1", 0, None, True, timeout, wait_for_workers=False, use_libuv=backend)
    baseline, relays, failures = server.num_keys(), [], []
    try:
        for rank in range(world):
            client = TCPStore("127.0.0.1", server.port, None, False, timeout, use_libuv=backend)
            registered = []
            relay = RetentionWakeRelay.create(SimpleNamespace(rank=rank, world=world, store=client), namespace,
                wake=wake, fatal=failures.append, register=registered.append)
            assert registered == [relay] and relay.timeout == timeout
            relays.append(relay)
        yield server, baseline, relays, failures
    finally:
        errors = []
        for relay in relays:
            try:
                relay.fence()
            except BaseException as error:
                errors.append(error)
        for relay in [*relays[1:], *relays[:1]]:
            if not relay.record()["closed"]:
                try:
                    relay.close(timeout=2)
                except BaseException as error:
                    errors.append(error)
        raise_failures(None, errors)


class Sender:
    def __init__(self, base, slot):
        self.base, self.slot, self.sets, self.pending = base, slot, 0, []

    def __getattr__(self, name):
        return getattr(self.base, name)

    def set(self, key, value):
        if key == self.slot:
            self.sets += 1
        self.base.set(key, value)


def test_global_coalescing_covers_every_published_rank_and_bounds_frames(backend):
    reached, observed = threading.Event(), []
    predicates = [False] * 8

    def wake():
        observed.append(tuple(predicates))
        reached.set()

    with cohort(wake, backend, world=8) as (server, baseline, relays, failures):
        for rank, follower in enumerate(relays[1:], 1):
            follower._sender = Sender(follower._sender, follower._slot)
            follower.start()
            predicates[rank] = True  # Actual caller publishes before notify.
            for _ in range(30):
                follower.notify()
            assert follower._sender.sets <= 1
        assert server.check([TRANSPORT_DOORBELL])
        assert server.num_keys() == baseline + 8 + 1 + 1  # claims, lease, one event slot
        relays[0].start()
        assert reached.wait(2) and observed == [tuple(predicates)]
        fence_all(relays)
        for relay in [*relays[1:], relays[0]]:
            relay.close(timeout=2)
        assert not failures and server.num_keys() == baseline and not server.check([TRANSPORT_DOORBELL])


def test_pre_delete_publications_coalesce_into_the_later_full_scan(backend):
    entered, release, reached = threading.Event(), threading.Event(), threading.Event()
    published, observed = ["first"], []

    def wake():
        observed.append(tuple(published))
        reached.set()

    with cohort(wake, backend) as (_server, _baseline, relays, failures):
        leader, follower = relays

        class DelayedDelete:
            def __init__(self, base):
                self.base = base

            def delete_key(self, key):
                entered.set()
                assert release.wait(2)
                return self.base.delete_key(key)

            def __getattr__(self, name):
                return getattr(self.base, name)

        leader._receiver = DelayedDelete(leader._receiver)
        follower._sender = Sender(follower._sender, follower._slot)
        leader.start()
        follower.start()
        follower.notify()
        try:
            assert entered.wait(2)
            published.append("second")
            follower.notify()
            assert follower._sender.sets == 1
        finally:
            release.set()
        assert reached.wait(2) and observed == [("first", "second")] and not failures


def test_post_delete_publication_survives_a_blocked_scheduler_wake(backend):
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def wake():
        calls.append(True)
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        else:
            second.set()

    with cohort(wake, backend) as (server, _baseline, relays, failures):
        leader, follower = relays
        leader.start()
        follower.start()
        follower.notify()
        try:
            assert entered.wait(2) and not server.check([TRANSPORT_DOORBELL])
            follower.notify()
            # SET returns after flush; another socket's CHECK cannot fence it.
            assert follower._sender.check([TRANSPORT_DOORBELL])
            assert server.check([TRANSPORT_DOORBELL])
        finally:
            release.set()
        assert second.wait(2) and calls == [True, True] and not failures


def test_each_same_sender_check_processes_its_previous_reply_free_set(backend):
    with cohort(lambda: None, backend) as (server, _baseline, relays, _failures):
        leader, follower = relays

        class DelayedSet(Sender):
            def set(self, key, value):
                if key == self.slot:
                    self.sets += 1
                    self.pending.append((key, value))
                else:
                    self.base.set(key, value)

            def flush(self):
                for key, value in self.pending:
                    self.base.set(key, value)
                self.pending.clear()

            def check(self, keys):
                self.flush()
                return self.base.check(keys)

            def get(self, key):
                self.flush()
                return self.base.get(key)

        sender = follower._sender = DelayedSet(follower._sender, follower._slot)
        follower.start()
        follower.notify()
        assert len(sender.pending) == 1 and not server.check([TRANSPORT_DOORBELL])
        follower.notify()
        assert sender.sets == 1 and not sender.pending and server.check([TRANSPORT_DOORBELL])
        leader.start()


def test_wait_timeout_has_no_business_wake_and_does_not_block_daemon(backend):
    calls = []
    with cohort(lambda: calls.append(True), backend, timeout=timedelta(milliseconds=20)) as (server, _baseline, relays, failures):
        for relay in relays:
            relay.start()
        time.sleep(.07)
        # An outstanding legacy WAIT registers a waiter; unrelated Store work
        # must still be serviced by the actual daemon, without polling wakes.
        server.set("independent-control", b"ready")
        assert server.get("independent-control") == b"ready"
        server.delete_key("independent-control")
        assert not calls and not failures and not relays[0]._worker.retired.is_set()


def test_missing_sender_fence_keeps_lease_and_prevents_stop(backend):
    with cohort(lambda: None, backend) as (server, baseline, relays, _failures):
        leader, follower = relays
        leader.start()
        follower.start()
        leader.fence()
        with pytest.raises(DistStoreError):
            leader.close(timeout=.02)
        assert not leader._stop_attempted and server.get(TRANSPORT_LEASE) == leader._nonce
        follower.fence()
        follower.close(timeout=2)
        leader.close(timeout=2)
        assert server.num_keys() == baseline


def test_failed_sender_reports_once_and_explicit_retirement_preserves_primary(backend):
    primary = OSError("doorbell SET refused")
    with cohort(lambda: None, backend) as (server, baseline, relays, failures):
        leader, follower = relays

        class Refused(Sender):
            def set(self, key, value):
                if key == self.slot:
                    raise primary
                self.base.set(key, value)

        follower._sender = Refused(follower._sender, follower._slot)
        leader.start()
        follower.start()
        with pytest.raises(OSError) as caught:
            follower.notify()
        assert caught.value is primary and failures == [primary]
        with pytest.raises(RuntimeError, match="failed"):
            follower.notify()
        fence_all(relays)
        with pytest.raises(OSError) as caught:
            follower.close(timeout=2)
        assert caught.value is primary and follower.record()["closed"]
        leader.close(timeout=2)
        assert server.num_keys() == baseline


def test_uncertain_receiver_drain_retains_owner_and_excludes_queue_and_kv(backend):
    entered, release = threading.Event(), threading.Event()

    def wake():
        entered.set()
        assert release.wait(2)

    with cohort(wake, backend) as (server, baseline, relays, _failures):
        leader, follower = relays
        leader.start()
        follower.start()
        follower.notify()
        try:
            assert entered.wait(2)
            fence_all(relays)
            with pytest.raises(TimeoutError):
                leader.close(timeout=.01)
            assert leader._receiver is not None and not leader.record()["closed"]
            for implementation in (RetentionWakeRelay, QueueRelay):
                refused = []
                with pytest.raises(ValueError, match="already owns a retention wake receiver"):
                    implementation.create(SimpleNamespace(rank=0, world=2, store=server), "activation-B",
                        wake=lambda: None, fatal=lambda _e: None, register=refused.append)
                assert refused[0].record()["closed"]
            assert server.get(TRANSPORT_LEASE) == leader._nonce
        finally:
            release.set()
        follower.close(timeout=2)
        leader.close(timeout=2)
        assert server.num_keys() == baseline


def test_fixed_slot_is_reused_after_all_rank_retirement(backend):
    server = TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=2), wait_for_workers=False, use_libuv=backend)
    baseline = server.num_keys()
    for turn in range(6):
        owners, reached = [], threading.Event()
        relays = [RetentionWakeRelay.create(SimpleNamespace(rank=rank, world=2, store=server), f"A-B-A/{turn}",
            wake=reached.set, fatal=lambda e: pytest.fail(type(e).__name__), register=owners.append) for rank in range(2)]
        try:
            for relay in relays:
                relay.start()
            relays[1].notify()
            assert reached.wait(2)
        finally:
            fence_all(relays)
            relays[1].close(timeout=2)
            relays[0].close(timeout=2)
        assert server.num_keys() == baseline and not server.check([TRANSPORT_DOORBELL, TRANSPORT_LEASE])


def test_unknown_sender_fence_keeps_lease_until_same_connection_retry(backend):
    primary = OSError("sender fence readback refused")
    with cohort(lambda: None, backend) as (server, baseline, relays, failures):
        leader, follower = relays
        sender = follower._sender

        class RefusedGet:
            def get(self, _key):
                raise primary

            def __getattr__(self, name):
                return getattr(sender, name)

        leader.start()
        follower.start()
        follower._sender = RefusedGet()
        leader.fence()
        with pytest.raises(OSError) as caught:
            follower.fence()
        assert caught.value is primary and failures == [primary]
        assert not follower.record()["sender_fenced"] and not server.check([follower._fence_key(1)])
        with pytest.raises(OSError):
            follower.close(timeout=2)
        with pytest.raises(DistStoreError):
            leader.close(timeout=.02)
        assert not leader._stop_attempted and server.get(TRANSPORT_LEASE) == leader._nonce
        follower._sender = sender  # Same actual retained socket, not a new clone.
        follower.fence()
        with pytest.raises(OSError) as caught:
            follower.close(timeout=2)
        assert caught.value is primary and follower.record()["closed"]
        leader.close(timeout=2)
        assert server.num_keys() == baseline


def test_stop_set_is_fenced_on_stable_claim_and_unknown_readback_keeps_owner(backend):
    primary = OSError("stop SET processing is uncertain")
    with cohort(lambda: None, backend) as (server, baseline, relays, _failures):
        leader, follower = relays
        control = leader._control

        class DelayedControl:
            def __init__(self):
                self.pending, self.refused = [], True

            def set(self, key, value):
                assert not self.pending  # Explicit retries cannot queue another uncertain SET.
                self.pending.append((key, value))

            def get(self, key):
                assert key != TRANSPORT_DOORBELL  # The receiver consumes it independently.
                if self.refused:
                    raise primary
                for item, value in self.pending:
                    control.set(item, value)
                self.pending.clear()
                return control.get(key)

            def __getattr__(self, name):
                return getattr(control, name)

        leader.start()
        follower.start()
        fence_all(relays)
        follower.close(timeout=2)
        delayed = leader._control = DelayedControl()
        with pytest.raises(OSError) as caught:
            leader.close(timeout=.02)
        assert caught.value is primary and not leader.record()["closed"]
        assert leader._receiver is not None and server.get(TRANSPORT_LEASE) == leader._nonce
        delayed.refused = False
        leader.close(timeout=2)  # Only this explicit retry flushes the retained control socket.
        assert not delayed.pending and leader.record()["closed"] and server.num_keys() == baseline


def test_receiver_delete_failure_preserves_primary_and_retires_only_after_quiescence(backend):
    fatal = threading.Event()
    with cohort(lambda: pytest.fail("unacknowledged doorbell woke scheduler"), backend) as (server, baseline, relays, failures):
        leader, follower = relays

        class MissingDelete:
            def __init__(self, base):
                self.base = base

            def delete_key(self, _key):
                return False

            def __getattr__(self, name):
                return getattr(self.base, name)

        original = leader._fatal
        leader._fatal = lambda error: (original(error), fatal.set())
        leader._receiver = MissingDelete(leader._receiver)
        leader.start()
        follower.start()
        follower.notify()
        assert fatal.wait(2) and leader._worker.retired.wait(2)
        assert len(failures) == 1 and isinstance(failures[0], RuntimeError)
        fence_all(relays)
        follower.close(timeout=2)
        with pytest.raises(RuntimeError) as caught:
            leader.close(timeout=2)
        assert caught.value is failures[0] and leader.record()["closed"]
        assert server.num_keys() == baseline and not server.check([TRANSPORT_DOORBELL])


def test_partial_receiver_acquisition_is_owned_before_clone_and_retires_keys(backend):
    server = TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=2), wait_for_workers=False, use_libuv=backend)
    baseline, owners = server.num_keys(), []
    primary = OSError("receiver clone failed")

    class Clones:
        timeout = server.timeout

        def __init__(self):
            self.calls = 0

        def clone(self):
            assert len(owners) == 1
            self.calls += 1
            if self.calls == 2:
                raise primary
            return server.clone()

    with pytest.raises(OSError) as caught:
        RetentionWakeRelay.create(SimpleNamespace(rank=0, world=2, store=Clones()), "partial",
            wake=lambda: None, fatal=lambda _e: None, register=owners.append)
    assert caught.value is primary and owners[0].record()["closed"]
    assert owners[0]._control is None and server.num_keys() == baseline


@pytest.mark.parametrize("blockage", ["input", "output", "unbounded-output"])
def test_real_eight_rank_scheduler_catalog_caller_uses_canonical_kv(backend, blockage, monkeypatch):
    import test_live_scheduler_cohort as live
    import test_retained_catalog_cohort as states

    assert live.RetentionWakeRelay is RetentionWakeRelay
    stores, relays = [], []
    create = RetentionWakeRelay.create

    def store(*args, **kwargs):
        result = TCPStore(*args, **kwargs, use_libuv=backend)
        stores.append(result)
        return result

    def relay(*args, **kwargs):
        result = create(*args, **kwargs)
        relays.append(result)
        return result

    monkeypatch.setattr(states, "TCPStore", store)
    monkeypatch.setattr(RetentionWakeRelay, "create", relay)
    # Exercise the maintained independent State/RAM/TP/fairness/resume oracle,
    # including its exact output, ticket, backing and all-rank key retirement
    # assertions. Only the model step is controlled; no CUDA claim is made.
    live.test_eligible_new_request_runs_while_old_same_request_is_in_ram_then_exact_ticket_resumes(blockage)
    assert len(stores) >= 9 and all(item.libuvBackend is backend for item in stores)
    assert sorted(item.rank for item in relays) == list(range(8))
    assert all(item.record()["closed"] for item in relays)


def test_single_rank_uses_no_store_or_extra_receiver():
    calls, owners = [], []
    relay = RetentionWakeRelay.create(None, "one", wake=lambda: calls.append(True),
        fatal=lambda e: pytest.fail(type(e).__name__), register=owners.append)
    relay.start()
    relay.notify()
    relay.fence()
    relay.close(timeout=0)
    assert calls == [True] and relay.record()["cpu_receiver_workers"] == 0 and relay.record()["closed"]

"""Real CPU TCPStore progress/coalescing/retirement, without GPU qualification."""

from contextlib import contextmanager
from datetime import timedelta
from types import SimpleNamespace
import threading
import time

import pytest

pytest.importorskip("torch")  # Guard the optional provider before backend imports/collection.
from torch.distributed import DistStoreError, TCPStore

from tensorfold.cleanup import raise_failures
from tensorfold.families.qwen4_exp.cuda.retention_wake import (
    MAX_WORD, TRANSPORT_LEASE, TRANSPORT_QUEUE, RetentionWakeRelay,
)

pytestmark = pytest.mark.torch


def fence_all(relays):
    # These controls have already quiesced every local data producer.
    for relay in relays:
        relay.fence()


@contextmanager
def cohort(wake, *, world=2, timeout=timedelta(seconds=2)):
    server = TCPStore("127.0.0.1", 0, None, True, timeout, wait_for_workers=False)
    baseline = server.num_keys()
    relays, failures = [], []
    try:
        for rank in range(world):
            client = TCPStore("127.0.0.1", server.port, None, False, timeout)
            comm = SimpleNamespace(rank=rank, world=world, store=client)
            registered = []
            relay = RetentionWakeRelay.create(comm, "activation-A", wake=wake,
                                               fatal=lambda error: failures.append(error), register=registered.append)
            assert registered == [relay]
            assert relay.timeout == timeout
            relays.append(relay)
        yield server, baseline, relays, failures
    finally:
        errors = []
        # Data producers in these controls have quiesced before leaving bodies.
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


def test_notifications_coalesce_until_ack_and_retire_all_owned_keys():
    reached = threading.Event()
    calls = []

    def wake():
        calls.append(True)
        reached.set()

    with cohort(wake) as (server, baseline, relays, failures):
        leader, follower = relays
        follower.start()
        for _ in range(12):
            follower.notify()
        assert server.queue_len(leader._queue) == 1
        assert follower.record()["posted_generations"] == 1
        assert leader.record()["cpu_receiver_workers"] == 1
        assert follower.record()["cpu_receiver_workers"] == 0
        leader.start()
        assert reached.wait(2)
        fence_all(relays)
        follower.close(timeout=2)
        leader.close(timeout=2)
        assert calls == [True] and failures == []
        assert server.num_keys() == baseline and server.queue_len(leader._queue) == 0
        assert leader.record()["cpu_receiver_workers"] == 0


def test_eight_rank_queue_and_ack_metadata_stay_bounded_by_actual_followers():
    reached, calls = threading.Event(), []

    def wake():
        calls.append(True)
        if len(calls) == 7:
            reached.set()

    with cohort(wake, world=8) as (server, baseline, relays, failures):
        for follower in relays[1:]:
            follower.start()
            follower.notify()
            follower.notify()
        assert server.queue_len(relays[0]._queue) == 7
        assert all(follower.record()["posted_generations"] == 1 for follower in relays[1:])
        relays[0].start()
        assert reached.wait(2) and server.num_keys() == baseline + 8 + 7 + 1
        fence_all(relays)
        for follower in relays[1:]:
            follower.close(timeout=2)
        relays[0].close(timeout=2)
        assert len(calls) == 7 and not failures and server.num_keys() == baseline


def test_ack_precedes_wake_so_a_new_generation_can_queue_while_actor_wake_waits():
    entered, release, second = threading.Event(), threading.Event(), threading.Event()
    calls = []

    def wake():
        calls.append(True)
        if len(calls) == 1:
            entered.set()
            assert release.wait(2)
        else:
            second.set()

    with cohort(wake) as (server, baseline, relays, failures):
        leader, follower = relays
        leader.start()
        follower.start()
        follower.notify()
        try:
            assert entered.wait(2)
            assert server.get(leader._ack(1, 1)) == b"1"
            follower.notify()
            follower.notify()  # generation2 has not been ACKed; it stays one packet.
            assert server.queue_len(leader._queue) == 1
            assert follower.record()["posted_generations"] == 2
        finally:
            release.set()
        assert second.wait(2)
        fence_all(relays)
        follower.close(timeout=2)
        leader.close(timeout=2)
        assert calls == [True, True] and failures == [] and server.num_keys() == baseline


def test_pre_ack_publications_are_covered_by_the_one_later_wake():
    entered, release, reached = threading.Event(), threading.Event(), threading.Event()
    observed, published = [], ["first"]

    def wake():
        observed.append(tuple(published))
        reached.set()

    with cohort(wake) as (_server, _baseline, relays, failures):
        leader, follower = relays

        class DelayedAck:
            def __init__(self, base):
                self.base = base

            def set(self, key, value):
                entered.set()
                assert release.wait(2)
                self.base.set(key, value)

            def __getattr__(self, name):
                return getattr(self.base, name)

        leader._receiver = DelayedAck(leader._receiver)
        leader.start()
        follower.start()
        follower.notify()
        try:
            assert entered.wait(2)
            published.append("second")
            follower.notify()
            assert follower.record()["posted_generations"] == 1
        finally:
            release.set()
        assert reached.wait(2) and observed == [("first", "second")] and failures == []


def test_normal_store_timeouts_do_not_trigger_business_wakes():
    calls = []
    with cohort(lambda: calls.append(True), timeout=timedelta(milliseconds=20)) as (_server, _baseline, relays, failures):
        leader, follower = relays
        leader.start()
        follower.start()
        # Only the transport wait expires; there is no work or wake to poll.
        time.sleep(.07)
        assert calls == [] and failures == [] and not leader._worker.retired.is_set()


@pytest.mark.parametrize("payload", ["1:0", "1:2", "2:1", "1:9223372036854775808", "01:1", "0:0", "x" * 65])
def test_invalid_packets_fail_once_and_retire_actual_receiver(payload):
    fatal = threading.Event()
    with cohort(lambda: pytest.fail("invalid packet woke business work")) as (server, baseline, relays, failures):
        leader, follower = relays
        original = leader._fatal
        leader._fatal = lambda error: (original(error), fatal.set())
        leader.start()
        follower.start()
        server.queue_push(leader._queue, payload)
        assert fatal.wait(2) and leader._worker.retired.wait(2)
        assert len(failures) == 1 and isinstance(failures[0], ValueError)
        fence_all(relays)
        with pytest.raises(ValueError) as caught:
            leader.close(timeout=2)
        assert caught.value is failures[0] and leader.record()["closed"]
        follower.close(timeout=2)
        assert server.num_keys() == baseline


def test_failed_receiver_discards_only_bounded_owned_notifications_after_quiescence():
    with cohort(lambda: pytest.fail("failed receiver woke business work")) as (server, baseline, relays, failures):
        leader, follower = relays
        follower.start()
        server.queue_push(leader._queue, "8:1")
        assert server.queue_len(leader._queue) == 1  # Order the injected failure before another socket's send.
        follower.notify()
        leader.start()
        assert leader._worker.retired.wait(2)
        assert server.queue_len(leader._queue) == 1
        fence_all(relays)
        with pytest.raises(ValueError):
            leader.close(timeout=2)
        follower.close(timeout=2)
        assert len(failures) == 1 and server.queue_len(leader._queue) == 0 and server.num_keys() == baseline


def test_send_failure_preserves_primary_and_reports_only_once():
    primary = OSError("queue transport refused")
    with cohort(lambda: None) as (server, baseline, relays, failures):
        leader, follower = relays

        class Refused:
            def __init__(self, base):
                self.base = base

            def queue_push(self, *_args):
                raise primary

            def __getattr__(self, name):
                return getattr(self.base, name)

        follower._sender = Refused(follower._sender)
        leader.start()
        follower.start()
        with pytest.raises(OSError) as caught:
            follower.notify()
        assert caught.value is primary and failures == [primary]
        assert follower.record()["posted_generations"] == 0
        with pytest.raises(RuntimeError, match="relay is failed"):
            follower.notify()
        fence_all(relays)
        with pytest.raises(OSError) as caught:
            follower.close(timeout=2)
        assert caught.value is primary and follower.record()["closed"]
        leader.close(timeout=2)
        assert server.num_keys() == baseline


def test_fatal_callback_failure_keeps_native_primary_without_foreign_formatting():
    class Opaque(BaseException):
        def __str__(self):
            raise AssertionError("foreign formatter")

    primary, secondary = Opaque(), RuntimeError("fatal containment also failed")
    with cohort(lambda: None) as (_server, _baseline, relays, _failures):
        leader, follower = relays

        class Refused:
            def __init__(self, base):
                self.base = base

            def queue_push(self, *_args):
                raise primary

            def __getattr__(self, name):
                return getattr(self.base, name)

        def failed_fatal(error):
            assert error is primary
            raise secondary

        follower._sender = Refused(follower._sender)
        follower._fatal = failed_fatal
        leader.start()
        follower.start()
        with pytest.raises(Opaque) as caught:
            follower.notify()
        assert caught.value is primary and primary.__cause__ is secondary
        fence_all(relays)
        with pytest.raises(Opaque) as caught:
            follower.close(timeout=2)
        assert caught.value is primary and follower.record()["closed"]


def test_uncertain_receiver_drain_keeps_owned_clones_until_explicit_retry():
    entered, release = threading.Event(), threading.Event()

    def wake():
        entered.set()
        assert release.wait(2)

    with cohort(wake) as (server, baseline, relays, failures):
        leader, follower = relays
        leader.start()
        follower.start()
        follower.notify()
        try:
            assert entered.wait(2)
            fence_all(relays)
            with pytest.raises(TimeoutError):
                leader.close(timeout=.01)
            assert not leader.record()["closed"] and leader._receiver is not None and leader._control is not None
            assert not leader._worker.retired.is_set()
            assert server.get(TRANSPORT_LEASE) == leader._nonce
        finally:
            release.set()
        follower.close(timeout=2)
        leader.close(timeout=2)
        assert failures == [] and server.num_keys() == baseline


def test_one_native_queue_name_is_reused_across_fully_retired_a_b_a_activations():
    server = TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=2), wait_for_workers=False)
    baseline, queues = server.num_keys(), set()
    comms = [SimpleNamespace(rank=rank, world=2, store=TCPStore(
        "127.0.0.1", server.port, None, False, timedelta(seconds=2))) for rank in range(2)]
    for turn in range(12):
        reached, owners, failures = threading.Event(), [], []
        relays = [RetentionWakeRelay.create(comm, f"model-{'A' if turn % 2 == 0 else 'B'}/activation-{turn}",
                    wake=reached.set, fatal=failures.append, register=owners.append) for comm in comms]
        try:
            assert owners == relays
            for relay in relays:
                relay.start()
                queues.add(relay._queue)
            assert server.get(TRANSPORT_LEASE) == relays[0]._nonce
            relays[1].notify()
            assert reached.wait(2) and not failures
        finally:
            fence_all(relays)
            relays[1].close(timeout=2)
            relays[0].close(timeout=2)
        assert server.num_keys() == baseline and server.queue_len(TRANSPORT_QUEUE) == 0
    # The pinned native queues_ map retains empty entries. These real repeated
    # activations create exactly ONE key in it, rather than one key per UUID.
    assert queues == {TRANSPORT_QUEUE}


def test_other_activation_cannot_take_transport_while_receiver_retirement_is_uncertain():
    entered, release = threading.Event(), threading.Event()

    def wake():
        entered.set()
        assert release.wait(2)

    with cohort(wake) as (server, baseline, relays, _failures):
        leader, follower = relays
        leader.start()
        follower.start()
        follower.notify()
        try:
            assert entered.wait(2)
            fence_all(relays)
            with pytest.raises(TimeoutError):
                leader.close(timeout=.01)
            refused = []
            with pytest.raises(ValueError, match="already owns a retention wake receiver"):
                RetentionWakeRelay.create(SimpleNamespace(rank=0, world=2, store=server), "activation-B",
                    wake=lambda: None, fatal=lambda _error: None, register=refused.append)
            assert refused[0].record()["closed"]
            assert server.get(TRANSPORT_LEASE) == leader._nonce
            assert server.queue_len(TRANSPORT_QUEUE) == 1  # The old issued stop remains untouched.
        finally:
            release.set()
        follower.close(timeout=2)
        leader.close(timeout=2)
        assert server.num_keys() == baseline


def test_late_sender_bytes_are_read_back_before_reusable_queue_can_retire():
    reached = threading.Event()
    with cohort(reached.set) as (server, baseline, relays, failures):
        leader, follower = relays

        class DelayedSender:
            def __init__(self, base):
                self.base, self.pending = base, []

            def queue_push(self, key, value):
                # A successful native queue_push only flushes socket bytes.
                # Model their delayed server processing until this connection
                # performs a reply-bearing operation, preserving stream order.
                self.pending.append((key, value))

            def get(self, key):
                for queue, value in self.pending:
                    self.base.queue_push(queue, value)
                self.pending.clear()
                return self.base.get(key)

            def __getattr__(self, name):
                return getattr(self.base, name)

        follower._sender = DelayedSender(follower._sender)
        leader.start()
        follower.start()
        follower.notify()
        assert follower.record()["posted_generations"] == 1
        assert server.queue_len(TRANSPORT_QUEUE) == 0 and not reached.is_set()
        with pytest.raises(RuntimeError, match="requires its sender fence"):
            leader.close(timeout=.02)
        leader.fence()
        with pytest.raises(DistStoreError):
            leader.close(timeout=.02)  # A main/control connection cannot fence the sender.
        assert not leader._stop_attempted and not leader.record()["closed"]
        assert server.get(TRANSPORT_LEASE) == leader._nonce
        follower.fence()  # Delivers the delayed packet, then actually reads it back.
        assert reached.wait(2) and follower.record()["sender_fenced"]
        with pytest.raises(RuntimeError, match="not open"):
            follower.notify()
        follower.close(timeout=2)
        leader.close(timeout=2)
        assert not failures and server.num_keys() == baseline and server.queue_len(TRANSPORT_QUEUE) == 0


def test_unknown_sender_fence_keeps_lease_and_connections_until_explicit_readback_retry():
    primary, reached = OSError("sender readback refused"), threading.Event()
    with cohort(reached.set) as (server, baseline, relays, failures):
        leader, follower = relays
        sender = follower._sender

        class RefusedReadback:
            def get(self, _key):
                raise primary

            def __getattr__(self, name):
                return getattr(sender, name)

        leader.start()
        follower.start()
        follower.notify()
        assert reached.wait(2)
        follower._sender = RefusedReadback()
        leader.fence()
        with pytest.raises(OSError) as caught:
            follower.fence()
        assert caught.value is primary and failures == [primary]
        assert not follower.record()["sender_fenced"] and not server.check([follower._fence_key(1)])
        with pytest.raises(OSError) as caught:
            follower.close(timeout=2)
        assert caught.value is primary and not follower.record()["closed"] and follower._sender is not None
        with pytest.raises(DistStoreError):
            leader.close(timeout=.02)
        assert not leader._stop_attempted and server.get(TRANSPORT_LEASE) == leader._nonce
        follower._sender = sender
        follower.fence()
        with pytest.raises(OSError) as caught:
            follower.close(timeout=2)
        assert caught.value is primary and follower.record()["closed"]
        leader.close(timeout=2)
        assert server.num_keys() == baseline and server.queue_len(TRANSPORT_QUEUE) == 0


@pytest.mark.parametrize("fail_readback", [False, True])
def test_terminal_receiver_ack_bytes_are_read_back_before_metadata_retirement(fail_readback):
    primary = OSError("scheduler wake failed after ACK publication")
    readback_error = OSError("receiver readback refused")

    def refused_wake():
        raise primary

    with cohort(refused_wake) as (server, baseline, relays, failures):
        leader, follower = relays

        class DelayedAck:
            def __init__(self, base):
                self.base, self.pending = base, []
                self.refused = fail_readback

            def set(self, key, value):
                self.pending.append((key, value))

            def get(self, key):
                if self.refused:
                    raise readback_error
                for ack, value in self.pending:
                    self.base.set(ack, value)
                self.pending.clear()
                return self.base.get(key)

            def __getattr__(self, name):
                return getattr(self.base, name)

        receiver = leader._receiver = DelayedAck(leader._receiver)
        leader.start()
        follower.start()
        follower.notify()
        assert leader._worker.retired.wait(2) and failures == [primary]
        assert len(receiver.pending) == 1 and not server.check([leader._ack(1, 1)])
        fence_all(relays)
        follower.close(timeout=2)
        if fail_readback:
            with pytest.raises(OSError) as caught:
                leader.close(timeout=2)
            assert caught.value is primary and primary.__cause__ is readback_error
            assert not leader.record()["closed"] and leader._receiver is receiver
            assert len(receiver.pending) == 1 and server.get(TRANSPORT_LEASE) == leader._nonce
            receiver.refused = False
        with pytest.raises(OSError) as caught:
            leader.close(timeout=2)
        assert caught.value is primary and leader.record()["closed"] and not receiver.pending
        assert server.num_keys() == baseline and not server.check([leader._ack(1, 1)])


def test_constructor_is_registered_before_clone_and_partial_acquisition_retires_keys():
    server = TCPStore("127.0.0.1", 0, None, True, timedelta(seconds=2), wait_for_workers=False)
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
                                   wake=lambda: None, fatal=lambda _error: None, register=owners.append)
    assert caught.value is primary and owners[0].record()["closed"]
    assert owners[0]._control is None and server.num_keys() == baseline


def test_registration_failure_acquires_no_store_or_thread():
    primary, owners = MemoryError("registration failed"), []

    class NoClone:
        def clone(self):
            pytest.fail("registration failure acquired a connection")

    def refused(owner):
        owners.append(owner)
        raise primary

    with pytest.raises(MemoryError) as caught:
        RetentionWakeRelay.create(SimpleNamespace(rank=0, world=2, store=NoClone()), "registration",
                                   wake=lambda: None, fatal=lambda _error: None, register=refused)
    assert caught.value is primary and owners[0].record()["closed"] and owners[0]._worker is None


def test_namespace_rank_claim_prevents_two_consumers_without_deleting_first_owner():
    with cohort(lambda: None) as (server, baseline, relays, _failures):
        leader = relays[0]
        owners = []
        comm = SimpleNamespace(rank=0, world=2, store=server)
        with pytest.raises(ValueError, match="already owned"):
            RetentionWakeRelay.create(comm, leader.namespace, wake=lambda: None,
                                       fatal=lambda _error: None, register=owners.append)
        assert owners[0].record()["closed"] and server.get(leader._claim) == leader._nonce
        assert server.num_keys() == baseline + 2 + 1


def test_generation_exhaustion_and_launch_failure_are_explicit(monkeypatch):
    with cohort(lambda: None) as (_server, _baseline, relays, failures):
        leader, follower = relays
        primary = KeyboardInterrupt("before native launch")
        monkeypatch.setattr(leader._thread, "start", lambda: (_ for _ in ()).throw(primary))
        with pytest.raises(KeyboardInterrupt) as caught:
            leader.start()
        assert caught.value is primary and leader._worker.retired.is_set()
        assert failures == [primary]
        follower._generation = MAX_WORD
        follower._control.set(follower._ack(1, MAX_WORD), b"1")
        assert follower._control.get(follower._ack(1, MAX_WORD)) == b"1"
        follower.start()
        with pytest.raises(OverflowError):
            follower.notify()
        # This synthetic terminal frontier has no live leader to own its ACK.
        follower._control.delete_key(follower._ack(1, MAX_WORD))
        fence_all(relays)
        with pytest.raises(KeyboardInterrupt):
            leader.close(timeout=2)
        assert leader.record()["closed"]
        with pytest.raises(OverflowError):
            follower.close(timeout=2)


def test_single_rank_has_no_store_or_receiver_and_closes_idempotently():
    calls, owners = [], []
    relay = RetentionWakeRelay.create(None, "single", wake=lambda: calls.append(True),
                                       fatal=lambda error: pytest.fail(type(error).__name__), register=owners.append)
    relay.start()
    relay.notify()
    assert calls == [True] and relay.record()["cpu_receiver_workers"] == 0
    relay.fence()
    relay.close(timeout=0)
    relay.close(timeout=0)
    assert relay.record()["closed"]

"""Actual bounded FIFO scheduler/ownership protocol with a controlled numerical boundary."""

import sys
import threading
from datetime import timedelta
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from contextlib import nullcontext
from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda.scheduler import Scheduler
from tensorfold.server.cancellation import RequestCancelled
from tensorfold.server.errors import CapacityError
from test_flashnext_tp_scheduler import cohort as _cohort

cohort = _cohort


class Decoder:
    def __init__(self, *, blocked=(), pause=None):
        self.arrived = lambda: False
        self.streams, self.filling = {}, []
        self.condition = threading.Condition()
        self.blocked = set(blocked)
        self.prepared, self.admissions = [], []
        self.next_id, self.pause = 0, pause
        self.in_round, self.release_round = threading.Event(), threading.Event()
        self.retirement_ready = threading.Event()
        self.retirement_ready.set()
        self.retire_called = threading.Event()
        self.retire_calls = 0
        self._wake = None

    def live(self):
        return len(self.streams)

    def validate_request(self, stream):
        if not stream.prompt or len(stream.prompt) > 4:
            raise ValueError("controlled infeasible context")

    def prepare_request(self, stream, *, wake):
        with self.condition:
            if self._wake is not None:
                assert self._wake is wake
            self._wake = wake
            self.prepared.append(stream.prompt[0])
            self.condition.notify_all()
            return stream.prompt[0] not in self.blocked

    def allow(self, token):
        with self.condition:
            self.blocked.discard(token)
            wake = self._wake
        wake()

    def admit(self, stream):
        with self.condition:
            self.admissions.append((stream.prompt[0], stream.admission_ticket,
                                    tuple(item.prompt[0] for item in self.streams.values())))
            stream.sid, self.next_id = self.next_id, self.next_id + 1
            stream.st = NS(pos=len(stream.prompt))
            stream.cached = len(stream.prompt) - 1 if stream.prompt[0] == 3 else 0
            self.streams[stream.sid] = stream
            self.condition.notify_all()

    def round(self):
        if self.pause is not None and any(stream.prompt[0] == self.pause for stream in self.streams.values()):
            self.in_round.set()
            assert self.release_round.wait(3)
        for stream in tuple(self.streams.values()):
            if not stream.done:
                stream.st.pos += 1
                stream.take([stream.prompt[0] + 1])
        return [stream for stream in self.streams.values() if stream.done]

    def retire_request(self, stream):
        self.retire_calls += 1
        self.retire_called.set()
        return self.retirement_ready.is_set()

    def finish(self, streams):
        for stream in streams:
            self.streams.pop(stream.sid, None)

    def drop(self):
        result = list(self.streams.values())
        self.streams.clear()
        return result


def fifo(decoder, *, streams=2, byte_limit=4096, principal_bytes=None, timeout=None):
    return Scheduler(decoder, max_streams=streams, max_pending_requests=8,
        admission_policy="fifo", max_waiting_input_bytes=byte_limit,
        max_waiting_input_bytes_per_principal=principal_bytes, queue_timeout_s=timeout).start()


def client(scheduler, token, count=1, *, prompt=None, principal=None, **kwargs):
    done, output, errors = threading.Event(), [], []
    def submit():
        try:
            with scheduler._request_limit.principal(principal) if principal is not None else nullcontext():
                scheduler.submit([token] if prompt is None else prompt, count, None, False,
                                 lambda values: output.extend(values), **kwargs)
        except BaseException as error:
            errors.append(error)
        finally:
            done.set()
    thread = threading.Thread(target=submit)
    thread.start()
    return NS(thread=thread, done=done, output=output, errors=errors)


def joined(*clients):
    for item in clients:
        assert item.done.wait(3)
        item.thread.join(3)
        assert not item.thread.is_alive()


def wait_input(decoder, token):
    with decoder.condition:
        assert decoder.condition.wait_for(lambda: token in decoder.prepared, 3)


def test_unready_old_ticket_does_not_block_ready_work_with_spare_gpu_capacity():
    decoder = Decoder(blocked=(2,), pause=99)
    scheduler = fifo(decoder)
    active = client(scheduler, 99, 3)
    assert decoder.in_round.wait(3)
    old = client(scheduler, 2)
    # Queue publication is authoritative even before the next model boundary.
    with scheduler.waiting.not_empty:
        assert scheduler.waiting.not_empty.wait_for(lambda: scheduler.waiting.revision >= 2, 3)
    newer = client(scheduler, 3, background=True)
    with scheduler.waiting.not_empty:
        assert scheduler.waiting.not_empty.wait_for(lambda: scheduler.waiting.revision >= 3, 3)
    decoder.release_round.set()
    try:
        joined(newer, active)
        assert not old.done.is_set()
        record = scheduler.queue_record()
        row, = record["waiting"]
        assert row["ticket"] == 1 and row["blocked_reason"] == "media"
        admission = next(item for item in decoder.admissions if item[0] == 3)
        assert 99 in admission[2]  # Ready work started while the earlier GPU sequence remained admitted.
        decoder.allow(2)
        joined(old)
    finally:
        decoder.release_round.set()
        scheduler.close()
    assert active.errors == old.errors == newer.errors == []
    assert [(token, ticket) for token, ticket, _active in decoder.admissions] == [(99, 0), (3, 2), (2, 1)]


def test_background_and_warm_cost_never_jump_an_eligible_fifo_head():
    decoder = Decoder(pause=99)
    scheduler = fifo(decoder, streams=1)
    active = client(scheduler, 99)
    assert decoder.in_round.wait(3)
    older = client(scheduler, 2, background=True, prompt=[2] * 4)
    with scheduler.waiting.not_empty:
        assert scheduler.waiting.not_empty.wait_for(lambda: scheduler.waiting.revision >= 2, 3)
    newer = client(scheduler, 3, prompt=[3] * 3)  # Warm two rows; only one remaining, still a later ticket.
    with scheduler.waiting.not_empty:
        assert scheduler.waiting.not_empty.wait_for(lambda: scheduler.waiting.revision >= 3, 3)
    decoder.release_round.set()
    try:
        joined(active, older, newer)
    finally:
        scheduler.close()
    assert [token for token, _ticket, _active in decoder.admissions] == [99, 2, 3]
    assert active.errors == older.errors == newer.errors == [] and scheduler.yields == 0


def test_waiting_byte_refusal_retires_only_rejected_owner_and_endpoint_stays_healthy():
    decoder = Decoder(blocked=(2,))
    one = sys.getsizeof(list((2,))) + sys.getsizeof(2)
    scheduler = fifo(decoder, byte_limit=one)
    old = client(scheduler, 2)
    wait_input(decoder, 2)
    rejected = client(scheduler, 3)
    try:
        joined(rejected)
        assert len(rejected.errors) == 1 and isinstance(rejected.errors[0], CapacityError)
        assert scheduler.pending_requests == 1 and not scheduler._closing
        decoder.allow(2)
        joined(old)
        accepted = client(scheduler, 4)
        joined(accepted)
        assert accepted.errors == [] and accepted.output == [5]
    finally:
        decoder.allow(2)
        scheduler.close()
    assert old.errors == [] and scheduler.pending_requests == 0


def test_waiting_principal_bytes_use_actual_auth_scope_without_consuming_other_principal_credit():
    decoder = Decoder(blocked=(2, 4))
    one = sys.getsizeof(list((2,))) + sys.getsizeof(2)
    scheduler = fifo(decoder, byte_limit=one * 3, principal_bytes=one)
    first = client(scheduler, 2, principal="trusted-a")
    wait_input(decoder, 2)
    other = client(scheduler, 4, principal="trusted-b")
    wait_input(decoder, 4)
    refused = client(scheduler, 3, principal="trusted-a")
    unbound = client(scheduler, 6)
    try:
        joined(refused, unbound)
        assert isinstance(refused.errors[0], CapacityError)
        assert "principal waiting-input byte" in str(refused.errors[0])
        assert isinstance(unbound.errors[0], ValueError) and "authenticated request principal" in str(unbound.errors[0])
        assert scheduler.pending_requests == 2 and not scheduler._closing
        decoder.allow(2)
        joined(first)
        replacement = client(scheduler, 5, principal="trusted-a")
        joined(replacement)
        assert replacement.errors == [] and replacement.output == [6]
        decoder.allow(4)
        joined(other)
    finally:
        decoder.allow(2)
        decoder.allow(4)
        scheduler.close()
    assert first.errors == other.errors == [] and scheduler.pending_requests == 0


def test_queue_deadline_retires_without_gpu_and_does_not_poll_expired_cpu_owner(monkeypatch):
    import tensorfold.cuda.scheduler as source

    clock = [100.]
    monkeypatch.setattr(source.time, "monotonic", lambda: clock[0])
    decoder = Decoder(blocked=(2,))
    decoder.retirement_ready.clear()
    scheduler = fifo(decoder, timeout=10.)
    parked = threading.Event()
    actual_wait = scheduler.waiting.get_control
    def wait(**kwargs):
        if decoder.retire_called.is_set() and not decoder.retirement_ready.is_set():
            parked.set()
            result = actual_wait(**kwargs)
            assert decoder.retirement_ready.is_set(), "expired input retried without a genuine completion event"
            return result
        return actual_wait(**kwargs)
    monkeypatch.setattr(scheduler.waiting, "get_control", wait)
    request = client(scheduler, 2)
    wait_input(decoder, 2)
    clock[0] = 115.
    scheduler.waiting.wake()
    try:
        assert parked.wait(3)
        row, = scheduler.queue_record()["waiting"]
        assert row["blocked_reason"] == "retiring" and row["pending_terminal_reason"] == "expired"
        assert decoder.retire_calls == 1 and not request.done.is_set()
        decoder.retirement_ready.set()
        scheduler.waiting.wake()
        joined(request)
        assert isinstance(request.errors[0], CapacityError) and decoder.admissions == []
        outcome = scheduler.queue_record()["last_outcome"]
        assert outcome["ticket"] == 0 and outcome["reason"] == "expired" and outcome["waited_s"] == 15.
    finally:
        decoder.retirement_ready.set()
        scheduler.waiting.wake()
        scheduler.close()


def test_shutdown_cancels_ineligible_queue_and_waits_for_cpu_retirement():
    decoder = Decoder(blocked=(2,))
    decoder.retirement_ready.clear()
    scheduler = fifo(decoder)
    request = client(scheduler, 2, eligibility=lambda: "client_input")
    with scheduler.waiting.not_empty:
        assert scheduler.waiting.not_empty.wait_for(lambda: scheduler.waiting.revision == 1, 3)
    closed, failures = threading.Event(), []
    def close():
        try:
            scheduler.close()
        except BaseException as error:
            failures.append(error)
        finally:
            closed.set()
    closer = threading.Thread(target=close)
    closer.start()
    try:
        assert decoder.retire_called.wait(3)
        assert not closed.is_set() and not request.done.is_set()
        decoder.retirement_ready.set()
        scheduler.waiting.wake()
        assert closed.wait(3)
        joined(request)
    finally:
        decoder.retirement_ready.set()
        scheduler.waiting.wake()
        closer.join(3)
        assert not closer.is_alive()
    assert failures == []
    assert decoder.admissions == [] and len(request.errors) == 1
    assert isinstance(request.errors[0], RequestCancelled)
    assert scheduler.queue_record()["outcomes"]["cancelled"] == 1


@pytest.mark.parametrize("timeout", [0, -1, True, float("nan"), float("inf"), 10 ** 400])
def test_timeout_domain_refuses_unsupported_values_before_worker_launch(timeout):
    with pytest.raises(ValueError, match="queue_timeout_s"):
        Scheduler(Decoder(), admission_policy="fifo", max_waiting_input_bytes=4096, queue_timeout_s=timeout)


def test_new_policy_limits_and_context_are_validated_before_queue_acceptance():
    with pytest.raises(ValueError, match="explicit max_waiting_input_bytes"):
        Scheduler(Decoder(), admission_policy="fifo")
    with pytest.raises(ValueError, match="require FIFO"):
        Scheduler(Decoder(), max_waiting_input_bytes=4096)
    decoder = Decoder()
    scheduler = fifo(decoder)
    try:
        with pytest.raises(ValueError, match="infeasible context"):
            scheduler.submit([1] * 5, 1, None, False, lambda _tokens: None)
        assert scheduler.waiting.revision == 0 and scheduler.pending_requests == 0 and decoder.admissions == []
    finally:
        scheduler.close()


@pytest.mark.parametrize("failure_at", ["eligibility", "readiness-type"])
def test_unknown_admission_callback_failure_is_contained_before_numerical_work(failure_at):
    decoder = Decoder()
    poisoned = []
    decoder.poison_control = poisoned.append
    failure = RuntimeError("unretired eligibility owner")
    def unknown():
        raise failure
    if failure_at == "readiness-type":
        decoder.prepare_request = lambda _stream, **_kwargs: None
        eligibility = None
    else:
        eligibility = unknown
    scheduler = fifo(decoder)
    request = client(scheduler, 2, eligibility=eligibility)
    joined(request)
    with pytest.raises(RuntimeError, match="worker failed"):
        scheduler.close()
    assert len(request.errors) == 1 and isinstance(request.errors[0], RuntimeError)
    assert poisoned == request.errors and decoder.admissions == []
    if failure_at == "eligibility":
        assert request.errors[0] is failure


def test_fifo_ticket_and_policy_reach_all_eight_actual_tp_controls(cohort):
    from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation

    store = cohort[0].comm.store
    def wait(keys, duration):
        assert duration in (timedelta(hours=1), timedelta(seconds=60))
        with store.condition:
            assert store.condition.wait_for(lambda: all(key in store.data for key in keys), 5), keys
    store.wait = wait
    for rank, peer in enumerate(cohort):
        decoder = peer.decoder
        decoder.w.positions = object()
        decoder.w.meta = {"rank": rank, "world": 8, "coordinate_profile_identity": "a" * 64}
        decoder.input_preparation_context = install_input_preparation(decoder.w)
        decoder.input_preparation_context.bind_decoder(decoder)
        original = decoder.admit
        def admit(stream, original=original, decoder=decoder):
            decoder.observed.append(("admission-ticket", stream.admission_ticket, stream.prompt[0]))
            return original(stream)
        decoder.admit = admit
    scheduler = fifo(cohort[0])
    first = client(scheduler, 4, 3)
    with scheduler.waiting.not_empty:
        assert scheduler.waiting.not_empty.wait_for(lambda: scheduler.waiting.revision >= 1, 3)
    second = client(scheduler, 8, 3, background=True)
    try:
        joined(first, second)
    finally:
        scheduler.close()
    assert first.errors == second.errors == []
    expected = [item for item in cohort[0].decoder.observed if item[0] == "admission-ticket"]
    assert [item[2] for item in expected] == [4, 8]
    for peer in cohort:
        assert peer.admission_policy == peer.decoder.admission_policy == "fifo"
        assert [item for item in peer.decoder.observed if item[0] == "admission-ticket"] == expected
        assert not peer.requests and peer.failure is None


def test_valid_shaped_peer_request_drift_is_rejected_before_any_gpu_slot(monkeypatch):
    from tensorfold.cuda.scheduler import AdmissionStatus
    from tensorfold.cuda.streams import Stream
    from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
    from test_flashnext_tp_scheduler import Decoder as PeerDecoder
    from test_input_preparation import Store

    store = Store()
    peers = []
    for rank in range(8):
        decoder = PeerDecoder(rank, 8, store)
        decoder.w.positions = object()
        decoder.w.meta = {"rank": rank, "world": 8, "coordinate_profile_identity": "a" * 64}
        decoder.input_preparation_context = install_input_preparation(decoder.w)
        decoder.input_preparation_context.bind_decoder(decoder)
        peer = TensorParallelDecoder(decoder, NS(_abort_comm=False))
        peer.configure_admission_policy("fifo")
        peers.append(peer)
    monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", store.abort)
    source = Stream([4], 2)
    source.admission_ticket = 7
    source.admission_status = AdmissionStatus(7, 100., 100, None)
    source.principal = None
    source.retention_intent = None
    body = {"kind": "admit", "wire": 0, "stream": peers[0]._pack(source), "points": [],
            "sequence": 0, "control": peers[0]._controls()}
    def execute(rank):
        local = deepcopy(body)
        if rank == 3:
            local["stream"]["prompt"].append(5)  # Individually legal, but changes a future growth/collective path.
        return peers[rank]._execute(local, source=source if rank == 0 else None)
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(execute, range(8)))
    assert all(error is not None for _result, error, _status in results)
    assert all(peer.failure is not None and peer.decoder.observed == [] and not peer.requests for peer in peers)


def test_common_restore_publication_retries_without_manufacturing_a_wake_or_new_ticket():
    class Restoring(Decoder):
        def __init__(self):
            super().__init__()
            self.claimed, self.ready, self.attempts = None, False, []
        def admission_claims(self):
            return int(self.claimed is not None)
        def owns_admission(self, stream):
            return self.claimed is stream
        def request_blocked_reason(self, stream):
            return "restoring" if self.claimed is stream and not self.ready else None
        def admit(self, stream):
            self.attempts.append(stream.admission_ticket)
            if not self.ready:
                self.claimed = stream
                return False
            assert self.claimed is stream
            self.claimed = None
            return super().admit(stream)
        def retention_boundary(self, *, idle, needs_slot, shutdown):
            if self.claimed is not None and not self.ready:
                assert idle
                self.ready = True
                # Completion was consumed at this same actor boundary. No
                # callback remains to wake it after this common publication.
                return {"changed": True, "drained": False}
            return {"changed": False, "drained": shutdown}
    decoder = Restoring()
    scheduler = fifo(decoder, streams=1)
    try:
        request = client(scheduler, 7)
        joined(request)
        assert request.errors == [] and request.output == [8]
        assert decoder.attempts == [0, 0]
        assert scheduler.queue_record()["outcomes"]["admitted"] == 1
    finally:
        scheduler.close()


def test_waiting_failure_frame_retirement_follows_positive_actor_retirement():
    from tensorfold.cuda.output_mailbox import OutputMailbox
    from tensorfold.cuda.scheduler import AdmissionStatus
    from tensorfold.cuda.streams import Stream

    decoder = Decoder()
    decoder.retirement_ready.clear()
    scheduler = Scheduler(decoder, max_pending_requests=2, admission_policy="fifo", max_waiting_input_bytes=4096)
    calls = []
    def retired(error):
        assert decoder.retirement_ready.is_set()
        assert error.__dict__["_tensorfold_request_state_retired"] is True
        calls.append(error)
        return error
    scheduler.serving_profiles = NS(error_outcome=retired, failure=None)
    stream = Stream([1], 1, cancel=threading.Event())
    stream.principal = None
    stream.admission_status = AdmissionStatus(-1, 1., 100, None)
    box, original = OutputMailbox(1), ValueError("completed input refusal")
    with scheduler._request_limit.request():
        scheduler._request_limit.borrow(box)
        scheduler.waiting.put((stream, box))
        _revision, (entry,) = scheduler.waiting.requests()
        assert not scheduler._retire_waiting(entry, original)
        assert calls == [] and box.record()["terminal_cells"] == 0
        decoder.retirement_ready.set()
        assert scheduler._retire_waiting(entry)
        assert calls == [original]
        assert box.get_nowait() == ("error", original)

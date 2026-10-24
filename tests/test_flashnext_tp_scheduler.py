"""Eight peer schedulers share control decisions; no CUDA/model qualification is claimed."""

from datetime import timedelta
from types import SimpleNamespace
import threading
import time

import pytest

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.cuda.scheduler import Scheduler
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder


class Store:
    def __init__(self):
        self.condition = threading.Condition()
        self.data = {}

    def set(self, key, value):
        with self.condition:
            self.data[key] = value.encode() if isinstance(value, str) else value
            self.condition.notify_all()

    def wait(self, keys, duration):
        assert duration == timedelta(hours=1)
        with self.condition:
            assert self.condition.wait_for(lambda: all(key in self.data for key in keys), 5), keys

    def get(self, key):
        with self.condition:
            return self.data[key]

    def delete_key(self, key):
        with self.condition:
            del self.data[key]


class BatchStore(Store):
    def multi_set(self, keys, values):
        for key, value in zip(keys, values):
            self.set(key, value)

    def multi_get(self, keys):
        return [self.get(key) for key in keys]


class Decoder:
    """Stateful multi-request stand-in with deliberately different private rank choices."""

    def __init__(self, rank, world, store):
        self.w = SimpleNamespace(comm=SimpleNamespace(rank=rank, world=world, store=store))
        self.rank = rank
        self.prefill_rows = 16
        self.points = lambda prompt: [rank + 1]
        self.arrived = lambda: bool(rank % 2)
        self.streams, self.filling = {}, []
        self.next_id = 0
        self.observed = []

    def _pass_rows(self):
        return 3 + self.rank

    def live(self):
        return len(self.streams)

    def admit(self, stream):
        if self.live() >= 2:
            raise NoRoom("two controlled slots")
        stream.sid, self.next_id = self.next_id, self.next_id + 1
        stream.st = SimpleNamespace(pos=len(stream.prompt))
        self.observed.append(("admit", self.points(stream.prompt)))
        self.streams[stream.sid] = stream

    def round(self):
        self.observed.append(("round", self.arrived(), self._pass_rows()))
        for stream in self.streams.values():
            if not stream.done:
                stream.st.pos += 1
                stream.take([(stream.prompt[-1] + len(stream.out) + 1) % 64])
        time.sleep(0.001)
        return [stream for stream in self.streams.values() if stream.done]

    def cancel(self, streams):
        for stream in streams:
            stream.done = True
        return streams

    def finish(self, streams):
        for stream in streams:
            self.streams.pop(stream.sid, None)

    def drop(self):
        streams = list(self.streams.values())
        self.streams.clear()
        return streams


@pytest.fixture(params=[Store, BatchStore], ids=["individual-store-operations", "batched-store-operations"])
def cohort(request):
    store = request.param()
    peers = [TensorParallelDecoder(Decoder(rank, 8, store), SimpleNamespace(
        _abort_comm=False, _closing=False, _lifecycle=threading.Condition())) for rank in range(8)]
    errors = []

    def follow(peer):
        try:
            while True:
                body = peer.receive()
                if body is None:
                    return
                peer.apply(body)
        except BaseException as error:
            errors.append(error)

    threads = [threading.Thread(target=follow, args=(peer,), daemon=True) for peer in peers[1:]]
    for thread in threads:
        thread.start()
    try:
        yield peers
    finally:
        peers[0].shutdown()
        for thread in threads:
            thread.join(7)
            assert not thread.is_alive()
        assert errors == []
        assert store.data == {}


def test_round_choices_and_request_identity_are_owned_by_rank_zero(cohort):
    root = cohort[0]
    root.arrived = lambda: True
    stream = Stream([3, 5], 3, draft=False)
    root.admit(stream)
    for _ in range(3):
        ended = root.round()
    assert stream.out == [6, 7, 8]
    assert ended == [stream]
    assert all(peer.decoder.observed == root.decoder.observed for peer in cohort)
    assert root.decoder.observed[0] == ("admit", [1])
    assert root.decoder.observed[-1] == ("round", True, 3)
    root.finish(ended)
    assert all(peer.requests == {} and peer.live() == 0 for peer in cohort)


def test_client_cancellation_is_frozen_and_shared_before_the_next_round(cohort):
    cancel = threading.Event()
    stream = Stream([7], 100, draft=False, cancel=cancel)
    root = cohort[0]
    root.admit(stream)
    assert root.round() == []
    cancel.set()
    ended = root.round()
    assert ended == [stream]
    assert all(next(iter(peer.requests.values())).done for peer in cohort)
    root.finish(ended)


def test_admission_wait_and_retry_do_not_lose_the_wire_identity(cohort):
    root = cohort[0]
    first, second, waiting = [Stream([token], 3) for token in (3, 8, 12)]
    root.admit(first)
    root.admit(second)
    with pytest.raises(NoRoom):
        root.admit(waiting)
    wire = root.wires[id(waiting)]
    root.finish([first])
    root.admit(waiting)
    assert root.wires[id(waiting)] == wire
    root.finish([second, waiting])


def test_cancelled_unadmitted_owner_is_retired_on_the_next_control_frame(cohort):
    root = cohort[0]
    first, second = Stream([3], 4), Stream([8], 4)
    cancel = threading.Event()
    waiting = Stream([11], 4, cancel=cancel)
    root.admit(first)
    root.admit(second)
    with pytest.raises(NoRoom):
        root.admit(waiting)
    cancel.set()
    assert waiting.cancel.is_set()  # Scheduler can observe cancellation outside a model operation.
    root.round()
    assert all(len(peer.requests) == 2 for peer in cohort)
    root.finish([first, second])


def test_recoverable_round_failure_keeps_cancelled_owners_until_scheduler_finish(cohort):
    root = cohort[0]
    cancelled, live = Stream([3], 4), Stream([8], 4)
    root.admit(cancelled)
    root.admit(live)
    root.cancel([cancelled])

    def fail_round():
        raise ValueError("recoverable frame refusal")

    def drop_unfinished(decoder):
        removed = [stream for stream in decoder.streams.values() if not stream.done]
        for stream in removed:
            decoder.streams.pop(stream.sid)
        return removed

    for peer in cohort:
        peer.decoder.round = fail_round
        peer.decoder.drop = lambda decoder=peer.decoder: drop_unfinished(decoder)
    with pytest.raises(ValueError, match="recoverable frame"):
        root.round()
    assert root.drop() == [live]
    assert all(len(peer.requests) == 1 for peer in cohort)
    root.finish([cancelled])
    assert all(peer.requests == {} and peer.live() == 0 for peer in cohort)


@pytest.mark.parametrize("field", ["drafts", "mtp_len", "mtp_drafted", "cur"])
def test_operation_digest_includes_mtp_control_state_on_every_rank(cohort, field):
    root = cohort[0]
    stream = Stream([3], 4)
    root.admit(stream)
    for peer in cohort:
        local = next(iter(peer.requests.values()))
        local.st.mtp_len, local.st.mtp_drafted, local.st.cur = 2, 0, [0, 0]
    assert len({peer._state_digest() for peer in cohort}) == 1
    remote = next(iter(cohort[7].requests.values()))
    if field == "drafts":
        remote.drafts = [12]
    elif field == "cur":
        remote.st.cur[0] = 1
    else:
        setattr(remote.st, field, getattr(remote.st, field) + 1)
    assert cohort[7]._state_digest() != root._state_digest()
    remote.drafts = []
    remote.st.mtp_len, remote.st.mtp_drafted, remote.st.cur = 2, 0, [0, 0]
    root.finish([stream])


def test_existing_scheduler_serves_two_clients_through_eight_mirrored_decoders(cohort):
    root = cohort[0]
    scheduler = Scheduler(root, max_streams=2).start()
    outputs, failures = {}, []

    def request(token):
        try:
            words = []
            scheduler.submit([token], 5, None, False, lambda new: words.extend(new))
            outputs[token] = words
        except BaseException as error:
            failures.append(error)

    clients = [threading.Thread(target=request, args=(token,)) for token in (3, 15)]
    try:
        for client in clients:
            client.start()
        for client in clients:
            client.join(10)
            assert not client.is_alive()
    finally:
        scheduler.close()
    assert failures == []
    assert outputs == {3: [4, 5, 6, 7, 8], 15: [16, 17, 18, 19, 20]}
    assert all(peer.requests == {} and peer.live() == 0 for peer in cohort)


def test_uncertain_control_publication_poisoning_prevents_later_operations():
    store = Store()
    failure = OSError("publication may have reached a peer")
    calls = []

    def interrupted(key, value):
        calls.append(key)
        raise failure

    store.set = interrupted
    engine = SimpleNamespace(_abort_comm=False)
    root = TensorParallelDecoder(Decoder(0, 8, store), engine)
    with pytest.raises(OSError) as caught:
        root.admit(Stream([1], 2))
    assert caught.value is failure
    assert root.failure is failure and engine._abort_comm
    with pytest.raises(RuntimeError, match="poisoned"):
        root.round()
    assert len(calls) == 1


@pytest.mark.parametrize("operation", ["multi_set", "multi_get"])
def test_uncertain_batch_failure_never_retries_individual_operations(operation):
    store = BatchStore()
    failure = OSError("a partial batch may have reached peers")
    calls = []

    def interrupted(*args):
        calls.append(args)
        if operation == "multi_set":
            Store.set(store, args[0][0], args[1][0])
        raise failure

    setattr(store, operation, interrupted)
    store.wait = lambda *_: None
    engine = SimpleNamespace(_abort_comm=False)
    root = TensorParallelDecoder(Decoder(0, 8, store), engine)
    with pytest.raises(OSError) as caught:
        root.round()
    assert caught.value is failure
    assert root.failure is failure and engine._abort_comm
    with pytest.raises(RuntimeError, match="poisoned"):
        root.round()
    assert len(calls) == 1


def test_incomplete_acknowledgment_batch_poisons_the_cohort():
    store = BatchStore()
    store.wait = lambda *_: None
    store.multi_get = lambda _: []
    engine = SimpleNamespace(_abort_comm=False)
    root = TensorParallelDecoder(Decoder(0, 8, store), engine)
    with pytest.raises(RuntimeError, match="incomplete acknowledgment batch"):
        root.round()
    assert root.failure is not None and engine._abort_comm

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
from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController


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
                body = None  # Match Engine._follow_concurrent: retire this received parent after apply.
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


def test_leader_execution_epoch_and_fair_order_override_private_peer_controls(cohort):
    for peer in cohort:
        decoder = peer.decoder
        decoder.controller = AdaptiveController(slots=2, row_budget=8, proposal_limit=1, prefill_tokens=16)
        decoder.freeze_execution = lambda dec=decoder: dec.controller.freeze(sorted(dec.streams))

        def accept(plan, dec=decoder):
            dec.controller.accept(plan)
            dec.execution = plan

        decoder.accept_execution = accept

        def round_(dec=decoder):
            selected = dec.controller.select([(sid, 1) for sid in dec.streams], dec.execution)
            dec.observed.append((dec.controller.epoch, selected))
            for sid in selected:
                stream = dec.streams[sid]
                stream.st.pos += 1
                stream.take([5])
            return [stream for stream in dec.streams.values() if stream.done]

        decoder.round = round_
    root = cohort[0]
    root.decoder.controller.request(proposal_limit=0, row_budget=1)
    streams = [Stream([token], 2) for token in (1, 2)]
    for stream in streams:
        root.admit(stream)
    for _ in range(4):
        root.round()
    assert [stream.out for stream in streams] == [[5, 5], [5, 5]]
    assert all(peer.decoder.observed == root.decoder.observed for peer in cohort)
    assert root.decoder.observed[-4:] == [(1, [0]), (2, [1]), (3, [0]), (4, [1])]
    root.finish(streams)


def test_worker_boundary_settings_reach_every_rank_before_admit_finish_and_manifest(cohort):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    proposal_reads = [0] * 8
    for peer in cohort:
        decoder = peer.decoder
        decoder.controller = AdaptiveController(slots=8, row_budget=32, proposal_limit=3, prefill_tokens=16)
        decoder.share, decoder.round_s, decoder.row_s = 0.0, None, None
        decoder.pass_rows_for = lambda settings, dec=decoder: MultiDecoder.pass_rows_for(dec, settings)
        original_admit, original_finish = decoder.admit, decoder.finish

        def observe(kind, dec=decoder):
            settings = dec.controller.effective
            dec.observed.append((kind, settings["row_budget"], MultiDecoder._proposal_limit(dec),
                                 dec._pass_rows(), settings["session_ceiling"]))

        def admit(stream, dec=decoder, original=original_admit, note=observe):
            note("prefill+mtp")
            proposal_reads[dec.rank] += MultiDecoder._proposal_limit(dec)
            original(stream)

        def finish(streams, original=original_finish, note=observe):
            note("finish")
            return original(streams)

        def manifest(name, note=observe):
            note("manifest")
            return {"name": name}

        decoder.admit, decoder.finish, decoder.install_execution_manifest = admit, finish, manifest
    root = cohort[0]
    root.decoder.controller.request(row_budget=16, proposal_limit=0, prefill_tokens=8, session_ceiling=1)
    root.decoder.controller.boundary()  # The real CUDA worker applies this before _admit.
    first = Stream([1], 2)
    root.admit(first)
    root.finish([first])  # No round/plan accept happened to synchronize peers.
    assert root.install_execution_manifest("profiles-next.json") == {"name": "profiles-next.json"}
    assert proposal_reads == [0] * 8
    assert all(peer.decoder.controller.epoch == 0 and peer.failure is None for peer in cohort)
    assert all(peer.decoder.observed == root.decoder.observed for peer in cohort)
    assert ("prefill+mtp", 16, 0, 8, 1) in root.decoder.observed
    assert ("finish", 16, 0, 8, 1) in root.decoder.observed
    assert ("manifest", 16, 0, 8, 1) in root.decoder.observed
    root.decoder.controller.request(row_budget=8, proposal_limit=1, prefill_tokens=4, session_ceiling=2)
    root.decoder.controller.boundary()
    second = Stream([2], 2)
    root.admit(second)
    root.finish([second])
    assert proposal_reads == [1] * 8
    assert all(peer.decoder.observed == root.decoder.observed for peer in cohort)


def test_new_round_profile_prefill_quota_does_not_reuse_the_previous_effective_quota(cohort):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    for peer in cohort:
        decoder = peer.decoder
        decoder.controller = AdaptiveController(slots=2, row_budget=8, proposal_limit=1, prefill_tokens=16)
        decoder.share, decoder.round_s, decoder.row_s = 0.0, None, None
        decoder.pass_rows_for = lambda settings, dec=decoder: MultiDecoder.pass_rows_for(dec, settings)

        def freeze(dec=decoder):
            plan = dec.controller.freeze(sorted(dec.streams))
            # Stand in for a newly selected bounded profile; table evidence and
            # selection are verified independently in controller/identity tests.
            return {**plan, "prefill_tokens": 4, "proposal_limit": 0, "row_budget": 1}

        decoder.freeze_execution = freeze
        decoder.accept_execution = lambda plan, dec=decoder: dec.controller.accept(plan)
    root = cohort[0]
    stream = Stream([1], 1)
    root.admit(stream)
    assert root.round() == [stream]
    assert all(peer.decoder.observed[-1] == ("round", False, 4) for peer in cohort)
    assert all(peer.decoder.controller.epoch == 1 for peer in cohort)
    root.finish([stream])


def test_partial_prefix_decision_publication_aborts_without_waiting_for_normal_acks():
    class BrokenPrefixStore(BatchStore):
        def multi_set(self, keys, values):
            if keys and "/prefix/" in keys[0]:
                self.set(keys[0], values[0])
                raise OSError("publication completion is uncertain")
            super().multi_set(keys, values)

        def wait(self, keys, duration):
            pytest.fail("a poisoned publication entered normal operation ACK aggregation")

    store = BrokenPrefixStore()
    engine = SimpleNamespace(_abort_comm=False)
    root = TensorParallelDecoder(Decoder(0, 2, store), engine)
    root.decoder.round = lambda: root.decoder.prefix_decisions([(1, False)])
    with pytest.raises(RuntimeError, match="control publication failed"):
        root.round()
    assert engine._abort_comm and isinstance(root.failure, OSError)


def test_uncertain_migration_failure_never_enters_normal_ack_join():
    class NoJoinStore(BatchStore):
        def wait(self, keys, duration):
            pytest.fail("a failed migration entered normal ACK aggregation")

    engine = SimpleNamespace(_abort_comm=False)
    root = TensorParallelDecoder(Decoder(0, 2, NoJoinStore()), engine)

    def failed_round():
        error = RuntimeError("migration ownership is uncertain")
        root.decoder.poison_control(error)
        raise error

    root.decoder.round = failed_round
    with pytest.raises(RuntimeError, match="control publication failed"):
        root.round()
    assert engine._abort_comm


def test_fatal_control_notifies_supervisor_once_and_keeps_the_first_error(monkeypatch):
    from tensorfold.cuda import cohort as containment

    sent = []
    monkeypatch.setattr(containment, "notify_failure", lambda: sent.append(True))
    root = TensorParallelDecoder(Decoder(0, 2, Store()), SimpleNamespace(_abort_comm=False))
    primary = RuntimeError("first fatal error")
    root._poison_control(primary)
    root._poison_control(OSError("secondary failure"))
    assert sent == [True] and root.failure is primary


def test_failure_notification_does_not_replace_the_original_control_failure(monkeypatch):
    from tensorfold.cuda import cohort as containment

    primary, secondary = RuntimeError("native control failed"), OSError("supervisor notification failed")

    def notify():
        raise secondary

    monkeypatch.setattr(containment, "notify_failure", notify)
    root = TensorParallelDecoder(Decoder(0, 2, Store()), SimpleNamespace(_abort_comm=False))
    with pytest.raises(RuntimeError) as caught:
        root._poison_control(primary)
    assert caught.value is primary and caught.value.__cause__ is secondary


def test_only_the_leader_calls_stop_policies_and_all_peers_keep_the_same_prefix(cohort):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    calls = []
    for peer in cohort:
        decoder = peer.decoder

        def round_(dec=decoder):
            streams = list(dec.streams.values())
            news = [[11, 12, 13] for _ in streams]
            decisions = MultiDecoder._prefix_limits(dec, streams, news)
            for stream, tokens, (count, stop) in zip(streams, news, decisions):
                stream.st.pos += count
                stream.take(tokens[:count])
                stream.done = stream.done or stop
            return [stream for stream in streams if stream.done]

        decoder.round = round_
    root = cohort[0]

    def stop(tokens, prefix):
        calls.append((list(tokens), list(prefix)))
        return 1, True

    first, second = Stream([1], 100, token_limit=stop), Stream([2], 100)
    root.admit(first)
    root.admit(second)
    assert root.round() == [first]
    assert calls == [([11, 12, 13], [])]
    assert first.out == [11] and second.out == [11, 12, 13]
    assert all(peer.decoder.streams[0].out == [11] and peer.decoder.streams[0].st.pos == 2
               for peer in cohort)
    root.finish([first, second])


def test_control_arriving_during_long_prefill_applies_before_the_next_global_chunk(cohort):
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    for peer in cohort:
        decoder = peer.decoder
        decoder.controller = AdaptiveController(slots=2, row_budget=8, proposal_limit=1, prefill_tokens=16)
        decoder.kept, decoder.retained, decoder.fills = [], {}, {}
        decoder.freeze_execution = lambda dec=decoder: MultiDecoder.freeze_execution(dec)

        def accept(plan, dec=decoder):
            dec.controller.accept(plan)
            dec.execution = plan

        decoder.accept_execution = accept
        admit = decoder.admit

        def filling_admit(stream, dec=decoder, original=admit):
            original(stream)
            dec.streams.pop(stream.sid)
            dec.filling.append(stream)
            dec.fills[stream.sid] = [None, False, 0, [], None]

        decoder.admit = filling_admit
        peer._pass_rows = lambda dec=decoder: dec.controller.effective["prefill_tokens"]

        def prompt_pass(dec=decoder):
            dec.observed.append(("prefill", dec.controller.epoch, dec._pass_rows()))
            for stream in dec.filling:
                dec.fills[stream.sid][2] += dec._pass_rows()
                stream.st.pos = dec.fills[stream.sid][2]
            if dec.rank == 0 and dec.controller.epoch == 1:
                dec.controller.request(prefill_tokens=4)
            return []

        decoder._pass = prompt_pass
        decoder.round = lambda dec=decoder: MultiDecoder._fill(dec)
        decoder.finish = lambda done, dec=decoder: setattr(dec, "filling", [stream for stream in dec.filling
                                                                         if all(stream is not s for s in done)])
    root = cohort[0]
    stream = Stream([1] * 1000, 10)
    root.admit(stream)
    assert root.round() == []
    assert root.decoder.controller.record()["pending"]["prefill_tokens"] == 4
    assert root.round() == []
    assert all(peer.decoder.observed[-2:] == [("prefill", 1, 16), ("prefill", 2, 4)] for peer in cohort)
    root.finish([stream])


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


def test_noroom_retries_retire_actual_rpc_parents_but_keep_live_stream_children(cohort):
    """Real dispatch/ACK paths, CPU backend only; GC is not the cleanup protocol."""
    import gc
    import weakref
    class Parent(dict):
        pass
    class Child(list):
        pass
    root = cohort[0]
    parents = []
    original_pack = root._pack
    def pack(stream):
        parent = Parent(original_pack(stream))
        parents.append(weakref.ref(parent))
        return parent
    root._pack = pack
    peer_parents = []
    for peer in cohort[1:]:
        original_receive = peer.receive
        def receive(original=original_receive):
            body = original()
            if body is None:
                return None
            parent = Parent(body)
            peer_parents.append(weakref.ref(parent))
            return parent
        peer.receive = receive
    first, second = Stream([1], 100), Stream([2], 100)
    root.admit(first)
    root.admit(second)
    retry = Stream(Child([3, 4]), 100)
    child = weakref.ref(retry.prompt)
    gc.collect()
    enabled = gc.isenabled()
    gc.disable()
    try:
        for _ in range(16):
            try:
                root.admit(retry)
            except NoRoom as error:
                # Same optional boundary called by both real Scheduler catches.
                scheduler = object.__new__(Scheduler)
                scheduler.decoder = root
                scheduler.serving_profiles = None
                scheduler._retire_no_room_frames(error)
            else:
                pytest.fail('controlled two-slot backend unexpectedly admitted a third stream')
        deadline = time.monotonic() + 2
        while any(parent() is not None for parent in peer_parents) and time.monotonic() < deadline:
            time.sleep(.001)  # ACK precedes peer apply/receive-loop local retirement.
        assert all(parent() is None for parent in parents)
        assert peer_parents and all(parent() is None for parent in peer_parents)
        assert child() is retry.prompt
        assert all(any(stream is retry for stream in peer.requests.values()) if peer is root
                   else len(peer.requests) == 3 for peer in cohort)
        assert all(peer.decoder.live() == 2 for peer in cohort)
    finally:
        if enabled:
            gc.enable()
    root.finish([first, second])


def test_noroom_preserves_authentic_error_diagnostics_and_foreign_frame_locals(cohort):
    root = cohort[0]
    cause, context = ValueError('authentic cause'), RuntimeError('authentic context')
    primary = NoRoom('authentic pressure')
    primary.__cause__, primary.__context__ = cause, context
    primary.add_note('owned diagnostic note')
    marker = object()
    original = root.decoder.admit
    for peer in cohort:
        def refuse(stream, selected=peer):
            foreign_marker = marker  # noqa: F841 - preserved foreign-frame diagnostic is the oracle
            if selected is root:
                raise primary
            raise NoRoom('authentic pressure')
        peer.decoder.admit = refuse
    try:
        with pytest.raises(NoRoom) as caught:
            root.admit(Stream([1, 2], 4))
        error = caught.value
        assert error is primary and error.__cause__ is cause and error.__context__ is context
        assert error.__notes__ == ['owned diagnostic note']
        traceback = error.__traceback__
        names, foreign = [], []
        while traceback is not None:
            names.append(traceback.tb_frame.f_code.co_name)
            if traceback.tb_frame.f_code.co_name == 'refuse':
                foreign.append(traceback.tb_frame.f_locals['foreign_marker'])
            traceback = traceback.tb_next
        assert '_execute' in names and '_dispatch' in names and 'admit' in names
        assert foreign == [marker]
        root.retire_admission_frames(error)
        assert error.__cause__ is cause and error.__context__ is context and error.__notes__ == ['owned diagnostic note']
    finally:
        root.decoder.admit = original


def test_noroom_frame_cleanup_failure_preserves_primary():
    primary, cleanup = NoRoom('pressure'), OSError('frame cleanup')
    scheduler = object.__new__(Scheduler)
    scheduler.serving_profiles = None
    scheduler.decoder = SimpleNamespace(retire_admission_frames=lambda error: (_ for _ in ()).throw(cleanup))
    with pytest.raises(NoRoom) as caught:
        scheduler._retire_no_room_frames(primary)
    assert caught.value is primary
    assert caught.value.__cause__ is cleanup or caught.value.__context__ is cleanup


@pytest.mark.parametrize('policy', ['legacy', 'fifo'])
def test_real_scheduler_noroom_catches_release_rpc_frames_before_live_retry(cohort, policy):
    import gc
    import weakref
    class Parent(dict):
        pass
    root = cohort[0]
    if policy == 'fifo':
        from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation
        store = root.comm.store
        def wait(keys, duration):
            assert duration in (timedelta(hours=1), timedelta(seconds=60))
            with store.condition:
                assert store.condition.wait_for(lambda: all(key in store.data for key in keys), 5), keys
        store.wait = wait
        for rank, peer in enumerate(cohort):
            decoder = peer.decoder
            decoder.logical_context = decoder.prefill_rows
            decoder.w.cfg = SimpleNamespace(vocab=64)
            decoder.w.positions = object()
            decoder.w.meta = {'rank': rank, 'world': 8, 'coordinate_profile_identity': 'a' * 64}
            decoder.input_preparation_context = install_input_preparation(decoder.w)
            decoder.input_preparation_context.bind_decoder(decoder)
    entered, release = threading.Event(), threading.Event()
    original_admit = root.decoder.admit
    def gated_admit(stream):
        if not release.is_set():
            entered.set()
            assert release.wait(5)
        return original_admit(stream)
    root.decoder.admit = gated_admit
    original_pack, parents = root._pack, []
    def pack(stream):
        parent = Parent(original_pack(stream))
        parents.append(weakref.ref(parent))
        return parent
    root._pack = pack
    options = {} if policy == 'legacy' else dict(admission_policy='fifo', max_pending_requests=8,
                                                 max_waiting_input_bytes=4096)
    scheduler = Scheduler(root, max_streams=3, **options).start()
    failures, outputs = [], {}
    def request(token):
        try:
            words = []
            scheduler.submit([token], 5, None, False, lambda new: words.extend(new))
            outputs[token] = words
        except BaseException as error:
            failures.append(error)
    clients = [threading.Thread(target=request, args=(token,)) for token in (3, 15, 31)]
    enabled = gc.isenabled()
    gc.collect()
    gc.disable()
    try:
        for client in clients:
            client.start()
        assert entered.wait(5)
        with scheduler.waiting.not_empty:
            assert scheduler.waiting.not_empty.wait_for(lambda: len(scheduler.waiting.queue) >= (3 if policy == "fifo" else 2), 5)
        release.set()  # All three published before admitting to the backend's two slots.
        for client in clients:
            client.join(10)
            assert not client.is_alive()
        assert failures == []
        assert len(parents) >= 4  # Three successes plus at least one NoRoom/retry.
        assert all(parent() is None for parent in parents)
        assert set(outputs) == {3, 15, 31}
        assert all(len(words) == 5 for words in outputs.values())
    finally:
        release.set()
        scheduler.close()
        for client in clients:
            client.join(5)
        if enabled:
            gc.enable()


def test_managed_noroom_preserves_raw_frames_for_existing_error_outcome_certifier(cohort):
    root, observed = cohort[0], []
    for peer in cohort:
        def certify(error, selected=peer):
            trace, bodies = error.__traceback__, []
            while trace is not None:
                if trace.tb_frame.f_code.co_name in ('_invoke', '_execute', '_dispatch'):
                    bodies.append(trace.tb_frame.f_locals.get('body'))
                trace = trace.tb_next
            assert any(body is not None for body in bodies)
            observed.append((selected.comm.rank, error))
            return error
        peer.engine.serving_profiles = SimpleNamespace(gate=SimpleNamespace(epoch=0, available=True),
                                                       error_outcome=certify)
    first, second = Stream([1], 4), Stream([2], 4)
    root.admit(first)
    root.admit(second)
    with pytest.raises(NoRoom) as caught:
        root.admit(Stream([3], 4))
    error = caught.value
    assert root._operation_error is error
    root.retire_admission_frames(error)  # Explicit guard refuses managed-frame mutation.
    scheduler = object.__new__(Scheduler)
    scheduler.decoder, scheduler.serving_profiles = root, root.engine.serving_profiles
    scheduler._retire_no_room_frames(error)
    scheduler._retire_admission_error(error)
    assert next(value for rank, value in observed if rank == 0) is error
    assert sorted(rank for rank, value in observed) == list(range(8))
    assert all(peer._operation_error is None for peer in cohort)
    root.finish([first, second])

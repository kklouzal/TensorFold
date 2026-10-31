"""Bounded request scheduling controls; no GPU or model qualification."""

import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import timedelta
from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch

from tensorfold.cuda.scheduler import Scheduler, Waiting
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda import media_requests
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from test_flashnext_media_producer import Cohort, planned
from test_flashnext_tp_scheduler import cohort as _cohort
from test_qwen4_exp_media_plan import NAMES, planner, probe, tokenizer as _tokenizer

cohort, tokenizer = _cohort, _tokenizer


class Work:
    def __init__(self):
        self.calls, self.completed = [], set()
        self.retired = False
    def set_wake(self, wake):
        self.wake = wake
    def request(self, prepared, index, **identity):
        self.calls.append((index, identity))
        return True
    def ready(self, prepared, index, **identity):
        return index in self.completed
    def check_ready(self, prepared, index, **identity):
        return self.ready(prepared, index, **identity)
    def cancel(self, prepared):
        return self.retired


def owner_request():
    plan = planned()
    closed = []
    prepared = NS(plan=plan, token_ids=plan.token_ids, close=lambda: closed.append(True))
    owner = NS(identity=plan.profile_identity, work=Work(), producer=NS(cached=lambda *_: False),
               check_media_health=lambda: None, profile=NS(execution=NS(input_rows=4)))
    stream = Stream(list(plan.token_ids), 1, vision=prepared)
    return owner, stream, closed


def test_cpu_epoch_survives_admission_retries_and_close_waits_for_retirement():
    owner, stream, closed = owner_request()
    assert not media_requests.prepare(stream, owner, 17, lambda: None)
    owner.work.completed.add(0)
    assert media_requests.prepare(stream, owner, 17, lambda: None)
    assert all(record[1]["generation"] == 17 for record in owner.work.calls)
    with pytest.raises(ValueError, match="epoch"):
        media_requests.bind(stream, owner, 18)
    assert not media_requests.retire(stream) and closed == []
    owner.work.retired = True
    assert media_requests.retire(stream) and closed == []
    assert media_requests.retire(stream) and closed == []


def test_admitted_request_retains_replay_until_explicit_all_rank_release():
    owner, stream, closed = owner_request()
    media_requests.bind(stream, owner, 19)
    stream.st = NS(slot_generation=203)
    owner.work.retired = True
    assert media_requests.retire(stream) and closed == []
    assert stream._media_request.epoch == 19
    media_requests.release(stream)
    assert closed == [] and stream.vision is None


def replay_request(tmp_path):
    from tensorfold.vision.media_frontend import PreparedReplayMedia
    from tensorfold.vision.replay import ReplayLease, ReplayStore
    from test_media_replay import envelope

    bounds = envelope()
    store = ReplayStore(tmp_path, bounds)
    directory = tmp_path / "encoded"
    directory.mkdir()
    lease = ReplayLease(store, directory, (), 1)
    owner, stream, _ = owner_request()
    owner.executor, owner.profile.cpu = object(), bounds
    stream.vision = PreparedReplayMedia(stream.vision.plan, (), lease, owner.executor)
    owner.requests = media_requests.MediaRequests(owner)
    return owner, stream, lease, directory


def test_registry_serializes_unclaimed_close_claim_and_real_lease_retirement(tmp_path):
    from tensorfold.vision.media_frontend import PreparedReplayMedia
    from tensorfold.vision.images_http import ImageInputError

    owner, stream, lease, directory = replay_request(tmp_path)
    foreign = Stream(list(stream.prompt), 1,
        vision=PreparedReplayMedia(stream.vision.plan, (), lease, object()))
    with pytest.raises(ValueError, match="actual prepared replay owner"):
        media_requests.bind(foreign, owner, 2)
    assert owner.requests.active == {} and not lease._closed
    request = media_requests.bind(stream, owner, 3)
    assert lease._borrows == 0  # Strong owner, no redundant third replay borrower.
    assert not owner.requests.close_if_unclaimed(stream.vision)
    with pytest.raises(RuntimeError, match="active media"):
        owner.requests.check_drained()
    owner.work.retired = True
    assert media_requests.retire(stream)
    owner.requests.check_drained()
    assert lease._closed and not directory.exists()
    late = Stream(list(stream.prompt), 1, vision=request.prepared)
    with pytest.raises(ImageInputError, match="retired before"):
        media_requests.bind(late, owner, 4)
    assert owner.requests.close_if_unclaimed(request.prepared)


def test_subrun_release_preserves_shared_replay_and_finalizer_can_defer_close(tmp_path):
    from tensorfold.vision.media_frontend import PreparedReplayMedia

    owner, first, lease, directory = replay_request(tmp_path)
    prepared = first.vision
    first_request = media_requests.bind(first, owner, 1)
    # Continuation views share the original physical replay, never a second
    # independent owner that could close underneath the first subrun.
    continued = PreparedReplayMedia(prepared.plan, (), lease, owner.executor)
    second = Stream(list(continued.token_ids), 1, vision=continued)
    with pytest.raises(ValueError, match="already claimed"):
        media_requests.bind(second, owner, 2)
    owner.work.retired = True
    assert media_requests.retire(first)
    owner.requests.check_drained()
    assert first_request.released and not lease._closed and directory.exists()
    media_requests.bind(second, owner, 2)
    assert not owner.requests.close_if_unclaimed(prepared)
    assert not lease._closed
    assert media_requests.retire(second)
    owner.requests.check_drained()
    assert lease._closed and not directory.exists()


@pytest.mark.parametrize("failure_at", ["stream-publication", "stream-attached", "claim-return", "rollback-failure"])
def test_bind_publication_failure_rolls_back_only_its_actual_registry_claim(tmp_path, failure_at, monkeypatch):
    owner, original, lease, directory = replay_request(tmp_path)
    failure = MemoryError("request publication allocation")
    if failure_at != "claim-return":
        class RefusingStream(Stream):
            def __setattr__(self, name, value):
                if name == "_media_request":
                    if failure_at == "stream-attached":
                        super().__setattr__(name, value)
                    raise failure
                super().__setattr__(name, value)
        stream = RefusingStream(list(original.prompt), 1, vision=original.vision)
    else:
        stream = original
        claim = owner.requests.claim
        def interrupted(request):
            claim(request)
            raise failure
        monkeypatch.setattr(owner.requests, "claim", interrupted)
    release = owner.requests.release
    cleanup = RuntimeError("claim rollback uncertain")
    if failure_at == "rollback-failure":
        def cannot_release(_request):
            raise cleanup
        monkeypatch.setattr(owner.requests, "release", cannot_release)
    with pytest.raises(MemoryError) as caught:
        media_requests.bind(stream, owner, 4)
    assert caught.value is failure
    if failure_at == "stream-attached":
        assert stream._media_request.released
        with pytest.raises(ValueError, match="claim is no longer active"):
            media_requests.prepare(stream, owner, 4, lambda: None)
    else:
        assert getattr(stream, "_media_request", None) is None
    if failure_at == "rollback-failure":
        request, = owner.requests.active.values()
        assert failure.__cause__ is cleanup
        assert request in failure._tensorfold_retained_owners
        assert not owner.requests.close_if_unclaimed(original.vision)
        assert not lease._closed and directory.exists()
        monkeypatch.setattr(owner.requests, "release", release)
        owner.requests.rollback_claim(request)
    assert owner.requests.active == {} and owner.work.calls == []
    if failure_at != "rollback-failure":
        assert not lease._closed and directory.exists()  # Caller still owns subsequent subruns/final close.
    assert owner.requests.close_if_unclaimed(original.vision)
    assert lease._closed and not directory.exists()


@pytest.mark.parametrize("refuse_rank", [None, 3], ids=["ready", "peer-refuses-before-admission"])
def test_eight_peer_request_metadata_and_replay_close_follow_final_ack(cohort, tokenizer, refuse_rank):
    from tensorfold.vision.media_frontend import PreparedReplayMedia
    from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation
    from test_media_replay import envelope

    peers = cohort
    selected = planner(tokenizer)
    image, _, begin, end = NAMES
    bounds = envelope()
    probes = (replace(probe("image", 0), envelope_id=bounds.identity),)
    plan = selected(f"hello {begin}{image}{end}", probes, max_prompt_tokens=512, max_visual_tokens=None)
    closed = []
    def close():
        assert all(not peer.requests for peer in peers)
        closed.append(True)
    prepared = PreparedReplayMedia(plan, probes, NS(close=close), NS())
    store = peers[0].comm.store
    # Scheduler and input-preparation protocols share the same actual store;
    # the latter has an explicit 60-second retirement contract.
    def wait(keys, duration):
        assert duration in (timedelta(hours=1), timedelta(seconds=60))
        with store.condition:
            assert store.condition.wait_for(lambda: all(key in store.data for key in keys), 5), keys
    store.wait = wait
    for rank, peer in enumerate(peers):
        work = Work() if rank == 0 else None
        if work is not None:
            work.completed.add(0)
            work.retired = True
        peer.decoder.vision = NS(is_media_runtime=True, identity=plan.profile_identity, work=work,
            producer=NS(cached=lambda *_: False), check_media_health=lambda: None, contract=selected.contract,
            profile=NS(cpu=bounds))
        peer.decoder.w.positions = object()
        peer.decoder.w.meta = {"rank": rank, "world": 8, "coordinate_profile_identity": plan.profile_identity}
        peer.decoder.input_preparation_context = install_input_preparation(peer.decoder.w)
        peer.decoder.input_preparation_context.bind_decoder(peer.decoder)
    leader = peers[0]
    stream = Stream(list(plan.token_ids), 2, vision=prepared)
    assert leader.prepare_request(stream, wake=lambda: None)
    cpu_epoch = stream._media_request.epoch
    if refuse_rank is not None:
        def refuse(_body):
            raise ValueError("malformed follower media metadata")
        peers[refuse_rank]._unpack = refuse
        with pytest.raises(ValueError):
            leader.admit(stream)
        assert all(peer.decoder.observed == [] and not peer.requests for peer in peers)
        assert all(peer.failure is None for peer in peers)
        assert leader.retire_request(stream)
        prepared.close()
        assert closed == [True]
        return
    leader.admit(stream)
    wire = leader.wires[id(stream)]
    for peer in peers[1:]:
        remote = peer.requests[wire]
        assert remote._media_request.epoch == cpu_epoch
        assert remote._media_request.plan == plan
        assert not hasattr(remote.vision, "executor") and not hasattr(remote.vision, "close")
    assert closed == []
    leader.finish([stream])
    assert closed == [] and all(not peer.requests for peer in peers)
    prepared.close()  # The logical caller closes after every generation subrun.
    assert closed == [True]


def test_no_unavailable_visual_rows_or_cross_unit_piece_and_shared_arena_bound():
    owner, first, _ = owner_request()
    _, second, _ = owner_request()
    for sid, stream in enumerate((first, second)):
        stream.sid = sid
        media_requests.bind(stream, owner, sid)
    decoder = object.__new__(MultiDecoder)
    decoder.media_runtime, decoder._media_limits = owner, None
    decoder.w = NS(meta={})
    decoder.filling = [first, second]
    decoder.passed, decoder._pass_rows = {}, lambda: 100
    decoder._points = lambda *_: []
    start = first._media_request.plan.units[0].run.start
    decoder.fills = {0: [None, False, start, [], None], 1: [None, False, start, [], None]}
    assert decoder._pieces() == []
    owner.work.completed.add(0)
    pieces = decoder._pieces()
    assert sum(count for _stream, _start, count in pieces) == 4
    assert all(at + count <= stream._media_request.plan.units[0].run.end for stream, at, count in pieces)
    decoder._media_limits = {0: (start, 1), 1: (start, 2)}
    assert [count for _stream, _start, count in decoder._pieces()] == [1, 2]
    decoder._media_limits[0] = (start + 1, 1)
    with pytest.raises(ValueError, match="boundary"):
        decoder._pieces()


def test_expected_cpu_failure_is_returned_before_gpu_work_and_poison_is_not_downgraded():
    from tensorfold.vision.images_http import ImageInputError

    owner, stream, _ = owner_request()
    media_requests.bind(stream, owner, 2)
    stream.sid = 0
    decoder = object.__new__(MultiDecoder)
    decoder.media_runtime, decoder.filling, decoder.prefill_rows = owner, [stream], 32
    decoder.fills = {0: [None, False, 0, [], None]}
    error = ImageInputError("invalid temporal unit")
    def fail(*_args, **_kwargs):
        raise error
    owner.work.ready = fail
    limits, failures = decoder.prepare_media_round()
    assert limits == {0: (0, 0)} and failures == [(stream, error)]
    owner.check_media_health = lambda: (_ for _ in ()).throw(RuntimeError("unretired worker"))
    poisoned = []
    decoder.poison_control = poisoned.append
    with pytest.raises(ImageInputError) as caught:
        decoder.prepare_media_round()
    assert caught.value is error and poisoned == [error]
    assert isinstance(error.__cause__, RuntimeError) and str(error.__cause__) == "unretired worker"


def test_tp_cpu_input_refusal_stays_local_but_unretired_worker_contains_cohort(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
    from tensorfold.vision.images_http import ImageInputError
    from test_flashnext_tp_scheduler import Decoder, Store

    owner, stream, _ = owner_request()
    owner.is_media_runtime = True
    decoder = Decoder(0, 8, Store())
    decoder.vision = owner
    engine = NS(_abort_comm=False)
    control = TensorParallelDecoder(decoder, engine)
    def refuse(*_args, **_kwargs):
        raise ImageInputError("malformed media unit")
    owner.work.ready = refuse
    with pytest.raises(ImageInputError, match="malformed media"):
        control.prepare_request(stream, wake=lambda: None)
    assert control.failure is None and not engine._abort_comm
    fatal = RuntimeError("worker retirement uncertain")
    def unhealthy():
        raise fatal
    owner.check_media_health = unhealthy
    notified = []
    monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notified.append(True))
    with pytest.raises(RuntimeError, match="retirement uncertain"):
        control.prepare_request(stream, wake=lambda: None)
    assert control.failure is fatal and engine._abort_comm and notified == [True]


@pytest.mark.parametrize("failure_at", ["import", "classifier", "runtime-lookup", "health-inspection"])
def test_input_refusal_inspection_failure_poison_preserves_original_before_drop(monkeypatch, failure_at):
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
    from tensorfold.families.qwen4_exp.cuda import execution_identity
    from tensorfold.vision.images_http import ImageInputError
    from test_flashnext_tp_scheduler import Decoder, Store

    owner, stream, _ = owner_request()
    owner.is_media_runtime = True
    store = Store()
    decoder = Decoder(0, 8, store)
    decoder.vision = owner
    engine = NS(_abort_comm=False)
    control = TensorParallelDecoder(decoder, engine)
    primary, inspection = ImageInputError("original bad input"), MemoryError("outcome inspection failed")
    def refuse(*_args, **_kwargs):
        raise primary
    owner.work.ready = refuse
    def fail(*_args, **_kwargs):
        raise inspection
    if failure_at == "import":
        import builtins

        actual_import = builtins.__import__
        def importing(name, *args, **kwargs):
            if name == "tensorfold.vision.images_http":
                fail()
            return actual_import(name, *args, **kwargs)
        monkeypatch.setattr(builtins, "__import__", importing)
    elif failure_at == "classifier":
        monkeypatch.setattr(execution_identity, "_unretired", fail)
    elif failure_at == "runtime-lookup":
        actual = media_requests.runtime
        looked = []
        def lookup(value):
            looked.append(True)
            return actual(value) if len(looked) == 1 else fail()
        monkeypatch.setattr(media_requests, "runtime", lookup)
    else:
        checked = []
        def health():
            checked.append(True)
            if len(checked) > 1:
                fail()
        owner.check_media_health = health
    notifications = []
    monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notifications.append(True))
    with pytest.raises(ImageInputError) as caught:
        control.prepare_request(stream, wake=lambda: None)
    assert caught.value is primary and primary.__cause__ is inspection
    assert control.failure is primary and engine._abort_comm and notifications == [True]
    with pytest.raises(RuntimeError, match="poisoned"):
        control.drop()
    assert control.sequence == 0 and store.data == {}


@pytest.mark.parametrize("unretired", [False, True], ids=["unknown-outcome", "unretired-input"])
def test_containment_failure_does_not_replace_original_cpu_outcome(monkeypatch, unretired):
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
    from tensorfold.vision.images_http import ImageInputError
    from test_flashnext_tp_scheduler import Decoder, Store

    owner, stream, _ = owner_request()
    owner.is_media_runtime = True
    store, engine = Store(), NS(_abort_comm=False)
    decoder = Decoder(0, 8, store)
    decoder.vision = owner
    control = TensorParallelDecoder(decoder, engine)
    primary = ImageInputError("unretired input") if unretired else RuntimeError("unknown CPU failure")
    if unretired:
        primary._tensorfold_retained_owners = [owner]
    def fail(*_args, **_kwargs):
        raise primary
    owner.work.ready = fail
    notification = MemoryError("cohort notification failed")
    def cannot_notify():
        raise notification
    monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", cannot_notify)
    with pytest.raises(type(primary)) as caught:
        control.prepare_request(stream, wake=lambda: None)
    assert caught.value is primary and primary.__cause__ is notification
    assert control.failure is primary and engine._abort_comm and control.sequence == 0 and store.data == {}


def test_rank_encoder_preserves_default_wire_and_explicit_compact_boundary(monkeypatch):
    from tensorfold.cuda import rank_protocol

    body = {"a": [1, 2], "label": "\u2603"}
    assert rank_protocol.bounded_json(body) == '{"a": [1, 2], "label": "\\u2603"}'
    compact = '{"a":[1,2],"label":"\\u2603"}'
    monkeypatch.setattr(rank_protocol, "MAX_MESSAGE_BYTES", len(compact))
    assert rank_protocol.bounded_json(body, separators=(",", ":")) == compact
    with pytest.raises(ValueError, match="transport envelope"):
        rank_protocol.bounded_json(body)


def test_combined_tp_wire_refusal_precedes_sequence_advance_and_publication(monkeypatch):
    from tensorfold.cuda import rank_protocol
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
    from test_flashnext_tp_scheduler import Decoder, Store

    store = Store()
    engine = NS(_abort_comm=False)
    control = TensorParallelDecoder(Decoder(0, 8, store), engine)
    monkeypatch.setattr(rank_protocol, "MAX_MESSAGE_BYTES", 192)
    # Each individual string is legal; the enclosing control/request is not.
    with pytest.raises(ValueError, match="transport envelope"):
        control._invoke({"kind": "round", "first": "a" * 80, "second": "b" * 80})
    assert control.sequence == 0 and store.data == {}
    assert control.failure is None and not engine._abort_comm


def test_poisoned_tp_readiness_contains_cohort_before_any_later_drop_publication(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
    from test_flashnext_tp_scheduler import Decoder, Store

    store = Store()
    engine, decoder = NS(_abort_comm=False), Decoder(0, 8, store)
    failure = RuntimeError("unretired media readiness owner")
    def unreadable():
        raise failure
    decoder.media_runnable = unreadable
    control = TensorParallelDecoder(decoder, engine)
    notifications = []
    monkeypatch.setattr("tensorfold.cuda.cohort.notify_failure", lambda: notifications.append(True))
    with pytest.raises(RuntimeError) as caught:
        control.media_runnable()
    assert caught.value is failure and control.failure is failure and engine._abort_comm
    assert notifications == [True] and store.data == {}
    with pytest.raises(RuntimeError, match="poisoned"):
        control.drop()
    assert store.data == {} and control.sequence == 0
    owner, stream, _ = owner_request()
    owner.is_media_runtime, decoder.vision = True, owner
    with pytest.raises(RuntimeError, match="poisoned"):
        control.prepare_request(stream, wake=lambda: None)
    assert owner.work.calls == [] and getattr(stream, "_media_request", None) is None


def gate_collective():
    """Reuse the producer's eight-peer owner/collective fixture for real CPU tensors."""
    group, values = Cohort(), {}
    peers = []
    for rank in range(8):
        peer = group.comm(rank)
        peer.device, peer.gathers = torch.device("cpu"), 0
        def gather(source, output, peer=peer):
            phase = peer.gathers
            peer.gathers += 1
            with group.condition:
                values[phase, peer.rank] = source.clone()
            group.barrier.wait()
            output.copy_(torch.cat([values[phase, other] for other in range(8)]))
            group.barrier.wait()
        peer.all_gather = gather
        peers.append(peer)
    return group, peers


def test_media_gate_keeps_actual_rank_ledgers_and_votes_every_allocation():
    from tensorfold.cuda.memory_gate import MemoryGate
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import SharedMemoryGate

    group, peers = gate_collective()
    live = [1007] * 8
    def run(rank):
        held = 100 + rank * 101
        reserve = 900 if rank == 5 else 100
        live[rank] = reserve + 7
        original = MemoryGate(held + reserve + 7, reserve, lambda: live[rank])
        original.held = held
        gate = SharedMemoryGate(original, peers[rank], rank_local=True)
        assert gate.room == original.room and gate.held == held and gate.reserve == reserve
        assert peers[rank].gathers == 0  # Existing fits vote suffices; no lossy global-room reduction.
        assert gate.fits(7)
        assert not gate.fits(8)
        if rank == 3:
            live[rank] -= 1
        group.barrier.wait()
        assert not gate.fits(7)  # One rank's live pressure refuses all eight allocations.
        assert peers[rank].gathers == 3
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))


def test_legacy_shared_gate_keeps_minimum_room_and_initial_collective():
    from tensorfold.cuda.memory_gate import MemoryGate
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import SharedMemoryGate

    _group, peers = gate_collective()
    def run(rank):
        original = MemoryGate(1000 + rank * 17, 200, lambda: 1000)
        original.held = 100
        gate = SharedMemoryGate(original, peers[rank])
        assert gate.room == 1000 and gate.reserve == 200 and gate.held == 100
        assert peers[rank].gathers == 1
        assert gate.fits(700)
        assert peers[rank].gathers == 2
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(run, range(8)))


def test_full_cohort_completion_wait_preserves_queued_request():
    waiting = Waiting()
    queued = (Stream([1], 1), object())
    waiting.put(queued)
    entered, finished, values = threading.Event(), threading.Event(), []
    def owner():
        entered.set()
        values.append(waiting.get_control())
        finished.set()
    thread = threading.Thread(target=owner)
    thread.start()
    assert entered.wait(5)
    assert not finished.is_set()
    waiting.wake()
    assert finished.wait(5)
    thread.join(5)
    assert len(values) == 1 and waiting.is_wake(values[0])
    assert waiting.get_nowait() == queued


def test_pixel_copy_has_no_mapping_owner_before_gpu_producer_reads():
    owner, stream, _ = owner_request()
    request = media_requests.bind(stream, owner, 4)
    closed = []
    values = np.ones((24, 1536), dtype=np.float32)
    class Borrow:
        def __enter__(self):
            return NS(values=values)
        def __exit__(self, *_):
            closed.append("borrow")
    lease = NS(borrow=Borrow, close=lambda: closed.append("lease"))
    owner.work.take_ready = lambda *_args, **_kwargs: lease
    copied = request.pixels(request.plan.units[0])
    assert closed == ["borrow", "lease"]
    assert copied.device.type == "cpu" and copied.dtype == torch.float32
    values.fill(0)
    assert bool(torch.all(copied == 1))


class PendingDecoder:
    """Explicit completion gates make waiting/cancellation tests deterministic."""
    def __init__(self):
        self.arrived = lambda: False
        self.streams, self.filling = {}, []
        self.requested, self.gpu = threading.Event(), threading.Event()
        self.ready, self.retired = False, False
        self.rounds, self.finishes = 0, []
    def live(self):
        return len(self.streams)
    def prepare_request(self, stream, *, wake):
        self.wake, self.stream = wake, stream
        self.requested.set()
        return self.ready
    def retire_request(self, stream):
        return self.retired
    def media_runnable(self):
        return any(not stream.done for stream in self.streams.values())
    def admit(self, stream):
        assert self.ready
        stream.sid, stream.st = 0, NS()
        self.streams[0] = stream
        self.gpu.set()
    def round(self):
        self.rounds += 1
        for stream in self.streams.values():
            if not stream.done:
                stream.take([7])
                return [stream]
        return []
    def finish(self, streams):
        for stream in streams:
            assert self.retired
            self.finishes.append(stream)
            self.streams.pop(stream.sid)
    def drop(self):
        result = list(self.streams.values())
        self.streams.clear()
        return result


def test_idle_request_wakes_on_ready_then_defers_reply_until_cpu_retirement():
    decoder = PendingDecoder()
    scheduler = Scheduler(decoder, max_streams=1).start()
    output, failures, finished = [], [], threading.Event()
    def client():
        try:
            scheduler.submit([1], 1, None, False, lambda tokens: output.extend(tokens))
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()
    thread = threading.Thread(target=client)
    thread.start()
    try:
        assert decoder.requested.wait(5)
        assert not decoder.gpu.is_set()
        decoder.ready = True
        decoder.wake()
        assert decoder.gpu.wait(5)
        assert not finished.is_set()
        decoder.retired = True
        decoder.wake()
        assert finished.wait(5)
    finally:
        scheduler.close()
        thread.join(5)
    assert not thread.is_alive() and failures == [] and output == [7]
    assert len(decoder.finishes) == 1


def test_unadmitted_cancellation_waits_for_cpu_retirement_without_gpu_execution():
    decoder = PendingDecoder()
    scheduler = Scheduler(decoder, max_streams=1).start()
    failures, finished = [], threading.Event()
    def client():
        try:
            scheduler.submit([1], 1, None, False, lambda _tokens: None)
        except BaseException as error:
            failures.append(error)
        finally:
            finished.set()
    thread = threading.Thread(target=client)
    thread.start()
    try:
        assert decoder.requested.wait(5)
        scheduler._request_cancel(decoder.stream.cancel)
        assert not finished.is_set()
        decoder.retired = True
        decoder.wake()
        assert finished.wait(5)
    finally:
        scheduler.close()
        thread.join(5)
    assert failures == [] and not decoder.gpu.is_set() and decoder.finishes == []

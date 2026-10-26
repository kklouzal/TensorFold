"""Actual CPU work/replay/producer bridge; controlled encoder, no CUDA/model claim."""

import threading
import time
from datetime import timedelta
from dataclasses import replace
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cleanup import finish, rollback
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda import media_mtp, media_requests
from tensorfold.families.qwen4_exp.cuda.media_producer import MediaInputProducer
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
from tensorfold.families.qwen4_exp.cuda.input_preparation import install_input_preparation
from tensorfold.vision.images_http import ImageInputError
from tensorfold.vision.replay import MediaBusy, MediaCancelled
from tensorfold.vision.media_work import MediaWork
from test_flashnext_media_mtp import state
from test_flashnext_media_producer import weights
from test_flashnext_tp_scheduler import cohort as _cohort
from test_media_work import image_source, make_prepared, pixel
from test_qwen4_exp_media_plan import NAMES, planner, tokenizer

cohort = _cohort


def full_prepared(tmp_path, monkeypatch, count=1):
    prepared = make_prepared(tmp_path, monkeypatch)
    if count == 2:
        replay = prepared.executor.store.stage([image_source(size=(32, 32)) for _ in range(2)],
            allow_urls=False, cancelled=lambda: False)
        prepared.close()
        prepared.replay = replay
        prepared.probes = tuple(replace(prepared.probes[0], ref=ref) for ref in replay.refs)
    selected = planner(tokenizer.__wrapped__(), profile_identity="a" * 64)
    image, _, begin, end = NAMES
    prepared.plan = selected("hello " + " ".join(f"{begin}{image}{end}" for _ in range(count)),
                             prepared.probes, max_prompt_tokens=512, max_visual_tokens=None)
    return prepared, selected.contract


def test_actual_work_claim_readiness_spans_and_second_subrun(tmp_path, monkeypatch):
    prepared, _contract = full_prepared(tmp_path, monkeypatch)
    entered, release, changed = threading.Event(), threading.Event(), threading.Event()
    made = []
    def decode(probe, index, *, cancelled):
        entered.set()
        assert release.wait(3)
        assert not cancelled()
        lease = pixel(prepared.executor, probe, index, value=7.)
        made.append(lease)
        return lease
    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    work = MediaWork(prepared.executor)
    w = weights(mtp=True)
    class Encoder:
        def _retire_inputs(self):
            pass
        def encode_device_unit(self, pixels, grid):
            assert made[-1]._closed  # Raw mapping retired before encoder work.
            assert pixels.dtype == torch.float16 and bool(torch.all(pixels == 7))
            return NS(features=torch.full((int(grid.prod()) // 4, 4), 13., dtype=torch.float16))
    producer = MediaInputProducer.create(w, encoder=Encoder(), encoder_rank=0, rows=64, patches=4,
                                         cache_rows=0, register=lambda _owner: None)
    owner = NS(identity=prepared.plan.profile_identity, work=work, executor=prepared.executor,
               profile=NS(cpu=prepared.executor.envelope), producer=producer,
               check_media_health=prepared.executor.check_health)
    owner.requests = media_requests.MediaRequests(owner)
    wake = changed.set  # One stable callback owns every subrun.
    def ready(stream, epoch):
        deadline = time.monotonic() + 3  # Test deadlock guard, not a service SLO.
        while not media_requests.prepare(stream, owner, epoch, wake):
            assert changed.wait(max(0., deadline - time.monotonic()))
            changed.clear()
    def close():
        finish((lambda: work.close(timeout=3), producer.close, prepared.close))
    def consume(stream, generation):
        stream.st = state()
        stream.st.slot_generation = generation
        stream.st.media_target_position, stream.st.media_mtp_position = w.positions.target, w.positions.mtp
        pieces = ((stream, 0, len(stream.prompt)),)
        media_requests.publish(owner, pieces)
        span = stream.st.media_input_span
        assert span.generation == generation and span.generation != stream._media_request.epoch
        expected = w.embed[0].table[torch.tensor(stream.prompt)].clone()
        unit, = prepared.plan.units
        expected[unit.run.start:unit.run.end] = 13
        with media_mtp.borrow_span(stream.st, "target") as lease:
            target = lease.rows(0, stream.prompt).clone()
        with media_mtp.borrow_span(stream.st, "mtp") as lease:
            actual_next = lease.rows(1, stream.prompt[1:]).clone()
        torch.testing.assert_close(target, expected, rtol=0, atol=0)
        torch.testing.assert_close(actual_next, expected[1:], rtol=0, atol=0)
        media_requests.retire_batch(owner, pieces, cancelled=False)
        assert span.closed and stream.st.media_input_span is None
        assert media_requests.retire(stream)
        media_requests.release(stream)  # CPU-only local rank's completed consumer proof.
    try:
        stream = Stream(list(prepared.token_ids), 1, vision=prepared)
        assert not media_requests.prepare(stream, owner, 8, wake)
        assert entered.wait(3)
        assert not media_requests.prepare(stream, owner, 8, wake)
        release.set()
        ready(stream, 8)
        consume(stream, 3)
        owner.requests.check_drained()
        assert not prepared.replay._closed and prepared.replay._borrows == 0
        second = Stream(list(prepared.token_ids), 1, vision=prepared)
        ready(second, 9)
        consume(second, 4)
        assert len(made) == 2 and all(lease._closed for lease in made)
        assert owner.requests.close_if_unclaimed(prepared)
        assert prepared.replay._closed and prepared.executor.store._staging == 0
    except BaseException as primary:
        rollback((work, prepared), primary, close)
    else:
        close()


@pytest.mark.parametrize("phase", ["initial", "next-unit"])
@pytest.mark.parametrize("outcome", ["input", "busy", "cancelled"])
def test_actual_bad_work_is_retired_before_tp_publication_and_next_unit_pixels(tmp_path, monkeypatch, cohort, phase, outcome):
    prepared, contract = full_prepared(tmp_path, monkeypatch, count=2)
    changed, bad_finished = threading.Event(), threading.Event()
    failure = {"input": ImageInputError, "busy": MediaBusy, "cancelled": MediaCancelled}[outcome](
        "controlled failed CPU media unit")
    if outcome == "busy":
        failure.__cause__ = ValueError("retired non-retryable outcome")  # Clean temporary Busy stays parked internally.
    def decode(probe, index, *, cancelled):
        assert not cancelled()
        if phase == "initial" or probe.ref.ordinal == 1:
            bad_finished.set()
            raise failure
        return pixel(prepared.executor, probe, index, value=7.)
    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    work = MediaWork(prepared.executor)
    w = weights(mtp=True)
    class Encoder:
        def _retire_inputs(self):
            pass
        def encode_device_unit(self, pixels, grid):
            return NS(features=torch.full((int(grid.prod()) // 4, 4), 13., dtype=torch.float16))
    producer = MediaInputProducer.create(w, encoder=Encoder(), encoder_rank=0, rows=64, patches=4,
                                         cache_rows=0, register=lambda _owner: None)
    owner = NS(is_media_runtime=True, identity=prepared.plan.profile_identity, work=work,
        executor=prepared.executor, profile=NS(cpu=prepared.executor.envelope, execution=NS(input_rows=64)),
        producer=producer, check_media_health=prepared.executor.check_health, contract=contract)
    owner.requests = media_requests.MediaRequests(owner)
    store = cohort[0].comm.store
    def wait(keys, duration):
        assert duration in (timedelta(hours=1), timedelta(seconds=60))
        with store.condition:
            assert store.condition.wait_for(lambda: all(key in store.data for key in keys), 5), keys
    store.wait = wait
    for rank, peer in enumerate(cohort):
        decoder = peer.decoder
        decoder.vision = owner if rank == 0 else NS(is_media_runtime=True, identity=owner.identity,
            work=None, profile=owner.profile, contract=contract)
        decoder.media_runtime, decoder.fills = decoder.vision, {}
        decoder.w.positions = object()
        decoder.w.meta = {"rank": rank, "world": 8, "coordinate_profile_identity": owner.identity}
        decoder.input_preparation_context = install_input_preparation(decoder.w)
        decoder.input_preparation_context.bind_decoder(decoder)
        original = decoder.admit
        def admit(stream, decoder=decoder, original=original):
            original(stream)
            decoder.streams.pop(stream.sid)
            stream.st = state()
            stream.st.media_target_position, stream.st.media_mtp_position = w.positions.target, w.positions.mtp
            decoder.filling.append(stream)
            decoder.fills[stream.sid] = [None, True, 0, [], None]
        def cancel(streams, decoder=decoder):
            for stream in streams:
                decoder.filling.remove(stream)
                decoder.fills.pop(stream.sid)
                stream.done = True
            return streams
        def empty_round(decoder=decoder):
            assert not decoder.filling  # Bad-unit failure must remove it before an execution round.
            return []
        decoder.admit, decoder.cancel, decoder.round = admit, cancel, empty_round
    leader = cohort[0]
    leader.decoder.prefill_rows = 64
    leader.decoder.prepare_media_round = lambda: MultiDecoder.prepare_media_round(leader.decoder)
    stream = Stream(list(prepared.token_ids), 2, vision=prepared)
    wake = changed.set
    def initial_ready():
        deadline = time.monotonic() + 3
        while not leader.prepare_request(stream, wake=wake):
            assert changed.wait(max(0., deadline - time.monotonic()))
            changed.clear()
    def close():
        finish((lambda: work.close(timeout=3), producer.close, prepared.close))
    try:
        if phase == "initial":
            with pytest.raises(ImageInputError) as caught:
                initial_ready()
            assert caught.value is failure
            assert producer.context.ordinal == 0 and leader.sequence == 0 and store.data == {}
        else:
            initial_ready()
            leader.admit(stream)
            count = prepared.plan.units[0].run.end
            pieces = ((stream, 0, count),)
            media_requests.publish(owner, pieces)
            with media_mtp.borrow_span(stream.st, "target") as lease:
                lease.rows(0, stream.prompt[:count])
            with media_mtp.borrow_span(stream.st, "mtp") as lease:
                lease.rows(0, stream.prompt[:count])
            media_requests.retire_batch(owner, pieces, cancelled=False)
            for peer in cohort:
                remote = peer.requests[leader.wires[id(stream)]]
                remote.st.pos = count
                peer.decoder.fills[remote.sid][2] = count
            assert bad_finished.wait(3)
            deadline = time.monotonic() + 3
            while not work.ready(prepared, 1, **stream._media_request._arguments()):
                assert changed.wait(max(0., deadline - time.monotonic()))
                changed.clear()
            before = producer.context.ordinal
            assert leader.round() == [stream]
            assert stream.error is failure and producer.context.ordinal == before
            assert all(not peer.failure and not peer.decoder.filling for peer in cohort)
        assert work.record()["owned_units"] == 0 and prepared.replay._borrows == 0
        assert leader.retire_request(stream) and leader.retire_request(stream)
        assert leader.failure is None and not leader.engine._abort_comm
        if phase == "next-unit":
            leader.finish([stream])
        owner.requests.check_drained()
        assert owner.requests.close_if_unclaimed(prepared)
    except BaseException as primary:
        rollback((work, producer, prepared), primary, close)
    else:
        close()

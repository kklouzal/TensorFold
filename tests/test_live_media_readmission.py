# ruff: noqa: E402 - optional Torch must be checked before backend imports.
"""Real replay/CPU work/producer/State RAM and saved-cursor readmission.

The encoder and codec pixel boundary are controlled; no vision-model/GPU claim.
"""
import threading
import time
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")

from tensorfold.cleanup import finish
from tensorfold.cuda.output_mailbox import OutputMailbox
from tensorfold.cuda.scheduler import DeliveryOwner
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda import forward, media_mtp, media_requests
from tensorfold.families.qwen4_exp.cuda.media_producer import MediaInputProducer
from tensorfold.families.qwen4_exp.cuda.retained_catalog import make_retention_catalog
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import FrozenCancellation, TensorParallelDecoder
from tensorfold.server.request_limits import RequestLimit
from tensorfold.vision.images_http import ImageInputError
from tensorfold.vision.media_work import MediaWork
from test_dspark_taps_forward import cpu_weights
from test_flashnext_media_mtp import bind_reference_inputs
from test_media_request_work import full_prepared
from test_media_work import pixel
from test_retained_multi_caller import decoder_for


@pytest.mark.parametrize("bad_ahead", [False, True], ids=["completed-current-unit", "retired-ahead-input-error"])
def test_saved_media_cursor_keeps_actual_ahead_job_epoch_and_never_restarts_consumed_first_unit(tmp_path, monkeypatch, bad_ahead):
    prepared, contract = full_prepared(tmp_path, monkeypatch, count=2)
    changed, ahead, release = threading.Event(), threading.Event(), threading.Event()
    calls = []
    failure = ImageInputError("controlled current visual unit refusal")
    def decode(probe, index, *, cancelled):
        calls.append(probe.ref.ordinal)
        if probe.ref.ordinal == 1:
            ahead.set()
            assert release.wait(5)
            assert not cancelled()  # Cursor0 on resume would wrongly supersede this actual unit1 job.
            if bad_ahead:
                raise failure
        return pixel(prepared.executor, probe, index, value=7.)
    monkeypatch.setattr(prepared.executor, "pixel_unit", decode)
    work = MediaWork(prepared.executor)
    weights = cpu_weights(monkeypatch)
    weights.meta.update(rank=0, media_inputs_enabled=True)
    source, destination = [State(weights, 64, 4, "fp16", limit=64) for _ in range(2)]
    # CPU math fixture only: attention scratch queries its nominal launch
    # geometry even for CPU tensors; this does not provision/report a GPU.
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_args: NS(multi_processor_count=80))
    target = Buffers(weights, 64, 64, prefill=True)
    target.staged = NS(synchronize=lambda: None, record=lambda: None)  # CPU copies complete synchronously.
    for state in (source, destination):
        bind_reference_inputs(weights, state, target)
        state.media_inputs_enabled = True
    from tensorfold.families.qwen4_exp.cuda import compact_indexer
    target.compact_indexer_stage = compact_indexer.Staging(64, 1, "cpu", mtp=False)
    target.target_indexer_coordinate_storage = target.compact_indexer_stage.coordinates
    target.target_indexer_position = weights.positions.target
    target.mtp_indexer_position = weights.positions.mtp
    catalog = make_retention_catalog(RetentionPolicy("ram", 2, 10, 1 << 30, 37, 74, 0),
        weights, [source, destination], prefix_limit=2, recovery_versions=2)
    catalog.configure_admission(max_waiting_input_bytes=1 << 20, max_pending_requests=2)
    class Encoder:
        def _retire_inputs(self):
            pass
        def encode_device_unit(self, values, grid):
            return NS(features=torch.full((int(grid.prod()) // 4, weights.cfg.hidden), 13., dtype=torch.float16))
    producer = MediaInputProducer.create(weights, encoder=Encoder(), encoder_rank=0, rows=64, patches=4,
        cache_rows=0, register=lambda _owner: None)
    def health():
        work.check_health()
        prepared.executor.check_health()
    owner = NS(is_media_runtime=True, identity=prepared.plan.profile_identity, work=work,
        executor=prepared.executor, profile=NS(cpu=prepared.executor.envelope, execution=NS(input_rows=64)),
        producer=producer, check_media_health=health, contract=contract)
    owner.requests = media_requests.MediaRequests(owner)
    decoder = decoder_for(catalog, monkeypatch)
    decoder.w, decoder.capacity, decoder.prefill_rows = weights, 64, 64
    decoder.pbuf, decoder.vision, decoder.media_runtime = target, owner, owner
    raw, limit, box = threading.Event(), RequestLimit(2), OutputMailbox(1)
    cancel = FrozenCancellation(raw)
    cancel.active = True
    stream = Stream(list(prepared.token_ids), 1, draft=False, vision=prepared, cancel=cancel, sid=4, st=source)
    stream.principal, stream.admission_ticket, stream.blocked_input = None, 17, "media"
    peer = object.__new__(TensorParallelDecoder)
    peer.decoder, peer.engine, peer.failure, peer.next_wire = decoder, NS(_abort_comm=False), None, 0
    wake = changed.set
    handle = None
    def ready():
        deadline = time.monotonic() + 5
        while not peer.prepare_request(stream, wake=wake):
            assert changed.wait(max(0., deadline - time.monotonic()))
            changed.clear()
    try:
        with limit.request():
            limit.borrow(box)
            stream.delivery_owner = DeliveryOwner(limit, box, raw, lambda _values: None, None)
            ready()
            epoch = stream._media_request.epoch
            end = prepared.plan.units[1].run.start
            media_mtp.bind_context(source, owner.identity, prepared.plan.media_identity)
            pieces = ((stream, 0, end),)
            media_requests.publish(owner, pieces)
            forward.forward(weights, source, target, stream.prompt[:end], logits=False)
            forward.commit(weights, source, target, end, end)
            media_requests.retire_batch(owner, pieces, cancelled=False)
            assert ahead.wait(5) and not producer.cached(prepared.plan, prepared.plan.units[0])
            engine = NS(stops=[])
            decoder.filling.append(stream)
            decoder.fills[stream.sid] = [engine, False, end, [], None]
            assert decoder.suspend_request(stream)
            handle = stream.retention_handle
            assert handle.transfer.retired.wait(5)
            catalog.progress()
            decoder._release_retained_states()
            assert not peer.prepare_request(stream, wake=wake)
            release.set()
            if bad_ahead:
                with pytest.raises(ImageInputError) as original:
                    ready()
                assert original.value is failure and work.record()["owned_units"] == 0
            else:
                ready()
                assert stream._media_request.epoch == epoch and calls == [0, 1]
                assert decoder.admit(stream) is False and stream.retention_request.restore_handle.transfer.retired.wait(5)
                catalog.progress()
                decoder.admit(stream)
                rest = ((stream, end, len(stream.prompt) - end),)
                media_requests.publish(owner, rest)
                forward.forward(weights, stream.st, target, stream.prompt[end:], logits=False)
                forward.commit(weights, stream.st, target, len(stream.prompt) - end, len(stream.prompt) - end)
                media_requests.retire_batch(owner, rest, cancelled=False)
                decoder.filling.clear()
                decoder.fills.clear()
            stream.done = True
            assert media_requests.retire(stream)
            assert decoder.retire_retained_request(stream)
            limit.release(box)
            stream.retirement_host.retire_backend()
            media_requests.release(stream)
        owner.requests.check_drained()
        assert prepared.replay._borrows == 0 and calls == [0, 1] and peer.failure is None
    finally:
        release.set()
        limit.release(box)
        if handle is not None and handle.internal_owner:
            if stream.retention_request is not None:
                stream.retention_request.close()
            catalog.release_temporary(handle)
        finish((lambda: work.close(timeout=5), producer.close, prepared.close, lambda: catalog.close(timeout=5)))

"""Original descriptor-word oracle and bounded whole-frame ownership controls."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.frame_descriptors import FrameDescriptors, envelope
from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.test_native_owner_adapter import _weights
from tests.frame_descriptor_controls import original_words


def prepared(count, rows_per_state=3, owner_rows=8, *, mtp=False):
    weights = _weights()
    if mtp:
        weights.mtp.layer = NS(index=12, linear=False)
    states = [State(weights, 128, 4, "int8", slot=index) for index in range(count)]
    owner = OwnerBuffers(weights, owner_rows, NS(idw=2052))
    owner.prepare_native(states)
    rows = count * rows_per_state
    buffers = NS(rows=rows, prefill=False, history_owner=owner, target_indexer_frame=None, mtp_indexer_frame=None)
    packet = FrameDescriptors(owner, 1 if mtp else 12, rows, mtp=mtp)
    owner.descriptor_frames = packet
    packet.bind(weights, buffers)
    segments = [(state, index * rows_per_state, (index + 1) * rows_per_state) for index, state in enumerate(states)]
    return weights, states, owner, packet, buffers, segments




@pytest.mark.parametrize("count,owner_rows", [(1, 4), (8, 8), (24, 32)])
@pytest.mark.parametrize("mtp", [False, True])
def test_one_packet_matches_original_descriptor_bytes_for_all_layers_states_and_chunks(count, owner_rows, mtp):
    weights, states, owner, packet, buffers, segments = prepared(count, owner_rows=owner_rows, mtp=mtp)
    layers = (weights.mtp.layer,) if mtp else weights.layers
    pointers = owner.descriptors.data_ptr(), owner.lane_rows.data_ptr(), owner.native_layer.data_ptr()
    for repeat in range(2):
        with packet.frame(segments, buffers):
            for layer in layers:
                for base in range(0, buffers.rows, owner.rows):
                    count_rows = min(owner.rows, buffers.rows - base)
                    bucket = owner.native_bucket(count_rows)
                    expected, lanes = original_words(owner, segments, layer, base, count_rows, bucket, mtp=mtp)
                    owner.stage_native_descriptors(segments, layer, base, count_rows, bucket_rows=bucket, mtp=mtp)
                    assert torch.equal(owner.descriptors[:bucket], expected)
                    assert torch.equal(owner.lane_rows[:bucket], lanes)
                    assert owner.native_layer.tolist() == [layer.index]
                    assert pointers == (owner.descriptors.data_ptr(), owner.lane_rows.data_ptr(), owner.native_layer.data_ptr())
        assert not packet.active and packet.submitted == repeat + 1
    assert packet.copied_layers == 2 * len(layers) * packet.chunks
    assert len(packet.registered) <= 2 * packet.layout["frame_rows"]
    owner.close()
    assert packet.closed and packet.device is packet.host is None
    assert all(packet not in state.history_native_readers for state in states)


def test_active_frame_blocks_reset_position_and_coordinate_restage_before_mutation():
    weights, states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    state = states[0]
    from tensorfold.families.qwen4_exp.cuda import compact_indexer
    with packet.frame(segments, buffers):
        before = state.slot_generation, state.pos, state.kv_status.clone()
        for operation in (lambda: state.reset(weights), lambda: state.reset(weights, new_sequence=False),
                          lambda: state.set_pos(1), lambda: state.set_rope_delta(1),
                          lambda: compact_indexer.stage_target_indexer_frame(buffers, segments, None)):
            with pytest.raises(RuntimeError, match="descriptor frame"):
                operation()
        assert (state.slot_generation, state.pos) == before[:2]
        assert torch.equal(state.kv_status, before[2])
        for layer in weights.layers:
            owner.stage_native_descriptors(segments, layer, 0, 3, bucket_rows=4)
    owner.close()


def test_original_storage_mismatch_fails_before_packet_or_device_table_publication():
    _weights0, states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    owner.descriptors.fill_(99)
    cache = next(cache for cache in states[0].kc if cache.k.numel())
    cache.k = cache.k.clone()
    with pytest.raises(RuntimeError, match="storage catalogue"):
        with packet.frame(segments, buffers):
            pytest.fail("invalid original storage acquired frame")
    assert bool((owner.descriptors == 99).all()) and packet.error is not None
    with pytest.raises(RuntimeError, match="failed"):
        with packet.frame(segments, buffers):
            pytest.fail("poisoned owner reused")
    owner.close()


def test_incomplete_frame_preserves_primary_and_poison_until_retirement():
    _weights0, _states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    primary = RuntimeError("operation")
    with pytest.raises(RuntimeError) as caught:
        with packet.frame(segments, buffers):
            raise primary
    assert caught.value is primary and packet.error is primary and not packet.active
    with pytest.raises(RuntimeError, match="failed"):
        with packet.frame(segments, buffers):
            pass
    owner.close()
    assert packet.error is primary


def test_metadata_envelope_scales_with_actual_accepted_bucket_catalogue():
    small = envelope(12, 72, 32, bucket_count=6)
    many = envelope(12, 72, 32, bucket_count=32)
    assert small["chunks"] == many["chunks"] == 3
    assert small["payload_bytes"] == many["payload_bytes"] == 12 * 3 * 4232
    assert many["host_metadata_bytes_bound"] > small["host_metadata_bytes_bound"]
    assert many["host_commitment_bytes"] == many["payload_bytes"] + many["host_metadata_bytes_bound"]


class Event:
    def __init__(self, *, wait_error=None, record_error=None):
        self.wait_error, self.record_error = wait_error, record_error
        self.waits = self.records = 0
    def synchronize(self):
        self.waits += 1
        if self.wait_error is not None:
            raise self.wait_error
    def record(self, stream):
        self.records += 1
        if self.record_error is not None:
            raise self.record_error


def test_failed_upload_wait_preserves_packet_and_blocks_all_future_submissions():
    _w, _states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    error = RuntimeError("host DMA uncertain")
    packet.upload, packet.pending_upload = Event(wait_error=error), True
    packet.device.fill_(17)
    before = packet.device.clone()
    with pytest.raises(RuntimeError) as caught:
        with packet.frame(segments, buffers):
            pytest.fail("uncertain DMA reused host bytes")
    assert caught.value is error and packet.error is error and torch.equal(packet.device, before)
    owner.close()  # CPU oracle has no asynchronous CUDA consumer.


def test_operation_and_completion_failure_preserve_primary_and_require_positive_fence(monkeypatch):
    _w, states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    primary, completion = RuntimeError("operation"), RuntimeError("completion unknown")
    packet.upload, packet.done = Event(), Event(record_error=completion)
    stream = NS(cuda_stream=7, wait_event=lambda event: None)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: stream)
    with pytest.raises(RuntimeError) as caught:
        with packet.frame(segments, buffers):
            raise primary
    assert caught.value is primary and primary.__cause__ is completion and packet.error is primary
    held = packet.device
    with pytest.raises(RuntimeError, match="global fence"):
        packet.retire(fenced=False)
    assert packet.device is held and packet in states[0].history_native_readers
    packet.retire(fenced=True)
    assert packet.error is primary and packet not in states[0].history_native_readers
    packet.done = None
    owner.close()


def test_changed_frame_or_rows_refuses_before_canonical_descriptor_overwrite():
    _w, _states, owner, packet, buffers, segments = prepared(1, owner_rows=4)
    with pytest.raises(RuntimeError, match="coordinate frame changed"):
        with packet.frame(segments, buffers):
            owner.descriptors.fill_(91)
            buffers.target_indexer_frame = object()
            owner.stage_native_descriptors(segments, packet.layers[0], 0, 3, bucket_rows=4)
    assert bool((owner.descriptors == 91).all())
    owner.close()

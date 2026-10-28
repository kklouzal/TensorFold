"""Real pinned whole-frame upload and original descriptor bytes on SM70."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.frame_descriptors import FrameDescriptors
from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.frame_descriptor_controls import original_words, weights as metadata_weights


@pytest.mark.parametrize("count,owner_rows", [(1, 4), (8, 8), (24, 32)])
@pytest.mark.parametrize("mtp", [False, True])
def test_real_sm70_packet_all_layers_states_chunks_and_cross_stream_retirement(count, owner_rows, mtp):
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires actual SM70 pinned source foundation")
    torch.cuda.set_device(0)
    weights = metadata_weights("cuda:0")
    if mtp:
        weights.mtp.layer = NS(index=12, linear=False)
    states = [State(weights, 128, 4, "int8", slot=index) for index in range(count)]
    owner = OwnerBuffers(weights, owner_rows, NS(idw=2052))
    owner.prepare_native(states)
    rows = count * 3
    packet = FrameDescriptors(owner, 1 if mtp else 12, rows, mtp=mtp)
    owner.descriptor_frames = packet
    buffers = NS(rows=rows, prefill=False, history_owner=owner, target_indexer_frame=None, mtp_indexer_frame=None)
    packet.bind(weights, buffers)
    segments = [(state, 3 * index, 3 * index + 3) for index, state in enumerate(states)]
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    try:
        with packet.frame(segments, buffers):
            for ordinal, layer in enumerate(packet.layers):
                for base in range(0, rows, owner.rows):
                    kept = min(owner.rows, rows - base)
                    bucket = owner.native_bucket(kept)
                    expected, lanes = original_words(owner, segments, layer, base, kept, bucket, mtp=mtp)
                    with torch.cuda.stream(streams[(ordinal + base) % 2]):
                        owner.stage_native_descriptors(segments, layer, base, kept, mtp=mtp, bucket_rows=bucket)
                        actual = owner.descriptors[:bucket].clone()
                        actual_lanes = owner.lane_rows[:bucket].clone()
                        actual_layer = owner.native_layer.clone()
                        # The synthetic last consumer stands in for the native
                        # pipeline; its event preserves the real reuse protocol.
                        owner.mark_native_submitted()
                    owner.wait_native_reuse()  # Join the synthetic consumer before the default-stream D2H oracle.
                    assert torch.equal(actual.cpu(), expected)
                    assert torch.equal(actual_lanes.cpu(), lanes)
                    assert actual_layer.cpu().tolist() == [layer.index]
        torch.cuda.synchronize()
        assert packet.submitted == 1 and packet.copied_layers == len(packet.layers) * packet.chunks
        assert packet.host.is_pinned() and not packet.active and packet.error is None
    finally:
        torch.cuda.synchronize()
        owner.close()
    assert packet.host is packet.device is None and packet.closed

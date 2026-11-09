"""Pinned descriptor publication on SM70; no model/performance qualification."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70.software_binding import issue_source_binding
from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.test_native_owner_adapter import _weights


@pytest.fixture(autouse=True)
def actual_target():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("requires actual pinned SM70; collection is not qualification")
    issue_source_binding(required=True)


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("base,count", [(0, 5), (1, 3), (3, 2)])
def test_pinned_words_match_original_rows_across_producer_streams(mtp, base, count):
    weights = _weights()
    weights.device = torch.device("cuda")
    weights.meta["attention_degree"] = 4
    states = [State(weights, 257, 4, "int8", slot=slot) for slot in range(2)]
    owner = OwnerBuffers(weights, 8, SimpleNamespace(idw=2052))
    try:
        owner.prepare_native(states, buckets=(8,))
        assert owner.descriptors_host.is_pinned()
        mirror = owner.descriptor_host_words
        assert mirror.ctypes.data == owner.descriptors_host.data_ptr()
        layer = SimpleNamespace(index=2)
        segments = [(states[0], 0, 3), (states[1], 3, 5)]
        expected = torch.zeros_like(owner.descriptors_host)
        lane = 0
        for state, first, last in segments:
            begin, end = max(base, first), min(base + count, last)
            if begin >= end:
                continue
            ai = state.history_placement.target_layers if mtp else state.att_index[layer.index]
            cache = state.mtp_kc if mtp else state.kc[ai]
            position = state.mtp_pos if mtp else state.pos_dev
            live = owner.lane_rows.data_ptr() + lane * owner.lane_rows.element_size()
            for row in range(begin, end):
                expected[row - base].copy_(torch.tensor((
                    cache.k.data_ptr(), cache.v.data_ptr(), cache.ks.data_ptr(), cache.vs.data_ptr(),
                    cache.page_offsets.data_ptr(), state.history_generations.data_ptr(), state.kv_status.data_ptr(),
                    position.data_ptr(), live, state.slot_generation, state.history_generation,
                    state.capacity, cache.capacity, cache.page_offsets.numel(), row - first,
                    state.history_placement.indexer_owner(state.history_slot, ai, include_mtp=True),
                ), dtype=torch.int64))
            lane += 1
        main = torch.cuda.current_stream()
        side = torch.cuda.Stream()
        side.wait_stream(main)
        for stream in (main, side):
            with torch.cuda.stream(stream):
                owner.stage_native_descriptors(segments, layer, base, count, mtp=mtp, bucket_rows=8)
            main.wait_event(owner.descriptor_upload)
            assert torch.equal(owner.descriptors.cpu(), expected)
            assert owner.descriptor_host_words is mirror
    finally:
        owner.close()

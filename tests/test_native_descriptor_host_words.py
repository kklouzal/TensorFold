"""CPU descriptor bits, shared mirror ownership and its publication frontier.

No CUDA pointer is dereferenced and this is not an integrated GPU performance test.
"""

import gc
from types import SimpleNamespace as NS
import weakref

import numpy as np
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.test_native_owner_adapter import _weights

pytestmark = pytest.mark.torch


def _workspace():
    weights = _weights()
    states = [State(weights, 257, 4, "int8", slot=slot) for slot in range(2)]
    owner = OwnerBuffers(weights, 8, NS(idw=2052))
    owner.prepare_native(states)
    return owner, states


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("base,count", [(0, 5), (1, 3), (3, 2)])
def test_view_matches_independent_tensor_rows_and_zero_tails(mtp, base, count, monkeypatch):
    owner, states = _workspace()
    state0, state1 = states
    state0.slot_generation, state0.history_generation = 2**63 - 1, 0
    state1.slot_generation, state1.history_generation = 0, 2**63 - 1
    layer = NS(index=2)
    segments = [(state0, 0, 3), (state1, 3, 5)]
    owner.descriptors_host.fill_(-91)
    mirror_pointer = owner.descriptors_host.data_ptr()
    view = owner.descriptor_host_words
    assert view.dtype == np.int64 and view.flags.c_contiguous and view.flags.writeable
    assert view.ctypes.data == mirror_pointer
    assert np.shares_memory(view, owner.descriptors_host.numpy())
    # The old row representation is the independent bit oracle. Compute it before
    # prohibiting hot torch.tensor construction in the affected operation.
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
    def no_tensor(*_args, **_kwargs):
        pytest.fail("hot descriptor row allocated a torch.tensor")
    monkeypatch.setattr(torch, "tensor", no_tensor)
    owner.stage_native_descriptors(segments, layer, base, count, mtp=mtp, bucket_rows=8)
    assert torch.equal(owner.descriptors_host, expected)
    assert torch.equal(owner.descriptors, expected)
    assert owner.descriptor_host_words is view
    assert owner.descriptors_host.data_ptr() == mirror_pointer
    owner.close()


def test_mirror_reuse_waits_before_host_or_device_mutation(monkeypatch):
    owner, states = _workspace()
    owner.descriptors_host.fill_(-29)
    owner.descriptors.fill_(-31)
    old_host, old_device = owner.descriptors_host.clone(), owner.descriptors.clone()
    order = []
    def dma_retire():
        assert torch.equal(owner.descriptors_host, old_host)
        assert torch.equal(owner.descriptors, old_device)
        order.append("dma")
    def readers():
        assert order == ["dma"]
        assert torch.equal(owner.descriptors_host, old_host)
        assert torch.equal(owner.descriptors, old_device)
        order.append("readers")
    owner._upload_pending = True
    owner.descriptor_upload = NS(synchronize=dma_retire, record=lambda *_: None)
    monkeypatch.setattr(owner, "wait_native_reuse", readers)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda *_: None)
    owner.stage_native_descriptors([(states[0], 0, 2)], NS(index=0), 0, 2, bucket_rows=8)
    assert order == ["dma", "readers"]
    owner.close()


def test_failed_dma_retirement_preserves_both_mirrors_and_view():
    owner, states = _workspace()
    owner.descriptors_host.fill_(-29)
    owner.descriptors.fill_(-31)
    old_host, old_device = owner.descriptors_host.clone(), owner.descriptors.clone()
    view = owner.descriptor_host_words
    primary = RuntimeError("DMA retirement uncertain")
    def fail():
        raise primary
    owner._upload_pending = True
    owner.descriptor_upload = NS(synchronize=fail)
    with pytest.raises(RuntimeError) as caught:
        owner.stage_native_descriptors([(states[0], 0, 2)], NS(index=0), 0, 2, bucket_rows=8)
    assert caught.value is primary
    assert owner.descriptor_host_words is view
    assert torch.equal(owner.descriptors_host, old_host)
    assert torch.equal(owner.descriptors, old_device)
    owner.close()


@pytest.mark.parametrize("fault", ["storage", "generation", "coverage"])
def test_all_spans_validate_before_wait_or_mirror_mutation(fault):
    owner, states = _workspace()
    owner.descriptors_host.fill_(-17)
    before = owner.descriptors_host.clone()
    def unexpected():
        pytest.fail("invalid spans reached the mirror retirement/mutation frontier")
    owner._upload_pending = True
    owner.descriptor_upload = NS(synchronize=unexpected)
    segments = [(states[0], 0, 2), (states[1], 2, 4)]
    if fault == "storage":
        states[1].kc[0].page_offsets = states[1].kc[0].page_offsets.clone()
        error = RuntimeError
    elif fault == "generation":
        states[1].history_generation = 2**63
        error = OverflowError
    else:
        segments[1] = (states[1], 3, 5)
        error = ValueError
    with pytest.raises(error):
        owner.stage_native_descriptors(segments, NS(index=0), 0, 4, bucket_rows=8)
    assert torch.equal(owner.descriptors_host, before)
    owner.close()


def test_view_remains_owned_on_failed_close_then_released_after_positive_retirement(monkeypatch):
    owner, _ = _workspace()
    view_ref = weakref.ref(owner.descriptor_host_words)
    mirror_ref = weakref.ref(owner.descriptors_host)
    original = owner.retire_native
    primary = RuntimeError("native readers not retired")
    def fail():
        raise primary
    monkeypatch.setattr(owner, "retire_native", fail)
    with pytest.raises(RuntimeError) as caught:
        owner.close()
    assert caught.value is primary
    assert not owner.closed and view_ref() is not None and mirror_ref() is not None
    monkeypatch.setattr(owner, "retire_native", original)
    owner.close()
    gc.collect()
    assert owner.closed and owner.descriptor_host_words is None
    assert view_ref() is None and mirror_ref() is None

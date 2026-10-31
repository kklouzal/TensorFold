"""Same-format full family execution across the GPU-history/RAM boundary."""

import pytest
import torch

from test_flashnext_sm70_family import _small_weights
from tensorfold.cuda.numa_memory import HostAllocator, configure_gpu_numa
from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda.forward import commit, forward
from tensorfold.families.qwen4_exp.cuda.indexer_history import HistoryPlan, HistorySession
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA archive family integration")


@pytest.mark.parametrize("cache", ["fp16", "int8", "rotorquant-signed128-4", "rotorquant-signed64-norm4"])
def test_lossless_archive_matches_full_gpu_history_through_prefill_rejection_and_resize(monkeypatch, cache):
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 archive family integration")
    monkeypatch.setattr(family_ops, "_NATIVE_FAMILY", True)
    weights = _small_weights(True, hidden=2560, lowrank=320, head_dim=256)
    baseline = State(weights, 64, 4, cache, limit=128)
    target = configure_gpu_numa(weights.device)
    plan = HistoryPlan(1, weights.cfg.index_dim, weights.cfg.index_ratio, 16, 4)
    allocator = HostAllocator(target, byte_limit=plan.host_bytes(128), pinned_limit=plan.pinned_bytes)
    session = HistorySession(plan, allocator, weights.device, maximum_owners=1)
    weights.meta.update(execution_provider="sm70", indexer_history="ram", indexer_history_session=session)
    candidate = State(weights, 64, 4, cache, limit=128)
    prompt = list(range(4, 35))
    try:
        for offset, count in ((0, 7), (7, 11), (18, 13)):
            outputs = []
            for state in (baseline, candidate):
                # The same immutable weights serve both explicit storage
                # policies; normal engine construction pins this per model.
                weights.meta["indexer_history"] = state.indexer_history_mode
                buffers = Buffers(weights, 16, state.capacity, prefill=True)
                outputs.append(forward(weights, state, buffers, prompt[offset:offset + count]).clone())
                commit(weights, state, buffers, count, count)
            assert torch.equal(*outputs)
            raw = candidate.indexer_history_owner.archive[0, :candidate.pos]
            assert torch.equal(raw.view(torch.int16), baseline.ikc[0][:baseline.pos].cpu().view(torch.int16))
            assert torch.equal(candidate.pooled[0], baseline.pooled[0])
        for tokens, keep in (([71, 33, 18], 1), ([29, 41], 2), ([61], 1)):
            outputs = []
            for state in (baseline, candidate):
                weights.meta["indexer_history"] = state.indexer_history_mode
                buffers = Buffers(weights, 4, state.capacity)
                outputs.append(forward(weights, state, buffers, tokens).clone())
                commit(weights, state, buffers, len(tokens), keep)
            assert torch.equal(*outputs)
            assert torch.equal(candidate.pooled[0][:candidate.pos // plan.ratio],
                               baseline.pooled[0][:baseline.pos // plan.ratio])
            assert torch.equal(candidate.rec[candidate.cur[0]], baseline.rec[baseline.cur[0]])
        for state in (baseline, candidate):
            state.resize(128)
        outputs = []
        for state in (baseline, candidate):
            weights.meta["indexer_history"] = state.indexer_history_mode
            buffers = Buffers(weights, 4, state.capacity)
            outputs.append(forward(weights, state, buffers, [92, 44, 17]).clone())
            commit(weights, state, buffers, 3, 3)
        assert torch.equal(*outputs)
        assert torch.equal(candidate.pooled[0], baseline.pooled[0])
        assert torch.equal(candidate.indexer_history_owner.archive[0, :candidate.pos].view(torch.int16),
                           baseline.ikc[0][:baseline.pos].cpu().view(torch.int16))
    finally:
        session.close()

"""Actual State/RAM-owner lifecycle and byte contracts on CPU, without NUMA/CUDA."""

from contextlib import contextmanager

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.indexer_history import HistoryPlan, HistorySession
from tensorfold.families.qwen4_exp.cuda.service_memory import cache_resize_overlap
from tensorfold.families.qwen4_exp.cuda.state import State
from tensorfold.families.qwen4_exp.cuda.state import KVNumericError
from tensorfold.families.qwen4_exp.kv_formats import get_pair
from tests.test_indexer_history import Allocator, data
from tests.test_sm70_rotorquant_state import model


@contextmanager
def runtime(*, mtp=True, maximum_owners=8):
    w = model(8, mtp=mtp)
    layers = sum(not layer.linear for layer in w.layers) + int(mtp)
    plan = HistoryPlan(layers, w.cfg.index_dim, w.cfg.index_ratio, 16, 4)
    session = HistorySession(plan, Allocator(), "cpu", maximum_owners=maximum_owners)
    w.meta.update(execution_provider="sm70", indexer_history="ram", indexer_history_session=session)
    try:
        yield w, plan, session
    finally:
        session.close()


def produce(state, rows, *, start=0):
    owner = state.indexer_history_owner
    frame = owner.begin("prefill", start, rows)
    for index in range(owner.plan.layers):
        offset = start - int(owner.origins_host[index])
        frame.stage(index)[offset:offset + rows].copy_(data(start, rows, index, owner.plan.index_dim))
    frame.enqueue()
    frame.mark_submitted()
    frame.validated(fenced=True)
    frame.commit(rows)
    state.set_pos(start + rows)
    state.set_mtp_len(start + rows if hasattr(state, "mtp_kc") else 0)


@pytest.mark.parametrize("mtp", [False, True])
@pytest.mark.parametrize("kv_dtype", ["fp16", "int8", "rotorquant-signed128-4"])
def test_ram_state_counts_bounded_device_owner_and_full_limit_host_archive(mtp, kv_dtype):
    with runtime(mtp=mtp) as (w, plan, session):
        state = State(w, 37, 4, kv_dtype, limit=128)
        pair = get_pair(kv_dtype)
        expected = plan.layers * (37 * pair.row_bytes(1, 256) + pair.dummy_bytes + 10 * 128 * 2)
        expected += plan.device_nbytes
        assert state.cache_bytes() == state.owned_cache_bytes() == expected
        assert state.host_cache_bytes() == plan.host_bytes(128)
        assert state.indexer_history_owner.limit == 128
        assert state.kv_status is not None and state._kv_peer_status.nbytes == 8 * 8
        assert all(handle.owner is state.indexer_history_owner for handle in state.ikc)
        before = tuple(t.data_ptr() for t in state.indexer_history_owner.device_tensors())
        host_before = state.host_cache_bytes()
        produce(state, 9)
        state.resize(73)
        assert state.cache_bytes() == state.owned_cache_bytes()
        assert tuple(t.data_ptr() for t in state.indexer_history_owner.device_tensors()) == before
        assert state.host_cache_bytes() == host_before
        assert cache_resize_overlap(state, 128) == state.layer_bytes(128) - plan.device_bytes_per_layer
        assert state.indexer_history_owner in session._owners


@pytest.mark.parametrize("mtp", [False, True])
def test_clone_historical_prefix_restore_and_reset_use_the_independent_full_archive(mtp):
    with runtime(mtp=mtp) as (w, plan, session):
        source = State(w, 37, 4, "fp16", limit=128)
        produce(source, 9)
        snapshot = source.snapshot()
        cloned = source.clone()
        assert cloned.kv_identity == source.kv_identity
        assert cloned.indexer_history_owner.archive.data_ptr() != source.indexer_history_owner.archive.data_ptr()
        assert torch.equal(cloned.indexer_history_owner.archive[:, :9], source.indexer_history_owner.archive[:, :9])
        assert all(handle.owner is cloned.indexer_history_owner for handle in cloned.ikc)
        if mtp:
            assert cloned.mtp_ikc.owner is cloned.indexer_history_owner
        # A later validated branch must not destroy the raw prefix needed by a
        # prior recurrent snapshot. Restore rebuilds the exact historical ring.
        produce(source, 8, start=9)
        source.restore(snapshot)
        for index in range(plan.layers):
            assert torch.equal(source.indexer_history_owner.ring[index, :9], data(0, 9, index, plan.index_dim))
        destination = State(w, 37, 4, "fp16", limit=128)
        # A reused destination can still carry longer main/MTP host cursors
        # while copy_prefix replaces its raw archive with a shorter lineage.
        produce(destination, 11)
        destination.copy_prefix(source, 7, 5 if mtp else 0)
        lengths = [7] * len(source.kc) + ([5] if mtp else [])
        cut = {**snapshot, "pos": 7, "mtp_len": 5 if mtp else 0,
               "indexer_history": source.indexer_history_owner.snapshot(lengths)}
        destination.restore(cut)
        for index, rows in enumerate(lengths):
            assert torch.equal(destination.indexer_history_owner.archive[index, :rows], data(0, rows, index, plan.index_dim))
        count, host_bytes = len(session._owners), destination.host_cache_bytes()
        destination.reset(w)
        assert (destination.pos, destination.mtp_len) == (0, 0)
        assert destination.host_cache_bytes() == host_bytes and len(session._owners) == count


def test_ram_and_gpu_snapshot_and_prefix_identities_cannot_cross_before_mutation():
    with runtime() as (w, _, _):
        ram = State(w, 37, 4, "fp16", limit=128)
        produce(ram, 9)
        w.meta["indexer_history"] = "gpu"
        gpu = State(w, 37, 4, "fp16", limit=128)
        before = gpu.rec.clone()
        with pytest.raises(ValueError, match="matching ordered formats"):
            gpu.restore(ram.snapshot())
        with pytest.raises(ValueError, match="matching cache formats"):
            gpu.copy_prefix(ram, 7, 7)
        assert torch.equal(gpu.rec, before) and gpu.pos == 0


def test_closed_history_fails_existing_state_check_and_unvalidated_snapshot_is_rejected():
    with runtime() as (w, _, _):
        state = State(w, 37, 4, "fp16", limit=128)
        snapshot = state.snapshot()
        before = state.rec.clone()
        snapshot["indexer_history"] = {"invalid": True}
        with pytest.raises(ValueError, match="snapshot"):
            state.restore(snapshot)
        assert torch.equal(state.rec, before) and state.pos == 0
        state.indexer_history_owner.close()
        with pytest.raises(RuntimeError, match="closed"):
            state.kv_check()


def pending(state, rows):
    owner = state.indexer_history_owner
    frame = owner.begin("prefill", 0, rows)
    for index in range(owner.plan.layers):
        frame.stage(index)[:rows].copy_(data(0, rows, index, owner.plan.index_dim))
    frame.enqueue()
    frame.mark_submitted()
    state.kv_begin()
    return owner


def test_existing_numeric_readback_validates_archive_before_setters_publish_independent_lengths():
    with runtime() as (w, plan, _):
        state = State(w, 37, 4, "fp16", limit=128)
        owner = pending(state, 9)
        assert owner._available == owner._published == [0] * plan.layers
        state.kv_check_words(0, -2)  # same status-consumption boundary as the TP cohort
        assert owner._available == [9] * plan.layers and owner._published == [0] * plan.layers
        state.set_pos(7)
        state.set_mtp_len(5)
        assert owner._published == [7] * len(state.kc) + [5]
        # A cut inside validated rows can precede the main cursor publication.
        assert state.indexer_snapshot_metadata(9, 8)["indexer_history"]["lengths"] == tuple([9] * len(state.kc) + [8])


def test_numeric_failure_never_archives_or_publishes_and_cold_reset_retires_pending_transfer():
    with runtime() as (w, plan, _):
        state = State(w, 37, 4, "fp16", limit=128)
        owner = pending(state, 9)
        with pytest.raises(KVNumericError):
            state.kv_check_words(16, 0)
        assert owner._available == owner._published == [0] * plan.layers
        with pytest.raises(KVNumericError):
            state.set_pos(7)
        state.reset(w)
        assert not owner._outstanding and state._kv_error is None and (state.pos, state.mtp_len) == (0, 0)


def test_archive_publication_failure_is_sticky_until_explicit_reset(monkeypatch):
    with runtime() as (w, _, _):
        state = State(w, 37, 4, "fp16", limit=128)
        owner = pending(state, 9)
        error = RuntimeError("controlled archive publication failure")

        def fail(**kwargs):
            raise error

        monkeypatch.setattr(owner, "validated", fail)
        with pytest.raises(RuntimeError, match="controlled archive"):
            state.kv_check_words(0, -2)
        assert state._kv_error is error and (state.pos, state.mtp_len) == (0, 0)
        with pytest.raises(RuntimeError, match="controlled archive"):
            state.set_mtp_len(1)
        state.reset(w)
        assert state._kv_error is None and not owner._outstanding

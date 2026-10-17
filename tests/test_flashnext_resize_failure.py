"""Failed layer-at-a-time growth preserves ownership, accounting and explicit slot retirement."""

import importlib
from types import SimpleNamespace
import weakref

import pytest

from tests.test_cuda_geometry import allocations as allocations
from tests.test_cuda_growing_caches import filled, weights

pytestmark = pytest.mark.torch


def attached_bytes(st):
    """Independent physical-storage sum, including both sides' scales and the MTP layer."""
    tensors = []
    for cache in [*st.kc, st.mtp_kc]:
        tensors.extend((cache.k, cache.v, cache.ks, cache.vs))
    tensors.extend([*st.ikc, *st.pooled, st.mtp_ikc, st.mtp_pooled])
    return sum(t.untyped_storage().nbytes() for t in tensors)


@pytest.mark.parametrize("dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("failed_row", range(1, 7))
def test_partial_layers_remain_charged_and_cannot_be_reset_or_published(allocations, monkeypatch, dtype, failed_row):
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    st = filled(state, torch, dtype)
    before, version = attached_bytes(st), st.version
    failure = MemoryError("controlled index/pool allocation failure")
    row, temporary = state._rows, []
    calls = 0

    def failed(array, rows, keep):
        nonlocal calls
        calls += 1
        if calls == failed_row:
            uncommitted = torch.zeros(16)
            temporary.append(weakref.ref(uncommitted))
            raise failure
        return row(array, rows, keep)

    with monkeypatch.context() as patch:
        patch.setattr(state, "_rows", failed)
        with pytest.raises(MemoryError) as caught:
            st.resize(512)
    assert caught.value is failure and st._resize_error is failure
    completed = (failed_row - 1) // 2
    expected = before + completed * (st.layer_bytes(512) - st.layer_bytes(256))
    assert st.capacity == 256 and st.version > version
    assert st.cache_bytes() == attached_bytes(st) == expected
    assert all(item() is None for item in temporary), "failed helper frames retained unaccounted replacement storage"
    for operation in (st.kv_begin, st.kv_check, st.kv_snapshot_metadata, st.clone,
                      lambda: st.kv_check_words(0, -2), lambda: st.ensure(257), lambda: st.reset(weights())):
        with pytest.raises(state.KVResizeError) as poisoned:
            operation()
        assert poisoned.value.__cause__ is failure
    delta = st.recover_resize(weights(), 256)
    assert delta == before - expected and st.cache_bytes() == attached_bytes(st) == before
    assert st._resize_error is st._kv_error is None and st.pos == st.mtp_len == 0
    assert all(torch.count_nonzero(cache.k) == 0 and torch.count_nonzero(cache.v) == 0
               for cache in [*st.kc, st.mtp_kc])
    assert st.ensure(300) > 0


def test_failed_recovery_stays_poisoned_and_reports_its_actual_attached_storage(allocations, monkeypatch):
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    st = filled(state, torch, "int4")
    original, victim = state.kvcache.KVCache.resized, st.kc[1]
    failure = OSError("controlled second-layer replacement failure")

    def fail(cache, *args):
        if cache is victim:
            raise failure
        return original(cache, *args)

    with monkeypatch.context() as patch:
        patch.setattr(state.kvcache.KVCache, "resized", fail)
        with pytest.raises(OSError, match="second-layer"):
            st.resize(512)
    assert st.cache_bytes() == attached_bytes(st) > st.cache_bytes(256)
    before = attached_bytes(st)
    reset = st.reset
    cleanup = RuntimeError("controlled final reset failure")
    with monkeypatch.context() as patch:
        patch.setattr(st, "reset", lambda w: (_ for _ in ()).throw(cleanup))
        with pytest.raises(RuntimeError) as caught:
            st.recover_resize(weights(), 256)
    assert caught.value is cleanup and st._resize_error is cleanup
    assert st.cache_bytes() == attached_bytes(st) < before
    with pytest.raises(state.KVResizeError):
        reset(weights())
    st.recover_resize(weights(), 256)
    assert st._resize_error is st._kv_error is None


def test_failed_stream_fence_keeps_all_attached_owners_before_retry(allocations, monkeypatch):
    import torch

    state = importlib.import_module("tensorfold.families.qwen4_exp.cuda.state")
    st = filled(state, torch, "bf16")
    st._poison_resize(MemoryError("controlled grow failure"))
    before = attached_bytes(st)
    pointers = [cache.k.data_ptr() for cache in [*st.kc, st.mtp_kc]]
    real_pos = st.pos_dev
    st.pos_dev = SimpleNamespace(is_cuda=True, device="cuda:0")
    fence = RuntimeError("controlled stream fence failure")

    def fail():
        raise fence

    with monkeypatch.context() as patch:
        patch.setattr(torch.cuda, "current_stream", lambda device: SimpleNamespace(synchronize=fail))
        with pytest.raises(RuntimeError) as caught:
            st.recover_resize(weights(), 256)
    assert caught.value is fence and st._resize_error is fence
    assert st.cache_bytes() == before and [cache.k.data_ptr() for cache in [*st.kc, st.mtp_kc]] == pointers
    st.pos_dev = real_pos
    st.recover_resize(weights(), 256)
    assert st._resize_error is None

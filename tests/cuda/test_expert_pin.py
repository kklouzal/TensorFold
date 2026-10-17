"""Actual CUDA host registration, direct-copy exactness and source retirement."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.expert_cache import HostExpertCache, entry_bytes_for
from tensorfold.cuda.expert_pin import ExpertPinPolicy, _native


def test_registered_authority_direct_dma_orders_cross_stream_eviction_and_unpins_after_close():
    shape = (7, 17)
    sizes = (7 * 17 * 4,) * 2
    policy = ExpertPinPolicy(sum(sizes), mode="on", allocation_bytes=sizes)
    cache = HostExpertCache(160, 160, "cuda", pin_policy=policy)
    try:
        source = tuple(cache.allocate_authority(shape, dtype=torch.int32, tag=f"field{i}") for i in range(2))
        for i, tensor in enumerate(source):
            tensor.copy_(torch.arange(tensor.numel(), dtype=torch.int32).reshape(shape) + 1000 * i)
        originals = tuple(t.clone() for t in source)
        assert all(t.is_pinned() for t in source)
        assert entry_bytes_for(source) == cache.entry_bytes
        cache.register(0, source)
        assert cache._layers[0].pinned == (True, True)
        assert not cache._layers[0].staging_required
        streams = [torch.cuda.Stream(), torch.cuda.Stream()]
        results = []
        for step in range(60):
            expert = step % shape[0]
            with torch.cuda.stream(streams[step % 2]), cache.lease(0, [expert]) as (views, mapping):
                results.append((expert, tuple(t[mapping[expert]].clone() for t in views)))
        cache.close()
        assert not any(cache._stage_pending)
        assert all(not t.is_pinned() for t in source)
        assert policy.receipt()["closed"]
        assert cache.copied_bytes == 60 * 2 * shape[1] * 4
        for expert, values in results:
            assert all(torch.equal(t.cpu(), expected[expert]) for t, expected in zip(values, originals))
    finally:
        cache.close()


def test_partial_startup_registration_rollback_retains_usable_cpu_bytes():
    policy = ExpertPinPolicy(96, mode="on", allocation_bytes=(32, 64))
    cache = HostExpertCache(64, 64, "cuda", pin_policy=policy)
    try:
        first = cache.allocate_authority((8,), dtype=torch.int32)
        first.fill_(17)
        assert first.is_pinned()
        # Model/header failure before cache.register must still retire the first allocation.
        with pytest.raises(ValueError, match="geometry"):
            cache.allocate_authority((4,), dtype=torch.int32)
    finally:
        cache.close()
    assert not first.is_pinned() and first.tolist() == [17] * 8


@pytest.mark.parametrize("cb,widths", [("3inst", ((4, 6, 4), (2, 4, 8), (8, 8, 8))),
                                      ("mcg", ((4, 4, 4), (6, 4, 2), (8, 8, 8))),
                                      ("mul1", ((3, 5, 7), (6, 4, 2), (8, 8, 8)))])
def test_registered_compact_exl3_waves_keep_native_float_bits(cb, widths):
    from test_exl3_host_cache_cuda import inputs, make_cached, resident
    from test_exl3_host_experts import PREFIX, SHARED, Checkpoint

    from tensorfold.cuda.exl3 import experts as native
    from tensorfold.cuda.exl3 import host_experts as host

    pk = Checkpoint(cb=cb, widths=widths)
    authority, _ = host.load_compact(pk, PREFIX, pk.count, SHARED)
    size = authority.payload_bytes
    policy = ExpertPinPolicy(size, mode="on", allocation_bytes=(size,))
    entry = authority.max_entry_bytes
    cache = host.Exl3HostExpertCache(entry, entry, "cuda", pin_policy=policy)
    try:
        _, _, cached = make_cached(pk, capacity=1, cache=cache)
        payload = cache._layers[0].authority.data
        assert payload.is_pinned() and torch.equal(payload, authority.data)
        cache.finish_loading()
        base = resident(pk)
        x, picks = inputs(pk)
        reference_scratch = native.Scratch(base, len(x), picks.shape[1])
        actual_scratch = native.Scratch(cached, len(x), picks.shape[1])
        reference_scratch.y.fill_(7.25)
        actual_scratch.y.fill_(7.25)
        expected = native.routed(x, picks, None, base, reference_scratch, None, len(x),
                                 act_mode=native.ACT_F32).clone()
        for _ in range(3):
            actual = host.routed_cached(x, picks, cached, actual_scratch, len(x)).clone()
            assert torch.equal(actual.view(torch.int32), expected.view(torch.int32))
        cache.close()
        assert not payload.is_pinned() and not any(cache._stage_pending)
    finally:
        cache.close()


def test_native_registration_rejects_invalid_or_reused_owners():
    native = _native()
    with pytest.raises(RuntimeError, match="CPU byte vector"):
        native.Registration(torch.empty(4096, dtype=torch.int32))
    policy = ExpertPinPolicy(8, mode="on", allocation_bytes=(8,))
    try:
        policy.empty((2,), dtype=torch.int32)
        with pytest.raises(RuntimeError, match="attempted"):
            policy._pins[0].pin()
    finally:
        policy.close()
    assert policy.receipt()["closed"]

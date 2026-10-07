"""Real CUDA copies, staging reuse, lease ordering and failure containment."""

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.expert_cache import CachedExperts, HostExpertCache, entry_bytes_for  # noqa: E402


def _source(layer=0):
    first = torch.arange(7 * 24, dtype=torch.int32).view(7, 3, 8) + layer * 1000
    second = torch.arange(7 * 12, dtype=torch.int32).view(7, 3, 4) - layer * 1000
    return first, second


def _cache(capacity=2, staging=2):
    source = _source()
    entry = entry_bytes_for(source)
    return HostExpertCache(capacity * entry + 1, entry, "cuda", staging_slots=staging)


def test_global_bound_contiguous_payloads_and_layer_identity():
    source, other = _source(), _source(1)
    cache = _cache()
    try:
        cache.register(0, source)
        cache.register(1, other)
        assert cache.gpu_bytes <= cache.budget_bytes and cache.gpu_bytes == cache.capacity * cache.entry_bytes
        assert cache.pinned_bytes == 2 * cache.entry_bytes
        assert cache.host_bytes == sum(t.numel() * t.element_size() for t in (*source, *other))
        assert cache.score_history_entries == 14
        for layer, tensors, ids in [(0, source, [0, 6]), (1, other, [0]), (0, source, [6])]:
            with cache.lease(layer, ids) as (payloads, mapping):
                assert all(t.is_contiguous() for t in payloads)
                for expected, actual in zip(tensors, payloads):
                    for expert in ids:
                        assert torch.equal(actual[mapping[expert]].cpu(), expected[expert])
        assert len(cache._policy.resident) <= cache.capacity
    finally:
        cache.close()
    cache.close()
    with pytest.raises(RuntimeError, match="closed"):
        with cache.lease(0, []):
            pass


def test_staging_ring_and_cross_stream_eviction_preserve_queued_consumers():
    source = _source()
    cache = _cache(1, 1)
    cache.register(0, source)
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    outputs = []
    try:
        for step in range(80):
            expert = step % 7
            with torch.cuda.stream(streams[step % 2]), cache.lease(0, [expert]) as (payloads, mapping):
                # Clone is the actual asynchronous consumer. Evict on the other
                # stream immediately; the next lease must wait for this read.
                outputs.append((expert, [tensor[mapping[expert]].clone() for tensor in payloads]))
        cache.close()
        for expert, payloads in outputs:
            assert all(torch.equal(actual.cpu(), expected[expert]) for actual, expected in zip(payloads, source))
        assert cache.misses == 80 and cache.copied_bytes == 80 * cache.entry_bytes
    finally:
        cache.close()


@pytest.mark.parametrize("ids", [[0, 0], [True], [-1], [7], [0, 1, 2], (0,)])
def test_invalid_and_nested_leases_leave_cache_valid(ids):
    cache = _cache()
    cache.register(0, _source())
    try:
        with pytest.raises(ValueError):
            with cache.lease(0, ids):
                pass
        with pytest.raises(ValueError, match="unknown"):
            with cache.lease(1, []):
                pass
        with cache.lease(0, []) as (_, mapping):
            assert mapping == {}
        with cache.lease(0, [0]):
            with pytest.raises(RuntimeError, match="nested"):
                with cache.lease(0, [1]):
                    pass
        assert cache.hits == 0 and cache.misses == 1
    finally:
        cache.close()


def test_source_mutation_and_layout_changes_fail_explicitly():
    cache = _cache()
    source = _source()
    cache.register(0, source)
    try:
        with pytest.raises(ValueError, match="already"):
            cache.register(0, source)
        with pytest.raises(ValueError, match="layout"):
            cache.register(1, (torch.ones(7, 1, dtype=torch.int32),))
        source[0].add_(1)
        with pytest.raises(ValueError, match="mutated"):
            with cache.lease(0, [0]):
                pass
    finally:
        cache.close()


def test_inference_sources_and_consumer_exception_preserve_valid_cache():
    with torch.inference_mode():
        cache = _cache()
        source = _source()
        cache.register(0, source)
    try:
        with pytest.raises(RuntimeError, match="consumer"):
            with cache.lease(0, [0]) as (payloads, mapping):
                assert all(not torch.is_inference(tensor) for tensor in payloads)
                result = payloads[0][mapping[0]].clone()
                raise RuntimeError("consumer failed")
        with cache.lease(0, [0]) as (payloads, mapping):
            assert torch.equal(payloads[0][mapping[0]].cpu(), source[0][0])
        assert torch.equal(result.cpu(), source[0][0])
        assert cache.hits == 1 and cache.misses == 1
    finally:
        cache.close()


def test_cleanup_failure_preserves_primary_consumer_error(monkeypatch):
    cache = _cache()
    cache.register(0, _source())

    class FailedRecord:
        def record(self, _):
            raise RuntimeError("injected event record failure")

    monkeypatch.setattr(cache, "_last_use", FailedRecord())
    try:
        with pytest.raises(ValueError, match="primary consumer") as caught:
            with cache.lease(0, [0]):
                raise ValueError("primary consumer failure")
        assert isinstance(caught.value.__cause__, RuntimeError)
        assert "event record" in str(caught.value.__cause__)
        assert cache._active is False and cache._failure is not None
    finally:
        cache.close()


def test_copy_failure_contains_cache_and_releases_lease_lock(monkeypatch):
    cache = _cache()
    cache.register(0, _source())

    def failure(*_):
        raise RuntimeError("injected copy failure")

    monkeypatch.setattr(cache, "_copy", failure)
    try:
        with pytest.raises(RuntimeError, match="injected copy"):
            with cache.lease(0, [0]):
                pass
        assert cache._active is False
        with pytest.raises(RuntimeError, match="unusable"):
            with cache.lease(0, [0]):
                pass
    finally:
        cache.close()


def test_lfu_and_lru_native_expert_replay_preserve_outputs_with_fewer_transfers(record_testsuite_property):
    from tensorfold.cuda import experts as grouped

    generator = torch.Generator().manual_seed(83)
    count, width = 18, 64

    def projection():
        words = torch.randint(-(2**31), 2**31 - 1, (count, width, width // 8),
                              generator=generator, dtype=torch.int32)
        scales = (torch.rand((count, width, width // 32), generator=generator) * 0.02 + 0.001).to(torch.bfloat16)
        biases = (torch.rand((count, width, width // 32), generator=generator) * 0.02 - 0.01).to(torch.bfloat16)
        return tuple(tensor.cuda() for tensor in (words, scales, biases))

    resident = grouped.make([projection(), projection()], projection(), 32)
    host = grouped.Experts(resident.up.cpu(), resident.down.cpu(), 32, width, width)
    plan = grouped.Plan(1, 1, count, "cuda")
    x = torch.randn((1, width), generator=generator).to(device="cuda", dtype=torch.bfloat16)
    activation = torch.empty((1, width), device="cuda", dtype=torch.bfloat16)
    output = torch.empty((1, width), device="cuda", dtype=torch.float32)
    picks = torch.empty((1, 1), device="cuda", dtype=torch.int32)

    def compute(expert, weights, physical=None):
        picks.fill_(expert)
        grouped.route(picks, plan)
        if physical is not None:
            plan.items[0, 0].fill_(physical)
        grouped.gate_up(x, weights, plan, activation, 1)
        grouped.down(activation, weights, plan, output, 1)
        return output.clone()

    oracle = torch.cat([compute(expert, resident) for expert in range(count)])
    trace = [0, 1, 0, 1, *range(2, count)] * 30
    totals = {}
    for name in ("lfu", "lru"):
        cache = HostExpertCache(4 * host.bytes_per_expert(), host.bytes_per_expert(), "cuda")
        cached = CachedExperts(host, cache, 0)
        if name == "lru":
            # Independent comparison policy lives only in the test. Empty
            # cells first, then oldest recency, with physical index tie-break.
            def victim(protected):
                available = [slot for slot in range(cache.capacity) if slot not in protected]
                return min(available, key=lambda slot: (cache._policy.keys[slot] is not None,
                                                       cache._policy.recency[slot], slot))

            cache._policy.victim = victim
        results = []
        try:
            for expert in trace:
                with cached.lease([expert]) as (hot, mapping):
                    results.append(compute(expert, hot, mapping[expert]))
            actual = torch.cat(results)
            expected = oracle[torch.tensor(trace, device="cuda")]
            assert torch.equal(actual, expected)
            totals[name] = {"misses": cache.misses, "hits": cache.hits, "copied_bytes": cache.copied_bytes}
        finally:
            cache.close()
    assert totals["lfu"]["misses"] < totals["lru"]["misses"]
    assert totals["lfu"]["copied_bytes"] < totals["lru"]["copied_bytes"]
    record_testsuite_property("controlled_native_expert_replay", json.dumps(totals, sort_keys=True))

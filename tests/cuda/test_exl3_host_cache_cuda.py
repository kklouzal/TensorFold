"""Real compact EXL3 upload, original-kernel parity and lease lifetime gates."""

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_exl3_host_experts import Checkpoint, PREFIX, SHARED  # noqa: E402
from tensorfold.cuda.exl3 import experts as native  # noqa: E402
from tensorfold.cuda.exl3 import host_experts as host  # noqa: E402


def make_cached(pk=None, capacity=2, layer=0, cache=None):
    pk = pk or Checkpoint()
    entry = max(sum(ks) * pk.dims * pk.width // 16 for ks in pk.widths)
    cache = cache or host.Exl3HostExpertCache(capacity * entry + 15, entry, "cuda")
    cached = host.load_cached(pk, PREFIX, pk.count, SHARED, cache, layer, cache.device)
    return pk, cache, cached


def resident(pk):
    return native.prepare(*pk.matrices("cuda"), pk.cb)


def inputs(pk, rows=7):
    gen = torch.Generator().manual_seed(101)
    x = torch.randn((rows, pk.dims), generator=gen).to(dtype=torch.bfloat16, device="cuda")
    # Includes shared, repeated logical IDs across rows and invalid native picks.
    # IDs within each row are distinct, as required by native grouped members.
    # Every row is active and the union deliberately exceeds two cells.
    picks = torch.tensor([[i % 2, 1 - (i % 2), 2, -1] for i in range(rows)], dtype=torch.int32, device="cuda")
    return x, picks


@pytest.mark.parametrize("cb,widths", [
    ("3inst", ((4, 6, 4), (2, 4, 8), (8, 8, 8))),
    ("mcg", ((4, 4, 4), (6, 4, 2), (8, 8, 8))),
    ("mul1", ((3, 5, 7), (6, 4, 2), (8, 8, 8))),
])
@pytest.mark.parametrize("capacity", [1, 2, 3])
def test_waved_act_f32_exact_native_outputs_all_codebooks_widths_and_cells(cb, widths, capacity):
    pk, cache, cached = make_cached(Checkpoint(cb=cb, widths=widths), capacity)
    base = resident(pk)
    x, picks = inputs(pk)
    s_base, s_hot = [native.Scratch(ex, x.shape[0], picks.shape[1]) for ex in (base, cached)]
    # A skipped row/slot must retain the caller's exact prior value through waves.
    s_base.y.fill_(7.25)
    s_hot.y.fill_(7.25)
    reference = native.routed(x, picks, None, base, s_base, None, len(x), act_mode=native.ACT_F32).clone()
    try:
        cache.finish_loading()
        assert cache._header_keys is None and cache._sealed
        for inference in (False, True, False):
            with torch.inference_mode(inference):
                actual = host.routed_cached(x, picks, cached, s_hot, len(x)).clone()
            assert torch.equal(actual.view(torch.int32), reference.view(torch.int32))
        assert not torch.is_inference(cache._publication_device)
        assert not torch.is_inference(s_hot.host_waves.device_pick)
        assert s_hot.host_waves.device_bytes == picks.numel() * 4
        assert s_hot.host_waves.host_bytes == picks.numel() * 8
        assert cache.capacity == capacity and cache.gpu_bytes == capacity * cache.entry_bytes
        assert len(cache._policy.resident) <= capacity
        assert torch.isfinite(actual).all()
        # Existing consuming combine applies the original weights exactly once.
        weights = torch.tensor([[.2, .3, 1., 0.]] * len(x), device="cuda")
        expected = (reference.view(len(x), picks.shape[1], -1) * weights[..., None]).sum(1)
        combined = (actual.view(len(x), picks.shape[1], -1) * weights[..., None]).sum(1)
        assert torch.equal(combined, expected)
    finally:
        cache.close()


@pytest.mark.parametrize("signs", [False, True])
def test_publication_points_at_original_compact_projection_bytes_and_copy_stats(signs):
    from tensorfold.families.qwen4_exp.cuda.weight_types import Weights

    pk, cache, cached = make_cached(Checkpoint(signs=signs), 2)
    weights = Weights(None, (cached, cached), [], None, None, torch.empty(0, device="cuda"))
    logical_bytes = cached.count * (36 + 6 * (cached.dims + cached.width))
    try:
        authority = cache._layers[0].authority
        assert cache.host_payload_bytes == authority.payload_bytes
        assert cache.metadata_host_bytes == 28 * cached.count
        assert cache.host_bytes == authority.payload_bytes + authority.metadata_bytes
        assert cache.control_device_bytes == cache._publication_device.nbytes == 32 * cache.capacity
        assert cache.control_host_bytes == cache._publication_host.nbytes == 64 * cache.capacity
        assert cache.pinned_bytes == cache._staging.nbytes == 2 * cache.entry_bytes
        assert cache.score_history_entries == cached.count
        # Shared pool and duplicate appearances in model fields count once;
        # pageable authority and pinned controls do not enter device accounting.
        assert weights.nbytes() == cache.gpu_bytes + cache.control_device_bytes + logical_bytes
        cache.finish_loading()
        for ids in ([0, 2], [0, 2], [1], [2]):
            with cached.lease(list(ids)) as tables:
                pointer_tables = [t.cpu().tolist() for t in (tables.gate_ptr, tables.up_ptr, tables.down_ptr)]
                for expert in ids:
                    slot = cache._policy.resident[(0, expert)]
                    cell = cache._cells[slot].cpu()
                    start = 0
                    for j in range(3):
                        original = authority.projection(expert, j).view(torch.uint8).reshape(-1)
                        assert torch.equal(cell[start:start + original.numel()], original)
                        assert pointer_tables[j][expert] == cache._pool.data_ptr() + slot * cache.entry_bytes + start
                        start += original.numel()
        assert cache.hits == 3 and cache.misses == 3
        assert cache.copied_bytes == sum(int(authority.trellis_bytes[i]) for i in (0, 2, 1))
        # A sealed loader must not grow or change the authority registry.
        with pytest.raises(RuntimeError, match="sealed"):
            cache.register(1, authority)
    finally:
        cache.close()
    cache.close()
    assert cache._publication_host is None and cache._publication_device is None and cache._cells is None
    assert list(cached.device_tensors()) == [cached._tables]
    assert weights.nbytes() == logical_bytes


def test_cross_stream_wave_evictions_keep_native_consumers_and_layer_identity():
    first, cache, a = make_cached(capacity=1)
    other = Checkpoint(seed=211)
    # Isolated layers share one authority/cache lifetime. The real loader uses
    # one reader with distinct prefixes; this fixture supplies a second layer's
    # original bytes directly so its random stream stays independently seeded.
    authority, tables = host.load_compact(other, PREFIX, other.count, SHARED, device=cache.device)
    b = host.CachedExl3Experts(authority, tables, cache, 1)
    bases = [resident(pk) for pk in (first, other)]
    x, _ = inputs(first, rows=1)
    scratches = [native.Scratch(ex, 1, 1) for ex in (a, b)]
    oracle = []
    for base in bases:
        s = native.Scratch(base, 1, 1)
        oracle.append([native.routed(x, torch.tensor([[i]], dtype=torch.int32, device="cuda"), None,
                                     base, s, None, 1).clone() for i in range(3)])
    streams = [torch.cuda.Stream(), torch.cuda.Stream()]
    for stream in streams:
        stream.wait_stream(torch.cuda.current_stream())
    results = []
    try:
        cache.finish_loading()
        for step in range(24):
            layer, expert = step % 2, step % 3
            with torch.cuda.stream(streams[layer]):
                pick = torch.tensor([[expert]], dtype=torch.int32, device="cuda")
                result = host.routed_cached(x, pick, (a, b)[layer], scratches[layer], 1).clone()
                results.append((layer, expert, result))
        cache.close()
        for stream in streams:
            stream.synchronize()
        for layer, expert, actual in results:
            assert torch.equal(actual, oracle[layer][expert])
        assert cache.misses == 24 and cache.hits == 0
        assert cache.gpu_bytes == cache.entry_bytes
        expected = sum(int((a, b)[layer].trellis_bytes[expert]) for layer, expert, _ in results)
        assert cache.copied_bytes == expected
    finally:
        cache.close()


def test_cache_rejects_second_checkpoint_reader_without_loading_payload():
    _, cache, _ = make_cached()
    other = Checkpoint(seed=211)
    try:
        with pytest.raises(ValueError, match="one checkpoint reader"):
            host.load_cached(other, PREFIX, other.count, SHARED, cache, 1, cache.device)
        assert other.reads == []
    finally:
        cache.close()


@pytest.mark.parametrize("mutation", ["payload", "starts", "widths", "counts", "scales", "gpu_widths"])
def test_authority_and_logical_metadata_mutation_fail_before_new_upload(mutation):
    _, cache, cached = make_cached()
    try:
        authority = cache._layers[0].authority
        target = {"payload": authority.data, "starts": authority.starts, "widths": authority.k2,
                  "counts": authority.trellis_bytes, "scales": cached._tables.suh_g,
                  "gpu_widths": cached._tables.gate_k2}[mutation]
        target.view(-1)[0].add_(1)
        with pytest.raises(ValueError, match="mutated"):
            with cached.lease([0]):
                pass
        assert cache.misses == 0 and cache.copied_bytes == 0
    finally:
        cache.close()


@pytest.mark.parametrize("point", ["copy", "publication"])
def test_copy_or_publication_failure_poison_cache_and_release_lease(monkeypatch, point):
    _, cache, cached = make_cached()

    def failure(*_):
        raise RuntimeError("injected compact scheduling failure")

    monkeypatch.setattr(cache, "_copy" if point == "copy" else "publish", failure)
    try:
        with pytest.raises(RuntimeError, match="injected compact"):
            with cached.lease([0]):
                pass
        assert not cache._active and cache._failure is not None
        with pytest.raises(RuntimeError, match="unusable"):
            with cached.lease([0]):
                pass
    finally:
        cache.close()


def test_consumer_failure_orders_queued_reads_and_does_not_poison_valid_cache():
    _, cache, cached = make_cached(capacity=1)
    try:
        with pytest.raises(RuntimeError, match="consumer failed"):
            with cached.lease([0]):
                result = cache._cells[0].clone()
                raise RuntimeError("consumer failed")
        with cached.lease([0]):
            assert torch.equal(cache._cells[0], result)
        assert cache._failure is None and cache.hits == cache.misses == 1
    finally:
        cache.close()


def test_graph_capture_rejected_before_publication_or_native_execution(monkeypatch):
    pk, cache, cached = make_cached()
    x, picks = inputs(pk)
    scratch = native.Scratch(cached, len(x), picks.shape[1])
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)
    try:
        with pytest.raises(RuntimeError, match="cannot be captured"):
            host.routed_cached(x, picks, cached, scratch, len(x))
        with pytest.raises(RuntimeError, match="cannot be captured"):
            with cached.lease([0]):
                pass
        assert cache.misses == 0 and not hasattr(scratch, "host_waves")
    finally:
        cache.close()


def test_duplicate_valid_routes_fail_before_lease_publication_or_native_consumer(monkeypatch):
    pk, cache, cached = make_cached()
    x, picks = inputs(pk)
    picks[0] = torch.tensor([0, 0, cached.count, -1], dtype=torch.int32, device="cuda")
    scratch = native.Scratch(cached, len(x), picks.shape[1])
    before_y = scratch.y.clone()
    before_ptrs = [t.clone() for t in (cached._tables.gate_ptr, cached._tables.up_ptr, cached._tables.down_ptr)]

    def forbidden(*args, **kwargs):
        pytest.fail("duplicate route reached an unsafe native consumer")

    monkeypatch.setattr(native, "routed", forbidden)
    monkeypatch.setattr(cache, "publish", forbidden)
    try:
        with pytest.raises(ValueError, match="distinct valid experts"):
            host.routed_cached(x, picks, cached, scratch, len(x))
        assert cache.misses == cache.hits == cache.copied_bytes == 0
        assert cache._policy.resident == {} and not cache._active and cache._failure is None
        assert torch.equal(scratch.y, before_y)
        assert all(torch.equal(actual, before) for actual, before in
                   zip((cached._tables.gate_ptr, cached._tables.up_ptr, cached._tables.down_ptr), before_ptrs))
    finally:
        cache.close()


def test_repeated_invalid_native_ids_are_skipped_and_retain_output_slots():
    pk, cache, cached = make_cached(capacity=1)
    base = resident(pk)
    x, _ = inputs(pk, rows=2)
    picks = torch.tensor([[cached.count, cached.count, -1, -1], [0, 99, 99, -1]],
                         dtype=torch.int32, device="cuda")
    normal, waved = [native.Scratch(ex, 2, 4) for ex in (base, cached)]
    normal.y.fill_(9.5)
    waved.y.fill_(9.5)
    expected = native.routed(x, picks, None, base, normal, None, 2).clone()
    try:
        actual = host.routed_cached(x, picks, cached, waved, 2)
        assert torch.equal(actual, expected)
        assert torch.equal(actual[:4], torch.full_like(actual[:4], 9.5))
        assert cache.misses == 1 and cache._failure is None
    finally:
        cache.close()

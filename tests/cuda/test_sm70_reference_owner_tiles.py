"""Exact preparation and original two-pass math, independent of native WMMA attention."""
import pytest
import torch

from tests.cuda.test_sm70_paged_attention import shard, sm70  # noqa: F401
from tests.cuda.test_sm70_paged_attention_diagnostic import cache_for
from tensorfold.cuda.sm70.reference_owner_tiles import ReferenceTiles, extension
from tensorfold.families.qwen4_exp import kv_formats
from tensorfold.families.qwen4_exp.cuda import history_owner as original


def exact(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def prepared(state):
    # Same already-paid communication extent as OwnerBuffers at rows2/heads3/D256.
    slab = torch.empty((8, 2, 12, 256), dtype=torch.float32, device="cuda")
    prep = ReferenceTiles(slab)
    ctx = prep.context(status=state.status, generations=state.generations,
                       slot_generation=0, owner_generation=0, layer=3)
    return prep, ctx


@pytest.mark.parametrize("name", ["int8", kv_formats.COMPACT_Q4, kv_formats.COMPACT_ISO4])
@pytest.mark.parametrize("degree", [1, 4])
@pytest.mark.parametrize("sparse", [False, True])
def test_fused_tiles_and_two_pass_outputs_match_original_bits(name, degree, sparse):
    extension()  # No compile/module search during a tile operation.
    pair = kv_formats.get_pair(name)
    fields = pair.key.bits, pair.key.codec
    gen = torch.Generator(device="cpu").manual_seed(4103)
    key, value = [(torch.randn((512, 1, 256), generator=gen) * .7).half().cuda() for _ in range(2)]
    states = [shard(degree, rank, key=fields, value=fields) for rank in range(4)]
    packed = [original._encode(v, f, states[0].status, 3, error=1)
              for v, f in [(key, pair.key), (value, pair.value)]]
    for state in states:
        for page, offset in enumerate(state.page_offsets.cpu().tolist()):
            if offset >= 0:
                for data, meta, (source, scale) in [(state.k, state.ks, packed[0]), (state.v, state.vs, packed[1])]:
                    data[offset:offset + 128].copy_(source[page * 128:(page + 1) * 128])
                    meta[offset:offset + 128].copy_(scale[page * 128:(page + 1) * 128])
    ids = (torch.cat((torch.tensor([-1], device="cuda"), torch.arange(0, 512, 2, device="cuda"),
                      torch.tensor([511, 512, 900], device="cuda"))).long()
           if sparse else torch.arange(512, device="cuda"))
    visible = ids < 260
    query = (torch.randn((12, 256), generator=gen) * .6).half().cuda()
    contexts, caches, stats_old, stats_new = [], [], [], []
    for state in states:
        cache = cache_for(state, pair)
        prep, ctx = prepared(state)
        contexts.append((prep, ctx))
        caches.append(cache)
        for first in range(0, ids.numel(), 128):
            positions, live = ids[first:first + 128], visible[first:first + 128]
            index, owned = cache.local_indices(positions)
            mask = owned & live
            expected_k = original._decode(torch.where(mask[:, None, None], cache.k[index], 0),
                torch.where(mask[:, None, None], cache.ks[index], 0), pair.key, 256)[:, 0] if cache.capacity else torch.zeros((positions.numel(), 256), device="cuda", dtype=torch.float16)
            expected_v = original._decode(torch.where(mask[:, None, None], cache.v[index], 0),
                torch.where(mask[:, None, None], cache.vs[index], 0), pair.value, 256)[:, 0] if cache.capacity else torch.zeros_like(expected_k)
            expected_k = torch.where(mask[:, None], expected_k, 0)
            expected_v = torch.where(mask[:, None], expected_v, 0)
            with ctx.tile(cache, positions, live) as tile:
                exact(tile.indices, index)
                exact(tile.mask, mask)
                exact(tile.key, expected_k)
                exact(tile.values(), expected_v)
            exact(state.status, torch.tensor([0, -2], dtype=torch.int32, device="cuda"))
        stats_old.append(original.partial_statistics(query, cache, ids, visible, 1/16))
        stats_new.append(original.partial_statistics(query, cache, ids, visible, 1/16, tile_preparer=ctx))
    for a, b in zip(stats_old, stats_new):
        exact(a[0], b[0])
        exact(a[1], b[1])
    old_global = original.merge_statistics(*(torch.stack([p[i] for p in stats_old]) for i in (0, 1)))
    new_global = original.merge_statistics(*(torch.stack([p[i] for p in stats_new]) for i in (0, 1)))
    exact(old_global[0], new_global[0])
    exact(old_global[1], new_global[1])
    old, new = [], []
    for cache, (prep, ctx) in zip(caches, contexts):
        old.append(original.normalized_partial(query, cache, ids, visible, 1/16, *old_global))
        new.append(original.normalized_partial(query, cache, ids, visible, 1/16, *new_global, tile_preparer=ctx))
        exact(old[-1], new[-1])
        assert prep.record()["additional_device_bytes"] == 0
    a, b = old[0], new[0]
    for rank in range(1, 4):
        a = a + old[rank]
        b = b + new[rank]
    exact(a, b)


def test_epoch_change_between_key_and_value_latches_without_reading_stale_payload():
    state = shard(4, 0)
    pair = kv_formats.get_pair("int8")
    cache = cache_for(state, pair)
    prep, ctx = prepared(state)
    ids = torch.arange(17, dtype=torch.int64, device="cuda")
    with ctx.tile(cache, ids, ids < 10) as tile:
        state.generations[1] += 1
        out = tile.values()
        assert torch.count_nonzero(out) == 0
    assert state.status.cpu().tolist() == [32, 3]
    assert tile.indices is tile.mask is tile.key is tile.cache is None
    with pytest.raises(RuntimeError, match="retired"):
        tile.values()
    prep.begin_retirement()
    prep.release_views()



def test_prepared_tail_and_reuse_keep_buffer_addresses_and_refuse_aliases():
    extension()
    state = shard(4, 0)
    cache = cache_for(state, kv_formats.get_pair("int8"))
    prep, ctx = prepared(state)
    ids = torch.arange(9, device="cuda", dtype=torch.int64)
    prep.half.fill_(17)
    addresses = tuple(t.data_ptr() for t in (prep.mask, prep.indices, prep.half))
    for _ in range(2):
        with ctx.tile(cache, ids, ids < 7) as tile:
            with pytest.raises(RuntimeError, match="already active"):
                with ctx.tile(cache, ids, ids < 7):
                    pass
            tile.values()
        assert torch.equal(prep.half[9:], torch.full_like(prep.half[9:], 17))
        assert tuple(t.data_ptr() for t in (prep.mask, prep.indices, prep.half)) == addresses
    with pytest.raises(RuntimeError, match="overlap"):
        extension().keys(cache.k, cache.ks, cache.page_offsets, ids, ids < 7, state.generations,
            prep.indices.view(torch.int32)[:2], prep.indices, prep.mask, prep.half,
            512, 128, 8, 0, 0, 0, 3)


@pytest.mark.parametrize("fault,expected", [("epoch", 32), ("page", 32), ("metadata", 8)])
def test_bad_epoch_mapping_or_metadata_is_contained_in_existing_status(fault, expected):
    state = shard(4, 0)
    cache = cache_for(state, kv_formats.get_pair("int8"))
    _, ctx = prepared(state)
    if fault == "epoch":
        state.generations[0] += 1
    elif fault == "page":
        state.page_offsets[0] = -2
    else:
        state.ks[0].fill_(float("nan"))
    ids = torch.zeros(1, dtype=torch.int64, device="cuda")
    with ctx.tile(cache, ids, ids == 0) as tile:
        assert torch.count_nonzero(tile.key) == 0
    assert state.status.cpu().tolist() == [expected, 3]

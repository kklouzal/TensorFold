"""Unique history accounting and execution against independent logical domains."""

from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement, owner_workspace_bytes


def _seed_zero_indexer_prefix(state, length):
    """Valid zero-key reference prefix without allocating discarded raw history."""
    import torch

    snapshot = state.snapshot()
    snapshot["pos"] = length
    lengths = state.compact_indexer.lengths(length, snapshot["mtp_len"])
    tails = []
    for entry, end in zip(state.compact_indexer.entries, lengths):
        if entry is None:
            tails.append(None)
            continue
        raw, coords = torch.zeros_like(entry.raw), torch.zeros_like(entry.coordinates)
        for position in range(end - end % 4, end):
            coords[position % 4].fill_(position)
        tails.append({"origin": 0, "length": end, "pooled_valid": end // 4, "raw": raw, "coordinates": coords})
    snapshot["compact_indexer"] = {"schema": "tensorfold.compact-indexer-v1", "lengths": lengths, "tails": tuple(tails)}
    state.restore(snapshot)


@pytest.mark.parametrize("degree", [1, 2, 4])
@pytest.mark.parametrize("rows", [1, 127, 128, 129, 262148])
def test_every_logical_head_row_has_one_owner_and_target_quartets_balance(degree, rows):
    placement = HistoryPlacement(8, 2, 12, degree=degree)
    physical_rows = (rows + 127) // 128 * 128
    per_rank = [sum(placement.local_rows(3, layer, rows, rank) for layer in range(12)) for rank in range(8)]
    assert per_rank == [3 * physical_rows] * 8
    for layer in range(13):
        assert sum(placement.local_rows(3, layer, rows, rank) for rank in range(8)) == 2 * physical_rows
    # Updated specification§10: partial logical tails still reserve complete
    # typed physical pages; validity and footprint are independent contracts.
    tail = (rows - 1) // 128
    assert placement.page_rows(rows, tail) == (rows - 1) % 128 + 1
    assert placement.physical_page_rows(rows, tail) == 128
    # Indexer ownership has exactly one canonical copy and whole 13-layer
    # inventories divide evenly across the stated first-candidate counts.
    for sessions in (16, 24):
        counts = [sum(placement.indexer_owner(slot, layer) == rank
                      for slot in range(sessions) for layer in range(13)) for rank in range(8)]
        assert counts == [sessions * 13 // 8] * 8


@pytest.mark.parametrize("degree", [1, 2, 4])
def test_cross_peer_merge_order_follows_logical_pages_not_session_slot(degree):
    """Regression: the FP32 cross-peer sums were taken in rank order, so a slot's
    rank rotation reordered >=3 page-residue partials (rows past 256 at degree 4)."""
    from tensorfold.families.qwen4_exp.cuda.history_owner import merge_order

    placement = HistoryPlacement(8, 2, 12, degree=degree)

    def decode(word):
        assert 0 <= word < 256
        peers = [j ^ (word >> (2 * j) & 3) for j in range(4)]
        assert sorted(peers) == [0, 1, 2, 3]
        return peers

    for head in (0, 1):
        for layer in range(placement.target_layers + 1):
            width = placement.group_size if layer == placement.target_layers else degree
            for pages in (1, 2, 3, 4, 9):
                residues = set()
                for slot in range(8):
                    owners = [placement.kv_owner(slot, layer, head, page) for page in range(pages)]
                    peers = decode(merge_order(owners))
                    # The j-th summand is the peer holding page j's residue class.
                    holders = [owners[page] % 4 for page in range(min(pages, width))]
                    assert peers[:len(holders)] == holders
                    assert peers[len(holders):] == sorted(set(range(4)) - set(holders))
                    residues.add(tuple(owners[page] % 4 for page in range(pages)))
                assert len(residues) > 1  # slots really rotate owners
    # Migration mixes maps: order still follows first owned logical pages.
    assert decode(merge_order([6, 6, 5, 7, 4])) == [2, 1, 3, 0]
    assert merge_order([0, 1, 2, 3]) == 0


def test_placement_rejects_unsupported_shapes_before_allocation():
    for args in ((4, 2, 12), (8, 4, 12), (8, 2, 0)):
        with pytest.raises(ValueError):
            HistoryPlacement(*args)
    for degree in (0, 3, 8, True):
        with pytest.raises(ValueError):
            HistoryPlacement(8, 2, 12, degree=degree)
    with pytest.raises(ValueError):
        HistoryPlacement(8, 2, 12, page_tokens=127)


@pytest.mark.torch
@pytest.mark.parametrize("degree", [1, 2, 4])
@pytest.mark.parametrize("format_name", ["fp16", "int8", "rotorquant-planar4-sm70-v1"])
def test_packed_pages_allocate_the_exact_rank_plan_and_keep_logical_rows(degree, format_name):
    import torch
    from tensorfold.families.qwen4_exp.cuda.history_owner import OwnedKVCache
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    placement = HistoryPlacement(8, 2, 12, degree=degree)
    pair = get_pair(format_name)
    rows, dim, slot, layer = 517, 256, 5, 3
    payload = torch.arange(rows, dtype=torch.int64)[:, None, None]
    for rank in range(8):
        cache = OwnedKVCache(rows, 1, dim, "cpu", pair=pair, placement=placement,
                             slot=slot, layer=layer, rank=rank)
        k = payload.expand(rows, 1, cache.k.shape[-1]).to(cache.k.dtype)
        v = k.clone()
        ks = torch.ones((rows, 1, dim // pair.key.group), dtype=getattr(torch, pair.key.scale_dtype))
        vs = torch.ones((rows, 1, dim // pair.value.group), dtype=getattr(torch, pair.value.scale_dtype))
        cache.append(0, k, v, ks, vs)
        expected = placement.local_rows(slot, layer, rows, rank) * pair.row_bytes(1, dim)
        expected += pair.dummy_bytes + len(cache.offsets) * 8
        assert cache.nbytes == expected
        logical = torch.arange(rows)
        offset, owned = cache.local_indices(logical)
        if owned.any():
            assert torch.equal(cache.k[offset[owned]], k[owned])
        clone = cache.clone()
        assert torch.equal(clone.k, cache.k) and torch.equal(clone.page_offsets, cache.page_offsets)
        assert not cache.k.numel() or clone.k.data_ptr() != cache.k.data_ptr()
        grown = cache.resized(901, rows)
        offset2, owned2 = grown.local_indices(logical)
        assert torch.equal(owned2, owned)
        if owned.any():
            assert torch.equal(grown.k[offset2[owned2]], k[owned])


@pytest.mark.torch
@pytest.mark.parametrize("degree", [1, 2, 4])
def test_disjoint_softmax_merge_matches_analytic_uniform_attention_and_masks_future_nan(degree):
    import torch
    from tensorfold.families.qwen4_exp.cuda.history_owner import (
        OwnedKVCache, partial_statistics, merge_statistics, normalized_partial,
    )
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    placement = HistoryPlacement(8, 2, 12, degree=degree)
    rows, dims = 512, 32
    key = torch.zeros((rows, 1, dims), dtype=torch.float16)
    value = torch.ones_like(key)
    value[:128].fill_(2)
    value[128:256].fill_(4)
    value[256:].fill_(float("nan"))
    selected = torch.arange(rows)
    query = torch.zeros((4, dims), dtype=torch.float16)
    partials, caches = [], []
    for rank in range(4):
        cache = OwnedKVCache(rows, 1, dims, "cpu", pair=get_pair("fp16"), placement=placement,
                             slot=0, layer=0, rank=rank)
        cache.append(0, key, value, None, None)
        caches.append(cache)
        partials.append(partial_statistics(query, cache, selected, selected < 256, dims ** -0.5))
    maximum, denominator = merge_statistics(*(torch.stack([part[index] for part in partials]) for index in range(2)))
    result = sum(normalized_partial(query, cache, selected, selected < 256, dims ** -0.5,
                                    maximum, denominator) for cache in caches)
    # Exactly representable powers-of-two denominators and integer values make
    # this an exact oracle: 128*2 + 128*4, divided by256, is3 in every lane.
    assert torch.equal(result, torch.full((4, dims), 3.0))


@pytest.mark.torch
def test_state_owns_only_canonical_histories_and_reports_no_gpu_observability_work():
    from tests.test_sm70_rotorquant_state import model
    from tensorfold.families.qwen4_exp.cuda.state import State

    rows, slot = 517, 2
    for rank in range(8):
        w = model(8)
        w.meta.update(full=NS(kv_heads=2), rank=rank, execution_provider="sm70",
                      history_layout="unique", attention_degree=2)
        state = State(w, rows, 4, "int8", limit=901, slot=slot)
        assert state.cache_bytes() == state.owned_cache_bytes()
        placement = state.history_placement
        for layer, (raw, pooled) in enumerate(zip(state.ikc, state.pooled)):
            expected = rank == placement.indexer_owner(slot, layer)
            assert raw.shape[0] == (4 if expected else 0)
            assert pooled.shape[0] == ((rows + 3) // 4 if expected else 0)
        with pytest.raises(ValueError, match="fixed full-window"):
            state.resize(901)
        assert state.capacity == rows and state.cache_bytes() == state.owned_cache_bytes()
        record = state.history_record()
        assert record["layout"] == "unique" and record["attention_degree"] == 2
        assert record["target_cached_length"] == record["draft_valid_length"] == 0
        page = state.history_page(0, 1, 1)
        assert page["owner_rank"] == placement.kv_owner(slot, 0, 1, 1)


def test_workspace_persistent_plan_matches_constructed_wire_slabs():
    text = NS(heads=4, head_dim=256, index_heads=4, index_dim=128,
              index_budget=2048, index_ratio=4, context=262148)
    small = owner_workspace_bytes(text, 32)
    large = owner_workspace_bytes(text, 256)
    assert small["persistent_bytes"] == (
        8 * 32 * 4 * 256 * 2 + 9 * 32 * (2052 + 2) * 4 + 9 * 32 * 16 * 256 * 4 + 4)
    assert large["persistent_bytes"] > small["persistent_bytes"]
    assert large["temporary_bytes"] > small["temporary_bytes"]


@pytest.mark.torch
@pytest.mark.parametrize("degree", [1, 2, 4])
def test_distributed_attention_retains_global_probability_half_rounding(degree):
    import torch
    from tensorfold.families.qwen4_exp.cuda.history_owner import (
        OwnedKVCache, partial_statistics, merge_statistics, normalized_partial,
    )
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    placement = HistoryPlacement(8, 2, 12, degree=degree)
    # Small gaps, large gaps, FP16-normal/subnormal probability boundaries,
    # opposing values, more than one tile and more than one physical shard.
    levels = torch.tensor([0, -2**-10, -0.5, -1, -4, -8, -12, -16], dtype=torch.float16)
    scores = levels.repeat(64)
    keys = torch.zeros((512, 1, 32), dtype=torch.float16)
    keys[:, 0, 0] = scores
    values = torch.arange(512, dtype=torch.int64).remainder(17).sub(8).half()[:, None, None].expand(512, 1, 32).clone()
    query = torch.zeros((1, 32), dtype=torch.float16)
    query[0, 0] = 1
    ids = torch.arange(512)
    visible = ids < 509
    caches, statistics = [], []
    for rank in range(4):
        cache = OwnedKVCache(512, 1, 32, "cpu", pair=get_pair("fp16"), placement=placement,
                             slot=0, layer=0, rank=rank)
        cache.append(0, keys, values, None, None)
        caches.append(cache)
        statistics.append(partial_statistics(query, cache, ids, visible, 1))
    maximum, denominator = merge_statistics(*(torch.stack([part[i] for part in statistics]) for i in (0, 1)))
    actual = sum(normalized_partial(query, cache, ids, visible, 1, maximum, denominator) for cache in caches).half()
    full_scores = scores.float().masked_fill(~visible, -torch.inf)
    expected = (full_scores.softmax(0).half().float() @ values[:, 0].float()).half()[None]
    assert torch.equal(actual, expected)


@pytest.mark.torch
def test_empty_global_attention_is_a_reported_invariant_failure():
    import torch
    from tensorfold.families.qwen4_exp.cuda.history_owner import merge_statistics

    with pytest.raises(ValueError, match="softmax mass"):
        merge_statistics(torch.full((4, 2), -torch.inf), torch.zeros((4, 2)))
    status = torch.tensor([0, -2], dtype=torch.int32)
    merge_statistics(torch.full((4, 2), -torch.inf), torch.zeros((4, 2)), status=status, layer=7)
    assert status.tolist() == [16, 7]


@pytest.mark.torch
@pytest.mark.parametrize("degree", [1, 2, 4])
@pytest.mark.parametrize("format_name", ["fp16", "int8", "rotorquant-planar4-sm70-v1"])
@pytest.mark.parametrize("budget,start", [(8, 0), (11, 9), (11, 19)])
def test_real_owner_layer_routes_both_tp_head_groups_with_batched_collectives(degree, format_name, budget, start):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    import torch
    from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers, attention_layer
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    from tensorfold.cuda.sm70 import family_ops
    from tests.test_sm70_rotorquant_state import model

    class World:
        def __init__(self):
            self.barrier = Barrier(8, timeout=15)
            self.sends = [None] * 8
            self.calls = [0] * 8

        def communicator(self, rank):
            def gather(send, recv):
                self.sends[rank] = send.clone()
                self.calls[rank] += 1
                self.barrier.wait()
                recv.copy_(torch.stack(self.sends))
                self.barrier.wait()
            return NS(world=8, rank=rank, all_gather=gather)

    world = World()
    workers = []
    for rank in range(8):
        w = model(8)
        c = w.cfg
        c.heads, c.index_heads, c.eps = 1, 2, 1e-6
        c.mrope_section, c.rope_attention_factor = (1, 1, 1), 1.0
        c.index_budget = budget
        w.inv_freq = torch.tensor([1.0])
        w.comm = world.communicator(rank)
        w.meta.update(full=NS(kv_heads=2), rank=rank, execution_provider="sm70",
                      history_layout="unique", attention_degree=degree)
        capacity = max(9, start + 6)
        state = State(w, capacity, 3, format_name)
        # This independent complete-history oracle uses the unbound native positional policy.
        assert w.positions is None and state.coordinate_profile_identity is None
        idw = budget + 4
        scratch = NS(idw=idw, budget=budget, ratio=4, qsa=True,
                     ids=torch.zeros((3, idw), dtype=torch.int32), nk=torch.zeros(3, dtype=torch.int32),
                     sparse=torch.zeros(3, dtype=torch.int32), scores=torch.zeros((3, (capacity + 3) // 4)))
        width = 2 * 256 + 2 * 256 + 3 * 128
        projection = torch.zeros((start + 3, width), dtype=torch.float16)
        projection[:, 3 * 256:4 * 256] = (torch.arange(start + 3).half() * 0.25 + (1 if rank < 4 else 2))[:, None]
        buffers = NS(pa=projection[start:].contiguous(), q=torch.empty((3, 1, 256), dtype=torch.float16),
                     iq=torch.empty((3, 2, 128), dtype=torch.float16),
                     attn_o=torch.empty((3, 1, 256), dtype=torch.float16), attn=scratch, prefill=False,
                     input_preparation_context=None)
        buffers.owner_kernel_family, buffers.owner_graph_family, buffers.owner_graphs = "torch-reference", "uncaptured", None
        from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging

        buffers.compact_indexer_stage = Staging(3, state.layers, "cpu", mtp=False)
        buffers.target_indexer_coordinates = buffers.target_indexer_frame = None
        buffers.target_indexer_position = None
        buffers.target_indexer_frame_id = 0
        buffers.target_indexer_coordinate_storage = buffers.compact_indexer_stage.coordinates
        buffers.history_owner = OwnerBuffers(w, 3, scratch)
        layer = NS(index=1, attn=NS(q_scale=torch.ones(256), k_scale=torch.ones(256),
                                  iq_scale=torch.ones(128), ik_scale=torch.ones(128)))
        pair = get_pair(format_name)
        cache = KVCache(capacity, 1, 256, "cpu", pair=pair)
        q = torch.empty((start + 3, 1, 256), dtype=torch.float16)
        iq = torch.empty((start + 3, 2, 128), dtype=torch.float16)
        raw = torch.zeros((capacity, 128), dtype=torch.float16)
        options = dict(k_bits=0 if pair.key.bits == 16 else pair.key.bits, k_codec=pair.key.codec,
                       v_bits=0 if pair.value.bits == 16 else pair.value.bits, v_codec=pair.value.codec)
        family_ops.attn_prep(projection, torch.zeros(1, dtype=torch.int32), layer.attn.q_scale, layer.attn.k_scale,
                             layer.attn.iq_scale, w.inv_freq, q, cache.k, cache.v, iq, raw, 1e-6,
                             q_heads=1, kv_heads=1, head_dim=256, index_heads=2, index_dim=128,
                             ks=cache.ks, vs=cache.vs, independent_rows=True, **options)
        state.kc[0].append(0, cache.k[:start], cache.v[:start], cache.ks[:start], cache.vs[:start])
        _seed_zero_indexer_prefix(state, start)
        buffers.reference = NS(cache=cache, q=q[start:], iq=iq[start:], options=options)
        workers.append((layer, w, [(state, 0, 3)], buffers))
    def execute(item):
        layer, w, segments, buffers = item
        return attention_layer(layer, w, segments, buffers, 3).clone()
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(execute, workers))
    for rank, output in enumerate(results):
        layer, w, _, buffers = workers[rank]
        # The pre-existing complete-history operator is independent of page
        # routing and owner preparation. Position-varying values expose any
        # duplicate position0 from reading beyond canonical nk, including a
        # nondivisible budget and an incomplete final pooling group.
        ref, pos = buffers.reference, torch.tensor([start], dtype=torch.int32)
        scratch = NS(qsa=True, budget=budget, ratio=4, out=torch.empty_like(output),
                     ids=torch.zeros((3, idw), dtype=torch.int32), nk=torch.zeros(3, dtype=torch.int32),
                     sparse=torch.zeros(3, dtype=torch.int32), scores=torch.zeros((3, (capacity + 3) // 4)))
        family_ops.qsa_rows(ref.iq, torch.zeros(((capacity + 3) // 4, 128), dtype=torch.float16),
                            pos, scratch, 3, context=start + 3)
        expected = family_ops.attention(ref.q, ref.cache.k, ref.cache.v, pos, scratch, 3, 256 ** -0.5,
                                        context=start + 3, ks=ref.cache.ks, vs=ref.cache.vs, **ref.options)
        assert torch.equal(output, expected)
    assert world.calls == [4] * 8  # Batched by layer, independent of request count.


@pytest.mark.torch
def test_compact_unowned_operands_are_zeroed_before_metadata_validation():
    import torch
    from tensorfold.families.qwen4_exp.cuda.history_owner import (
        allocate_target_caches, partial_statistics, normalized_partial,
    )
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    caches = allocate_target_caches(128, 256, "cpu", pair=get_pair("rotorquant-planar4-sm70-v1"),
                                    placement=HistoryPlacement(8, 2, 12), slot=1, rank=0)
    cache = caches[0]  # Its page belongs to rank1; arena row0 belongs to another layer.
    cache.ks.view(torch.int32).fill_(-65536)  # Deliberately invalid reserved bits in an invisible operand.
    cache.vs.view(torch.int32).fill_(-65536)
    query = torch.zeros((1, 256), dtype=torch.float16)
    ids = torch.arange(128)
    maximum, denominator = partial_statistics(query, cache, ids, ids < 128, 1)
    assert torch.isneginf(maximum).all() and torch.equal(denominator, torch.zeros(1))
    numerator = normalized_partial(query, cache, ids, ids < 128, 1, torch.zeros(1), torch.ones(1))
    assert torch.equal(numerator, torch.zeros((1, 256)))


@pytest.mark.torch
def test_independent_clone_preserves_actual_pages_without_sharing_page_tables():
    from tensorfold.families.qwen4_exp.cuda.state import State
    from tests.test_sm70_rotorquant_state import model

    w = model(8)
    w.layers = [NS(index=layer, linear=False) for layer in range(12)]
    w.meta.update(full=NS(kv_heads=2), rank=0, execution_provider="sm70", history_layout="unique")
    state = State(w, 257, 4, "int8")
    cloned = state.clone()
    assert cloned.kv_identity == state.kv_identity
    for before, after in zip(state.kc, cloned.kc):
        assert before.owners == after.owners and before.owners is not after.owners
        assert before.page_offsets.data_ptr() != after.page_offsets.data_ptr()



@pytest.mark.torch
@pytest.mark.parametrize("target_layers", [6, 12])
@pytest.mark.parametrize("rows,slot", [(1, 0), (129, 3)])
def test_prepared_fixed_metadata_reaches_actual_state_on_all_ranks(target_layers, rows, slot):
    from tests.test_sm70_rotorquant_state import model
    from tensorfold.families.qwen4_exp.cuda.state import State

    placement = HistoryPlacement(8, 2, target_layers, degree=4)
    for rank in range(8):
        w = model(8)
        w.layers = [NS(index=layer, linear=False) for layer in range(target_layers)]
        w.meta.update(full=NS(kv_heads=2), rank=rank, execution_provider="sm70",
                      history_layout="unique", attention_degree=4)
        assert w.meta["history_layout"] == "unique" and w.meta["attention_degree"] == 4
        state = State(w, rows, 4, "int8", limit=rows, slot=slot)
        assert state.history_placement == placement
        assert state.history_record()["attention_degree"] == 4
        assert state.cache_bytes() == state.owned_cache_bytes()
        for layer in range(target_layers + 1):
            assert state.history_placement.local_rows(slot, layer, rows, rank) == placement.local_rows(slot, layer, rows, rank)


@pytest.mark.parametrize("target_layers", [6, 12])
@pytest.mark.parametrize("rows", [1, 127, 128, 129, 1025, 262148])
@pytest.mark.parametrize("slot", [0, 3, 23])
def test_fixed_four_way_worst_rank_pages_include_tails_and_mtp(target_layers, rows, slot):
    placement = HistoryPlacement(8, 2, target_layers, degree=4)
    rank_totals = [0] * 8
    pages = (rows + 127) // 128
    for layer in range(target_layers + 1):
        expected = [0] * 8
        for page in range(pages):
            for head in range(2):
                rank = head * 4 + (((slot + layer) % 4) ^ (page % 4))
                assert placement.kv_owner(slot, layer, head, page) == rank
                expected[rank] += 128
        assert sum(expected) == 2 * pages * 128
        for rank in range(8):
            assert placement.local_rows(slot, layer, rows, rank) == expected[rank]
            rank_totals[rank] += expected[rank]
        assert sum(placement.page_rows(rows, page) for page in range(pages)) == rows
    # Six-layer stages keep their actual skew; no twelve-layer balance shortcut.
    assert sum(rank_totals) == (target_layers + 1) * 2 * pages * 128
    assert max(rank_totals) == max(sum(placement.local_rows(slot, layer, rows, rank)
                                     for layer in range(target_layers + 1)) for rank in range(8))

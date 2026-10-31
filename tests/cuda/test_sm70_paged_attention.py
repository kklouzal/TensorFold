"""Exact SM70 page/mask/epoch controls; full-model numerical gates are separate.

The uniform-score fixtures have eight visible keys and represented probability
1/8. Values repeat exactly across keys, so their FP16 products and FP32 sums
are exact for every ownership degree. No empirical floating tolerance is used.
"""

from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import paged_attention as operator
from tensorfold.cuda.sm70 import rotorquant_compact as compact


@pytest.fixture(autouse=True)
def sm70():
    if not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("native paged owner controls require an actual SM70 GPU")


def shard(degree, rank, *, rows=2, group=0, sparse=True, key=(8, 0), value=(8, 0)):
    page_tokens, logical = 128, 512
    offsets, physical = [], 0
    for page in range(4):
        owned = page % degree == rank
        offsets.append(physical if owned else -1)
        physical += page_tokens if owned else 0

    def payload(format, *, is_value):
        bits, codec = format
        if bits == 8:
            data = torch.zeros((physical, 1, 256), dtype=torch.int8, device="cuda")
            metadata = torch.full((physical, 1, 8), 256. if is_value else 0.,
                                  dtype=torch.float16, device="cuda")
        else:
            pattern = (torch.arange(128, device="cuda", dtype=torch.int32) % 16 * 17).byte()
            data = pattern.expand(physical, 1, 128).clone()
            words = torch.full((physical, 1, 2), 0x3c00 if is_value else 0, dtype=torch.int32, device="cuda")
            metadata = words.view(torch.float32)
        return data, metadata

    k, ks = payload(key, is_value=False)
    v, vs = payload(value, is_value=True)
    page_offsets = torch.tensor(offsets, dtype=torch.int64, device="cuda")
    generations = torch.zeros(2, dtype=torch.int64, device="cuda")
    status = torch.tensor([0, -2], dtype=torch.int32, device="cuda")
    position = torch.tensor([510 if sparse else 6], dtype=torch.int32, device="cuda")
    live = torch.tensor([rows], dtype=torch.int32, device="cuda")
    layer = torch.tensor([3], dtype=torch.int32, device="cuda")
    records = [[k.data_ptr(), v.data_ptr(), ks.data_ptr(), vs.data_ptr(), page_offsets.data_ptr(),
                generations.data_ptr(), status.data_ptr(), position.data_ptr(), live.data_ptr(),
                0, 0, logical, physical, 4, row, 0] for row in range(rows)]
    descriptor = torch.tensor(records, dtype=torch.int64, device="cuda")
    # Row0dense has7keys, so this fixture uses one dense row at end8 below.
    if not sparse:
        position.fill_(7)
        live.fill_(1)
    query = torch.zeros((8, rows, 3, 256), dtype=torch.float16, device="cuda")
    selection = torch.zeros((8, rows, 12), dtype=torch.int32, device="cuda")
    if sparse:
        selection[0, :, :10] = torch.tensor([-1, 0, 1, 128, 129, 256, 257, 384, 385, 512],
                                            dtype=torch.int32, device="cuda")
        selection[0, :, -2:] = torch.tensor([10, 1], dtype=torch.int32, device="cuda")
    stats = torch.empty((rows, 12, 2), dtype=torch.float32, device="cuda")
    global_stats = torch.empty_like(stats)
    values = torch.empty((rows, 12, 256), dtype=torch.float32, device="cuda")
    out = torch.empty((rows, 3, 256), dtype=torch.float16, device="cuda")
    return NS(k=k, v=v, ks=ks, vs=vs, page_offsets=page_offsets, generations=generations, status=status,
              position=position, live=live, layer=layer, table=descriptor, query=query, selection=selection,
              stats=stats, global_stats=global_stats, values=values, out=out, key=key, value=value, group=group)


def statistics(state):
    return operator.local_statistics(state.query, state.selection, state.table, state.stats,
                                     group_start=state.group, page_tokens=128, layer=state.layer,
                                     key_bits=state.key[0], key_codec=state.key[1], scale=1/16)


def expected_value(format):
    if format == (8, 0):
        result = torch.zeros(256, dtype=torch.float16)
        result[::32] = 32 * (1 / 32**0.5)
        return result
    data = (torch.arange(128, dtype=torch.int32) % 16 * 17).byte().reshape(1, 1, 128)
    metadata = torch.full((1, 1, 2), 0x3c00, dtype=torch.int32).view(torch.float32)
    represented = compact.dequant_ref(data, metadata, codec=format[1]).half().float()
    return compact.rotate_ref(represented, codec=format[1], inverse=True).half().reshape(256)


@pytest.mark.parametrize("degree", [1, 2, 4])
@pytest.mark.parametrize("key,value", [((8, 0), (8, 0)), ((4, 10), (4, 10)), ((4, 9), (4, 9))])
@pytest.mark.parametrize("group", [0, 4])
@pytest.mark.parametrize("sparse", [False, True])
def test_uniform_global_probability_and_exact_value_merge_across_unique_owners(degree, key, value, group, sparse):
    states = [shard(degree, rank, group=group, sparse=sparse, key=key, value=value) for rank in range(4)]
    gathered_stats = torch.zeros((8, 2, 12, 2), dtype=torch.float32, device="cuda")
    gathered_stats[..., 0].fill_(-torch.inf)
    for rank, state in enumerate(states):
        gathered_stats[group + rank].copy_(statistics(state))
    gathered_values = torch.zeros((8, 2, 12, 256), dtype=torch.float32, device="cuda")
    for rank, state in enumerate(states):
        operator.merge_statistics(gathered_stats, state.table, state.global_stats,
                                   group_start=group, page_tokens=128, layer=state.layer)
        operator.normalized_partial(state.query, state.selection, state.table, state.global_stats, state.values,
                                     group_start=group, page_tokens=128, layer=state.layer,
                                     key_bits=key[0], key_codec=key[1], value_bits=value[0], value_codec=value[1], scale=1/16)
        gathered_values[group + rank].copy_(state.values)
    expected = expected_value(value).to("cuda")
    for rank, state in enumerate(states):
        operator.merge_values(gathered_values, state.table, state.out, rank=group + rank,
                               page_tokens=128, layer=state.layer, value_bits=value[0], value_codec=value[1])
        assert state.status.cpu().tolist() == [0, -2]
        assert torch.equal(state.out[0], expected.expand(3, 256))
        if sparse:
            assert torch.equal(state.out[1], expected.expand(3, 256))
        else:
            assert torch.equal(state.out[1], torch.zeros_like(state.out[1]))


def test_stale_generation_and_inactive_rows_never_read_query_or_page_payload():
    state = shard(1, 0)
    state.query.fill_(float("nan"))
    state.generations[0] = 1
    statistics(state)
    assert state.status.cpu().tolist() == [32, 3]
    assert torch.isneginf(state.stats[..., 0]).all() and (state.stats[..., 1] == 0).all()
    state.status.copy_(torch.tensor([0, -2], dtype=torch.int32, device="cuda"))
    state.live.zero_()
    statistics(state)
    assert state.status.cpu().tolist() == [0, -2]


def test_global_normalization_precedes_half_rounding_with_unequal_shard_mass():
    states = [shard(4, rank) for rank in range(4)]
    chosen = torch.tensor([0, 1, 2, 128, 129, 256, 257, 384, 385], dtype=torch.int32, device="cuda")
    gathered_stats = torch.zeros((8, 2, 12, 2), dtype=torch.float32, device="cuda")
    gathered_stats[..., 0].fill_(-torch.inf)
    for rank, state in enumerate(states):
        state.vs.fill_(256 * 13)  # Every decoded value is exactly13.
        state.selection[0, :, :9] = chosen
        state.selection[0, :, -2] = 9
        gathered_stats[rank].copy_(statistics(state))
    all_values = torch.zeros((8, 2, 12, 256), dtype=torch.float32, device="cuda")
    for rank, state in enumerate(states):
        operator.merge_statistics(gathered_stats, state.table, state.global_stats,
                                   group_start=0, page_tokens=128, layer=state.layer)
        operator.normalized_partial(state.query, state.selection, state.table, state.global_stats, state.values,
                                     group_start=0, page_tokens=128, layer=state.layer,
                                     key_bits=8, key_codec=0, value_bits=8, value_codec=0, scale=1/16)
        all_values[rank].copy_(state.values)
    represented = float(torch.tensor(1/9, dtype=torch.float16))
    expected_scalar = (torch.tensor(32 * 13 * 9 * represented, dtype=torch.float32)
                       * torch.tensor(1 / 32**0.5, dtype=torch.float32)).half()
    unrounded_scalar = torch.tensor(13 * 32**0.5, dtype=torch.float16)
    assert expected_scalar != unrounded_scalar  # The fixture detects delayed Half rounding.
    expected = torch.zeros((2, 3, 256), dtype=torch.float16, device="cuda")
    expected[:, :, ::32] = expected_scalar.to("cuda")
    for rank, state in enumerate(states):
        operator.merge_values(all_values, state.table, state.out, rank=rank, page_tokens=128,
                               layer=state.layer, value_bits=8, value_codec=0)
        assert state.status.cpu().tolist() == [0, -2]
        assert torch.equal(state.out, expected)


def test_invalid_page_offset_is_contained_before_packed_cache_access():
    state = shard(1, 0)
    state.page_offsets[0] = 2**63 - 1
    statistics(state)
    assert state.status.cpu().tolist() == [32, 3]
    assert (state.stats[..., 1] == 0).all()


def test_inconsistent_partial_softmax_state_fails_even_if_other_shards_have_mass():
    state = shard(1, 0)
    gathered = torch.zeros((8, 2, 12, 2), dtype=torch.float32, device="cuda")
    gathered[:4, ..., 1] = 2
    gathered[0, ..., 1] = 0  # A finite maximum with zero mass is not an empty shard.
    operator.merge_statistics(gathered, state.table, state.global_stats,
                               group_start=0, page_tokens=128, layer=state.layer)
    assert state.status.cpu().tolist() == [16, 3]


def test_extreme_finite_scores_use_global_maximum_without_exp_overflow():
    states = [shard(4, rank) for rank in range(4)]
    all_stats = torch.zeros((8, 2, 12, 2), dtype=torch.float32, device="cuda")
    all_stats[..., 0].fill_(-torch.inf)
    for rank, state in enumerate(states):
        state.query.fill_(65504)
        state.vs.fill_(256 * 13)
        if rank == 0:
            state.k[0].fill_(127)
            state.ks[0].fill_(65504)
            state.vs[0].fill_(256)
        all_stats[rank].copy_(statistics(state))
    all_values = torch.zeros((8, 2, 12, 256), dtype=torch.float32, device="cuda")
    for rank, state in enumerate(states):
        operator.merge_statistics(all_stats, state.table, state.global_stats,
                                   group_start=0, page_tokens=128, layer=state.layer)
        operator.normalized_partial(state.query, state.selection, state.table, state.global_stats, state.values,
                                     group_start=0, page_tokens=128, layer=state.layer,
                                     key_bits=8, key_codec=0, value_bits=8, value_codec=0, scale=1/16)
        all_values[rank].copy_(state.values)
    expected = expected_value((8, 0)).to("cuda").expand(2, 3, 256)
    for rank, state in enumerate(states):
        operator.merge_values(all_values, state.table, state.out, rank=rank, page_tokens=128,
                               layer=state.layer, value_bits=8, value_codec=0)
        assert state.status.cpu().tolist() == [0, -2]
        assert torch.equal(state.out, expected)


def test_captured_operator_reads_current_layer_and_generation_controls():
    state = shard(1, 0)
    state.generations[1] = 1
    statistics(state)  # Prepare extension and allocator before capture.
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        statistics(state)
    state.layer.fill_(17)
    state.status.copy_(torch.tensor([0, -2], dtype=torch.int32, device="cuda"))
    graph.replay()
    assert state.status.cpu().tolist() == [32, 17]

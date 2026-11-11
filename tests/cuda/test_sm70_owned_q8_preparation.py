"""Owned-Q8 decode preparation versus unchanged per-State Torch program.

Strict component gate; numerical mismatches remain failures, not a tolerance.
No compact span/pool transaction or model-quality equivalence is inferred.
"""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda import history_owner as history
from tensorfold.families.qwen4_exp.kv_formats import get_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='owned Q8 preparation requires actual SM70')


def exact(actual, expected):
    assert actual.dtype == expected.dtype and actual.shape == expected.shape
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))


def fixture(start=64, *, rank=0, owned=True, pattern='random', capacity=385, owned_pages=None):
    device = torch.device('cuda', torch.cuda.current_device())
    gen = torch.Generator(device=device).manual_seed(1207)
    cfg = NS(heads=3, head_dim=256, index_heads=4, index_dim=128, eps=1e-6,
             mrope_section=(11, 11, 10), rope_attention_factor=1.0)
    placement = NS(group_size=4, page_tokens=128,
                   physical_page_rows=lambda capacity, page: min(128, capacity - page * 128),
                   indexer_owner=lambda *args, **kwargs: (rank + 1) % 8)
    pages = (capacity + placement.page_tokens - 1) // placement.page_tokens
    foreign = rank // 4 * 4 + (rank + 1) % 4
    owners = [rank if owned and (owned_pages is None or page in owned_pages) else foreign
              for page in range(pages)]
    cache = history.OwnedKVCache(capacity, 1, 256, device, pair=get_pair('int8'),
                                 placement=placement, slot=0, layer=0, rank=rank, owners=owners)
    for t in (cache.k, cache.v, cache.ks, cache.vs):
        t.fill_(7)
    state = NS(pos=start, att_index={0: 0}, kc=[cache], history_rank=rank,
               history_placement=placement, history_slot=0, slot_generation=7, history_generation=3,
               history_generations=torch.tensor([7, 3], device=device, dtype=torch.int64),
               kv_status=torch.tensor([0, -2], device=device, dtype=torch.int32),
               compact_indexer=object(), pooled=[None], target_tentative_length=start)
    def norm(width):
        return torch.rand(width, device=device, generator=gen) + .5
    layer = NS(index=0, attn=NS(q_scale=norm(256), k_scale=norm(256), iq_scale=norm(128)))
    frequency = torch.linspace(.0001, .03, 64, device=device)
    weights = NS(cfg=cfg, inv_freq=frequency)
    width = 2 * 3 * 256 + 512 + 5 * 128
    p = torch.randn((1, width), device=device, dtype=torch.float16, generator=gen)
    if pattern == 'zeros':
        p.zero_()
        p[:, 1::2] = -0.0
    elif pattern == 'cancellation':
        p.fill_(32)
        p[:, 1::2] = -32
    elif pattern == 'subnormal':
        p.fill_(2**-24)
        p[:, 1::2] = -(2**-24)
    elif pattern == 'nonfinite':
        for begin in (0, 1536, 1792, 2048, 2560):
            p[0, begin] = torch.nan
    elif pattern == 'raw-index-nonfinite':
        p[0, 2560] = torch.nan
    elif pattern == 'overflow':
        p.fill_(65504)
        layer.attn.q_scale.fill_(65504)
        layer.attn.k_scale.fill_(65504)
    axes = torch.tensor([[start, start + 3, -start]], device=device, dtype=torch.int64)
    buffers = NS(prefill=False, target_indexer_coordinates=axes,
                 target_indexer_position=NS(inv_freq=frequency, sections=(11, 11, 10), amplitude=1.0),
                 q=torch.full((1, 3, 256), 13, device=device, dtype=torch.float16),
                 iq=torch.full((1, 4, 128), 13, device=device, dtype=torch.float16),
                 owner_kernel_family='sm70-paged-owner-rn-ex2-p2051-v1', owner_graph_family='uncaptured',
                 history_owner=NS(owned_q8_preparation=None))
    return p, state, layer, weights, buffers


@pytest.mark.parametrize('start,rank,owned,pattern', [
    (64, 0, True, 'random'), (127, 3, True, 'random'), (128, 4, True, 'random'),
    (384, 7, True, 'random'), (64, 1, False, 'random'), (128, 5, False, 'nonfinite'),
    (64, 0, True, 'zeros'), (96, 0, True, 'cancellation'), (64, 0, True, 'subnormal'),
    (64, 0, True, 'nonfinite'), (64, 0, True, 'raw-index-nonfinite'), (64, 0, True, 'overflow'),
])
@torch.no_grad()
def test_unchanged_torch_program_raw_outputs_status_coordinates_and_versions(start, rank, owned, pattern):
    old = fixture(start, rank=rank, owned=owned, pattern=pattern)
    new = fixture(start, rank=rank, owned=owned, pattern=pattern)
    module = ops.prepare_owned_q8(new[0].device)
    new[4].history_owner.owned_q8_preparation = module
    def tensors(item):
        _, state, _, _, buffers = item
        cache = state.kc[0]
        return (buffers.q, buffers.iq, cache.k, cache.v, cache.ks, cache.vs, state.kv_status)
    before = [[t._version for t in tensors(x)] for x in (old, new)]
    axes = new[4].target_indexer_coordinates.clone()
    generations = new[1].history_generations.clone()
    history._prepare_reference(old[0], old[1], old[2], old[3], old[4], 0, 1, mtp=False)
    history.prepare(new[0], new[1], new[2], new[3], new[4], 0, 1, mtp=False)
    torch.cuda.synchronize()
    for actual, expected in zip(tensors(new), tensors(old)):
        exact(actual, expected)
    assert [t._version - n for t, n in zip(tensors(new), before[1])] == [
        t._version - n for t, n in zip(tensors(old), before[0])]
    exact(new[4].target_indexer_coordinates, axes)
    exact(new[1].history_generations, generations)
    assert new[1].target_tentative_length == old[1].target_tentative_length == start + 1


@pytest.mark.parametrize('fault', ['generation', 'bad-offset', 'out-of-bounds-offset'])
@torch.no_grad()
def test_generation_and_page_errors_poison_without_payload_or_query_publication(fault):
    p, state, layer, weights, buffers = fixture()
    module = ops.prepare_owned_q8(p.device)
    cache = state.kc[0]
    before = [t.clone() for t in (buffers.q, buffers.iq, cache.k, cache.v, cache.ks, cache.vs)]
    if fault == 'generation':
        state.history_generations[0] += 1
    else:
        cache.page_offsets[0] = -2 if fault == 'bad-offset' else cache.capacity
    assert ops.owned_q8_prepare(module, p, buffers.target_indexer_coordinates, weights, cache, state, layer,
        buffers.q, buffers.iq, frequency=weights.inv_freq, sections=(11, 11, 10), amplitude=1., start=64) is not None
    torch.cuda.synchronize()
    assert state.kv_status.cpu().tolist() == [32, 0]
    for actual, expected in zip((buffers.q, buffers.iq, cache.k, cache.v, cache.ks, cache.vs), before):
        exact(actual, expected)


@torch.no_grad()
def test_nondefault_stream_replay_uses_changed_coordinates_input_and_sticky_status():
    old = fixture()
    new = fixture()
    new[4].history_owner.owned_q8_preparation = ops.prepare_owned_q8(new[0].device)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for index in range(3):
            for p, state, _layer, _weights, buffers in (old, new):
                p.mul_(.5)
                buffers.target_indexer_coordinates.add_(index + 1)
                state.kv_status.copy_(torch.tensor([8, 2], device=p.device, dtype=torch.int32))
            history._prepare_reference(old[0], old[1], old[2], old[3], old[4], 0, 1, mtp=False)
            history.prepare(new[0], new[1], new[2], new[3], new[4], 0, 1, mtp=False)
    stream.synchronize()
    for actual, expected in zip((new[4].q, new[4].iq, new[1].kc[0].k, new[1].kc[0].v, new[1].kv_status),
                                (old[4].q, old[4].iq, old[1].kc[0].k, old[1].kc[0].v, old[1].kv_status)):
        exact(actual, expected)


@torch.no_grad()
def test_missing_compact_owner_retains_original_primary_status_and_no_output_publication():
    old, new = fixture(pattern='nonfinite'), fixture(pattern='nonfinite')
    new[4].history_owner.owned_q8_preparation = ops.prepare_owned_q8(new[0].device)
    for item in (old, new):
        item[1].compact_indexer = None
    def watched(item):
        return (item[4].q, item[4].iq, item[1].kc[0].k, item[1].kc[0].v)
    snapshots = [[t.clone() for t in watched(x)] for x in (old, new)]
    errors = []
    for function, item in ((history._prepare_reference, old), (history.prepare, new)):
        with pytest.raises(RuntimeError) as caught:
            function(item[0], item[1], item[2], item[3], item[4], 0, 1, mtp=False)
        errors.append(str(caught.value))
    torch.cuda.synchronize()
    assert errors[0] == errors[1]
    exact(new[1].kv_status, old[1].kv_status)
    for item, before in zip((old, new), snapshots):
        for actual, expected in zip(watched(item), before):
            exact(actual, expected)


@pytest.mark.parametrize('fault', ['output-alias', 'axes-shape', 'axes-dtype'])
@torch.no_grad()
def test_binding_refuses_invalid_boundaries_before_native_publication(fault):
    p, state, layer, weights, buffers = fixture()
    module = ops.prepare_owned_q8(p.device)
    if fault == 'output-alias':
        buffers.iq = buffers.q.view(-1)[:512].view(1, 4, 128)
    elif fault == 'axes-shape':
        buffers.target_indexer_coordinates = buffers.target_indexer_coordinates[:, :2].contiguous()
    else:
        buffers.target_indexer_coordinates = buffers.target_indexer_coordinates.float()
    before = [t.clone() for t in (buffers.q, buffers.iq, state.kc[0].k, state.kc[0].v)]
    with pytest.raises(RuntimeError):
        ops.owned_q8_prepare(module, p, buffers.target_indexer_coordinates, weights, state.kc[0], state, layer,
            buffers.q, buffers.iq, frequency=weights.inv_freq, sections=(11, 11, 10), amplitude=1., start=64)
    torch.cuda.synchronize()
    for actual, expected in zip((buffers.q, buffers.iq, state.kc[0].k, state.kc[0].v), before):
        exact(actual, expected)


@pytest.mark.parametrize('start', [2047, 2048, 2051, 2052, 262140, 262141, 262142, 262143, 262144])
@pytest.mark.parametrize('owned', [True, False])
@torch.no_grad()
def test_long_absolute_positions_all_pool_phases_and_final_context_writer_match_reference(start, owned):
    # Full logical allowance and complete2049-entry table, but at most ONE
    # physical page is owned here. This is writer/domain testing, not a claim
    # that near-full history was populated or full-model quality was exercised.
    page = start // 128
    old = fixture(start, rank=0, owned=owned, capacity=262145, owned_pages={page})
    new = fixture(start, rank=0, owned=owned, capacity=262145, owned_pages={page})
    new[4].history_owner.owned_q8_preparation = ops.prepare_owned_q8(new[0].device)
    def watched(item):
        state, buffers = item[1], item[4]
        cache = state.kc[0]
        return (buffers.q, buffers.iq, cache.k, cache.v, cache.ks, cache.vs, state.kv_status)
    before = [[t._version for t in watched(item)] for item in (old, new)]
    axes = new[4].target_indexer_coordinates.clone()
    generations = new[1].history_generations.clone()
    offsets = new[1].kc[0].page_offsets.clone()
    assert new[1].kc[0].page_offsets.numel() == 2049
    assert new[1].kc[0].capacity <= 128
    history._prepare_reference(old[0], old[1], old[2], old[3], old[4], 0, 1, mtp=False)
    history.prepare(new[0], new[1], new[2], new[3], new[4], 0, 1, mtp=False)
    torch.cuda.synchronize()
    for actual, expected in zip(watched(new), watched(old)):
        exact(actual, expected)
    assert new[1].kv_status.cpu().tolist() == [0, -2]
    assert [t._version - n for t, n in zip(watched(new), before[1])] == [
        t._version - n for t, n in zip(watched(old), before[0])]
    exact(new[4].target_indexer_coordinates, axes)
    exact(new[1].history_generations, generations)
    exact(new[1].kc[0].page_offsets, offsets)
    assert new[1].target_tentative_length == old[1].target_tentative_length == start + 1


@pytest.mark.parametrize('axes', [(0, 262144, 131072), (262144, 0, 262143)])
@torch.no_grad()
def test_long_issued_axis_selection_uses_each_original_coordinate_without_text_position_substitution(axes):
    old = fixture(262144, rank=4, capacity=262145, owned_pages={2048})
    new = fixture(262144, rank=4, capacity=262145, owned_pages={2048})
    for item in (old, new):
        item[4].target_indexer_coordinates.copy_(torch.tensor([axes], device=item[0].device, dtype=torch.int64))
    new[4].history_owner.owned_q8_preparation = ops.prepare_owned_q8(new[0].device)
    history._prepare_reference(old[0], old[1], old[2], old[3], old[4], 0, 1, mtp=False)
    history.prepare(new[0], new[1], new[2], new[3], new[4], 0, 1, mtp=False)
    torch.cuda.synchronize()
    for actual, expected in ((new[4].q, old[4].q), (new[4].iq, old[4].iq),
            (new[1].kc[0].k, old[1].kc[0].k), (new[1].kc[0].v, old[1].kc[0].v),
            (new[1].kc[0].ks, old[1].kc[0].ks), (new[1].kc[0].vs, old[1].kc[0].vs),
            (new[1].kv_status, old[1].kv_status)):
        exact(actual, expected)
    assert new[1].kv_status.cpu().tolist() == [0, -2]

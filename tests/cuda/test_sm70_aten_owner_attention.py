"""Owned Q8 gather -> ATen -> ordered four-shard merge controls on SM70.

Logical shard collectives are materialized by copies in this single-GPU fixture;
actual TP8 transport, model/MTP/PPL and serving/resource acceptance are separate.
"""
import json
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import attention_backends as backend, paged_attention as native
from tests.cuda.test_sm70_paged_attention import shard, expected_value

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires actual SM70')


def fixture(degree=4, group=0, *, query_layout=None):
    states = [shard(degree, rank, group=group) for rank in range(4)]
    rows, keys = 2, backend.OWNER_MAX_KEYS
    spec = backend.TileSpec(rows, 1, keys, 12, 1, .0625, masked=True, query_layout=query_layout)
    primitive = backend.prepare(spec, device='cuda:0',
        device_budget_bytes=backend.tensor_requirements(spec)['known_peak_tensor_bytes'])
    owners = []
    for state in states:
        owners.append(NS(state=state, query=torch.empty(rows, 1, 12, 256, device='cuda', dtype=torch.float16),
            key=torch.empty(rows, keys, 1, 256, device='cuda', dtype=torch.float16),
            value=torch.empty(rows, keys, 1, 256, device='cuda', dtype=torch.float16),
            valid=torch.empty(rows, 1, keys, device='cuda', dtype=torch.bool),
            normalized=torch.empty(rows, 12, 256, device='cuda', dtype=torch.float16),
            lse=torch.empty(rows, 12, device='cuda')))
    all_stats = torch.empty(8, rows, 12, 2, device='cuda')
    all_values = torch.empty(8, rows, 12, 256, device='cuda')
    return NS(owners=owners, primitive=primitive, group=group, all_stats=all_stats, all_values=all_values)


def pipeline(frame):
    frame.all_stats[..., 0].fill_(-torch.inf)
    frame.all_stats[..., 1].zero_()
    frame.all_values.zero_()
    for rank, owner in enumerate(frame.owners):
        state = owner.state
        native.pack_owner_queries(state.query, state.table, owner.query,
            group_start=frame.group, page_tokens=128, layer=state.layer)
        native.gather_owner_q8(state.selection, state.table, owner.key, owner.value, owner.valid,
            first=0, group_start=frame.group, page_tokens=128, layer=state.layer)
        result = frame.primitive.run(owner.query, owner.key, owner.value, valid_keys=owner.valid)
        owner.normalized.copy_(result.output[:, 0])
        owner.lse.copy_(result.logsumexp[:, :, 0])
        state.stats[..., 0].copy_(owner.lse)
        state.stats[..., 1].copy_(result.valid_rows.expand(2, 12))
        frame.all_stats[frame.group + rank].copy_(state.stats)
    for rank, owner in enumerate(frame.owners):
        state = owner.state
        native.merge_statistics(frame.all_stats, state.table, state.global_stats,
            group_start=frame.group, page_tokens=128, layer=state.layer)
        native.weight_owner_normalized(owner.normalized, owner.lse, state.global_stats, state.table, state.values,
            group_start=frame.group, page_tokens=128, layer=state.layer)
        frame.all_values[frame.group + rank].copy_(state.values)
    for rank, owner in enumerate(frame.owners):
        state = owner.state
        native.merge_values(frame.all_values, state.table, state.out, rank=frame.group + rank,
            page_tokens=128, layer=state.layer, value_bits=8, value_codec=0)


@pytest.mark.parametrize('degree', [1, 2, 4])
@pytest.mark.parametrize('group', [0, 4])
@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_uniform_exact_result_with_empty_local_owners_and_both_head_groups(degree, group, query_layout):
    frame = fixture(degree, group, query_layout=query_layout)
    pipeline(frame)
    expected = expected_value((8, 0)).to('cuda')
    for owner in frame.owners:
        assert owner.state.status.cpu().tolist() == [0, -2]
        assert torch.equal(owner.state.out, expected.expand(2, 3, 256))


def test_pack_head_order_stale_generation_and_inactive_rows():
    state = shard(1, 0, group=4)
    state.query.copy_(torch.arange(8 * 2 * 3, device='cuda').half().view(8, 2, 3, 1).expand_as(state.query))
    out = torch.empty(2, 1, 12, 256, device='cuda', dtype=torch.float16)
    native.pack_owner_queries(state.query, state.table, out, group_start=4, page_tokens=128, layer=state.layer)
    assert torch.equal(out[:, 0], state.query[4:8].permute(1, 0, 2, 3).reshape(2, 12, 256))
    state.generations[0] = 1
    state.query.fill_(torch.nan)
    native.pack_owner_queries(state.query, state.table, out, group_start=4, page_tokens=128, layer=state.layer)
    assert torch.equal(out, torch.zeros_like(out)) and state.status.cpu().tolist() == [32, 3]
    state.status.copy_(torch.tensor([0, -2], dtype=torch.int32, device='cuda'))
    state.live.zero_()
    native.pack_owner_queries(state.query, state.table, out, group_start=4, page_tokens=128, layer=state.layer)
    assert state.status.cpu().tolist() == [0, -2]


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_captured_pipeline_reads_changed_q8_values_and_selected_key_masks(query_layout):
    frame = fixture(query_layout=query_layout)
    pipeline(frame)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            pipeline(frame)
        frame.owners[0].state.vs.fill_(768.)  # Decodes selected local values3 ratherthan1.
        graph.replay()
        expected = (expected_value((8, 0)).float() * 1.5).half().to('cuda')
        for owner in frame.owners:
            assert torch.equal(owner.state.out, expected.expand_as(owner.state.out))
        # Remove both rank0 keys; that owner's selected set is explicitly empty.
        for owner in frame.owners:
            owner.state.selection[0, :, 1:3].fill_(-1)
        frame.owners[0].state.k.fill_(-128)
        frame.owners[0].state.ks.fill_(torch.nan)
        frame.owners[0].state.vs.fill_(torch.nan)
        graph.replay()
        for owner in frame.owners:
            assert owner.state.status.cpu().tolist() == [0, -2]
            assert torch.equal(owner.state.out, expected_value((8, 0)).to('cuda').expand_as(owner.state.out))
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize('field,flag', [('query', 1), ('ks', 2), ('vs', 4)])
@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_nonfinite_valid_data_latches_original_state_failure_authority(field, flag, query_layout):
    frame = fixture(query_layout=query_layout)
    getattr(frame.owners[0].state, field).fill_(torch.nan)
    pipeline(frame)
    assert frame.owners[0].state.status.cpu().tolist() == [flag, 3]


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_unequal_local_mass_reports_changed_numeric_family_without_fitted_tolerance(record_property, query_layout):
    frame = fixture(query_layout=query_layout)
    chosen = torch.tensor([0, 1, 2, 128, 129, 256, 257, 384, 385], dtype=torch.int32, device='cuda')
    for rank, owner in enumerate(frame.owners):
        owner.state.selection[0, :, :9] = chosen
        owner.state.selection[0, :, -2] = 9
        owner.state.vs.fill_(256 * (rank + 1))
    pipeline(frame)
    ideal = torch.zeros(256, device='cuda', dtype=torch.float64)
    ideal[::32] = (21 / 9) * 32**.5
    errors = []
    for owner in frame.owners:
        assert owner.state.status.cpu().tolist() == [0, -2] and torch.isfinite(owner.state.out).all()
        errors.append(float((owner.state.out.double() - ideal).abs().max().item()))
    record_property('aten_owner_numeric_diagnostic', json.dumps({'max_abs_vs_ideal': max(errors),
        'family': backend.OWNER_FAMILY, 'query_layout': query_layout, 'quality_accepted': False, 'MTP_model_qualified': False, 'P3_qualified': False}))


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_fixed_key_capacity_live_row_embedding_preserves_output_and_lse_bits(record_property, query_layout):
    # A numerical profile may differ from the incumbent, but serial/verify
    # regrouping within this profile must preserve each live row's arithmetic.
    generator = torch.Generator(device='cuda').manual_seed(117)
    q = torch.randn(4, 1, 12, 256, generator=generator, device='cuda').half()
    k = torch.randn(4, backend.OWNER_MAX_KEYS, 1, 256, generator=generator, device='cuda').half()
    v = torch.randn(k.shape, generator=generator, device='cuda').half()
    mask = torch.arange(backend.OWNER_MAX_KEYS, device='cuda')[None, None] < torch.tensor(
        [1, 17, 65, 2048], device='cuda')[:, None, None]
    def run(q, k, v, mask, *, layout=query_layout):
        spec = backend.TileSpec(len(q), 1, backend.OWNER_MAX_KEYS, 12, 1, .0625, masked=True, query_layout=layout)
        prepared = backend.prepare(spec, device='cuda:0',
            device_budget_bytes=backend.tensor_requirements(spec)['known_peak_tensor_bytes'])
        return prepared.run(q, k, v, valid_keys=mask)
    batched = run(q, k, v, mask)
    differences = []
    for row in range(4):
        single = run(q[row:row + 1], k[row:row + 1], v[row:row + 1], mask[row:row + 1])
        differences.append([int((single.output.view(torch.int16) != batched.output[row:row + 1].view(torch.int16)).sum()),
            int((single.logsumexp.view(torch.int32) != batched.logsumexp[row:row + 1].view(torch.int32)).sum())])
    permutation = torch.tensor([3, 1, 0, 2], device='cuda')
    reordered = run(q[permutation], k[permutation], v[permutation], mask[permutation])
    differences.append([int((reordered.output.view(torch.int16) != batched.output[permutation].view(torch.int16)).sum()),
        int((reordered.logsumexp.view(torch.int32) != batched.logsumexp[permutation].view(torch.int32)).sum())])
    record_property('row_embedding_bit_differences', json.dumps(differences))
    assert differences == [[0, 0]] * 5
    if query_layout == backend.HEAD_QUERY_LAYOUT:
        reference = run(q, k, v, mask, layout=None)
        assert torch.isfinite(reference.output).all() and torch.isfinite(reference.logsumexp).all()
        assert torch.isfinite(batched.output).all() and torch.isfinite(batched.logsumexp).all()
        record_property('original_vs_head_rows_diagnostic', json.dumps({
            'output_bit_differences': int((reference.output.view(torch.int16) != batched.output.view(torch.int16)).sum()),
            'lse_bit_differences': int((reference.logsumexp.view(torch.int32) != batched.logsumexp.view(torch.int32)).sum()),
            'max_abs_output_delta': float((reference.output.float() - batched.output.float()).abs().max()),
            'max_abs_lse_delta': float((reference.logsumexp - batched.logsumexp).abs().max()),
            'quality_accepted': False, 'P3_qualified': False}))


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_visible_qsa_mask_never_reads_future_poisoned_q8_rows(query_layout):
    frame = fixture(degree=1, query_layout=query_layout)
    for owner in frame.owners:
        state = owner.state
        state.position.zero_()
        state.live.fill_(1)
        state.selection[0, :, :2] = torch.tensor([0,1], dtype=torch.int32, device='cuda')
        state.selection[0, :, -2] = 2
        # Only logical0 is visible to active row0. Row1 is inactive; logical1
        # must not become visible merely because twelve heads became query rows.
        if state.k.numel():
            state.ks[1].fill_(torch.nan)
            state.vs[1].fill_(torch.nan)
    pipeline(frame)
    expected = expected_value((8,0)).to('cuda')
    for owner in frame.owners:
        assert owner.state.status.cpu().tolist() == [0,-2]
        assert torch.equal(owner.state.out[0], expected.expand(3,256))
        assert torch.count_nonzero(owner.state.out[1]) == 0

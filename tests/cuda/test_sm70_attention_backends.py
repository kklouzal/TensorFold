"""Unselected ATen/CUTLASS tile integration, current-stream and capture controls."""

import json
import math

import pytest
import torch

from tensorfold.cuda.sm70 import attention_backends as backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires actual SM70')


def prepare(*, rows=3, masked=True, queries=1, heads=12, kv_heads=1, query_layout=None, batch=2):
    spec = backend.TileSpec(batch, queries, rows, heads, kv_heads, .0625, masked=masked, query_layout=query_layout)
    return backend.prepare(spec, device=torch.device('cuda', torch.cuda.current_device()),
                           device_budget_bytes=backend.tensor_requirements(spec)['known_peak_tensor_bytes'])


def operands(spec):
    q = torch.zeros((spec.batch, spec.query_rows, spec.query_heads, 256), dtype=torch.float16, device='cuda')
    k = torch.zeros((spec.batch, spec.key_rows, spec.kv_heads, 256), dtype=torch.float16, device='cuda')
    v = torch.arange(spec.key_rows, dtype=torch.float16, device='cuda')[None, :, None, None].expand(
        spec.batch, spec.key_rows, spec.kv_heads, 256).contiguous()
    mask = torch.zeros((spec.batch, spec.query_rows, spec.key_rows), dtype=torch.bool, device='cuda')
    if spec.key_rows:
        mask[0, :, 1 if spec.key_rows > 1 else 0] = True
    return q, k, v, mask


def assert_single_key_identity(result, selected):
    assert result.output.dtype == torch.float16 and result.logsumexp.dtype == torch.float32
    assert torch.equal(result.output[0], torch.full_like(result.output[0], selected))
    assert torch.equal(result.logsumexp[0], torch.zeros_like(result.logsumexp[0]))
    assert torch.equal(result.output[1], torch.zeros_like(result.output[1]))
    assert torch.isneginf(result.logsumexp[1]).all()
    assert result.policy == backend.POLICY


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_head256_gqa_masked_tail_and_empty_owner_obey_integer_identity(query_layout):
    prepared = prepare(rows=3, query_layout=query_layout)
    q, k, v, mask = operands(prepared.spec)
    result = prepared.run(q, k, v, valid_keys=mask)
    assert_single_key_identity(result, 1)
    assert result.output.shape == q.shape and result.logsumexp.shape == (2, 12, 1)
    assert not any(torch._C._overlaps(result.output, item) for item in (q, k, v))
    assert not any(torch._C._overlaps(result.logsumexp, item) for item in (q, k, v))
    assert torch.equal(result.valid_rows, torch.tensor([[True], [False]], device='cuda'))


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_explicit_zero_key_capacity_never_invokes_native_uninitialized_lse(query_layout):
    prepared = prepare(rows=0, query_layout=query_layout)
    q, k, v, mask = operands(prepared.spec)
    result = prepared.run(q, k, v, valid_keys=mask)
    assert torch.equal(result.output, torch.zeros_like(q))
    assert torch.isneginf(result.logsumexp).all() and not result.valid_rows.any()


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_current_stream_and_graph_replay_use_changed_key_masks_and_values(query_layout):
    prepared = prepare(rows=8, query_layout=query_layout)
    q, k, v, mask = operands(prepared.spec)
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        warm = prepared.run(q, k, v, valid_keys=mask)
    torch.cuda.current_stream().wait_stream(side)
    assert_single_key_identity(warm, 1)
    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph, stream=side):
            result = prepared.run(q, k, v, valid_keys=mask)
        torch.cuda.current_stream().wait_stream(side)
        addresses = (result.output.data_ptr(), result.logsumexp.data_ptr())
        mask.zero_()
        mask[0, :, 5] = True
        v[:, 5].fill_(13)
        graph.replay()
        assert_single_key_identity(result, 13)
        assert addresses == (result.output.data_ptr(), result.logsumexp.data_ptr())
    finally:
        torch.cuda.synchronize()
        graph.reset()


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_nonempty_nan_result_is_not_repaired_by_empty_owner_identity(query_layout):
    prepared = prepare(rows=3, query_layout=query_layout)
    q, k, v, mask = operands(prepared.spec)
    v[0, 1].fill_(math.nan)
    result = prepared.run(q, k, v, valid_keys=mask)
    assert torch.isnan(result.output[0]).any()
    assert torch.equal(result.output[1], torch.zeros_like(result.output[1]))


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_decoded_input_gradient_and_unfunded_mask_fail_before_native_dispatch(query_layout):
    prepared = prepare(rows=3, masked=False, query_layout=query_layout)
    q, k, v, mask = operands(prepared.spec)
    q.requires_grad_(True)
    with pytest.raises(ValueError, match='without gradients'):
        prepared.run(q, k, v)
    q.requires_grad_(False)
    with pytest.raises(ValueError, match='byte admission'):
        prepared.run(q, k, v, valid_keys=mask)


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
def test_singleton_lse_keeps_distinct_head_order_and_empty_owner_identity(query_layout, record_property):
    prepared = prepare(query_layout=query_layout)
    q, k, v, mask = operands(prepared.spec)
    q[0, 0, :, 0] = torch.arange(12, dtype=torch.float16, device='cuda') * 2
    k[0, 1, 0, 0] = 1
    result = prepared.run(q, k, v, valid_keys=mask)
    # Single selected value is independent of Q; singleton logsumexp is its
    # score h/8. Check the discrete head order and exact zero, not a new margin.
    assert torch.equal(result.output[0], torch.ones_like(result.output[0]))
    assert result.logsumexp.shape == (2, 12, 1)
    actual = result.logsumexp[0, :, 0]
    assert actual[0] == 0 and torch.isfinite(actual).all()
    assert (actual[1:] > actual[:-1]).all()
    assert torch.count_nonzero(result.output[1]) == 0 and torch.isneginf(result.logsumexp[1]).all()
    record_property('singleton_head_lse_diagnostic', json.dumps({
        'query_layout': query_layout, 'actual': actual.cpu().tolist(),
        'analytic': [head / 8 for head in range(12)], 'quality_accepted': False}))


@pytest.mark.parametrize('changes', [{'queries':2}, {'heads':6}, {'kv_heads':2}, {'masked':False}])
def test_ineligible_tile_regimes_keep_actual_native_gqa_request_and_outputs(changes):
    original = prepare(**changes)
    candidate = prepare(**changes, query_layout=backend.HEAD_QUERY_LAYOUT)
    assert not candidate.spec.heads_as_query_rows
    q, k, v, mask = operands(candidate.spec)
    kwargs = {'valid_keys':mask} if candidate.spec.masked else {}
    reference, result = original.run(q, k, v, **kwargs), candidate.run(q, k, v, **kwargs)
    # These requests are structurally identical, unlike the eligible new layout.
    assert torch.equal(reference.output.view(torch.int16), result.output.view(torch.int16))
    assert torch.equal(reference.logsumexp.view(torch.int32), result.logsumexp.view(torch.int32))


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
@pytest.mark.parametrize('codec', ['q8', 'iso4-working'])
def test_registered_working_operands_keep_exact_uniform_value_identity(codec, query_layout):
    from tensorfold.cuda.sm70 import family_ops, rotorquant_compact, rotorquant_q4_native
    count = 8  # The existing uniform-score oracle uses exactly represented1/8.
    if codec == 'q8':
        codes = torch.zeros((count, 1, 256), dtype=torch.int8, device='cuda')
        metadata = torch.full((count, 1, 8), 256., dtype=torch.float16, device='cuda')
        working = family_ops.dequantize_int8(codes, metadata)
        key_working = family_ops.dequantize_int8(codes, torch.zeros_like(metadata))
        assert torch.equal(working, torch.ones_like(working))
    else:
        # Decode registered codec10 into its working basis using the existing
        # independent compact reference. This does NOT admit an Iso4 owner.
        codes = (torch.arange(128, dtype=torch.int32, device='cuda') % 16 * 17).byte()
        codes = codes.expand(count, 1, 128).clone()
        metadata = torch.full((count, 1, 2), 0x3c00, dtype=torch.int32, device='cuda').view(torch.float32)
        working = torch.empty((count, 1, 256), dtype=torch.float16, device='cuda')
        status = torch.tensor([0,-2], dtype=torch.int32, device='cuda')
        rotorquant_q4_native.decode(codes, metadata, working, codec=10, status=status)
        zero_metadata = torch.zeros_like(metadata)
        key_working = torch.empty_like(working)
        rotorquant_q4_native.decode(codes, zero_metadata, key_working, codec=10, status=status)
        expected_key = rotorquant_compact.dequant_ref(codes.cpu(), zero_metadata.cpu(), codec=10).half()
        assert torch.equal(key_working.cpu().view(torch.int16), expected_key.view(torch.int16))
        expected = rotorquant_compact.dequant_ref(codes.cpu(), metadata.cpu(), codec=10).half()
        assert torch.equal(working.cpu().view(torch.int16), expected.view(torch.int16))
        assert status.cpu().tolist() == [0,-2]
    prepared = prepare(rows=count, query_layout=query_layout)
    q, _k, _v, mask = operands(prepared.spec)
    assert torch.count_nonzero(key_working) == 0
    k = key_working[None].expand(2, count, 1, 256).contiguous()
    v = working[None].expand(2, count, 1, 256).contiguous()
    mask[0].fill_(True)
    result = prepared.run(q, k, v, valid_keys=mask)
    # Equal values and exactly represented probability1/8 reuse the independent
    # uniform-value oracle. No cross-layout random-data tolerance is introduced.
    assert torch.equal(result.output[0, 0], working[0, 0].expand(12, 256))
    assert torch.isfinite(result.logsumexp[0]).all()
    assert torch.equal(result.logsumexp[0], result.logsumexp[0, :1].expand(12, 1))
    assert torch.count_nonzero(result.output[1]) == 0 and torch.isneginf(result.logsumexp[1]).all()


def test_eligible_head_rows_reject_gradient_and_per_head_validity_before_native_work():
    prepared = prepare(query_layout=backend.HEAD_QUERY_LAYOUT)
    assert prepared.spec.heads_as_query_rows
    q, k, v, mask = operands(prepared.spec)
    q.requires_grad_(True)
    with pytest.raises(ValueError, match='without gradients'):
        prepared.run(q, k, v, valid_keys=mask)
    q.requires_grad_(False)
    with pytest.raises(ValueError, match='validity'):
        prepared.run(q, k, v, valid_keys=mask.expand(2, 12, 3).contiguous())


@pytest.mark.parametrize('query_layout', [None, backend.HEAD_QUERY_LAYOUT])
@pytest.mark.parametrize('batch', [1,15,16])
def test_owner_key_width_and_existing_panel16_tail_preserve_singleton_answers(query_layout, batch):
    prepared = prepare(rows=backend.OWNER_MAX_KEYS, query_layout=query_layout, batch=batch)
    q, k, v, mask = operands(prepared.spec)
    mask.zero_()
    for row in range(batch):
        mask[row, 0, row] = True
    if batch > 1:
        mask[-1].zero_()
    result = prepared.run(q, k, v, valid_keys=mask)
    expected = torch.arange(batch, device='cuda', dtype=torch.float16)[:,None,None,None].expand_as(q).clone()
    active = batch if batch == 1 else batch - 1
    if batch > 1:
        expected[-1].zero_()
    assert torch.equal(result.output, expected)
    assert torch.equal(result.logsumexp[:active], torch.zeros_like(result.logsumexp[:active]))
    assert torch.isneginf(result.logsumexp[active:]).all()
    assert result.valid_rows[:active].all() and not result.valid_rows[active:].any()

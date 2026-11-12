"""Unselected ATen/CUTLASS tile integration, current-stream and capture controls."""

import math

import pytest
import torch

from tensorfold.cuda.sm70 import attention_backends as backend

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires actual SM70')


def prepare(*, rows=3, masked=True, queries=1, heads=12):
    spec = backend.TileSpec(2, queries, rows, heads, 1, .0625, masked=masked)
    return backend.prepare(spec, device=torch.device('cuda', torch.cuda.current_device()),
                           device_budget_bytes=backend.tensor_requirements(spec)['known_peak_tensor_bytes'])


def operands(spec):
    q = torch.zeros((spec.batch, spec.query_rows, spec.query_heads, 256), dtype=torch.float16, device='cuda')
    k = torch.zeros((spec.batch, spec.key_rows, 1, 256), dtype=torch.float16, device='cuda')
    v = torch.arange(spec.key_rows, dtype=torch.float16, device='cuda')[None, :, None, None].expand(
        spec.batch, spec.key_rows, 1, 256).contiguous()
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


def test_head256_gqa_masked_tail_and_empty_owner_obey_integer_identity():
    prepared = prepare(rows=3)
    q, k, v, mask = operands(prepared.spec)
    result = prepared.run(q, k, v, valid_keys=mask)
    assert_single_key_identity(result, 1)
    assert torch.equal(result.valid_rows, torch.tensor([[True], [False]], device='cuda'))


def test_explicit_zero_key_capacity_never_invokes_native_uninitialized_lse():
    prepared = prepare(rows=0)
    q, k, v, mask = operands(prepared.spec)
    result = prepared.run(q, k, v, valid_keys=mask)
    assert torch.equal(result.output, torch.zeros_like(q))
    assert torch.isneginf(result.logsumexp).all() and not result.valid_rows.any()


def test_current_stream_and_graph_replay_use_changed_key_masks_and_values():
    prepared = prepare(rows=8)
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
        mask.zero_()
        mask[0, :, 5] = True
        v[:, 5].fill_(13)
        graph.replay()
        assert_single_key_identity(result, 13)
    finally:
        torch.cuda.synchronize()
        graph.reset()


def test_nonempty_nan_result_is_not_repaired_by_empty_owner_identity():
    prepared = prepare(rows=3)
    q, k, v, mask = operands(prepared.spec)
    v[0, 1].fill_(math.nan)
    result = prepared.run(q, k, v, valid_keys=mask)
    assert torch.isnan(result.output[0]).any()
    assert torch.equal(result.output[1], torch.zeros_like(result.output[1]))


def test_decoded_input_gradient_and_unfunded_mask_fail_before_native_dispatch():
    prepared = prepare(rows=3, masked=False)
    q, k, v, mask = operands(prepared.spec)
    q.requires_grad_(True)
    with pytest.raises(ValueError, match='without gradients'):
        prepared.run(q, k, v)
    q.requires_grad_(False)
    with pytest.raises(ValueError, match='byte admission'):
        prepared.run(q, k, v, valid_keys=mask)

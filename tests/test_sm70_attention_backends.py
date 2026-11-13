"""Decoded-tile metadata/resource/boundary controls; no CUDA execution claim."""

import math
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda.sm70 import attention_backends as backend


def spec(**changes):
    return backend.TileSpec(**({'batch': 2, 'query_rows': 1, 'key_rows': 3,
        'query_heads': 12, 'kv_heads': 1, 'scale': .0625, 'masked': True} | changes))


@pytest.mark.parametrize('change', [dict(batch=0), dict(query_rows=True), dict(key_rows=-1),
    dict(query_heads=13, kv_heads=2), dict(head_dim=128), dict(masked=1),
    dict(scale=float('nan')), dict(scale=float('inf')), dict(scale=0), dict(scale=True),
    dict(scale=1e100), dict(scale=10**1000)])
def test_invalid_geometry_and_scale_fail_before_any_operator_or_device_work(change):
    with pytest.raises(ValueError):
        spec(**change)


def test_tensor_forecast_pays_native_accumulator_padded_lse_and_mask_publication():
    facts = backend.tensor_requirements(spec())
    expected = {'output_half_bytes': 2 * 1 * 12 * 256 * 2,
                'native_lse_padded_bytes': 2 * 12 * 32 * 4,
                'native_accumulator_bound_bytes': 2 * 1 * 12 * 256 * 4,
                'native_rng_placeholder_bytes': 16,
                'aligned_bias_bytes': 2 * 1 * 8 * 2,
                'mask_inversion_bytes': 2 * 1 * 3, 'validity_bytes': 2,
                'published_masked_output_bytes': 2 * 1 * 12 * 256 * 2,
                'published_masked_lse_bytes': 2 * 12 * 1 * 4}
    assert {name: facts[name] for name in expected} == expected
    assert facts['known_peak_tensor_bytes'] == sum(expected.values())
    assert 'allocator' in facts['scope']
    assert backend.tensor_requirements(spec(masked=False))['aligned_bias_bytes'] == 0


def test_explicit_zero_key_owner_requires_only_zero_identity_output_tensors():
    facts = backend.tensor_requirements(spec(key_rows=0))
    assert facts['native_accumulator_bound_bytes'] == facts['native_rng_placeholder_bytes'] == 0
    assert facts['known_peak_tensor_bytes'] == 2 * 12 * 256 * 2 + 2 * 12 * 4 + 2


def test_size_arithmetic_refuses_unaddressable_geometry():
    with pytest.raises(OverflowError):
        spec(batch=(1 << 31) - 1, query_rows=(1 << 31) - 1)


def test_actual_operator_schema_inspection_does_not_activate_cuda():
    initialized = torch.cuda.is_initialized()
    found = backend.operator_identity()
    assert found['schema'] == backend.SCHEMA
    assert found['cuda_registered'] is True and found['policy'] == backend.POLICY
    assert torch.cuda.is_initialized() is initialized


def test_schema_mismatch_is_not_a_fallback_to_another_attention_operator(monkeypatch):
    monkeypatch.setattr(torch.ops.aten._efficient_attention_forward, 'default',
                        SimpleNamespace(_schema='another ABI'))
    with pytest.raises(ValueError, match='schema/CUDA registration'):
        backend.operator_identity()


def test_byte_admission_refuses_before_device_work(monkeypatch):
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: pytest.fail('CUDA must remain untouched'))
    needed = backend.tensor_requirements(spec())['known_peak_tensor_bytes']
    with pytest.raises(MemoryError, match='byte credit'):
        backend.prepare(spec(), device='cuda:0', device_budget_bytes=needed - 1)


def test_preparation_checks_actual_device_once_with_no_native_invocation(monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: calls.append(device) or (7, 0))
    prepared = backend.prepare(spec(), device='cuda:0',
        device_budget_bytes=backend.tensor_requirements(spec())['known_peak_tensor_bytes'])
    assert prepared.spec == spec() and calls == [torch.device('cuda:0')]
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (8, 0))
    with pytest.raises(ValueError, match='actual SM70'):
        backend.prepare(spec(), device='cuda:0', device_budget_bytes=prepared.device_budget_bytes)


def test_cpu_inputs_are_refused_without_native_dispatch(monkeypatch):
    s = spec()
    prepared = backend.Prepared(s, torch.device('cuda:0'),
        backend.tensor_requirements(s)['known_peak_tensor_bytes'], {})
    q = torch.zeros((2, 1, 12, 256), dtype=torch.float16)
    kv = torch.zeros((2, 3, 1, 256), dtype=torch.float16)
    monkeypatch.setattr(torch.ops.aten._efficient_attention_forward, 'default',
                        lambda *args, **kwargs: pytest.fail('invalid inputs reached native dispatch'))
    with pytest.raises(ValueError, match='contiguous CUDA FP16'):
        prepared.run(q, kv, kv, valid_keys=torch.ones((2, 1, 3), dtype=torch.bool))


def test_lazy_negative_tile_is_not_an_uncharged_aten_materialization():
    tensor = torch.ones((2, 1, 12, 256), dtype=torch.float16)
    negative = torch._neg_view(tensor)
    assert negative.is_contiguous() and negative.is_neg()
    assert backend._plain_tile(tensor, tuple(tensor.shape), torch.float16, torch.device('cpu'))
    assert not backend._plain_tile(negative, tuple(tensor.shape), torch.float16, torch.device('cpu'))


def test_explicit_empty_validity_is_the_only_zero_identity_publication():
    output = torch.full((1, 2, 1, 256), math.nan, dtype=torch.float16)
    lse = torch.full((1, 1, 2), math.nan, dtype=torch.float32)
    valid = torch.tensor([[True, False]])
    result = backend._publish_masked(output, lse, valid)
    assert torch.isnan(result.output[:, 0]).all() and torch.isnan(result.logsumexp[:, :, 0]).all()
    assert torch.equal(result.output[:, 1], torch.zeros_like(result.output[:, 1]))
    assert torch.isneginf(result.logsumexp[:, :, 1]).all()
    assert result.valid_rows is valid


@pytest.mark.parametrize('layout', [None, backend.NATIVE_QUERY_LAYOUT, backend.HEAD_QUERY_LAYOUT])
def test_query_layout_credit_retains_native_gqa_bound(layout):
    assert backend.tensor_requirements(spec(query_layout=layout)) == backend.tensor_requirements(spec())


@pytest.mark.parametrize('layout', [True, 1, {}, 'unknown'])
def test_unknown_query_layout_fails_before_operator_work(layout):
    with pytest.raises(ValueError):
        spec(query_layout=layout)


def _cpu_request(monkeypatch, selected, *, nonfinite=False):
    """Exercise owned mapping/publication with real CPU tensors, stub only CUDA op.

    The stub echoes rows and supplies distinct LSE entries plus poisoned padding;
    it is a layout oracle, not an implementation of attention or CUDA evidence.
    """
    q = torch.arange(selected.batch * selected.query_rows * selected.query_heads * 256,
                     dtype=torch.float32).half().view(selected.batch, selected.query_rows, selected.query_heads, 256)
    k = torch.zeros((selected.batch, selected.key_rows, selected.kv_heads, 256), dtype=torch.float16)
    v = torch.ones_like(k)
    valid = torch.ones((selected.batch, selected.query_rows, selected.key_rows), dtype=torch.bool)
    valid[-1].zero_()
    calls = []
    def operator(query, key, value, bias, seq_q, seq_k, max_q, max_k, dropout, custom_mask, lse_needed, **kwargs):
        calls.append((query, key, value, bias))
        assert seq_q is seq_k is max_q is max_k is None
        assert dropout == 0 and custom_mask == 0 and lse_needed is True
        assert kwargs == {'scale': .0625}
        native_rows, native_heads = query.shape[1:3]
        lse = torch.full((selected.batch, native_heads, (native_rows + 31) // 32 * 32), math.nan)
        lse[:, :, :native_rows].copy_(query[..., 0].transpose(1, 2).float())
        output = query.clone()
        if nonfinite:
            output[0].fill_(math.nan)
            lse[0, :, :native_rows].fill_(math.nan)
        return output, lse, None, None, native_rows, selected.key_rows
    monkeypatch.setattr(torch.ops.aten._efficient_attention_forward, 'default', operator)
    prepared = backend.Prepared(selected, torch.device('cpu'),
        backend.tensor_requirements(selected)['known_peak_tensor_bytes'], {})
    result = prepared.run(q, k, v, valid_keys=valid if selected.masked else None)
    return q, k, v, valid, result, calls


@pytest.mark.parametrize('batch', [1, 2, 17])
def test_fixed12_head_rows_preserves_batch_head_values_lse_and_shared_mask(monkeypatch, batch):
    selected = spec(batch=batch, query_layout=backend.HEAD_QUERY_LAYOUT)
    q, k, v, valid, result, calls = _cpu_request(monkeypatch, selected)
    assert selected.heads_as_query_rows and len(calls) == 1
    nq, nk, nv, bias = calls[0]
    assert nq.shape == (batch, 12, 1, 256) and nq.data_ptr() == q.data_ptr()
    assert nk is k and nv is v  # No KV replication/materialization.
    assert bias.shape == (batch, 1, 12, 3) and bias.stride(2) == 0
    assert bias.untyped_storage().nbytes() == batch * 8 * 2  # One aligned shared mask.
    for head in range(12):
        assert torch.equal(bias[:, 0, head], torch.where(valid[:, 0], 0., -math.inf).half())
    assert result.output.shape == q.shape and result.logsumexp.shape == (batch, 12, 1)
    assert torch.equal(result.output[:-1], q[:-1])
    assert torch.equal(result.logsumexp[:-1], q[:-1, :, :, 0].transpose(1, 2).float())
    assert torch.equal(result.output[-1], torch.zeros_like(q[-1]))
    assert torch.isneginf(result.logsumexp[-1]).all()


@pytest.mark.parametrize('changes', [{'query_rows':2}, {'query_heads':6}, {'kv_heads':2}, {'masked':False}])
def test_fixed12_selector_keeps_other_valid_tile_regimes_on_native_gqa(monkeypatch, changes):
    selected = spec(query_layout=backend.HEAD_QUERY_LAYOUT, **changes)
    q, _k, _v, _valid, result, calls = _cpu_request(monkeypatch, selected)
    assert not selected.heads_as_query_rows and calls[0][0] is q
    assert result.output.shape == q.shape
    assert result.logsumexp.shape == (selected.batch, selected.query_heads, selected.query_rows)


def test_fixed12_nonempty_nan_is_not_repaired_by_masked_empty_publication(monkeypatch):
    _q, _k, _v, _valid, result, _calls = _cpu_request(monkeypatch,
        spec(query_layout=backend.HEAD_QUERY_LAYOUT), nonfinite=True)
    assert torch.isnan(result.output[0]).all() and torch.isnan(result.logsumexp[0]).all()
    assert torch.count_nonzero(result.output[1]) == 0 and torch.isneginf(result.logsumexp[1]).all()


def test_fixed12_rejects_per_head_masks_before_any_native_request(monkeypatch):
    selected = spec(query_layout=backend.HEAD_QUERY_LAYOUT)
    prepared = backend.Prepared(selected, torch.device('cpu'),
        backend.tensor_requirements(selected)['known_peak_tensor_bytes'], {})
    q = torch.zeros((2, 1, 12, 256), dtype=torch.float16)
    kv = torch.zeros((2, 3, 1, 256), dtype=torch.float16)
    monkeypatch.setattr(torch.ops.aten._efficient_attention_forward, 'default',
                        lambda *a, **k: pytest.fail('per-head mask entered native work'))
    with pytest.raises(ValueError, match='validity'):
        prepared.run(q, kv, kv, valid_keys=torch.ones((2, 12, 3), dtype=torch.bool))


def test_fixed12_zero_key_owner_retains_original_empty_identity(monkeypatch):
    selected = spec(key_rows=0, query_layout=backend.HEAD_QUERY_LAYOUT)
    prepared = backend.Prepared(selected, torch.device('cpu'),
        backend.tensor_requirements(selected)['known_peak_tensor_bytes'], {})
    q = torch.zeros((2, 1, 12, 256), dtype=torch.float16)
    kv = torch.zeros((2, 0, 1, 256), dtype=torch.float16)
    monkeypatch.setattr(torch.ops.aten._efficient_attention_forward, 'default',
                        lambda *a, **k: pytest.fail('empty owner entered native work'))
    result = prepared.run(q, kv, kv, valid_keys=torch.zeros((2, 1, 0), dtype=torch.bool))
    assert torch.count_nonzero(result.output) == 0 and torch.isneginf(result.logsumexp).all()

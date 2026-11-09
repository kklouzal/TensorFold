"""Paid opposite-half query packing; original row reshape is the oracle."""
import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.history_owner import _reference_query_rows

pytestmark = pytest.mark.torch


def query_slab(rows, device):
    # Cyclic raw Half encodings include zeros/subnormals/Inf/all NaN payloads.
    words = (torch.arange(8 * rows * 3 * 256, dtype=torch.int32, device=device) % 65536).to(torch.int16)
    return words.view(torch.float16).view(8, rows, 3, 256)


def oracle(query, group):
    return torch.stack([query[group:group + 4, row].reshape(12, 256) for row in range(query.shape[1])])


@pytest.mark.parametrize('rows', [1, 2, 3, 5, 40, 160, 256])
@pytest.mark.parametrize('group', [0, 4])
def test_raw_bits_active_half_and_single_copy_or_zero_copy(rows, group, monkeypatch):
    query = query_slab(rows, 'cpu')
    before = query.view(torch.int16).clone()
    expected = oracle(query, group).view(torch.int16)
    integer_expected = oracle(before, group)
    storage = query.untyped_storage().data_ptr()
    calls = []
    copy = torch.Tensor.copy_
    def copied(destination, source, *args, **kwargs):
        calls.append(destination.data_ptr())
        return copy(destination, source, *args, **kwargs)
    def allocation(*args, **kwargs):
        pytest.fail('query packing allocated instead of borrowing the opposite half')
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, 'copy_', copied)
        patch.setattr(torch.Tensor, 'clone', allocation)
        patch.setattr(torch, 'empty', allocation)
        patch.setattr(torch, 'empty_like', allocation)
        patch.setattr(torch, 'empty_strided', allocation)
        actual = _reference_query_rows(query, group)
    assert len(calls) == int(rows > 1)
    assert actual.untyped_storage().data_ptr() == storage
    assert actual.shape == (rows, 12, 256) and actual.stride() == (3072, 256, 1)
    assert torch.equal(actual.view(torch.int16), expected)
    assert torch.equal(actual.view(torch.int16), integer_expected)
    assert torch.equal(query[group:group + 4].view(torch.int16), before[group:group + 4])
    other = 4 - group
    if rows == 1:
        assert actual.data_ptr() == query[group].data_ptr()
        assert torch.equal(query.view(torch.int16), before)
    else:
        assert actual.data_ptr() == query[other].data_ptr()
        assert torch.equal(query[other:other + 4].view(rows, 12, 256).view(torch.int16), expected)


@pytest.mark.parametrize('group', [-4, 1, 8, True])
def test_invalid_group_refuses_without_mutation(group):
    query = query_slab(3, 'cpu')
    before = query.view(torch.int16).clone()
    with pytest.raises(ValueError, match='rank-major'):
        _reference_query_rows(query, group)
    assert torch.equal(query.view(torch.int16), before)


def test_actual_private_caller_packs_once_and_reuses_rows(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import history_owner as module
    from tests.test_sm70_global_normalization import test_actual_reference_layer_selects_batch_and_preserves_four_phase_order
    calls = []
    original = module._reference_query_rows
    def packed(query, group):
        calls.append((query.shape, group))
        return original(query, group)
    monkeypatch.setattr(module, '_reference_query_rows', packed)
    test_actual_reference_layer_selects_batch_and_preserves_four_phase_order(monkeypatch)
    assert calls == [(torch.Size([8, 5, 3, 256]), 0)]


def test_all_empty_private_jobs_do_not_pack(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import history_owner as module
    from tests.test_sm70_global_normalization import test_actual_reference_layer_selects_batch_and_preserves_four_phase_order
    monkeypatch.setattr(module, '_PRIVATE_EMPTY_PREFIX', 'cpu-empty-v1')
    monkeypatch.setattr(module, '_first_page_empty', lambda *a: True)
    monkeypatch.setattr(module, '_reference_query_rows', lambda *a: pytest.fail('all-empty rows gained a copy'))
    test_actual_reference_layer_selects_batch_and_preserves_four_phase_order(monkeypatch)


def test_default_none_caller_keeps_original_row_path(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import history_owner as module
    from tests.test_flashnext_history_placement import test_real_owner_layer_routes_both_tp_head_groups_with_batched_collectives
    monkeypatch.setattr(module, '_reference_query_rows', lambda *a: pytest.fail('legacy selection packed queries'))
    test_real_owner_layer_routes_both_tp_head_groups_with_batched_collectives(4, 'int8', 8, 0)

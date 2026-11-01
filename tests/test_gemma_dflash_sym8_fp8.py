"""DF-only private-copy adapter payload/ownership controls, no physical CUDA."""
from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda import b16_sym8 as sym
from tensorfold.cuda.b16_sym8_layout import SCHEMA
from tensorfold.cuda.requant import requantize
from tensorfold.families.gemma4.cuda.dflash import _sym8_fp8_head


def factory_fixture(monkeypatch, *, fail_lookup=None, fail_fence=False):
    # Actual dtype quantization/permutation runs on CPU. Only CUDA allocation,
    # synchronization and raw-row callback boundaries are represented here.
    source = torch.arange(5 * 64, dtype=torch.float32).reshape(5, 64).to(torch.bfloat16) / 17
    source = source.to(torch.bfloat16)
    calls = []
    primary = RuntimeError('synthetic native factory failure')
    primary.__cause__ = ValueError('native root')
    class Table:
        layout, n, k, lookup_limit = SCHEMA, 5, 64, 2
        device = torch.device('cuda')
        def embedding(self, ids):
            calls.append(('lookup', ids.tolist()))
            if fail_lookup is not None and len([c for c in calls if c[0] == 'lookup']) == fail_lookup:
                raise primary
            return source[ids]
    monkeypatch.setattr(sym, 'Sym8Table', Table)
    empty, arange = torch.empty, torch.arange
    def allocate(*args, **kwargs):
        if torch.device(kwargs.get('device', 'cpu')).type == 'cuda':
            kwargs['device'] = 'cpu'
        return empty(*args, **kwargs)
    def ids(*args, **kwargs):
        if torch.device(kwargs.get('device', 'cpu')).type == 'cuda':
            kwargs['device'] = 'cpu'
        return arange(*args, **kwargs)
    def fence(device):
        calls.append(('fence', str(device)))
        if fail_fence:
            raise primary
    monkeypatch.setattr(torch, 'empty', allocate)
    monkeypatch.setattr(torch, 'arange', ids)
    monkeypatch.setattr(torch.cuda, 'synchronize', fence)
    return Table(), source, calls, primary


def test_chunked_copy_matches_existing_raw_source_fp8_codes_scales(monkeypatch):
    table, original, calls, _ = factory_fixture(monkeypatch)
    result = _sym8_fp8_head(table, softcap=30.)
    expected = requantize(original, 'fp8')
    assert torch.equal(result.linear.w8, expected.w8)
    assert torch.equal(result.linear.cols.view(torch.int32), expected.cols.view(torch.int32))
    assert result.softcap == 30. and not hasattr(table, 'weight')
    assert [row[1] for row in calls if row[0] == 'lookup'] == [[0, 1], [2, 3], [4]]
    assert len([row for row in calls if row[0] == 'fence']) == 4  # three chunks + packed-copy handoff


@pytest.mark.parametrize('kind', ['lookup', 'fence'])
def test_native_failure_never_retries_a_fence_and_retains_actual_frame_owners(monkeypatch, kind):
    table, _, calls, primary = factory_fixture(monkeypatch, fail_lookup=1 if kind == 'lookup' else None,
                                               fail_fence=kind == 'fence')
    with pytest.raises(RuntimeError) as caught:
        _sym8_fp8_head(table, softcap=30.)
    assert caught.value is primary and isinstance(primary.__cause__, ValueError)
    assert len([row for row in calls if row[0] == 'fence']) == (1 if kind == 'fence' else 0)
    tb = primary.__traceback__
    while tb is not None and tb.tb_frame.f_code.co_name != '_sym8_fp8_head':
        tb = tb.tb_next
    assert tb is not None and tb.tb_frame.f_locals['codes'] is not None
    assert tb.tb_frame.f_locals['scales'] is not None
    assert tb.tb_frame.f_locals['table'] is table


def test_layout_label_without_actual_owned_type_is_refused():
    with pytest.raises(ValueError, match='actual raw Sym8'):
        _sym8_fp8_head(SimpleNamespace(layout=SCHEMA, k=64), softcap=30.)

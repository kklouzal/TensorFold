"""Real-CUDA private-witness controls; diagnostic, never throughput evidence."""
import json

import numpy as np
import pytest
import torch

from tools import v100_owned_qualification as owned
from tests.test_v100_normalization_parity import fixture
from tests.test_sm70_global_normalization import scalar_oracle, assert_bits

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires the pinned actual SM70 diagnostic runtime')


@pytest.mark.parametrize('rows,group', [(1, 0), (5, 4), (160, 0)])
def test_actual_cuda_same_input_return_status_and_retirement(tmp_path, monkeypatch, rows, group):
    engine, witness, module, data, local, output, ranges, made, _, failures = fixture(
        tmp_path, monkeypatch, device='cuda', rows=rows, maximum=256, group=group,
        phase=6 if rows == 1 else 0, capture_writer=owned._save_router_tensor)
    old_ranges = [(s.clone(), first, last) for s, first, last in ranges]
    expected = scalar_oracle(data, group, old_ranges, 17)
    result = module.merge_global_statistics(data, local, output, group, ranges, 17)
    assert result is made[0]
    for actual, old in zip(result, expected):
        assert_bits(actual, old)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(ranges, old_ranges))
    assert not failures and not witness.failed
    assert witness.published['geometry']['decode'] == {'min_rows': rows, 'max_rows': rows}
    assert engine.close() is engine.token
    assert not witness.roles and witness.engine is None
    record = json.loads((witness.directory / 'summary.json').read_text())
    assert record['closed'] and not record['failed'] and record['diagnostic_only']
    assert record['counts']['decode:17']['calls'] == 1


def test_actual_cuda_wrong_oracle_bit_saves_all_binary_evidence_before_abort(tmp_path, monkeypatch):
    engine, witness, module, data, local, output, ranges, _, _, failures = fixture(
        tmp_path, monkeypatch, device='cuda', rows=5, maximum=256, group=4,
        fault='value', capture_writer=owned._save_router_tensor)
    before = torch.stack([s for s, _, _ in ranges]).cpu().numpy().copy()
    with pytest.raises(RuntimeError, match='raw-bit/status parity mismatch') as caught:
        module.merge_global_statistics(data, local, output, 4, ranges, 17)
    assert failures == [caught.value] and witness.failed
    record = json.loads((witness.directory / 'mismatch.json').read_text())
    assert len(record['files']) == 9
    assert len(list(witness.directory.glob('*.npy'))) == 9
    statuses = np.load(witness.directory / 'status-before.npy', allow_pickle=False).view(np.int32)
    np.testing.assert_array_equal(statuses, before)
    for info in record['files'].values():
        assert (witness.directory / info['file']).stat().st_size == info['bytes']
    engine.close()

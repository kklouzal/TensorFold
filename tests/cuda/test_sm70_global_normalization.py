"""Exact SM70 global normalization component; no model/P3 qualification claim."""
import pytest
import torch

from tests.test_sm70_global_normalization import aligned_workspace, sample, statuses, scalar_oracle, assert_bits
from tensorfold.cuda.sm70.reference_owner_tiles import merge_global_statistics

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='requires actual pinned SM70 normalization')


@pytest.mark.parametrize('rows', [1, 5, 40, 160, 256])
@pytest.mark.parametrize('group', [0, 4])
def test_actual_sm70_scalar_oracle_and_receive_to_PV_lifetime(rows, group):
    for phase in (range(8) if rows == 1 else (0, 5) if rows == 5 else (0,)):
        data = sample(rows, 'cuda', group, phase=phase)
        workspace = aligned_workspace(rows, 'cuda')
        output = torch.full((rows, 12, 2), 19., device='cuda')
        expected_status, actual_status = statuses(rows, 'cuda'), statuses(rows, 'cuda')
        expected = scalar_oracle(data, group, expected_status, 37)
        actual = merge_global_statistics(data, workspace, output, group, actual_status, 37)
        # No fence/readback before the original phase's next queued writers.
        data.fill_(float('nan'))
        workspace.fill_(-777.)
        for a, e in zip(actual, expected):
            assert_bits(a, e)
        for (a, _, _), (e, _, _) in zip(actual_status, expected_status):
            assert torch.equal(a, e)


@pytest.mark.parametrize('rows', [1, 5, 40, 160, 256])
def test_actual_sm70_graph_replay_uses_current_mass_and_sticky_status(rows):
    data = sample(rows, 'cuda', 0)
    workspace = aligned_workspace(rows, 'cuda')
    output = torch.empty((rows, 12, 2), device='cuda')
    ranges = statuses(rows, 'cuda')
    merge_global_statistics(data, workspace, output, 0, ranges, 17)
    capture = torch.cuda.CUDAGraph()
    with torch.cuda.graph(capture):
        actual = merge_global_statistics(data, workspace, output, 0, ranges, 17)
    for group in (0, 4):
        data.copy_(sample(rows, 'cuda', group).roll(-group, dims=0))
        for status, _, _ in ranges:
            status.copy_(torch.tensor([0, -2], dtype=torch.int32, device='cuda'))
        expected_ranges = statuses(rows, 'cuda')
        for status, _, _ in expected_ranges:
            status.copy_(torch.tensor([0, -2], dtype=torch.int32, device='cuda'))
        expected = scalar_oracle(data, 0, expected_ranges, 17)
        capture.replay()
        for a, e in zip(actual, expected):
            assert_bits(a, e)
        for (a, _, _), (e, _, _) in zip(ranges, expected_ranges):
            assert torch.equal(a, e)

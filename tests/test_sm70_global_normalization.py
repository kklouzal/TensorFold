"""Independent scalar-merge oracle for paid-slab global normalization batching."""
import pytest
import torch

from tensorfold.cuda.sm70.reference_owner_tiles import merge_global_statistics
from tensorfold.families.qwen4_exp.cuda.history_owner import merge_statistics

pytestmark = pytest.mark.torch


def aligned_workspace(rows, device):
    allocation = torch.empty(rows * 12 * 256 + 128, dtype=torch.float32, device=device)
    shift = (-allocation.data_ptr() % 512) // 4
    return allocation[shift:shift + rows * 12 * 256]


def sample(rows, device, group, phase=0):
    torch.manual_seed(2157 + rows)
    data = torch.randn((8, rows, 12, 2), dtype=torch.float32, device=device)
    data[..., 0].mul_(10)
    data[..., 1].abs_().add_(.1)
    peers = data[group:group + 4]
    # Finite extremes, all-empty, nonfinite input, zero signs and failed mass.
    for row in range(rows):
        case = (row + phase) % 8
        if case == 0:
            peers[:, row, :, 0] = -torch.inf
            peers[:, row, :, 1] = 0
        elif case == 1:
            peers[0, row, 0, 0] = float('nan')
            peers[1, row, 1, 1] = float('nan')
        elif case == 2:
            peers[:, row, :, 0] = -0.
            peers[1, row, :, 0] = 0.
            peers[:, row, :, 1] = -0.
        elif case == 3:
            peers[0, row, :, 0] = torch.inf
            peers[1, row, :, 1] = torch.inf
        elif case == 4:
            peers[:, row, :, 0] = torch.tensor([-1e20, 1e20, -1e10, 0.], device=device)[:, None]
        elif case == 5:
            peers[:, row, :, 0] = 0
            peers[:, row, :, 1] = torch.tensor([2**24, 1, -(2**24), 3], device=device)[:, None]
            peers[:, row, 0, 1] = -1
        elif case == 7:
            peers[:, row, :, 0] = 0
            peers[:, row, :, 1] = 2**-145
    return data


def statuses(rows, device):
    ends = sorted(set([0, min(2, rows), min(7, rows), rows]))
    return [(torch.tensor(([2, 9] if i == 1 else [0, -2]), dtype=torch.int32, device=device), first, last)
            for i, (first, last) in enumerate(zip(ends, ends[1:]))]


def scalar_oracle(data, group, ranges, layer):
    result = []
    for row in range(data.shape[1]):
        status = next(status for status, first, last in ranges if first <= row < last)
        pieces = data[group:group + 4, row]
        result.append(merge_statistics(pieces[..., 0], pieces[..., 1], status=status, layer=layer))
    return torch.stack([m for m, _ in result]), torch.stack([s for _, s in result])


def assert_bits(actual, expected):
    assert torch.equal(actual.contiguous().view(torch.int32), expected.contiguous().view(torch.int32))


@pytest.mark.parametrize('rows', [1, 5, 40, 160, 256])
@pytest.mark.parametrize('group', [0, 4])
def test_global_batch_matches_unchanged_scalar_shapes_status_and_storage(rows, group):
    data = sample(rows, 'cpu', group)
    before = data.clone()
    expected_status, actual_status = statuses(rows, 'cpu'), statuses(rows, 'cpu')
    expected = scalar_oracle(data, group, expected_status, 37)
    workspace = aligned_workspace(rows, 'cpu')
    output = torch.full((rows + 3, 12, 2), 19., dtype=torch.float32)
    actual = merge_global_statistics(data, workspace, output, group, actual_status, 37)
    for a, e in zip(actual, expected):
        assert_bits(a, e)
    for (a, _, _), (e, _, _) in zip(actual_status, expected_status):
        assert torch.equal(a, e)
    assert_bits(data, before)
    planes = output.view(2, rows + 3, 12)
    assert torch.equal(planes[:, rows:], torch.full_like(planes[:, rows:], 19.))
    # The next reference tile overwrites receive storage, then PV overwrites
    # local storage. Final normalization must own neither temporary slab.
    data.fill_(float('nan'))
    workspace.fill_(-777.)
    for a, e in zip(actual, expected):
        assert_bits(a, e)


@pytest.mark.parametrize('failure', ['input-alias', 'output-alias', 'status-alias', 'missing-range', 'small-workspace'])
def test_invalid_phase_storage_refuses_before_writing(failure):
    rows = 5
    data = sample(rows, 'cpu', 0)
    workspace = aligned_workspace(rows, 'cpu')
    output = torch.full((rows, 12, 2), 19.)
    ranges = statuses(rows, 'cpu')
    if failure == 'input-alias':
        workspace = data.view(-1)
    elif failure == 'output-alias':
        output = workspace[:rows * 24].view(rows, 12, 2)
    elif failure == 'status-alias':
        ranges = [(workspace[:2].view(torch.int32), 0, rows)]
    elif failure == 'missing-range':
        ranges = ranges[:-1]
    else:
        workspace = workspace[:10]
    before = output.clone()
    with pytest.raises(ValueError):
        merge_global_statistics(data, workspace, output, 0, ranges, 1)
    assert_bits(output, before)


def test_reference_preparation_records_required_batching_and_retires_views():
    from tensorfold.cuda.sm70.reference_owner_tiles import ReferenceTiles, global_normalization_workspace_bytes

    received = aligned_workspace(8 * 5, 'cpu')
    local = aligned_workspace(5, 'cpu')
    output = torch.empty((5, 12, 2))
    preparer = ReferenceTiles(received)
    assert not preparer.record()['global_normalization']['prepared']
    preparer.bind_global_statistics(local, output)
    record = preparer.record()['global_normalization']
    assert record == {'policy': 'scalar-reductions-batched-pointwise-v1', 'prepared': True,
                      'additional_device_bytes': 0, 'rows': 5}
    assert preparer.normalization_workspace is local and preparer.normalization_output is output
    assert global_normalization_workspace_bytes(5) <= local.nbytes
    with pytest.raises(ValueError, match='fresh disjoint'):
        preparer.bind_global_statistics(local, output)
    preparer.begin_retirement()
    preparer.release_views()
    assert preparer.normalization_workspace is preparer.normalization_output is None


def test_selected_reference_owner_cannot_silently_fall_back_without_global_slab():
    from types import SimpleNamespace as NS
    from tensorfold.families.qwen4_exp.cuda.history_owner import _attention_layer

    def forbidden(*args):
        pytest.fail('unprepared normalization entered device or collective work')
    owner = NS(reference_tiles=NS(normalization_workspace=None, normalization_output=None),
               global_stats=None, partial=aligned_workspace(1, 'cpu'), wait_native_reuse=forbidden)
    weights = NS(cfg=NS(), meta={'rank': 0}, comm=NS(world=8, rank=0, all_gather=forbidden))
    buffers = NS(history_owner=owner, owner_kernel_family='torch-reference',
                 owner_graph_family='uncaptured', owner_graphs=None)
    with pytest.raises(RuntimeError, match='global normalization was not prepared'):
        _attention_layer(None, weights, [], buffers, 0)


def test_valid_mass_keeps_unrelated_error_and_unset_layer_unchanged():
    data = torch.zeros((8, 5, 12, 2))
    data[..., 1] = 1
    ranges = statuses(5, 'cpu')
    ranges[0][0].copy_(torch.tensor([1, -2], dtype=torch.int32))
    before = [s.clone() for s, _, _ in ranges]
    out = merge_global_statistics(data, aligned_workspace(5, 'cpu'), torch.empty((5, 12, 2)), 4, ranges, 19)
    assert_bits(out[0], torch.zeros((5, 12)))
    assert_bits(out[1], torch.full((5, 12), 4.))
    for (status, _, _), expected in zip(ranges, before):
        assert torch.equal(status, expected)


def test_actual_reference_layer_selects_batch_and_preserves_four_phase_order(monkeypatch):
    from types import SimpleNamespace as NS
    from tensorfold.families.qwen4_exp.cuda import history_owner as h, compact_indexer
    from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement
    from tensorfold.cuda.sm70 import reference_owner_tiles as tile_module

    rows, events = 5, []
    def gather(send, recv):
        events.append('gather')
        recv.copy_(send[None].expand_as(recv))
        if events.count('gather') == 3:
            recv[..., 0].fill_(-torch.inf)
            recv[..., 1].zero_()
            recv[1, ..., 0].zero_()
            recv[1, ..., 1].fill_(4)
        elif events.count('gather') == 4:
            recv.zero_()
            recv[1].fill_(2)
    cfg = NS(heads=3, head_dim=256, index_budget=2048, index_ratio=4)
    weights = NS(cfg=cfg, device=torch.device('cpu'), meta={'rank': 0, 'execution_provider': 'sm70'},
                 comm=NS(world=8, rank=0, all_gather=gather))
    owner = h.OwnerBuffers(weights, rows, NS(idw=2052))
    owner.partial = aligned_workspace(rows, 'cpu').view(rows, 12, 256)
    owner.partials = aligned_workspace(8 * rows, 'cpu').view(8, rows, 12, 256)
    owner.global_stats = torch.empty((rows, 12, 2))
    owner.reference_tiles = tile_module.ReferenceTiles(owner.partials)
    owner.reference_tiles.bind_global_statistics(owner.partial, owner.global_stats)
    state = NS(history_placement=HistoryPlacement(8, 2, 12), history_slot=1, pos=0,
               att_index={0: 0}, graph_rows=None, kv_status=torch.tensor([0, -2], dtype=torch.int32),
               history_generations=torch.tensor([0, 0], dtype=torch.int64), slot_generation=0, history_generation=0,
               kc=[NS(capacity=0, value_format=NS(codec=0, quantized=False))])
    buffers = NS(history_owner=owner, owner_kernel_family='torch-reference', owner_graph_family='uncaptured',
                 owner_graphs=None, q=torch.zeros((rows, 3, 256)), pa=None,
                 compact_indexer_stage=NS(begin=lambda **kw: events.append('stage')),
                 attn_o=torch.empty((rows, 3, 256)))
    monkeypatch.setattr(compact_indexer, 'validate_target_indexer_frame', lambda *a: events.append('validate'))
    monkeypatch.setattr(h, 'prepare', lambda *a, **kw: events.append('prepare'))
    monkeypatch.setattr(h, 'merge_statistics', lambda *a, **kw: pytest.fail('selected owner fell back to scalar merge'))
    original = tile_module.merge_global_statistics
    def batch(*a):
        assert events.count('gather') == 3
        events.append('batch')
        return original(*a)
    monkeypatch.setattr(tile_module, 'merge_global_statistics', batch)
    actual = h._attention_layer(NS(index=0), weights, [(state, 0, rows)], buffers, rows)
    assert events == ['validate', 'stage', 'prepare', 'gather', 'gather', 'gather', 'batch', 'gather']
    assert_bits(actual.float(), torch.full((rows, 3, 256), 2.))
    assert state.kv_status.tolist() == [0, -2]

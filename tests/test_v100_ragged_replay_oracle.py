"""CPU controls for the native oracle's fixed workload and exact comparator.

These do not execute or qualify graph replay. The real idle TP8 decoder is the
only execution path of the native component gate.
"""
import pytest
import torch

from tools.v100_ragged_replay_oracle import LAYOUTS, SEED_LENGTHS, _equal


def test_layouts_preserve_requested_key_and_change_cold_last_row_maps():
    seed = (4, 4, 4, 4, 3, 3, 3, 3)
    maps = set()
    for counts in (seed, *LAYOUTS):
        assert len(counts) == 8 and sum(counts) == 28
        assert min(counts) >= 1 and max(counts) <= 5
        cursor, last = 0, []
        for count in counts:
            cursor += count
            last.append(cursor - 1)
        maps.add(tuple(last))
    assert len(maps) == 3


def test_exact_comparator_accepts_independent_same_values():
    expected = {'logits': torch.tensor([0.0, 1.0]), 'pos': 3}
    _equal({'logits': expected['logits'].clone(), 'pos': 3}, expected)


@pytest.mark.parametrize('actual', [
    {'logits': torch.tensor([0.0, 1.000001]), 'pos': 3},
    {'logits': torch.tensor([0.0, 1.0]), 'pos': 4},
    {'logits': torch.tensor([0.0, 1.0])},
    {'logits': torch.tensor([float('nan'), 1.0]), 'pos': 3},
    {'logits': torch.tensor([-0.0, 1.0]), 'pos': 3},
])
def test_exact_comparator_refuses_changed_output_metadata_or_nonfinite(actual):
    with pytest.raises(AssertionError):
        _equal(actual, {'logits': torch.tensor([0.0, 1.0]), 'pos': 3})


def test_fixed_operation_input_is_closed_bounded_and_duplicate_free(tmp_path):
    import json

    from tools.v100_owned_qualification import _ragged_input
    data = {'schema': 'tensorfold.native-ragged-replay-input-v1', 'key': [28, 8],
            'layouts': [list(counts) for counts in LAYOUTS], 'slots': 10,
            'maximum_context_rows': 10, 'comparison': 'bit-exact'}
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(data))
    assert len(_ragged_input(path)) == 64
    for raw in (json.dumps({**data, 'slots': 10.0}), json.dumps({**data, 'unknown': 1}),
                '{"slots":10,"slots":10}', ' ' * 4097):
        path.write_text(raw)
        with pytest.raises(ValueError):
            _ragged_input(path)


def test_seed_and_ragged_windows_fit_prepaid_short_prefix_with_unequal_contexts():
    contexts = tuple(length + (index % 2 == 0) for index, length in enumerate(SEED_LENGTHS))
    assert min(contexts) > 0 and len(set(contexts)) > 1
    for counts in LAYOUTS:
        assert max(context + count for context, count in zip(contexts, counts)) <= 10


@pytest.fixture
def accounting_inputs(monkeypatch):
    """Controlled metadata schema only; never GPU/model/replay proof."""
    import copy

    for name in ('TENSORFOLD_MEMORY_RESERVE_GIB', 'TENSORFOLD_CUDA_MEMORY_RESERVE_GIB',
                 'TENSORFOLD_CUDA_MEMORY_LIMIT_GB', 'TENSORFOLD_MULTI_GRAPH_MIB'):
        monkeypatch.delenv(name, raising=False)
    keys = [['verify', n, 1] for n in (1, 2, 3, 4, 5)] + [['mtp', n, 1] for n in (1, 2, 3, 4, 5)]
    keys += [[role, n * width, n] for n in (2, 4, 8, 16, 24) for width in (1, 5) for role in ('verify', 'mtp')]
    keys += [['verify', 28, 8], ['mtp', 28, 8]]
    measured = [{'bytes': 1, 'pieces': 2, 'accounting': {
        'schema': 'tensorfold.exact-graph-key-v1', 'rank': 0, 'role': role, 'rows': rows, 'participants': count,
        'pieces': 2, 'prior_keys': i, 'prior_bytes': i, 'physical_delta_bytes': 1,
        'allocated_delta_bytes': 0, 'reserved_delta_bytes': 0, 'budget_bytes': 100, 'admitted': True}}
        for i, (role, rows, count) in enumerate(keys)]
    allocator = {'backend': 'native', 'accounting': 'native-pool-inactive', 'physical_free_bytes': 8 * 2**30,
                 'available_after_floor_bytes': 4 * 2**30, 'reserved_bytes': 100, 'allocated_bytes': 90,
                 'private_pool_spare_bytes': 5}
    record = {'verify_keys': 16, 'mtp_keys': 16, 'bytes': 32, 'settled_bytes': 30, 'room_bytes': 32,
              'budget_bytes': 100, 'pieces': 64, 'stopped': None, 'rejected': None,
              'capture_during_serving': False, 'keys': measured, 'allocator': allocator,
              'exact_graph_catalogue': {'policy': 'observed-demand-v1', 'evidence_sha256': 'a' * 64,
                  **{name: copy.deepcopy(keys) for name in ('available_keys', 'requested_keys', 'selected_keys', 'captured_keys')},
                  'declined_keys': []}}
    capacity = {'native_window': 262144, 'requested_context': 262144, 'context_window': 262144,
        'allocation_baseline': {'device_total_bytes': 32 * 2**30, 'effective_device_floor_bytes': 4 * 2**30},
        'resource_ledger_bytes': {'serving_workspace': 2 * 2**30},
        'multi_graphs': {**copy.deepcopy(record), 'automatic': True, 'captured': True, 'reason': None,
            'future_service': {'growth_bytes': 0, 'copy_overlap_bytes': 0, 'snapshot_bytes': 10,
                'transient_bytes': 10, 'workspace_bytes': 2 * 2**30, 'future_bytes': 2 * 2**30 + 10}},
        'kv_full_windows': {'requested': 24, 'reserved': 24, 'stream_slots': 24, 'context_window': 262144,
            'cache_slots_each': 262149, 'workspace_bytes': 2 * 2**30, 'materialized': True,
            'startup_capacities': [262149] * 24}}
    sample = {'physical_free_bytes': 8 * 2**30, 'device_total_bytes': 32 * 2**30,
              'allocated_bytes': 100, 'reserved_bytes': 110, 'observed_at_unix_ns': 10,
              'start_monotonic_ns': 20, 'end_monotonic_ns': 21}
    states = [{'slot': i, 'logical_context': 262144, 'capacity': 262149, 'pos': 0,
               'mtp_len': 0, 'slot_generation': 5} for i in range(24)]
    return record, capacity, sample, states


def test_accounting_projects_actual_selected_fields_and_keeps_phase_distinction(accounting_inputs):
    import json

    from tools.v100_owned_qualification import _ragged_accounting
    record, capacity, sample, states = accounting_inputs
    record['unused_large_archive'] = 'unrelated' * 10000
    result = _ragged_accounting(record, capacity, rank=0, graph_budget=100, sample=sample, states=states)
    assert result['capture_settlement']['settled_bytes'] == 30
    assert result['capture_settlement']['allocator']['allocated_bytes'] == 90
    assert result['quiescent_sample']['allocated_bytes'] == 100
    assert len(result['capture_settlement']['keys']) == 32
    assert len(result['actual_states']) == 24
    assert 'unused_large_archive' not in result
    assert len(json.dumps(result).encode()) < 4096 + 64 * 1024


@pytest.mark.parametrize('case', ['declined', 'missing', 'duplicate', 'budget', 'rank', 'charge',
    'false_admission', 'noninteger', 'startup_drift', 'floor', 'workspace', 'context', 'busy_state', 'state_capacity'])
def test_accounting_rejects_incomplete_or_weakened_gate_before_publication(accounting_inputs, case):
    from tools.v100_owned_qualification import _ragged_accounting
    record, capacity, sample, states = accounting_inputs
    if case == 'declined':
        record['exact_graph_catalogue']['declined_keys'] = [{'key': ['verify', 28, 8], 'reason': 'no room'}]
    elif case == 'missing':
        record['exact_graph_catalogue']['captured_keys'].pop()
    elif case == 'duplicate':
        record['exact_graph_catalogue']['requested_keys'][-1] = ['verify', 1, 1]
    elif case == 'budget':
        record['room_bytes'] = 101
    elif case == 'rank':
        record['keys'][0]['accounting']['rank'] = 1
    elif case == 'charge':
        record['keys'][1]['accounting']['prior_bytes'] = 0
    elif case == 'false_admission':
        record['keys'][0]['accounting']['admitted'] = False
    elif case == 'noninteger':
        record['keys'][0]['bytes'] = True
    elif case == 'startup_drift':
        capacity['multi_graphs']['settled_bytes'] = 31
    elif case == 'floor':
        sample['physical_free_bytes'] = 4 * 2**30 - 1
    elif case == 'workspace':
        capacity['kv_full_windows']['workspace_bytes'] = 2**30
    elif case == 'context':
        capacity['native_window'] = 8192
    elif case == 'busy_state':
        states[-1]['mtp_len'] = 1
    elif case == 'state_capacity':
        states[-1]['capacity'] = 10
    with pytest.raises(ValueError):
        _ragged_accounting(record, capacity, rank=0, graph_budget=100, sample=sample, states=states)


def test_accounting_metadata_is_paid_in_pre_engine_host_credit(tmp_path, monkeypatch):
    import json
    from types import SimpleNamespace

    from tensorfold.families.qwen4_exp.cuda.weight_types import Config
    from tools.v100_owned_qualification import _history_resources, _ragged_resources
    cfg = SimpleNamespace(layer_types=['linear'] * 36 + ['attention'] * 12, layers=48,
        nv=48, nk=16, dv=128, dk=128, conv_kernel=4, ple_kernel=4, ngram_size=3,
        streams=4, hidden=2560, head_dim=256, index_dim=128, index_ratio=4, vocab=262144)
    monkeypatch.setattr(Config, 'read', lambda _path: cfg)
    for name in ('TENSORFOLD_MEMORY_RESERVE_GIB', 'TENSORFOLD_CUDA_MEMORY_RESERVE_GIB',
                 'TENSORFOLD_CUDA_MEMORY_LIMIT_GB', 'TENSORFOLD_MULTI_GRAPH_MIB'):
        monkeypatch.delenv(name, raising=False)
    data = {'schema': 'tensorfold.native-ragged-replay-input-v1', 'key': [28, 8],
            'layouts': [list(counts) for counts in LAYOUTS], 'slots': 10,
            'maximum_context_rows': 10, 'comparison': 'bit-exact'}
    path = tmp_path / 'input.json'
    path.write_text(json.dumps(data))
    args = SimpleNamespace(draft_method='mtp', mtp_drafts=4, no_drafts=False, no_graphs=False,
                           parallel=24, context=262144, model='header-fixture')
    cli = SimpleNamespace(_parallel=lambda n: n, _config_dir=lambda _model: tmp_path)
    paid = _ragged_resources(args, cli, path)
    expected = _history_resources(32 + 24 + 8, 4096)
    assert paid['accounting_bytes_limit'] == expected['receipt_bytes_limit']
    assert paid['accounting_eager_encoding_bytes'] == 4 * expected['receipt_bytes_limit']
    assert paid['accounting_host_bytes'] == 2 * expected['per_rank_payload_bytes'] + paid['accounting_eager_encoding_bytes']
    assert paid['global_host_obligation_bytes'] == 8 * paid['per_rank_payload_bytes']
    assert paid['per_rank_payload_bytes'] > 3 * paid['snapshot_payload_bytes'] + paid['accounting_host_bytes']
    monkeypatch.setenv('TENSORFOLD_CUDA_MEMORY_RESERVE_GIB', '0')
    with pytest.raises(ValueError, match='unchanged default'):
        _ragged_resources(args, cli, path)

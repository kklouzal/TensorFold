"""CPU fixture production/retirement contracts; no device qualification."""
import gc
from types import SimpleNamespace as NS
import weakref

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda.compact_indexer import Staging, coordinate_rows
from tensorfold.families.qwen4_exp.cuda.history_owner import _encode
from tensorfold.families.qwen4_exp.cuda.state import State, KVNumericError
from tensorfold.families.qwen4_exp.kv_formats import get_pair
from tests.cuda import test_sm70_owner_graphs as fixture
from tests.test_native_owner_adapter import _weights

pytestmark = pytest.mark.torch


def state_for(rank):
    weights = _weights(rank)
    weights.mtp = None
    weights.meta['attention_degree'] = 1
    return weights, State(weights, 257, 4, 'int8')


def inputs(state):
    raw = torch.arange(133 * 128, dtype=torch.float32).reshape(133, 128).half()
    coordinates = coordinate_rows(state, torch.arange(133, dtype=torch.int64))
    return Staging(133, 12, 'cpu', mtp=False), raw, coordinates, torch.ones(128), torch.zeros(1)


@pytest.mark.parametrize('rank', range(8))
def test_real_prefill_span_pool_and_cut_preserve_kv_and_owned_tail(rank):
    weights, state = state_for(rank)
    pair = get_pair('int8')
    state.kv_begin()
    key = torch.zeros((133, 1, 256), dtype=torch.float16)
    value = torch.full_like(key, 1 if rank < 4 else 2)
    codes, scales = _encode(key, pair.key, state.kv_status, 0, error=2)
    values, value_scales = _encode(value, pair.value, state.kv_status, 0, error=4)
    for cache in state.kc:
        cache.append(0, codes, values, scales, value_scales)
    before = [(cache.k.clone(), cache.v.clone(), cache.ks.clone(), cache.vs.clone()) for cache in state.kc]
    stage, raw, coordinates, gamma, frequencies = inputs(state)
    fixture._populate_compact_indexer(state, stage, raw, coordinates, gamma, frequencies,
                                      1e-6, (1, 1, 1), 1.0, 129)
    state.kv_check()
    assert state.pos == int(state.pos_dev[0]) == state.target_tentative_length == 129
    assert state.capacity == state.logical_context == 257
    assert state.kv_status.tolist() == [0, -2]
    expected_tail = torch.zeros((4, 128), dtype=torch.float16)
    expected_tail[0].copy_(raw[128])
    expected_coordinates = torch.zeros((4, 3), dtype=torch.int64)
    expected_coordinates[0].copy_(coordinates[128])
    owned = next(i for i, entry in enumerate(state.compact_indexer.entries) if entry is not None)
    reference = torch.zeros_like(state.pooled[owned])
    ops.qsa_pool(raw, reference, torch.tensor([0], dtype=torch.int32), gamma, frequencies,
                 1e-6, NS(ratio=4), 133, rope=coordinates, length=133, sections=(1, 1, 1), rope_scale=1.0)
    for ordinal, entry in enumerate(state.compact_indexer.entries):
        if entry is not None:
            assert entry.n == 129 and entry.span.start == 0 and entry.span.end == 133
            assert entry.span.expected == (state.slot_generation, state.history_generation)
            assert entry.span.staging is stage and entry.span.resolved == 129
            assert torch.equal(entry.raw, expected_tail)
            assert torch.equal(entry.coordinates, expected_coordinates)
            assert torch.equal(state.pooled[ordinal][:33], reference[:33])
        for actual, old in zip((state.kc[ordinal].k, state.kc[ordinal].v, state.kc[ordinal].ks, state.kc[ordinal].vs), before[ordinal]):
            assert torch.equal(actual, old)
    stage_ref = weakref.ref(stage)
    del stage
    gc.collect()
    assert stage_ref() is not None  # The produced span still owns its staging.
    state.reset(weights)
    gc.collect()
    assert stage_ref() is None  # Production reset retires and revokes the span.


def test_original_unproduced_commit_failure_remains_a_reproducer():
    _, state = state_for(0)
    with pytest.raises(ValueError, match='produced cache span/generation'):
        state.set_pos(129)
    assert state.pos == 0


def test_pool_failure_does_not_publish_position_or_drop_produced_staging(monkeypatch):
    _, state = state_for(0)
    stage, raw, coordinates, gamma, frequencies = inputs(state)
    entry = next(item for item in state.compact_indexer.entries if item is not None)
    primary = OSError('authentic pool failure')
    def fail(*args):
        raise primary
    monkeypatch.setattr(entry, 'pool', fail)
    with pytest.raises(OSError) as caught:
        fixture._populate_compact_indexer(state, stage, raw, coordinates, gamma, frequencies,
                                          1e-6, (1, 1, 1), 1.0, 129)
    assert caught.value is primary
    assert state.pos == 0 and entry.n == 0
    assert entry.span.staging is stage and entry.span.resolved is None


class Owner:
    def __init__(self, name, events, failure=None):
        self.name, self.events, self.failure = name, events, failure
        self.closed = self.retired = False

    def close(self, **kwargs):
        self.events.append((self.name, kwargs))
        if self.failure is not None:
            raise self.failure
        self.closed = self.retired = True


def owners(failure=None):
    events = []
    result = [Owner(name, events, failure.get(name) if failure else None)
              for name in ('graphs', 'workspace', 'reference', 'head_comm', 'comm')]
    return events, result


def test_successful_retirement_order_and_body_failure_transport():
    events, objects = owners()
    errors, retired = fixture._retire_worker_resources(0, True, *objects, failed=True)
    assert retired and errors == []
    assert [name for name, _ in events] == ['graphs', 'workspace', 'reference', 'head_comm', 'comm']
    assert events[-2:] == [('head_comm', {'abort': True}), ('comm', {'abort': True})]
    primary = ValueError('authentic population failure')
    with pytest.raises(ValueError) as caught:
        fixture._worker_outcome(0, True, primary, errors, retired)
    assert caught.value is primary


def test_graph_failure_does_not_release_any_dependent_native_owner():
    error = OSError('graph retirement uncertain')
    events, objects = owners({'graphs': error})
    errors, retired = fixture._retire_worker_resources(0, True, *objects, failed=True)
    assert not retired and errors == [error]
    assert events == [('graphs', {})]
    assert not any(owner.closed for owner in objects)


def test_both_independent_workspace_failures_are_preserved_and_comms_retained():
    first, second = OSError('main workspace uncertain'), ValueError('reference workspace uncertain')
    events, objects = owners({'workspace': first, 'reference': second})
    errors, retired = fixture._retire_worker_resources(0, True, *objects, failed=True)
    assert not retired and errors == [first, second]
    assert [name for name, _ in events] == ['graphs', 'workspace', 'reference']
    assert not objects[-1].retired and not objects[-2].retired


def test_head_release_failure_is_not_retried_and_keeps_parent_store_owner():
    error = OSError('native pointer release entered but uncertain')
    events, objects = owners({'head_comm': error})
    errors, retired = fixture._retire_worker_resources(0, True, *objects, failed=True)
    assert not retired and errors == [error]
    assert [name for name, _ in events] == ['graphs', 'workspace', 'reference', 'head_comm']
    assert not objects[-1].retired


def test_unproved_return_is_not_treated_as_retirement(monkeypatch):
    events, objects = owners()
    monkeypatch.setattr(objects[0], 'close', lambda: None)
    errors, retired = fixture._retire_worker_resources(0, False, *objects, failed=False)
    assert not retired and 'did not prove retirement' in str(errors[0])
    assert events == []


def test_uncertain_owners_use_process_containment_with_primary_and_all_errors(monkeypatch):
    primary = ValueError('authentic body')
    first, second = OSError('cleanup1'), RuntimeError('cleanup2')
    observed = []
    class ProcessContained(BaseException):
        pass
    def terminal(code):
        observed.append(('exit', code))
        raise ProcessContained()
    monkeypatch.setattr(fixture.os, '_exit', terminal)
    monkeypatch.setattr(fixture, '_diagnostic_exception', lambda rank, head4, error: observed.append(('primary', error)))
    with pytest.raises(ProcessContained):
        fixture._worker_outcome(0, False, primary, [first, second], False)
    assert observed == [('primary', primary), ('exit', 1)]
    assert primary.__cause__ is not None
    assert 'OSError' in '\n'.join(primary.__notes__) and 'RuntimeError' in '\n'.join(primary.__notes__)


def test_compact_population_preserves_prior_kv_error():
    _, state = state_for(0)
    state.kv_begin()
    state.kv_status[0] = 2
    state.kv_status[1] = 7
    stage, raw, coordinates, gamma, frequencies = inputs(state)
    with pytest.raises(KVNumericError):
        fixture._populate_compact_indexer(state, stage, raw, coordinates, gamma, frequencies,
                                          1e-6, (1, 1, 1), 1.0, 129)
    assert state.kv_status.tolist() == [2, 7]
    assert state.pos == 0


def test_cleanup_preparation_exhaustion_keeps_primary_and_live_owners(monkeypatch):
    primary, exhaustion = ValueError('body before cleanup'), MemoryError('cleanup preparation')
    events, objects = owners()
    stage = object()
    observed = []
    class ProcessContained(BaseException):
        pass
    def fail(*args, **kwargs):
        raise exhaustion
    def terminal(code):
        assert not any(owner.closed for owner in objects)
        observed.append(('exit', code))
        raise ProcessContained()
    monkeypatch.setattr(fixture, '_retire_worker_resources', fail)
    monkeypatch.setattr(fixture, '_diagnostic_exception', lambda rank, head4, error: observed.append(error))
    monkeypatch.setattr(fixture.os, '_exit', terminal)
    with pytest.raises(ProcessContained):
        fixture._finish_worker(0, True, primary, *objects[:3], stage, *objects[3:], failed=True)
    assert events == []
    assert observed == [primary, exhaustion, ('exit', 1)]


def graph_metadata(head4):
    """Stored host fields only; no graph construction or CUDA execution."""
    from tensorfold.families.qwen4_exp.cuda.owner_graphs import OwnerAttentionGraphs
    weights = _weights(0)
    weights.mtp = None
    weights.cfg.index_budget = 8
    graphs = object.__new__(OwnerAttentionGraphs)
    values = dict(kernel_family='sm70-paged-owner-q8-compact-v1', heads_per_block=None,
                  prepared=False, closed=False, failure=MemoryError('fixture budget'), buckets=(1, 2, 4),
                  graphs={1: object(), 2: object()}, budget=64 << 20, room_bytes=96 << 20,
                  allocator_record={'accounting': 'physical-only'}, private_spare_bytes=None, votes=None,
                  capture_s=0, hits=0, eager_calls=0, comm=NS(world=8),
                  measurements=[{'rows': 1, 'physical_free_before': 1000, 'physical_free_after': 1100,
                                 'raw_capture_bytes': -100},
                                {'rows': 2, 'physical_free_before': 1000, 'physical_free_after': 900,
                                 'raw_capture_bytes': 100}])
    for key, value in values.items():
        setattr(graphs, key, value)
    workspace = NS(native_pair=get_pair('int8'), nbytes=123456, native_comm=NS(world=4 if head4 else 8))
    return graphs, workspace, weights


@pytest.mark.parametrize('head4', [False, True])
def test_resource_record_is_stored_metadata_only_and_atomic_bounded(monkeypatch, head4):
    import json
    graphs, workspace, weights = graph_metadata(head4)
    def forbidden(*args, **kwargs):
        raise AssertionError('measurement must not issue device work or allocation')
    for name in ('synchronize', 'empty_cache', 'mem_get_info', 'memory_snapshot'):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    for name in ('empty', 'zeros', 'arange', 'tensor'):
        monkeypatch.setattr(torch, name, forbidden)
    monkeypatch.setattr(graphs, '_free', forbidden)
    records = []
    def write(descriptor, value):
        records.append((descriptor, value))
        return len(value)
    monkeypatch.setattr(fixture.os, 'write', write)
    fixture._emit_graph_resource_context(0, head4, graphs, workspace, weights)
    assert len(records) == 1 and records[0][0] == 2
    raw = records[0][1]
    assert len(raw) <= 4096 and raw.endswith(b'\n')
    record = json.loads(raw.removeprefix(b'OWNER_GRAPH_RESOURCES '))
    assert record['graph_record']['graph_budget_bytes'] == 64 << 20
    assert record['graph_record']['observed_preparation_residency_bytes'] == 96 << 20
    assert record['graph_record']['captures'][0]['raw_capture_bytes'] == -100
    assert record['graph_record']['private_pool_spare_bytes'] is None
    assert record['remaining_capture_rows'] == [4]
    assert record['prepare_start_free'] is None
    assert record['future_schedule']['compact_stage']['tensor_plan_bytes'] == 71288
    assert record['future_schedule']['reference_owner']['enabled'] is head4
    assert record['capture_transport_world'] == (4 if head4 else 8) and record['vote_world'] == 8
    assert not graphs.prepared and graphs.budget == 64 << 20


def test_resource_report_callback_size_and_short_write_fail_explicitly(monkeypatch):
    graphs, workspace, weights = graph_metadata(True)
    primary = OSError('authentic metadata callback failure')
    def fail():
        raise primary
    monkeypatch.setattr(graphs, 'record', fail)
    with pytest.raises(OSError) as caught:
        fixture._emit_graph_resource_context(0, True, graphs, workspace, weights)
    assert caught.value is primary
    monkeypatch.setattr(fixture, '_graph_resource_context', lambda *args: {'oversized': 'x' * 5000})
    with pytest.raises(ValueError, match='4096-byte'):
        fixture._emit_graph_resource_context(0, True, graphs, workspace, weights)
    monkeypatch.setattr(fixture, '_graph_resource_context', lambda *args: {'stored': True})
    monkeypatch.setattr(fixture.os, 'write', lambda descriptor, value: len(value) - 1)
    with pytest.raises(OSError, match='incomplete'):
        fixture._emit_graph_resource_context(0, True, graphs, workspace, weights)


def test_resource_diagnostic_failure_preserves_budget_primary_and_cleanup(monkeypatch):
    primary = MemoryError('original 64MiB refusal')
    diagnostic = OSError('measurement sink failed')
    cleanup = RuntimeError('poisoned graph retains owners')
    events, objects = owners({'graphs': cleanup})
    observed = []
    class ProcessContained(BaseException):
        pass
    def terminal(code):
        observed.append(('exit', code))
        raise ProcessContained()
    monkeypatch.setattr(fixture.os, '_exit', terminal)
    monkeypatch.setattr(fixture, '_diagnostic_exception', lambda rank, head4, error: observed.append(error))
    with pytest.raises(ProcessContained):
        fixture._finish_worker(0, True, primary, *objects[:3], object(), *objects[3:],
                               failed=True, diagnostic=diagnostic)
    assert events == [('graphs', {})]
    assert observed == [primary, ('exit', 1)]
    assert diagnostic in primary.__cause__.exceptions and cleanup in primary.__cause__.exceptions
    assert not any(owner.closed for owner in objects)


@pytest.mark.parametrize('rank,expected', [(0, 1538320), (4, 1519440)])
@pytest.mark.parametrize('head4', [False, True])
def test_positive_caller_plan_has_complete_rank_payload_and_existing_workspace(rank, expected, head4):
    weights = _weights(rank)
    weights.mtp = None
    weights.cfg.index_budget = 8
    workspace = NS(native_pair=get_pair('int8'), nbytes=494456)
    plan = fixture._fixture_future_plan(weights, head4, workspace)
    assert sum(plan['persistent'].values()) + plan['clone_peak'] == expected - (0 if head4 else 494456)
    assert plan['project_temporary'] == 23943328
    assert plan['workspace'] == 2 * 2**30
    assert fixture._fixture_remaining(plan) == expected - (0 if head4 else 494456) + 2 * 2**30


def test_fixed_payload_counts_aliases_once_and_excludes_host_descriptor_mirror():
    data = torch.empty((100,), dtype=torch.uint8)
    view = data[7:90]
    mirror = torch.empty((50,), dtype=torch.int64)
    first = NS(data=data, alias=view, descriptors_host=mirror)
    assert fixture._fixture_tensor_bytes(first, data, view) == 100


class FundingComm:
    world = 8
    def __init__(self, reject_peer=False):
        self.reject_peer = reject_peer
    def all_gather(self, send, receive):
        receive.copy_(send.view(1, 3).expand(8, 3))
        if self.reject_peer:
            receive[7, 0] = 0


def funding_inputs():
    weights = _weights(0)
    weights.mtp = None
    weights.cfg.index_budget = 8
    workspace = NS(native_pair=get_pair('int8'), nbytes=494456,
                   data=torch.empty((494456,), dtype=torch.uint8))
    scratch = NS(data=torch.empty((1000,), dtype=torch.uint8))
    send, receive = torch.empty((3,), dtype=torch.int64), torch.empty((8, 3), dtype=torch.int64)
    return weights, workspace, scratch, send, receive


def test_graph_grant_is_joint_remainder_not_observed_capture_or_double_debit(monkeypatch):
    from tensorfold.cuda import capacity
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    weights, workspace, scratch, send, receive = funding_inputs()
    plan = fixture._fixture_future_plan(weights, True, workspace)
    fixed = fixture._fixture_tensor_bytes(workspace, scratch, send, receive)
    extra = 1234567
    cap = fixed + fixture._fixture_remaining(plan) + extra
    paid = []
    def baseline(runtime, *, preallocated_device_bytes):
        paid.append(preallocated_device_bytes)
        return {'usable_serving_budget_bytes': cap - preallocated_device_bytes,
                'physical_free_bytes': 12 * 2**30, 'effective_device_floor_bytes': 4 * 2**30,
                'absolute_grant_cap_bytes': cap}
    monkeypatch.setattr(capacity, 'device_allocation_baseline', baseline)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 100)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 200)
    result = fixture._fixture_admit(weights, True, workspace, scratch, FundingComm(), send, receive)
    assert result['graph_bytes'] == extra
    assert paid == [fixed, fixed]  # A new sample does not cumulatively debit fixed storage.
    result['graph_charge'] = 1000
    fixture._fixture_check(result, FundingComm(), send, receive, 'state-built', done=('state',))
    expected_paid = fixed + 1000 + plan['persistent']['state']
    assert paid[-1] == expected_paid
    assert result['checks'][-1]['available'] == cap - expected_paid
    assert result['checks'][-1]['required_future'] == fixture._fixture_remaining(plan, ('state',))
    assert result['checks'][-1]['torch_peak_reserved'] == 200


def test_caller_pressure_and_peer_refusal_stop_before_next_allocation(monkeypatch):
    from tensorfold.cuda import capacity
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    weights, workspace, scratch, send, receive = funding_inputs()
    plan = fixture._fixture_future_plan(weights, True, workspace)
    future = fixture._fixture_remaining(plan)
    def baseline(runtime, *, preallocated_device_bytes):
        return {'usable_serving_budget_bytes': future - 1,
                'physical_free_bytes': 12 * 2**30, 'effective_device_floor_bytes': 4 * 2**30,
                'absolute_grant_cap_bytes': None}
    monkeypatch.setattr(capacity, 'device_allocation_baseline', baseline)
    with pytest.raises(MemoryError, match='complete future liabilities'):
        fixture._fixture_admit(weights, True, workspace, scratch, FundingComm(), send, receive)
    funding = {'fixed_tensor_bytes': 0, 'graph_charge': 0, 'initial_usable': future + 100,
               'plan': plan, 'checks': []}
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    with pytest.raises(MemoryError, match='next'):
        fixture._fixture_check(funding, FundingComm(), send, receive, 'next')
    assert funding['checks'][-1]['available'] < funding['checks'][-1]['required_future']
    monkeypatch.setattr(capacity, 'device_allocation_baseline', lambda *args, **kwargs:
        {'usable_serving_budget_bytes': future + 100, 'physical_free_bytes': 12 * 2**30,
         'effective_device_floor_bytes': 4 * 2**30, 'absolute_grant_cap_bytes': None})
    with pytest.raises(MemoryError, match='peer'):
        fixture._fixture_check(funding, FundingComm(reject_peer=True), send, receive, 'peer')


def test_clone_ownership_transfer_and_final_floor_check(monkeypatch):
    from tensorfold.cuda import capacity
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *args: None)
    weights, workspace, scratch, send, receive = funding_inputs()
    plan = fixture._fixture_future_plan(weights, True, workspace)
    done = tuple(plan['persistent'])
    live = 3 * 4 * 3 * 256 * 2
    assert fixture._fixture_remaining(plan, done, live) == plan['workspace'] + 6144
    with pytest.raises(ValueError, match='ownership plan'):
        fixture._fixture_remaining(plan, done, plan['clone_peak'] + 1)
    funding = {'fixed_tensor_bytes': 0, 'graph_charge': 0, 'initial_usable': 10 * 2**30,
               'plan': plan, 'checks': []}
    monkeypatch.setattr(capacity, 'device_allocation_baseline', lambda *args, **kwargs:
        {'usable_serving_budget_bytes': 0, 'physical_free_bytes': 1,
         'effective_device_floor_bytes': 4 * 2**30, 'absolute_grant_cap_bytes': None})
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 0)
    monkeypatch.setattr(torch.cuda, 'max_memory_reserved', lambda: 0)
    with pytest.raises(MemoryError, match='complete'):
        fixture._fixture_check(funding, FundingComm(), send, receive, 'complete', done=done,
                               live_clones=live, complete=True)
    assert funding['checks'][-1]['required_future'] == 0


def test_funding_telemetry_failure_keeps_original_and_cleanup_outcome(monkeypatch):
    primary = OSError('funding receipt sink')
    def fail(*args):
        raise primary
    monkeypatch.setattr(fixture.os, 'write', fail)
    with pytest.raises(OSError) as caught:
        fixture._emit_fixture_funding(0, True, {'plan': {}, 'checks': []})
    assert caught.value is primary
    body = MemoryError('future grant refused')
    events, objects = owners()
    with pytest.raises(MemoryError) as transported:
        fixture._finish_worker(0, True, body, *objects[:3], object(), *objects[3:], failed=True, diagnostic=primary)
    assert transported.value is body and primary is body.__cause__
    assert [name for name, _ in events] == ['graphs', 'workspace', 'reference', 'head_comm', 'comm']


def test_each_reporter_is_marked_before_callback_and_not_retried(monkeypatch):
    attempted, errors, calls = [False, False], [None, None], []
    primary = OSError('first funding sink failure')
    def funding(*args):
        assert attempted[0]
        calls.append('funding')
        raise primary
    def resources(*args):
        assert attempted[1]
        calls.append('resources')
    monkeypatch.setattr(fixture, '_emit_fixture_funding', funding)
    monkeypatch.setattr(fixture, '_emit_graph_resource_context', resources)
    fixture._fixture_reports(0, True, {}, object(), None, None, attempted, errors)
    fixture._fixture_reports(0, True, {}, object(), None, None, attempted, errors)
    assert calls == ['funding', 'resources']
    assert attempted == [True, True] and errors == [primary, None]


def test_partial_first_write_does_not_suppress_resource_report_or_retry(monkeypatch):
    graphs, workspace, weights = graph_metadata(True)
    attempted, errors, writes = [False, False], [None, None], []
    def write(descriptor, value):
        writes.append(value)
        return len(value) - 1 if len(writes) == 1 else len(value)
    monkeypatch.setattr(fixture.os, 'write', write)
    fixture._fixture_reports(0, True, {}, graphs, workspace, weights, attempted, errors)
    fixture._fixture_reports(0, True, {}, graphs, workspace, weights, attempted, errors)
    assert len(writes) == 2
    assert writes[0].startswith(b'OWNER_FIXTURE_FUNDING ') and writes[1].startswith(b'OWNER_GRAPH_RESOURCES ')
    assert isinstance(errors[0], OSError) and errors[1] is None


def test_both_report_failures_keep_original_body_and_all_cleanup(monkeypatch):
    body, first, second = MemoryError('original grant refusal'), OSError('funding sink'), RuntimeError('resource sink')
    attempted, errors = [False, False], [None, None]
    def one(*args):
        raise first
    def two(*args):
        raise second
    monkeypatch.setattr(fixture, '_emit_fixture_funding', one)
    monkeypatch.setattr(fixture, '_emit_graph_resource_context', two)
    fixture._fixture_reports(0, True, {}, object(), None, None, attempted, errors)
    events, objects = owners()
    with pytest.raises(MemoryError) as caught:
        fixture._finish_worker(0, True, body, *objects[:3], object(), *objects[3:], failed=True, report_errors=errors)
    assert caught.value is body
    assert body.__cause__.exceptions == (first, second)
    assert [name for name, _ in events] == ['graphs', 'workspace', 'reference', 'head_comm', 'comm']

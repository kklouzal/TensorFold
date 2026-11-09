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

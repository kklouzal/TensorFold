"""Host comparator integration and lifetime controls; kernels stay independent."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.owner_preparation import parse_preparation, storage_binding_host_bytes
from tensorfold.families.qwen4_exp.cuda.owner_execution import KERNEL_FAMILY, UNCAPTURED
from tensorfold.families.qwen4_exp.cuda.state import State
from tests.test_native_owner_adapter import _weights


def workspace():
    weights = _weights()
    states = [State(weights, 257, 4, 'int8', slot=i) for i in range(2)]
    owner = OwnerBuffers(weights, 8, NS(idw=2052))
    owner.prepare_native(states)
    return owner, states


def test_optional_preparation_and_bounded_host_future():
    old = dict(kernel_family=KERNEL_FAMILY, graph_family=UNCAPTURED, graph_budget_bytes=0, trial=True)
    assert 'storage_bindings' not in parse_preparation(old)
    selected = parse_preparation(dict(old, storage_bindings=True))
    assert storage_binding_host_bytes(selected, {'decode': 32, 'mtp': 32, 'prefill': 256}) == 24 * 2 * 16384
    assert storage_binding_host_bytes(parse_preparation(old), {'decode': 32}) == 0
    with pytest.raises(ValueError):
        parse_preparation(dict(old, storage_bindings='1'))


def test_native_mismatch_keeps_exact_boundary_before_descriptor_write():
    owner, states = workspace()
    calls = []
    owner.native_storage_bindings = {id(state): NS(matches=lambda current: calls.append(current) or False) for state in states}
    owner.descriptors_host.fill_(7)
    with pytest.raises(RuntimeError, match='retained canonical storage catalogue'):
        owner.stage_native_descriptors([(states[0], 0, 2)], NS(index=0), 0, 2)
    assert len(calls) == 1 and bool((owner.descriptors_host == 7).all())


def test_unsupported_native_tuple_keeps_original_python_predicate(monkeypatch):
    owner, states = workspace()
    state = states[0]
    owner.native_storage_bindings = {id(state): NS(matches=lambda current: None)}
    original = owner._storage_signature
    calls = []
    monkeypatch.setattr(owner, '_storage_signature', lambda values: calls.append(values) or original(values))
    owner.stage_native_descriptors([(state, 0, 2)], NS(index=0), 0, 2)
    assert len(calls) == 1


def test_native_result_does_not_skip_generation_guard_or_frontier():
    owner, states = workspace()
    state = states[0]
    owner.native_storage_bindings = {id(state): NS(matches=lambda current: True)}
    state.slot_generation = 2**63
    owner.descriptors_host.fill_(7)
    with pytest.raises(OverflowError, match='generation'):
        owner.stage_native_descriptors([(state, 0, 2)], NS(index=0), 0, 2)
    assert bool((owner.descriptors_host == 7).all())


def test_failed_retirement_keeps_binding_anchors(monkeypatch):
    owner, states = workspace()
    sentinel = object()
    owner.native_storage_bindings = {id(states[0]): sentinel}
    original = owner.native_states
    # A required failure before the existing release cut must retain bindings.
    class Refusal:
        def zero_(self):
            raise OSError('retire zero')
    owner.descriptors = Refusal()
    with pytest.raises(OSError, match='retire zero'):
        owner.retire_native()
    assert owner.native_states is original and owner.native_storage_bindings[id(states[0])] is sentinel

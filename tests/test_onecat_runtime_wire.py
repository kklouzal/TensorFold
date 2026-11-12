"""Actual producer records must cross the unchanged strict runtime JSON wire."""
from itertools import product
import json
from types import SimpleNamespace as NS

import pytest

from tensorfold.engine.model_identity import canonical_runtime_json
from tensorfold.families.qwen4_exp.cuda import onecat_experts as experts
from tensorfold.families.qwen4_exp.cuda.onecat_frame_experts import OneCatFrameOwner


def selectors():
    for method, down, source, prefill in product(experts.METHODS, experts.DOWN_METHODS,
                                                ('active-frame', 'raw'), ('tensorfold', 'candidate')):
        if method == 'tensorfold' and source == 'raw':
            continue
        yield experts.OneCatGU(method, 1 << 30, 1 << 20, down, source, prefill)


@pytest.mark.parametrize('spec', list(selectors()))
@pytest.mark.parametrize('owner_type', [experts.OneCatOwner, OneCatFrameOwner])
def test_real_native_preparation_is_strict_json_without_changing_selector_identity(monkeypatch, spec, owner_type):
    from tensorfold.cuda.sm70 import gguf_ops, onecat_moe_gu, onecat_moe_down, onecat_gpu_packing

    def operator(name):
        return NS(module=NS(__name__=name))
    monkeypatch.setattr(gguf_ops, 'prepare_owned_gate_up', lambda device: operator('tf-gu'))
    monkeypatch.setattr(gguf_ops, 'prepare_owned_from_activation', lambda device: operator('tf-down'))
    monkeypatch.setattr(onecat_moe_gu, 'prepare', lambda device: operator('onecat-gu'))
    monkeypatch.setattr(onecat_moe_gu, 'prepare_raw_iq2', lambda device: operator('raw-iq2'))
    monkeypatch.setattr(onecat_moe_down, 'prepare', lambda device: operator('q8-down'))
    monkeypatch.setattr(onecat_gpu_packing, 'prepare', lambda device: operator('packer'))
    original = spec.identity
    record = owner_type(spec, 'cpu').prepare_native()
    outcome = {'ok': True, 'value': {'onecat_gu': record}}
    assert json.loads(canonical_runtime_json(outcome, max_bytes=4 << 20)) == outcome
    assert record['selector'] == list(original)
    assert spec.identity == original and type(spec.identity) is tuple
    hash(spec.identity)


@pytest.mark.parametrize('spec', list(selectors()))
def test_active_frame_forecast_and_profile_are_wire_data(monkeypatch, spec):
    from test_onecat_runtime_bridge import prototype
    proto, _ = prototype()
    owner = OneCatFrameOwner(spec, 'cpu')
    record = owner.forecast(proto, 32 << 20)
    assert json.loads(canonical_runtime_json(record, max_bytes=4 << 20)) == record
    owner.complete = True
    owner.gu = owner.down = NS(module=NS(__name__='fixture-only'))
    profile = owner.profile()
    assert json.loads(canonical_runtime_json(profile, max_bytes=4 << 20)) == profile


@pytest.mark.parametrize('spec', list(selectors()))
def test_actual_owned_plan_wire_preserves_internal_nested_tuple(monkeypatch, spec):
    from tensorfold.families.qwen4_exp.cuda import owned_experts
    from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition
    monkeypatch.setattr(owned_experts, '_GROUPED_DOWN', 'grouped-v1')
    plan = owned_experts.Plan(ExpertPartition(8, 0, 512, 640, 4), 2560, 11, 120, 256, True, _gate_up=spec)
    original = plan.identity
    result = owned_experts.wire_identity(plan)
    assert json.loads(canonical_runtime_json({'identity': result}, max_bytes=1 << 20)) == {'identity': result}
    assert result[-1] == list(spec.identity)
    resources = plan.resources(120)
    assert json.loads(canonical_runtime_json(resources, max_bytes=1 << 20)) == resources
    assert plan.identity == original and type(plan.identity[-1]) is tuple
    hash(plan.identity)


def test_wire_adapter_does_not_stringify_unknown_objects():
    from tensorfold.families.qwen4_exp.cuda.owned_experts import wire_identity
    with pytest.raises(ValueError, match='non-JSON'):
        canonical_runtime_json({'identity': wire_identity(NS(identity=('known', object())))}, max_bytes=1024)

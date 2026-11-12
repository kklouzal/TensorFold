"""Exercise the real family boundary before its GPU-owning constructor."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from tensorfold.families import qwen4_exp as family
from tensorfold.cuda.sm70.gguf_ops import _GroupedGateUp
from tensorfold.families.qwen4_exp.cuda.onecat_experts import OneCatGU


@pytest.fixture
def adapter(monkeypatch):
    from tensorfold import gguf_file
    from tensorfold.families.qwen4_exp.cuda import engine, serving_profiles, speculation

    calls = []
    monkeypatch.setattr(gguf_file, 'resolve_gguf', lambda path: Path('owned.gguf'))
    monkeypatch.setattr(family, 'has_mtp', lambda path: True)
    monkeypatch.setattr(serving_profiles, 'validate_service_options', lambda *a: None)
    monkeypatch.setattr(speculation, 'checkpoint_method', lambda path: 'mtp')
    sentinel = object()
    monkeypatch.setattr(engine, 'FlashNextEngine', lambda path, **kwargs: calls.append((path, kwargs)) or sentinel)
    return calls, sentinel


def selected():
    return OneCatGU('onecat-half-epilogue', 1, 1, down_method='onecat-q8-slots',
                    iq2_source='raw', prefill_method='candidate')


@pytest.mark.parametrize('method', ['none', 'mtp'])
@pytest.mark.parametrize('capability', [lambda: _GroupedGateUp(4), selected])
def test_family_forwards_exact_typed_owned_capability_to_engine(adapter, method, capability):
    calls, sentinel = adapter
    value = capability()
    result = family.cuda_engine('owned.gguf', tp=8, expert_tensor_parallel_size=4,
        drafter='draft' if method == 'mtp' else '', draft_method=method,
        _owned_expert_native=True, _owned_expert_gate_up=value, graphs=True,
        execution_profile_manifest='profile.json')
    assert result is sentinel and len(calls) == 1
    options = calls[0][1]
    assert options['_owned_expert_gate_up'] is value and options['_owned_expert_native'] is True
    assert options['tp'] == 8 and options['expert_tensor_parallel_size'] == 4
    assert options['adaptive_sessions'] is True and options['graphs'] is True
    assert options['execution_profile_manifest'] == Path('profile.json')
    assert options['depth'] == (0 if method == 'none' else 4)
    assert ('mtp_model_dir' in options) is (method == 'mtp')


@pytest.mark.parametrize('value', [True, 32, 'onecat', SimpleNamespace(rows=32), object()])
def test_family_rejects_untyped_capability_before_engine(adapter, value):
    calls, _ = adapter
    with pytest.raises(TypeError, match='typed owned'):
        family.cuda_engine('owned.gguf', tp=8, expert_tensor_parallel_size=4,
            draft_method='none', _owned_expert_native=True, _owned_expert_gate_up=value)
    assert not calls


def test_family_rejects_subclass_and_capability_without_owned_scope(adapter):
    calls, _ = adapter
    class Derived(OneCatGU):
        pass
    values = [Derived('onecat-half-epilogue', 1, 1), selected(), _GroupedGateUp(4)]
    for value, owned in [(values[0], True), (values[1], False), (values[2], False)]:
        with pytest.raises(TypeError, match='typed owned'):
            family.cuda_engine('owned.gguf', tp=8, expert_tensor_parallel_size=4,
                draft_method='none', _owned_expert_native=owned, _owned_expert_gate_up=value)
    assert not calls

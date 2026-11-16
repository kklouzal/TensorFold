"""Worker placement is verified during startup, outside inference hot paths."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine


@pytest.fixture
def env(monkeypatch):
    for name in ('TENSORFOLD_NUMA_PLAN', 'TENSORFOLD_NUMA_PLAN_SHA256', 'TENSORFOLD_NUMA_RANK'):
        monkeypatch.delenv(name, raising=False)


def test_legacy_launch_without_numa_authority_does_not_touch_cuda(env):
    obj = object.__new__(FlashNextEngine)
    obj._verify_numa_placement('post-cuda', NS(), expected_rank=0)
    assert not hasattr(obj, '_numa_placement')


@pytest.mark.parametrize('stage', ['post-cuda', 'ready'])
def test_enforced_worker_proof_is_recorded_only_for_the_actual_engine_rank(env, monkeypatch, stage):
    from tensorfold.cuda import rank_placement
    monkeypatch.setenv('TENSORFOLD_NUMA_PLAN', '/operator/numa/plan.json')
    calls = []
    record = {'rank': 3, 'stage': stage}
    monkeypatch.setattr(rank_placement, 'verify_worker_from_env',
        lambda module, device, **kw: calls.append((module, device, kw)) or record)
    runtime = NS(device=lambda kind, index: (kind, index), cuda=NS(current_device=lambda: 0))
    obj = object.__new__(FlashNextEngine)
    obj._verify_numa_placement(stage, runtime, expected_rank=3)
    assert calls == [(runtime, ('cuda', 0), {'stage': stage})]
    assert obj._numa_placement[stage] is record
    with pytest.raises(ValueError, match='actual Engine rank'):
        obj._verify_numa_placement(stage, runtime, expected_rank=4)
    assert obj._numa_placement == {stage: record}


def test_policy_failures_propagate_without_publishing_success(env, monkeypatch):
    from tensorfold.cuda import rank_placement
    monkeypatch.setenv('TENSORFOLD_NUMA_RANK', '0')
    error = PermissionError('NUMA policy unavailable')
    def fail(*a, **k):
        raise error
    monkeypatch.setattr(rank_placement, 'verify_worker_from_env', fail)
    obj = object.__new__(FlashNextEngine)
    runtime = NS(device=lambda *a: 'cuda:0', cuda=NS(current_device=lambda: 0))
    with pytest.raises(PermissionError) as caught:
        obj._verify_numa_placement('post-cuda', runtime, expected_rank=0)
    assert caught.value is error and not hasattr(obj, '_numa_placement')


def test_real_startup_places_post_cuda_check_before_native_preparation_and_ready_before_return():
    source = Path(__file__).resolve().parents[1]/'src/tensorfold/families/qwen4_exp/cuda/engine.py'
    text = source.read_text()
    assert text.index('torch.cuda.set_device(0)') < text.index('self._verify_numa_placement("post-cuda"') < text.index('self._triton_driver_preparation = prepare_driver')
    tree = ast.parse(text)
    constructor = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    calls = [node for node in ast.walk(constructor) if isinstance(node, ast.Call)]
    initialize = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == '_initialize')
    ready = next(node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == '_verify_numa_placement')
    assert initialize.lineno < ready.lineno


def _bind_calls(monkeypatch, discovered):
    from tensorfold.cuda import numa_memory
    calls = []
    def configure(device):
        calls.append(('configure', device))
        if isinstance(discovered, BaseException):
            raise discovered
        return discovered
    monkeypatch.setattr(numa_memory, 'configure_gpu_numa', configure)
    monkeypatch.setattr(numa_memory, 'bind_process_to_placement', lambda placement: calls.append(('bind', placement)))
    return calls


@pytest.mark.parametrize('name', ['TENSORFOLD_NUMA_PLAN', 'TENSORFOLD_NUMA_PLAN_SHA256', 'TENSORFOLD_NUMA_RANK'])
def test_rank_binding_defers_to_a_sealed_plan_and_skips_single_rank(env, monkeypatch, name):
    calls = _bind_calls(monkeypatch, NS(cpus=(0,), pci_bus_id='0000:1a:00.0', node=0))
    FlashNextEngine._bind_local_node(1)
    monkeypatch.setenv(name, 'x')
    FlashNextEngine._bind_local_node(8)
    assert calls == []


def test_rank_binding_binds_observed_node_and_logs_unobservable_topology(env, monkeypatch, capsys):
    placement = NS(cpus=(0, 1), pci_bus_id='0000:1a:00.0', node=0)
    calls = _bind_calls(monkeypatch, placement)
    FlashNextEngine._bind_local_node(8)
    assert calls == [('configure', 0), ('bind', placement)]
    assert 'NUMA node 0 (2 CPUs)' in capsys.readouterr().out
    calls = _bind_calls(monkeypatch, RuntimeError('GPU has no observable online host NUMA node'))
    FlashNextEngine._bind_local_node(8)
    assert calls == [('configure', 0)]
    assert 'left unbound' in capsys.readouterr().out


def test_rank_binding_affinity_faults_propagate(env, monkeypatch):
    from tensorfold.cuda import numa_memory
    monkeypatch.setattr(numa_memory, 'configure_gpu_numa', lambda device: NS(cpus=(0,), pci_bus_id='b', node=0))
    def refuse(placement):
        raise OSError(22, 'Invalid argument')
    monkeypatch.setattr(numa_memory, 'bind_process_to_placement', refuse)
    with pytest.raises(OSError):
        FlashNextEngine._bind_local_node(8)

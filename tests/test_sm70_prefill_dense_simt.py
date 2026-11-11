"""SIMT owner protocol and header/package boundaries; no CUDA math proof."""
import importlib.resources
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import prefill_dense_simt as simt, foundation_headers, provider
from test_sm70_prefill_dense import Native


@pytest.fixture
def factory(monkeypatch):
    calls, failures, bindings = [], {}, []
    def create(x, weight, out):
        bindings.append((x, weight, out))
        return Native((x, weight, out), failures, calls)
    monkeypatch.setattr(simt, 'extension', lambda: NS(Prepared=create))
    return calls, failures, bindings


def inputs(inference=False):
    with torch.inference_mode(inference):
        return torch.ones((3, 7)).half(), torch.ones((4, 7)).half(), torch.empty((3, 4)).half()


@pytest.mark.parametrize('inference', [False, True])
def test_direct_half_operation_reuses_lifetime_and_publishes_one_output(factory, inference):
    calls, _, bindings = factory
    args = inputs(inference)
    before = None if inference else args[-1]._version
    with torch.inference_mode(inference), simt.prepare(*args) as plan:
        assert all(a is b for a, b in zip(bindings[0], args))
        assert plan.run(*args) is args[-1]
        assert len(plan._operands) == 3 and all(a is b for a, b in zip(plan._operands, args))
        with pytest.raises(RuntimeError, match='pending'):
            plan.run(*args)
        plan.finish()
        assert plan._operands is None
    assert calls == ['configure', 'run', 'finish', 'close']
    if not inference:
        assert args[-1]._version == before + 1


@pytest.mark.parametrize('index', [0, 1, 2])
def test_all_bound_operands_reject_grad_before_native_acquisition(factory, index):
    _, _, bindings = factory
    args = list(inputs())
    args[index].requires_grad_(True)
    with pytest.raises(ValueError, match='plain'):
        simt.prepare(*args)
    assert bindings == []


def test_failed_configure_retains_bound_operands_and_authentic_cleanup(factory):
    calls, failures, _ = factory
    primary, cleanup = RuntimeError('configure'), RuntimeError('close')
    failures.update(configure=primary, close=cleanup)
    args = inputs()
    with pytest.raises(RuntimeError) as found:
        simt.prepare(*args)
    assert found.value is primary and primary.__cause__ is cleanup
    owner = primary._tensorfold_prefill_dense_owner
    assert all(a is b for a, b in zip(owner._binding, args))
    assert calls == ['configure', 'close']


def test_wrong_sdk_receipt_refuses_before_provider_or_compilation(monkeypatch):
    simt.extension.cache_clear()
    monkeypatch.setattr(foundation_headers, 'selected_headers', lambda: NS(manifest_sha256='0' * 64))
    monkeypatch.setattr(provider, '_native', lambda: pytest.fail('wrong SDK reached provider'))
    with pytest.raises(RuntimeError, match='CUTLASS2c85'):
        simt.extension()
    simt.extension.cache_clear()


def test_both_owned_headers_are_installed_package_resources():
    root = importlib.resources.files('tensorfold.cuda.sm70')
    for name in ('prefill_dense_contracts.cuh', 'prefill_dense_simt.cuh'):
        assert root.joinpath(name).is_file()

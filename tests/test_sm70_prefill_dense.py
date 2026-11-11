"""CPU ownership controls; these do not execute cuBLAS or CUDA arithmetic."""
from types import SimpleNamespace
import threading

import pytest
import torch

from tensorfold.cuda.sm70 import prefill_dense as dense


class Native:
    def __init__(self, workspace, failures, calls):
        self.workspace, self.failures, self.calls = workspace, failures, calls
    def invoke(self, name):
        self.calls.append(name)
        if name in self.failures:
            raise self.failures[name]
    def configure(self):
        self.invoke('configure')
    def run(self, *args):
        self.invoke('run')
    def finish(self):
        self.invoke('finish')
    def close(self):
        self.invoke('close')
    def settings(self):
        self.invoke('settings')
        return {'private': True}


@pytest.fixture
def factory(monkeypatch):
    calls, failures, natives = [], {}, []
    def create(workspace):
        native = Native(workspace, failures, calls)
        natives.append(native)
        return native
    monkeypatch.setattr(dense, 'extension', lambda: SimpleNamespace(Prepared=create))
    return calls, failures, natives


def operands(inference=False):
    with torch.inference_mode(inference):
        return (torch.ones((2, 3)).half(), torch.ones((4, 3)).half(),
                torch.empty((2, 4)), torch.empty((2, 4)).half())


@pytest.mark.parametrize('inference', [False, True])
def test_pending_operands_retained_until_positive_fence_and_mutations_once(factory, inference):
    calls, _, _ = factory
    args = operands(inference)
    before = None if inference else tuple(t._version for t in args[2:])
    with torch.inference_mode(inference), dense.prepare(torch.empty(256, dtype=torch.uint8)) as plan:
        assert plan.run(*args) is args[-1]
        assert all(a is b for a, b in zip(plan._operands, args))
        with pytest.raises(RuntimeError, match='pending'):
            plan.run(*args)
        plan.finish()
        assert plan._operands is None
        assert plan.settings() == {'private': True}
    assert calls == ['configure', 'run', 'finish', 'settings', 'close']
    if not inference:
        assert tuple(t._version for t in args[2:]) == tuple(v + 1 for v in before)
    plan.close()
    assert calls.count('close') == 1


@pytest.mark.parametrize('operation', ['run', 'finish', 'close'])
def test_failure_retains_loans_and_cleanup_is_not_retried_when_uncertain(factory, operation):
    calls, failures, _ = factory
    plan = dense.prepare(torch.empty(256, dtype=torch.uint8))
    args = operands()
    primary = RuntimeError(operation)
    failures[operation] = primary
    if operation == 'run':
        with pytest.raises(RuntimeError) as found:
            plan.run(*args)
        assert found.value is primary
        assert plan._operands is not None
        plan.close()  # One positive retirement is still required after operation failure.
    else:
        plan.run(*args)
        with pytest.raises(RuntimeError) as found:
            getattr(plan, operation)()
        assert found.value is primary
        assert plan._operands is not None and plan._workspace is not None
        with pytest.raises(RuntimeError, match='uncertain'):
            plan.close()
        assert calls.count(operation) == 1
    with pytest.raises(RuntimeError):
        plan.run(*args)


def test_configure_failure_and_cleanup_failure_preserve_original_chain_and_owner(factory):
    _, failures, _ = factory
    origin = KeyError('original context')
    primary, cleanup = RuntimeError('configure'), RuntimeError('close')
    primary.__cause__ = origin
    failures.update(configure=primary, close=cleanup)
    with pytest.raises(RuntimeError) as found:
        dense.prepare(torch.empty(256, dtype=torch.uint8))
    assert found.value is primary
    assert getattr(primary, '_tensorfold_prefill_dense_owner')
    assert primary.__cause__ is not None
    assert origin in primary.__cause__.exceptions and cleanup in primary.__cause__.exceptions


def test_context_preserves_body_failure_and_failed_retirement(factory):
    _, failures, _ = factory
    body, cleanup = ValueError('body'), RuntimeError('retirement')
    with pytest.raises(ValueError) as found:
        with dense.prepare(torch.empty(256, dtype=torch.uint8)):
            failures['close'] = cleanup
            raise body
    assert found.value is body
    assert body.__cause__ is cleanup


def test_failed_run_never_publishes_successful_output_versions(factory):
    _, failures, _ = factory
    plan = dense.prepare(torch.empty(256, dtype=torch.uint8))
    args = operands()
    before = tuple(t._version for t in args[2:])
    failures['run'] = RuntimeError('before write')
    with pytest.raises(RuntimeError):
        plan.run(*args)
    assert tuple(t._version for t in args[2:]) == before
    plan.close()


@pytest.mark.parametrize('domain', ['grad', 'negative', 'subclass'])
def test_generic_tensor_domains_refused_before_native_work(factory, domain):
    calls, _, _ = factory
    args = list(operands())
    if domain == 'grad':
        args[0].requires_grad_(True)
    elif domain == 'negative':
        args[0] = torch._neg_view(args[0])
    else:
        class Other(torch.Tensor):
            pass
        args[0] = args[0].as_subclass(Other)
    with dense.prepare(torch.empty(256, dtype=torch.uint8)) as plan:
        with pytest.raises(ValueError, match='plain'):
            plan.run(*args)
    assert calls == ['configure', 'close']


def test_preparing_thread_owns_handle(factory):
    calls, _, _ = factory
    with dense.prepare(torch.empty(256, dtype=torch.uint8)) as plan:
        found = []
        def other():
            try:
                plan.run(*operands())
            except BaseException as error:
                found.append(error)
        thread = threading.Thread(target=other)
        thread.start()
        thread.join(timeout=5)
        assert not thread.is_alive() and len(found) == 1
        assert isinstance(found[0], RuntimeError)
    assert calls == ['configure', 'close']


@pytest.mark.parametrize('cleanup_fails', [False, True])
def test_reporting_exhaustion_never_precedes_close_or_replaces_primary(factory, monkeypatch, cleanup_fails):
    calls, failures, _ = factory
    origin, primary = KeyError('original'), RuntimeError('configuration')
    primary.__cause__ = origin
    failures['configure'] = primary
    cleanup = RuntimeError('cleanup')
    if cleanup_fails:
        failures['close'] = cleanup
    exhausted = MemoryError('report transport')
    def transport(*args):
        assert calls[-1] == 'close'
        raise exhausted
    monkeypatch.setattr(dense, 'raise_failures', transport)
    with pytest.raises(RuntimeError) as found:
        dense.prepare(torch.empty(256, dtype=torch.uint8))
    assert found.value is primary and primary.__cause__ is exhausted
    assert calls == ['configure', 'close']
    if cleanup_fails:
        owner = primary._tensorfold_prefill_dense_owner
        assert owner._cleanup_one[0] is cleanup
        assert owner._native_chain[0] is origin
        assert owner._workspace is not None

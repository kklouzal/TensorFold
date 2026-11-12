"""Owned forward-wrapper CPU lifetime controls; no FA arithmetic qualification."""
from types import SimpleNamespace
import threading

import pytest
import torch

from tensorfold.cuda.sm70 import flash_v100_forward as fa


@pytest.fixture
def native():
    calls, failures = [], {}
    class Prepared:
        def __init__(self, *args):
            calls.append('acquire')
        def invoke(self, operation):
            calls.append(operation)
            if operation in failures:
                raise failures[operation]
        def configure(self):
            self.invoke('configure')
        def warm(self):
            self.invoke('warm')
        def run(self, *args):
            self.invoke('run')
        def close(self):
            self.invoke('close')
        def handoff_to_current_thread(self):
            self.invoke('handoff')
        def settings(self):
            return {'source_commit': fa.SOURCE_COMMIT}
    return SimpleNamespace(Prepared=Prepared), calls, failures


def operands():
    q = torch.ones(1, 1, 2, 256).half()
    return q, torch.ones(1, 16, 1, 256).half(), torch.ones(1, 16, 1, 256).half(), torch.empty_like(q), torch.empty(1, 2, 1)


def test_warm_and_run_mutation_publication_and_single_explicit_retirement(native):
    module, calls, _ = native
    args = operands()
    versions = args[3]._version, args[4]._version
    with fa.prepare(module, *args) as owner:
        result = owner.run(*args)
        assert result[0] is args[3] and result[1] is args[4]
        assert owner.settings()['source_commit'] == fa.SOURCE_COMMIT
        assert tuple(t._version for t in args[3:]) == tuple(v + 2 for v in versions)
        assert 'close' not in calls
    owner.close()
    assert calls == ['acquire', 'configure', 'warm', 'run', 'close']


def test_warm_primary_preserved_with_cleanup_failure(native):
    module, calls, failures = native
    primary, cleanup = RuntimeError('warm failure'), RuntimeError('cleanup failure')
    failures.update(warm=primary, close=cleanup)
    with pytest.raises(RuntimeError) as found:
        fa.prepare(module, *operands())
    assert found.value is primary and calls[-1] == 'close'


def test_failed_retirement_is_terminal_without_native_retry(native):
    module, calls, failures = native
    owner = fa.prepare(module, *operands())
    failures['close'] = RuntimeError('uncertain fence')
    with pytest.raises(RuntimeError, match='fence'):
        owner.close()
    with pytest.raises(RuntimeError, match='retry'):
        owner.close()
    assert calls.count('close') == 1


def test_explicit_thread_handoff_then_control_thread_close(native):
    module, calls, _ = native
    args = operands()
    owner = fa.prepare(module, *args)
    errors = []
    def worker():
        try:
            owner.handoff_to_current_thread()
            owner.run(*args)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors == []
    with pytest.raises(RuntimeError, match='Thread'):
        owner.run(*args)
    owner.close()
    assert calls[-3:] == ['handoff', 'run', 'close']


def test_untrusted_tensor_domain_refused_before_acquisition(native):
    module, calls, _ = native
    args = list(operands())
    args[0].requires_grad_()
    with pytest.raises(ValueError, match='plain non-grad'):
        fa.prepare(module, *args)
    assert calls == []


def test_mutation_publication_failure_poisoned_after_native_launch(native, monkeypatch):
    module, calls, _ = native
    args = operands()
    owner = fa.prepare(module, *args)
    def fail(*args):
        raise MemoryError('mutation publication')
    monkeypatch.setattr(fa, '_publish', fail)
    with pytest.raises(MemoryError, match='publication'):
        owner.run(*args)
    with pytest.raises(RuntimeError, match='failed'):
        owner.run(*args)
    owner.close()
    assert calls.count('run') == 1

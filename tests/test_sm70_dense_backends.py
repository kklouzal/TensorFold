"""CPU lifecycle controls; no CUDA arithmetic or capture qualification."""
from types import SimpleNamespace
import threading

import pytest
import torch

from tensorfold.cuda.sm70 import dense_backends as dense


@pytest.fixture
def factory(monkeypatch):
    calls, failures, plans = [], {}, []

    class NativeContext:
        def __init__(self, *args):
            self.retired = False

        def configure(self):
            calls.append('context.configure')

        def handoff_to_current_thread(self):
            calls.append('context.handoff')
            if 'context.handoff' in failures:
                raise failures['context.handoff']

        def close(self):
            calls.append('context.close')
            if 'context.close' in failures:
                raise failures['context.close']
            self.retired = True
            if 'context.handle' in failures:
                raise failures['context.handle']

        def retirement_fenced(self):
            return self.retired

    class NativePlan:
        def __init__(self, context, *args):
            self.context = context
            plans.append(self)

        def invoke(self, operation):
            calls.append(operation)
            if operation in failures:
                raise failures[operation]

        def configure(self):
            self.invoke('configure')

        def warm(self, *args):
            self.invoke('warm')

        def run(self, *args):
            self.invoke('run')

        def close(self):
            assert self.context.retired
            self.invoke('close')

        def settings(self):
            return {'workspace_bytes': 0}

    monkeypatch.setattr(dense, 'extension', lambda: SimpleNamespace(Context=NativeContext, Prepared=NativePlan))
    return calls, failures, plans


def operands():
    return torch.ones(2, 8).half(), torch.ones(4, 8).half(), torch.empty(2, 4).half()


def test_reusable_pointers_no_per_run_retirement_and_one_context_close(factory):
    calls, _, _ = factory
    with dense.Context('cuda:0', backend='cublas', max_plans=2) as context:
        first, second = operands(), operands()
        version = first[-1]._version
        plan = dense.prepare(context, *first)
        assert first[-1]._version == version + 1  # Warm writes output.
        for args in (first, second, first):
            before = args[-1]._version
            assert plan.run(*args) is args[-1]
            assert args[-1]._version == before + 1
        assert 'context.close' not in calls and 'close' not in calls
        assert plan.settings()['workspace_bytes'] == 0
    assert calls == ['context.configure', 'configure', 'warm', 'run', 'run', 'run', 'context.close', 'close']
    context.close()
    with pytest.raises(RuntimeError, match='closed'):
        plan.run(*first)


def test_plan_capacity_checked_before_acquisition(factory):
    _, _, natives = factory
    with dense.Context('cuda:0', backend='cutlass', max_plans=1) as context:
        dense.prepare(context, *operands())
        with pytest.raises(RuntimeError, match='capacity'):
            dense.prepare(context, *operands())
        assert len(natives) == 1


def test_partial_prepare_journal_and_primary_survives_cleanup_failure(factory):
    calls, failures, _ = factory
    primary, cleanup = RuntimeError('warm'), RuntimeError('cleanup')
    failures.update(warm=primary, close=cleanup)
    with pytest.raises(RuntimeError) as found:
        with dense.Context('cuda:0', backend='cublaslt', max_plans=1) as context:
            dense.prepare(context, *operands())
    assert found.value is primary
    assert calls[-2:] == ['context.close', 'close']
    with pytest.raises(RuntimeError, match='retry'):
        context.close()


def test_context_failed_fence_preserves_plans_without_attempted_descriptor_release(factory):
    calls, failures, _ = factory
    context = dense.Context('cuda:0', backend='cublas', max_plans=1)
    dense.prepare(context, *operands())
    failures['context.close'] = RuntimeError('fence')
    with pytest.raises(RuntimeError, match='fence'):
        context.close()
    assert 'close' not in calls
    with pytest.raises(RuntimeError, match='retry'):
        context.close()
    assert calls.count('context.close') == 1


def test_each_plan_cleanup_attempted_once_after_context_retirement(factory):
    calls, failures, _ = factory
    context = dense.Context('cuda:0', backend='cublaslt', max_plans=2)
    for _ in range(2):
        dense.prepare(context, *operands())
    failures['close'] = RuntimeError('descriptor')
    with pytest.raises(RuntimeError):
        context.close()
    assert calls.count('close') == 2


def test_handle_retirement_failure_after_fence_still_retires_independent_plans(factory):
    calls, failures, _ = factory
    context = dense.Context('cuda:0', backend='cublaslt', max_plans=2)
    for _ in range(2):
        dense.prepare(context, *operands())
    primary, cleanup = RuntimeError('handle'), RuntimeError('descriptor')
    failures.update({'context.handle': primary, 'close': cleanup})
    with pytest.raises(RuntimeError) as found:
        context.close()
    assert found.value is primary
    assert calls.count('close') == 2
    with pytest.raises(RuntimeError, match='retry'):
        context.close()


def test_thread_boundary_rejects_before_native_launch(factory):
    calls, _, _ = factory
    with dense.Context('cuda:0', backend='cublas', max_plans=1) as context:
        args = operands()
        plan = dense.prepare(context, *args)
        errors = []
        def worker():
            try:
                plan.run(*args)
            except RuntimeError as error:
                errors.append(error)
        thread = threading.Thread(target=worker)
        thread.start()
        thread.join()
        assert len(errors) == 1 and 'thread' in str(errors[0])
        assert 'run' not in calls


def test_explicit_handoff_moves_all_existing_plans_and_control_thread_can_close(factory):
    calls, _, _ = factory
    context = dense.Context('cuda:0', backend='cublaslt', max_plans=2)
    args = operands()
    plans = [dense.prepare(context, *args) for _ in range(2)]
    errors = []
    def worker():
        try:
            context.handoff_to_current_thread()
            for plan in plans:
                plan.run(*args)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=worker)
    thread.start()
    thread.join()
    assert errors == []
    assert calls.count('context.handoff') == 1 and calls.count('run') == 2
    with pytest.raises(RuntimeError, match='host thread'):
        plans[0].run(*args)
    # The previous control thread retires after the scheduler thread exits.
    context.close()
    assert calls[-3:] == ['context.close', 'close', 'close']


def test_handoff_failure_never_publishes_new_thread_and_is_terminal(factory):
    calls, failures, _ = factory
    context = dense.Context('cuda:0', backend='cublas', max_plans=1)
    args = operands()
    plan = dense.prepare(context, *args)
    original_thread = context._thread
    failure = RuntimeError('handoff fence')
    failures['context.handoff'] = failure
    with pytest.raises(RuntimeError) as found:
        context.handoff_to_current_thread()
    assert found.value is failure and context._thread == original_thread
    with pytest.raises(RuntimeError, match='uncertain'):
        plan.run(*args)
    with pytest.raises(RuntimeError, match='uncertain'):
        context.handoff_to_current_thread()
    with pytest.raises(RuntimeError, match='retry'):
        context.close()
    assert calls.count('context.handoff') == 1 and 'context.close' not in calls


def test_closed_context_cannot_be_handed_off(factory):
    calls, _, _ = factory
    context = dense.Context('cuda:0', backend='cutlass', max_plans=1)
    context.close()
    with pytest.raises(RuntimeError, match='closed'):
        context.handoff_to_current_thread()
    assert 'context.handoff' not in calls


def test_reused_numeric_thread_id_does_not_authorize_a_different_thread_object(factory, monkeypatch):
    calls, _, _ = factory
    with dense.Context('cuda:0', backend='cublas', max_plans=1) as context:
        args = operands()
        plan = dense.prepare(context, *args)
        previous_owner = context._thread
        replacement = SimpleNamespace(ident=previous_owner.ident)
        with monkeypatch.context() as scoped:
            scoped.setattr(dense.threading, 'current_thread', lambda: replacement)
            with pytest.raises(RuntimeError, match='host thread'):
                plan.run(*args)
        assert 'run' not in calls


@pytest.mark.parametrize('capacity', [0, -1, True, 1.5])
def test_invalid_capacity_never_acquires_native(factory, capacity):
    calls, _, _ = factory
    with pytest.raises(ValueError, match='max_plans'):
        dense.Context('cuda:0', backend='cublas', max_plans=capacity)
    assert calls == []

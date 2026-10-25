"""All-rank preparation ordering, one model counter and fatal containment."""

from concurrent.futures import ThreadPoolExecutor
from threading import Barrier, Condition
from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen4_exp.cuda.input_preparation import InputPreparationContext, install_input_preparation


class Store:
    def __init__(self):
        self.condition, self.values, self.aborted = Condition(), {}, False
    def abort(self):
        with self.condition:
            self.aborted = True
            self.condition.notify_all()
    def set(self, key, value):
        with self.condition:
            if self.aborted:
                raise RuntimeError("test supervisor contained the cohort")
            self.values[key] = value
            self.condition.notify_all()
    def get(self, key):
        with self.condition:
            if self.aborted:
                raise RuntimeError("test supervisor contained the cohort")
            return self.values[key]
    def wait(self, keys, timeout):
        with self.condition:
            if not self.condition.wait_for(lambda: self.aborted or all(k in self.values for k in keys), timeout=5):
                raise TimeoutError("test preparation cohort was stranded")
            if self.aborted:
                raise RuntimeError("test supervisor contained the cohort")
    def delete_key(self, key):
        with self.condition:
            return self.values.pop(key, None) is not None


@pytest.mark.parametrize("mode", ["success", "pure_refusal", "copy_oom", "unretired_value", "invalid_window"])
@pytest.mark.parametrize("serving", [False, True])
def test_eight_rank_windows_finish_preparation_before_collectives_or_abort(monkeypatch, mode, serving):
    from tensorfold.cuda import cohort

    store, barrier = Store(), Barrier(8, timeout=5)
    contexts, owners, decoded, notifications, forward = [], [], [], [], []
    def notify():
        notifications.append(True)
        store.abort()
    monkeypatch.setattr(cohort, "notify_failure", notify)
    class Decoder:
        def __init__(self, weights):
            self.w, self.poison_control = weights, None
    for rank in range(8):
        weights = NS(positions=object(), comm=NS(world=8, rank=rank, store=store),
                     meta={"world": 8, "rank": rank, "coordinate_profile_identity": "a" * 64})
        context = install_input_preparation(weights)
        contexts.append(context)
        owners.append(weights)
        if serving:
            decoder = Decoder(weights)
            context.bind_decoder(decoder)
            decoder.poison_control = lambda error: notify()  # Installed after the startup context.
            decoded.append(decoder)
    cause = MemoryError("one-rank coordinate allocation") if mode == "copy_oom" else ValueError("one-rank input proof")
    if mode == "unretired_value":
        cause._tensorfold_retained_owners = (object(),)
    def worker(rank):
        context = contexts[rank]
        def operation():
            assert context.ordinal == 1  # Allocated before local lease/window construction.
            context.prepare("nested-mtp-stage", lambda: None, may_refuse=mode in ("pure_refusal", "unretired_value"))
            barrier.wait()
            if mode == "invalid_window":
                from test_compact_position_frames import prepared
                from tensorfold.families.qwen4_exp.cuda.compact_indexer import stage_target_indexer_frame

                buffers, _position, _states, windows, segments = prepared(profile_identity=context.identity)
                buffers.input_preparation_context = context
                if rank == 3:
                    windows[1].source.tensor.add_(1)  # Actual stale producer binding before forward.
                stage_target_indexer_frame(buffers, segments, windows)
            elif rank == 3 and mode != "success":
                raise cause
            return rank
        try:
            value = context.prepare("target-inputs", operation, may_refuse=mode in ("pure_refusal", "unretired_value"))
            forward.append(rank)
            assert value == rank
            return value
        except BaseException as error:
            return error
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(worker, range(8)))
    assert all(context.ordinal == 1 for context in contexts)
    assert all(context.cohort.multi is None and context._active is None for context in contexts)
    if mode == "success":
        assert results == list(range(8)) and not notifications and not store.values
    elif mode == "pure_refusal":
        assert all(isinstance(result, ValueError) for result in results)
        assert not notifications and not store.values
        with ThreadPoolExecutor(max_workers=8) as pool:
            assert list(pool.map(lambda c: c.prepare("next-inputs", lambda: 7), contexts)) == [7] * 8
        assert not store.values and all(context.ordinal == 2 for context in contexts)
    else:
        if mode == "invalid_window":
            assert isinstance(results[3], ValueError) and "storage changed" in str(results[3])
        else:
            assert results[3] is cause
        assert all(isinstance(result, BaseException) for result in results) and not forward
        assert notifications and store.aborted
        assert all(context.cohort._execution_manifest_uncertain for context in contexts)
        # No normal retirement rendezvous follows uncertain preparation.
        assert not any("/ready/retirement/" in key for key in store.values)


def test_single_rank_nested_calls_share_one_context_without_a_listener():
    weights = NS(positions=object(), comm=None, meta={"coordinate_profile_identity": "a" * 64})
    context = install_input_preparation(weights)
    assert install_input_preparation(weights) is context
    assert context.prepare("target-stage", lambda: context.prepare("mtp-stage", lambda: object())) is not None
    assert context.ordinal == 1 and context.cohort.comm is None


def test_pure_outer_scope_cannot_hide_device_staging(monkeypatch):
    from tensorfold.cuda import cohort

    notified = []
    monkeypatch.setattr(cohort, "notify_failure", lambda: notified.append(True))
    context = InputPreparationContext(NS(positions=object(), comm=None,
                                        meta={"coordinate_profile_identity": "a" * 64}))
    with pytest.raises(RuntimeError, match="pure input preflight"):
        context.prepare("pure", lambda: context.prepare("device", lambda: None), may_refuse=True)
    assert notified == [True] and context.cohort._execution_manifest_uncertain


def test_store_retirement_preserves_an_agreed_refusal_and_reports_its_failure(monkeypatch):
    from tensorfold.cuda import cohort
    from tensorfold.families.qwen4_exp.cuda import input_preparation as module

    context = InputPreparationContext(NS(positions=object(), comm=NS(world=2, rank=0, store=object()),
        meta={"rank": 0, "world": 2, "coordinate_profile_identity": "a" * 64}))
    primary, retirement, notifications = module._ManifestRefusal("pure input refused"), OSError("key retirement"), []
    def phase(*args, **kwargs):
        raise primary
    def retire(*args, **kwargs):
        raise retirement
    monkeypatch.setattr(module, "_phase", phase)
    monkeypatch.setattr(module, "_retire_store", retire)
    monkeypatch.setattr(cohort, "notify_failure", lambda: notifications.append(True))
    with pytest.raises(ValueError) as caught:
        context.prepare("pure", lambda: None, may_refuse=True)
    assert caught.value is primary and primary.__cause__ is retirement
    assert notifications == [True] and context.cohort._execution_manifest_uncertain

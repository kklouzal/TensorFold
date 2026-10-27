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


@pytest.mark.parametrize("mode", ["legacy", "matching", "mismatch", "missing"])
def test_eight_rank_publication_contract_agreement_precedes_every_collective(monkeypatch, mode):
    from tensorfold.cuda import cohort

    store, contexts, notifications, collectives = Store(), [], [], []
    def notify():
        notifications.append(True)
        store.abort()
    monkeypatch.setattr(cohort, "notify_failure", notify)
    for rank in range(8):
        contexts.append(install_input_preparation(NS(positions=object(),
            comm=NS(world=8, rank=rank, store=store),
            meta={"world": 8, "rank": rank, "coordinate_profile_identity": "a" * 64})))
    def worker(rank):
        options = {} if mode == "legacy" else {"contract": "b" * 64}
        if rank == 3 and mode in ("mismatch", "missing"):
            options["contract"] = "c" * 64 if mode == "mismatch" else None
        try:
            result = contexts[rank].prepare("span-publication", lambda: rank, **options)
            collectives.append(rank)
            return result
        except BaseException as error:
            return error
    with ThreadPoolExecutor(max_workers=8) as pool:
        outcomes = list(pool.map(worker, range(8)))
    assert all(context.ordinal == 1 and context._active is None for context in contexts)
    if mode in ("legacy", "matching"):
        assert outcomes == list(range(8)) and sorted(collectives) == list(range(8))
        assert not notifications and not store.values
    else:
        assert all(isinstance(value, BaseException) for value in outcomes)
        assert any("publication contract" in str(value) for value in outcomes)
        assert notifications and not collectives
        assert all(context.cohort._execution_manifest_uncertain for context in contexts)
        assert not any("/ready/retirement/" in key for key in store.values)


@pytest.mark.parametrize("contract", [False, 0, "", "A" * 64, "b" * 63, object()])
def test_invalid_publication_contract_is_refused_before_local_work(monkeypatch, contract):
    from tensorfold.cuda import cohort

    notifications = []
    monkeypatch.setattr(cohort, "notify_failure", lambda: notifications.append(True))
    context = install_input_preparation(NS(positions=object(), comm=None,
        meta={"coordinate_profile_identity": "a" * 64}))
    with pytest.raises(ValueError, match="SHA256"):
        context.prepare("span", lambda: pytest.fail("work before contract validation"), contract=contract)
    assert context.ordinal == 0 and notifications == [True]


def test_nested_publication_contract_cannot_bypass_rank_agreement(monkeypatch):
    from tensorfold.cuda import cohort

    notifications = []
    monkeypatch.setattr(cohort, "notify_failure", lambda: notifications.append(True))
    context = install_input_preparation(NS(positions=object(), comm=None,
        meta={"coordinate_profile_identity": "a" * 64}))
    with pytest.raises(RuntimeError, match="top-level"):
        context.prepare("outer", lambda: context.prepare("nested", lambda: pytest.fail("nested contract work"),
                                                        contract="b" * 64))
    assert context.ordinal == 1 and notifications == [True]


def test_prepared_rejection_preserves_primary_if_cohort_notification_fails():
    from tensorfold.families.qwen4_exp.cuda.input_preparation import reject_prepared

    primary, secondary = ValueError("invalid prepared input"), RuntimeError("notification failed")

    def failed(error):
        assert error is primary
        raise secondary

    with pytest.raises(ValueError) as caught:
        reject_prepared(NS(input_preparation_context=NS(fail=failed)), primary)
    assert caught.value is primary and primary.__cause__ is secondary


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


@pytest.mark.parametrize("bad", [None, "short", "invalid-json", "wrong-type"])
def test_batch_outcomes_use_one_read_and_preserve_containment(monkeypatch, bad):
    from tensorfold.cuda import cohort

    class Batch(Store):
        def __init__(self):
            super().__init__()
            self.calls = []
        def multi_get(self, keys):
            self.calls.append(tuple(keys))
            values = [self.values[key].encode() for key in keys]
            if bad == "short":
                return values[:-1]
            if bad == "invalid-json":
                values[0] = b"["
            if bad == "wrong-type":
                values[0] = "unconverted-native-string"
            return values
        def get(self, key):
            pytest.fail("batch-capable input preparation used scalar GET")

    store, notifications = Batch(), []
    def notify():
        notifications.append(True)
        store.abort()
    monkeypatch.setattr(cohort, "notify_failure", notify)
    contexts = [InputPreparationContext(NS(positions=object(), comm=NS(rank=r, world=8, store=store),
        meta={"rank": r, "world": 8, "coordinate_profile_identity": "a" * 64})) for r in range(8)]
    def work(context):
        try:
            return context.prepare("batch", lambda: 7, contract="b" * 64)
        except BaseException as error:
            return error
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, contexts))
    if bad is None:
        assert results == [7] * 8
        assert len(store.calls) == 8 and all(len(keys) == 8 for keys in store.calls)
        assert not store.values and not notifications
    else:
        assert all(isinstance(result, BaseException) for result in results)
        assert notifications and all(c.cohort._execution_manifest_uncertain for c in contexts)


def test_actual_native_eight_rank_batch_preparation_retires_all_keys():
    torch = pytest.importorskip("torch")
    from datetime import timedelta

    server = torch.distributed.TCPStore("127.0.0.1", 0, 8, True, timedelta(seconds=10),
                                        wait_for_workers=False, use_libuv=False)
    contexts = [InputPreparationContext(NS(positions=object(),
        comm=NS(rank=r, world=8, store=server.clone()),
        meta={"rank": r, "world": 8, "coordinate_profile_identity": "c" * 64})) for r in range(8)]
    baseline = server.num_keys()
    for epoch in range(3):
        with ThreadPoolExecutor(max_workers=8) as pool:
            results = list(pool.map(lambda c: c.prepare("native-batch", lambda: epoch, contract="d" * 64), contexts))
        assert results == [epoch] * 8
        assert server.num_keys() == baseline + 8  # One reusable outcome slot per rank, no epoch keys survive.
    assert all(c.ordinal == 3 for c in contexts)
    for context in contexts:
        context.close()
    assert server.num_keys() == baseline


def test_batch_phase_raw_buffers_have_explicit_global_funding():
    import struct
    import sys
    from tensorfold.families.qwen4_exp.cuda.execution_identity import input_preparation_value_bytes
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        native_sync_control_payload, retention_header_resources,
    )

    for world in (2, 8):
        frame = input_preparation_value_bytes()
        record = native_sync_control_payload(world=world, chunk_bytes=0, ram=False)
        assert record["input_preparation_batch_native_extra_bytes"] == world * (
            (world + 4) * frame + 32 * world + 48) + frame
        assert record["input_preparation_python_raw_bytes"] == world * (
            world * (sys.getsizeof(b"") + frame)
            + sys.getsizeof([]) + struct.calcsize("P") * world)
        assert record["global_python_preparation_payload_bytes"] == sum(record[key] for key in (
            "input_preparation_python_raw_bytes", "input_preparation_python_graph_bytes",
            "input_preparation_python_metadata_bytes"))
        header = retention_header_resources(None, world)
        assert header.global_host_preparation_bytes == record["global_python_preparation_payload_bytes"]
        assert header.record()["schema"] == "tensorfold.retention-header-resources-v2"
    assert retention_header_resources(None, 1).global_host_bytes == 0


class MailboxStore(Store):
    def __init__(self):
        super().__init__()
        self.deletions, self.maximum_keys = [], 0
    def set(self, key, value):
        with self.condition:
            super().set(key, value)
            self.maximum_keys = max(self.maximum_keys, len(self.values))
    def add(self, key, increment):
        with self.condition:
            if self.aborted:
                raise RuntimeError("test supervisor contained the cohort")
            value = int(self.values.get(key, "0")) + increment
            self.values[key] = str(value)
            self.maximum_keys = max(self.maximum_keys, len(self.values))
            self.condition.notify_all()
            return value
    def multi_get(self, keys):
        with self.condition:
            if self.aborted:
                raise RuntimeError("test supervisor contained the cohort")
            return [self.values[key].encode() for key in keys]
    def delete_key(self, key):
        self.deletions.append(key)
        return super().delete_key(key)


def mailbox_contexts(store, world=8):
    return [InputPreparationContext(NS(positions=object(), comm=NS(rank=r, world=world, store=store),
        meta={"rank": r, "world": world, "coordinate_profile_identity": "e" * 64})) for r in range(world)]


def test_reusable_mailbox_bounds_keys_and_delete_work_across_model_lifetimes():
    store = MailboxStore()
    for _generation in range(3):
        contexts = mailbox_contexts(store)
        for epoch in range(16):
            with ThreadPoolExecutor(max_workers=8) as pool:
                assert list(pool.map(lambda c: c.prepare("reused", lambda: epoch), contexts)) == [epoch] * 8
            assert len(store.values) == 8
        for context in contexts:
            context.close()
            context.close()  # Known completed close is idempotent.
            with pytest.raises(RuntimeError, match="retired"):
                context.prepare("late", lambda: pytest.fail("work after model retirement"))
        assert not store.values
    assert len(store.deletions) == 3 * (16 * 4 + 8)
    assert store.maximum_keys <= 8 + 5


def test_delayed_consumed_wait_cannot_be_deleted_before_its_reader_finishes():
    from threading import Event

    delayed, release = Event(), Event()
    class Delayed(MailboxStore):
        def wait(self, keys, timeout):
            if keys[0].endswith("/consumed") and not delayed.is_set():
                with self.condition:
                    if not delayed.is_set():
                        delayed.set()
                        first = True
                    else:
                        first = False
                if first:
                    if not release.wait(5):
                        raise TimeoutError("test did not release delayed WAIT submission")
            return super().wait(keys, timeout)
    store = Delayed()
    contexts = mailbox_contexts(store)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = [pool.submit(c.prepare, "delayed", lambda: 7) for c in contexts]
        assert delayed.wait(5)
        assert not any(key.endswith("/consumed") for key in store.deletions)
        release.set()
        assert [f.result(timeout=5) for f in futures] == [7] * 8
    for context in contexts:
        context.close()
    assert not store.values


@pytest.mark.parametrize("fault", ["stale", "label", "contract", "bad-arrival", "after-set", "after-add", "delete"])
def test_mailbox_uncertainty_poison_preserves_original_failure(monkeypatch, fault):
    import json
    from tensorfold.cuda import cohort

    primary = OSError("injected mailbox transport failure")
    class Broken(MailboxStore):
        def set(self, key, value):
            super().set(key, value)
            if fault == "after-set" and "/rank/3" in key:
                raise primary
        def add(self, key, increment):
            value = super().add(key, increment)
            if fault == "after-add" and value == 3:
                raise primary
            return 999 if fault == "bad-arrival" and value == 3 else value
        def multi_get(self, keys):
            result = super().multi_get(keys)
            if fault == "stale":
                row = json.loads(result[0])
                row["ordinal"] -= 1
                result[0] = json.dumps(row).encode()
            return result
        def delete_key(self, key):
            value = super().delete_key(key)
            if fault == "delete" and key.endswith("/count"):
                raise primary
            return value
    store, notifications = Broken(), []
    def notify():
        notifications.append(True)
        store.abort()
    monkeypatch.setattr(cohort, "notify_failure", notify)
    contexts = mailbox_contexts(store)
    def work(context):
        try:
            label = "other-label" if fault == "label" and context.cohort.rank == 3 else "fault"
            contract = "a" * 64 if fault == "contract" and context.cohort.rank == 3 else "b" * 64
            return context.prepare(label, lambda: 7, contract=contract)
        except BaseException as error:
            return error
    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(work, contexts))
    assert notifications
    assert (any(c.cohort._execution_manifest_uncertain for c in contexts) if fault == "delete" else
            all(c.cohort._execution_manifest_uncertain for c in contexts))
    # Peers can return before rank0's final deletion fails, as in the original
    # retirement protocol. That failure still poisons every later collective.
    if fault != "delete":
        assert all(isinstance(result, BaseException) for result in results)
    if fault in ("after-set", "after-add", "delete"):
        assert any(result is primary for result in results)
    if fault == "delete":
        assert len(store.deletions) == 4
    for context in contexts:
        if context.cohort._execution_manifest_uncertain:
            with pytest.raises(RuntimeError, match="poisoned"):
                context.prepare("next", lambda: pytest.fail("work after uncertain epoch"))


def test_mailbox_agreed_refusal_is_reusable_and_epoch_exhaustion_precedes_work(monkeypatch):
    from tensorfold.cuda import cohort

    store = MailboxStore()
    contexts = mailbox_contexts(store)
    notifications = []
    monkeypatch.setattr(cohort, "notify_failure", lambda: notifications.append(True))
    def refuse(context):
        try:
            def operation():
                if context.cohort.rank == 3:
                    raise ValueError("bounded pure refusal")
            context.prepare("refusal", operation, may_refuse=True)
        except ValueError as error:
            return error
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert all(isinstance(e, ValueError) for e in pool.map(refuse, contexts))
        assert list(pool.map(lambda c: c.prepare("next", lambda: 8), contexts)) == [8] * 8
    assert not notifications and len(store.values) == 8
    for context in contexts:
        context.ordinal = (1 << 63) - 1
        with pytest.raises(RuntimeError, match="ordinal exhausted"):
            context.prepare("overflow", lambda: pytest.fail("work after ordinal exhaustion"))
    assert notifications == [True] * 8


def test_model_close_failure_blocks_new_preparation_and_retains_retry_owner():
    primary = OSError("slot delete succeeded before native return failed")
    class CloseFault(MailboxStore):
        fail = True
        def delete_key(self, key):
            result = super().delete_key(key)
            if "/rank/3" in key and self.fail:
                self.fail = False
                raise primary
            return result
    store = CloseFault()
    contexts = mailbox_contexts(store)
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda c: c.prepare("last", lambda: 7), contexts)) == [7] * 8
    with pytest.raises(OSError) as caught:
        contexts[0].close()
    assert caught.value is primary
    assert contexts[0].cohort._phase_mailbox.store is store and not contexts[0]._closed
    with pytest.raises(RuntimeError, match="retired"):
        contexts[0].prepare("new", lambda: pytest.fail("new work after failed model close"))
    contexts[0].close()
    for context in contexts[1:]:
        context.close()
    assert not store.values
    assert all(c._closed and c.cohort._phase_mailbox is None for c in contexts)


@pytest.mark.parametrize("after_begin", [False, True])
@pytest.mark.parametrize("notification_fails", [False, True])
@pytest.mark.parametrize("serving", [False, True])
def test_epoch_setup_failure_notifies_before_any_operation_and_preserves_primary(
        monkeypatch, after_begin, notification_fails, serving):
    from tensorfold.cuda import cohort

    context = mailbox_contexts(MailboxStore(), world=2)[0]
    primary, secondary = MemoryError("mailbox epoch allocation"), OSError("supervisor notification")
    notifications, poisoned, work = [], [], []
    def notify():
        notifications.append(True)
        if notification_fails:
            raise secondary
    monkeypatch.setattr(cohort, "notify_failure", notify)
    if serving:
        weights = NS(meta={"input_preparation_context": context})
        def poison(error):
            poisoned.append(error)
            notify()
        class Decoder:
            pass
        decoder = Decoder()
        decoder.w, decoder.poison_control = weights, poison
        context.bind_decoder(decoder)
    mailbox = context.cohort._phase_mailbox
    begin = mailbox.begin
    def fail_begin(*args):
        if after_begin:
            begin(*args)
        raise primary
    monkeypatch.setattr(mailbox, "begin", fail_begin)
    with pytest.raises(MemoryError) as caught:
        context.prepare("epoch", lambda: work.append(True))
    assert caught.value is primary
    assert primary.__cause__ is (secondary if notification_fails else None)
    assert context.ordinal == 1 and notifications == [True] and not work
    assert context.cohort._execution_manifest_uncertain and context.cohort._abort_comm
    assert context.cohort.multi is None and context._active is None and context._pure is False
    assert bool(mailbox.active) is after_begin
    assert poisoned == ([primary] if serving else [])
    with pytest.raises(RuntimeError, match="poisoned"):
        context.prepare("next", lambda: pytest.fail("work after failed epoch setup"))

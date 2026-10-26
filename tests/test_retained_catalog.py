# ruff: noqa: E402 - optional torch capability must be checked before CUDA-family imports.
"""Actual State bytes through the common bounded RAM-transfer implementation."""

import threading
from types import SimpleNamespace

import pytest
torch = pytest.importorskip("torch")

from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.retained_catalog import LogicalReference, RetentionCapacity, make_retention_catalog
from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy
from tensorfold.families.qwen4_exp.cuda.state_retention import state_inventory
pytest_plugins = ("test_state_retention",)


def owner(state, *, tier="ram", prefix_limit=2, ram_bytes=None):
    future = state_inventory(state, recovery_versions=1).future_bytes
    policy = (RetentionPolicy(tier, 2, 10, future * 6 + (64 << 20) if ram_bytes is None else ram_bytes, 37, 74, 0) if tier == "ram" else
              RetentionPolicy(tier, 1, 10))
    wake = threading.Event()
    weights = SimpleNamespace(meta={"rank": 0}, comm=None)
    catalog = make_retention_catalog(policy, weights, [state, state.clone()], prefix_limit=prefix_limit, recovery_versions=1, wake=wake.set)
    return catalog, catalog.future_claim(), wake


def captured(catalog, state, future):
    ref = LogicalReference("local-unauthenticated", "response:one", 1)
    handle = catalog.reserve(ref, future_bytes=future)
    stream = Stream([3, 7, 11, 9], 8, draft=False, st=state)
    catalog.capture_idle(stream, handle=handle, recoveries=(state.snapshot(),))
    return ref, handle


def test_actual_request_temporary_checkpoint_roundtrips_without_a_logical_reference(materialized):
    catalog, future, _wake = owner(materialized)
    catalog.configure_admission(max_waiting_input_bytes=100000, max_pending_requests=2)
    handle = catalog.reserve_temporary("local-unauthenticated")
    stream = Stream([3, 7, 11, 9], 8, draft=False, st=materialized)
    catalog.capture_idle(stream, handle=handle)
    expected = {part.name: part.tensor.clone() for part in handle.inventory.parts}
    assert not handle.owners and handle.future_bytes == future
    assert catalog.suspend(handle).retired.wait(5)
    assert catalog.saved(handle)
    request = catalog.borrow(handle, continuation="same-request")
    destination = catalog.states[1]
    assert request.claim(destination, stream=stream) is None
    assert handle.transfer.retired.wait(5)
    handoff = catalog.restored(request)
    assert handoff is not None and handoff["continuation"] == "same-request" and handle.state is destination
    current = state_inventory(destination, stream=stream, recovery_versions=catalog.recovery_versions)
    assert all(torch.equal(part.tensor, expected[part.name]) for part in current.parts)
    catalog.release_temporary(handle)
    assert catalog.has_retained  # The actual consumer, not an alias, still owns the checkpoint.
    request.close()
    assert not catalog.has_retained
    catalog.close(timeout=5)


def test_optional_ram_prefix_is_usable_by_disposable_request_and_principal_isolated(materialized):
    catalog, _future, _wake = owner(materialized)
    catalog.configure_admission(max_waiting_input_bytes=100000, max_pending_requests=2)
    stream = Stream([3, 7, 11, 9], 8, draft=False, st=materialized)
    handle = catalog.capture_prefix(stream, principal="principal-a")
    assert catalog.suspend(handle).retired.wait(5)
    assert catalog.saved(handle)
    assert catalog.best_prefix([3, 7, 11, 9, 4], principal="principal-a") is handle
    assert catalog.best_prefix([3, 7, 11, 9, 4], principal="principal-b") is None
    request = catalog.borrow_prefix(handle, principal="principal-a")
    fresh = Stream([3, 7, 11, 9, 4], 7, draft=False)
    assert request.claim(catalog.states[1], stream=fresh) is None
    assert request.restore_handle.transfer.retired.wait(5)
    assert catalog.restored(request)["continuation"] == "new-turn"
    assert request.target_handle.future_bytes == 0 and not request.target_handle.owners
    request.close()
    catalog.release_owner(handle)
    assert not catalog.has_retained
    catalog.close(timeout=5)


def test_optional_prefix_entry_bound_reclaims_only_unborrowed_generation(materialized):
    catalog, _future, _wake = owner(materialized, prefix_limit=1)
    catalog.configure_admission(max_waiting_input_bytes=100000, max_pending_requests=1)
    stream = Stream([3, 7, 11, 9], 8, draft=False, st=materialized)
    old = catalog.capture_prefix(stream, principal="principal-a")
    assert catalog.suspend(old).retired.wait(5)
    assert catalog.saved(old)
    reader = catalog.borrow_prefix(old, principal="principal-a")
    with pytest.raises(RetentionCapacity, match="role capacity"):
        catalog.capture_prefix(stream, principal="principal-a")
    reader.close()
    current = catalog.capture_prefix(stream, principal="principal-a")
    assert current.serial > old.serial and old.serial not in catalog._handles
    assert catalog.record()["ram_prefix_generations"] == 1
    catalog.release_owner(current)
    catalog.close(timeout=5)


def test_temporary_role_is_bounded_by_actual_request_envelope_not_logical_aliases(materialized):
    catalog, _future, _wake = owner(materialized)
    catalog.configure_admission(max_waiting_input_bytes=100000, max_pending_requests=1)
    temporary = catalog.reserve_temporary("local-unauthenticated")
    with pytest.raises(RetentionCapacity, match="role capacity"):
        catalog.reserve_temporary("local-unauthenticated")
    logical = catalog.reserve(LogicalReference("local-unauthenticated", "response:independent", 1),
                              future_bytes=catalog.future_claim())
    assert catalog.record()["sessions"] == 1
    assert catalog.record()["temporary_checkpoints"] == 1
    catalog.release_temporary(temporary)
    catalog.release_owner(logical)
    catalog.close(timeout=5)


def test_required_future_claim_reclaims_optional_ram_only_after_its_reader_retires(materialized):
    probe, future, _wake = owner(materialized, prefix_limit=1)
    probe.close(timeout=5)
    catalog, actual, _wake = owner(materialized, prefix_limit=1, ram_bytes=2 * future)
    assert actual == future
    catalog.configure_admission(max_waiting_input_bytes=100000, max_pending_requests=1)
    prefix = catalog.capture_prefix(Stream([3, 7, 11, 9], 8, draft=False, st=materialized), principal="p")
    temporary = catalog.reserve_temporary("p")
    reader = catalog.borrow_prefix(prefix, principal="p")
    reference = LogicalReference("p", "response:required", 1)
    assert catalog.reserve_eligibility() == "tier_unavailable"
    with pytest.raises(RetentionCapacity, match="future RAM"):
        catalog.reserve(reference, future_bytes=future)
    assert prefix.serial in catalog._handles
    reader.close()
    assert catalog.reserve_eligibility() is None
    logical = catalog.reserve(reference, future_bytes=future)
    assert prefix.serial not in catalog._handles
    assert catalog.record()["future_backing_bytes"] == 2 * future
    catalog.release_temporary(temporary)
    catalog.release_owner(logical)
    catalog.close(timeout=5)


def test_optional_prefix_capture_does_not_retain_original_stream_or_delivery_callbacks(materialized):
    import gc
    import weakref

    class Delivery:
        def emit(self, _tokens):
            return False
    catalog, _future, _wake = owner(materialized)
    catalog.configure_admission(max_waiting_input_bytes=100000, max_pending_requests=1)
    delivery = Delivery()
    stream = Stream([3, 7, 11, 9], 8, draft=False, st=materialized, emit=delivery.emit)
    stream.delivery_owner = delivery
    stream_ref, delivery_ref = weakref.ref(stream), weakref.ref(delivery)
    handle = catalog.capture_prefix(stream, principal="p")
    assert not hasattr(handle, "capture_stream")
    stream.prompt[0] = 2
    assert handle.canonical_ids.tolist() == [3, 7, 11]
    del stream, delivery
    gc.collect()
    assert stream_ref() is None and delivery_ref() is None
    assert catalog.suspend(handle).retired.wait(5)
    assert catalog.saved(handle)
    catalog.release_owner(handle)
    catalog.close(timeout=5)


def test_completed_temporary_promotion_reuses_quota_and_hands_off_real_cpu_owner(materialized):
    catalog, _future, _wake = owner(materialized)
    catalog.configure_admission(max_waiting_input_bytes=100000, max_pending_requests=1)
    temporary = catalog.reserve_temporary("p")
    stream = Stream([3, 7, 11, 9], 8, draft=False, st=materialized)
    host = object()
    catalog.capture_idle(stream, handle=temporary, host=host)
    reserved = catalog.record()["future_backing_bytes"]
    with pytest.raises(ValueError, match="CPU host-owner handoff"):
        catalog.capture_prefix(stream, principal="p", from_handle=temporary)
    assert temporary.role == "temporary" and temporary.host is host
    journal = []
    prefix = catalog.capture_prefix(stream, principal="p", from_handle=temporary, detach_host=journal.append)
    assert prefix is temporary and prefix.role == "prefix"
    assert journal == [host] and prefix.host is None and prefix.ready_handoff["host"] is None
    assert catalog.record()["future_backing_bytes"] == reserved
    with pytest.raises(ValueError, match="temporary retirement"):
        catalog.release_temporary(prefix)
    catalog.release_owner(prefix)
    catalog.close(timeout=5)


@pytest.mark.parametrize("room", [False, True])
def test_actual_restore_enforces_header_recovery_pool_before_target_allocation(materialized, monkeypatch, room):
    snapshot = materialized.snapshot()
    inventory = state_inventory(materialized, recoveries=(snapshot,), recovery_versions=2)
    needed = sum(part.tensor.nbytes for part in inventory.parts if part.name.startswith("recovery."))
    catalog, future, _wake = owner(materialized)
    catalog.configure_device_resources(recovery_pool_bytes=needed if room else needed - 1,
                                       pending_hc_bytes=0, snapshot_bytes=needed)
    _ref, handle = captured(catalog, materialized, future)
    assert catalog.suspend(handle).retired.wait(5) and catalog.saved(handle)
    request = catalog.borrow(handle)
    original, allocations = torch.empty, []
    def counted(*args, **kwargs):
        value = original(*args, **kwargs)
        allocations.append(value.nbytes)
        return value
    monkeypatch.setattr(torch, "empty", counted)
    stream = Stream([3, 7, 11, 9], 8, draft=False)
    if room:
        assert request.claim(catalog.states[1], stream=stream) is None
        assert handle.transfer.retired.wait(5)
        assert catalog.restored(request) is not None
        assert sum(allocations) >= needed
    else:
        with pytest.raises(RetentionCapacity, match="combined GPU overlap"):
            request.claim(catalog.states[1], stream=stream)
        assert allocations == [] and handle.transfer is None and handle.phase == "ram"
    request.close()
    catalog.release_owner(handle)
    catalog.close(timeout=5)


def test_failed_temporary_backend_requires_real_reset_and_preserves_backing_until_owner_release(materialized, materialized_family):
    catalog, future, _wake = owner(materialized)
    catalog.configure_admission(max_waiting_input_bytes=100000, max_pending_requests=1)
    handle = catalog.reserve_temporary("p")
    host, primary = object(), ValueError("original cancelled backend")
    catalog.capture_idle(Stream([3, 7, 11, 9], 8, draft=False, st=materialized), handle=handle, host=host)
    with pytest.raises(RuntimeError, match="healthy reset"):
        catalog.retire_backend(handle, state=materialized, fenced=True, error=primary)
    assert handle.state is materialized and handle.phase == "gpu"
    materialized.reset(materialized_family[0])
    journal = []
    assert catalog.retire_backend(handle, state=materialized, fenced=True, error=primary,
                                  detach_host=journal.append) == (materialized,)
    assert journal == [host] and handle.state is None and handle.failure is primary
    assert catalog.record()["future_backing_bytes"] == future
    assert catalog.take_released_states() == (materialized,)
    assert catalog.retire_backend(handle, state=materialized, fenced=True, error=primary) == ()
    catalog.release_temporary(handle)
    assert catalog.record()["future_backing_bytes"] == 0
    catalog.close(timeout=5)


def test_failed_pending_target_retirement_does_not_drop_published_parent(materialized, materialized_family):
    catalog, future, _wake = owner(materialized)
    reference, parent = captured(catalog, materialized, future)
    catalog.logical_reference_committed(reference.principal, reference.sequence, generation=reference.generation)
    catalog.progress()
    target = catalog.reserve(LogicalReference(reference.principal, "response:child", 1), future_bytes=future)
    destination = catalog.states[1]
    catalog.capture_idle(Stream([3, 7, 11, 9], 8, draft=False, st=destination), handle=target)
    primary = ValueError("failed independent child")
    destination.reset(materialized_family[0])
    catalog.retire_backend(target, state=destination, fenced=True, error=primary)
    assert parent.state is materialized and parent.phase == "gpu" and parent.failure is None
    assert catalog.lookup(reference) is parent
    with pytest.raises(RuntimeError, match="published ownership"):
        catalog.retire_backend(parent, state=materialized, fenced=True, error=primary)
    catalog.release_owner(target)
    catalog.release_owner(parent)
    catalog.close(timeout=5)


def test_restore_releases_each_ordinary_chunk_before_allocating_the_next(materialized, monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.ram_backing import RamReader

    catalog, future, _wake = owner(materialized)
    _reference, handle = captured(catalog, materialized, future)
    assert catalog.suspend(handle).retired.wait(5) and catalog.saved(handle)
    original = RamReader.read
    class TrackedBytes(bytes):
        live = 0
        calls = 0
        def __new__(cls, value):
            result = super().__new__(cls, value)
            cls.live += 1
            return result
        def __del__(self):
            type(self).live -= 1
    def tracked(reader, offset, count):
        # Assert BEFORE the real allocator/read, not after assignment has
        # already destroyed the old local. Byte subclass is a lifetime oracle;
        # actual RAM bytes and all ordinary restore machinery remain in use.
        assert TrackedBytes.live == 0, "previous ordinary chunk still owned at the next allocation"
        TrackedBytes.calls += 1
        return TrackedBytes(original(reader, offset, count))
    monkeypatch.setattr(RamReader, "read", tracked)
    request = catalog.borrow(handle)
    assert request.claim(catalog.states[1], stream=Stream([3, 7, 11, 9], 8, draft=False)) is None
    transfer = handle.transfer
    assert transfer.retired.wait(5)
    assert catalog.restored(request) is not None, transfer.failure
    assert TrackedBytes.calls > 1 and TrackedBytes.live == 0
    request.close()
    catalog.release_owner(handle)
    catalog.close(timeout=5)


def test_actual_ram_roundtrip_and_core_protected_recovery(materialized):
    catalog, future, wake = owner(materialized)
    ref, handle = captured(catalog, materialized, future)
    original = state_inventory(materialized, recoveries=(materialized.snapshot(),), recovery_versions=1)
    expected = {part.name: part.tensor.clone() for part in original.parts}
    save = catalog.suspend(handle)
    assert save.retired.wait(5)
    assert catalog.saved(handle), save.failure
    assert handle.state is None and handle.inventory is None
    destination = catalog.states[1]
    destination.begin_sequence_generation()
    destination.set_pos(0)
    for cache in destination.kc:
        cache.k.fill_(float("nan"))
        cache.v.fill_(float("nan"))
    request = catalog.borrow(handle)
    assert request.claim(destination, stream=Stream([3, 7, 11, 9], 8, draft=False)) is None
    restore = handle.transfer
    assert restore.retired.wait(5)
    handoff = catalog.restored(request)
    assert handoff is not None, restore.failure
    inventory = state_inventory(destination, recoveries=handoff["recoveries"], recovery_versions=1)
    for part in inventory.parts:
        assert torch.equal(part.tensor, expected[part.name]), part.name
    assert destination.pos == 3 and destination.pos_dev.item() == 3
    assert wake.is_set()
    request.close()
    assert catalog.has_retained
    catalog.logical_reference_unavailable(ref.principal, ref.sequence, generation=1, reason="deleted")
    catalog.progress()
    assert not catalog.has_retained
    catalog.close(timeout=5)


def test_queued_cancel_and_response_alias_do_not_delete_conversation_owner(materialized):
    catalog, future, _wake = owner(materialized, tier="gpu")
    ref, handle = captured(catalog, materialized, future)
    conv = LogicalReference(ref.principal, "conversation:c", 1)
    catalog.associate(handle, conv)
    catalog.logical_reference_committed(conv.principal, conv.sequence, generation=1)
    catalog.progress()
    request = catalog.borrow(handle)
    request.close()
    assert catalog.lookup(conv) is handle
    catalog.logical_reference_unavailable(ref.principal, ref.sequence, generation=1, reason="deleted")
    catalog.progress()
    assert catalog.lookup(conv) is handle and catalog.has_retained
    with pytest.raises(RuntimeError, match="drain"):
        catalog.close(timeout=0)
    catalog.release_owner(handle)
    catalog.close(timeout=0)


def test_prefix_entries_share_catalog_and_complete_promises_are_not_current_rows(materialized):
    catalog, future, _wake = owner(materialized)
    entry = ([3, 7], materialized, materialized.snapshot(), None)
    catalog.prefix_entries = [entry]
    assert catalog.prefix_entries[0] is entry
    with pytest.raises(RetentionCapacity, match="future"):
        catalog.reserve(LogicalReference("p", "response:large", 1), future_bytes=catalog.policy.ram_bytes + 1)
    assert catalog.record()["future_backing_bytes"] == 0
    catalog.prefix_entries = []
    catalog.close(timeout=5)


def test_corrupt_ram_is_a_retired_session_error_before_any_destination_write(materialized):
    from tensorfold.families.qwen4_exp.cuda.retained_catalog import RetentionCorrupt

    catalog, future, _wake = owner(materialized)
    _ref, handle = captured(catalog, materialized, future)
    save = catalog.suspend(handle)
    assert save.retired.wait(5) and catalog.saved(handle)
    key = next(key for key in handle.keys if catalog._ram.descriptor(key)["bytes"])
    catalog._ram._objects[key].data[0] ^= 1  # Fault in the actual sole backing object.
    destination = catalog.states[1]
    before = destination.rec.clone()
    request = catalog.borrow(handle)
    request.claim(destination, stream=Stream([3, 7, 11, 9], 8, draft=False))
    transfer = handle.transfer
    assert transfer.retired.wait(5)
    with pytest.raises(RetentionCorrupt):
        catalog.restored(request)
    assert catalog.retire_failed_destination(request, fenced=True) == (destination,)
    assert handle.state is None and not catalog.protected_states()
    catalog.check_health()
    assert torch.equal(destination.rec, before)
    request.close()
    catalog.release_owner(handle)
    catalog.close(timeout=5)


def test_partial_factory_owner_is_published_before_validation_and_is_close_safe(materialized):
    owners = []
    with pytest.raises(ValueError, match="prefix"):
        make_retention_catalog(None, SimpleNamespace(meta={"rank": 0}, comm=None), [materialized],
                               prefix_limit=-1, recovery_versions=0, register=owners.append)
    assert len(owners) == 1
    owners[0].close(timeout=0)


def test_cuda_completion_is_issued_and_observed_only_by_actor(monkeypatch, materialized):
    catalog, _future, wake = owner(materialized)
    actor = threading.get_ident()
    calls = []

    class Event:
        def record(self, stream):
            calls.append(("record", threading.get_ident()))

        def query(self):
            calls.append(("query", threading.get_ident()))
            return False

        def synchronize(self):
            calls.append(("fence", threading.get_ident()))

    monkeypatch.setattr(torch.cuda, "Event", Event)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device: object())
    result = []
    worker = threading.Thread(target=lambda: result.append(catalog._actor(lambda: 7, SimpleNamespace(type="cuda"))))
    worker.start()
    assert wake.wait(5)
    assert catalog.progress()
    assert worker.is_alive()
    assert not catalog.progress(idle=True)
    worker.join(5)
    assert result == [7] and calls and all(thread == actor for _name, thread in calls)
    catalog.close(timeout=5)


def test_terminal_service_close_preserves_consumer_until_its_actual_release(materialized):
    from tensorfold.families.qwen4_exp.cuda.retained_catalog import RetentionError

    catalog, future, _wake = owner(materialized, tier="gpu")
    _ref, handle = captured(catalog, materialized, future)
    request = catalog.borrow(handle)
    catalog.begin_shutdown()
    assert not catalog.shutdown_progress()
    assert catalog.protected_states() == (materialized,)
    with pytest.raises(RetentionError, match="shutting"):
        catalog.reserve(LogicalReference("p", "response:late", 1), future_bytes=future)
    request.close()
    assert catalog.shutdown_progress()
    assert catalog.take_released_states() == (materialized,)
    catalog.close(timeout=5)


def test_queued_branch_does_not_pin_idle_gpu_and_restore_owns_independent_target(materialized):
    catalog, future, _wake = owner(materialized)
    ref, parent = captured(catalog, materialized, future)
    target = catalog.reserve(LogicalReference(ref.principal, "response:branch", 1), future_bytes=future)
    request = catalog.borrow(parent, continuation="new-turn", target_handle=target)
    saved = catalog.suspend(parent)  # Accepted queue reference is not a GPU reader.
    assert saved.retired.wait(5) and catalog.saved(parent)
    destination = catalog.states[1]
    stream = Stream([3, 7, 11, 9, 5], 8, draft=False, st=destination)
    request.claim(destination, stream=stream)
    assert parent.phase == "ram" and target.phase == "restoring"
    assert target.transfer.retired.wait(5)
    assert catalog.restored(request) is not None
    assert parent.phase == "ram" and target.phase == "gpu" and target.state is destination
    request.close()
    catalog.capture_idle(stream, handle=target)
    catalog.release_owner(parent)
    assert catalog._ram.record()["objects"] > 0  # Target owns shared backing independently.
    transfer = catalog.suspend(target)
    assert transfer.retired.wait(5) and catalog.saved(target)
    catalog.release_owner(target)
    catalog.close(timeout=5)


def test_parallel_gpu_response_branches_preserve_the_original_required_version(materialized):
    states = [materialized, materialized.clone(), materialized.clone()]
    catalog = make_retention_catalog(RetentionPolicy("gpu", 3, 10), SimpleNamespace(meta={"rank": 0}, comm=None),
                                     states, prefix_limit=0, recovery_versions=1)
    ref, parent = captured(catalog, materialized, catalog.future_claim())
    targets = [catalog.reserve(LogicalReference(ref.principal, f"response:branch{i}", 1),
                               future_bytes=catalog.future_claim()) for i in range(2)]
    requests = [catalog.borrow(parent, continuation="new-turn", target_handle=target) for target in targets]
    original = materialized.rec.clone()
    for request, state in zip(requests, states[1:]):
        result = request.claim(state, stream=Stream([3, 7, 11, 9, 5], 8, draft=False))
        assert result["forked"]
        state.rec.add_(11)
        request.close()
    assert torch.equal(materialized.rec, original)
    assert len(catalog.protected_states()) == 3
    for handle in [parent, *targets]:
        catalog.release_owner(handle)
    catalog.close(timeout=0)


def test_registration_faults_leave_no_invisible_promise_alias_or_consumer(materialized):
    class FailedInsert(dict):
        def __setitem__(self, key, value):
            super().__setitem__(key, value)
            raise MemoryError("injected insertion failure after publication")

    class FailedAdd(set):
        def add(self, value):
            super().add(value)
            raise MemoryError("injected target hold failure after publication")

    catalog, future, _wake = owner(materialized)
    first = LogicalReference("local-unauthenticated", "response:invisible", 1)
    catalog._references = FailedInsert()
    with pytest.raises(MemoryError):
        catalog.reserve(first, future_bytes=future)
    assert catalog.record()["sessions"] == catalog.record()["future_backing_bytes"] == 0
    assert catalog._serial == 1 and not catalog._references and catalog._registration_owner is None
    catalog._references = {}
    ref, handle = captured(catalog, materialized, future)
    alias = LogicalReference(ref.principal, "conversation:alias", 1)
    catalog._references = FailedInsert(catalog._references)
    with pytest.raises(MemoryError):
        catalog.associate(handle, alias)
    assert alias not in handle.owners and alias not in catalog._references
    catalog._references = dict(catalog._references)
    target = catalog.reserve(LogicalReference(ref.principal, "response:target", 1), future_bytes=future)
    target.request_holds = FailedAdd()
    with pytest.raises(MemoryError):
        catalog.borrow(handle, continuation="new-turn", target_handle=target)
    assert not handle.consumers and not target.request_holds and catalog._registration_owner is None
    catalog.check_health()
    catalog.release_owner(target)
    catalog.release_owner(handle)
    catalog.close(timeout=5)


def test_new_turn_backing_ref_failure_keeps_transfer_and_rolls_back_exact_journal(monkeypatch, materialized):
    catalog, future, _wake = owner(materialized)
    ref, parent = captured(catalog, materialized, future)
    save = catalog.suspend(parent)
    assert save.retired.wait(5) and catalog.saved(parent)
    target = catalog.reserve(LogicalReference(ref.principal, "response:ref-fault", 1), future_bytes=future)
    request = catalog.borrow(parent, continuation="new-turn", target_handle=target)
    request.claim(catalog.states[1], stream=Stream([3, 7, 11, 9, 5], 8, draft=False))
    transfer = target.transfer
    assert transfer.retired.wait(5)
    original = catalog._ram.retain_into
    calls = 0

    def fail_after_publication(key, journal, index):
        nonlocal calls
        original(key, journal, index)
        calls += 1
        if calls == 2:
            raise MemoryError("injected retained-key publication failure")

    with monkeypatch.context() as faults:
        faults.setattr(catalog._ram, "retain_into", fail_after_publication)
        with pytest.raises(MemoryError):
            catalog.restored(request)
    assert target.transfer is transfer and target.phase == "restoring"
    assert target.keys == () and all(key is None for key in transfer.retained_keys)
    assert all(catalog._ram._objects[key].references == 1 for key in parent.keys)
    assert catalog.restored(request) is not None
    request.close()
    catalog.release_owner(target)
    catalog.release_owner(parent)
    catalog.close(timeout=5)


def test_disposable_ram_prefix_consumer_reserves_no_new_retained_backing(materialized):
    catalog, future, _wake = owner(materialized)
    ref, parent = captured(catalog, materialized, future)
    catalog.logical_reference_committed(ref.principal, ref.sequence, generation=1)
    catalog.progress()
    transfer = catalog.suspend(parent)
    assert transfer.retired.wait(5) and catalog.saved(parent)
    catalog.configure_admission(max_waiting_input_bytes=4096, max_pending_requests=2)
    assert catalog.best_prefix([3, 7, 11, 9, 5], principal=ref.principal) is parent
    reserved = catalog.record()["future_backing_bytes"]
    journals = []
    request = catalog.borrow_prefix(parent, principal=ref.principal, register=journals.append)
    assert journals == [request] and request.target_handle.transient
    assert catalog.record()["sessions"] == 1 and catalog.record()["future_backing_bytes"] == reserved
    request.claim(catalog.states[1], stream=Stream([3, 7, 11, 9, 5], 8, draft=False))
    assert request.restore_handle.transfer.retired.wait(5)
    assert catalog.restored(request) is not None
    assert not request.source_active and request.target_handle.keys == ()
    catalog.release_owner(parent)
    assert catalog._ram.record()["payload_bytes"] == 0
    assert catalog.record()["sessions"] == 0 and catalog.record()["transient_requests"] == 1
    request.close()
    assert not catalog.has_retained
    catalog.close(timeout=5)


def test_pending_subruns_reuse_one_claim_without_extending_absolute_lifetime(materialized):
    import time

    catalog, future, _wake = owner(materialized)
    ref, handle = captured(catalog, materialized, future)
    original = handle.expires
    assert catalog.reserve_pending(ref, future_bytes=future) is handle
    assert catalog.replace_pending(handle) is None  # RAM target releases its idle GPU entitlement.
    assert handle.phase == "reserved" and handle.expires == original
    assert catalog.record()["future_backing_bytes"] == future
    stream = Stream([3, 7, 11, 9, 5], 7, draft=False, st=catalog.states[1])
    catalog.capture_idle(stream, handle=handle)
    handle.expires = time.monotonic() - 1
    catalog.expire((handle.serial,))
    assert handle.pending and handle.expired and catalog.has_retained
    assert catalog.reserve_pending(ref, future_bytes=future) is handle
    catalog.logical_reference_committed(ref.principal, ref.sequence, generation=1)
    catalog.progress()
    assert catalog.lookup(ref) is None and not catalog.has_retained
    catalog.close(timeout=5)


def test_successful_conversation_replaces_private_generation_only(materialized):
    catalog, future, _wake = owner(materialized)
    principal = "local-unauthenticated"
    first = LogicalReference(principal, "conversation:c", 1)
    old = catalog.reserve(first, future_bytes=future)
    catalog.associate(old, LogicalReference(principal, "response:old", 1))
    catalog.capture_idle(Stream([3, 7, 11, 9], 8, draft=False, st=materialized), handle=old)
    for sequence in ("conversation:c", "response:old"):
        catalog.logical_reference_committed(principal, sequence, generation=1)
    catalog.progress()
    second = LogicalReference(principal, "conversation:c", 2)
    new = catalog.reserve(second, future_bytes=future)
    catalog.capture_idle(Stream([3, 7, 11, 9, 5], 8, draft=False, st=catalog.states[1]), handle=new)
    catalog.logical_reference_committed(principal, second.sequence, generation=2)
    catalog.progress()
    assert catalog.lookup(first) is None and catalog.lookup(second) is new
    assert catalog.lookup(LogicalReference(principal, "response:old", 1)) is old
    catalog.release_owner(new)
    catalog.release_owner(old)
    catalog.close(timeout=5)


def test_retention_error_with_failed_reader_retirement_is_never_request_local(monkeypatch, materialized):
    from tensorfold.families.qwen4_exp.cuda import ram_backing
    from tensorfold.families.qwen4_exp.cuda.retained_catalog import RetentionCorrupt

    catalog, future, _wake = owner(materialized)
    _ref, handle = captured(catalog, materialized, future)
    save = catalog.suspend(handle)
    assert save.retired.wait(5) and catalog.saved(handle)
    primary = RetentionCorrupt("injected checksum failure with an unretired real reader")
    real_close = ram_backing.RamReader.close

    def verify(_reader):
        raise primary

    def close(_reader):
        raise RuntimeError("injected reader retirement failure")

    request = catalog.borrow(handle)
    with monkeypatch.context() as faults:
        faults.setattr(ram_backing.RamReader, "verify", verify)
        faults.setattr(ram_backing.RamReader, "close", close)
        request.claim(catalog.states[1], stream=Stream([3, 7, 11, 9], 8, draft=False))
        transfer = handle.transfer
        assert transfer.retired.wait(5)
    assert transfer.failure is primary and handle.phase == "restoring" and handle.transfer_readers == 1
    with pytest.raises(RuntimeError, match="unretired"):
        catalog.check_health()
    with pytest.raises(RuntimeError):
        catalog.retire_failed_destination(request, fenced=True)
    # Only the injected reader can now be positively closed. The poisoned
    # model/catalog authority remains retained; a real service contains its
    # process. This CPU fixture joins the already-quiescent callback only.
    for reader in primary.__dict__["_tensorfold_retained_owners"]:
        if isinstance(reader, ram_backing.RamReader):
            real_close(reader)
    catalog._worker.drain(catalog._thread, timeout=5, cancel_unentered=False)


def test_pending_reset_failure_after_old_refs_retire_is_latched_and_retained(monkeypatch, materialized):
    from tensorfold.families.qwen4_exp.cuda.execution_identity import _unretired

    catalog, future, _wake = owner(materialized)
    _ref, handle = captured(catalog, materialized, future)
    transfer = catalog.suspend(handle)
    assert transfer.retired.wait(5) and catalog.saved(handle)
    primary = MemoryError("reset publication failed after refs retired")

    def fail(_plan):
        raise primary

    with monkeypatch.context() as faults:
        faults.setattr(catalog, "_apply_release", fail)
        with pytest.raises(MemoryError) as caught:
            catalog.replace_pending(handle)
    assert caught.value is primary and _unretired(primary)
    with pytest.raises(RuntimeError, match="unretired"):
        catalog.check_health()
    assert catalog.has_retained and catalog._registration_owner is handle
    catalog._worker.drain(catalog._thread, timeout=5, cancel_unentered=False)


def test_failed_retained_ref_rollback_is_terminal_before_diagnostic_transport(monkeypatch, materialized):
    from tensorfold.families.qwen4_exp.cuda.retained_catalog import RetentionCorrupt

    catalog, future, _wake = owner(materialized)
    ref, parent = captured(catalog, materialized, future)
    save = catalog.suspend(parent)
    assert save.retired.wait(5) and catalog.saved(parent)
    target = catalog.reserve(LogicalReference(ref.principal, "response:rollback-fault", 1), future_bytes=future)
    request = catalog.borrow(parent, continuation="new-turn", target_handle=target)
    request.claim(catalog.states[1], stream=Stream([3, 7, 11, 9, 5], 8, draft=False))
    transfer = target.transfer
    assert transfer.retired.wait(5)
    primary = RetentionCorrupt("injected ref acquisition error")
    real_retain = catalog._ram.retain_into

    def acquire(key, journal, index):
        real_retain(key, journal, index)
        raise primary

    def release(_key):
        raise RuntimeError("ref rollback failed")

    with monkeypatch.context() as faults:
        faults.setattr(catalog._ram, "retain_into", acquire)
        faults.setattr(catalog._ram, "release", release)
        with pytest.raises(RetentionCorrupt) as caught:
            catalog.restored(request)
    assert caught.value is primary and target.transfer is transfer
    assert any(key is not None for key in transfer.retained_keys)
    with pytest.raises(RuntimeError, match="unretired"):
        catalog.check_health()
    catalog._worker.drain(catalog._thread, timeout=5, cancel_unentered=False)


def test_failure_after_complete_ref_journal_never_retains_again_on_retry(monkeypatch, materialized):
    from tensorfold.families.qwen4_exp.cuda.retained_catalog import RetainedHandle

    catalog, future, _wake = owner(materialized)
    ref, parent = captured(catalog, materialized, future)
    save = catalog.suspend(parent)
    assert save.retired.wait(5) and catalog.saved(parent)
    target = catalog.reserve(LogicalReference(ref.principal, "response:publish-fault", 1), future_bytes=future)
    request = catalog.borrow(parent, continuation="new-turn", target_handle=target)
    request.claim(catalog.states[1], stream=Stream([3, 7, 11, 9, 5], 8, draft=False))
    transfer = target.transfer
    assert transfer.retired.wait(5)
    original = RetainedHandle.__setattr__
    primary = MemoryError("after-ref publication failure")

    def assign(owner, name, value):
        if owner is target and name == "ready_handoff":
            raise primary
        original(owner, name, value)

    with monkeypatch.context() as faults:
        faults.setattr(RetainedHandle, "__setattr__", assign)
        with pytest.raises(MemoryError) as caught:
            catalog.restored(request)
    assert caught.value is primary and target.transfer is transfer
    before = [catalog._ram._objects[key].references for key in parent.keys]
    with pytest.raises(RuntimeError, match="unretired"):
        catalog.restored(request)
    assert before == [catalog._ram._objects[key].references for key in parent.keys]
    catalog._worker.drain(catalog._thread, timeout=5, cancel_unentered=False)


def test_public_catalog_health_observes_store_poison_before_worker_classification(materialized):
    from tensorfold.families.qwen4_exp.cuda.retained_catalog import RetentionCorrupt

    catalog, future, _wake = owner(materialized)
    builder = catalog._ram.begin("a" * 64, 4)
    builder.write(b"abcd")
    key = builder.finish()
    reader = catalog._ram.acquire(key)
    primary = RetentionCorrupt("durable store poison before catalog error transport")
    with pytest.raises(RetentionCorrupt):
        catalog._ram._poison(primary, reader)
    assert catalog._failure is None
    with pytest.raises(RuntimeError, match="poisoned"):
        catalog.check_health()
    with pytest.raises(RuntimeError, match="poisoned"):
        catalog.reserve(LogicalReference("p", "response:late", 1), future_bytes=future)
    # Continue the real classifier only after observing the deterministic
    # interval; unknown ownership remains retained, callback alone is joined.
    assert catalog._clean_retention_error(primary) is False
    catalog._latch_failure(primary)  # The existing _serve outcome for a nonclean error.
    catalog._worker.drain(catalog._thread, timeout=5, cancel_unentered=False)

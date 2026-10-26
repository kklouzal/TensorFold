# ruff: noqa: E402 - optional Torch precedes the actual State/catalog controls.
"""Project Store bounds and generation exhaustion before transfer publication."""

from types import SimpleNamespace

import pytest
pytest.importorskip("torch")

from tensorfold.cuda.store_fields import MAX_KEY_BYTES, validate_store_key
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.retained_catalog import (
    LogicalReference, RetainedCatalog, RetentionCapacity, RetentionError,
)
from test_retained_catalog import captured, owner

pytest_plugins = ("test_state_retention",)
MAX_GENERATION = (1 << 63) - 1


@pytest.mark.parametrize("key", ["a" * (MAX_KEY_BYTES - 1), "é" * 4095 + "a", "😀" * 2047 + "abc"])
def test_project_key_bound_counts_utf8_and_the_native_prefix(key):
    validate_store_key(key)


@pytest.mark.parametrize("key", ["a" * MAX_KEY_BYTES, "é" * 4096, "😀" * 2048, "a" * 100000])
def test_project_key_bound_refuses_before_native_io(key):
    with pytest.raises(ValueError, match="key field"):
        validate_store_key(key)


def test_catalog_enforces_project_keys_without_a_native_backend_cap(materialized):
    catalog, _future, _wake = owner(materialized)
    try:
        assert catalog._native_key_limit is None
        catalog._store_payload("a" * (MAX_KEY_BYTES - 1), b"bounded")
        with pytest.raises(RetentionError, match="key field"):
            catalog._store_payload("a" * MAX_KEY_BYTES, b"bounded")
    finally:
        catalog.close(timeout=5)


@pytest.mark.parametrize("unchanged_gpu", [False, True])
def test_last_save_generation_is_usable_then_restore_refuses_without_reusing_it(materialized, unchanged_gpu, monkeypatch):
    catalog, future, _wake = owner(materialized)
    request = None
    _reference, handle = captured(catalog, materialized, future)
    try:
        handle.transfer_generation = MAX_GENERATION - (3 if unchanged_gpu else 1)
        transfer = catalog.suspend(handle)
        assert transfer.retired.wait(5) and catalog.saved(handle)
        if unchanged_gpu:
            request = catalog.borrow(handle, continuation="same-request")
            assert request.claim(catalog.states[1], stream=Stream([3, 7, 11, 9], 8, draft=False)) is None
            assert handle.transfer.retired.wait(5)
            assert catalog.restored(request) is not None
            request.close()
            request = None
            assert handle.inventory is None  # Reuse unchanged RAM authority.
            transfer = catalog.suspend(handle)
            assert transfer.retired.wait(5) and catalog.saved(handle)
        assert transfer.generation == handle.transfer_generation == MAX_GENERATION
        request = catalog.borrow(handle, continuation="same-request")
        before = (handle.phase, handle.state, handle.manifest, handle.future_bytes, catalog._reserved)
        import tensorfold.families.qwen4_exp.cuda.retained_catalog as source
        monkeypatch.setattr(source, "state_restore_plan", lambda *_a, **_k:
            pytest.fail("exhausted generation allocated a restore plan"))
        with pytest.raises(RetentionCapacity, match="transfer generation exhausted"):
            request.claim(catalog.states[0], stream=Stream([3, 7, 11, 9], 8, draft=False))
        assert handle.transfer is None and not request.claimed
        assert handle.transfer_generation == MAX_GENERATION
        assert (handle.phase, handle.state, handle.manifest, handle.future_bytes, catalog._reserved) == before
        assert not catalog._pending
    finally:
        if request is not None:
            request.close()
        catalog.release_owner(handle)
        catalog.close(timeout=5)


def test_suspension_exhaustion_keeps_its_gpu_authority(materialized):
    catalog, future, _wake = owner(materialized)
    _reference, handle = captured(catalog, materialized, future)
    try:
        handle.transfer_generation = MAX_GENERATION
        inventory = handle.inventory
        with pytest.raises(RetentionCapacity, match="transfer generation exhausted"):
            catalog.suspend(handle)
        assert handle.inventory is inventory and handle.state is materialized
        assert handle.phase == "gpu" and handle.transfer is None
        assert handle.transfer_generation == MAX_GENERATION and not catalog._pending
    finally:
        catalog.release_owner(handle)
        catalog.close(timeout=5)


def test_new_turn_fork_exhaustion_preserves_source_and_destination(materialized):
    catalog, future, _wake = owner(materialized)
    request = None
    target = None
    _reference, handle = captured(catalog, materialized, future)
    try:
        target = catalog.reserve(LogicalReference("local-unauthenticated", "response:next", 1), future_bytes=future)
        request = catalog.borrow(handle, continuation="new-turn", target_handle=target)
        target.transfer_generation = MAX_GENERATION
        destination = catalog.states[1]
        position = destination.pos
        with pytest.raises(RetentionCapacity, match="transfer generation exhausted"):
            request.claim(destination, stream=Stream([3, 7, 11, 9, 4], 8, draft=False))
        assert target.transfer_generation == MAX_GENERATION and target.transfer is None
        assert not request.claimed and destination.pos == position
        assert handle.state is materialized and handle.phase == "gpu"
        assert not catalog._pending
    finally:
        if request is not None:
            request.close()
        if target is not None:
            catalog.release_owner(target)
        catalog.release_owner(handle)
        catalog.close(timeout=5)


def test_settlement_exhaustion_precedes_any_cohort_phase(monkeypatch):
    import tensorfold.families.qwen4_exp.cuda.retained_catalog as source
    # Only the pre-publication decision is projected here. No fake collective
    # or GPU outcome is asserted; any attempted phase is an independent failure.
    probe = SimpleNamespace(world=8, _settlement_generation=MAX_GENERATION, check_health=lambda: None)
    monkeypatch.setattr(source, "_phase", lambda *_a, **_k: pytest.fail("exhausted generation reached Store IO"))
    with pytest.raises(RetentionCapacity, match="settlement generation exhausted"):
        RetainedCatalog.settle_ready(probe)
    assert probe._settlement_generation == MAX_GENERATION

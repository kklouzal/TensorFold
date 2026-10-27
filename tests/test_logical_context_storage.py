# ruff: noqa: E402 - optional Torch is checked before CUDA-family imports.
"""Logical positional lifetime is independent of physical speculative cache rows."""
from types import SimpleNamespace as NS

import pytest
torch = pytest.importorskip("torch")

from tensorfold.families.qwen4_exp.cuda.state import State
from tensorfold.families.qwen4_exp.cuda.decode import Engine as DecodeEngine
from tensorfold.families.qwen4_exp.cuda.state_retention import state_inventory, state_restore_plan
from tests.test_sm70_rotorquant_state import model


def weights(domain=128, *, mtp=True):
    w = model(8, mtp=mtp)
    w.layers = [NS(index=index, linear=False) for index in range(4)]
    w.meta.update(full=NS(kv_heads=2), rank=0, execution_provider="sm70", history_layout="unique",
                  coordinate_profile_identity="a" * 64)
    def role(name):
        return NS(domain=NS(max_rows=domain), role=name)
    w.positions = NS(target=role("target"), mtp=role("mtp") if mtp else None,
                     profile=NS(identity="b" * 64, epoch=0))
    return w


@pytest.mark.parametrize("logical", [128, 262144])
def test_full_window_keeps_all_physical_guards_with_exact_logical_domain(logical):
    w = weights(logical)
    st = State(w, logical + 3, 3, "fp16", limit=logical + 3, logical_context=logical)
    assert st.capacity == st.limit == logical + 3
    assert st.logical_context == logical
    assert st.media_target_position.domain.max_rows == st.media_mtp_position.domain.max_rows == logical
    assert st.cache_bytes(logical + 3) == st.cache_bytes()
    assert st.cache_bytes(logical + 3) > st.cache_bytes(logical)
    with pytest.raises(ValueError, match="logical"):
        st.set_pos(logical + 1)
    with pytest.raises(ValueError, match="logical"):
        st.set_mtp_len(logical + 1)


@pytest.mark.parametrize("logical", [129, 132, 0, True])
def test_bad_logical_allowance_refuses_before_device_allocation(monkeypatch, logical):
    monkeypatch.setattr(torch, "zeros", lambda *_a, **_k: pytest.fail("allocation preceded logical admission"))
    with pytest.raises(ValueError, match="logical|positional"):
        State(weights(), 131, 3, "fp16", limit=131, logical_context=logical)


def test_clone_and_snapshot_restore_preserve_both_limits_and_refuse_guard_frontiers():
    w = weights()
    st = State(w, 131, 3, "fp16", limit=131, logical_context=128)
    cloned = st.clone()
    assert (cloned.logical_context, cloned.capacity, cloned.limit) == (128, 131, 131)
    assert cloned.kv_identity == st.kv_identity
    snap = st.snapshot()
    for key in ("pos", "mtp_len"):
        bad = dict(snap, **{key: 129})
        with pytest.raises(ValueError, match="snapshot positions"):
            st.restore(bad)
    with pytest.raises(ValueError, match="logical"):
        st.indexer_snapshot_metadata(129, 0)
    with pytest.raises(ValueError, match="prefix copy"):
        st.copy_prefix(cloned, 129, 0)
    other = State(w, 131, 3, "fp16", limit=131, logical_context=127)
    with pytest.raises(ValueError, match="matching cache formats"):
        other.copy_prefix(st, 0, 0)


def test_physical_resize_and_retirement_preserve_logical_context():
    w = model(1)
    st = State(w, 16, 3, "fp16", limit=131, logical_context=128)
    st.resize(131)
    assert (st.logical_context, st.capacity, st.limit) == (128, 131, 131)
    with pytest.raises(ValueError, match="physical"):
        st.resize(132)
    st._resize_error = RuntimeError("controlled retired cache")
    st.recover_resize(w, 16)
    assert (st.logical_context, st.capacity, st.limit) == (128, 16, 131)


def test_serial_twin_keeps_logical_context_separate_from_storage(monkeypatch):
    from tests.test_dspark_taps_forward import cpu_weights
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda *_a: NS(multi_processor_count=80))
    w = cpu_weights(monkeypatch)
    decoder = DecodeEngine(w, capacity=131, max_rows=3, prefill_rows=4, kv_dtype="fp16", logical_context=128)
    twin = decoder.twin()
    assert (twin.capacity, twin.logical_context, twin.st.limit, twin.st.logical_context) == (131, 128, 131, 128)
    assert twin.st.kv_identity == decoder.st.kv_identity


def test_canonical_positions_and_retention_cannot_promote_guard_rows():
    w = model(1)
    st = State(w, 131, 3, "fp16", limit=131, logical_context=128)
    st.set_pos(128)
    st.set_mtp_len(128)
    assert st.pos == st.mtp_len == st.logical_context
    st.reset(w)
    inventory = state_inventory(st, recovery_versions=0)
    for changes in ({"pos": 129}, {"mtp_len": 129, "mtp_drafted": 0},
                    {"mtp_len": 129, "mtp_drafted": 1}):
        metadata = dict(inventory.metadata, **changes)
        with pytest.raises(ValueError, match="frontier"):
            state_restore_plan(st, ((metadata, inventory.specs),), continuation="same-request", recovery_versions=0)
    st.mtp_len, st.mtp_drafted = 129, 1
    with pytest.raises(ValueError, match="frontier"):
        state_inventory(st, recovery_versions=0)


@pytest.mark.parametrize("extra_storage", [3, 4, 9])
def test_engine_uses_admitted_logical_window_without_inventing_fixed_guard_count(extra_storage):
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
    owner = object.__new__(FlashNextEngine)
    owner.max_len, owner.depth = 262144 + extra_storage, 2
    owner.capacity_plan = {"context_window": 262144, "cache_slots": owner.max_len}
    assert owner.context_window == 262144
    assert owner._limit([0] * 262143, 3) == 1
    with pytest.raises(ValueError, match="no room"):
        owner._limit([0] * 262144, 1)


@pytest.mark.parametrize("mtp_domain", [None, 127])
def test_mtp_logical_domain_is_checked_independently_before_allocation(monkeypatch, mtp_domain):
    w = weights()
    w.positions.mtp = None if mtp_domain is None else NS(domain=NS(max_rows=mtp_domain))
    monkeypatch.setattr(torch, "zeros", lambda *_a, **_k: pytest.fail("allocation preceded MTP admission"))
    with pytest.raises(ValueError, match="positional"):
        State(w, 131, 3, "fp16", limit=131, logical_context=128)

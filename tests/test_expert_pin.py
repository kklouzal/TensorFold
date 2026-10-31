"""Host pin admission and interrupted/failed registration ownership."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.cuda import expert_pin as pin


class Native:
    memory_allocation, not_supported = 2, 801

    def __init__(self, status=0, interrupted=False, close_status=0):
        self.status, self.interrupted, self.close_status = status, interrupted, close_status
        self.owners = []

    def Registration(self, pages):
        owner = SimpleNamespace(pages=pages, active=False)

        def register():
            owner.active = self.status == 0
            if self.interrupted:
                raise KeyboardInterrupt
            return self.status

        def close():
            if not self.close_status:
                owner.active = False
            return self.close_status

        owner.pin, owner.close = register, close
        owner.contains = lambda t: owner.active and owner.pages.data_ptr() <= t.data_ptr() and \
            t.data_ptr() + t.nbytes <= owner.pages.data_ptr() + owner.pages.nbytes
        self.owners.append(owner)
        return owner

    @staticmethod
    def error_text(code):
        return f"controlled status {code}"


def ready(monkeypatch, native=None, *, host_room=1 << 30, cgroup_room=None):
    monkeypatch.setattr(pin, "_host", lambda path: {"room_bytes": host_room, "reserve_bytes": 0})
    monkeypatch.setattr(pin, "host_cgroup_bytes", lambda: cgroup_room)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    native = Native() if native is None else native
    monkeypatch.setattr(pin, "_native", lambda: native)
    return native


def test_disabled_policy_does_not_probe_or_register_and_keeps_ordinary_storage(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("disabled pin path observed optional resources")

    monkeypatch.setattr(pin, "_host", unexpected)
    monkeypatch.setattr(pin, "host_cgroup_bytes", unexpected)
    policy = pin.ExpertPinPolicy(8)
    with torch.inference_mode():
        result = policy.empty((4,), dtype=torch.int16)
    assert not torch.is_inference(result) and not result.is_pinned()
    assert policy.receipt()["allocated_bytes"] == 8
    with pytest.raises(ValueError, match="budget"):
        policy.empty((1,), dtype=torch.int16)
    policy.close()
    policy.close()


@pytest.mark.parametrize("kwargs", [{"mode": "yes"}, {"planned_bytes": -1}, {"future_bytes": True},
                                     {"allocation_bytes": (7,)}, {"allocation_bytes": (8, 0)}])
def test_invalid_plan_fails_before_resource_observation(kwargs):
    with pytest.raises(ValueError):
        pin.ExpertPinPolicy(**{"planned_bytes": 8, **kwargs})


def test_full_model_backing_future_and_limiting_cgroup_are_charged_once(monkeypatch):
    footprint = pin.backing_bytes(8) + pin.backing_bytes(16)
    ready(monkeypatch, host_room=footprint + 32, cgroup_room=footprint + 31)
    policy = pin.ExpertPinPolicy(24, mode="auto", allocation_bytes=(8, 16), future_bytes=32)
    assert not policy.admission.admitted
    assert policy.admission.reason == "insufficient_host_or_cgroup_room"
    assert policy.admission.backing_bytes == footprint
    assert not policy.admission.snapshot_is_reservation
    with pytest.raises(pin.PinUnavailable, match="required"):
        pin.ExpertPinPolicy(24, mode="on", allocation_bytes=(8, 16), future_bytes=32)


def test_missing_geometry_refuses_only_optional_pin_capability(monkeypatch):
    ready(monkeypatch)
    policy = pin.ExpertPinPolicy(8, mode="auto")
    assert policy.admission.reason == "allocation_geometry_unavailable"
    assert not policy.empty((8,), dtype=torch.uint8).is_pinned()
    policy.close()


def test_source_geometry_and_page_bound_are_validated_before_registration(monkeypatch):
    native = ready(monkeypatch)
    policy = pin.ExpertPinPolicy(24, mode="on", allocation_bytes=(8, 16))
    with pytest.raises(ValueError, match="geometry"):
        policy.empty((3,), dtype=torch.int32)
    result = policy.empty((2,), dtype=torch.int32)
    result.copy_(torch.tensor([17, -3], dtype=torch.int32))
    owner = native.owners[0]
    assert owner.active and owner.pages.data_ptr() % pin.PAGESIZE == 0
    assert owner.pages.numel() % pin.PAGESIZE == 0
    assert result.data_ptr() == owner.pages.data_ptr()
    assert owner.pages.untyped_storage().nbytes() <= pin.backing_bytes(8)
    assert policy.receipt()["pinned_bytes"] == 8
    assert policy.owns(result) and policy.owns(result[1:])
    assert not policy.owns(torch.empty_like(result))
    policy.close()
    assert not owner.active and result.tolist() == [17, -3]


@pytest.mark.parametrize("code", [2, 801])
def test_explicit_auto_refusal_retains_same_pageable_allocation(monkeypatch, code):
    native = ready(monkeypatch, Native(status=code))
    policy = pin.ExpertPinPolicy(24, mode="auto", allocation_bytes=(8, 16))
    result = policy.empty((2,), dtype=torch.int32)
    assert result.data_ptr() == native.owners[0].pages.data_ptr()
    assert not native.owners[0].active
    other = policy.empty((4,), dtype=torch.int32)
    assert other.shape == (4,) and len(native.owners) == 1
    assert policy.receipt()["registration_refusals"] == 1
    assert not policy.receipt()["pin_enabled"]
    policy.close()


def test_unknown_registration_error_never_becomes_pageable_fallback(monkeypatch):
    ready(monkeypatch, Native(status=700))
    policy = pin.ExpertPinPolicy(8, mode="auto", allocation_bytes=(8,))
    with pytest.raises(pin.PinUnavailable, match="controlled status 700"):
        policy.empty((2,), dtype=torch.int32)
    with pytest.raises(RuntimeError, match="failed"):
        policy.empty((2,), dtype=torch.int32)
    policy.close()


def test_interrupted_successful_registration_stays_owned_for_rollback(monkeypatch):
    native = ready(monkeypatch, Native(interrupted=True))
    policy = pin.ExpertPinPolicy(8, mode="on", allocation_bytes=(8,))
    with pytest.raises(KeyboardInterrupt):
        policy.empty((2,), dtype=torch.int32)
    assert native.owners[0].active
    with pytest.raises(RuntimeError, match="failed"):
        policy.empty((2,), dtype=torch.int32)
    policy.close()
    assert not native.owners[0].active


def test_post_pin_bookkeeping_interruption_latches_failure_before_reuse(monkeypatch):
    native = ready(monkeypatch)
    policy = pin.ExpertPinPolicy(8, mode="on", allocation_bytes=(8,))
    primary = KeyboardInterrupt()

    class InterruptedCounter(dict):
        def __setitem__(self, key, value):
            raise primary

    policy._remaining = InterruptedCounter(policy._remaining)
    with pytest.raises(KeyboardInterrupt) as failure:
        policy.empty((2,), dtype=torch.int32)
    assert failure.value is primary and native.owners[0].active
    with pytest.raises(RuntimeError, match="failed"):
        policy.empty((2,), dtype=torch.int32)
    assert len(native.owners) == 1
    policy.close()
    assert not native.owners[0].active


def test_close_failure_retains_registration_owner_in_owning_error(monkeypatch):
    native = ready(monkeypatch, Native(close_status=700))
    policy = pin.ExpertPinPolicy(8, mode="on", allocation_bytes=(8,))
    policy.empty((2,), dtype=torch.int32)
    with pytest.raises(pin.PinCleanupError) as failure:
        policy.close()
    assert failure.value.owner is policy and native.owners[0].active
    assert policy._pins and not policy.receipt()["closed"]


def test_close_exception_preserves_primary_and_retires_independent_owners(monkeypatch):
    native = ready(monkeypatch)
    policy = pin.ExpertPinPolicy(24, mode="on", allocation_bytes=(8, 16))
    policy.empty((2,), dtype=torch.int32)
    policy.empty((4,), dtype=torch.int32)
    primary = KeyboardInterrupt()

    def interrupted():
        raise primary

    native.owners[1].close = interrupted
    with pytest.raises(KeyboardInterrupt) as failure:
        policy.close()
    assert failure.value is primary
    assert not native.owners[0].active and native.owners[1].active
    assert isinstance(primary.__cause__, pin.PinCleanupError)
    assert primary.__cause__.owner is policy
    assert not policy.receipt()["closed"]


def test_close_diagnostic_exhaustion_preserves_primary_after_every_native_retirement(monkeypatch):
    native = ready(monkeypatch)
    policy = pin.ExpertPinPolicy(24, mode="on", allocation_bytes=(8, 8, 8))
    for _ in range(3):
        policy.empty((2,), dtype=torch.int32)
    primary = KeyboardInterrupt("controlled independent close interruption")
    exhausted = MemoryError("controlled cleanup diagnostic exhaustion")
    closed = []

    def close(index):
        closed.append(index)
        if index == 2:
            raise primary
        if index == 1:
            return 700
        native.owners[index].active = False
        return 0

    for index, owner in enumerate(native.owners):
        owner.close = lambda index=index: close(index)

    def no_diagnostic(*args):
        raise exhausted

    monkeypatch.setattr(pin, "PinCleanupError", no_diagnostic)
    with pytest.raises(KeyboardInterrupt) as failure:
        policy.close()
    assert failure.value is primary and primary.__cause__ is exhausted
    assert closed == [2, 1, 0]
    assert policy._close_errors[2] is primary and policy._close_codes[1] == 700
    assert native.owners[2].active and native.owners[1].active and not native.owners[0].active
    assert policy._pins and not policy.receipt()["closed"]
    with pytest.raises(RuntimeError, match="failed"):
        policy.empty((0,), dtype=torch.int32)

"""Startup budget provenance from actual API samples; no physical GPU claim."""

from types import SimpleNamespace as NS
import struct

import pytest

from tensorfold.cuda import capacity
from test_adaptive_session_capacity import admission as _admission

GIB = 2**30
admission = _admission  # Register the existing fixture without another wrapper or implementation.


class Device:
    def __init__(self, free=20 * GIB):
        self.samples = 0
        self.free = free

    def mem_get_info(self):
        self.samples += 1
        assert self.samples == 1, "the budget resampled physical availability"
        return self.free, 32 * GIB

    def memory_allocated(self):
        return 5 * GIB

    def memory_reserved(self):
        return 7 * GIB


@pytest.fixture
def device(monkeypatch):
    monkeypatch.setattr(capacity, "unified", lambda _torch: False)
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_RESERVE_GIB", "4")
    monkeypatch.delenv(capacity.LIMIT_ENV, raising=False)
    return Device()


@pytest.mark.parametrize("cap,expected", [(None, 16), ("16", 12), ("10", 6), ("1e300", 16)])
def test_one_sample_and_remaining_absolute_grant_do_not_recharge_physical_free(device, monkeypatch, cap, expected):
    if cap is not None:
        monkeypatch.setenv(capacity.LIMIT_ENV, cap)
    result = capacity.device_allocation_baseline(NS(cuda=device), preallocated_device_bytes=4 * GIB)
    assert device.samples == 1 and result["physical_free_bytes"] == 20 * GIB
    assert result["usable_serving_budget_bytes"] == expected * GIB
    assert result["allocator_allocated_bytes"] == 5 * GIB
    assert result["allocator_reserved_bytes"] == 7 * GIB  # Never add allocated to reserved.
    assert result["preallocated_project_bytes"] == 4 * GIB
    assert result["observed_at_unix_ns"] > 0
    assert 0 <= result["interval_monotonic_ns"][0] <= result["interval_monotonic_ns"][1]


def test_allocator_unavailability_is_explicit_and_never_fabricated_as_zero(device, monkeypatch):
    def unavailable():
        raise RuntimeError("counter unavailable")
    monkeypatch.setattr(device, "memory_allocated", unavailable)
    result = capacity.device_allocation_baseline(NS(cuda=device))
    assert result["allocator_allocated_bytes"] is None
    assert result["allocator_observation_status"]["memory_allocated"] == "unavailable:RuntimeError"
    assert result["allocator_reserved_bytes"] == 7 * GIB and device.samples == 1


def test_counter_exhaustion_is_not_swallowed_as_an_unknown_observation(device, monkeypatch):
    primary = MemoryError("counter allocation failed")
    def exhausted():
        raise primary
    monkeypatch.setattr(device, "memory_allocated", exhausted)
    with pytest.raises(MemoryError) as caught:
        capacity.device_allocation_baseline(NS(cuda=device))
    assert caught.value is primary and device.samples == 0


def test_selected_baseline_floor_does_not_change_legacy_live_budget_api(device, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_RESERVE_GIB", "3")
    with pytest.raises(ValueError, match="4 GiB"):
        capacity.device_allocation_baseline(NS(cuda=device))
    legacy = Device()
    assert capacity.available_bytes(NS(cuda=legacy)) == 17 * GIB


@pytest.mark.parametrize("cap", [None, "16", "1e300"])
def test_admission_and_rank_receipt_use_the_same_baseline_and_preserve_large_caps(admission, device, monkeypatch, cap):
    if cap is not None:
        monkeypatch.setenv(capacity.LIMIT_ENV, cap)
    monkeypatch.setattr(capacity, "available_bytes", lambda *_a, **_kw: pytest.fail("startup budget resampled"))
    result = capacity.admit(**{**admission, "torch": NS(cuda=device)}, session_ceiling=18,
                            preallocated_device_bytes=4 * GIB, capture_device_baseline=True)
    assert device.samples == 1 and result["context_window"] == 100
    local = result["allocation_baseline"]
    rank = result["session_capacity"]["ranks"][0]
    assert rank["budget_bytes"] == local["usable_serving_budget_bytes"] == result["budget_bytes"]
    for key in ("physical_free_bytes", "absolute_grant_cap_bytes", "remaining_absolute_grant_bytes",
                "preallocated_project_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes"):
        assert rank["allocation_baseline"][key] == local[key]


def test_mixed_baseline_modes_refuse_before_variable_length_rank_exchange(admission, device):
    calls = []
    def gather(packet):
        calls.append(packet)
        assert len(packet) == 8
        peer = list(packet)
        peer[7] = 18  # Same ceiling, legacy/no-capture mode.
        return [packet, peer]
    with pytest.raises(ValueError, match="baseline modes"):
        capacity.admit(**{**admission, "torch": NS(cuda=device)}, world=2, gather=gather,
                       session_ceiling=18, capture_device_baseline=True)
    assert len(calls) == 1


def test_distinct_rank_facts_stay_bound_to_each_budget(admission, device, monkeypatch):
    monkeypatch.setenv(capacity.LIMIT_ENV, "16")
    def gather(packet):
        if len(packet) == 8:
            return [packet, list(packet)]
        # The existing header ledger precedes the versioned startup facts.
        assert len(packet) == 16
        cap_word = struct.unpack("<q", struct.pack("<d", 16.0))[0]
        peer = [1, 12 * GIB, 100, 50, 2000, 2100, 1,
                packet[7] + 2, 18 * GIB, 32 * GIB, 4 * GIB, 4 * GIB, 12 * GIB,
                6 * GIB, 8 * GIB, cap_word]
        return [packet, peer]
    result = capacity.admit(**{**admission, "torch": NS(cuda=device)}, world=2, gather=gather,
                            session_ceiling=18, preallocated_device_bytes=4 * GIB,
                            capture_device_baseline=True)
    ranks = result["session_capacity"]["ranks"]
    assert [row["allocation_baseline"]["physical_free_bytes"] for row in ranks] == [20 * GIB, 18 * GIB]
    assert [row["allocation_baseline"]["allocator_reserved_bytes"] for row in ranks] == [7 * GIB, 8 * GIB]
    assert all(row["budget_bytes"] == row["allocation_baseline"]["usable_serving_budget_bytes"] == 12 * GIB
               for row in ranks)

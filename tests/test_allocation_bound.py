"""Native allocator capability/identity/rounding boundaries, without CUDA work."""

import gc
import json
import weakref
from copy import deepcopy
from fractions import Fraction
from types import SimpleNamespace

import pytest

from tensorfold.cuda.allocation_bound import UnsupportedAllocator, pool_allocation_bound

MIB = 1 << 20


class Bootstrap:
    def __init__(self, amount, pointer):
        self.amount, self.pointer = amount, pointer
        self.device = SimpleNamespace(type="cuda", index=0)
        self.offset, self.contiguous = 0, True

    def is_contiguous(self):
        return self.contiguous

    def storage_offset(self):
        return self.offset

    def untyped_storage(self):
        return SimpleNamespace(nbytes=lambda: self.amount, data_ptr=lambda: self.pointer)

    def data_ptr(self):
        return self.pointer

    def numel(self):
        return self.amount

    def element_size(self):
        return 1


def scenario(*, expandable=True, quantum=32 * MIB, amount=3 * MIB // 2, divisions=None):
    pointer = 0xAAAABBBBCCCCDDDD
    divisions = [0] * 16 if divisions is None else divisions
    settings = {"expandable_segments": expandable,
                "PYTORCH_CUDA_ALLOC_CONF": f"expandable_segments:{expandable},large_segment_size_mb:{quantum // MIB}",
                "roundup_power2_divisions": {str(1 << index): value for index, value in enumerate(divisions)}}
    # The common fixture is deliberately512-aligned and exactly a power-of-two
    # division boundary, so its expected block is independent of production rounding.
    after = {"allocator_settings": deepcopy(settings), "segments": [{
        "device": 0, "address": pointer, "total_size": quantum, "segment_type": "large",
        "segment_pool_id": (0, 0), "is_expandable": expandable,
        "allocated_size": amount, "active_size": amount,
        "blocks": [{"address": pointer, "size": amount, "requested_size": amount, "state": "active_allocated"},
                   {"address": pointer + amount, "size": quantum - amount, "state": "inactive"}]}]}
    before = {"allocator_settings": deepcopy(settings), "segments": []}
    state = {"backend": "native", "after": after}
    torch = SimpleNamespace(__version__="probe-supported-version", version=SimpleNamespace(git_version="observed-git"),
                            cuda=SimpleNamespace(memory=SimpleNamespace(
                                get_allocator_backend=lambda: state["backend"],
                                _snapshot=lambda: deepcopy(state["after"]))))
    return torch, Bootstrap(amount, pointer), before, state


def make(**options):
    torch, bootstrap, before, state = scenario(**options)
    return pool_allocation_bound(torch, bootstrap, before_snapshot=before), state


@pytest.mark.parametrize("expandable", [False, True])
def test_observed_configured_segment_and_actual_runtime_are_recorded(expandable):
    bound, _ = make(expandable=expandable, quantum=48 * MIB)
    assert bound.large_quantum == 48 * MIB
    assert bound(0) == 0 and bound(1) == 2 * MIB
    assert bound(2 * MIB) == 48 * MIB
    assert bound(49 * MIB) == (96 * MIB if expandable else 50 * MIB)
    receipt = bound.receipt
    assert receipt["torch_version"] == "probe-supported-version"
    assert receipt["torch_git"] == "observed-git"
    assert receipt["large_segment_bytes"] == 48 * MIB
    assert "address" not in json.dumps(receipt) and str(0xAAAABBBBCCCCDDDD) not in json.dumps(receipt)
    receipt["allocator_settings"]["expandable_segments"] = "mutated"
    bound.validate_current()


def test_nonexpandable_bound_is_monotone_across_native_medium_to_large_drop():
    bound, _ = make(expandable=False)
    values = [bound(size) for size in (MIB, MIB + 1, 9 * MIB, 10 * MIB, 20 * MIB, 32 * MIB, 32 * MIB + 1)]
    assert values == sorted(values)
    assert bound(10 * MIB) == 32 * MIB  # actual native fresh request10MiB; monotone envelope keeps observed32MiB.


def independent_rounded(amount, divisions):
    if amount < 512:
        return 512
    bucket = min(max(amount.bit_length() - 1 - 20, 0), 15)
    count = divisions[bucket]
    if count <= 1 or amount <= 512 * count:
        return ((amount - 1) // 512 + 1) * 512
    lower = 2 ** (amount.bit_length() - 1)
    if lower == amount:
        return lower
    # Enumerate the official interval grid as exact rationals, independent of
    # the production bit-shift/mask formulation.
    return int(next(Fraction(lower) + Fraction(lower * step, count)
                    for step in range(count + 1)
                    if Fraction(lower) + Fraction(lower * step, count) >= amount))


@pytest.mark.parametrize("expandable", [False, True])
def test_all_power2_buckets_obey_native_rounding_and_reservation(expandable):
    divisions = [2, 4, 8, 16] * 4
    bound, _ = make(expandable=expandable, divisions=divisions)
    amounts = {1, 511, 512, 513, MIB - 1, MIB, 10 * MIB - 1, 10 * MIB}
    for exponent in range(20, 39):
        lower = 1 << exponent
        amounts.update((lower, lower + 1, lower + lower // 3, 2 * lower - 1))
    previous = 0
    for amount in sorted(amounts):
        rounded = independent_rounded(amount, divisions)
        actual = (2 * MIB if rounded <= MIB else
                  -(-rounded // bound.large_quantum) * bound.large_quantum if expandable else
                  bound.large_quantum if rounded < 10 * MIB else -(-rounded // (2 * MIB)) * (2 * MIB))
        assert bound(amount) >= actual >= amount
        assert bound(amount) >= previous
        previous = bound(amount)


def test_other_devices_and_host_segments_do_not_invalidate_device_freshness():
    torch, bootstrap, before, state = scenario()
    unrelated = [{"device": -1}, {"device": 1}]
    before["segments"] += deepcopy(unrelated)
    state["after"]["segments"] += deepcopy(unrelated)
    pool_allocation_bound(torch, bootstrap, before_snapshot=before)


def test_bound_retains_no_bootstrap_tensor_or_snapshots():
    torch, bootstrap, before, _ = scenario()
    reference = weakref.ref(bootstrap)
    bound = pool_allocation_bound(torch, bootstrap, before_snapshot=before)
    del bootstrap, before
    gc.collect()
    assert reference() is None
    assert bound(2 * MIB) == 32 * MIB


@pytest.mark.parametrize("change", ["backend", "expandable", "configuration", "divisions"])
def test_validation_refuses_changed_allocator_configuration(change):
    bound, state = make()
    if change == "backend":
        state["backend"] = "cudaMallocAsync"
    elif change == "divisions":
        state["after"]["allocator_settings"]["roundup_power2_divisions"]["1"] = 2
    else:
        field = "expandable_segments" if change == "expandable" else "PYTORCH_CUDA_ALLOC_CONF"
        state["after"]["allocator_settings"][field] = False if change == "expandable" else "changed"
    with pytest.raises(UnsupportedAllocator):
        bound.validate_current()


@pytest.mark.parametrize("change", ["prior_segment", "private_pool", "interior_pointer", "requested_bytes",
                                   "rounded_bytes", "extra_live_block", "awaiting_free", "block_gap",
                                   "unknown_mapping", "small_segment", "quantum_not_mb", "missing_settings",
                                   "missing_divisions", "nonpower_divisions", "overflow_divisions", "settings_drift"])
def test_unproved_segment_identity_and_configuration_fail_explicitly(change):
    torch, bootstrap, before, state = scenario()
    segment = state["after"]["segments"][0]
    if change == "prior_segment":
        before["segments"] = [{"device": 0}]
    elif change == "private_pool":
        segment["segment_pool_id"] = (1, 7)
    elif change == "interior_pointer":
        segment["address"] += 512
    elif change == "requested_bytes":
        segment["blocks"][0]["requested_size"] += 1
    elif change == "rounded_bytes":
        segment["blocks"][0]["size"] += 512
    elif change == "extra_live_block":
        segment["blocks"][1]["state"] = "active_allocated"
    elif change == "awaiting_free":
        segment["blocks"][1]["state"] = "active_awaiting_free"
    elif change == "block_gap":
        segment["blocks"][1]["address"] += 512
    elif change == "unknown_mapping":
        del segment["is_expandable"]
    elif change == "small_segment":
        segment["segment_type"] = "small"
    elif change == "quantum_not_mb":
        segment["total_size"] += 512
    elif change == "missing_settings":
        del state["after"]["allocator_settings"]
    elif change == "missing_divisions":
        del state["after"]["allocator_settings"]["roundup_power2_divisions"]["1"]
    elif change == "nonpower_divisions":
        state["after"]["allocator_settings"]["roundup_power2_divisions"]["1"] = 3
    elif change == "overflow_divisions":
        state["after"]["allocator_settings"]["roundup_power2_divisions"]["1"] = 1 << 63
    else:
        before["allocator_settings"]["expandable_segments"] = False
    with pytest.raises(UnsupportedAllocator):
        pool_allocation_bound(torch, bootstrap, before_snapshot=before)


@pytest.mark.parametrize("change", ["cpu", "offset", "strided", "partial_storage", "zero_pointer",
                                   "small", "large", "missing_snapshot", "async_backend"])
def test_unsupported_bootstrap_or_inspection_capabilities_are_not_guessed(change):
    torch, bootstrap, before, state = scenario()
    if change == "cpu":
        bootstrap.device.type = "cpu"
    elif change == "offset":
        bootstrap.offset = 1
    elif change == "strided":
        bootstrap.contiguous = False
    elif change == "partial_storage":
        bootstrap.numel = lambda: 1
    elif change == "zero_pointer":
        bootstrap.pointer = 0
    elif change == "small":
        bootstrap.amount = MIB
    elif change == "large":
        bootstrap.amount = 10 * MIB
    elif change == "missing_snapshot":
        torch.cuda.memory._snapshot = None
    else:
        state["backend"] = "cudaMallocAsync"
    with pytest.raises(UnsupportedAllocator):
        pool_allocation_bound(torch, bootstrap, before_snapshot=before)


@pytest.mark.parametrize("amount", [-1, True, 1.5, 1 << 63, (1 << 63) - 1])
def test_request_range_and_rounded_size_exhaustion_are_explicit(amount):
    bound, _ = make()
    with pytest.raises(UnsupportedAllocator):
        bound(amount)

"""CPU boundaries and independent bounded replacement-policy checks."""

from collections import OrderedDict
import random

import pytest
import torch

from expert_cache_oracle import ScalarPolicy, same_tensor_bits
from tensorfold.cuda.expert_cache import HostExpertCache, _HALVE, _Policy, _VECTOR_MIN_CAPACITY, entry_bytes_for


def test_tensor_bit_oracle_distinguishes_zero_sign_and_nan_payloads():
    positive, negative = torch.tensor(0.0), torch.tensor(-0.0)
    assert torch.equal(positive, negative)
    assert not same_tensor_bits(positive, negative)
    first = torch.tensor([0x7FC00001], dtype=torch.int32).view(torch.float32)
    second = torch.tensor([0x7FC00002], dtype=torch.int32).view(torch.float32)
    assert torch.isnan(first).all() and torch.isnan(second).all()
    assert not torch.equal(first, first.clone())
    assert same_tensor_bits(first, first.clone())
    assert not same_tensor_bits(first, second)
    values = torch.arange(6, dtype=torch.float32).reshape(2, 3).T
    assert not values.is_contiguous()
    assert same_tensor_bits(values, values.contiguous())
    assert not same_tensor_bits(first, first.reshape(1, 1))
    assert not same_tensor_bits(first, first.view(torch.int32))


@pytest.mark.parametrize("payloads", [(), [], (torch.empty(0, 2),), (torch.empty(2, 0),),
    (torch.empty(2, 3).T,), (torch.empty(2, 3, requires_grad=True),), (torch.empty(2, 3), torch.empty(3, 3))])
def test_invalid_authoritative_payloads_fail_before_allocation(payloads):
    with pytest.raises(ValueError):
        entry_bytes_for(payloads)


def test_entry_layout_alignment_and_exact_affine_bytes():
    assert entry_bytes_for((torch.empty(7, 3, dtype=torch.int32), torch.empty(7, 5, dtype=torch.int16))) == 32
    up = torch.empty(5, 4, 8, 2, 160, dtype=torch.int32)
    down = torch.empty(5, 8, 4, 1, 160, dtype=torch.int32)
    assert entry_bytes_for((up, down)) == (up[0].numel() + down[0].numel()) * 4


@pytest.mark.parametrize("arguments", [(True, 16, "cuda"), (16, False, "cuda"), (0, 16, "cuda"),
    (15, 16, "cuda"), (128, 17, "cuda"), (16, 16, "cpu"), (10**400, 16, "cuda"), (32, 10**400, "cuda")])
def test_invalid_pool_contract_fails_before_cuda_initialization(arguments):
    with pytest.raises(ValueError):
        HostExpertCache(*arguments)


def test_aging_table_matches_independent_integer_oracle():
    assert bytearray(range(256)).translate(_HALVE) == bytearray(value // 2 for value in range(256))
    policy = _Policy(2)
    policy.register(0, 256)
    policy.frequency[0][:] = bytes(range(256))
    policy.touches = policy.interval - 1
    policy.touch([(0, 255)])
    expected = bytearray(value // 2 for value in range(256))
    expected[255] += 1
    assert policy.frequency[0] == expected
    assert len(policy.frequency[0]) == 256


def test_policy_frequency_saturates_and_resident_ties_are_stable():
    policy = _Policy(2)
    policy.register(0, 4)
    policy.frequency[0][0] = 255
    policy.touch([(0, 0)])
    assert policy.frequency[0][0] == 255
    policy.install(0, (0, 1))
    policy.install(1, (0, 2))
    assert policy.victim(set()) == 0
    assert policy.victim({0}) == 1
    with pytest.raises(ValueError):
        policy.victim({0, 1})
    with pytest.raises(ValueError):
        policy.register(0, 4)


def _lfu_misses(trace, capacity, count):
    policy = _Policy(capacity)
    policy.register(0, count)
    misses = 0
    for expert in trace:
        key = (0, expert)
        policy.touch([key])
        slot = policy.resident.get(key)
        if slot is None:
            misses += 1
            slot = policy.victim(set())
            policy.remove(slot)
            policy.install(slot, key)
        else:
            policy.recency[slot] = policy.tick
        assert len(policy.resident) <= capacity and len(policy.frequency[0]) == count
    return misses, policy


def _lru_misses(trace, capacity):
    resident = OrderedDict()
    misses = 0
    for expert in trace:
        if expert in resident:
            resident.move_to_end(expert)
        else:
            misses += 1
            if len(resident) == capacity:
                resident.popitem(last=False)
            resident[expert] = None
    return misses


def test_full_layer_scans_retain_frequently_reused_keys_over_lru():
    # A fixed controlled trace models a reused pair at each layer then one cold
    # expert per cycle. This compares transfer counts, not PCIe service speed.
    trace = ([0, 1, 0, 1, *range(2, 18)] * 30)
    misses, _ = _lfu_misses(trace, 4, 18)
    assert misses < _lru_misses(trace, 4)
    assert _lfu_misses(trace, 4, 18)[0] == misses


def test_policy_ages_to_new_routing_distribution():
    trace = [0, 1] * 40 + [2, 3] * 200
    _, policy = _lfu_misses(trace, 2, 4)
    assert set(policy.resident) == {(0, 2), (0, 3)}


@pytest.mark.parametrize("capacity", [4, _VECTOR_MIN_CAPACITY])
def test_recency_rebase_preserves_every_order_and_equal_time_tie(capacity):
    policy = _Policy(capacity)
    policy.register(0, 4)
    original = [0] * capacity
    original[:4] = [policy.interval, policy.interval, policy.interval + capacity, 42]
    policy.recency[:] = original
    policy.tick = policy.interval + capacity
    policy.touches = policy.interval - 1
    policy.touch([(0, 0)])
    for first in range(capacity):
        for second in range(capacity):
            assert (policy.recency[first] < policy.recency[second]) == (original[first] < original[second])
            assert (policy.recency[first] == policy.recency[second]) == (original[first] == original[second])
    assert policy.tick > max(policy.recency) and policy.touches == 0
    clock = policy.tick
    for _ in range(1000):
        policy.touch([])
    assert policy.tick == clock


def test_bounded_policy_replays_identical_slots_against_unbounded_clock_oracle():
    capacity, counts = 4, {0: 16, 1: 12}
    policy = _Policy(capacity)
    for layer, count in counts.items():
        policy.register(layer, count)
    # The independent oracle uses lifetime clocks and dictionaries rather than
    # byte arrays or rebasing. Victims follow frequency, time, physical index.
    scores = {(layer, expert): 0 for layer, count in counts.items() for expert in range(count)}
    keys, stamps = [None] * capacity, [0] * capacity
    tick = touches = ages = 0
    generator = random.Random(7)
    for invocation in range(1800):
        layer = invocation % 2
        requested = generator.sample(range(counts[layer]), generator.randrange(capacity + 1))
        wanted = [(layer, expert) for expert in requested]
        policy.touch(wanted)
        for key in wanted:
            touches += 1
            if touches % (64 * capacity) == 0:
                ages += 1
                scores = {key: value // 2 for key, value in scores.items()}
            scores[key] = min(255, scores[key] + 1)
        tick += 1
        oracle_protected = {slot for slot, key in enumerate(keys) if key in wanted}
        actual_protected = {policy.resident[key] for key in wanted if key in policy.resident}
        for key in wanted:
            if key in keys:
                expected = keys.index(key)
            else:
                expected = min((slot for slot in range(capacity) if slot not in oracle_protected),
                    key=lambda slot: (-1 if keys[slot] is None else scores[keys[slot]], stamps[slot], slot))
                keys[expected] = key
            stamps[expected] = tick
            oracle_protected.add(expected)
            slot = policy.resident.get(key)
            if slot is None:
                slot = policy.victim(actual_protected)
                policy.remove(slot)
                policy.install(slot, key)
            else:
                policy.recency[slot] = policy.tick
            actual_protected.add(slot)
            assert slot == expected
        assert policy.keys == keys and len(policy.resident) <= capacity
        assert sum(len(values) for values in policy.frequency.values()) == sum(counts.values())
        assert all(policy.frequency[layer][expert] == value for (layer, expert), value in scores.items())
        assert 0 <= policy.touches < policy.interval and policy.tick <= policy.interval + capacity + 1
    assert ages >= 10


def _request_policy(policy, wanted):
    policy.touch(wanted)
    protected = {policy.resident[key] for key in wanted if key in policy.resident}
    mapping = []
    for key in wanted:
        slot = policy.resident.get(key)
        if slot is None:
            slot = policy.victim(protected)
            policy.remove(slot)
            policy.install(slot, key)
        else:
            policy.recency[slot] = policy.tick
        protected.add(slot)
        mapping.append(slot)
    return mapping


@pytest.mark.parametrize("capacity", [2, _VECTOR_MIN_CAPACITY - 1, _VECTOR_MIN_CAPACITY])
def test_registered_history_rebinds_without_losing_residents_or_byte_bounds(capacity):
    actual, reference = _Policy(capacity), ScalarPolicy(capacity)
    for policy in (actual, reference):
        policy.register(0, 256)
        policy.frequency[0][:] = bytes(range(256))
        policy.touch([(0, 255)])
        assert policy.frequency[0][255] == 255
        policy.install(0, (0, 255))
        policy.register(1, 256)
        policy.frequency[1][:] = bytes(reversed(range(256)))
        policy.touches = policy.interval - 1
        policy.touch([(1, 0), (0, 255)])
    assert actual.frequency == reference.frequency
    assert actual.frequency[0][255] == 128 and actual.frequency[1][0] == 128
    assert actual.keys == reference.keys and actual.resident == reference.resident
    assert list(actual.recency) == reference.recency
    assert actual.victim({0}) == reference.victim({0})
    assert sum(map(len, actual.frequency.values())) == 512 and len(actual._flat) == 513
    snapshot = actual.tick, actual.touches, bytes(actual._flat)
    for _ in range(1000):
        actual.touch([])
    assert (actual.tick, actual.touches, bytes(actual._flat)) == snapshot


@pytest.mark.parametrize("capacity", [2, _VECTOR_MIN_CAPACITY - 1, _VECTOR_MIN_CAPACITY, 64, 512, 2796])
def test_canonical_policy_exactly_replays_independent_scalar_reference(capacity):
    counts = ({layer: 513 for layer in range(49)} if capacity > 512 else
              {0: max(16, 2 * capacity), 1: max(12, capacity + 4)})
    actual, reference = _Policy(capacity), ScalarPolicy(capacity)
    for policy in (actual, reference):
        for layer, count in counts.items():
            policy.register(layer, count)
    width = min(capacity, 8)
    remaining = capacity
    for layer, count in counts.items():
        take = min(remaining, count)
        for start in range(0, take, width):
            wanted = [(layer, expert) for expert in range(start, min(start + width, take))]
            assert _request_policy(actual, wanted) == _request_policy(reference, wanted)
        remaining -= take
    assert len(actual.resident) == capacity
    actual.touches = reference.touches = actual.interval - 1
    generator = random.Random(73 + capacity)
    layers = tuple(counts)
    for invocation in range(2200):
        layer = layers[invocation % len(layers)]
        requested = generator.sample(range(counts[layer]), generator.randrange(width + 1))
        wanted = [(layer, expert) for expert in requested]
        assert _request_policy(actual, wanted) == _request_policy(reference, wanted)
        assert actual.keys == reference.keys and actual.resident == reference.resident
        assert actual.frequency == reference.frequency
        assert list(actual.recency) == reference.recency
        assert actual.tick == reference.tick and actual.touches == reference.touches
        assert 0 <= actual.touches < actual.interval and actual.tick <= actual.interval + capacity + 1
    assert sum(map(len, actual.frequency.values())) == sum(counts.values())
    with pytest.raises(ValueError, match="capacity"):
        actual.victim(set(range(capacity)))


def test_policy_rejects_capacity_above_supported_integer_bounds_before_allocation():
    with pytest.raises(ValueError, match="32-bit"):
        _Policy(2**31)

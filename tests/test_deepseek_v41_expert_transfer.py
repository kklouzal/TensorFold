from random import Random

import pytest

from tensorfold.families.deepseek_v4.cuda.expert_transfer import ExpertSwap, plan_expert_transfers


def _minimum_swaps(current, proposed):
    # Count cycles of old-slot to final-slot, independently of the swap schedule.
    destinations = [proposed.index(expert) for expert in current]
    visited = set()
    cycle_count = 0
    for origin in range(len(destinations)):
        if origin in visited:
            continue
        cycle_count += 1
        slot = origin
        while slot not in visited:
            visited.add(slot)
            slot = destinations[slot]
    return len(destinations) - cycle_count


@pytest.mark.parametrize("etp,group_slots", [(1, 48), (2, 96), (4, 192)])
def test_minimal_rounds_reach_full_placement_without_losing_etp_lanes(etp, group_slots):
    rng = Random(20261125)
    current = tuple(tuple(rng.sample(range(384), 384)) for _ in range(3))
    proposed = (tuple(rng.sample(range(384), 384)), tuple(reversed(current[1])), current[2])
    plan = plan_expert_transfers(current, proposed, etp=etp)
    assert plan.etp == etp and plan.group_slots == group_slots
    assert plan.current == current and plan.proposed == proposed
    assert plan == plan_expert_transfers(current, proposed, etp=etp)
    assert sum(map(len, plan.rounds)) == sum(_minimum_swaps(before, after) for before, after in
                                             zip(current, proposed, strict=True))

    def carriers(rows):
        return [
            [[(expert, rank % etp) for expert in row[start:start + group_slots]] for row in rows]
            for rank in range(8)
            for start in [rank // etp * group_slots]
        ]

    banks = carriers(current)
    for round_swaps in plan.rounds:
        owners = set()
        assert round_swaps
        for swap in round_swaps:
            assert 0 <= swap.left < swap.right < 384 and 0 <= swap.layer < len(current)
            owner_a, slot_a = divmod(swap.left, group_slots)
            owner_b, slot_b = divmod(swap.right, group_slots)
            assert owners.isdisjoint((owner_a, owner_b))
            owners.update((owner_a, owner_b))
            for part in range(etp):
                bank_a = banks[owner_a * etp + part][swap.layer]
                bank_b = banks[owner_b * etp + part][swap.layer]
                bank_a[slot_a], bank_b[slot_b] = bank_b[slot_b], bank_a[slot_a]
    assert banks == carriers(proposed)


def test_rounds_pack_disjoint_owners_and_preserve_owner_order():
    row = tuple(range(384))
    target = list(row)
    for left, right in [(0, 48), (1, 49), (2, 3), (96, 144)]:
        target[left], target[right] = target[right], target[left]
    second = list(row)
    second[4], second[5] = second[5], second[4]
    plan = plan_expert_transfers((row, row), (tuple(target), tuple(second)), etp=1)
    assert plan.rounds == (
        (ExpertSwap(0, 0, 48), ExpertSwap(0, 96, 144)),
        (ExpertSwap(0, 1, 49),),
        (ExpertSwap(0, 2, 3),),
        (ExpertSwap(1, 4, 5),),
    )


def test_equal_maps_have_no_transfers():
    row = (tuple(reversed(range(384))),)
    plan = plan_expert_transfers(row, row, etp=2)
    assert plan.current == plan.proposed == row
    assert plan.rounds == ()


@pytest.mark.parametrize(("current", "proposed", "etp", "expected"), [
    ((), (), 4, "immutable layer rows"),
    (((0, 1, 2),), ((0, 1, 2),), 4, "divide evenly"),
    (((0, 1),), ((0, 1),), True, "require TP8"),
    (((0, 1),), ((0, 1),), 8, "require TP8"),
    (((0, 0),), ((0, 1),), 4, "current placement layer 0"),
    (((0, 1),), ((0, True),), 4, "proposed placement layer 0"),
    (((0, 1),), ((0, 2),), 4, "proposed placement layer 0"),
    (((0, 1),), ([0, 1],), 4, "proposed placement layer 0"),
    (((0, 1),), ((0, 1), (0, 1)), 4, "contain 1 immutable layer"),
])
def test_invalid_maps_are_rejected_before_any_transfer(current, proposed, etp, expected):
    with pytest.raises(ValueError, match=expected):
        plan_expert_transfers(current, proposed, etp=etp)

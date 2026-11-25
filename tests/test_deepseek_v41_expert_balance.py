"""CPU contracts for observed expert loads and balanced move proposals."""
import pytest

from tensorfold.families.deepseek_v4.cuda.expert_balance import plan_expert_placement, rolling_expert_load


def test_route_decay_and_idle_layer():
    initial = rolling_expert_load(None, ((4, 0), (0, 4)), half_life_routes=4)
    assert initial == ((1.0, 0.0), (0.0, 1.0))
    second = rolling_expert_load(initial, ((0, 4), (0, 0)), half_life_routes=4)
    assert second == ((0.5, 0.5), initial[1])
    assert rolling_expert_load(second, ((0, 0), (0, 0)), half_life_routes=4) == second
    assert rolling_expert_load(initial, ((0, 10**400), (0, 0)), half_life_routes=4)[0] == (0.0, 1.0)
    assert rolling_expert_load(None, ((0, 0),), half_life_routes=4) == ((0.0, 0.0),)


def worst(placement, phase, groups):
    width = len(placement) // groups
    return max(sum(phase[expert] for expert in placement[start:start + width])
               for start in range(0, len(placement), width))


@pytest.mark.parametrize("etp", [1, 2, 4])
def test_balanced_proposal_is_deterministic_and_improves_both_phases(etp):
    groups = 8 // etp
    experts = 2 * groups
    incumbent = tuple(range(experts))
    phases = ((8, 6, *(1,) * (experts - 2)), (9, 5, *(1,) * (experts - 2)))
    loads = tuple((phase,) for phase in phases)
    [proposal] = plan_expert_placement(loads, etp=etp)
    assert [proposal] == list(plan_expert_placement(loads, etp=etp, current=(incumbent,)))
    assert sorted(proposal) == list(incumbent)
    assert sum(a != b for a, b in zip(incumbent, proposal, strict=True)) == 2
    assert all(worst(proposal, phase, groups) < worst(incumbent, phase, groups) for phase in phases)


def test_swap_ordering_retains_improvements_below_float_resolution():
    high = 2**60
    phase = (high + 2, high + 4, high, high - 2)
    [proposal] = plan_expert_placement(((phase,),), etp=4)
    # The first improving swap (0, 2) leaves a higher worst load. Two
    # optimal swaps follow, so existing slot order still chooses (0, 3).
    assert proposal == (3, 1, 2, 0)
    assert worst(proposal, phase, 2) == 2 * high + 2


def test_ties_and_incompatible_phase_do_not_move_experts():
    incumbent = ((1, 0, 3, 2),)
    assert plan_expert_placement((((7, 7, 1, 1),), ((4, 4, 0, 8),)), etp=4, current=incumbent) == incumbent
    assert plan_expert_placement((((0, 0, 0, 0),),), etp=4, current=incumbent) == incumbent
    assert plan_expert_placement((((4, 4, 4, 4),),), etp=4, current=incumbent) == incumbent


def test_reject_invalid_work_or_map():
    with pytest.raises(ValueError, match="integer counts"):
        rolling_expert_load(None, ((True, 0),), half_life_routes=4)
    with pytest.raises(ValueError, match="every expert exactly once"):
        plan_expert_placement((((7, 7, 1, 1),),), etp=4, current=((0, 0, 2, 3),))
    with pytest.raises(ValueError, match="matching shapes"):
        plan_expert_placement((((7, 7, 1, 1),), ((4, 4),)), etp=4)

"""Plan in-place DeepSeek expert-bank transfers for a fixed TP8 ETP degree."""

from dataclasses import dataclass


Placement = tuple[tuple[int, ...], ...]


def local_expert_layout(row: tuple[int, ...], *, owner: int, group_slots: int):
    """Return the local slot order and inverse map for an already validated placement row."""
    start = owner * group_slots
    stop = start + group_slots
    selected = row[start:stop]
    if selected == tuple(range(start, stop)):
        return None
    inverse = [-1] * len(row)
    for slot, logical in enumerate(selected):
        inverse[logical] = slot
    return selected, inverse


@dataclass(frozen=True, slots=True)
class ExpertSwap:
    layer: int
    left: int
    right: int


@dataclass(frozen=True, slots=True)
class ExpertTransferPlan:
    """An immutable schedule tied to one source placement.

    Before executing, the owner must match both the active epoch and
    `current`, then exclude weight readers until the transfers and route-map
    publication have completed. A completed plan cannot be replayed as-is.
    """

    etp: int
    group_slots: int
    current: Placement
    proposed: Placement
    rounds: tuple[tuple[ExpertSwap, ...], ...]


def _validate(placement, *, label, layers, experts):
    if type(placement) is not tuple or len(placement) != layers:
        raise ValueError(f"{label} placement must contain {layers} immutable layer rows")
    expected = set(range(experts))
    for layer, row in enumerate(placement):
        if (type(row) is not tuple or len(row) != experts
                or any(type(expert) is not int for expert in row) or set(row) != expected):
            raise ValueError(f"{label} placement layer {layer} must contain each of its {experts} experts exactly once")


def _layer_swaps(current, proposed):
    slots = list(current)
    position = [0] * len(slots)
    for slot, expert in enumerate(slots):
        position[expert] = slot
    for left, wanted in enumerate(proposed):
        right = position[wanted]
        if left == right:
            continue
        # Moving the wanted expert here splits its cycle, fixing at least
        # this slot. Each length-k cycle therefore uses exactly k-1 swaps.
        displaced = slots[left]
        slots[left], slots[right] = wanted, displaced
        position[wanted], position[displaced] = left, right
        yield left, right


def plan_expert_transfers(current: Placement, proposed: Placement, *, etp: int) -> ExpertTransferPlan:
    """Compute the minimum number of swaps for two physical-slot maps.

    For each swap, `divmod(slot, group_slots)` gives its owner and local
    slot. A cross-owner transfer pairs identical ETP lanes, so part `p` on
    owner `g` lives on rank `g * etp + p`. Within-owner swaps are local to
    each rank in that group. Each round touches an owner at most once and
    preserves that owner's swap order, including between layers.
    """
    if type(etp) is not int or etp not in (1, 2, 4):
        raise ValueError("expert transfers require TP8 ETP1, ETP2 or ETP4")
    if type(current) is not tuple or not current or type(current[0]) is not tuple:
        raise ValueError("current placement must contain immutable layer rows")
    groups = 8 // etp
    experts = len(current[0])
    if not experts or experts % groups:
        raise ValueError("expert count must divide evenly across EP groups")
    layers = len(current)
    _validate(current, label="current", layers=layers, experts=experts)
    _validate(proposed, label="proposed", layers=layers, experts=experts)
    group_slots = experts // groups
    last_round = [-1] * groups
    rounds = []
    for layer, (before, after) in enumerate(zip(current, proposed, strict=True)):
        if before == after:
            continue
        for left, right in _layer_swaps(before, after):
            left_owner = left // group_slots
            right_owner = right // group_slots
            # Only swaps with disjoint owners can move into an earlier round.
            step = 1 + max(last_round[left_owner], last_round[right_owner])
            if step == len(rounds):
                rounds.append([])
            rounds[step].append(ExpertSwap(layer, left, right))
            last_round[left_owner] = last_round[right_owner] = step
    return ExpertTransferPlan(etp, group_slots, current, proposed, tuple(tuple(step) for step in rounds))

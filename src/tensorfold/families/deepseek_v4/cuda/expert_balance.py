"""CPU load estimates and balanced, non-replicated expert placement proposals."""
from itertools import combinations
from math import exp, expm1, fsum, isfinite, lcm, log


_LN_2 = log(2.0)


def _shape(rows):
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("expert load requires a nonempty rectangular layer table")
    return len(rows), len(rows[0])


def _check_load(values):
    if any(type(value) not in (int, float) or value < 0
           or (type(value) is float and not isfinite(value)) for value in values):
        raise ValueError("expert loads must be finite nonnegative numbers")


def rolling_expert_load(previous, counts, *, half_life_routes):
    """Return normalized load rows for one phase, indexed by layer and global expert.

    Decay is in observed routes per layer. A nonempty batch contributes its route
    proportions, and an idle row keeps its history. None starts a new history.
    Observation cadence and the decision to propose placement belong to the caller.
    """
    if (type(half_life_routes) not in (int, float) or half_life_routes <= 0
            or (type(half_life_routes) is float and not isfinite(half_life_routes))):
        raise ValueError("half_life_routes must be finite and positive")
    half, scale = half_life_routes.as_integer_ratio()
    shape = _shape(counts)
    if previous is None:
        previous = [(0.0,) * shape[1]] * shape[0]
    elif _shape(previous) != shape:
        raise ValueError("previous expert load and counts must have the same shape")
    result = []
    for history, row in zip(previous, counts, strict=True):
        _check_load(history)
        if any(value > 1 for value in history):
            raise ValueError("previous expert load must contain proportions")
        if any(type(count) is not int or count < 0 for count in row):
            raise ValueError("observed routes must be nonnegative integer counts")
        total = sum(row)
        if not total:
            result.append(tuple(history))
            continue
        fresh = tuple(count / total for count in row)
        prior = fsum(history)
        if not prior:
            result.append(fresh)
            continue
        try:
            decay = -_LN_2 * (total * scale / half)
        except OverflowError:
            result.append(fresh)
            continue
        keep, update = exp(decay), -expm1(decay)
        mixed = tuple(keep * (old / prior) + update * observed for old, observed in zip(history, fresh, strict=True))
        normalizer = fsum(mixed)
        result.append(tuple(value / normalizer for value in mixed))
    return tuple(result)


def _scaled_loads(values):
    # A common binary denominator makes comparisons of group sums exact.
    # In particular, a mathematically tied swap cannot appear beneficial
    # because it changes the rounding order of the old group totals.
    _check_load(values)
    ratios = [value.as_integer_ratio() for value in values]
    scale = max(denominator for _, denominator in ratios)
    return tuple(value * (scale // denominator) for value, denominator in ratios)


def _balanced_swap(placement, loads, groups):
    size = len(placement) // groups
    weights = [tuple(phase[expert] for expert in placement) for phase in loads]
    totals = [tuple(sum(phase[i:i + size]) for i in range(0, len(placement), size)) for phase in weights]
    worst = tuple(max(phase) for phase in totals)
    mass = tuple(sum(phase) for phase in totals)
    if all(high * groups == total for high, total in zip(worst, mass, strict=True)):
        return placement
    # Normalize every phase to one denominator. Integer keys retain close
    # improvements without constructing fractions in the swap loop.
    denominator = lcm(*(total for total in mass if total))
    units = tuple(denominator // total if total else 0 for total in mass)
    best = None
    move = None
    for left, right in combinations(range(groups), 2):
        outside = tuple(max((value for group, value in enumerate(phase) if group not in (left, right)), default=0)
                        for phase in totals)
        for i in range(left * size, (left + 1) * size):
            for j in range(right * size, (right + 1) * size):
                updated = []
                for phase, group_load, rest, limit in zip(weights, totals, outside, worst, strict=True):
                    shift = phase[j] - phase[i]
                    high = max(rest, group_load[left] + shift, group_load[right] - shift)
                    if high > limit:
                        break
                    updated.append(high)
                else:
                    if not any(high < limit for high, limit in zip(updated, worst, strict=True)):
                        continue
                    score = tuple(sorted((high * unit for high, unit in zip(updated, units, strict=True) if unit),
                                         reverse=True))
                    if best is None or score < best:
                        best = score
                        move = i, j
    if move is None:
        return placement
    i, j = move
    slots = list(placement)
    slots[i], slots[j] = slots[j], slots[i]
    return tuple(slots)


def _pack(placement, loads, groups):
    """Pack heaviest experts first with a fixed number of slots per group."""
    size = len(placement) // groups
    assigned = [set() for _ in range(groups)]
    totals = [(0,) * len(loads) for _ in range(groups)]
    weights = tuple(zip(*loads, strict=True))
    for expert in sorted(range(len(placement)), key=lambda e: -max(weights[e])):
        def priority(group):
            merged = (a + b for a, b in zip(totals[group], weights[expert], strict=True))
            return tuple(sorted(merged, reverse=True)), group != expert // size, group

        chosen = min((group for group in range(groups) if len(assigned[group]) < size), key=priority)
        assigned[chosen].add(expert)
        totals[chosen] = tuple(a + b for a, b in zip(totals[chosen], weights[expert], strict=True))
    slots = []
    for group, selected in enumerate(assigned):
        incumbent = placement[group * size:(group + 1) * size]
        # Keep the existing physical slots of every expert that stays local.
        incoming = iter(sorted(selected.difference(incumbent)))
        slots.extend(expert if expert in selected else next(incoming) for expert in incumbent)
    return tuple(slots)


def _worst_groups(placement, loads, groups):
    size = len(placement) // groups
    return tuple(max(sum(phase[expert] for expert in placement[start:start + size])
                     for start in range(0, len(placement), size)) for phase in loads)


def _initial_placement(placement, loads, groups):
    active = [(phase, sum(phase)) for phase in loads]
    active = [(phase, total) for phase, total in active if total]
    denominator = lcm(*(total for _, total in active))
    normalized = [tuple(weight * (denominator // total) for weight in phase) for phase, total in active]
    baseline = _worst_groups(placement, normalized, groups)
    if all(high * groups == denominator for high in baseline):
        return placement

    best = _balanced_swap(placement, loads, groups)
    def key(slots, worst):
        moves = sum(a != b for a, b in zip(placement, slots, strict=True))
        return tuple(sorted(worst, reverse=True)), moves

    best_key = key(best, _worst_groups(best, normalized, groups))
    # Try each active phase on its own and a joint packing on the same scale.
    seeds = [(phase,) for phase in normalized]
    if len(normalized) > 1:
        seeds.append(tuple(normalized))
    for seed in seeds:
        slots = _pack(placement, seed, groups)
        worst = _worst_groups(slots, normalized, groups)
        if worst == baseline or any(new > old for new, old in zip(worst, baseline, strict=True)):
            continue
        score = key(slots, worst)
        if score < best_key:
            best, best_key = slots, score
    return best


def plan_expert_placement(loads, *, etp, current=None):
    """Return [layer][physical slot] global expert IDs for TP8 ETP1/2/4.

    Each consecutive equal-size group of slots is one EP group. Initial
    placement tries equal-cardinality packing for each phase and a joint
    packing, with a single swap as a fallback. For an active map, consider
    only one swap per layer. Every proposal improves at least one phase's
    worst group without increasing any other phase's worst group. Minimize
    the worst normalized phase first, then avoid unnecessary movement.
    Movement costs and observed persistence are separate decisions.
    """
    if type(etp) is not int or etp not in (1, 2, 4):
        raise ValueError("expert placement requires TP8 ETP1, ETP2 or ETP4")
    if not loads:
        raise ValueError("expert placement requires at least one phase")
    layers, experts = _shape(loads[0])
    if any(_shape(phase) != (layers, experts) for phase in loads):
        raise ValueError("expert load phases must have matching shapes")
    groups = 8 // etp
    if experts % groups:
        raise ValueError("expert count must divide evenly across EP groups")
    identity = tuple(range(experts))
    propose = _initial_placement if current is None else _balanced_swap
    if current is None:
        current = (identity,) * layers
    elif _shape(current) != (layers, experts):
        raise ValueError("current expert placement must match the load shape")
    expected = set(identity)
    result = []
    for layer, row in enumerate(current):
        if any(type(expert) is not int for expert in row) or set(row) != expected:
            raise ValueError("each layer must contain every expert exactly once")
        weights = [_scaled_loads(phase[layer]) for phase in loads]
        result.append(propose(tuple(row), weights, groups))
    return tuple(result)

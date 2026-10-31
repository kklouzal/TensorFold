"""Standard rejection laws, immutable ownership and request-keyed truncation controls."""

from fractions import Fraction

import numpy as np
import pytest

from tensorfold.engine import rejection
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import (
    PROPOSAL,
    RESIDUAL,
    ChainPacket,
    ProbabilityRow,
    draw,
    node,
    proposal_row,
    verify,
)


def row(weights):
    return ProbabilityRow(list(weights), list(weights.values()))


def test_exact_single_step_identity_by_independent_rational_branch_enumeration():
    # Independent finite calculation: q(x)*accept(x) plus rejected mass times the residual law equals p.
    p = [Fraction(2, 5), Fraction(1, 4), Fraction(1, 5), Fraction(1, 10), Fraction(1, 20)]
    q = [Fraction(1, 10), Fraction(1, 2), Fraction(1, 20), Fraction(3, 10), Fraction(1, 20)]
    accepted = [min(a, b) for a, b in zip(p, q)]
    rejected = 1 - sum(accepted)
    residual = [max(Fraction(0), a - b) for a, b in zip(p, q)]
    assert [a + rejected * r / sum(residual) for a, r in zip(accepted, residual)] == p


@pytest.mark.parametrize("children", [1, 2, 3, 5])
def test_archived_without_replacement_node_distribution_gate(children):
    # Existing archive's 20k-seed/0.999 chi-square gate, unchanged sensitivity.
    p, q = row({0: .4, 1: .25, 2: .2, 3: .1, 4: .05}), row({0: .1, 1: .5, 2: .05, 3: .3, 4: .05})
    counts = np.zeros(5)
    for seed in range(20000):
        from tensorfold.engine.exact_sampling import uniform

        order = np.argsort(-(np.log(q.probs) - np.log(-np.log(uniform(seed, 17, q.ids)))), kind="stable")
        kids = q.ids[order[:children]].tolist()
        index, token = node(p, q, kids, seed, 17)
        assert index < 0 or kids[index] == token
        counts[token] += 1
    expected = p.probs * counts.sum()
    assert float(((counts - expected) ** 2 / expected).sum()) < 18.47


def test_q_equal_p_is_always_accepted_and_disjoint_support_always_rejected():
    p = row({0: .25, 1: .75})
    for seed in range(100):
        token = draw(p, seed, 9)
        assert node(p, p, [token], seed, 9) == (0, token)
        index, actual = node(p, row({2: 1}), [2], seed, 9)
        assert index == -1 and actual in (0, 1)


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_block_chain_joint_distribution_retains_archived_gate(depth):
    # Existing archive's 30k-seed/0.999 joint-distribution gate, with actual independent proposal-domain draws.
    target = {None: row({0: .5, 1: .3, 2: .15, 3: .05}), 0: row({0: .1, 1: .6, 2: .2, 3: .1}),
              1: row({0: .25, 1: .25, 2: .25, 3: .25}), 2: row({0: .7, 1: .1, 2: .1, 3: .1}),
              3: row({0: .05, 1: .05, 2: .1, 3: .8})}
    draft = [row({0: .3, 1: .3, 2: .3, 3: .1}), row({0: .1, 1: .7, 2: .1, 3: .1}),
             row({0: .4, 1: .1, 2: .4, 3: .1})]
    counts = np.zeros((4, 4))
    for seed in range(30000):
        out = []
        while len(out) < 2:
            first = len(out)
            qs = tuple(draft[min(first + j, 2)] for j in range(depth))
            tokens = tuple(draw(q, seed, first + j) for j, q in enumerate(qs))
            p_rows = [target[out[-1] if out else None], *(target[token] for token in tokens)]
            path, terminal = verify(ChainPacket(tokens, qs, first), p_rows, seed, room=2 - first)
            out.extend(tokens[i - 1] for i in path[1:])
            out.append(terminal)
        counts[out[0], out[1]] += 1
    probabilities = np.array([[target[None].probability(a) * target[a].probability(b) for b in range(4)]
                              for a in range(4)])
    expected = probabilities * 30000
    assert float(((counts - expected) ** 2 / expected).sum()) < 37.70


def test_final_room_performs_rejection_and_preserves_every_prefix_of_the_logical_sequence():
    q = row({0: 1})
    chain = ChainPacket((0, 0), (q, q), 17)
    ps = [row({0: .5, 1: .5}), q, q]
    for seed in range(100):
        path, terminal = verify(chain, ps, seed)
        complete = [chain.tokens[i - 1] for i in path[1:]] + [terminal]
        for room in (1, 2, 3):
            path, terminal = verify(chain, ps, seed, room=room)
            assert [chain.tokens[i - 1] for i in path[1:]] + [terminal] == complete[:room]
    assert verify(chain, ps, 6, room=1) == ([0], 1)  # real keyed schedule counterexample: never serial bonus0
    assert verify(ChainPacket((0,), (q,), 17), [q, None], 6, room=1) == ([0], 0)


def test_eos_and_unreachable_grammar_rows_keep_the_pending_terminal_out_of_commit():
    q, p = row({2: 1}), row({0: 1})
    chain = ChainPacket((2,), (q,), 5)
    assert verify(chain, [p, None], 10) == ([0], 0)
    assert verify(chain, [q, None], 10, eos=(2,)) == ([0], 2)
    with pytest.raises(ValueError, match="bonus row"):
        verify(chain, [q, None], 10)


def test_probability_rows_and_packets_own_deeply_read_only_snapshots():
    ids, probabilities = np.array([1, 4]), np.array([.25, .75])
    p = ProbabilityRow(ids, probabilities)
    ids[:] = 9
    probabilities[:] = 0
    assert p.ids.tolist() == [1, 4] and p.probs.tolist() == [.25, .75]
    for array in (p.ids, p.probs):
        with pytest.raises(ValueError):
            array.setflags(write=True)
    packet = ChainPacket([1], [p], 7, [.5])
    assert packet.tokens == (1,) and packet.q_rows == (p,)


@pytest.mark.parametrize("ids,weights", [([1, 1], [.5, .5]), ([1.5], [1]), ([-1], [1]), ([1], [0]),
                                        ([1], [-1]), ([1], [float("nan")]), ([1], [float("inf")])])
def test_malformed_probability_rows_fail_at_the_owning_boundary(ids, weights):
    with pytest.raises(ValueError):
        ProbabilityRow(ids, weights)


def test_subnormal_proposal_never_accepts_zero_target_mass():
    q = ProbabilityRow([1, 0, 2], [np.nextafter(0., 1.), .5, .5])
    for seed in range(20):
        assert node(row({0: 1}), q, [1], seed, 1)[0] == -1


def test_sparse_proposal_is_the_actual_temperature_filtered_distribution():
    sampling = Sampling(7, temperature=.5, top_k=2, top_p=.8, min_p=.1)
    q = proposal_row(np.array([0., 2., 1.]), np.array([9, 4, 2]), sampling)
    assert q.ids.tolist() == [4] and q.probs.tolist() == [1]
    assert draw(q, 9, 3) == 4


def test_endpoint_witness_cannot_promote_a_tiny_mass_or_grammar_mask():
    witness = 4199689772251414973
    p, forbidden = row({0: 1e-300, 1: 1.}), row({2: 1.})
    # Force the historical endpoint under each actual stochastic domain, retaining full target support.
    with np.errstate(all="raise"):
        assert draw(p, witness ^ PROPOSAL, 1, domain=PROPOSAL) == 1
        assert draw(p, witness ^ RESIDUAL, 1, domain=RESIDUAL) == 1
        assert node(p, forbidden, [2], witness ^ RESIDUAL, 1) == (-1, 1)
        chain = ChainPacket((2,), (forbidden,), 1)
        assert verify(chain, [p, None], witness ^ RESIDUAL) == ([0], 1)
        assert verify(chain, [row({1: 1.}), None], witness, room=1) == ([0], 1)


def fraction_residual(p, q):
    """Independent exact categorical fractions, conditioned before one nearest-FP64 conversion."""
    a = {int(i): Fraction.from_float(float(w)) for i, w in zip(p.ids, p.probs)}
    b = {int(i): Fraction.from_float(float(w)) for i, w in zip(q.ids, q.probs)}
    ap, bq = sum(a.values()), sum(b.values())
    residual = {i: max(w / ap - b.get(i, Fraction()) / bq, Fraction()) for i, w in a.items()}
    total = sum(residual.values())
    if not total:
        return {}
    return {i: float(w / total) for i, w in residual.items() if float(w / total) > 0}


def test_near_equal_rejection_keeps_the_exact_conditioned_residual():
    p, q = row({0: .3, 1: .7}), row({0: np.nextafter(.3, 1.), 1: .7})
    seed = 5783094707825395892
    assert fraction_residual(p, q) == {1: 1.}
    assert node(p, q, [0], seed, 1) == (-1, 1)
    assert node(p, q, [0, 1], seed, 1) == (1, 1)
    result = rejection._residual(p, q)
    assert result.ids.tolist() == [1] and result.probs.tolist() == [1.]


@pytest.mark.parametrize("a,b", [
    ({0: .3, 1: .7}, {0: np.nextafter(.3, 1.), 1: .7}),
    ({0: np.nextafter(0., 1.), 1: .5, 2: .5}, {1: .5, 2: .5}),
    ({0: 1.}, {1: 1.}),
    ({0: np.nextafter(0., 1.), 1: 1., 2: 1e-200}, {0: 2 * np.nextafter(0., 1.), 1: 1., 2: 1e-300}),
    ({0: .1, 1: .2, 2: .7}, {0: .1, 1: .3, 2: .6}),
])
def test_integer_residual_matches_independent_fraction_conditioning(a, b):
    p, q = row(a), row(b)
    expected = fraction_residual(p, q)
    actual = rejection._residual(p, q)
    assert dict(zip(actual.ids.tolist(), actual.probs.tolist())) == expected
    assert actual.probs.max() > 0 and np.isfinite(actual.probs).all()


def test_acceptance_bounds_and_exact_fallback_match_rational_comparison(monkeypatch):
    from tensorfold.engine.exact_sampling import uniform

    fixtures = [
        (row({0: .3, 1: .7}), row({0: np.nextafter(.3, 1.), 1: .7})),
        (row({0: np.nextafter(0., 1.), 1: 1.}), row({0: 2 * np.nextafter(0., 1.), 1: 1.})),
        (row({0: .1, 1: .2, 2: .7}), row({0: .1, 1: .3, 2: .6})),
    ]
    for p, q in fixtures:
        for child in q.ids.tolist():
            pc, qc = p.probability(child), q.probability(child)
            for seed in (0, 6, 5783094707825395892):
                u = float(uniform(seed ^ rejection.ACCEPT, 1, np.array([0], dtype=np.int64))[0])
                ap, bq = sum(map(Fraction.from_float, p.probs.tolist())), sum(map(Fraction.from_float, q.probs.tolist()))
                expected = Fraction.from_float(u) * Fraction.from_float(qc) * ap <= Fraction.from_float(pc) * bq
                assert rejection._accept(p, q, pc, qc, u)[0] == expected
    # Widely separated decisions do not need exact sums/integers; this proves the common fast path exists.
    monkeypatch.setattr(rejection, "_grid", lambda *args: pytest.fail("unnecessary exact acceptance fallback"))
    p, q = row({0: .9, 1: .1}), row({0: .1, 1: .9})
    assert rejection._accept(p, q, .9, .1, .9) == (True, None)
    assert rejection._accept(p, q, .1, .9, .9) == (False, None)


def test_sum_and_product_intervals_enclose_exact_subnormal_to_unit_values():
    values = [np.nextafter(0., 1.), np.finfo(np.float64).tiny, .1, .3, .7, 1.]
    for n in (1, 2, 7, 128):
        a = np.resize(np.array(values), n)
        lo, hi = rejection._sum_bounds(a)
        exact = sum(map(Fraction.from_float, a.tolist()))
        assert Fraction.from_float(lo) <= exact <= Fraction.from_float(hi)
        for factor in values:
            lower, upper = rejection._product_bounds((lo, hi), factor)
            product = exact * Fraction.from_float(factor)
            assert Fraction.from_float(lower) <= product <= Fraction.from_float(upper)


@pytest.mark.parametrize("support", [32, 256])
def test_one_visited_row_bigint_peak_fits_derived_host_workspace(support):
    import tracemalloc

    p = ProbabilityRow(np.arange(support), np.ones(support))
    weights = np.full(support, np.nextafter(0., 1.))
    weights[0] = 1.
    q = ProbabilityRow(np.arange(support), weights)
    tracemalloc.start()
    try:
        tracemalloc.reset_peak()
        before = tracemalloc.get_traced_memory()[0]
        result = rejection._residual(p, q)
        peak = tracemalloc.get_traced_memory()[1] - before
    finally:
        tracemalloc.stop()
    assert peak <= rejection.residual_workspace(support) + support * (7 * 8 + 1)
    assert result.ids.tolist() == list(range(1, support))
    assert dict(zip(result.ids.tolist(), result.probs.tolist())) == fraction_residual(p, q)

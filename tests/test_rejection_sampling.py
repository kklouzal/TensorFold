"""Standard rejection laws, immutable ownership and request-keyed truncation controls."""

from fractions import Fraction

import numpy as np
import pytest

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import (
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

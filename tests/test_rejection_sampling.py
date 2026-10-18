"""Keyed speculative rejection sampling (tensorfold.engine.rejection): output distributed exactly as the target.

Chi-square tests over many request seeds on synthetic models; the critical values are the 0.999 quantiles, so a
correct sampler fails one of these checks about once in a thousand runs per statistic (the seeds are fixed, so the
outcome is reproducible).
"""

import numpy as np
import pytest

from tensorfold.engine.exact_sampling import uniform
from tensorfold.engine.rejection import node, walk

CHI2_999 = {3: 16.27, 4: 18.47, 5: 20.52, 15: 37.70}


def _chi2(counts, probs):
    expected = np.asarray(probs) * counts.sum()
    return float(((counts - expected) ** 2 / expected).sum())


def _draft(q, seed, position, count):
    """Gumbel-top-k of q under the target's keyed noise: ``count`` draws without replacement (Kool et al. 2019)."""
    ids = np.array(sorted(q), dtype=np.int64)
    noise = -np.log(-np.log(uniform(seed, position, ids)))
    order = np.argsort(-(np.log([q[int(t)] for t in ids]) + noise), kind="stable")
    return [int(ids[i]) for i in order[:count]]


def _serial(p, seed, position):
    return _draft(p, seed, position, 1)[0]


P = {0: 0.40, 1: 0.25, 2: 0.20, 3: 0.10, 4: 0.05}
Q = {0: 0.10, 1: 0.50, 2: 0.05, 3: 0.30, 4: 0.05}


@pytest.mark.parametrize("children", [1, 2, 3, 5])
def test_one_node_emits_the_target_distribution(children):
    counts = np.zeros(5)
    for seed in range(20000):
        kids = _draft(Q, seed, 17, children)
        i, token = node(P, Q, kids, seed, 17)
        assert (i < 0) or kids[i] == token
        counts[token] += 1
    assert _chi2(counts, [P[t] for t in range(5)]) < CHI2_999[4]


def test_drafts_outside_the_target_support_are_always_rejected():
    p = {0: 0.7, 1: 0.3}
    q = {2: 0.9, 0: 0.1}
    for seed in range(200):
        i, token = node(p, q, [2], seed, 3)
        assert i == -1 and token in (0, 1)


def _conditional(previous):
    rows = {None: [0.5, 0.3, 0.15, 0.05], 0: [0.1, 0.6, 0.2, 0.1], 1: [0.25, 0.25, 0.25, 0.25],
            2: [0.7, 0.1, 0.1, 0.1], 3: [0.05, 0.05, 0.1, 0.8]}
    return dict(enumerate(rows[previous]))


DRAFTER = [{0: 0.3, 1: 0.3, 2: 0.3, 3: 0.1}, {0: 0.1, 1: 0.7, 2: 0.1, 3: 0.1}, {0: 0.4, 1: 0.1, 2: 0.4, 3: 0.1}]


def _generate(seed, depth, length=2):
    """Rounds of a block drafter (row j drafted from DRAFTER[j]'s fixed distribution, independent of earlier drafts)
    verified by ``walk`` until ``length`` tokens exist; the target's row distribution depends on the last token."""
    out = []
    while len(out) < length:
        first = len(out)
        anchor = out[-1] if out else None
        drafts = [_draft(DRAFTER[min(first + j, 2)], seed, first + j, 1)[0] for j in range(depth)]
        tokens = [anchor] + drafts
        context = [anchor] + drafts
        p_rows = [_conditional(t) for t in context]
        q_rows = [DRAFTER[min(first + j, 2)] for j in range(depth)] + [None]
        serial = [_serial(p, seed, first + j) for j, p in enumerate(p_rows)]
        path, terminal = walk(tokens, list(range(-1, depth)), [first + j for j in range(depth + 1)], seed,
                              p_rows, q_rows, serial)
        out.extend([tokens[r] for r in path[1:]] + [terminal])
    return tuple(out[:length])


@pytest.mark.parametrize("depth", [1, 2, 4])
def test_chains_of_block_drafts_emit_the_target_joint_distribution(depth):
    counts = {}
    for seed in range(30000):
        pair = _generate(seed, depth)
        counts[pair] = counts.get(pair, 0) + 1
    pairs = [(a, b) for a in range(4) for b in range(4)]
    probs = [_conditional(None)[a] * _conditional(a)[b] for a, b in pairs]
    assert _chi2(np.array([counts.get(pair, 0) for pair in pairs], dtype=float), probs) < CHI2_999[15]
    assert _generate(123, depth, 6) == _generate(123, depth, 6)          # keyed: same seed, same tokens


def test_a_tree_tests_siblings_in_draft_order_and_follows_the_accepted_branch():
    counts = np.zeros(4)
    for seed in range(20000):
        p0, q0 = _conditional(None), DRAFTER[0]
        kids = _draft(q0, seed, 5, 2)
        tokens = [9, *kids, 3]
        parents = [-1, 0, 0, 1]
        p_rows = [p0, _conditional(kids[0]), _conditional(kids[1]), _conditional(3)]
        q_rows = [q0, {3: 1.0}, None, None]
        serial = [_serial(p, seed, 5 + d) for p, d in zip(p_rows, [0, 1, 1, 2])]
        path, terminal = walk(tokens, parents, [5, 6, 6, 7], seed, p_rows, q_rows, serial)
        assert path[0] == 0 and all(parents[b] == a for a, b in zip(path, path[1:]))
        counts[tokens[path[1]] if len(path) > 1 else terminal] += 1
    assert _chi2(counts, [_conditional(None)[t] for t in range(4)]) < CHI2_999[3]


def test_room_and_end_tokens_cut_the_path():
    p = {1: 1.0}
    path, terminal = walk([0, 1, 1, 1], [-1, 0, 1, 2], [0, 1, 2, 3], 7, [p] * 4, [p] * 4, [1] * 4, room=2)
    assert path == [0, 1] and terminal == 1
    path, terminal = walk([0, 1, 1], [-1, 0, 1], [0, 1, 2], 7, [p] * 3, [p] * 3, [1] * 3, eos=(1,))
    assert path == [0] and terminal == 1
    with pytest.raises(ValueError):
        walk([0, 1], [-1, 0], [0, 1], 7, [None, None], [None, None], [1, 1])

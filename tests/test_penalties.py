"""Presence/frequency/repetition penalties (tensorfold.engine.penalties): the OpenAI/vLLM rule, per verify row."""

import numpy as np
import pytest

from tensorfold.engine.penalties import History, Penalties, penalize

torch = pytest.importorskip("torch")


def _reference_row(row, reply, prompt, p):
    """vLLM apply_penalties on one float32 row: repetition over prompt|reply, then frequency and presence."""
    out = row.astype(np.float32).copy()
    counts = np.bincount(np.asarray(reply, dtype=np.int64), minlength=len(row)).astype(np.float32)
    in_prompt = np.zeros(len(row), bool)
    in_prompt[list(prompt)] = True
    if p.repetition != 1.0:
        hit = in_prompt | (counts > 0)
        r = np.float32(p.repetition)
        scaling = np.where(out[hit] > 0, np.float32(1) / r, r)      # vLLM: logits *= where(logits > 0, 1 / r, r)
        out[hit] = out[hit] * scaling
    out = out - np.float32(p.frequency) * counts
    return out - np.float32(p.presence) * (counts > 0).astype(np.float32)


POLICIES = [Penalties(presence=1.5), Penalties(frequency=0.7), Penalties(repetition=1.3),
            Penalties(presence=-0.5, frequency=0.25, repetition=0.8), Penalties(presence=2.0, frequency=-2.0,
                                                                                repetition=1.05)]


@pytest.mark.parametrize("p", POLICIES)
@pytest.mark.parametrize("tree", [False, True])
def test_rows_follow_the_reply_and_their_path(p, tree):
    rng = np.random.default_rng(3)
    vocab = 50
    prompt = [3, 9, 9, 17, 4]
    reply = [9, 22, 22, 5]
    history = History(p, prompt)
    history.extend(reply)
    # row 0 is the committed anchor (reply[-1]); rows 1.. are drafts, a chain or a tree
    tokens = [reply[-1], 22, 31, 22, 8, 9]
    parents = [-1, 0, 1, 0, 3, 3] if tree else None
    logits = torch.tensor(rng.normal(size=(len(tokens), vocab)) * 4, dtype=torch.float32)
    expected = []
    for row in range(len(tokens)):
        path, node = [], row
        while node > 0:
            path.append(tokens[node])
            node = parents[node] if tree else node - 1
        expected.append(_reference_row(logits[row].numpy(), reply + path, prompt, p))
    got = history.apply(logits.clone(), tokens, parents)
    assert got.dtype == torch.float32
    np.testing.assert_array_equal(got.numpy(), np.stack(expected))


def test_inputs_stay_untouched_when_copied_and_bf16_widens_first():
    p = Penalties(presence=1.0, repetition=1.2)
    history = History(p, [1, 2])
    history.extend([3])
    source = torch.randn(2, 8)
    kept = source.clone()
    out = history.apply(source, copy=True)
    assert torch.equal(source, kept) and out is not source
    half = source.bfloat16()
    widened = history.apply(half)
    assert widened.dtype == torch.float32 and torch.equal(half.float()[:, 0], widened[:, 0])
    in_place = source.clone()
    assert history.apply(in_place) is in_place


def test_continuation_tokens_past_the_request_prompt_count_as_reply():
    p = Penalties(presence=1.0, frequency=0.5, prompt_length=3)
    history = History(p, [4, 4, 6, 7, 7])
    assert history.count(4) == 0 and history.count(7) == 2 and history.count(6) == 0
    history.extend([7, 8])
    assert history.count(7) == 3 and history.count(8) == 1
    with pytest.raises(ValueError):
        History(Penalties(presence=1.0, prompt_length=9), [1, 2])


def test_candidate_scores_match_the_row_rule():
    p = Penalties(presence=0.5, frequency=0.25, repetition=1.4)
    prompt, reply = [5, 6], [6, 7, 7]
    history = History(p, prompt)
    history.extend(reply)
    rng = np.random.default_rng(0)
    full = (rng.normal(size=12) * 3).astype(np.float32)
    ids = np.array([[7, 0, 5, 11, 6]])
    want = _reference_row(full, reply + [11], prompt, p)[ids[0]]
    np.testing.assert_array_equal(history.candidates(full[ids], ids, [11])[0], want)


def test_penalty_ranges_and_inactive_history_are_refused():
    for bad in (dict(presence=2.5), dict(frequency=-3.0), dict(repetition=0.0), dict(repetition=float("nan"))):
        with pytest.raises(ValueError):
            Penalties(**bad)
    assert not Penalties().active and Penalties(repetition=1.1).active
    with pytest.raises(ValueError):
        History(Penalties(), [1])
    np.testing.assert_array_equal(penalize(np.array([2.0, -2.0]), np.array([0, 0]), np.array([True, True]),
                                           Penalties(repetition=2.0)), np.array([1.0, -4.0], np.float32))


@pytest.mark.parametrize("r", [1e-38, 1e-50, 1e39])
def test_extreme_repetition_never_makes_a_non_score(r):
    """Accepted values whose vLLM float32 product overflows (or is 0 * inf): finite scores saturate, zeros and -inf
    masks stay, so no request hands the sampler +inf or NaN (which would fail every stream sharing its round)."""
    p = Penalties(repetition=r)
    history = History(p, [1, 2, 3, 4])
    row = torch.tensor([[5.0, 6.0, 0.0, -4.0, float("-inf"), 2.0]])
    got = history.apply(row.clone())
    host = penalize(row.numpy(), np.zeros(6), np.isin(np.arange(6), [1, 2, 3, 4]), p)
    np.testing.assert_array_equal(got.numpy(), host)
    assert torch.isfinite(got[0, :4]).all() and got[0, 4] == float("-inf")
    assert (got[0, 0], got[0, 2], got[0, 5]) == (5.0, 0.0, 2.0)


def test_the_dflash2_tree_policy_penalizes_each_node_by_its_path():
    """A node's children are ranked by penalized candidate scores counting the reply and the node's own path."""
    from tensorfold.families.qwen3_5.cuda.draft_tree import best_first

    cands = np.array([[5, 6, 7, 8], [5, 6, 7, 8]])
    unary = np.array([[4.0, 3.0, 0.0, -1.0], [4.0, 3.5, 0.0, -1.0]])
    pred, succ, projected = np.zeros((16, 3)), np.zeros((16, 3)), np.zeros((2, 3))

    def first_child(tokens, parents):
        return tokens[parents.index(0)]

    plain = best_first(cands, unary, projected, pred, succ, 1, 8, None, 10)
    assert plain[0][0] == 5 and first_child(*plain[:2]) == 5
    history = History(Penalties(presence=2.0), [1])
    penalized = best_first(cands, unary, projected, pred, succ, 1, 8, None, 10, history)
    assert penalized[0][0] == 5 and first_child(*penalized[:2]) == 6      # 5 is on the child's path
    history.extend([5])                                                   # now in the reply: the root ranks 6 first
    assert best_first(cands, unary, projected, pred, succ, 1, 8, None, 10, history)[0][0] == 6

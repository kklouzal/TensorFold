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

"""Request-local additive scores and the actual filtered/rejected target law."""
from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest

from tensorfold.engine.exact_sampling import Sampling, choose, uniform
from tensorfold.engine.penalties import History, Penalties, bias_workspace, penalize
from tensorfold.engine.rejection import ChainPacket, ProbabilityRow, verify
from tensorfold.server.errors import RequestError
from tensorfold.server.request_options import parse_logit_bias

torch = pytest.importorskip("torch")


@pytest.fixture
def cpu_sampling(monkeypatch):
    from tests import test_rejection_target_rows

    test_rejection_target_rows.cpu_device_control.__wrapped__(monkeypatch)


@pytest.mark.parametrize("value", [[], False, 1, {"01": 1}, {"+1": 1}, {"-1": 1}, {"1.0": 1},
                                    {" 1": 1}, {"١": 1}, {1: 1}, {"8": 0}, {"8": 1},
                                    {"1": True}, {"1": "1"}, {"1": None}, {"1": float("nan")},
                                    {"1": float("inf")}, {"1": -101}, {"1": 101}, {"1": 10**200}])
def test_invalid_bias_is_rejected_at_the_request_boundary(value):
    with pytest.raises(RequestError, match="logit_bias"):
        parse_logit_bias(value, 8)


def test_bias_policy_is_immutable_canonical_and_keeps_positional_continuations():
    bias = parse_logit_bias({"7": -100, "0": 2.5, "3": 0}, 8)
    assert bias == ((0, 2.5), (7, -100.0))
    p = Penalties(1, .5, 1.2, 3, bias)
    assert p.bias is bias and p.active
    assert replace(p, prompt_length=5).bias == bias
    with pytest.raises(FrozenInstanceError):
        p.bias = ()
    assert not Penalties(bias=((1, 0),)).active
    assert parse_logit_bias(None, 8) == parse_logit_bias({}, 8) == parse_logit_bias({"1": 0}, 8) == ()
    for value in [[(1, 1)], ((True, 1),), ((2, 1), (1, 1)), ((1, 1), (1, 2)), ((2**64, 1),)]:
        with pytest.raises(ValueError, match="bias"):
            Penalties(bias=value)


@pytest.mark.parametrize("tree", [False, True])
def test_bias_combines_after_penalties_on_each_verify_path_and_preserves_masks(tree):
    from tests.test_penalties import _reference_row

    p = Penalties(.5, .25, 1.2, bias=((0, 2.5), (1, 100.0), (5, -100.0)))
    prompt, reply = [1, 2], [2, 3, 3]
    history = History(p, prompt)
    history.extend(reply)
    tokens, parents = [3, 5, 2, 5], [-1, 0, 0, 2] if tree else None
    raw = torch.tensor([[4., -float("inf"), 2., -1., .5, 6., 0.]] * 4)
    expected = []
    for row in range(4):
        path, at = [], row
        while at > 0:
            path.append(tokens[at])
            at = parents[at] if tree else at - 1
        scores = _reference_row(raw[row].numpy(), reply + path, prompt, p)
        for token, value in p.bias:
            scores[token] = np.float32(scores[token] + np.float32(value))
        expected.append(scores)
    got = history.apply(raw, tokens, parents, copy=True)
    np.testing.assert_array_equal(got.numpy(), expected)
    assert torch.isneginf(got[:, 1]).all()
    assert torch.equal(raw, torch.tensor([[4., -float("inf"), 2., -1., .5, 6., 0.]] * 4))


def test_full_host_rows_and_global_candidate_ids_obey_the_same_bias():
    p = Penalties(bias=((0, 10.0), (4, -3.0), (6, 2.0)))
    history = History(p, [1, 2])
    raw = np.array([0., 2., -1., 3., 8., 0., 1.], np.float32)
    expected = raw.copy()
    for token, bias in p.bias:
        expected[token] += np.float32(bias)
    ids = np.array([[6, 0, 3], [4, 1, 6]], np.int64)
    np.testing.assert_array_equal(history.candidates(raw[ids], ids), expected[ids])
    np.testing.assert_array_equal(penalize(raw, np.zeros(7), np.zeros(7, bool), p), expected)
    np.testing.assert_array_equal(history.apply(torch.from_numpy(raw[None].copy())).numpy()[0], expected)
    assert history.count(0) == 0  # bias is not a generated occurrence


def test_bias_only_after_commits_skips_identity_gather_but_retains_counts_and_paths(monkeypatch):
    history = History(Penalties(bias=((0, -5.0),)), [])
    history.extend([1, 1, 2])
    raw = torch.tensor([[4., 2., 1., 0.]] * 3)
    original, gathered = torch.Tensor.index_select, []
    def record(tensor, dimension, ids):
        gathered.append(ids.tolist())
        return original(tensor, dimension, ids)
    monkeypatch.setattr(torch.Tensor, "index_select", record)
    actual = history.apply(raw, [2, 3, 1], [-1, 0, 1], copy=True)
    assert gathered == [[0]]  # one required bias gather, no identity penalty gather/table path
    assert torch.equal(actual, torch.tensor([[-1., 2., 1., 0.]] * 3))
    assert torch.equal(raw, torch.tensor([[4., 2., 1., 0.]] * 3))
    assert history.count(1) == 2 and history.count(2) == 1 and history.count(3) == 0
    with pytest.raises(ValueError, match="one input token and parent"):
        history.apply(raw, [2, 3])
    with pytest.raises(ValueError, match="parents must precede"):
        history.apply(raw, [2, 3, 1], [-1, 2, 1])
    with pytest.raises(ValueError, match="tokens exceed"):
        history.apply(raw, [2, 99, 1])


def test_bias_changes_greedy_and_sampled_scores_before_filters_without_mutating_cache(cpu_sampling):
    from tensorfold.cuda.sampling import sample_rows

    raw = torch.tensor([[3., 2., 1., -2.]])
    saved = raw.clone()
    p = Penalties(bias=((0, -10.0), (2, 4.0)))
    biased = History(p, []).apply(raw, copy=True)
    expected = np.array([-7., 2., 5., -2.], np.float32)
    assert sample_rows(biased, [17], None) == [int(np.argmax(expected))]
    assert sample_rows(raw, [17], None) == [0] and torch.equal(raw, saved)
    for seed in [0, 11, 4196]:
        policy = Sampling(seed, .75, 2, .95, 0)
        ids = np.argsort(-expected, kind="stable")[:2]
        scaled = (expected[ids].astype(np.float64) - float(expected[ids[0]])) / .75
        mass = np.exp(scaled)
        mass /= mass.sum()
        keep = np.searchsorted(np.cumsum(mass), .95) + 1
        selected = ids[:keep]
        noise = -np.log(-np.log(uniform(seed, 17, selected)))
        wanted = int(selected[np.argmax(scaled[:keep] + noise)])
        assert sample_rows(biased, [17], policy) == [wanted]
        assert choose(biased.numpy()[0], np.arange(4), 17, policy) == wanted


def test_actual_target_p_rejects_an_unmodified_proposal_under_strong_bias(cpu_sampling):
    from tensorfold.cuda.sampling import effective_rows

    raw = torch.tensor([[2., 1.], [2., 1.]])
    policy = Sampling(11, .1, 2, 1, 0)
    q = ProbabilityRow([0], [1.])
    packet = ChainPacket((0,), (q,), 17)
    biased = History(Penalties(bias=((0, -100.0),)), []).apply(raw, copy=True)
    rows = effective_rows(biased, policy)
    # Independent FP64 target softmax: exp(-990) underflows. This specific
    # numerical case has zero mass; finite -100 is not a universal hard ban.
    assert np.exp(np.float64(-990)) == 0
    assert rows[0].probability(0) == 0 and rows[0].probability(1) == 1
    assert verify(packet, rows, policy.seed) == ([0], 1)
    assert q.probability(0) == 1 and packet.q_rows[0] is q
    assert torch.equal(raw, torch.tensor([[2., 1.], [2., 1.]]))


def test_workspace_bounds_maps_gathers_and_host_candidate_temporaries():
    assert bias_workspace(16, 262144)["device_bytes"] == (4 * 16 + 12) * 262144
    one, two = bias_workspace(4, 8), bias_workspace(4, 8, streams=2)
    assert two == {key: 2 * value for key, value in one.items()}
    assert one["device_bytes"] == 224
    assert one["host_bytes"] >= sum(np.empty((4, 8), dtype).nbytes for dtype in
                                   [np.int64, np.int64, np.int64, bool, np.float32, np.float32, np.float32]) + 12 * 8

"""Effective target laws reuse existing filtering; probability packets preserve readback ownership."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import sampling, speculative
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.rejection import draw, proposal_row


@pytest.fixture(autouse=True)
def cpu_device_control(monkeypatch):
    # Keep all policy/score/shape checks; substitute only the CUDA-device boundary in these CPU arithmetic controls.
    def validate(logits, positions, policy):
        if (not isinstance(logits, torch.Tensor) or logits.ndim != 2 or logits.is_cuda
                or logits.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64)
                or logits.shape[1] < 1 or len(positions) != len(logits)):
            raise ValueError("invalid controlled row geometry")
        sampling.validate_policy(positions, policy)

    monkeypatch.setattr(sampling, "validate_rows", validate)


@pytest.mark.parametrize("top_k,top_p,min_p", [(2, .95, 0), (0, .7, 0), (0, 1, .2), (3, .9, .1)])
def test_effective_rows_match_target_draws_on_cpu_controls(top_k, top_p, min_p):
    logits = torch.tensor([[0., -1., -2., -3.], [-2., 1., -4., 0.]])
    policy = Sampling(11, temperature=.7, top_k=top_k, top_p=top_p, min_p=min_p)
    rows = sampling.effective_rows(logits, policy)
    for seed in range(100):
        actual = sampling.sample_rows(logits, [7, 9], Sampling(seed, .7, top_k, top_p, min_p))
        assert [draw(row, seed, position, domain=0) for row, position in zip(rows, [7, 9])] == actual
    assert all(np.isclose(row.probs.sum(), 1) for row in rows)


def test_target_effective_ties_are_repaired_using_complete_global_id_order():
    logits = torch.zeros((1, 80))
    mapping = np.arange(80, dtype=np.int64)[::-1].copy()
    p = sampling.effective_rows(logits, Sampling(1, top_k=2, top_p=1), id_map=mapping)[0]
    assert p.ids.tolist() == [0, 1] and p.probs.tolist() == [.5, .5]


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_target_rows_refuse_bad_complete_score_domain(bad):
    logits = torch.ones(1, 32)
    if bad == -float("inf"):
        logits[:] = bad
    else:
        logits[0, -1] = bad
    with pytest.raises(ValueError):
        sampling.effective_rows(logits, Sampling(1, top_k=1))


def test_sampled_chain_has_one_completed_readback_and_detached_probability_ownership():
    logits = torch.tensor([[2., 1., 0.], [0., 2., 1.]])
    owner = object()
    chains = speculative.Chains(logits, [None], 2, owners=owner, sampled=True)
    original = chains.words

    class Transfer:
        calls = 0

        def cpu(self):
            self.calls += 1
            assert chains.owners is owner
            return original.cpu()

    transfer = Transfer()
    chains.words = transfer
    a = chains.sample(0, 2, Sampling(12), 10)
    b = chains.sample(0, 2, Sampling(12), 10)
    assert a.tokens == b.tokens and transfer.calls == 1 and chains.owners is chains.words is None
    logits[:] = float("nan")
    assert chains.sample(0, 2, Sampling(12), 10).tokens == a.tokens
    assert all(q.probability(t) > 0 for t, q in zip(a.tokens, a.q_rows))


def test_workspace_receipts_cover_packet_and_fp64_output_storage():
    n, vocab = 16, 248320
    charge = speculative.chain_workspace(n, vocab)
    assert charge["device_bytes"] >= n * (1 + 2 * 256) * 8
    assert charge["host_bytes"] >= n * (1 + 2 * 256) * 8 + n * 256 * 16
    complete = sampling.effective_workspace(n, vocab)
    bounded = sampling.effective_workspace(n, vocab, top_k=20)
    assert complete["device_bytes"] >= n * vocab * 16
    assert complete["host_bytes"] >= n * vocab * 16
    assert bounded["host_bytes"] < complete["host_bytes"]


@pytest.mark.parametrize("tied", [False, True])
def test_candidate_boundary_repair_matches_complete_stable_sort(tied):
    width, count = 1024, 256
    values = torch.arange(width, dtype=torch.float32)[None].expand(3, -1).clone()
    if tied:
        values[0] = 0
        values[1, 400:] = 1000  # a large tied retained boundary
    mapping = np.arange(width, dtype=np.int64)[::-1].copy() + (1 << 54)
    fields, ids, flags = sampling.candidate_fields(values, count, id_map=mapping)
    repaired = []

    def full_row(row):
        repaired.append(row)
        return values[row].numpy(), mapping

    result = sampling.canonical_candidates(fields.numpy(), ids.numpy(), count, width, full_row)
    for row in range(3):
        order = np.lexsort((mapping, -values[row].numpy()))[:count]
        np.testing.assert_array_equal(result[0][row], values[row].numpy()[order])
        np.testing.assert_array_equal(result[1][row], mapping[order])
    assert repaired == ([0, 1] if tied else []) and not flags.any()


def test_tied_sampled_packet_repairs_all_rows_before_retiring_source(monkeypatch):
    logits, owner = torch.zeros(4, 1024), object()
    packet = speculative.Chains(logits, [None, None], 2, owners=owner, sampled=True)
    original = speculative.canonical_candidates
    repairs = []

    def canonical(values, ids, count, width, complete):
        def read(row):
            assert packet.owners is owner and packet._source is logits
            repairs.append(row)
            return complete(row)
        return original(values, ids, count, width, read)

    monkeypatch.setattr(speculative, "canonical_candidates", canonical)
    policy = Sampling(12, top_k=0, top_p=1)
    together = packet.sample(1, 2, policy, 10)
    assert repairs == [0, 1, 2, 3] and packet.owners is packet.words is packet._source is None
    assert all(q.ids.tolist() == list(range(256)) for q in together.q_rows)
    solo = speculative.Chains(torch.zeros(2, 1024), [None], 2, sampled=True)
    monkeypatch.setattr(speculative, "canonical_candidates", original)
    assert together == solo.sample(0, 2, policy, 10)


@pytest.mark.parametrize("offset", [1e30, -1e30])
@pytest.mark.parametrize("top_k,top_p,min_p", [(0, 1., 0.), (2, .9, .2)])
def test_extreme_effective_support_matches_corrected_target_law(offset, top_k, top_p, min_p):
    logits = torch.full((2, 4), offset)
    baseline = torch.zeros_like(logits)
    policy = Sampling(1, .7, top_k, top_p, min_p)
    rows, expected = sampling.effective_rows(logits, policy), sampling.effective_rows(baseline, policy)
    assert rows == expected
    assert proposal_row(logits[0].numpy(), np.arange(4), policy) == expected[0]
    for seed in range(16):
        policy = Sampling(seed, .7, top_k, top_p, min_p)
        selected = sampling.sample_rows(logits, [17, 19], policy)
        assert selected == sampling.sample_rows(baseline, [17, 19], policy)
        assert selected == [draw(p, seed, pos, domain=0) for p, pos in zip(rows, [17, 19])]


def test_stochastic_confidence_keeps_extreme_offset_separate_from_log_mass():
    logits = torch.full((2, 4), 1e30)
    policy = Sampling(3, .7, 2, 1.)
    tokens, confidence = sampling.sample_rows(logits, [17, 19], policy, with_prob=True)
    assert tokens == sampling.sample_rows(torch.zeros_like(logits), [17, 19], policy)
    # The legacy FP32 log-mass is retained, but its addition to 1e30 must not round away.
    assert confidence == [float(np.exp(-float(np.float32(np.log(4)))))] * 2

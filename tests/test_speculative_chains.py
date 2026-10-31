"""Shared block-drafter readback/stop contracts under CPU controls; native packet checks live in tests/cuda."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import speculative
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.penalties import History, Penalties


class Readback:
    """Control the transfer completion boundary without replacing the packet decoder or finish policy."""

    def __init__(self, chains, *, fail=False):
        self.chains, self.value, self.owner, self.fail = chains, chains.words, chains.owners, fail
        self.calls = 0
        chains.words = self

    def cpu(self):
        self.calls += 1
        assert self.chains.owners is self.owner
        if self.fail:
            raise RuntimeError("readback did not complete")
        return self.value.cpu()


def _known_packet():
    chains = speculative.Chains(torch.zeros(4, 16), [None], 4)
    confidence = torch.tensor([0.25, 0.5, 1.25, 2.0], dtype=torch.float32)
    chains.words = torch.stack((torch.tensor([2, 4, 6, 8]), confidence.view(torch.int32).to(torch.int64)), dim=1)
    return chains


@pytest.mark.parametrize("stop,count", [(None, 4), (0.0, 1), (0.25, 1), (0.5, 2), (1.25, 3), (2.0, 4)])
@pytest.mark.parametrize("cap", [0, 1, 2, 4])
def test_stop_retains_the_exact_existing_strict_threshold_and_node_cap(stop, count, cap):
    ids, parents, scores = _known_packet().finish(0, cap, stop=stop)
    kept = min(cap, count)
    assert ids == [2, 4, 6, 8][:kept]
    assert parents == [-1, 0, 1, 2][:kept]
    assert scores == [0.25, 0.5, 1.25, 2.0][:kept]


def test_one_completed_readback_validates_all_streams_once_and_keeps_returned_lists_private(monkeypatch):
    chains = speculative.Chains(torch.arange(24, dtype=torch.float32).reshape(4, 6), [None, None], 2,
                                owners=object())
    readback = Readback(chains)
    first = chains.finish(0, 2)
    assert readback.calls == 1 and chains.words is chains.owners is None
    expected = chains.host[0][1], [-1, 0], chains.host[1][1]
    first[0][0], first[2][0] = -1, float("nan")
    monkeypatch.setattr(speculative.np, "isfinite", lambda *a, **k: pytest.fail("unchanged readback revalidated"))
    assert chains.finish(1, 2) == expected
    assert chains.finish(0, 2)[0] == [5, 5] and readback.calls == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("cap", [0, 1, 2])
def test_nonfinite_confidence_in_another_stream_or_beyond_the_cap_refuses_publication(bad, cap):
    owner = object()
    chains = speculative.Chains(torch.ones(4, 3), [None, None], 2, owners=owner)
    chains.words[3, 1] = torch.tensor(bad).view(torch.int32).to(torch.int64)
    readback = Readback(chains)
    with pytest.raises(ValueError, match="draft confidence scores must be finite"):
        chains.finish(0, cap, stop=0.0)
    assert chains.host is None and chains.owners is owner and chains.words is readback
    assert readback.calls == 1


def test_failed_transfer_keeps_the_actual_packet_and_kernel_owners():
    owner = object()
    chains = speculative.Chains(torch.ones(2, 3), [None], 2, owners=owner)
    readback = Readback(chains, fail=True)
    with pytest.raises(RuntimeError, match="readback did not complete"):
        chains.finish(0, 2)
    assert chains.host is None and chains.owners is owner and chains.words is readback


@pytest.mark.parametrize("penalized", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_malformed_plain_or_penalized_logits_do_not_publish_a_nonfinite_chain(bad, penalized):
    logits = torch.tensor([[1.0, 0.5, -float("inf")], [2.0, 1.0, 0.0]])
    if bad == -float("inf"):
        logits[1] = bad
    else:
        logits[1, 0] = bad
    history = History(Penalties(presence=1.0), [1]) if penalized else None
    chains = speculative.Chains(logits, [history], 2)
    with pytest.raises(ValueError, match="draft confidence scores must be finite"):
        chains.finish(0, 2, history)
    assert chains.host is None


def test_finite_rows_with_negative_infinity_masks_keep_their_proposals():
    logits = torch.tensor([[2.0, -float("inf"), 1.0], [-float("inf"), 0.0, 3.0]])
    chains = speculative.Chains(logits, [None], 2)
    ids, parents, confidence = chains.finish(0, 2)
    assert ids == [0, 2] and parents == [-1, 0]
    assert all(np.isfinite(confidence)) and confidence[0] <= confidence[1]


def test_mixed_penalty_histories_keep_path_aware_picks_and_the_shared_stop_rule():
    logits = torch.tensor([[3.0, 2.5, 0.0], [2.9, 2.4, 0.1], [3.0, 2.5, 0.0], [2.9, 2.4, 0.1]])
    history = History(Penalties(presence=2.0), [2])
    history.extend([0])
    chains = speculative.Chains(logits, [None, history], 2)
    assert chains.finish(0, 2)[0] == [0, 0]
    ids, parents, confidence = chains.finish(1, 2, history)
    assert ids == [1, 0] and parents == [-1, 0] and all(np.isfinite(confidence))
    assert chains.finish(1, 2, history, stop=0.0) == (ids[:1], [-1], confidence[:1])
    with pytest.raises(ValueError, match="penalties it was launched with"):
        chains.finish(1, 2)


def test_newly_computed_penalty_confidences_are_refused_if_nonfinite():
    class BrokenHistory:
        def candidates(self, values, ids, path=()):
            return np.full_like(values, np.nan)

    values, ids = np.array([[1.0, 0.0]], dtype=np.float32), np.array([[0, 1]], dtype=np.int64)
    with pytest.raises(ValueError, match="draft confidence scores must be finite"):
        speculative.penalized(values, ids, BrokenHistory())


@pytest.mark.parametrize("top_k", [1, 20, 64, 255, 256])
@pytest.mark.parametrize("kind", ["distinct", "all_tied", "boundary", "old_boundary", "masked", "positive", "negative"])
def test_policy_bound_q_is_byte_identical_to_full256_support(top_k, kind):
    """Independent full-row ranking also checks both packet paths, not only their token choices."""
    from tensorfold.engine.rejection import proposal_row

    logits = torch.randn(4, 2049, generator=torch.Generator().manual_seed(62))
    if kind == "all_tied":
        logits.zero_()
    elif kind == "boundary":
        logits.zero_()
        logits[:, :top_k - 1] = 3
        logits[:, top_k - 1:top_k + 333] = 2
    elif kind == "old_boundary":
        logits.zero_()
        logits[:, :64] = 3
        logits[:, 64:1500] = 2
    elif kind == "masked":
        logits[:, 100:] = -float("inf")
    elif kind in ("positive", "negative"):
        logits.fill_(1e30 if kind == "positive" else -1e30)
    policy = Sampling(19, .7, top_k, .95, .03)
    bounded = speculative.Chains(logits, [None, None], 2, sampled=True, sampling=policy)
    reference = speculative.Chains(logits, [None, None], 2, sampled=True)
    assert bounded.words.shape == (4, 1 + 2 * (top_k + 32))
    assert reference.words.shape == (4, 1 + 2 * 288)
    raw = logits.numpy()
    for stream in (1, 0):
        packet = bounded.sample(stream, 2, policy, 17 + stream * 100)
        assert packet == reference.sample(stream, 2, policy, 17 + stream * 100)
        assert packet == bounded.sample(stream, 2, Sampling(19, .7, top_k, .95, .03), 17 + stream * 100)
        for j, q in enumerate(packet.q_rows):
            row = raw[stream * 2 + j]
            ids = np.lexsort((np.arange(len(row)), -row))[:256]
            expected = proposal_row(row[ids], ids, policy)
            assert q.ids.tobytes() == expected.ids.tobytes()
            assert q.probs.tobytes() == expected.probs.tobytes()
            assert not q.ids.flags.writeable and not q.probs.flags.writeable
    assert bounded._source is bounded.words is bounded.owners is None


@pytest.mark.parametrize("top_k", [0, 257, 1024])
def test_unselected_top_k_policies_retain_full256_packet(top_k):
    chains = speculative.Chains(torch.zeros(2, 1024), [None], 2, sampled=True,
                                sampling=Sampling(1, top_k=top_k))
    assert chains._proposal_count == 256 and chains._bound_top_k is None
    assert chains.words.shape == (2, 577)


def test_any_active_history_keeps_full256_for_all_streams():
    history = History(Penalties(presence=1.0), [0])
    logits = torch.arange(2048, dtype=torch.float32).reshape(2, 1024)
    policy = Sampling(31, top_k=20)
    bounded = speculative.Chains(logits, [None, history], 1, sampled=True, sampling=policy)
    reference = speculative.Chains(logits, [None, history], 1, sampled=True)
    assert bounded._proposal_count == 256 and bounded._bound_top_k is None
    assert bounded.words.shape == reference.words.shape == (2, 577)
    for row, penalties in enumerate((None, history)):
        assert bounded.sample(row, 1, policy, 29, penalties) == reference.sample(row, 1, policy, 29, penalties)


@pytest.mark.parametrize("changed", [0, 1, 21, 256, 512])
def test_changed_bound_top_k_refuses_before_any_readback(changed):
    owner = object()
    policy = Sampling(3, top_k=20)
    chains = speculative.Chains(torch.zeros(2, 1024), [None], 2, owners=owner, sampled=True, sampling=policy)
    readback = Readback(chains, fail=True)
    with pytest.raises(ValueError, match="construction top_k"):
        chains.sample(0, 2, Sampling(3, top_k=changed), 19)
    assert readback.calls == 0 and chains.host is None and chains.owners is owner


def test_other_sampling_fields_can_change_after_identical_top_k_filtering():
    logits = torch.randn(3, 1024, generator=torch.Generator().manual_seed(3))
    packet = speculative.Chains(logits, [None], 3, sampled=True, sampling=Sampling(4, top_k=20))
    for policy in (Sampling(4, .7, 20, .95), Sampling(67, 1.3, 20, .6, .1)):
        reference = speculative.Chains(logits, [None], 3, sampled=True).sample(0, 3, policy, 29)
        assert packet.sample(0, 3, policy, 29) == reference


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_bound_packet_preserves_whole_source_domain_failure_and_owners(bad):
    owner = object()
    logits = torch.ones(3, 1024)
    logits[-1, -1] = bad
    if bad == -float("inf"):
        logits[-1] = bad
    policy = Sampling(3, top_k=20)
    packet = speculative.Chains(logits, [None], 3, owners=owner, sampled=True, sampling=policy)
    with pytest.raises(ValueError):
        packet.sample(0, 0, policy, 19)
    assert packet.host is None and packet.owners is owner and packet._source is logits


def test_construction_policy_is_not_accepted_by_greedy_or_foreign_sampling():
    logits = torch.zeros(2, 1024)
    with pytest.raises(ValueError, match="sampled-only packet"):
        speculative.Chains(logits, [None], 2, sampling=Sampling(1))
    with pytest.raises(ValueError, match="sampled-only packet"):
        speculative.Chains(logits, [None], 2, sampled=True, sampling=object())


def test_bound_tie_repair_holds_source_and_foreign_owners_until_real_readback(monkeypatch):
    logits, owner = torch.zeros(3, 2049), object()
    policy = Sampling(5, top_k=20)
    chains = speculative.Chains(logits, [None], 3, owners=owner, sampled=True, sampling=policy)
    transfer = Readback(chains)
    canonical, repaired = speculative.canonical_candidates, []

    def checked(values, ids, count, width, full_row):
        def complete(row):
            assert chains._source is logits and chains.owners is owner
            repaired.append(row)
            return full_row(row)
        return canonical(values, ids, count, width, complete)

    monkeypatch.setattr(speculative, "canonical_candidates", checked)
    packet = chains.sample(0, 3, policy, 17)
    assert repaired == [0, 1, 2] and transfer.calls == 1
    assert chains._source is chains.owners is chains.words is None
    assert len(packet.tokens) == 3

"""Exact provisional guard geometry; no model, CUDA execution or performance claim."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import speculative
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.penalties import History, Penalties
from tensorfold.engine.rejection import ChainPacket, draw, proposal_row


def expected(logits, histories, rows, policy, desired):
    """Independent complete-row lexsort, then the existing penalty/q/draw contract."""
    result = []
    for stream, history in enumerate(histories):
        first, tokens, distributions = 17 + stream * 100, [], []
        for j in range(rows):
            scores = logits[stream * rows + j].float().numpy()
            ids = np.lexsort((np.arange(len(scores)), -scores))[:desired]
            values = scores[ids]
            if history is not None:
                values = history.candidates(values[None], ids[None], tokens)[0]
            q = proposal_row(values, ids, policy)
            tokens.append(draw(q, policy.seed, first + j))
            distributions.append(q)
        result.append(ChainPacket(tuple(tokens), tuple(distributions), first))
    return result


@pytest.mark.parametrize("guard", [1, 8, 32])
@pytest.mark.parametrize("mode", ["bound64", "unbound256", "penalized256"])
@pytest.mark.parametrize("kind", ["distinct", "plateau7", "plateau31", "spanning", "all_tied", "masked", "bf16"])
def test_guard_keeps_exact_desired_pool_before_penalties_and_normalization(monkeypatch, guard, mode, kind):
    rows, width = 2, 4099
    policy = Sampling(53, .7, 0 if mode == "unbound256" else 64, .95, .03)
    history = History(Penalties(presence=1.5, frequency=.3, repetition=1.2), [0, 1, 2, 63])
    histories = [None, history if mode == "penalized256" else None]
    desired = 64 if mode == "bound64" else 256
    logits = torch.randn(4, width, generator=torch.Generator().manual_seed(81))
    if kind.startswith("plateau") or kind == "spanning":
        plateau = int(kind[7:]) if kind.startswith("plateau") else 1000
        logits.zero_()
        logits[:, :desired - 1] = 3
        logits[:, desired - 1:desired - 1 + plateau] = 2
    elif kind == "all_tied":
        logits.zero_()
    elif kind == "masked":
        logits[:, 100:] = -float("inf")
    elif kind == "bf16":
        logits = logits.bfloat16().float()
    want = expected(logits, histories, rows, policy, desired)
    with monkeypatch.context() as scope:
        scope.setattr(speculative, "_PROPOSAL_GUARD", guard)
        packet = speculative.Chains(logits, histories, rows, sampled=True, sampling=policy)
        assert packet._proposal_count == desired
        assert packet._proposal_width == min(width, desired + guard)
        assert packet.words.shape == (4, 1 + 2 * min(width, desired + guard))
        for stream in (1, 0):
            actual = packet.sample(stream, rows, policy, 17 + stream * 100, histories[stream])
            assert actual == want[stream]
            for left, right in zip(actual.q_rows, want[stream].q_rows):
                assert left.ids.tobytes() == right.ids.tobytes()
                assert left.probs.tobytes() == right.probs.tobytes()
            assert actual == packet.sample(stream, rows, policy, 17 + stream * 100, histories[stream])
        assert packet._source is packet.words is packet.owners is None


@pytest.mark.parametrize("guard", [1, 8, 32])
@pytest.mark.parametrize("plateau", [7, 31, 1000])
def test_guard_avoids_only_provably_contained_ties_and_preserves_repair_owners(monkeypatch, guard, plateau):
    desired, rows = 64, 3
    logits = torch.zeros(rows, 4099)
    logits[:, :desired - 1] = 3
    logits[:, desired - 1:desired - 1 + plateau] = 2
    policy, owner = Sampling(51, top_k=desired), object()
    canonical, repaired = speculative.canonical_candidates, []
    packet = None

    def checked(values, ids, count, width, full_row):
        assert count == desired and values.shape[1] == desired + guard
        def complete(row):
            assert packet._source is logits and packet.owners is owner
            repaired.append(row)
            return full_row(row)
        return canonical(values, ids, count, width, complete)

    with monkeypatch.context() as scope:
        scope.setattr(speculative, "_PROPOSAL_GUARD", guard)
        scope.setattr(speculative, "canonical_candidates", checked)
        packet = speculative.Chains(logits, [None], rows, owners=owner, sampled=True, sampling=policy)
        actual = packet.sample(0, rows, policy, 17)
        assert repaired == (list(range(rows)) if plateau > guard else [])
        assert actual == expected(logits, [None], rows, policy, desired)[0]
        assert packet._source is packet.words is packet.owners is None


@pytest.mark.parametrize("guard", [1, 8, 32])
@pytest.mark.parametrize("width,top_k", [(23, 20), (23, 0), (256, 0), (257, 0)])
def test_complete_vocabulary_clipping_keeps_the_same_q(monkeypatch, guard, width, top_k):
    policy = Sampling(19, .8, top_k, .95)
    logits = torch.zeros(2, width)
    desired = top_k or 256
    with monkeypatch.context() as scope:
        scope.setattr(speculative, "_PROPOSAL_GUARD", guard)
        packet = speculative.Chains(logits, [None], 2, sampled=True, sampling=policy)
        assert packet._proposal_width == min(width, desired + guard)
        assert packet.sample(0, 2, policy, 17) == expected(logits, [None], 2, policy, desired)[0]


def test_workspace_charges_the_actual_extra_words_and_decoded_columns(monkeypatch):
    rows, width, charges, payloads = 4, 8192, {}, {}
    for guard in (1, 8, 32):
        with monkeypatch.context() as scope:
            scope.setattr(speculative, "_PROPOSAL_GUARD", guard)
            charge = speculative.chain_workspace(rows, width)
            packet = speculative.Chains(torch.zeros(rows, width), [None], rows, sampled=True)
            words = packet.words.numel() * packet.words.element_size()
            decoded = rows * packet._proposal_width * np.dtype(np.float32).itemsize
            assert words == {1: 16480, 8: 16928, 32: 18464}[guard]
            packet.sample(0, rows, Sampling(13, top_k=0), 17)
            charges[guard], payloads[guard] = charge, words + decoded
    for guard in (8, 32):
        # Full-vocabulary row scratch dominates for this geometry; immutable D
        # and headers are fixed, leaving only actual raw/readback columns.
        assert charges[guard]["host_bytes"] - charges[1]["host_bytes"] == payloads[guard] - payloads[1]
        assert charges[guard]["device_bytes"] > charges[1]["device_bytes"]
    with monkeypatch.context() as scope:
        scope.setattr(speculative, "_PROPOSAL_GUARD", 1)
        small = speculative.chain_workspace(rows, 23)
        scope.setattr(speculative, "_PROPOSAL_GUARD", 32)
        assert speculative.chain_workspace(rows, 23) == small


@pytest.mark.parametrize("guard", [1, 8, 32])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_guard_does_not_hide_full_source_failure_or_release_unread_owners(monkeypatch, guard, bad):
    logits, owner, policy = torch.ones(3, 4099), object(), Sampling(3, top_k=64)
    logits[-1, -1] = bad
    if bad == -float("inf"):
        logits[-1] = bad
    with monkeypatch.context() as scope:
        scope.setattr(speculative, "_PROPOSAL_GUARD", guard)
        packet = speculative.Chains(logits, [None], 3, owners=owner, sampled=True, sampling=policy)
        with pytest.raises(ValueError):
            packet.sample(0, 0, policy, 17)
        assert packet.host is None and packet.owners is owner and packet._source is logits

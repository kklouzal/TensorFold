"""Shared block-drafter readback/stop contracts under CPU controls; native packet checks live in tests/cuda."""

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tensorfold.cuda import speculative
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

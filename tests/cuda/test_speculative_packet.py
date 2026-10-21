"""Block-drafter chains on CUDA: argmax proposals, and penalized ones for streams with penalties, in one readback."""

from contextlib import nullcontext

import numpy as np
import pytest
import torch

from tensorfold.cuda import speculative
from tensorfold.engine.exact_sampling import Sampling
from tensorfold.engine.penalties import History, Penalties

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def test_mixed_streams_propose_their_argmax_or_penalized_argmax():
    vocab, rows = 4099, 5
    generator = torch.Generator(device="cuda").manual_seed(7)
    logits = torch.randn((2 * rows, vocab), generator=generator, device="cuda") * 3
    top = logits.argmax(-1).cpu().tolist()
    history = History(Penalties(presence=2.0, frequency=1.0), [top[rows], 11])
    history.extend(top[rows:rows + 2])
    chains = speculative.Chains(logits, [None, history], rows)
    assert speculative.Chains(logits, [None, None], rows).words.shape == (2 * rows, 2)
    ids, parents, scores = chains.finish(0, rows)
    assert ids == top[:rows] and parents == list(range(-1, rows - 1)) and all(a <= b for a, b in zip(scores, scores[1:]))
    host = logits[rows:].cpu().numpy()
    want = []
    for row in host:
        want.append(int(np.argmax(history.candidates(row[None], np.arange(vocab)[None], want)[0])))
    assert chains.finish(1, rows, history)[0] == want != top[rows:]
    assert chains.finish(0, rows, stop=0.0)[0] == top[:1]
    assert chains.finish(1, rows, history, stop=0.0)[0] == want[:1]
    with pytest.raises(ValueError):
        chains.finish(1, rows)
    with pytest.raises(ValueError):
        speculative.Chains(logits, [None], rows)


@pytest.mark.parametrize("penalized", [False, True])
@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_device_confidence_refuses_the_whole_packet_before_cutting(bad, penalized):
    logits = torch.ones(4, 7, device="cuda")
    if bad == -float("inf"):
        logits[3] = bad
    else:
        logits[3, 1] = bad
    history = History(Penalties(presence=1.0), [1]) if penalized else None
    owner = torch.ones(8, device="cuda")
    chains = speculative.Chains(logits, [None, history], 2, owners=owner)
    with pytest.raises(ValueError, match="draft confidence scores must be finite"):
        chains.finish(0, 1, stop=0.0)
    assert chains.host is None and chains.owners is owner


@pytest.mark.parametrize("nondefault", [False, True])
def test_a_successful_readback_finishes_packet_work_before_releasing_source_owners(nondefault):
    stream = torch.cuda.Stream() if nondefault else None
    with torch.cuda.stream(stream) if nondefault else nullcontext():
        owner = torch.ones(8, device="cuda")
        chains = speculative.Chains(torch.randn(6, 19, device="cuda"), [None, None], 3, owners=owner)
        done = torch.cuda.Event()
        done.record()
        assert chains.owners is owner
        assert len(chains.finish(0, 3)[0]) == 3
        assert done.query() and chains.words is chains.owners is None
        assert len(chains.finish(1, 3)[0]) == 3


@pytest.mark.parametrize("top_k", [1, 20, 64, 256])
@pytest.mark.parametrize("kind", ["random", "all_tied", "boundary", "masked", "extreme"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("guard", [1, 8, 32])
def test_bound_native_q_bytes_and_same_config_replay_match_full256(monkeypatch, top_k, kind, dtype, guard):
    from tensorfold.engine.rejection import proposal_row

    monkeypatch.setattr(speculative, "_PROPOSAL_GUARD", guard)
    logits = torch.randn((6, 4099), generator=torch.Generator(device="cuda").manual_seed(91),
                         device="cuda", dtype=dtype)
    if kind == "all_tied":
        logits.zero_()
    elif kind == "boundary":
        logits.zero_()
        logits[:, :top_k - 1] = 3
        logits[:, top_k - 1:top_k + 400] = 2
    elif kind == "masked":
        logits[:, 150:] = -float("inf")
    elif kind == "extreme":
        logits.fill_(1e30)
    policy = Sampling(31, .7, top_k, .95, .02)
    owner = object()
    bounded = speculative.Chains(logits, [None, None], 3, owners=owner, sampled=True, sampling=policy)
    reference = speculative.Chains(logits, [None, None], 3, sampled=True)
    assert bounded.words.shape == (6, 1 + 2 * (top_k + guard))
    raw = logits.float().cpu().numpy()
    for stream in (1, 0):
        first = 17 + 100 * stream
        packet = bounded.sample(stream, 3, policy, first)
        assert packet == reference.sample(stream, 3, policy, first)
        assert packet == bounded.sample(stream, 3, Sampling(31, .7, top_k, .95, .02), first)
        for row, actual in enumerate(packet.q_rows):
            scores = raw[stream * 3 + row]
            ids = np.lexsort((np.arange(len(scores)), -scores))[:256]
            expected = proposal_row(scores[ids], ids, policy)
            assert actual.ids.tobytes() == expected.ids.tobytes()
            assert actual.probs.tobytes() == expected.probs.tobytes()
    assert bounded._source is bounded.words is bounded.owners is None


@pytest.mark.parametrize("guard", [1, 8, 32])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_guard_packet_and_full_tie_repair_fit_the_matching_workspace(monkeypatch, guard, dtype):
    monkeypatch.setattr(speculative, "_PROPOSAL_GUARD", guard)
    logits = torch.zeros((4, 8192), dtype=dtype, device="cuda")
    charge = speculative.chain_workspace(4, 8192)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    packet = speculative.Chains(logits, [None], 4, sampled=True)
    q = packet.sample(0, 4, Sampling(17, top_k=0), 19)
    torch.cuda.synchronize()
    assert torch.cuda.max_memory_allocated() - before <= charge["device_bytes"]
    assert sum(row.ids.nbytes + row.probs.nbytes for row in q.q_rows) <= charge["host_bytes"]
    assert packet._proposal_width == 256 + guard
    assert packet._source is packet.words is packet.owners is None

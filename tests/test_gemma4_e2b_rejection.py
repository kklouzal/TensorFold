"""Full-q sampling at a strict logical target-cache boundary."""
from types import SimpleNamespace
import pytest

from tensorfold.engine.exact_sampling import Sampling
from test_gemma_cuda_engine import _engine


@pytest.mark.parametrize("budget", [2, 3, 4, 8, 9])
@pytest.mark.parametrize("grammar", [False, True])
def test_sampled_target_prefix_fits_context_without_changing_q_packet(monkeypatch, budget, grammar):
    sampling = Sampling(seed=173, temperature=.8, top_k=0, top_p=.91, min_p=.01)
    prompt = ([1, 3, 4, 8] * 6)[:22]
    reference = _engine(monkeypatch)
    reference.context_window = 96
    expected = []
    reference.generate(prompt, 35, sampling, lambda ids: expected.extend(ids), stop_eos=False)

    engine = _engine(monkeypatch)
    engine._clip_sampled_verification_to_room = True
    engine.context_window = len(prompt) + budget
    forward, propose = engine.target.forward, engine.head.propose
    proposals, packets, launches, output = [], [], [], []

    def full_proposal(state, anchor, count, history=None, *, sampled=False, sampling=None):
        proposals.append((count, sampled))
        chains = propose(state, anchor, count, history, sampled=sampled, sampling=sampling)

        def sample(*args):
            packet = chains.sample(*args)
            packets.append(len(packet.tokens))
            return packet

        return SimpleNamespace(sample=sample)

    def bounded_forward(tokens, state, **kwargs):
        if not kwargs.get("prompt", False):
            launches.append((len(tokens), budget - len(output)))
            assert state.position + len(tokens) <= engine.context_window
        return forward(tokens, state, **kwargs)

    class Constraint:
        finished = False

        def mask(self, logits, window=None):
            if window is not None:
                assert len(logits) == len(window.tokens)

        def window(self, tokens, parents):
            assert len(tokens) <= budget - len(output)  # truncate BEFORE grammar expansion
            return SimpleNamespace(tokens=tokens)

        def advance(self, tokens):
            pass

    engine.target.forward, engine.head.propose = bounded_forward, full_proposal
    engine.generate(prompt, budget, sampling, lambda ids: output.extend(ids), stop_eos=False,
                    constraint=Constraint() if grammar else None)
    assert output == expected[:budget]
    assert proposals and all(item == (7, True) for item in proposals)
    assert packets and all(count == 7 for count in packets)
    assert launches and all(rows <= room for rows, room in launches)
    assert engine._state.tokens == engine._draft.taps == prompt + output[:-1]

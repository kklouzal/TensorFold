"""Engine boundaries and exact accepted-path bookkeeping, without model weights."""

from contextlib import nullcontext
import threading
from types import SimpleNamespace

import pytest

from tensorfold.families.gemma4.cuda.engine import Gemma4Engine, _accepted, _prefill_end, _prompt_ids


@pytest.mark.parametrize("prompt", [[], [True], [1.0], [-1], [16], "1"])
def test_prompt_boundary_rejects_invalid_token_identity(prompt):
    with pytest.raises(ValueError, match="prompt token"):
        _prompt_ids(prompt, 16)


@pytest.mark.parametrize("option", [{"streams": 2}, {"drafter_bits": 8}, {"depth": 0}, {"depth": 2}])
def test_unsupported_settings_stop_before_any_torch_or_weight_load(option, tmp_path):
    with pytest.raises(ValueError):
        Gemma4Engine(tmp_path, **option)


def test_visual_chunk_keeps_whole_bidirectional_block():
    assert _prefill_end(0, 600, 128, ((100, 380),)) == 100
    assert _prefill_end(100, 600, 128, ((100, 380),)) == 380
    assert _prefill_end(380, 600, 128, ((100, 380),)) == 508
    with pytest.raises(ValueError, match="inside a visual block"):
        _prefill_end(101, 600, 128, ((100, 380),))


def test_acceptance_stops_at_first_mismatch_room_or_eos():
    assert _accepted([4, 9, 7], [2, 4, 6], 9, ()) == [4, 9]
    assert _accepted([4, 6, 7], [2, 4, 6], 2, ()) == [4, 6]
    assert _accepted([4, 6, 7], [2, 4, 6], 9, (6,)) == [4, 6]


def _engine(monkeypatch, *, with_head=True):
    torch = pytest.importorskip("torch")
    import numpy as np
    from tensorfold.engine.exact_sampling import choose

    class State:
        def __init__(self):
            self.tokens = []

        @property
        def position(self):
            return len(self.tokens)

        def rollback(self, position):
            del self.tokens[position:]

        def snapshot(self):
            return tuple(self.tokens)

        def restore(self, snap):
            self.tokens = list(snap)

    def scores(total):
        # A unique top, multiple plausible sampled choices, no accidental EOS.
        result = torch.tensor([-3.0 - ((i - total * 7) % 19) / 11 for i in range(19)])
        result[total * 7 % 19] = 2.0
        return result

    class Target:
        config = SimpleNamespace(vocab=19)
        device = torch.device("cpu")

        def make_state(self, *args, **kwargs):
            return State()

        def project_logits(self, hidden):
            return torch.stack([scores(int(row[1])) for row in hidden])

        def forward(self, tokens, state, *, output_logits=True, **kwargs):
            rows = []
            for token in tokens:
                state.tokens.append(token)
                rows.append([token, sum(state.tokens)])
            hidden = torch.tensor(rows, dtype=torch.bfloat16)
            return SimpleNamespace(hidden=hidden, logits=self.project_logits(hidden) if output_logits else None,
                                   taps={1: hidden})

    class Head:
        layers = 1
        tap_layers = (1,)

        def add_taps_streams(self, snaps, taps):
            result = []
            for snap, value in zip(snaps, taps):
                before = [] if snap[0][0] is None else snap[0][0]
                after = [*before, *[int(v) for v in value[:, 0]]]
                result.append(([after], [None], len(after), len(after)))
            return result

        def launch_blocks(self, snaps, pending, limit, samplings=None):
            before = snaps[0][0][0]
            total = sum(before) + pending[0]
            proposals = []
            for _ in range(limit):
                token = int(scores(total).argmax())
                proposals.append(token)
                total += token
            # Force rejection after a correct prefix to expose rollback/tap bugs.
            if len(proposals) > 2:
                proposals[2] = (proposals[2] + 1) % 19
            return [proposals]

        def finish_tree(self, launched, *args):
            return launched, list(range(-1, len(launched) - 1)), []

    def sample(logits, positions, sampling):
        if sampling is None:
            return logits.argmax(-1).tolist()
        return [choose(row.numpy(), np.arange(19), position, sampling)
                for row, position in zip(logits.float(), positions)]

    engine = Gemma4Engine.__new__(Gemma4Engine)
    engine.target = Target()
    engine.head = Head() if with_head else None
    engine.context_window, engine.max_rows, engine.eos = 400, 8, (18,)
    engine.vision = None
    engine._closed, engine._prefix, engine._owner = False, None, None
    engine._run_lock, engine._closed_lock, engine._prefix_lock = threading.Lock(), threading.Lock(), threading.Lock()
    engine._stream = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr("tensorfold.cuda.sampling.sample_rows", sample)
    return engine


@pytest.mark.parametrize("sampled", [False, True])
def test_dflash_rejections_and_prompt_resume_equal_serial(monkeypatch, sampled):
    engine = _engine(monkeypatch)
    from tensorfold.engine.exact_sampling import Sampling
    sampling = Sampling(seed=912, temperature=.7, top_k=7, top_p=.9) if sampled else None

    def run(prompt, draft):
        output = []
        stats = engine.generate(prompt, 35, sampling, lambda tokens: output.extend(tokens),
                                draft=draft, stop_eos=False)
        return output, stats

    serial, _ = run([3, 1, 4, 1, 5], False)
    drafted, stats = run([3, 1, 4, 1, 5], True)
    assert drafted == serial
    assert stats["cached"] == 5
    assert stats["drafted"] > 0
    assert stats["accepted"] < stats["drafted"]
    resumed, resumed_stats = run([3, 1, 4, 1, 5, 9, 2], True)
    fresh_engine = _engine(monkeypatch)
    fresh = []
    fresh_engine.generate([3, 1, 4, 1, 5, 9, 2], 35, sampling, lambda tokens: fresh.extend(tokens),
                          draft=False, stop_eos=False)
    assert resumed == fresh
    assert resumed_stats["cached"] == 5
    assert engine._prefix.tokens == (3, 1, 4, 1, 5, 9, 2)


@pytest.mark.parametrize("policy", [dict(temperature=1.0, top_k=7, top_p=.95), dict(temperature=.6, top_k=0, top_p=.9),
                                    dict(temperature=1.3, top_k=0, top_p=1.0, min_p=.05)])
def test_coupled_proposals_of_an_exact_drafter_are_all_accepted(monkeypatch, policy):
    """A drafter equal to the target proposes under the target's keyed noise at anchor + 1 + j: every proposal is the
    target's own sampled token, so each round accepts its whole block and output equals serial decoding."""
    torch = pytest.importorskip("torch")
    from tensorfold.cuda import speculative
    from tensorfold.engine.exact_sampling import Sampling
    engine = _engine(monkeypatch)
    monkeypatch.undo()                      # the real CUDA sampler's host rule, on CPU scores
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr("tensorfold.cuda.sampling.validate_rows", lambda logits, positions, sampling: None)
    sampling = Sampling(seed=4321, **policy)

    def scores(position):                   # the target's scores for the token at ``position``, whatever came before
        return torch.randn(19, generator=torch.Generator().manual_seed(position)) * 2

    def forward(tokens, state, *, output_logits=True, **kwargs):
        rows = []
        for token in tokens:
            rows.append([token, len(state.tokens)])
            state.tokens.append(token)
        hidden = torch.tensor(rows, dtype=torch.float32)
        return SimpleNamespace(hidden=hidden, logits=project(hidden) if output_logits else None, taps={1: hidden})

    def project(hidden):
        return torch.stack([scores(int(row[1]) + 1) for row in hidden])

    engine.target.forward, engine.target.project_logits = forward, project
    seen = []

    class Exact(type(engine.head)):
        def launch_blocks(self, snaps, pending, limit, samplings=None):
            first = snaps[0][3] + 1                 # the committed context ends at the anchor
            logits = torch.stack([scores(first + j) for j in range(limit)])
            return [(speculative.launch(logits, samplings, limit), first, logits.argmax(-1).tolist())]

        def finish_tree(self, launched, context_length, limit, sampling):
            packet, first, greedy = launched
            seen.append((context_length, first))
            ids = (speculative.chain(speculative.read(packet), 0, limit, context_length, sampling)[0]
                   if packet is not None else greedy)
            return ids, list(range(-1, limit - 1)), []

    engine.head = Exact()

    def run(draft):
        output = []
        stats = engine.generate([3, 1, 4, 1, 5], 40, sampling, lambda tokens: output.extend(tokens),
                                draft=draft, stop_eos=False)
        return output, stats

    serial, _ = run(False)
    drafted, stats = run(True)
    assert drafted == serial and len(drafted) == 40
    assert stats["drafted"] > 0 and stats["accepted"] == stats["drafted"]
    assert seen and all(length == first for length, first in seen)


def test_callback_stop_and_failure_do_not_publish_generated_context(monkeypatch):
    engine = _engine(monkeypatch)
    prompt = [4, 2]
    seen = []
    engine.generate(prompt, 35, None, lambda tokens: seen.extend(tokens) or True)
    assert len(seen) == 1
    assert engine._prefix.state == tuple(prompt)
    with pytest.raises(RuntimeError, match="callback failed"):
        engine.generate(prompt, 35, None, lambda tokens: (_ for _ in ()).throw(RuntimeError("callback failed")))
    assert engine._owner is None
    fresh = []
    engine.generate(prompt, 3, None, lambda tokens: fresh.extend(tokens), stop_eos=False)
    assert fresh[:1] == seen
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate(prompt, 3, None, lambda tokens: None)


def test_cancelled_prefill_does_not_replace_previous_prefix(monkeypatch):
    engine = _engine(monkeypatch)
    engine.generate([1, 2], 1, None, lambda tokens: None)
    before = engine._prefix
    from tensorfold.server.cancellation import RequestCancelled
    with pytest.raises(RequestCancelled):
        engine.generate([3] * 260, 1, None, lambda tokens: None, progress=lambda *args: True)
    assert engine._prefix is before


def test_reentrant_callback_is_rejected_without_stranding_engine(monkeypatch):
    engine = _engine(monkeypatch)
    def reenter(tokens):
        engine.generate([1, 2], 1, None, lambda tokens: None)
    with pytest.raises(RuntimeError, match="reenter"):
        engine.generate([1, 2], 1, None, reenter)
    assert engine._owner is None
    engine.generate([1, 2], 1, None, lambda tokens: None)


def test_constraint_masks_accepted_chain_without_mutating_cached_logits(monkeypatch):
    engine = _engine(monkeypatch)
    torch = pytest.importorskip("torch")

    class Constraint:
        finished = False

        def mask(self, logits, window=None):
            saved = logits[:, 6].clone()
            logits[:] = -float("inf")
            logits[:, 6] = saved

        def window(self, tokens, parents):
            # Same semantics as rejecting every proposal that violates grammar.
            selected = [tokens[0]]
            for token in tokens[1:]:
                if token != 6:
                    break
                selected.append(token)
            return SimpleNamespace(tokens=selected)

        def advance(self, tokens):
            assert all(token == 6 for token in tokens)

    prompt = [1, 3, 5]
    engine.generate(prompt, 1, None, lambda tokens: None)
    before = engine._prefix.logits.clone()
    output = []
    engine.generate(prompt, 11, None, lambda tokens: output.extend(tokens), constraint=Constraint(), stop_eos=False)
    assert output == [6] * 11
    assert torch.equal(engine._prefix.logits, before)

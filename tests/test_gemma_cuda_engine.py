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


@pytest.mark.parametrize("option", [{"streams": 2}, {"drafter_bits": 4}, {"depth": 0}, {"depth": 2}])
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
            self.prompt_rows = []       # per position: computed by a prompt (prefill) forward

        @property
        def position(self):
            return len(self.tokens)

        def rollback(self, position):
            del self.tokens[position:]
            del self.prompt_rows[position:]

        def snapshot(self):
            return tuple(self.tokens), tuple(self.prompt_rows)

        def restore(self, snap):
            self.tokens, self.prompt_rows = list(snap[0]), list(snap[1])

        def reset(self):
            self.tokens, self.prompt_rows = [], []

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

        def forward(self, tokens, state, *, output_logits=True, prompt=False, **kwargs):
            rows = []
            for token in tokens:
                state.tokens.append(token)
                state.prompt_rows.append(prompt)
                rows.append([token, sum(state.tokens)])
            hidden = torch.tensor(rows, dtype=torch.bfloat16)
            return SimpleNamespace(hidden=hidden, logits=self.project_logits(hidden) if output_logits else None,
                                   taps={1: hidden})

    class Draft:
        def __init__(self):
            self.taps = []

        def reset(self):
            self.taps = []

        def snapshot(self):
            return tuple(self.taps)

        def restore(self, snap):
            self.taps = list(snap)

    class Head:
        tap_layers = (1,)

        def make_state(self):
            return Draft()

        def append(self, draft, taps):
            draft.taps.extend(int(v) for v in taps[:, 0])

        def propose(self, draft, anchor, limit, history=None):
            if not draft.taps:
                return None
            total = sum(draft.taps) + anchor
            proposals = []
            for _ in range(limit):
                token = int(scores(total).argmax())
                proposals.append(token)
                total += token
            # Force rejection after a correct prefix to expose rollback/tap bugs.
            if len(proposals) > 2:
                proposals[2] = (proposals[2] + 1) % 19
            return SimpleNamespace(finish=lambda row, count, history=None: (proposals[:count],
                                                                            list(range(-1, min(count, limit) - 1)),
                                                                            []))

    def sample(logits, positions, sampling):
        if sampling is None:
            return logits.argmax(-1).tolist()
        return [choose(row.numpy(), np.arange(19), position, sampling)
                for row, position in zip(logits.float(), positions)]

    engine = Gemma4Engine.__new__(Gemma4Engine)
    engine.target = Target()
    engine._state = engine.target.make_state()
    engine.head = Head() if with_head else None
    engine._draft = engine.head.make_state() if with_head else None
    engine.context_window, engine.max_rows, engine.eos = 400, 8, (18,)
    engine.vision = None
    engine._closed, engine._prefix, engine._owner, engine._turn = False, None, None, None
    engine._run_lock, engine._closed_lock, engine._prefix_lock = threading.Lock(), threading.Lock(), threading.Lock()
    engine._stream = SimpleNamespace(synchronize=lambda: None)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr("tensorfold.cuda.sampling.sample_rows", sample)
    return engine


def _assert_prompt_path(engine, prompt):
    """Every prompt position, and only those, ran a prompt forward, whatever the prefix cache supplied."""
    state = engine._state
    assert state.tokens[:len(prompt)] == list(prompt)
    assert state.prompt_rows == [True] * len(prompt) + [False] * (state.position - len(prompt))


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
    for e in (engine, fresh_engine):
        _assert_prompt_path(e, [3, 1, 4, 1, 5, 9, 2])
    assert engine._state.tokens == fresh_engine._state.tokens


@pytest.mark.parametrize("policy", [dict(temperature=1.0, top_k=7, top_p=.95), dict(temperature=.6, top_k=0, top_p=.9),
                                    None])
@pytest.mark.parametrize("penalized", [False, True])
def test_proposals_of_an_exact_drafter_keep_serial_output(monkeypatch, policy, penalized):
    """A drafter equal to the target proposes its argmax at anchor + 1 + j, penalized as the target's verify rows
    are: greedy rounds accept every proposal (penalized or not), sampled ones keep the serial output."""
    torch = pytest.importorskip("torch")
    from tensorfold.cuda import speculative
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.penalties import Penalties
    engine = _engine(monkeypatch)
    monkeypatch.undo()                      # the real CUDA sampler's host rule, on CPU scores
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr("tensorfold.cuda.sampling.validate_rows", lambda logits, positions, sampling: None)
    sampling = Sampling(seed=4321, **policy) if policy is not None else None
    penalties = Penalties(presence=0.8, frequency=0.4, repetition=1.2) if penalized else None

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
        def propose(self, draft, anchor, limit, history=None):
            first = len(draft.taps) + 1             # the committed context ends at the anchor
            seen.append(first == engine._state.position + 1)
            logits = torch.stack([scores(first + j) for j in range(limit)])
            return speculative.Chains(logits, [history], limit)

    engine.head = Exact()

    def run(draft, penalties=penalties):
        output = []
        stats = engine.generate([3, 1, 4, 1, 5], 40, sampling, lambda tokens: output.extend(tokens),
                                draft=draft, stop_eos=False, penalties=penalties)
        return output, stats

    serial, _ = run(False)
    drafted, stats = run(True)
    assert drafted == serial and len(drafted) == 40 and stats["drafted"] > 0
    assert seen and all(seen)
    if sampling is None:
        assert stats["accepted"] == stats["drafted"]
    if penalized:
        assert run(False, None)[0] != serial              # the penalties do change this reply


@pytest.mark.parametrize("sampled", [False, True])
def test_penalized_rows_keep_drafted_output_serial_with_unaware_drafts(monkeypatch, sampled):
    """Proposals that ignore the penalties are still verified against penalized rows (each by its path)."""
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.penalties import Penalties
    engine = _engine(monkeypatch)
    sampling = Sampling(seed=5, temperature=.8, top_k=9, top_p=.95) if sampled else None

    def run(draft, penalties):
        output = []
        stats = engine.generate([3, 1, 4, 1, 5], 35, sampling, lambda tokens: output.extend(tokens), draft=draft,
                                stop_eos=False, penalties=penalties)
        return output, stats

    penalties = Penalties(presence=2.0, frequency=1.5, repetition=1.5)
    serial, _ = run(False, penalties)
    drafted, stats = run(True, penalties)
    assert drafted == serial and stats["drafted"] > 0
    assert serial != run(False, None)[0]


def test_callback_stop_and_failure_do_not_publish_generated_context(monkeypatch):
    engine = _engine(monkeypatch)
    prompt = [4, 2]
    seen = []
    engine.generate(prompt, 35, None, lambda tokens: seen.extend(tokens) or True)
    assert len(seen) == 1
    assert engine._prefix.checkpoints[-1].state == (tuple(prompt), (True,) * len(prompt))
    with pytest.raises(RuntimeError, match="callback failed"):
        engine.generate(prompt, 35, None, lambda tokens: (_ for _ in ()).throw(RuntimeError("callback failed")))
    assert engine._owner is None
    fresh = []
    engine.generate(prompt, 3, None, lambda tokens: fresh.extend(tokens), stop_eos=False)
    assert fresh[:1] == seen
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate(prompt, 3, None, lambda tokens: None)


@pytest.mark.parametrize("turn", [None, 11])
def test_cancelled_prefill_does_not_replace_previous_prefix(monkeypatch, turn):
    """Cancelled before or after this prompt's turn-marker checkpoint was taken: the previous entry stays."""
    engine = _engine(monkeypatch)
    engine._turn = turn
    engine.generate([1, 2], 1, None, lambda tokens: None)
    before = engine._prefix
    taken, checkpoint = [], engine._checkpoint
    engine._checkpoint = lambda state, logits=None: taken.append(state.position) or checkpoint(state, logits)
    from tensorfold.server.cancellation import RequestCancelled
    with pytest.raises(RequestCancelled) as cancelled:
        engine.generate([3] * 5 + [11] + [3] * 254, 1, None, lambda tokens: None,
                        progress=lambda total, cached, done, ms: done > 5)
    assert taken == ([] if turn is None else [5])
    assert engine._prefix is before
    # The live traceback keeps the prefill frame: it must no longer hold the unpublished checkpoint.
    tb = cancelled.value.__traceback__
    while tb.tb_frame.f_code.co_name != "_prefill":
        tb = tb.tb_next
    assert tb.tb_frame.f_locals["kept"] == []


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
    before = engine._prefix.checkpoints[-1].logits.clone()
    output = []
    engine.generate(prompt, 11, None, lambda tokens: output.extend(tokens), constraint=Constraint(), stop_eos=False)
    assert output == [6] * 11
    assert torch.equal(engine._prefix.checkpoints[-1].logits, before)


@pytest.mark.parametrize("sampled", [False, True])
def test_next_turn_resumes_at_the_last_turn_marker_and_equals_a_fresh_prefill(monkeypatch, sampled):
    """A chat next turn re-renders the previous reply without the generation prompt's suffix: the shared prefix
    ends at the last turn marker, where a checkpoint resumes; replies equal a cold engine's."""
    from tensorfold.engine.exact_sampling import Sampling
    sampling = Sampling(seed=5, temperature=.7, top_k=7, top_p=.9) if sampled else None
    turn = 11
    first = [3, 1, 4, turn, 1, 5, 9]            # ... <|turn> model <generation suffix>
    second = first[:4] + [1, 2, 6, 5, turn, 1, 5, 9]   # history re-rendered differently after the marker
    engine = _engine(monkeypatch)
    engine._turn = turn
    engine.generate(first, 6, sampling, lambda t: None, stop_eos=False)
    assert [c.position for c in engine._prefix.checkpoints] == [3, 7]
    warm = []
    stats = engine.generate(second, 9, sampling, lambda t: warm.extend(t), stop_eos=False)
    assert stats["cached"] == 3
    assert [c.position for c in engine._prefix.checkpoints] == [8, 12]
    cold_engine = _engine(monkeypatch)
    cold_engine._turn = turn
    cold = []
    assert cold_engine.generate(second, 9, sampling, lambda t: cold.extend(t), stop_eos=False)["cached"] == 0
    assert warm == cold
    # Checkpoints hold prompt rows only, at prefill positions; resumed and cold positions ran the same path.
    assert [c.state for c in engine._prefix.checkpoints] == [(tuple(second[:p]), (True,) * p) for p in (8, 12)]
    for e in (engine, cold_engine):
        _assert_prompt_path(e, second)
    assert engine._state.tokens == cold_engine._state.tokens
    # The whole-prompt checkpoint still serves an exact repeat (with its logits) and extensions.
    assert engine.generate(second, 9, sampling, lambda t: None, stop_eos=False)["cached"] == 12


def test_turn_boundary_is_the_last_marker_past_the_cached_prefix_outside_visual_blocks(monkeypatch):
    engine = _engine(monkeypatch)
    turn = 11
    prompt = [2, turn, 5, 5, 5, turn, 7, 7, turn, 3]
    assert engine._turn_boundary(prompt, 0, ()) == 0            # no marker token: whole-prompt checkpoints only
    engine._turn = turn
    assert engine._turn_boundary(prompt, 0, ()) == 8
    assert engine._turn_boundary(prompt, 8, ()) == 0            # nothing past the cached prefix to keep
    assert engine._turn_boundary(prompt, 0, ((6, 10),)) == 0    # never inside a bidirectional visual block
    assert engine._turn_boundary(prompt, 0, ((2, 8),)) == 8     # a block may end at the boundary
    assert engine._turn_boundary([2, 5, 5], 0, ()) == 0


@pytest.mark.parametrize("with_head", [False, True])
def test_graph_runner_serves_decode_rounds_and_prefill_stays_eager(monkeypatch, with_head):
    engine = _engine(monkeypatch, with_head=with_head)
    calls = []

    class Graphs:
        def forward(self, tokens, state, *, output_taps):
            calls.append((list(tokens), state.position, output_taps))
            return engine.target.forward(tokens, state)

    engine.graphs = Graphs()
    got = []
    engine.generate(list(range(1, 12)), 24, None, lambda t: got.extend(t) and None, stop_eos=False)
    reference = _engine(monkeypatch, with_head=with_head)
    want = []
    reference.generate(list(range(1, 12)), 24, None, lambda t: want.extend(t) and None, stop_eos=False)
    assert got == want and calls
    # Every replayed round starts at a committed decode position and carries the first token plus drafts.
    assert all(position >= 11 and 1 <= len(tokens) <= engine.max_rows and taps == with_head
               for tokens, position, taps in calls)

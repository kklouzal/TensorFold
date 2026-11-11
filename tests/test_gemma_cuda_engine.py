"""Engine boundaries and exact accepted-path bookkeeping, without model weights."""

import threading
import time
from contextlib import nullcontext
from itertools import pairwise
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


def _effective_cpu(logits, sampling):
    """Fixture probability boundary; shared sampler extraction has its own CUDA/numerical controls."""
    import numpy as np

    from tensorfold.engine.rejection import proposal_row

    return [proposal_row(row.float().numpy(), np.arange(row.numel()), sampling) for row in logits]


def _engine(monkeypatch, *, with_head=True):
    torch = pytest.importorskip("torch")
    import numpy as np
    from tensorfold.engine.exact_sampling import choose

    class State:
        def __init__(self):
            self.tokens = []
            self.high_water, self.poisoned, self.global_cache = 0, False, 'none'
            self.device, self.context, self.rows = torch.device('cpu'), 400, 400
            self.prompt_rows = []       # per position: computed by a prompt (prefill) forward

        @property
        def position(self):
            return len(self.tokens)

        @position.setter
        def position(self, value):
            self.tokens = self.raw[0, :value, 0, 0].int().tolist()
            self.prompt_rows = self.inv[0, :value, 0].bool().tolist()

        def rollback(self, position):
            del self.tokens[position:]
            del self.prompt_rows[position:]

        def snapshot(self):
            return tuple(self.tokens), tuple(self.prompt_rows)

        def restore(self, snap):
            self.tokens, self.prompt_rows = list(snap[0]), list(snap[1])

        def reset(self):
            self.tokens, self.prompt_rows = [], []
            self.high_water, self.poisoned = 0, False

    from tensorfold.families.gemma4.cuda.state import State as CacheState
    for name in ('cache_group_bytes', 'cache_export_rows', 'cache_import_rows', 'cache_finish_restore'):
        setattr(State, name, getattr(CacheState, name))

    def scores(total):
        # A unique top, multiple plausible sampled choices, no accidental EOS.
        result = torch.tensor([-3.0 - ((i - total * 7) % 19) / 11 for i in range(19)])
        result[total * 7 % 19] = 2.0
        return result

    class Target:
        config = SimpleNamespace(vocab=19, hidden=2)
        device = torch.device("cpu")

        def make_state(self, *args, **kwargs):
            return State()

        def project_logits(self, hidden):
            return torch.stack([scores(int(row[1])) for row in hidden])

        def forward(self, tokens, state, *, output_logits=True, prompt=False, **kwargs):
            rows = []
            for token in tokens:
                position = state.position
                state.raw[0, position, 0, 0] = token
                state.inv[0, position, 0] = prompt
                state.keys[0, position % state.ring, 0, 0] = token
                state.values[0, position % state.ring, 0, 0] = prompt
                state.tokens.append(token)
                state.prompt_rows.append(prompt)
                state.high_water = max(state.high_water, state.position)
                rows.append([token, sum(state.tokens)])
            hidden = torch.tensor(rows, dtype=torch.bfloat16)
            return SimpleNamespace(hidden=hidden, logits=self.project_logits(hidden) if output_logits else None,
                                   taps={1: hidden})

    class Draft:
        def __init__(self):
            self.taps = []

        @property
        def length(self):
            return len(self.taps)

        @length.setter
        def length(self, value):
            self.taps = self.keys[0][:value, 0, 0].int().tolist()

        def reset(self):
            self.taps = []

        def snapshot(self):
            return tuple(self.taps)

        def restore(self, snap):
            self.taps = list(snap)

    from tensorfold.families.gemma4.cuda.dflash import DraftState as CacheDraft
    for name in ('cache_group_bytes', 'cache_export_rows', 'cache_import_rows'):
        setattr(Draft, name, getattr(CacheDraft, name))

    class Head:
        tap_layers = (1,)

        def __init__(self):
            self.append_roles = []

        def make_state(self):
            return Draft()

        def append(self, draft, taps, *, prompt=False):
            self.append_roles.append((len(taps), prompt))
            for value in taps[:, 0]:
                draft.keys[0][len(draft.taps)].fill_(int(value))
                draft.values[0][len(draft.taps)].fill_(int(value))
                draft.taps.append(int(value))

        def propose(self, draft, anchor, limit, history=None, *, sampled=False, sampling=None):
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
            def packet(row, count, sampling, first_position, history=None):
                from tensorfold.engine.rejection import ChainPacket, ProbabilityRow
                # An intentionally unaware deterministic proposal is valid q: target rejection corrects it.
                tokens = tuple(proposals[:count])
                return ChainPacket(tokens, tuple(ProbabilityRow([t], [1.0]) for t in tokens), first_position)

            return SimpleNamespace(sample=packet, finish=lambda row, count, history=None: (proposals[:count],
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
    engine._closed, engine._owner, engine._prefix_error = False, None, None
    engine._native_retirement_error = None
    engine._prefix_budget_bytes, engine._prefix_draft_kind = 1 << 20, 'dflash'
    engine._state.ring = 400
    engine._state.keys = torch.empty((1, 400, 1, 1), dtype=torch.bfloat16)
    engine._state.values = torch.empty_like(engine._state.keys)
    engine._state.raw = torch.empty((1, 400, 1, 1), dtype=torch.bfloat16)
    engine._state.inv = torch.empty((1, 400, 1))
    if engine._draft is not None:
        engine._draft.rows, engine._draft.layer_types = 400, ('full_attention',)
        engine._draft.pos = torch.zeros(1, dtype=torch.int32)
        engine._draft.rings = (400,)
        engine._draft.keys = [torch.empty((400, 1, 1), dtype=torch.bfloat16)]
        engine._draft.values = [torch.empty((400, 1, 1), dtype=torch.bfloat16)]
    engine._run_lock, engine._closed_lock = threading.Lock(), threading.Lock()
    engine._stream = SimpleNamespace(synchronize=lambda: None)
    from tensorfold.families.gemma4.cuda.prefix_archive import GemmaPrefixArchive
    engine._prefix_archive = GemmaPrefixArchive(engine, engine._prefix_budget_bytes)
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr("tensorfold.cuda.sampling.sample_rows", sample)
    monkeypatch.setattr("tensorfold.cuda.sampling.effective_rows", _effective_cpu)
    # Delivery math oracle follows the real frontend-deferral route. Each
    # logical test request owns/cancels its own intent; no response postpass
    # can hide a decode/rollback bug. Dedicated integration controls test raw
    # publication separately with the original bound method.
    generate = engine.generate
    engine._delivery_generate = generate
    def request(*args, **kwargs):
        if '_cache_publication' in kwargs:
            return generate(*args, **kwargs)
        intent = engine.cache_publication_intent('frontend')
        try:
            return generate(*args, _cache_publication=intent, **kwargs)
        finally:
            engine.cancel_cache_publication(intent)
    engine.generate = request
    return engine


def _assert_prompt_path(engine, prompt):
    """Every prompt position, and only those, ran a prompt forward, whatever the prefix cache supplied."""
    state = engine._state
    assert state.tokens[:len(prompt)] == list(prompt)
    assert state.prompt_rows == [True] * len(prompt) + [False] * (state.position - len(prompt))


@pytest.mark.parametrize("with_head", [False, True])
@pytest.mark.parametrize("cached,total,processed", [
    (0, 10, [4, 8, 10]),
    (8, 10, [10]),
    (5, 15, [9, 13, 15]),
    (10, 10, []),
])
def test_prefill_progress_includes_cached_tokens(monkeypatch, with_head, cached, total, processed):
    engine = _engine(monkeypatch, with_head=with_head)
    monkeypatch.setattr("tensorfold.families.gemma4.cuda.engine.PREFILL_ROWS", 4)
    prompt = [3] * total
    if cached:
        engine._prefill(prompt[:cached], None, None, time.perf_counter())
    reports = []

    def progress(length, reused, held, elapsed):
        reports.append((length, reused, held, elapsed))
        assert 0 <= reused <= held <= length
        return False

    _, _, reused = engine._prefill(prompt, None, progress, time.perf_counter())

    assert reused == cached
    assert [item[2] for item in reports] == processed
    assert all(item[:2] == (total, cached) and item[3] >= 0 for item in reports)
    assert all(a[3] <= b[3] for a, b in pairwise(reports))
    assert engine._state.position == total
    assert engine.prefix_cache_identity(prompt).position == len(prompt)
    if reports:
        assert reports[-1][2] == total


@pytest.mark.parametrize("with_head", [False, True])
def test_cached_prefill_progress_cancellation_keeps_the_published_prefix(monkeypatch, with_head):
    from tensorfold.server.cancellation import RequestCancelled

    engine = _engine(monkeypatch, with_head=with_head)
    monkeypatch.setattr("tensorfold.families.gemma4.cuda.engine.PREFILL_ROWS", 4)
    prefix, prompt = [3] * 5, [3] * 15
    engine._prefill(prefix, None, None, time.perf_counter())
    reports = []

    def cancel(length, reused, held, elapsed):
        reports.append((length, reused, held))
        return True

    with pytest.raises(RequestCancelled, match="cancelled during Gemma4 prefill"):
        engine._prefill(prompt, None, cancel, time.perf_counter())

    assert reports == [(15, 5, 9)]
    assert engine._state.position == 9
    assert engine.prefix_cache_identity(prefix).position == len(prefix)
    resumed = []
    engine._prefill(prompt, None, lambda total, cached, held, elapsed: resumed.append((cached, held)),
                    time.perf_counter())
    assert resumed == [(9, 13), (9, 15)]  # committed cancelled input is reusable, not just the old point
    assert engine._state.position == 15
    assert engine.prefix_cache_identity(prompt).position == len(prompt)


def test_companion_prompt_role_covers_tiny_turn_suffix_and_cached_extension(monkeypatch):
    engine = _engine(monkeypatch)
    prompt = [3] * 20 + [17, 1, 2]
    engine.generate(prompt[:20], 1, None, lambda tokens: None, stop_eos=False)
    engine.generate(prompt, 1, None, lambda tokens: None, stop_eos=False)
    assert engine.head.append_roles == [(20, True), (3, True)]
    engine.head.append_roles.clear()
    engine.generate(prompt + [2] * 40, 8, None, lambda tokens: None, stop_eos=False)
    assert engine.head.append_roles[0] == (40, True)
    assert all(not role for _, role in engine.head.append_roles[1:])  # accepted decode rows retain their lane role


@pytest.mark.parametrize("sampled", [False, True])
def test_dflash_rejections_and_prompt_resume_keep_request_contract(monkeypatch, sampled):
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
    assert drafted == (run([3, 1, 4, 1, 5], True)[0] if sampled else serial)
    assert stats["cached"] == 5
    assert stats["drafted"] > 0
    assert stats["accepted"] < stats["drafted"]
    resumed, resumed_stats = run([3, 1, 4, 1, 5, 9, 2], True)
    fresh_engine = _engine(monkeypatch)
    fresh = []
    fresh_engine.generate([3, 1, 4, 1, 5, 9, 2], 35, sampling, lambda tokens: fresh.extend(tokens),
                          draft=sampled, stop_eos=False)
    assert resumed == fresh
    assert resumed_stats["cached"] == 5
    assert engine.prefix_cache_identity([3, 1, 4, 1, 5, 9, 2]).position == 7
    for e in (engine, fresh_engine):
        _assert_prompt_path(e, [3, 1, 4, 1, 5, 9, 2])
    assert engine._state.tokens == fresh_engine._state.tokens


@pytest.mark.parametrize("policy", [dict(temperature=1.0, top_k=7, top_p=.95), dict(temperature=.6, top_k=0, top_p=.9),
                                    None])
@pytest.mark.parametrize("penalized", [False, True])
def test_exact_drafter_preserves_greedy_and_accepts_sampled_target_law(monkeypatch, policy, penalized):
    """A drafter equal to the target proposes its argmax at anchor + 1 + j, penalized as the target's verify rows
    are. Greedy keeps serial output; sampled equal-q/p proposals all accept under their actual target law."""
    torch = pytest.importorskip("torch")
    from tensorfold.cuda import speculative
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.penalties import Penalties
    engine = _engine(monkeypatch)
    monkeypatch.undo()                      # the real CUDA sampler's host rule, on CPU scores
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    monkeypatch.setattr("tensorfold.cuda.sampling.validate_rows", lambda logits, positions, sampling: None)
    monkeypatch.setattr("tensorfold.cuda.sampling.effective_rows", _effective_cpu)
    sampling = Sampling(seed=4321, **policy) if policy is not None else None
    penalties = Penalties(presence=0.8, frequency=0.4, repetition=1.2) if penalized else None

    def scores(position):                   # the target's scores for the token at ``position``, whatever came before
        return torch.randn(19, generator=torch.Generator().manual_seed(position)) * 2

    def forward(tokens, state, *, output_logits=True, **kwargs):
        rows = []
        for token in tokens:
            position = state.position
            state.raw[0, position, 0, 0] = token
            state.inv[0, position, 0] = kwargs.get('prompt', False)
            state.keys[0, position % state.ring, 0, 0] = token
            state.values[0, position % state.ring, 0, 0] = kwargs.get('prompt', False)
            rows.append([token, position])
            state.tokens.append(token)
            state.prompt_rows.append(kwargs.get('prompt', False))
            state.high_water = max(state.high_water, state.position)
        hidden = torch.tensor(rows, dtype=torch.bfloat16)
        return SimpleNamespace(hidden=hidden, logits=project(hidden) if output_logits else None, taps={1: hidden})

    def project(hidden):
        return torch.stack([scores(int(row[1]) + 1) for row in hidden])

    engine.target.forward, engine.target.project_logits = forward, project
    seen = []

    class Exact(type(engine.head)):
        def propose(self, draft, anchor, limit, history=None, *, sampled=False, sampling=None):
            first = len(draft.taps) + 1             # the committed context ends at the anchor
            seen.append(first == engine._state.position + 1)
            logits = torch.stack([scores(first + j) for j in range(limit)])
            return speculative.Chains(logits, [history], limit, sampled=sampled, sampling=sampling)

    engine.head = Exact()

    def run(draft, penalties=penalties):
        output = []
        stats = engine.generate([3, 1, 4, 1, 5], 41, sampling, lambda tokens: output.extend(tokens),
                                draft=draft, stop_eos=False, penalties=penalties)
        return output, stats

    serial, _ = run(False)
    drafted, stats = run(True)
    assert drafted == (run(True)[0] if sampling is not None else serial)
    assert len(drafted) == 41 and stats["drafted"] > 0
    assert seen and all(seen)
    assert stats["accepted"] == stats["drafted"]
    if penalized:
        assert run(False, None)[0] != serial              # the penalties do change this reply


@pytest.mark.parametrize("sampled", [False, True])
def test_penalized_rows_keep_target_contract_with_unaware_drafts(monkeypatch, sampled):
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
    assert drafted == (run(True, penalties)[0] if sampled else serial)
    assert stats["drafted"] > 0
    assert serial != run(False, None)[0]
    if sampled:
        assert drafted != run(True, None)[0]


def test_callback_stop_and_failure_do_not_publish_generated_context(monkeypatch):
    engine = _engine(monkeypatch)
    prompt = [4, 2]
    seen = []
    engine.generate(prompt, 35, None, lambda tokens: seen.extend(tokens) or True)
    assert len(seen) == 1
    assert engine.prefix_cache_identity(prompt).position == len(prompt)
    assert engine._prefill(prompt, None, None, time.perf_counter())[2] == len(prompt)
    assert engine._state.tokens == prompt and all(engine._state.prompt_rows)
    with pytest.raises(RuntimeError, match="callback failed"):
        engine.generate(prompt, 35, None, lambda tokens: (_ for _ in ()).throw(RuntimeError("callback failed")))
    assert engine._owner is None
    fresh = []
    engine.generate(prompt, 3, None, lambda tokens: fresh.extend(tokens), stop_eos=False)
    assert fresh[:1] == seen
    engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        engine.generate(prompt, 3, None, lambda tokens: None)




def test_reentrant_callback_is_rejected_without_stranding_engine(monkeypatch):
    engine = _engine(monkeypatch)
    def reenter(tokens):
        engine.generate([1, 2], 1, None, lambda tokens: None)
    with pytest.raises(RuntimeError, match="reenter"):
        engine.generate([1, 2], 1, None, reenter)
    assert engine._owner is None
    engine.generate([1, 2], 1, None, lambda tokens: None)


@pytest.mark.parametrize("sampled", [False, True])
def test_constraint_masks_accepted_chain_without_mutating_cached_logits(monkeypatch, sampled):
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
    before = engine._prefill(prompt, None, None, time.perf_counter())[1].clone()
    output = []
    from tensorfold.engine.exact_sampling import Sampling
    sampling = Sampling(seed=5, temperature=.7, top_k=7, top_p=.9) if sampled else None
    engine.generate(prompt, 11, sampling, lambda tokens: output.extend(tokens), constraint=Constraint(), stop_eos=False)
    assert output == [6] * 11
    assert torch.equal(engine._prefill(prompt, None, None, time.perf_counter())[1], before)


@pytest.mark.parametrize("sampled", [False, True])
def test_arbitrary_interior_next_turn_lcp_equals_fresh_prefill(monkeypatch, sampled):
    """A chat next turn re-renders the previous reply without the generation prompt's suffix: the shared prefix
    ends at the last turn marker, where a checkpoint resumes; replies equal a cold engine's."""
    from tensorfold.engine.exact_sampling import Sampling
    sampling = Sampling(seed=5, temperature=.7, top_k=7, top_p=.9) if sampled else None
    turn = 11
    first = [3, 1, 4, turn, 1, 5, 9]            # ... <|turn> model <generation suffix>
    second = first[:4] + [1, 2, 6, 5, turn, 1, 5, 9]   # history re-rendered differently after the marker
    engine = _engine(monkeypatch)
    engine.generate(first, 6, sampling, lambda t: None, stop_eos=False)
    assert engine.prefix_cache_identity(first).position == len(first)
    warm = []
    stats = engine.generate(second, 9, sampling, lambda t: warm.extend(t), stop_eos=False)
    assert stats["cached"] == 5  # exact token LCP, not the old last-turn checkpoint
    assert engine.prefix_cache_identity(second).position == len(second)
    cold_engine = _engine(monkeypatch)
    cold = []
    assert cold_engine.generate(second, 9, sampling, lambda t: cold.extend(t), stop_eos=False)["cached"] == 0
    assert warm == cold
    # Checkpoints hold prompt rows only, at prefill positions; resumed and cold positions ran the same path.
    assert engine.prefix_cache_identity(second).position == len(second)
    for e in (engine, cold_engine):
        _assert_prompt_path(e, second)
    assert engine._state.tokens == cold_engine._state.tokens
    # The whole-prompt checkpoint still serves an exact repeat (with its logits) and extensions.
    assert engine.generate(second, 9, sampling, lambda t: None, stop_eos=False)["cached"] == 12


@pytest.mark.parametrize("budget", [2, 3, 8, 9, 17, 35])
@pytest.mark.parametrize("penalized", [False, True])
def test_sampled_logical_packets_keep_prefix_across_reply_budgets_and_cache(monkeypatch, budget, penalized):
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.penalties import Penalties

    engine = _engine(monkeypatch)
    sampling = Sampling(seed=173, temperature=.8, top_k=0, top_p=.91, min_p=.01)
    penalties = Penalties(presence=.4, frequency=.2, repetition=1.2) if penalized else None
    launch = engine.head.propose
    widths = []

    def propose(state, anchor, limit, history=None, *, sampled=False, sampling=None):
        widths.append((limit, sampled))
        return launch(state, anchor, limit, history, sampled=sampled, sampling=sampling)

    engine.head.propose = propose

    def run(count):
        tokens = []
        stats = engine.generate([3, 1, 4], count, sampling, lambda t: tokens.extend(t),
                                stop_eos=False, penalties=penalties)
        assert engine._draft.taps == engine._state.tokens
        return tokens, stats

    reference, cold = run(35)
    short, cached = run(budget)
    assert short == reference[:budget] and cold["cached"] == 0 and cached["cached"] == 3
    assert widths and all(width == (7, True) for width in widths)
    assert engine._state.tokens == [3, 1, 4] + short[:-1]


@pytest.mark.parametrize("sampled_eos", [False, True])
def test_sampled_terminal_eos_or_residual_stays_pending_outside_committed_taps(monkeypatch, sampled_eos):
    torch = pytest.importorskip("torch")
    from tensorfold.engine.exact_sampling import Sampling
    from tensorfold.engine.rejection import ChainPacket, ProbabilityRow

    engine = _engine(monkeypatch)
    engine.eos = (6,)
    prompt = [2, 3]

    def project(hidden):
        logits = torch.full((len(hidden), 19), -float("inf"))
        logits[:, 1 if engine._state.position == len(prompt) else 6] = 0
        return logits

    def propose(state, anchor, limit, history=None, *, sampled=False, sampling=None):
        assert sampled and limit == 7
        token = 6 if sampled_eos else 9  # actual q accepts EOS or is surely rejected into the target's EOS
        return SimpleNamespace(sample=lambda row, count, policy, position, history=None:
                               ChainPacket((token,) * count, (ProbabilityRow([token], [1.0]),) * count, position))

    engine.target.project_logits, engine.head.propose = project, propose
    output = []
    engine.generate(prompt, 20, Sampling(seed=41, temperature=.7, top_k=7), lambda t: output.extend(t))
    assert output == [1, 6]
    assert engine._draft.taps == engine._state.tokens == prompt + [1]
    assert engine.prefix_cache_identity(prompt).position == len(prompt)


def test_sampled_probability_failure_never_publishes_speculative_context(monkeypatch):
    from tensorfold.engine.exact_sampling import Sampling

    engine = _engine(monkeypatch)
    policy = Sampling(seed=271, temperature=.7, top_k=7)
    prompt = [4, 1, 3]
    output = []

    def failed(_logits, _sampling):
        raise ValueError("invalid effective probability row")

    monkeypatch.setattr("tensorfold.cuda.sampling.effective_rows", failed)
    with pytest.raises(ValueError, match="invalid effective probability"):
        engine.generate(prompt, 17, policy, lambda t: output.extend(t), stop_eos=False)
    assert len(output) == 1 and engine._owner is None and engine.prefix_cache_identity(prompt).position == len(prompt)
    monkeypatch.setattr("tensorfold.cuda.sampling.effective_rows", _effective_cpu)
    resumed, fresh = [], []
    engine.generate(prompt, 17, policy, lambda t: resumed.extend(t), stop_eos=False)
    other = _engine(monkeypatch)
    other.generate(prompt, 17, policy, lambda t: fresh.extend(t), stop_eos=False)
    assert resumed == fresh and engine._draft.taps == engine._state.tokens == other._state.tokens




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


@pytest.mark.parametrize("context,smallest", [(256, 1), (4096, 8), (16384, 16)])
def test_graph_startup_prepares_the_smallest_grid_even_when_the_context_caps_it(context, smallest):
    pytest.importorskip("torch")
    pytest.importorskip("triton")
    from tensorfold.families.gemma4.cuda.graphs import Graphs
    from tensorfold.families.gemma4.cuda.state import GLOBAL_CHUNK

    state = SimpleNamespace(context=context, position=0)
    state.rollback = lambda position: setattr(state, "position", position)
    graphs = Graphs.__new__(Graphs)
    graphs.state, graphs.max_rows, graphs.span = state, 3, 16
    graphs.limit, graphs.graphs = -(-context // GLOBAL_CHUNK), {}
    calls = []

    def forward(tokens, seen, *, output_taps):
        assert seen is state and seen.position == 0
        key = (len(tokens), graphs.chunks(len(tokens)), output_taps)
        calls.append(key)
        graphs.graphs[key] = None
        seen.position += len(tokens)

    graphs.forward = forward
    assert graphs.prepare() == 6 and state.position == 0
    assert calls == [(rows, smallest, taps) for rows in range(1, 4) for taps in (False, True)]


def test_gate_continuation_preserves_original_prompt_checkpoints(monkeypatch):
    engine = _engine(monkeypatch)
    prompt = [3, 5, 8, 2]
    engine.generate(prompt, 3, None, lambda _: False)
    original = engine.prefix_cache_identity(prompt)
    assert original.position == len(prompt)
    continued = [*prompt, 4, 6, 9]
    engine.generate(continued, 2, None, lambda _: False, cache_prompt=False)
    assert engine.prefix_cache_identity(prompt) == original
    result = engine.generate(prompt, 1, None, lambda _: False)
    assert result['cached'] == len(prompt)





























def _native_metadata_cache(monkeypatch):
    """Actual deployed State/DFlash snapshot shapes on Torch meta, no model/GPU.

    The cache and snapshot implementations are real. Meta replaces only tensor
    data, so this checks logical allocation/ownership without claiming physical
    CUDA allocator peaks or model output correctness.
    """
    import json
    from pathlib import Path
    import torch
    from tensorfold.families.gemma4.cuda.state import State, capacity_bytes, PREFILL_ROWS
    from tensorfold.families.gemma4.cuda.weights import Config
    from tensorfold.families.gemma4.cuda.dflash import DraftState
    cfg = Config.from_dict(json.loads((Path(__file__).parent / 'fixtures/gemma26b_modelopt_metadata.json').read_text())['raw'])
    engine = _engine(monkeypatch)
    engine.context_window = 262144
    engine._state = State(cfg, 'meta', engine.context_window, rows=PREFILL_ROWS)
    draft_cfg = SimpleNamespace(layer_types=['sliding_attention'] * 4 + ['full_attention'],
                                sliding_window=2048, kv_heads=8, head_dim=128)
    engine._draft = DraftState(draft_cfg, 'meta', engine.context_window, rows=PREFILL_ROWS, block=5)
    engine._prefix_budget_bytes = 3 * (capacity_bytes(cfg, engine.context_window, PREFILL_ROWS)
        + sum(t.numel() * t.element_size() for t in (*engine._draft.keys, *engine._draft.values)))
    class Target:
        config, device = cfg, torch.device('meta')
        def forward(self, ids, state, **kwargs):
            state.position += len(ids)
            state.high_water = state.position
            return SimpleNamespace(hidden=torch.empty((len(ids), cfg.hidden), dtype=torch.bfloat16, device='meta'))
        def project_logits(self, hidden):
            return torch.empty((1, cfg.vocab), dtype=torch.bfloat16, device='meta')
    engine.target = Target()
    engine._append_head = lambda draft, result, ids, **kwargs: setattr(draft, 'length', draft.length + len(ids))
    from tensorfold.families.gemma4.cuda.prefix_archive import GemmaPrefixArchive
    engine._prefix_archive = GemmaPrefixArchive(engine, engine._prefix_budget_bytes)
    return engine


@pytest.mark.parametrize('short_count', [6, 10])
def test_native_40k_conversation_survives_short_interleaving_when_byte_budget_allows(monkeypatch, short_count):
    engine = _native_metadata_cache(monkeypatch)
    voice = [3] * 39992 + [11] + [4] * 7
    engine._prefill(voice, None, None, time.perf_counter())
    long_checkpoint = engine.prefix_cache_identity(voice)
    assert long_checkpoint.position == 40000
    for index in range(short_count):
        length = 2200 + index * 400
        prompt = [20 + index] + [5] * (length - 1)
        engine._prefill(prompt, None, None, time.perf_counter())
        stats = engine.prefix_cache_stats()
        assert stats['charged_bytes'] <= stats['byte_budget']
    assert engine.prefix_cache_identity(voice) == long_checkpoint
    _, _, cached = engine._prefill(voice, None, None, time.perf_counter())
    assert cached == 40000
    stats = engine.prefix_cache_stats()
    assert stats['capacity'] is stats['max_entries'] is stats['entries'] is None
    assert stats['last_lookup']['matched_tokens'] == 40000
    assert stats['indexed_tokens'] >= 40000
    assert len(stats['eviction_history']) <= 32











def test_cancelled_prefill_preserves_prior_backing_and_only_committed_input_rows(monkeypatch):
    from tensorfold.server.cancellation import RequestCancelled
    engine = _engine(monkeypatch)
    old = [1, 2]
    engine.generate(old, 1, None, lambda tokens: None)
    identity = engine.prefix_cache_identity(old)
    with pytest.raises(RequestCancelled):
        engine.generate([3] * 20, 1, None, lambda tokens: pytest.fail('output after prefillcancel'),
                        progress=lambda total, cached, done, ms: True)
    assert engine.prefix_cache_identity(old) == identity
    assert engine._draft.length == engine._state.position == 20
    assert not engine._prefix_archive.pending
    # Complete committed input may be cached; no uncomputed response is claimed.
    assert engine.prefix_cache_identity([3] * 20).position == 20



def test_positive_archive_credit_must_fund_core_index_before_runtime(tmp_path, monkeypatch):
    import torch
    monkeypatch.setattr(torch.cuda, 'set_device', lambda *_: pytest.fail('late cache-credit refusal'))
    with pytest.raises(ValueError, match='empty index metadata'):
        Gemma4Engine(tmp_path, prefix_cache_bytes=1)

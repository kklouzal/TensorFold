"""Flash Next drafting behind the benchmark switches "draft_rule" (the cost-aware rule, design 3) and "suffix" (session
drafts, design 4); "old" values are the A/B baseline, removed after selection. Drafts never change a reply: every
switch value, cost, chance and proposal (true continuations, true prefixes then garbage, end tokens, ids outside
the draft vocabulary) emits each stream's solo tokens, through joins, finishes, prompt passes, captured rounds, a
background replay, a grammar and a background index build; a failed frame's codec status still wins; a stream whose
drafting fails ends alone with the error, the others keeping their solo tokens.

The synthetic model never repeats its text, so its session index never matches on its own: proposals here are the
real index's calls with their result replaced (``Texts``), which drives every path after them for real."""

import random
import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402
from test_flashnext_multi_graphs import RQ, _switch  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.drafters import draft_rule, session_index  # noqa: E402
from tensorfold.drafters.session_index import SessionIndex  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import KVNumericError  # noqa: E402

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
LOOP = [7, 300, 41, 9, 1500, 22, 64, 5] * 8                       # repetitive text: the real index matches it
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=1.0, min_p=0.1),
             Sampling(seed=99, top_k=0, top_p=0.9), None]


def _vocab_model(seed=3):
    """Drafts over a token subset, as the real model's draft head (ids not in it are verified, never drafted)."""

    w = _model(seed)
    words, scales, biases = qmm.to_mlx(w.head)
    ids = torch.arange(1, V, 3, device="cuda")
    w.draft_ids = ids
    w.draft_head = qmm.make_q4(words[ids], scales[ids], biases[ids])
    return w


def _solo(w, prompts, samplings, counts, kv_dtype="bf16", capacity=1024, prefill_rows=16, constraints=None):
    refs = []
    for i, (prompt, sampling, count) in enumerate(zip(prompts, samplings, counts)):
        e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=prefill_rows, kv_dtype=kv_dtype)
        c = constraints[i]() if constraints is not None else None
        refs.append(serial_decode(e, prefill(e, prompt, sampling, constraint=c), count, sampling,
                                  constraint=c).tokens)
    return refs


class Texts:
    """Session (``SessionIndex.propose``) and copy (``CopyIndex.propose``) proposals: the real call runs, then its
    result is replaced by one of ``kinds`` built from the stream's solo reply (``refs``, by prompt): the true
    continuation, a true prefix then garbage, garbage, an end token inside, ids outside the draft vocabulary, a
    short match (4-7 tokens: no rule takes it), or nothing. Records (kind, match length, tokens) a proposal."""

    KINDS = ("truth", "truth", "prefix", "garbage", "eos", "outside", "short", "none")

    def __init__(self, refs: dict[tuple, list[int]], seed: int = 0, kinds=KINDS) -> None:
        self.refs, self.rng, self.kinds, self.made = refs, random.Random(seed), kinds, []

    def _make(self, prompt: tuple, at: int, count: int) -> tuple[int, list[int]]:
        ref, rng = self.refs[prompt], self.rng
        kind = rng.choice(self.kinds)
        truth = ref[at:at + count]
        garbage = [rng.randrange(1, V) for _ in range(count)]
        cut = rng.randrange(count + 1)
        tokens = {"truth": truth + garbage, "prefix": truth[:cut] + garbage, "garbage": garbage,
                  "eos": truth[:cut] + [0] + garbage, "outside": [3 * rng.randrange(1, V // 3) for _ in range(count)],
                  "short": truth + garbage, "none": []}[kind][:count]
        length = 0 if kind == "none" else rng.randint(4, 7) if kind == "short" else rng.randint(8, 64)
        self.made.append((kind, length, tokens))
        return (length, tokens) if length else (0, [])

    def install(self, monkeypatch) -> "Texts":
        real = SessionIndex.propose

        def propose(index, count):
            got = real(index, count)                     # the real index runs (and adopts a finished build)
            if count <= 0 or not index.ready():
                return got
            return self._make(tuple(index.tokens[:index.n - index.reply].tolist()), index.reply, count)

        monkeypatch.setattr(SessionIndex, "propose", propose)
        from tensorfold.families.qwen3_5.cuda.decode import CopyIndex

        real_copy = CopyIndex.propose

        def copy(index, context, max_nodes=127):
            real_copy(index, context, max_nodes)
            prompt = next(p for p in self.refs if len(p) <= len(context) and tuple(context[:len(p)]) == p)
            length, tokens = self._make(prompt, len(context) - len(prompt), max_nodes)
            return tokens if length >= 8 else []          # a copy needs its 8-token needle

        monkeypatch.setattr(CopyIndex, "propose", copy)
        return self


def _taken(monkeypatch) -> list:
    """(source, drafts taken, chances or None) for every text a stream drafted from."""

    taken, real = [], MultiDecoder._from_text

    def spy(self, s, text, chances, base, joined, agree=None):
        real(self, s, text, chances, base, joined, agree)
        taken.append((text[0], list(s.drafts), chances, s.sid, base))

    monkeypatch.setattr(MultiDecoder, "_from_text", spy)
    return taken


def _run(dec, streams, later=(), each_round=None):
    """Admit ``streams`` (``later``: (round, index) admitted then) and decode until every one is done."""

    pending = dict((i, r) for r, i in later)
    for i, s in enumerate(streams):
        if i not in pending:
            dec.admit(s)
    rounds = 0
    while dec.live() or pending:
        for i, r in list(pending.items()):
            if rounds >= r:
                dec.admit(streams[i])
                del pending[i]
        if each_round is not None:
            each_round(dec, rounds)
        dec.finish(dec.round())
        rounds += 1
    return rounds


@pytest.mark.parametrize("rule,suffix", [("new", "new"), ("new", "old"), ("old", "new"), ("old", "old")])
@pytest.mark.parametrize("depth", [1, 4])
def test_every_drafting_switch_emits_each_streams_solo_tokens(monkeypatch, tmp_path, rule, suffix, depth):
    """Greedy, keyed top-k/top-p, min-p and top_k=0 streams and a serial one; two join mid-run (one filling its prompt
    in the rounds' forwards: mixed rounds), all finish at different lengths; texts true, partly true or adversarial."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    w = _vocab_model()
    g = torch.Generator().manual_seed(11)
    long = torch.randint(1, V, (70,), generator=g).tolist()
    prompts = [LOOP, PROMPTS[1], PROMPTS[0], long, PROMPTS[3]]
    counts = [26, 30, 18, 24, 12]
    refs = _solo(w, prompts, SAMPLINGS, counts)
    texts = Texts({tuple(p): r for p, r in zip(prompts, refs)}, seed=depth).install(monkeypatch)
    taken = _taken(monkeypatch)
    _switch(monkeypatch, tmp_path, draft_rule=rule, suffix=suffix)
    dec = MultiDecoder(w, slots=5, capacity=1024, depth=depth, confidence=0.5, prefill_rows=16)
    streams = [Stream(list(p), c, smp, draft=i != 4) for i, (p, smp, c) in enumerate(zip(prompts, SAMPLINGS, counts))]
    mixed = []
    _run(dec, streams, later=[(2, 3), (4, 4)], each_round=lambda d, r: mixed.append(bool(d.filling) and d.converged))
    assert [s.out for s in streams] == refs
    assert any(mixed)                                    # a prompt filled in the rounds' forwards
    assert texts.made and any(src == ("suffix" if suffix == "new" else "copy") and drafts for src, drafts, *_ in taken)
    assert all(len(drafts) <= depth for _, drafts, *_ in taken)
    assert sum(s.accepted for s in streams[:4]) > 0 and streams[4].drafted == 0
    if rule == "new":                                    # the rule's chances, never above 1, never rising
        assert all(c is not None and all(b <= a <= 1 for a, b in zip(c, c[1:])) for _, _, c, *_ in taken)
    else:
        assert all(c is None for _, _, c, *_ in taken)


@pytest.mark.parametrize("kv_dtype", ["int8", "int4"])
def test_text_drafts_keep_each_cache_formats_solo_tokens(monkeypatch, tmp_path, kv_dtype):
    """Whole and partly true text windows committed into int8 and int4 caches: each stream's solo tokens."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    w = _model()
    prompts, counts = [LOOP, PROMPTS[1], PROMPTS[3]], [24, 20, 16]
    refs = _solo(w, prompts, SAMPLINGS[:3], counts, kv_dtype)
    Texts({tuple(p): r for p, r in zip(prompts, refs)}, seed=4, kinds=("truth", "prefix")).install(monkeypatch)
    _switch(monkeypatch, tmp_path, draft_rule="new", suffix="new")
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=4, confidence=0.5, kv_dtype=kv_dtype)
    streams = [Stream(list(p), c, smp) for p, smp, c in zip(prompts, SAMPLINGS, counts)]
    _run(dec, streams)
    assert [s.out for s in streams] == refs and sum(s.accepted for s in streams) > 0


@pytest.mark.parametrize("rule,suffix", [("new", "new"), ("old", "old")])
def test_text_drafts_through_captured_rounds(monkeypatch, tmp_path, rule, suffix):
    """Captured verify forwards and MTP steps (switch "graphs") replay rounds whose windows hold text drafts up to the
    depth (within the captured keys) and absorb up to depth + 1 rows; a stream past the attention budget too."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    w = _model(7)
    g = torch.Generator().manual_seed(9)
    prompts = [torch.randint(1, V, (2300,), generator=g).tolist(), LOOP, PROMPTS[1]]
    samplings = [Sampling(seed=5, top_k=20, top_p=0.95), None, Sampling(seed=6, top_k=20, top_p=0.95)]
    counts = [20, 24, 16]
    refs = _solo(w, prompts, samplings, counts, RQ, capacity=4096, prefill_rows=256)
    Texts({tuple(p): r for p, r in zip(prompts, refs)}, seed=3, kinds=("truth", "prefix", "eos")).install(monkeypatch)
    taken = _taken(monkeypatch)
    _switch(monkeypatch, tmp_path, draft_rule=rule, suffix=suffix)
    dec = MultiDecoder(w, slots=3, capacity=4096, depth=3, confidence=0.5, kv_dtype=RQ, prefill_rows=256,
                       graphs=True)
    dec.capture_graphs()
    streams = [Stream(list(p), c, smp) for p, smp, c in zip(prompts, samplings, counts)]
    replays = []
    _run(dec, streams, later=[(2, 2)], each_round=lambda d, r: replays.append(d.replays))
    assert [s.out for s in streams] == refs
    assert sum(replays) > 0 and any(len(drafts) == 3 for _, drafts, *_ in taken)


def test_random_costs_chances_and_adversarial_texts_keep_every_token(monkeypatch, tmp_path):
    """The cost-aware rule with random costs (zero-draft rounds to whole windows), random chances and step values
    (steps that one stream, several together or none pay for), and adversarial texts: every stream emits its solo
    tokens."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    w = _vocab_model(5)
    prompts = [LOOP, PROMPTS[1], PROMPTS[0], PROMPTS[3]]
    counts = [30, 24, 28, 20]
    refs = _solo(w, prompts, SAMPLINGS[:4], counts)
    Texts({tuple(p): r for p, r in zip(prompts, refs)}, seed=8).install(monkeypatch)
    rng = random.Random(17)

    monkeypatch.setattr(draft_rule.CostModel, "lam", lambda self, live, mixed, context: 1.0)   # costs in tokens
    monkeypatch.setattr(draft_rule.CostModel, "row_ms", lambda self, context, mixed: rng.choice([0.0, 0.1, 0.3, 0.6,
                                                                                                0.95, 1.5]))
    monkeypatch.setattr(draft_rule.CostModel, "step_ms", lambda self, mixed: rng.choice([0.0, 0.05, 0.5]))
    monkeypatch.setattr(draft_rule.CostModel, "step_stream_ms", lambda self, mixed: rng.choice([0.0, 0.02, 0.2]))
    monkeypatch.setattr(draft_rule.Table, "chance", lambda self, depth, score: rng.random())
    monkeypatch.setattr(draft_rule.Table, "value", lambda self, depth, c, row: rng.random() * 0.6)
    monkeypatch.setattr(draft_rule.Table, "optimistic", lambda self, depth: rng.random())
    _switch(monkeypatch, tmp_path, draft_rule="new", suffix="new")
    windows, real = [], MultiDecoder._trim

    def trim(self, live):
        real(self, live)
        windows.extend(len(s.drafts) for s in live if s.draft)

    monkeypatch.setattr(MultiDecoder, "_trim", trim)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=4, confidence=0.5, prefill_rows=16)
    streams = [Stream(list(p), c, smp) for p, smp, c in zip(prompts, SAMPLINGS, counts)]
    _run(dec, streams, later=[(3, 3)])
    assert [s.out for s in streams] == refs
    assert 0 in windows and 4 in windows                 # zero-draft rounds and whole windows


@pytest.mark.parametrize("rule", ["new", "old"])         # "old": A/B baseline, remove after selection
def test_a_background_replay_drafts_the_tokens_it_owes(monkeypatch, tmp_path, rule):
    """A background stream gives way (``Stream.continued``) and replays from its prompt: it drafts what it owes
    (certain), whole windows kept, then drafts as any stream; its reply is the solo one."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    w = _model()
    sampling = Sampling(seed=31, top_k=20, top_p=0.95)
    ref = _solo(w, [PROMPTS[1]], [sampling], [40])[0]
    taken = _taken(monkeypatch)
    _switch(monkeypatch, tmp_path, draft_rule=rule, suffix="new")
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=4, confidence=0.5, prefill_rows=16)
    s = Stream(list(PROMPTS[1]), 40, sampling, background=True)
    dec.admit(s)
    while len(s.out) < 17:
        dec.finish(dec.round())
    dec.finish([s])                                      # it gives way, as the scheduler's _yield does
    again = s.continued()
    dec.admit(again)
    while dec.live():
        dec.finish(dec.round())
    assert again.error is None and again.out == ref
    owed = [(drafts, base) for src, drafts, _, sid, base in taken if src == "owed" and sid == again.sid]
    assert owed and all(drafts == ref[base:base + len(drafts)] and len(drafts) == min(4, len(s.out) - base)
                        for drafts, base in owed)
    bases = [base for _, base in owed]                   # whole windows kept, round after round, to what it owes
    assert bases == list(range(bases[0], bases[0] + 5 * len(bases), 5)) and bases[-1] + 5 >= len(s.out) - 1


def test_a_long_prompts_index_builds_off_the_engine_thread(monkeypatch, tmp_path):
    """A prompt past ``BULK`` builds its index on the builder's thread: its stream decodes on MTP drafts until the
    build is in, then drafts from its text; a stream that finishes first leaves its build unread; shutdown stops the
    builder."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    monkeypatch.setattr(session_index, "BULK", 32)
    release, real = threading.Event(), session_index.build

    def slow(prompt, k):
        assert release.wait(60)
        return real(prompt, k)

    monkeypatch.setattr(session_index, "build", slow)
    w = _model()
    g = torch.Generator().manual_seed(4)
    prompts = [torch.randint(1, V, (90,), generator=g).tolist(), torch.randint(1, V, (60,), generator=g).tolist()]
    samplings = [Sampling(seed=2, top_k=20, top_p=0.95), None]
    counts = [36, 6]
    refs = _solo(w, prompts, samplings, counts)
    Texts({tuple(p): r for p, r in zip(prompts, refs)}, seed=1, kinds=("truth",)).install(monkeypatch)
    taken = _taken(monkeypatch)
    _switch(monkeypatch, tmp_path, draft_rule="new", suffix="new")
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.5, prefill_rows=64)
    streams = [Stream(list(p), c, smp) for p, smp, c in zip(prompts, samplings, counts)]
    futures = []
    try:
        for s in streams:
            dec.admit(s)
            futures.append(s.session.future)
        rounds = 0
        while dec.live():
            if rounds == 8:
                assert not taken and not streams[1].session    # MTP drafts only so far; the short one is done
                release.set()
                futures[0].result(30)
            dec.finish(dec.round())
            rounds += 1
        assert [s.out for s in streams] == refs
        assert taken and all(sid == streams[0].sid for *_, sid, _ in taken)
    finally:
        release.set()
        dec.close_drafting()
    assert futures[1].cancelled() or futures[1].done()
    assert dec.session_builder.pool is None and not any(t.name.startswith("tensorfold-session-index")
                                                         for t in threading.enumerate())


@pytest.mark.parametrize("rule", ["new", "old"])         # "old": A/B baseline, remove after selection
def test_a_constrained_stream_with_text_drafts_emits_its_solo_tokens(monkeypatch, tmp_path, rule):
    """A grammar prunes text drafts no accepted path can hold (true continuations pass, garbage is cut) and the rule's
    chances with them; the constrained stream and its unconstrained neighbour emit their solo tokens."""

    pytest.importorskip("xgrammar")
    from toy_grammar import toy

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    w = _model()
    grammars, compiled = toy(V)
    prompts, samplings, counts = [PROMPTS[1], LOOP], [Sampling(seed=1234, top_k=20, top_p=0.95), None], [24, 20]
    constraint = lambda: grammars.constraint(compiled)  # noqa: E731
    refs = _solo(w, prompts, samplings, counts, constraints=[constraint, lambda: None])
    Texts({tuple(p): r for p, r in zip(prompts, refs)}, seed=5, kinds=("truth", "prefix", "garbage")) \
        .install(monkeypatch)
    _switch(monkeypatch, tmp_path, draft_rule=rule, suffix="new")
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=4, confidence=0.5, prefill_rows=16)
    streams = [Stream(list(prompts[0]), counts[0], samplings[0], constraint=constraint()),
               Stream(list(prompts[1]), counts[1], samplings[1])]
    _run(dec, streams)
    assert [s.out for s in streams] == refs and streams[0].accepted > 0


@pytest.mark.parametrize("broken", [False, True])
@pytest.mark.parametrize("where", ["verify", "mtp"])
def test_a_failed_frame_still_wins_when_streams_draft_from_text(monkeypatch, tmp_path, where, broken):
    """A NaN query scale fails a step's codec status while streams hold text drafts (the rule's pick of their first
    MTP draft shares the absorb's readback): the round raises the codec failure, publishes nothing, and reset slots
    serve again. ``broken``: one stream's session query also fails in that round; its contained drafting failure
    does not hide the frame's."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    w = _model()
    prompts = [LOOP, PROMPTS[1]]
    samplings = [SAMPLINGS[1], None]
    refs = _solo(w, prompts, samplings, [16, 16], RQ, capacity=256)
    Texts({tuple(p): r for p, r in zip(prompts, refs)}, seed=2, kinds=("truth",)).install(monkeypatch)
    _switch(monkeypatch, tmp_path, draft_rule="new", suffix="new")
    emitted, real_propose = [], SessionIndex.propose
    dec = MultiDecoder(w, slots=2, capacity=256, depth=3, confidence=0.5, kv_dtype=RQ)
    streams = [Stream(list(p), 16, smp, emit=lambda new: emitted.extend(new)) for p, smp in zip(prompts, samplings)]
    for s in streams:
        dec.admit(s)
    dec.finish(dec.round())
    dec.finish(dec.round())
    assert any(s.drafts for s in streams)
    before = list(emitted)
    if broken:
        real = SessionIndex.propose

        def propose(index, count):
            if index.tokens[:len(prompts[1])].tolist() == prompts[1]:
                raise MemoryError("injected drafting failure")
            return real(index, count)

        monkeypatch.setattr(SessionIndex, "propose", propose)
    scale = w.mtp.layer.attn.q_scale if where == "mtp" else w.layers[1].attn.q_scale
    saved = scale.clone()
    scale.fill_(float("nan"))
    try:
        with pytest.raises(KVNumericError, match="layer=MTP" if where == "mtp" else "layer=1"):
            dec.round()
    finally:
        scale.copy_(saved)
        monkeypatch.setattr(SessionIndex, "propose", real_propose)
    assert emitted == before
    dec.drop()
    again = Stream(list(prompts[1]), 16, None)
    dec.admit(again)
    while dec.live():
        dec.finish(dec.round())
    assert again.error is None and again.out == refs[1]


def test_admission_keeps_the_context_edge_clamp_under_every_switch(monkeypatch, tmp_path):
    """X9 (design 2.3): a prompt near the context edge gets the same token limit whatever the drafting switches; an
    unknown "suffix" value fails the request's admission and frees its slot."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    w = _model()
    counts = set()
    for rule in ("new", "old"):
        for suffix in ("new", "old"):
            _switch(monkeypatch, tmp_path, draft_rule=rule, suffix=suffix)
            dec = MultiDecoder(w, slots=1, capacity=256, depth=4, prefill_rows=64)
            s = Stream(list(range(1, 241)), 100, None)
            dec.admit(s)
            counts.add(s.count)
            assert (s.session is not None) == (suffix == "new") and (s.copies is not None) == (suffix == "old")
            dec.drop()
    assert counts == {256 - 240 - 4 - 1}
    _switch(monkeypatch, tmp_path, suffix="bogus")       # an unknown value fails the admission, its slot freed
    dec = MultiDecoder(w, slots=1, capacity=256, depth=4, prefill_rows=64)
    with pytest.raises(ValueError, match="suffix"):
        dec.admit(Stream(list(range(1, 241)), 100, None))
    assert len(dec.free) == 1 and not dec.filling and not dec.fills


FAILURES = [("new", "build"), ("old", "build"), ("new", "join"), ("new", "propose"), ("old", "propose"),
            ("new", "rule"), ("new", "copy"), ("old", "copy")]                # "old": A/B baseline, remove after


@pytest.mark.parametrize("rule,where", FAILURES)
def test_a_drafting_failure_ends_its_request_alone(monkeypatch, tmp_path, rule, where):
    """A stream's drafting fails (its session index's background build, raised where a round adopts it; its query,
    as its prompt joins or in a later round; the cost-aware rule's decision for it; its copy index): that request
    ends with an error naming drafting, whose cause is the failure, having emitted a prefix of its solo tokens; the
    other streams emit their solo tokens; its slot is freed and the decoder serves the next request."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    if where == "build":                                 # the target's 6-token prompt builds in the background
        monkeypatch.setattr(session_index, "BULK", 6)
    w = _vocab_model(9)
    prompts, counts = [LOOP, PROMPTS[1], PROMPTS[0], PROMPTS[3]], [20, 24, 18, 16]
    refs = _solo(w, prompts, SAMPLINGS[:4], counts)
    target = 0 if where == "rule" else 1                 # the rule's failure: the one greedy drafting stream's
    armed, calls, cause = [True], [0], MemoryError("injected drafting failure")

    def fires(prompt) -> bool:
        if not armed[0] or list(prompt[:len(prompts[target])]) != prompts[target]:
            return False
        calls[0] += 1
        return calls[0] >= (3 if where == "propose" else 1)    # "propose": a later round's query

    real_build, real_propose, real_chance = session_index.build, SessionIndex.propose, draft_rule.Table.chance
    greedy_mtp = []                                      # the decoder's greedy MTP table (its predictor's)

    def build(prompt, k):
        if where == "build" and fires(prompt):
            raise cause
        return real_build(prompt, k)

    def propose(index, count):
        if where in ("join", "propose") and index.ready() and fires(index.tokens[:index.n].tolist()):
            raise cause
        return real_propose(index, count)

    def chance(self, depth, score):
        if where == "rule" and armed[0] and greedy_mtp and self is greedy_mtp[0]:
            raise cause
        return real_chance(self, depth, score)

    monkeypatch.setattr(session_index, "build", build)
    monkeypatch.setattr(SessionIndex, "propose", propose)
    monkeypatch.setattr(draft_rule.Table, "chance", chance)
    from tensorfold.families.qwen3_5.cuda.decode import CopyIndex

    real_copy = CopyIndex.propose

    def copy(index, context, max_nodes=127):
        if where == "copy" and fires(context):
            raise cause
        return real_copy(index, context, max_nodes)

    monkeypatch.setattr(CopyIndex, "propose", copy)
    _switch(monkeypatch, tmp_path, draft_rule=rule, suffix="old" if where == "copy" else "new")
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.5, prefill_rows=16)
    greedy_mtp.append(dec.predictor.mtp_tables[True])
    streams = [Stream(list(p), c, smp) for p, smp, c in zip(prompts, SAMPLINGS, counts)]
    try:
        _run(dec, streams)
        failed = streams[target]
        assert isinstance(failed.error, RuntimeError) and "drafting for this request failed" in str(failed.error)
        assert failed.error.__cause__ is cause and failed.done
        assert failed.out and failed.out == refs[target][:len(failed.out)] and len(failed.out) < counts[target]
        if where == "join":                              # it failed drafting as its prompt joined: its first token
            assert len(failed.out) == 1
        assert all(s.error is None and s.out == ref for i, (s, ref) in enumerate(zip(streams, refs)) if i != target)
        assert not dec.streams and not dec.filling
        assert len(dec.free) + len({id(k[1]) for k in dec.kept if all(k[1] is not f for f in dec.free)}) == 4
        armed[0] = False                                 # the decoder goes on serving
        again = Stream(list(prompts[target]), counts[target], SAMPLINGS[target])
        _run(dec, [again])
        assert again.error is None and again.out == refs[target]
    finally:
        dec.close_drafting()


def test_drafting_judges_costs_at_the_streams_the_next_round_will_have(monkeypatch, tmp_path):
    """The cost-aware rule drafts for the next round at that round's expected live streams: never more than the
    round-start trim then finds (streams that finish in a round, wait or ended are not counted; a prompt that joins
    may add one), and exactly those when none joins, here as streams finish one by one."""

    w = _model()
    prompts, counts = [PROMPTS[1], PROMPTS[0], PROMPTS[3]], [30, 12, 20]
    refs = _solo(w, prompts, SAMPLINGS[:3], counts)
    _switch(monkeypatch, tmp_path, draft_rule="new")
    calls, at = [], [0]
    real_chain, real_trim = MultiDecoder._chain, MultiDecoder._trim

    def chain(self, active, logits, room, base, words, rule, texts, joined, live):
        calls.append(("chain", at[0], live))
        return real_chain(self, active, logits, room, base, words, rule, texts, joined, live)

    def trim(self, live):
        calls.append(("trim", at[0], len(live)))
        return real_trim(self, live)

    monkeypatch.setattr(MultiDecoder, "_chain", chain)
    monkeypatch.setattr(MultiDecoder, "_trim", trim)
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, confidence=0.5, prefill_rows=16)
    streams = [Stream(list(p), c, smp) for p, smp, c in zip(prompts, SAMPLINGS, counts)]
    _run(dec, streams, each_round=lambda d, r: at.__setitem__(0, r))
    assert [s.out for s in streams] == refs
    pairs = []
    for i, (kind, r, actual) in enumerate(calls):
        if kind != "trim":
            continue
        drafted = [x for x in calls[:i] if x[0] == "chain" and x[1] in (r - 1, r)]
        if drafted:
            pairs.append((drafted[-1][2], actual))
    assert pairs and all(predicted <= actual for predicted, actual in pairs)
    assert (2, 2) in pairs and (1, 1) in pairs           # after the first and the second stream finished


def test_a_failed_admission_step_frees_its_slot_before_any_index_build(monkeypatch, tmp_path):
    """The prompt's kept points are found before its session index is built: a failure there frees the slot and
    leaves no build behind."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    monkeypatch.setattr(session_index, "BULK", 8)
    _switch(monkeypatch, tmp_path, suffix="new")

    def points(ids):
        raise ValueError("no message starts")

    dec = MultiDecoder(_model(), slots=1, capacity=256, depth=4, prefill_rows=64, points=points)
    with pytest.raises(ValueError, match="message starts"):
        dec.admit(Stream(list(range(1, 41)), 10, None))
    assert len(dec.free) == 1 and not dec.filling and not dec.fills and dec.session_builder.pool is None

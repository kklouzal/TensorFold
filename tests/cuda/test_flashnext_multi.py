"""Flash Next concurrent rounds: each row gets its own stream's bits, and streams together emit what each does alone."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.families.qwen4_exp.cuda import attention as attn_mod, qmm  # noqa: E402

from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, mtp_decode, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import commit, compute, forward, stage  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import Buffers, State  # noqa: E402

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]


def test_segments_give_each_stream_its_own_rows():
    w = _model()
    b = Buffers(w, 32, 1024)
    chains = [[401, 33, 2048], [5], [77, 1500, 9, 10, 11], [3, 4]]
    states = []
    for prompt in PROMPTS:
        st = State(w, 1024, 16)
        forward(w, st, b, prompt)
        commit(w, st, b, len(prompt), len(prompt))
        states.append(st)
    alone = [st.clone() for st in states]
    ref = []
    for st, chain in zip(alone, chains):
        lg = forward(w, st, b, chain)
        ref.append((lg[:len(chain)].clone(), b.streams[:len(chain)].clone()))
        keep = max(1, len(chain) // 2)
        commit(w, st, b, len(chain), keep)
        ref[-1] += (forward(w, st, b, [chain[keep] if keep < len(chain) else 1])[0].clone(),)
    segs = stage(w, b, list(zip(states, chains)))
    lg = compute(w, segs, b)
    for (st, a0, a1), (rl, rs, _) in zip(segs, ref):
        assert torch.equal(lg[a0:a1], rl) and torch.equal(b.streams[a0:a1], rs), (a0, a1)
    for (st, a0, a1), chain in zip(segs, chains):
        commit(w, st, b, a1 - a0, max(1, (a1 - a0) // 2), at=a0)
    for st, chain, (_, _, nxt) in zip(states, chains, ref):
        keep = max(1, len(chain) // 2)
        assert torch.equal(forward(w, st, b, [chain[keep] if keep < len(chain) else 1])[0], nxt)


@pytest.mark.parametrize("confidence,vocab,kv_dtype", [(0.0, False, "bf16"), (0.3, False, "bf16"), (0.3, True, "bf16"),
                                                       (0.3, True, "int8"), (0.0, False, "int4")])
def test_streams_decoded_together_equal_each_alone(confidence, vocab, kv_dtype):
    w = _model()
    if vocab:                                            # drafts over a token subset (the real model's draft head)
        words, scales, biases = qmm.to_mlx(w.head)
        ids = torch.arange(1, V, 3, device="cuda")
        w.draft_ids = ids
        w.draft_head = qmm.make_q4(words[ids], scales[ids], biases[ids])
    samplings = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95), None]
    refs = []
    for prompt, sampling in zip(PROMPTS, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        first = prefill(e, prompt, sampling)
        refs.append(serial_decode(e, first, 20, sampling).tokens)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=confidence, kv_dtype=kv_dtype)
    assert all(st.kv_dtype == kv_dtype and st.kc[0].dtype == kv_dtype for st in dec.free)
    streams = []
    for i, (prompt, sampling) in enumerate(zip(PROMPTS, samplings)):
        got: list[int] = []
        s = Stream(prompt, 20, sampling, draft=i != 3, emit=lambda new, got=got: got.extend(new))
        dec.admit(s)
        streams.append((s, got))
    while dec.live():
        dec.finish(dec.round())
    for i, (s, got) in enumerate(streams):
        assert got == refs[i] and s.out == refs[i], i
        assert s.min_rows >= (2 if s.draft else 1), (i, s.min_rows)
    assert len(dec.free) + len({id(k[1]) for k in dec.kept}) == 4 and not dec.live()     # every slot back or kept


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8", "int4"])
@pytest.mark.parametrize("sampling", [None, Sampling(seed=31, top_k=20, top_p=0.95)])
def test_prompts_that_extend_a_finished_stream_resume_from_its_slot(sampling, kv_dtype):
    w = _model()
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype)

    def run(prompt, count, draft=True):
        s = Stream(list(prompt), count, sampling, draft=draft)
        dec.admit(s)
        while dec.live():
            dec.finish(dec.round())
        return s

    def fresh(prompt, count):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        return serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens

    first = run(PROMPTS[1], 12)
    longer = PROMPTS[1] + first.out[:-1] + [42, 43]          # the reply's committed tokens, then new ones
    warm = run(longer, 10)
    assert warm.cached == len(PROMPTS[1]) - 1 and warm.out == fresh(longer, 10)   # kept one token early
    same = run(longer, 10)                                    # the same prompt again: all but its last token kept
    assert same.cached == len(longer) - 1 and same.out == warm.out
    again = run(longer, 10)                                   # and a third time: every resend hits, not every other
    assert again.cached == len(longer) - 1 and again.out == warm.out
    ext = PROMPTS[0] + [7, 8]                                 # a prompt kept at admission, extended
    run(PROMPTS[0], 6)
    other = run(ext, 8)
    assert other.cached > 0 and other.out == fresh(ext, 8)
    serial = run(longer, 10, draft=False)
    assert serial.cached == 0 and serial.out == warm.out


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_packed_passes_keep_each_prompt_one_token_early(kv_dtype):
    """Prompts sharing passes keep a fresh prefill's state one token early; a resend or a next turn resumes there."""

    w = _model(5)
    g = torch.Generator().manual_seed(13)
    long = torch.randint(1, V, (70,), generator=g).tolist()
    # 32-row passes: three points inside the first, one inside the long prompt's third piece, one ending a piece
    prompts = [PROMPTS[3], PROMPTS[1], [13, 400, 9, 21], long, [(5 * i + 2) % (V - 1) + 1 for i in range(12)]]
    samplings = [None, Sampling(seed=8, top_k=20, top_p=0.95), None, Sampling(seed=9, top_k=20, top_p=0.95), None]

    def fresh(prompt, sampling):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        return serial_decode(e, prefill(e, prompt, sampling), 12, sampling).tokens

    dec = MultiDecoder(w, slots=5, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=32)
    streams = [Stream(p, 12, smp) for p, smp in zip(prompts, samplings)]
    for s in streams:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == [fresh(p, smp) for p, smp in zip(prompts, samplings)]
    for p in prompts:                                        # what a fresh prefill of all but the last token leaves
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        prefill(e, p[:-1], None)
        want = e.st.snapshot()
        _, _, snap, tail = next(k for k in dec.kept if k[0] == p[:-1])
        assert all(torch.equal(snap[k], want[k]) for k in ("rec", "conv", "ple_tail")), len(p)
        assert (snap["pos"], snap["mtp_len"]) == (want["pos"], want["mtp_len"]), len(p)
        assert torch.equal(tail, e.last_streams), len(p)
    for p, smp in zip(prompts, samplings):
        for q in (p, p[:-1] + [271, 77]):                    # the same prompt again, then a next turn
            s = Stream(list(q), 12, smp)
            dec.admit(s)
            while dec.live():
                dec.finish(dec.round())
            assert s.cached == len(p) - 1 and s.out == fresh(q, smp), (len(p), q[-2:])


def _t_at_small_windows(monkeypatch) -> None:
    """T at every block count, its tile cut to 512 keys (one warp) so this file's 1,024-block windows still split
    each row over slices (the placement seeded by earlier slices' counts)."""

    monkeypatch.setattr(attn_mod, "MULTI_MIN_BLOCKS", 0)
    monkeypatch.setattr(attn_mod, "SELECT_TB", 512)


def _qsa_variant(monkeypatch, variant: str) -> dict:
    """The QSA path a test's rounds take: "batched" (every stream's rows in one select; T past its threshold) or
    "T" (T at every block count). Counts the launches of T and the slices of its widest split."""

    if variant == "T":
        _t_at_small_windows(monkeypatch)
    counts = {"T": 0, "slices": 0}
    real_multi = attn_mod._select_multi

    def multi(*a, **kw):
        counts["T"] += 1
        return real_multi(*a, **kw)
    monkeypatch.setattr(attn_mod, "_select_multi", multi)

    def slices(*a, real=attn_mod._slices):
        sl, n = real(*a)
        counts["slices"] = max(counts["slices"], n)
        return sl, n
    monkeypatch.setattr(attn_mod, "_slices", slices)
    return counts


@pytest.mark.parametrize("variant", ["batched", "T"])
@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_sparse_streams_decode_together_as_alone(monkeypatch, kv_dtype, variant):
    """Past the attention budget a long stream selects blocks beside short dense ones, each emitting its solo run
    (solo runs select with the one-CTA kernels; the rounds by each variant's)."""

    w = _model(7)
    g = torch.Generator().manual_seed(9)
    prompts = [torch.randint(1, V, (2400,), generator=g).tolist(), PROMPTS[0], PROMPTS[1]]
    samplings = [Sampling(seed=5, top_k=20, top_p=0.95), None, Sampling(seed=6, top_k=20, top_p=0.95)]
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=256, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 16, sampling).tokens)
    counts = _qsa_variant(monkeypatch, variant)
    dec = MultiDecoder(w, slots=3, capacity=4096, depth=3, confidence=0.3, kv_dtype=kv_dtype)
    streams = [Stream(p, 16, smp) for p, smp in zip(prompts, samplings)]
    for s in streams:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == refs
    assert (counts["T"] > 0) == (variant == "T"), counts
    assert counts["slices"] > 1 or variant != "T", counts


@pytest.mark.parametrize("graphs", [False, True])
def test_long_stream_selects_the_same_blocks_through_t(monkeypatch, graphs):
    """One stream past the budget: logits of every window size, tokens and MTP-drafted tokens are the same when its
    selection runs T (forced at this small window, captured in the graphs too) as through ``_select``."""

    w = _model(7)
    g = torch.Generator().manual_seed(11)
    prompt = torch.randint(1, V, (2600,), generator=g).tolist()
    sampling = Sampling(seed=8, top_k=20, top_p=0.95)

    def run():
        e = Engine(w, capacity=4096, max_rows=8, prefill_rows=256, graphs=graphs)
        first = prefill(e, prompt, sampling)
        logits = [e.forward([first] + list(range(3, 3 + n)))[:n + 1].clone() for n in range(6)]   # uncommitted
        tokens = serial_decode(e, first, 12, sampling).tokens
        prefill(e, prompt, sampling)
        return logits, tokens, mtp_decode(e, first, 12, sampling, depth=3, confidence=0.0).tokens

    want = run()
    _t_at_small_windows(monkeypatch)
    calls = []
    real = attn_mod._slices
    monkeypatch.setattr(attn_mod, "_slices", lambda *a: calls.append(real(*a)[1]) or real(*a))
    got = run()
    assert calls and max(calls) > 1 and all(torch.equal(a, b) for a, b in zip(got[0], want[0]))
    assert got[1:] == want[1:] and want[1] == want[2]


def test_streams_keep_their_bits_when_their_caches_move():
    """A stream's caches may be reallocated between rounds (growth): the step tables read addresses each step."""

    w = _model()
    samplings = [Sampling(seed=11, top_k=20, top_p=0.95), None]
    refs = []
    for prompt, sampling in zip(PROMPTS[:2], samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 24, sampling).tokens)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3)
    streams = [Stream(p, 24, smp) for p, smp in zip(PROMPTS[:2], samplings)]
    for s in streams:
        dec.admit(s)
    rounds = 0
    while dec.live():
        dec.finish(dec.round())
        rounds += 1
        if rounds % 2 == 0:                          # move every live stream's caches and states to new storage
            for s in streams:
                st = s.st
                for kc in st.kc + [st.mtp_kc]:
                    kc.k, kc.v, kc.ks, kc.vs = kc.k.clone(), kc.v.clone(), kc.ks.clone(), kc.vs.clone()
                st.ikc = [t.clone() for t in st.ikc]
                st.pooled = [t.clone() for t in st.pooled]
                st.mtp_ikc, st.mtp_pooled = st.mtp_ikc.clone(), st.mtp_pooled.clone()
                st.conv, st.rec = st.conv.clone(), st.rec.clone()
    assert [s.out for s in streams] == refs


@pytest.mark.parametrize("kv_dtype", ["bf16", "int4"])
def test_streams_that_grow_past_their_first_rows_equal_each_alone(kv_dtype):
    """Slots start at 256 rows; two streams decoding past them grow (a copy of every committed row) mid-reply."""

    w = _model()
    samplings = [None, Sampling(seed=5, top_k=20, top_p=0.95)]
    prompts = [[(7 * i + 3) % (V - 1) + 1 for i in range(240)], [(11 * i + 5) % (V - 1) + 1 for i in range(250)]]
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 40, sampling).tokens)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype)
    streams = []
    for prompt, sampling in zip(prompts, samplings):
        s = Stream(list(prompt), 40, sampling, draft=True)
        dec.admit(s)
        streams.append(s)
    assert all(s.st.capacity == 256 for s in streams)
    while dec.live():
        dec.finish(dec.round())
    for s, ref in zip(streams, refs):
        assert s.out == ref and s.st.version >= 1


def test_a_window_and_a_prompt_pass_share_each_layers_experts_and_keep_their_bits(monkeypatch):
    """One forward for a decode window and a prompt pass launches each layer's experts once for both rows, and both
    get the rows they get apart."""

    from tensorfold.families.qwen4_exp.cuda import forward as fwd

    w = _model()
    db, pb = Buffers(w, 16, 1024, moe_prefill=True), Buffers(w, 48, 1024, prefill=True)
    chains, pieces = [[401, 33, 2048], [5, 6]], [PROMPTS[2] + [70, 71], PROMPTS[3]]
    dstates = []
    for prompt in PROMPTS[:2]:
        st = State(w, 1024, 16)
        forward(w, st, db, prompt)
        commit(w, st, db, len(prompt), len(prompt))
        dstates.append(st)
    pstates = [State(w, 1024, 16) for _ in pieces]

    def run(mixed: bool):
        d, p = [st.clone() for st in dstates], [st.clone() for st in pstates]
        segs, psegs = stage(w, db, list(zip(d, chains))), stage(w, pb, list(zip(p, pieces)))
        ends = [a1 - 1 for _, _, a1 in psegs]
        if mixed:
            lg, heads = fwd.compute_mixed(w, segs, db, psegs, pb, ends=ends)
        else:
            lg = compute(w, segs, db).clone()
            heads = compute(w, psegs, pb, logits=True, ends=ends)
        rd, rp = segs[-1][2], psegs[-1][2]
        return lg[:rd].clone(), db.streams[:rd].clone(), heads[:len(ends)].clone(), pb.streams[:rp].clone()

    apart = run(False)
    calls, real = [], fwd.moe_block
    monkeypatch.setattr(fwd, "moe_block", lambda layer, w_, b, R: calls.append(R) or real(layer, w_, b, R))
    together = run(True)
    assert calls == [sum(map(len, chains)) + sum(map(len, pieces))] * len(w.layers)
    assert all(torch.equal(x, y) for x, y in zip(together, apart))


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_prompts_fill_between_rounds_while_streams_decode(kv_dtype):
    """A prompt admitted beside decoding streams prefills a chunk a round, in the round's own forward, while they
    keep decoding; a burst queues behind it, oldest first; every stream emits its solo run."""

    w = _model()
    g = torch.Generator().manual_seed(11)
    long = torch.randint(1, V, (70,), generator=g).tolist()           # five 16-row chunks
    prompts = [PROMPTS[0], long, PROMPTS[1], PROMPTS[0] + [7, 8]]
    samplings = [Sampling(seed=3, top_k=20, top_p=0.95), None, Sampling(seed=4, top_k=20, top_p=0.95), None]
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 24, sampling).tokens)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=16)
    first = Stream(prompts[0], 24, samplings[0])
    dec.admit(first)
    assert dec.live() == 1 and not first.out                          # queued: the prompt fills in the rounds
    dec.finish(dec.round())                                           # alone: the whole prompt, then a round
    assert len(first.out) > 1
    rest = [Stream(p, 24, smp) for p, smp in zip(prompts[1:], samplings[1:])]
    for s in rest:
        dec.admit(s)
    grew = []
    while dec.filling and rest[0] in dec.filling:
        before = len(first.out)
        dec.finish(dec.round())
        grew.append(len(first.out) - before)
    assert len(grew) >= 4 and all(n > 0 for n in grew[:3])           # a chunk a round; the first stream decodes
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in [first, *rest]] == refs


def test_prompts_fill_between_rounds_where_experts_cannot_share_a_launch(monkeypatch):
    """Experts that share no launch (NVFP4, EXL3): a long prompt fills between rounds beside a decoding stream."""

    from tensorfold.families.qwen4_exp.cuda import multi

    monkeypatch.setattr(multi, "converges", lambda w: False)
    w = _model()
    g = torch.Generator().manual_seed(11)
    prompts = [PROMPTS[0], torch.randint(1, V, (70,), generator=g).tolist()]          # the second: five passes
    samplings = [Sampling(seed=3, top_k=20, top_p=0.95), None]
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 24, sampling).tokens)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, prefill_rows=16)
    assert not dec.converged and dec.pbuf.rows == 16
    streams = [Stream(p, 24, smp) for p, smp in zip(prompts, samplings)]
    dec.admit(streams[0])
    dec.finish(dec.round())
    dec.admit(streams[1])
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == refs


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
@pytest.mark.parametrize("decoding", [False, True])
def test_packed_prompt_passes_keep_each_prompt_its_solo_run(kv_dtype, decoding):
    """Several prompts share each prompt pass (short ones end together, a long one spans passes), beside a decoding
    stream or alone; every stream emits its solo run."""

    w = _model(5)
    g = torch.Generator().manual_seed(12)
    long = torch.randint(1, V, (70,), generator=g).tolist()
    prompts = [PROMPTS[3], PROMPTS[1], [13, 400, 9, 21], long, PROMPTS[0]]
    samplings = [None, Sampling(seed=8, top_k=20, top_p=0.95), None, Sampling(seed=9, top_k=20, top_p=0.95), None]
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), 18, sampling).tokens)
    dec = MultiDecoder(w, slots=5, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=32)
    streams = [Stream(p, 18, smp) for p, smp in zip(prompts, samplings)]
    if decoding:
        dec.admit(streams[0])
        dec.finish(dec.round())
    for s in streams[1 if decoding else 0:]:
        dec.admit(s)
    ends = []
    while dec.filling:
        before = len(dec.filling)
        dec.finish(dec.round())
        ends.append(before - len(dec.filling))
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == refs
    assert max(ends) >= 2                                             # a pass that ended two prompts at once


def test_waiting_stream_keeps_its_recurrence_when_other_rounds_reuse_shared_scratch(monkeypatch):
    from test_flashnext_forward import _model
    from tensorfold.families.qwen4_exp.cuda import multi

    monkeypatch.setattr(multi, "FIRST", 8)
    monkeypatch.setattr(multi, "STEP", 16)
    real = MultiDecoder._grow
    def grow(dec, st, rows, *, alone=False):
        blocked = dec.streams.get(1)
        if (getattr(dec, "test_wait", False) and blocked is not None and st is blocked.st
                and rows > st.capacity and not alone):
            return False
        return real(dec, st, rows, alone=alone)
    monkeypatch.setattr(MultiDecoder, "_grow", grow)
    w = _model()
    dec = MultiDecoder(w, slots=3, capacity=1024, depth=3, prefill_rows=64)
    dec.test_wait = True
    streams = [Stream(PROMPTS[0], 60, stop_eos=False), Stream(PROMPTS[1], 20, stop_eos=False)]
    for s in streams:
        dec.admit(s)
    waits = 0
    while dec.live():
        dec.finish(dec.round())
        if streams[1].waiting:
            waits += 1
        if waits >= 4:
            dec.test_wait = False
    assert waits >= 4
    for s in streams:
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=64)
        assert s.out == serial_decode(e, prefill(e, s.prompt, None), s.count, None).tokens


@pytest.mark.parametrize("kv_dtype", ["bf16", "int8"])
def test_several_kept_points_in_one_pass_each_keep_a_fresh_prefills_state(kv_dtype):
    """A pass may hold up to CUTS kept points (message starts, the prompt end): each keeps the state a fresh prefill of
    the prompt up to it leaves, the reply is the one decoded alone, and a prompt diverging after a point resumes there.
    A point past CUTS ends the pass instead (one 128-row piece here holds four)."""

    from tensorfold.families.qwen4_exp.cuda.prompt_plan import CUTS

    w = _model(5)
    g = torch.Generator().manual_seed(17)
    prompt = torch.randint(1, V, (700,), generator=g).tolist()
    marks = [300, 310, 333, 350]                             # inside the piece [256, 384), one past CUTS
    assert CUTS == 3
    sampling = Sampling(seed=4, top_k=20, top_p=0.95)

    def fresh(p, count=12):
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        return serial_decode(e, prefill(e, p, sampling), count, sampling).tokens

    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, confidence=0.3, kv_dtype=kv_dtype, prefill_rows=128,
                       points=lambda ids: [p for p in marks if p < len(ids)])
    s = Stream(list(prompt), 12, sampling)
    dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert s.out == fresh(prompt)
    for p in marks + [len(prompt) - 1]:                      # the prompt end: kept one token early
        e = Engine(w, capacity=1024, max_rows=8, prefill_rows=16, kv_dtype=kv_dtype)
        prefill(e, prompt[:p], None)
        want = e.st.snapshot()
        _, _, snap, tail = next(k for k in dec.kept if k[0] == prompt[:p])
        assert all(torch.equal(snap[k], want[k]) for k in ("rec", "conv", "ple_tail")), p
        assert (snap["pos"], snap["mtp_len"]) == (want["pos"], want["mtp_len"]), p
        assert torch.equal(tail, e.last_streams), p
    branch = prompt[:333] + [271, 77, 5]                     # diverges right after a point inside the pass
    t = Stream(list(branch), 12, sampling)
    dec.admit(t)
    while dec.live():
        dec.finish(dec.round())
    assert t.cached == 333 and t.out == fresh(branch)

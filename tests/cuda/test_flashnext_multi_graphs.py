"""Flash Next concurrent rounds through captured CUDA graphs (benchmark switch "graphs"): the static tables and fixed
launch geometry give the old eager bits, every captured key replays them, and decoding emits each stream's solo run
through every round shape (joins, finishes, copy drafts, sparse rows, prompt passes, growth, failures)."""

import json

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.cuda import ab_switch  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import attn_multi, gdn_multi, multi_graphs  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.decode import Engine, prefill, serial_decode  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.forward import _kv_check, compute, stage  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.mtp import mtp_compute, mtp_stage  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.state import KVNumericError  # noqa: E402

RQ = "rotorquant6-norm"
PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=0.95), None]


def _switch(monkeypatch, tmp_path, **values) -> None:
    control = tmp_path / "ab.json"
    control.write_text(json.dumps(values))
    monkeypatch.setattr(ab_switch, "_path", str(control))
    monkeypatch.setattr(ab_switch, "_values", {})
    monkeypatch.setattr(ab_switch, "_stamp", None)


def _solo(w, prompts, samplings, count, kv_dtype, capacity=4096, prefill_rows=256):
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=prefill_rows, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), count, sampling).tokens)
    return refs


def _prefilled(w, prompts, kv_dtype, capacity):
    """States holding each prompt's committed rows (their own caches), for forward-level comparisons."""

    states = []
    for prompt in prompts:
        e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=256, kv_dtype=kv_dtype)
        prefill(e, prompt, None)
        states.append(e.st)
    return states


def _frame(states, segs, mtp: bool):
    """What a forward wrote of each stream's frame: its cache rows (and scales), index keys, pooled blocks, status."""

    got = []
    for st, a0, a1 in segs:
        p0 = st.mtp_len if mtp else st.pos
        n = a1 - a0
        caches = [(st.mtp_kc, st.mtp_ikc, st.mtp_pooled)] if mtp else list(zip(st.kc, st.ikc, st.pooled))
        for kc, ikc, pooled in caches:
            got += [kc.k[p0:p0 + n], kc.v[p0:p0 + n], kc.ks[p0:p0 + n], kc.vs[p0:p0 + n], ikc[p0:p0 + n],
                    pooled[p0 // 4:(p0 + n) // 4 + 1]]
        if st.kv_status is not None:
            got.append(st.kv_status)
        got.append(st.rec)
    return [t.clone() for t in got]


def _verify(dec, w, states, windows, held, parity, mode):
    """One verify forward over clones of ``states``: "old" (fresh tables, the step's geometry), "eager" (static
    tables, fixed geometry) or "graph" (the captured key replayed); its logits, streams, frames and DeltaNet rows."""

    sts = [st.clone() for st in states]
    b = dec.buf
    segs = stage(w, b, list(zip(sts, windows)))
    dec.gdn.parity = parity
    static = mode != "old"
    b.gdn_tables = gdn_multi.Tables(w, dec.gdn, segs, held, static=dec.round_tables if static else None)
    b.attn_step = attn_multi.Step(w, segs, mtp=False, tables=dec.step_tables if static else None)
    try:
        if mode == "graph":
            logits = dec.graphs.verify_forward(segs)
            assert logits is not None, [len(segs), segs[-1][2], parity]
        else:
            logits = compute(w, segs, b, check=False)
        _kv_check(segs)
    finally:
        b.gdn_tables = b.attn_step = None
    R = segs[-1][2]
    sc = dec.gdn
    rows = [sc.k[parity, :, :R], sc.v[parity, :, :R], sc.g[parity, :, :R], sc.beta[parity, :, :R]]
    return [logits[:R].clone(), b.streams[:R].clone(), *_frame(sts, segs, False), *[t.clone() for t in rows]]


def _equal(a, b, what):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape and torch.equal(x, y), (what, i)


@pytest.fixture(scope="module")
def captured():
    """A captured 3-slot decoder over a model whose QSA engages past 2,048 keys, and prefilled states: one long
    (2,400 rows: sparse), two short (dense)."""

    w = _model(7)
    dec = MultiDecoder(w, slots=3, capacity=4096, depth=3, confidence=0.3, kv_dtype=RQ, graphs=True)
    dec.capture_graphs()
    g = torch.Generator().manual_seed(9)
    long = torch.randint(1, V, (2400,), generator=g).tolist()
    states = _prefilled(w, [long, PROMPTS[0], PROMPTS[1]], RQ, 4096)
    return w, dec, states


@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("n,split", [(1, [1]), (1, [4]), (2, [1, 4]), (2, [4, 3]), (3, [1, 1, 1]), (3, [4, 2, 4]),
                                     (3, [4, 4, 4])])
def test_static_tables_and_fixed_geometry_and_replays_give_the_old_bits(captured, n, split, parity):
    """T1/T4: per key, the eager forward on static tables at the fixed geometry, and its captured replay, write the
    old path's logits, streams, cache rows, scales, index keys, pooled blocks, status words, committed recurrent
    states (after folding the last round's kept rows) and DeltaNet rows, sparse (2,400-row) and dense streams."""

    w, dec, states = captured
    order = [states[i] for i in ((0, 1, 2) if parity == 0 else (2, 0, 1))][:n]
    g = torch.Generator().manual_seed(100 * n + parity)
    windows = [torch.randint(1, V, (k,), generator=g).tolist() for k in split]
    # last round's rows in the other parity's scratch: a forward there, then kept rows to fold first
    _verify(dec, w, order, windows, [[] for _ in order], 1 - parity, "eager")
    held = [list(range(min(2, k))) if i % 2 == 0 else [] for i, k in enumerate(split)]
    want = _verify(dec, w, order, windows, held, parity, "old")
    _equal(_verify(dec, w, order, windows, held, parity, "eager"), want, "eager")
    _equal(_verify(dec, w, order, windows, held, parity, "graph"), want, "graph")


def test_static_chain_entries_equal_the_plan_builder():
    """T2: the static tables' chain entries and starts are ``gdn.plan_host``'s for every split."""

    import numpy as np

    from tensorfold.cuda.kernels import gdn as shared

    w = _model()
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=4)
    t = dec.round_tables
    for split in ([1], [5], [2, 3], [5, 5, 1, 4], [1, 1, 1, 1]):
        segs, a0 = [], 0
        for st, k in zip(dec.free, split):
            segs.append((st, a0, a0 + k))
            a0 += k
        t.write(segs, [[] for _ in segs])
        torch.cuda.synchronize()
        entries, starts, slots, _ = shared.plan_host([list(range(-1, k - 1)) for k in split])
        R, at = a0, t.at
        got = t.i32[at["entries"]:at["entries"] + 3 * R].cpu().numpy()
        assert slots == 0 and np.array_equal(got, np.asarray(entries, dtype=np.int32)), split
        assert t.i32[at["starts"]:at["starts"] + len(split) + 1].tolist() == starts, split


@pytest.mark.parametrize("n,split", [(1, [1]), (1, [4]), (2, [2, 3]), (3, [4, 1, 4]), (3, [1, 1, 1])])
def test_mtp_steps_replay_the_eager_bits(captured, n, split):
    """T4 (MTP): absorb and chain steps of every row count replay the eager static step's logits and frames."""

    w, dec, states = captured
    g = torch.Generator().manual_seed(7 * n + len(split))
    tokens = [torch.randint(1, V, (k,), generator=g).tolist() for k in split]
    streams = torch.randn((sum(split), dec.buf.streams.shape[1]), generator=torch.Generator().manual_seed(3)) \
        .to(torch.bfloat16).cuda() * 0.1
    got = {}
    for mode in ("old", "eager", "graph"):
        sts = [st.clone() for st in states[:n]]
        b, at, windows = dec.mbuf, 0, []
        for st, k, tok in zip(sts, split, tokens):
            windows.append((st, tok, streams[at:at + k]))
            at += k
        segs = mtp_stage(w, b, windows)
        b.attn_step = attn_multi.Step(w, segs, mtp=True, tables=dec.mtp_tables if mode != "old" else None)
        try:
            out = dec.graphs.mtp_step(segs) if mode == "graph" else mtp_compute(w, segs, b, check=False)
            assert out is not None
            _kv_check(segs)
        finally:
            b.attn_step = None
        got[mode] = [out.clone(), b.streams[:sum(split)].clone(), *_frame(sts, segs, True)]
    _equal(got["eager"], got["old"], "eager")
    _equal(got["graph"], got["old"], "graph")


def _decode(w, prompts, samplings, count, *, graphs, kv_dtype=RQ, capacity=4096, depth=3, confidence=0.3,
            prefill_rows=256, draft=None, later=(), each_round=None):
    """Decode ``prompts`` together (``later``: (round, prompt index) admitted mid-run); each stream's tokens and the
    decoder (its replay count over the run in ``dec.total_replays``)."""

    dec = MultiDecoder(w, slots=len(prompts), capacity=capacity, depth=depth, confidence=confidence,
                       kv_dtype=kv_dtype, prefill_rows=prefill_rows, graphs=graphs)
    if graphs:
        dec.capture_graphs()
    streams = [Stream(list(p), count, smp, draft=True if draft is None else draft[i])
               for i, (p, smp) in enumerate(zip(prompts, samplings))]
    pending = {i: r for r, i in later}
    for i, s in enumerate(streams):
        if i not in pending:
            dec.admit(s)
    rounds = dec.total_replays = 0
    while dec.live() or any(i in pending for i in range(len(streams))):
        for i, r in list(pending.items()):
            if rounds >= r:
                dec.admit(streams[i])
                del pending[i]
        if each_round is not None:
            each_round(rounds)
        dec.finish(dec.round())
        dec.total_replays += dec.replays
        rounds += 1
    return [s.out for s in streams], dec


@pytest.mark.parametrize("depth,confidence", [(1, 0.0), (3, 0.3), (5, 0.5)])
def test_graph_rounds_emit_each_streams_solo_tokens(depth, confidence):
    """T5/T6/T8: streams of greedy and keyed policies, drafting or serial, joining mid-run and finishing at different
    lengths, one past the attention budget (sparse rows), decode through replayed rounds and MTP steps as each does
    alone, at MTP depths 1, 3 and 5."""

    w = _model(7)
    g = torch.Generator().manual_seed(9)
    prompts = [torch.randint(1, V, (2300,), generator=g).tolist(), PROMPTS[0], PROMPTS[1], PROMPTS[3]]
    samplings = [Sampling(seed=5, top_k=20, top_p=0.95), None, Sampling(seed=6, top_k=20, top_p=0.95), None]
    counts = [14, 9, 18, 11]
    refs = [_solo(w, [p], [smp], c, RQ)[0] for p, smp, c in zip(prompts, samplings, counts)]
    dec = MultiDecoder(w, slots=4, capacity=4096, depth=depth, confidence=confidence, kv_dtype=RQ, prefill_rows=256,
                       graphs=True)
    dec.capture_graphs()
    assert dec.graph_record["captured"] and dec.graphs.verify and dec.graphs.mtp
    streams = [Stream(list(p), c, smp, draft=i != 3) for i, (p, smp, c) in enumerate(zip(prompts, samplings, counts))]
    for s in streams[:2]:
        dec.admit(s)
    rounds, replays = 0, 0
    while dec.live() or rounds < 3:
        if rounds == 2:                                  # two more join while the first two decode
            for s in streams[2:]:
                dec.admit(s)
        dec.finish(dec.round())
        replays += dec.replays
        rounds += 1
    assert [s.out for s in streams] == refs
    assert replays > 0


def test_copy_drafts_and_prompt_passes_through_graph_rounds(monkeypatch):
    """T5/T7: copy drafts (TENSORFOLD_MTP_COPY=1, repetitive prompts) and a prompt filling in the rounds' forwards
    (eager mixed rounds between replayed ones) emit each stream's solo tokens; slots grow past their first rows."""

    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    w = _model(5)
    g = torch.Generator().manual_seed(11)
    loop = [7, 300, 41, 9, 1500, 22, 64, 5] * 8
    prompts = [loop, torch.randint(1, V, (300,), generator=g).tolist(), loop[3:] + [11, 12], PROMPTS[1]]
    samplings = [None, Sampling(seed=4, top_k=20, top_p=0.95), Sampling(seed=8, top_k=20, top_p=0.95), None]
    refs = _solo(w, prompts, samplings, 40, RQ, capacity=1024, prefill_rows=64)
    outs, dec = _decode(w, prompts, samplings, 40, graphs=True, capacity=1024, prefill_rows=64, later=[(2, 1), (4, 3)])
    assert outs == refs and dec.total_replays > 0
    assert max(st.version for st in dec.free) >= 1                  # a slot grew past its first 256 rows


def test_switching_graphs_every_round_keeps_the_bits(monkeypatch, tmp_path):
    """The A/B switch flips between replays on static tables and the old eager path each round (kept rows folded
    across the two): every stream still emits its solo run."""

    w = _model(7)
    refs = _solo(w, PROMPTS, SAMPLINGS, 24, RQ)
    control = tmp_path / "ab.json"
    _switch(monkeypatch, tmp_path, graphs="new")

    def flip(r):
        control.write_text(json.dumps({"graphs": "old" if r % 2 else "new", "b1": "old" if r % 3 == 1 else "new"}))
        import os
        os.utime(control, ns=(r * 10**9 + 1, r * 10**9 + 1))

    outs, dec = _decode(w, PROMPTS, SAMPLINGS, 24, graphs=True, each_round=flip)
    assert outs == refs and dec.total_replays > 0


@pytest.mark.parametrize("where", ["verify", "mtp"])
def test_failed_status_in_a_replayed_step_raises_and_recovers(monkeypatch, where):
    """T10: a NaN query scale fails a replayed step's status: the round raises the codec failure (not the sampler's or
    the draft confidence's), nothing is published, no capture happens again, and reset slots serve again."""

    w = _model(7)
    emitted = []
    dec = MultiDecoder(w, slots=2, capacity=4096, depth=3, confidence=0.0, kv_dtype=RQ, prefill_rows=256, graphs=True)
    dec.capture_graphs()
    pieces = sum(len(p.graphs) for p in (*dec.graphs.verify.values(), *dec.graphs.mtp.values()))
    streams = [Stream(list(PROMPTS[0]), 12, SAMPLINGS[1], emit=lambda new: emitted.extend(new)),
               Stream(list(PROMPTS[1]), 12, None, emit=lambda new: emitted.extend(new))]
    for s in streams:
        dec.admit(s)
    dec.finish(dec.round())
    before = list(emitted)
    scale = w.mtp.layer.attn.q_scale if where == "mtp" else w.layers[1].attn.q_scale
    saved = scale.clone()
    scale.fill_(float("nan"))
    try:
        with pytest.raises(KVNumericError, match="layer=MTP" if where == "mtp" else "layer=1"):
            dec.round()
        assert dec.replays > 0
    finally:
        scale.copy_(saved)
    assert emitted == before and all(s.st._kv_error is not None for s in streams)
    dec.drop()
    again = Stream(list(PROMPTS[0]), 8, SAMPLINGS[1])
    dec.admit(again)
    while dec.live():
        dec.finish(dec.round())
    assert again.error is None and again.out == _solo(w, PROMPTS[:1], SAMPLINGS[1:2], 8, RQ)[0]
    assert sum(len(p.graphs) for p in (*dec.graphs.verify.values(), *dec.graphs.mtp.values())) == pieces


def test_uncapturable_experts_decline_graphs_and_decode_eagerly(monkeypatch):
    """T11: experts that read their plan on the host decline capture at construction with the reason; the rounds run
    the eager path."""

    w = _model()
    for layer in w.layers:
        monkeypatch.setattr(layer.moe.experts, "capturable", False, raising=False)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, graphs=True)
    assert dec.graph_budget == 0 and "host" in dec.graph_record["reason"]
    dec.warm()
    assert dec.graphs is None


def test_a_budget_too_small_stops_capture_and_rounds_run_eagerly(monkeypatch):
    """T13 (budget): a key whose memory passes TENSORFOLD_MULTI_GRAPH_MIB is dropped and capture stops there; the
    remaining rounds run eagerly with the same tokens; 0 declines graphs."""

    w = _model()
    refs = _solo(w, PROMPTS[:2], SAMPLINGS[:2], 12, RQ, capacity=1024, prefill_rows=16)
    monkeypatch.setenv(multi_graphs.ENV, "0")
    assert MultiDecoder(w, slots=2, capacity=1024, depth=3, kv_dtype=RQ, graphs=True).graph_budget == 0
    monkeypatch.setenv(multi_graphs.ENV, "1")
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, kv_dtype=RQ, prefill_rows=16, graphs=True)
    dec.capture_graphs()
    record = dec.graph_record
    assert record["captured"] and record["stopped"] and record["bytes"] <= 1 << 20
    assert len(dec.graphs.verify) + len(dec.graphs.mtp) == len(record["keys"])
    streams = [Stream(list(p), 12, smp) for p, smp in zip(PROMPTS[:2], SAMPLINGS[:2])]
    for s in streams:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == refs


def test_graph_memory_is_measured_bounded_and_taken_from_the_cache_room(tmp_path):
    """T13: every key of 4 slots at depth 4 is captured within the default budget; the measured bytes come off the
    memory gate's room and are recorded (logged here for the GB10 budget calibration)."""

    w = _model()
    dec = MultiDecoder(w, slots=4, capacity=4096, depth=4, kv_dtype=RQ, prefill_rows=64, graphs=True)
    room = dec.memory_gate.room
    dec.capture_graphs()
    record = dec.graph_record
    assert record["verify_keys"] == 2 * (5 + 9 + 13 + 17) and record["mtp_keys"] == 5 + 9 + 13 + 17
    assert not record["stopped"] and 0 < record["bytes"] <= record["budget_bytes"]
    assert dec.memory_gate.room == room - record["bytes"]
    print(json.dumps({k: v for k, v in record.items() if k != "keys"}))
    per = [k["bytes"] for k in record["keys"]]
    print("per-key bytes: min", min(per), "max", max(per), "pieces", record["pieces"])

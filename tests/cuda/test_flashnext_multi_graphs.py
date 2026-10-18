"""Flash Next concurrent rounds through captured CUDA graphs: the static tables and fixed launch geometry give each
stream's own per-stream forward bits, every captured key replays them, and decoding emits each stream's solo run
through every round shape (joins, finishes, copy drafts, sparse rows, prompt passes, growth, failures)."""

import gc
import json
import traceback
import weakref
from types import SimpleNamespace

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.cuda import pdl  # noqa: E402
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


def _solo(w, prompts, samplings, count, kv_dtype, capacity=4096, prefill_rows=256):
    refs = []
    for prompt, sampling in zip(prompts, samplings):
        e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=prefill_rows, kv_dtype=kv_dtype)
        refs.append(serial_decode(e, prefill(e, prompt, sampling), count, sampling, stop_eos=False).tokens)
    return refs


def _prefilled(w, prompts, kv_dtype, capacity):
    """States holding each prompt's committed rows (their own caches), for forward-level comparisons."""

    states = []
    for prompt in prompts:
        e = Engine(w, capacity=capacity, max_rows=8, prefill_rows=256, kv_dtype=kv_dtype)
        prefill(e, prompt, None)
        states.append(e.st)
    return states


def _frame(segs, mtp: bool, recurrent: bool = True):
    """What a forward wrote of each stream's frame: its cache rows (and scales), index keys, pooled blocks, status and
    (``recurrent``) its committed DeltaNet states."""

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
        if recurrent:
            got.append(st.rec)
    return [t.clone() for t in got]


def _alone(w, b, states, windows):
    """The oracle of a verify forward: each stream's rows through the per-stream path (no round tables: its own
    attention and DeltaNet launches, which commit its recurrent state at once) over clones of ``states``; its
    logits, streams and frames without the recurrent states."""

    sts = [st.clone() for st in states]
    segs = stage(w, b, list(zip(sts, windows)))
    logits = compute(w, segs, b, check=False)
    _kv_check(segs)
    R = segs[-1][2]
    return [logits[:R].clone(), b.streams[:R].clone(), *_frame(segs, False, recurrent=False)]


def _flushed(dec, states, held, parity):
    """Clones of ``states`` with each stream's ``held`` rows (last round's, in the DeltaNet scratch's other parity)
    replayed into its recurrent state as the decoder flushes a waiting stream's rows (``_flush``: the replay kernel,
    not the trees' fold), for ``_alone`` to run the window from."""

    sts = [st.clone() for st in states]
    dec.gdn.parity = parity                              # this round's parity: the held rows sit in the other one
    for i, (st, rows) in enumerate(zip(sts, held)):
        dec.held[-1 - i] = list(rows)                    # an id no stream has; _flush pops it
        dec._flush(SimpleNamespace(sid=-1 - i, st=st))
    return sts


def _verify(dec, w, states, windows, held, parity, mode):
    """One verify forward over clones of ``states``: "eager" (static tables, fixed geometry) or "graph" (the captured
    key replayed); its logits, streams, frames (``_alone``'s first) and DeltaNet rows."""

    sts = [st.clone() for st in states]
    b = dec.buf
    segs = stage(w, b, list(zip(sts, windows)))
    dec.gdn.parity = parity
    b.gdn_tables = gdn_multi.Tables(dec.gdn, segs, held, static=dec.round_tables)
    b.attn_step = attn_multi.Step(w, segs, mtp=False, tables=dec.step_tables)
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
    return [logits[:R].clone(), b.streams[:R].clone(), *_frame(segs, False, recurrent=False),
            *[st.rec.clone() for st in sts], *[t.clone() for t in rows]]


def _equal(a, b, what):
    assert len(a) == len(b)
    for i, (x, y) in enumerate(zip(a, b)):
        assert x.shape == y.shape and torch.equal(x, y), (what, i)


@pytest.fixture(scope="module", params=[RQ, "rotorquant8-norm"])
def captured(request):
    """A captured 3-slot decoder over a model whose QSA engages past 2,048 keys, and prefilled states: one long
    (2,400 rows: sparse), two short (dense); six- and eight-bit caches, whose concurrent attention decodes in
    quarters (``attention._chunk_quarters`` through ``attn_multi._chunks_multi``)."""

    kv_dtype = request.param
    w = _model(7)
    dec = MultiDecoder(w, slots=3, capacity=4096, depth=3, confidence=0.3, kv_dtype=kv_dtype, graphs=True)
    dec.capture_graphs()
    g = torch.Generator().manual_seed(9)
    long = torch.randint(1, V, (2400,), generator=g).tolist()
    states = _prefilled(w, [long, PROMPTS[0], PROMPTS[1]], kv_dtype, 4096)
    return w, dec, states


@pytest.mark.parametrize("parity", [0, 1])
@pytest.mark.parametrize("n,split", [(1, [1]), (1, [4]), (2, [1, 4]), (2, [4, 3]), (3, [1, 1, 1]), (3, [4, 2, 4]),
                                     (3, [4, 4, 4])])
def test_static_tables_and_fixed_geometry_and_replays_give_each_streams_own_bits(captured, n, split, parity):
    """T1/T4: per key, the eager forward on static tables at the fixed geometry writes each stream's own per-stream
    forward's logits, streams, cache rows, scales, index keys, pooled blocks and status words (``_alone``), sparse
    (2,400-row) and dense streams. Folding the last round's kept rows first, it writes what the per-stream forward
    writes once those rows are replayed into each recurrent state (``_flushed``), and the same committed recurrent
    states. Its captured replay writes the eager forward's logits, streams, frames, committed recurrent states and
    DeltaNet rows, either way."""

    w, dec, states = captured
    order = [states[i] for i in ((0, 1, 2) if parity == 0 else (2, 0, 1))][:n]
    g = torch.Generator().manual_seed(100 * n + parity)
    windows = [torch.randint(1, V, (k,), generator=g).tolist() for k in split]
    none = [[] for _ in order]
    alone = _alone(w, dec.buf, order, windows)
    eager = _verify(dec, w, order, windows, none, parity, "eager")
    _equal(eager[:len(alone)], alone, "alone")
    _equal(_verify(dec, w, order, windows, none, parity, "graph"), eager, "graph")
    # last round's rows in the other parity's scratch: a forward there, then kept rows to fold first
    _verify(dec, w, order, windows, none, 1 - parity, "eager")
    held = [list(range(min(2, k))) if i % 2 == 0 else [] for i, k in enumerate(split)]
    want = _verify(dec, w, order, windows, held, parity, "eager")
    _equal(_verify(dec, w, order, windows, held, parity, "graph"), want, "graph, folded")
    flushed = _flushed(dec, order, held, parity)          # the fold's oracle: replayed rows, then the per-stream path
    alone = _alone(w, dec.buf, flushed, windows) + [st.rec.clone() for st in flushed]
    _equal(want[:len(alone)], alone, "alone, folded")


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
    """T4 (MTP): absorb and chain steps of every row count, eager on the static step tables and replayed, write each
    stream's own per-stream step's logits and frames."""

    w, dec, states = captured
    g = torch.Generator().manual_seed(7 * n + len(split))
    tokens = [torch.randint(1, V, (k,), generator=g).tolist() for k in split]
    streams = torch.randn((sum(split), dec.buf.streams.shape[1]), generator=torch.Generator().manual_seed(3)) \
        .to(torch.bfloat16).cuda() * 0.1
    got = {}
    for mode in ("alone", "eager", "graph"):
        sts = [st.clone() for st in states[:n]]
        b, at, windows = dec.mbuf, 0, []
        for st, k, tok in zip(sts, split, tokens):
            windows.append((st, tok, streams[at:at + k]))
            at += k
        segs = mtp_stage(b, windows)
        b.attn_step = None if mode == "alone" else attn_multi.Step(w, segs, mtp=True, tables=dec.mtp_tables)
        try:
            out = dec.graphs.mtp_step(segs) if mode == "graph" else mtp_compute(w, segs, b, check=False)
            assert out is not None
            _kv_check(segs)
        finally:
            b.attn_step = None
        got[mode] = [out.clone(), b.streams[:sum(split)].clone(), *_frame(segs, True)]
    _equal(got["eager"], got["alone"], "eager")
    _equal(got["graph"], got["alone"], "graph")


def _replays(dec) -> list[int]:
    """A counter of the captured forwards and MTP steps ``dec`` replays (its graphs' calls that serve a step)."""

    count = [0]

    def counting(real):
        def counted(segs):
            out = real(segs)
            count[0] += out is not None
            return out
        return counted

    for name in ("verify_forward", "mtp_step"):
        setattr(dec.graphs, name, counting(getattr(dec.graphs, name)))
    return count


def _decode(w, prompts, samplings, count, *, graphs, kv_dtype=RQ, capacity=4096, depth=3, confidence=0.3,
            prefill_rows=256, draft=None, later=()):
    """Fixed-count decoding ignores EOS like ``_solo``; later streams join at their scheduled rounds."""

    dec = MultiDecoder(w, slots=len(prompts), capacity=capacity, depth=depth, confidence=confidence,
                       kv_dtype=kv_dtype, prefill_rows=prefill_rows, graphs=graphs)
    replays = [0]
    if graphs:
        dec.capture_graphs()
        replays = _replays(dec)
    streams = [Stream(list(p), count, smp, draft=True if draft is None else draft[i], stop_eos=False)
               for i, (p, smp) in enumerate(zip(prompts, samplings))]
    pending = {i: r for r, i in later}
    for i, s in enumerate(streams):
        if i not in pending:
            dec.admit(s)
    rounds = 0
    while dec.live() or any(i in pending for i in range(len(streams))):
        for i, r in list(pending.items()):
            if rounds >= r:
                dec.admit(streams[i])
                del pending[i]
        dec.finish(dec.round())
        rounds += 1
    assert all(s.error is None and len(s.out) == count for s in streams)
    return [s.out for s in streams], dec, replays[0]


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
    replays = _replays(dec)
    streams = [Stream(list(p), c, smp, draft=i != 3, stop_eos=False)
               for i, (p, smp, c) in enumerate(zip(prompts, samplings, counts))]
    for s in streams[:2]:
        dec.admit(s)
    rounds = 0
    while dec.live() or rounds < 3:
        if rounds == 2:                                  # two more join while the first two decode
            for s in streams[2:]:
                dec.admit(s)
        dec.finish(dec.round())
        rounds += 1
    assert [s.out for s in streams] == refs
    assert [len(s.out) for s in streams] == counts
    assert replays[0] > 0


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
    outs, dec, replays = _decode(w, prompts, samplings, 40, graphs=True, capacity=1024, prefill_rows=64,
                                 later=[(2, 1), (4, 3)])
    assert outs == refs and replays > 0
    assert all(len(out) == 40 for out in outs)
    slots = dec.free + [k[1] for k in dec.kept]                     # kept prompt ends hold their slots
    assert max(st.version for st in slots) >= 1                     # a slot grew past its first 256 rows


@pytest.mark.parametrize("where", ["verify", "mtp"])
def test_failed_status_in_a_replayed_step_raises_and_recovers(where):
    """T10: a NaN query scale fails a replayed step's status: the round raises the codec failure (not the sampler's or
    the draft confidence's), nothing is published, no capture happens again, and reset slots serve again."""

    w = _model(7)
    emitted = []
    dec = MultiDecoder(w, slots=2, capacity=4096, depth=3, confidence=0.0, kv_dtype=RQ, prefill_rows=256, graphs=True)
    dec.capture_graphs()
    pieces = sum(len(p.graphs) for p in (*dec.graphs.verify.values(), *dec.graphs.mtp.values()))
    streams = [Stream(list(PROMPTS[0]), 12, SAMPLINGS[1], stop_eos=False, emit=lambda new: emitted.extend(new)),
               Stream(list(PROMPTS[1]), 12, None, stop_eos=False, emit=lambda new: emitted.extend(new))]
    for s in streams:
        dec.admit(s)
    dec.finish(dec.round())
    before = list(emitted)
    scale = w.mtp.layer.attn.q_scale if where == "mtp" else w.layers[1].attn.q_scale
    saved = scale.clone()
    scale.fill_(float("nan"))
    replays = _replays(dec)
    try:
        with pytest.raises(KVNumericError, match="layer=MTP" if where == "mtp" else "layer=1"):
            dec.round()
        assert replays[0] > 0
    finally:
        scale.copy_(saved)
    assert emitted == before and all(s.st._kv_error is not None for s in streams)
    dec.drop()
    again = Stream(list(PROMPTS[0]), 8, SAMPLINGS[1], stop_eos=False)
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
    assert record["bytes"] == sum(max(0, key["bytes"]) for key in record["keys"])
    streams = [Stream(list(p), 12, smp, stop_eos=False) for p, smp in zip(PROMPTS[:2], SAMPLINGS[:2])]
    for s in streams:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == refs


def test_graph_memory_is_measured_bounded_and_taken_from_the_cache_room():
    """T13: every key of 4 slots at depth 4 is captured within the default budget; each key's capture is measured
    alone (raw), the kept keys' sum is bounded by the budget, and the larger of it and the settled drop of the whole
    capture comes off the memory gate's room; all recorded (logged here for the GB10 budget calibration)."""

    w = _model()
    dec = MultiDecoder(w, slots=4, capacity=4096, depth=4, kv_dtype=RQ, prefill_rows=64, graphs=True)
    room = dec.memory_gate.room
    dec.capture_graphs()
    record = dec.graph_record
    assert record["verify_keys"] == 2 * (5 + 9 + 13 + 17) and record["mtp_keys"] == 5 + 9 + 13 + 17
    assert not record["stopped"] and 0 < record["bytes"] <= record["budget_bytes"]
    assert record["bytes"] == sum(max(0, key["bytes"]) for key in record["keys"])
    assert record["room_bytes"] == max(record["bytes"], record["settled_bytes"])
    assert dec.memory_gate.room == room - record["room_bytes"]
    from tensorfold.families.qwen4_exp.cuda.service_memory import private_pool_spare

    if record["allocator"]["accounting"] == "native-pool-inactive":
        snapshot = torch.cuda.memory_snapshot(mempool_id=dec.graphs.pool, include_traces=False)
        unavailable = private_pool_spare(snapshot, dec.graphs.pool, w.device.index or 0)
        assert record["allocator"]["private_pool_spare_bytes"] == unavailable
        assert dec.graphs.private_spare_bytes == unavailable
        assert dec.memory_gate.live.__defaults__ == (dec.graphs,)
    else:
        assert dec.graphs.private_spare_bytes is None and dec.memory_gate.live is multi_graphs.live_bytes
    print(json.dumps({k: v for k, v in record.items() if k != "keys"}))
    per = [k["bytes"] for k in record["keys"]]
    print("per-key bytes: min", min(per), "max", max(per), "pieces", record["pieces"])


def test_a_failed_capture_raises_its_own_error_and_leaves_the_decoder_eager(monkeypatch):
    """C9: a key's capture failing midway raises that error itself; the launch policy, garbage collection, DeltaNet
    parity and step tables are restored and the slots reset. The admitted failed pool remains owned/accounted;
    rounds still decode the solo tokens eagerly."""

    w = _model()
    refs = _solo(w, PROMPTS[:2], SAMPLINGS[:2], 12, RQ, capacity=1024, prefill_rows=16)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, kv_dtype=RQ, prefill_rows=16, graphs=True)
    room, launch, parity = dec.memory_gate.room, pdl.ON, dec.gdn.parity
    real, calls, boom = multi_graphs.RoundGraphs._pieces, [], RuntimeError("the third key's capture failed")

    def pieces(graphs, run, b):
        calls.append(1)
        if len(calls) == 3:
            raise boom
        return real(graphs, run, b)

    monkeypatch.setattr(multi_graphs.RoundGraphs, "_pieces", pieces)
    with pytest.raises(RuntimeError) as caught:
        dec.capture_graphs()
    assert caught.value is boom and caught.value.__cause__ is None
    assert dec.graphs is None and not dec.graph_record["captured"]
    assert dec._failed_graphs is not None and dec._memory_error is None
    assert dec.memory_gate.room == room - dec.graph_record["room_bytes"]
    assert gc.isenabled() and pdl.ON == launch and dec.gdn.parity == parity
    assert dec.buf.attn_step is None and dec.buf.gdn_tables is None and dec.mbuf.attn_step is None
    assert all(not st._kv_pending and st._kv_error is None and st.pos == 0 for st in dec.free)
    streams = [Stream(list(p), 12, smp, stop_eos=False) for p, smp in zip(PROMPTS[:2], SAMPLINGS[:2])]
    for s in streams:
        dec.admit(s)
    while dec.live():
        dec.finish(dec.round())
    assert [s.out for s in streams] == refs


def test_a_failed_slot_restore_after_a_failed_capture_is_attached_to_the_capture_error(monkeypatch):
    """C9: when restoring the slots also fails after a key's capture failed, the capture's error is still the one
    raised, with the restore failure attached (its cause and a note); the launch policy and collection are restored."""

    w = _model()
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, kv_dtype=RQ, prefill_rows=16, graphs=True)
    launch = pdl.ON
    boom, restore = RuntimeError("the first key's capture failed"), RuntimeError("a slot could not be reset")
    monkeypatch.setattr(multi_graphs.RoundGraphs, "_pieces", lambda graphs, run, b: (_ for _ in ()).throw(boom))
    monkeypatch.setattr(dec.free[1], "reset", lambda w: (_ for _ in ()).throw(restore))
    with pytest.raises(RuntimeError) as caught:
        dec.capture_graphs()
    assert caught.value is boom and caught.value.__cause__ is restore
    assert any("cleanup also failed" in note for note in caught.value.__notes__)
    assert dec.graphs is None and gc.isenabled() and pdl.ON == launch


def test_failed_capture_tensor_locals_survive_real_frame_clear_and_keep_pool_receipt(monkeypatch):
    """assertRaises-style traceback clearing cannot change the failed pool's admitted inactive-byte constant."""
    from tensorfold.families.qwen4_exp.cuda.service_memory import private_pool_spare

    w = _model()
    refs = _solo(w, PROMPTS[:1], SAMPLINGS[:1], 8, RQ, capacity=1024, prefill_rows=16)
    dec = MultiDecoder(w, slots=2, capacity=1024, depth=3, kv_dtype=RQ, prefill_rows=16, graphs=True)
    original, probes = multi_graphs.RoundGraphs._pieces, []
    boom = RuntimeError("inside the captured forward")
    def pieces(graphs, run, b):
        def fail():
            temporary = torch.empty(4096, dtype=torch.bfloat16, device=w.device)
            probes.append(weakref.ref(temporary))
            run()
            raise boom
        return original(graphs, fail, b)
    monkeypatch.setattr(multi_graphs.RoundGraphs, "_pieces", pieces)
    with pytest.raises(RuntimeError) as caught:
        dec.capture_graphs()
    assert caught.value is boom and dec._memory_error is None
    owner = dec._failed_graphs
    assert owner is not None and owner.failure_frames
    traceback.clear_frames(owner.failure_traceback)
    boom.with_traceback(None)
    gc.collect()
    torch.cuda.synchronize()
    assert probes[0]() is not None
    if owner.allocator_backend == "native":
        snapshot = torch.cuda.memory_snapshot(mempool_id=owner.pool, include_traces=False)
        assert private_pool_spare(snapshot, owner.pool, w.device.index or 0) == owner.private_spare_bytes
    else:
        assert owner.private_spare_bytes is None and dec.memory_gate.live is multi_graphs.live_bytes
    stream = Stream(list(PROMPTS[0]), 8, SAMPLINGS[0], stop_eos=False)
    dec.admit(stream)
    while dec.live():
        dec.finish(dec.round())
    assert stream.error is None and stream.out == refs[0]


def _side_capturing(graphs) -> bool:
    with torch.cuda.stream(graphs.side):
        return torch.cuda.is_current_stream_capturing()


def test_capture_pieces_end_only_begun_captures_and_keep_the_first_failure(monkeypatch):
    """C9 (``RoundGraphs._pieces``): a failure inside the captured forward stays the raised error once its open piece
    is ended; a capture that cannot begin at a split is never ended (the piece before it ends once); an invalidating
    call mid-capture is the raised error, with the failed end of its capture attached; capture works afterwards."""

    graphs = multi_graphs.RoundGraphs(SimpleNamespace(), 0)
    b = SimpleNamespace(split=None)
    x = torch.zeros(16, device="cuda")
    torch.cuda.synchronize()

    def failing():
        x.add_(1)
        b.split(("select", None))
        x.mul_(2)
        raise ValueError("the forward failed mid-capture")

    with pytest.raises(ValueError, match="mid-capture") as caught:
        graphs._pieces(failing, b)
    assert caught.value.__cause__ is None and b.split is None and not _side_capturing(graphs)

    begun, ended = [], []

    class Graph(torch.cuda.CUDAGraph):
        def capture_begin(self, *args, **kwargs):
            if begun:
                raise RuntimeError("the second piece cannot begin")
            begun.append(self)
            super().capture_begin(*args, **kwargs)

        def capture_end(self):
            ended.append(self)
            super().capture_end()

    with monkeypatch.context() as patched:
        patched.setattr(torch.cuda, "CUDAGraph", Graph)
        with pytest.raises(RuntimeError, match="cannot begin") as caught:
            graphs._pieces(lambda: (x.add_(1), b.split(("ple_conv", None)), x.mul_(2)), b)
    assert caught.value.__cause__ is None and ended == begun and len(begun) == 1 and not _side_capturing(graphs)

    def invalidating():
        x.add_(1)
        torch.cuda.synchronize()                         # not permitted while this thread captures

    with pytest.raises(Exception) as caught:
        graphs._pieces(invalidating, b)
    first = caught.value
    assert "capturing" in str(first) and first.__cause__ is not None and first.__cause__ is not first
    assert any("cleanup also failed" in note for note in getattr(first, "__notes__", []))
    assert not _side_capturing(graphs)
    torch.cuda.synchronize()

    pieces = graphs._pieces(lambda: (x.add_(1), b.split(("ple_conv", None)), x.mul_(3)), b)
    assert len(pieces.graphs) == 2 and pieces.inserts == [("ple_conv", None)]
    x.zero_()
    for graph in pieces.graphs:
        graph.replay()
    torch.cuda.synchronize()
    assert torch.equal(x, torch.full_like(x, 3.0))


def test_the_engine_refuses_a_decoder_whose_graph_budget_differs_from_its_plan(tmp_path, monkeypatch):
    """Admission plans the captured rounds' budget from what the concurrent decoder will capture; a decoder that
    declines (here: every model) while the plan reserved it is an internal inconsistency, refused at startup."""

    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    from test_flashnext_tp import _checkpoint

    _checkpoint(tmp_path)
    monkeypatch.setenv(multi_graphs.ENV, "512")  # an explicit reservation must still agree if capture declines
    monkeypatch.setattr(multi_graphs, "capturable", lambda w: "declined by this test")
    with pytest.raises(RuntimeError, match=r"startup plan reserved .* \(declined by this test\)"):
        FlashNextEngine(tmp_path, depth=3, draft_vocab=None, max_len=512, prefetch=False, streams=2)

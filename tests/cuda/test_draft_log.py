"""Draft telemetry on concurrent rounds: off creates nothing; the log (with every round probed) and the trace leave
every verified draft and every emitted token as they were, and the trace replays round for round offline."""

import importlib.util
import json
import os
from pathlib import Path
import stat
import sys
import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from test_flashnext_forward import V, _model  # noqa: E402

from tensorfold.cuda import ab_switch, draft_log  # noqa: E402
from tensorfold.drafters import draft_rule  # noqa: E402
from tensorfold.cuda.streams import Stream  # noqa: E402
from tensorfold.engine.exact_sampling import Sampling  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import qmm  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder  # noqa: E402

TOOL = Path(__file__).resolve().parents[2] / "tools/draft_replay.py"
_spec = importlib.util.spec_from_file_location("draft_replay", TOOL)
replay = importlib.util.module_from_spec(_spec)
sys.modules["draft_replay"] = replay
_spec.loader.exec_module(replay)

PROMPTS = [[5, 17, 99, 250], [1023, 7, 64, 300, 11, 12], [13], [8, 8, 9, 2000, 31]]
SAMPLINGS = [None, Sampling(seed=1234, top_k=20, top_p=0.95), Sampling(seed=7, top_k=20, top_p=1.0, min_p=0.1),
             Sampling(seed=99, top_k=0, top_p=0.9)]
ENVS = (draft_log.LOG_ENV, draft_log.PROBE_ENV, draft_log.TRACE_ENV)
DRAFT_KEYS = {"j", "p", "m", "e", "g", "z", "c", "L", "a0", "inc", "hit", "label"}
STREAM_KEYS = {"rid", "ctx", "n", "greedy", "src", "probe", "join", "constrained", "verified", "accepted", "done",
               "drafts"}
LINE_KEYS = {"t", "live", "rows", "mixed", "pass_rows", "steps", "chain_steps", "chain_ms", "probed", "joined", "ms",
             "cost", "ab", "streams", "dropped", "forced"}
DECODER = {"depth": 3, "copy_match": 8, "anchor": 4, "recent": 32, "longest": 64}


def _switch(monkeypatch, tmp_path, **values) -> None:
    """The benchmark switches the decoder reads (A/B baseline values: removed after selection). The cost-aware
    rule's costs are pinned at their prior, so its decisions repeat run to run (they follow wall times otherwise)."""

    control = tmp_path / "ab.json"
    control.write_text(json.dumps(values))
    monkeypatch.setattr(ab_switch, "_path", str(control))
    monkeypatch.setattr(ab_switch, "_values", {})
    monkeypatch.setattr(ab_switch, "_stamp", None)
    monkeypatch.setattr(draft_rule.CostModel, "observe", lambda self, **kw: None)


def _decode(w, monkeypatch, confidence, count=32, *, prompts=PROMPTS, samplings=SAMPLINGS, counts=None, warm=False):
    """Decode every prompt together; returns the decoder, replies, stats and each round's verified drafts."""

    drafted, real = [], MultiDecoder._draft_all

    def spy(self, streams, rule, *args, **kw):
        steps = real(self, streams, rule, *args, **kw)
        drafted.append(sorted((s.sid, tuple(s.drafts)) for s, *_ in streams))
        return steps

    monkeypatch.setattr(MultiDecoder, "_draft_all", spy)
    dec = MultiDecoder(w, slots=4, capacity=1024, depth=3, confidence=confidence, prefill_rows=16)
    if warm:
        dec.warm()                                       # a synthetic request: never in the telemetry
    streams = []
    for i, (prompt, sampling) in enumerate(zip(prompts, samplings)):
        s = Stream(list(prompt), count if counts is None else counts[i], sampling)
        dec.admit(s)
        streams.append(s)
    while dec.live():
        dec.finish(dec.round())
    monkeypatch.setattr(MultiDecoder, "_draft_all", real)
    return dec, [s.out for s in streams], [(s.rounds, s.drafted, s.accepted) for s in streams], drafted


def _model_for(vocab: bool):
    w = _model()
    if vocab:                                            # drafts over a token subset (the real model's draft head)
        words, scales, biases = qmm.to_mlx(w.head)
        ids = torch.arange(1, V, 3, device="cuda")
        w.draft_ids = ids
        w.draft_head = qmm.make_q4(words[ids], scales[ids], biases[ids])
    return w


def test_everything_off_creates_nothing(monkeypatch, tmp_path):
    for env in ENVS:
        monkeypatch.delenv(env, raising=False)
    created, margins = [], []
    monkeypatch.setattr(draft_log.DraftWatch, "__init__", lambda *a, **k: created.append(1))
    real = MultiDecoder._picks

    def picks(self, logits, positions, samplings, margins_on=False, *readback):
        margins.append(margins_on)
        return real(self, logits, positions, samplings, margins_on, *readback)

    monkeypatch.setattr(MultiDecoder, "_picks", picks)
    before = {t.name for t in threading.enumerate()}
    monkeypatch.chdir(tmp_path)
    dec, outs, _, _ = _decode(_model(), monkeypatch, 0.3, count=16)
    assert dec.watch is None and not created and all(outs)
    assert margins and not any(margins)                  # no margins: today's single readback, unchanged
    assert not any(t.name.startswith("tensorfold-draft") for t in threading.enumerate() if t.name not in before)
    assert not os.listdir(tmp_path)


@pytest.mark.parametrize("rule", ["new", "old"])            # "old": A/B baseline, remove after selection
@pytest.mark.parametrize("confidence,copy,vocab", [(0.3, False, False), (0.9, True, True), (0.0, True, False)])
def test_log_probes_and_trace_change_no_draft_or_token(confidence, copy, vocab, rule, monkeypatch, tmp_path):
    w = _model_for(vocab)
    _switch(monkeypatch, tmp_path, draft_rule=rule)
    for env in ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1" if copy else "0")
    warm = confidence == 0.9
    _, plain, plain_stats, plain_drafts = _decode(w, monkeypatch, confidence, warm=warm)
    log, trace = tmp_path / "drafts.jsonl", tmp_path / "trace"
    monkeypatch.setenv(draft_log.LOG_ENV, str(log))
    monkeypatch.setenv(draft_log.PROBE_ENV, "1.0")
    monkeypatch.setenv(draft_log.TRACE_ENV, str(trace))
    dec, outs, stats, drafts = _decode(w, monkeypatch, confidence, warm=warm)
    assert outs == plain and stats == plain_stats and drafts == plain_drafts
    watch = dec.watch
    dec.close_telemetry()
    assert dec.watch is None and watch.log.failed is None and watch.trace.failed is None
    assert not watch.lines and not watch.chains and not watch.pending and not watch.traces

    lines = [json.loads(line) for line in log.read_text().splitlines()]
    assert lines and all(set(line) == LINE_KEYS and line["dropped"] == line["forced"] == 0 for line in lines)
    entries = [st for line in lines for st in line["streams"]]
    assert all(set(st) == STREAM_KEYS and all(set(d) == DRAFT_KEYS for d in st["drafts"]) for st in entries)
    assert sum(len(line["streams"]) for line in lines) == sum(r for r, _, _ in stats)
    assert len({st["rid"] for st in entries}) == len(PROMPTS)
    mtp = [d for st in entries if st["src"] == "mtp" and (not st["join"] or rule == "new") for d in st["drafts"]]
    assert mtp and all(d["p"] is not None and d["m"] is not None and d["g"] is not None for d in mtp)
    assert all((d["c"] is not None) == (rule == "new") for d in mtp)        # the rule's chance, under the rule
    assert all((line["cost"] is not None) == (rule == "new") and line["ab"] == {"draft_rule": rule} for line in lines)
    assert all(line["chain_steps"] <= max(0, line["steps"] - 1) and 0 <= line["chain_ms"] <= line["ms"]
               for line in lines)
    assert any(line["chain_steps"] for line in lines)    # every probe chains on: timed steps
    assert all(d["m"] == "inf" or d["m"] >= 0 for d in mtp)
    shadows = [d for st in entries for d in st["drafts"] if not d["inc"]]
    if confidence > 0:
        assert shadows and any(d["label"] is not None for d in shadows)    # probes label drafts past the rule
    assert all(line["probed"] for line in lines if line["steps"] > 1)       # every chain ran in a probe
    for st in entries:                                   # acceptance is the labelled prefix, room permitting
        assert st["verified"] == sum(d["inc"] for d in st["drafts"])
        lead = 0
        for d in st["drafts"]:
            if not d["inc"] or d["label"] != 1:
                break
            lead += 1
        assert st["accepted"] == lead or st["done"], st

    files = sorted(trace.iterdir())
    assert len(files) == len(PROMPTS)
    docs = [json.loads(f.read_text()) for f in files]
    assert all(stat.S_IMODE(os.stat(f).st_mode) == 0o600 for f in files)
    assert sorted(d["out"] for d in docs) == sorted(outs)
    for doc in docs:
        assert sum(r["acc"] + 1 for r in doc["rounds"]) == len(doc["out"]) - 1
        assert doc["decoder"] == {**DECODER, "confidence": confidence, "copy": copy}
        assert doc["text"] == ("suffix" if copy else None) and {r["rule"] for r in doc["rounds"]} == {rule}
    result = replay.run(replay.load([str(trace)]), ARGS)
    if copy and rule == "old":                           # this model's replies never repeat: MTP rounds only
        assert result["parity_suffix"] == {"checked": len(PROMPTS), "equal": len(PROMPTS), "differ": []}
    assert result["results"]["recorded"]["all"]["tokens"] == sum(len(o) - 1 for o in outs)


ARGS = type("Args", (), dict(t0=34.6, row_ms=6.5, step_ms=1.6, anchors=[4], windows=[5], recent=32, longest=64,
                             corpus_tokens=0, corpus_segment=1024, mtp_drafts=2, mtp_accept=0.75, seed=0,
                             tables=None, lam=2.33 / 50, confidence=0.5))()
QUOTE_A, QUOTE_B = list(range(10, 70)), list(range(300, 340))
SCRIPTS = {1: [5, 6] + QUOTE_A[:30] + [7] + QUOTE_A[10:40] + [8, 9],      # replies that quote their prompts
           2: [11] + QUOTE_B[5:25] + [12, 13] + QUOTE_B[:12] + [14]}


@pytest.mark.parametrize("rule,suffix", [("old", "old"), ("old", "new"), ("new", "new"), ("new", "old")])
def test_offline_copy_drafts_replay_the_decoders_round_for_round(monkeypatch, tmp_path, rule, suffix):
    """The target's draws are scripted (by seed) to quote the prompt, so the decoder drafts from its own text (the
    session index: switch "suffix" new; CopyIndex: old); with the log, probes and trace on, its drafts and rounds
    are unchanged, text drafts are labelled, and under the A/B baseline's rule the offline replay reproduces the
    rounds (copy8, suffix-old)."""

    from tensorfold.families.qwen4_exp.cuda import multi as multi_module
    from tensorfold.families.qwen4_exp.cuda.decode import Engine

    w = _model()
    prompts, samplings = [QUOTE_A, QUOTE_B], [Sampling(seed=1), Sampling(seed=2)]
    starts = {1: len(QUOTE_A), 2: len(QUOTE_B)}

    def scripted(seed, position):                        # past the reply: rows accept never keeps
        reply = SCRIPTS[seed]
        return reply[position - starts[seed]] if position - starts[seed] < len(reply) else 1

    real_sample = Engine.sample

    def first(self, logits, positions, sampling, *, draft=False, gathered=True):
        if draft:
            return real_sample(self, logits, positions, sampling, draft=draft, gathered=gathered)
        return [scripted(sampling.seed, int(p)) for p in positions]

    monkeypatch.setattr(Engine, "sample", first)
    monkeypatch.setattr(multi_module, "sample_streams", lambda logits, starts_, positions, smps: [
        [scripted(smp.seed, int(p)) for p in pos] for pos, smp in zip(positions, smps)])
    real_collect = multi_module.collect_streams

    def collect(packets, readback=None, waited=None):     # the round's packets and status words, then the script
        real_collect(packets, readback, waited)
        return [[scripted(smp.seed, int(p)) for p in pos] for pos, smp in zip(packets.positions, packets.samplings)]

    monkeypatch.setattr(multi_module, "collect_streams", collect)
    _switch(monkeypatch, tmp_path, draft_rule=rule, suffix=suffix)
    for env in ENVS:
        monkeypatch.delenv(env, raising=False)
    monkeypatch.setenv("TENSORFOLD_MTP_COPY", "1")
    counts = [len(SCRIPTS[1]), len(SCRIPTS[2])]
    _, plain, plain_stats, plain_drafts = _decode(w, monkeypatch, 0.5, prompts=prompts, samplings=samplings,
                                                  counts=counts)
    assert plain == [SCRIPTS[1], SCRIPTS[2]]
    log, trace = tmp_path / "drafts.jsonl", tmp_path / "trace"
    monkeypatch.setenv(draft_log.LOG_ENV, str(log))
    monkeypatch.setenv(draft_log.PROBE_ENV, "1.0")
    monkeypatch.setenv(draft_log.TRACE_ENV, str(trace))
    dec, outs, stats, drafts = _decode(w, monkeypatch, 0.5, prompts=prompts, samplings=samplings, counts=counts)
    assert outs == plain and stats == plain_stats and drafts == plain_drafts
    dec.close_telemetry()
    entries = [st for line in log.read_text().splitlines() for st in json.loads(line)["streams"]]
    src = "suffix" if suffix == "new" else "copy"
    copies = [st for st in entries if st["src"] == src]
    assert copies and any(st["accepted"] > 0 for st in copies)
    for st in copies:                                    # text drafts: no head features, exact labels
        assert all(d["p"] is None and d["m"] is None and d["L"] >= 4 and d["hit"] is not None for d in st["drafts"])
        assert all((d["c"] is not None) == (rule == "new") and (d["a0"] is not None) == (rule == "new")
                   for d in st["drafts"])
        lead = next((i for i, d in enumerate(st["drafts"]) if d["label"] != 1 or not d["inc"]), len(st["drafts"]))
        assert st["accepted"] == lead or st["done"]
    result = replay.run(replay.load([str(trace)]), ARGS)
    parity = result["parity" if suffix == "old" else "parity_suffix"]
    assert parity == ({"checked": 2, "equal": 2, "differ": []} if rule == "old" else
                      {"checked": 0, "equal": 0, "differ": []})
    copy8 = result["results"]["copy8" if suffix == "old" else "suffix-old"]["all"]
    assert copy8["suffix_tokens"] > 0 and copy8["tokens"] == sum(len(s) - 1 for s in SCRIPTS.values())

"""The offline draft replay (tools/draft_replay.py) on hand-computed traces."""

import importlib.util
import json
from pathlib import Path
import random
import sys

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools/draft_replay.py"
spec = importlib.util.spec_from_file_location("draft_replay", TOOL)
replay = importlib.util.module_from_spec(spec)
sys.modules["draft_replay"] = replay
spec.loader.exec_module(replay)
draft_rule = replay.draft_rule

P = list(range(100, 130))                                  # the prompt: 30 distinct tokens
Y = [200, 201, 202] + P[:20] + [400, 401, 402]             # the reply quotes the prompt's first 20 tokens


def _trace(rounds, *, rid="a", order=1, after=0, process="p", out=Y, prompt=P, copy=False, background=False):
    return {"v": 1, "rid": rid, "start": 0.0, "end": 1.0, "prompt": list(prompt), "out": list(out),
            "count": len(out), "stop_eos": True, "sampling": None, "draft": True, "constrained": False, "cached": 0,
            "background": background, "error": None, "process": process, "order": order, "after": after,
            "dropped": 0, "decoder": {"depth": 4, "confidence": 0.5, "copy": copy, "copy_match": 8},
            "rounds": [{"n": n, "rows": rows, "acc": acc, "src": src, "live": 1, "mixed": False, "ms": 40.0}
                       for n, rows, acc, src in rounds]}


SERIAL = [(i, 1, 0, "") for i in range(1, len(Y))]          # recorded: one token a round, no drafts


def _args(**kw):
    ns = dict(t0=34.6, row_ms=6.5, step_ms=1.6, anchors=[4, 8], windows=[5, 8], recent=32, longest=64,
              corpus_tokens=1_000_000, corpus_segment=262_144, mtp_drafts=2, mtp_accept=0.75, seed=0,
              tables=None, lam=2.33 / 50, confidence=0.5)
    ns.update(kw)
    return type("Args", (), ns)()


def _sim(tr, policy, name="p"):
    report = replay.Report(lambda rows, steps: 34.6 + (rows - 1) * 6.5 + steps * 1.6)
    got = replay.simulate(tr, policy, replay.Mtp([tr], 2, 0.75), report, name, random.Random(0))
    return got, report.summary()[name]["all"]


def test_copy8_replays_todays_copy_drafts_exactly():
    tr = _trace(SERIAL)
    copies = replay.copy_index_class()(8)
    seq = P + Y

    def copy8(i, context, room):
        drafts = copies.propose(replay.Prefix(seq, context), max(min(4, room), 8))[:min(4, room)]
        return (drafts, "copy") if drafts else None

    got, st = _sim(tr, copy8)
    # the context's last 8 tokens first repeat the prompt at i = 11 (P[0:8]); then P[5:13], then P[10:18]
    want = [(i, 1, 0, "") for i in range(1, 11)] + [(11, 5, 4, "copy"), (16, 5, 4, "copy"), (21, 5, 2, "copy"),
                                                     (24, 1, 0, ""), (25, 1, 0, "")]
    assert got == want
    assert st["tokens"] == len(Y) - 1 and st["suffix_tokens"] == 5 + 5 + 3 and st["mtp_exact"] == 12


def test_session_s1_anchor_four_matches_earlier_and_reaches_its_window():
    tr = _trace(SERIAL)
    seq = P + Y
    index = replay.Index(4)
    index.extend(seq)

    def s1(i, context, room):
        length, e = replay.match(index, seq, context, context - 1, 32, 64)
        return (replay.continuation(seq, e, context, min(7, room)), "session") if length else None

    got, st = _sim(tr, s1)
    # P[0:4] first repeats at i = 7; at i = 23 the continuation P[20:23] (room 3) misses the reply's own ending
    want = [(i, 1, 0, "") for i in range(1, 7)] + [(7, 8, 7, "session"), (15, 8, 7, "session"), (23, 4, 0, "session"),
                                                    (24, 1, 0, ""), (25, 1, 0, "")]
    assert got == want
    assert st["coverage"] == pytest.approx(17 / 25)


def test_matches_prefer_the_longest_backward_match_then_the_newest_and_continue_periodically():
    seq = [1, 2, 3, 9, 7, 7, 2, 3, 9, 8, 5, 2, 3, 9]
    index = replay.Index(3)
    index.extend(seq)
    n = len(seq)
    # [2, 3, 9] ends at 3 (preceded by 1), 8 (preceded by 7) and 13 (the needle itself); the needle is preceded by 5
    assert replay.match(index, seq, n, n - 1, 32, 64) == (3, 8)            # equal lengths: the newest
    seq2 = [5, 2, 3, 9, 4, 0, 2, 3, 9, 6, 5, 2, 3, 9]
    index2 = replay.Index(3)
    index2.extend(seq2)
    assert replay.match(index2, seq2, len(seq2), len(seq2) - 1, 32, 64) == (4, 3)   # the longer, older match
    assert replay.match(index2, seq2, len(seq2), len(seq2) - 1, 1, 64) == (3, 8)    # only the newest compared
    periodic = [4, 1, 2, 1, 2, 1, 2]                                       # "1 2" repeats up to the end
    index3 = replay.Index(2)
    index3.extend(periodic)
    length, e = replay.match(index3, periodic, 7, 6, 32, 64)
    assert (length, e) == (4, 4)
    assert replay.continuation(periodic, e, 7, 5) == [1, 2, 1, 2, 1]


def test_the_oracle_fills_its_window_and_rounds_cost_the_model():
    tr = _trace(SERIAL)
    got, st = _sim(tr, lambda i, context, room: (Y[i:i + min(4, room)], "oracle"))
    assert got == [(1, 1, 0, ""), (2, 5, 4, "oracle"), (7, 5, 4, "oracle"), (12, 5, 4, "oracle"),
                   (17, 5, 4, "oracle"), (22, 5, 3, "oracle")]
    assert st["ms"] == pytest.approx(36.2 + 5 * 62.2)
    assert st["tokens_per_round"] == pytest.approx(25 / 6)


def test_mtp_rounds_reuse_recorded_ones_then_the_bucket_then_tier0():
    rounds = [(1, 3, 2, "mtp"), (4, 3, 0, "mtp"), (5, 2, 1, "mtp")]
    tr = _trace(rounds, out=Y[:7])
    mtp = replay.Mtp([tr], 2, 0.75)
    rng = random.Random(1)
    assert mtp.round({4: (3, 0, "mtp")}, 4, 34, 3, 3, rng) == (2, 0, "mtp", "exact")
    drafts, acc, src, how = mtp.round({}, 2, 34, 1, 5, rng)                # the bucket's rounds, capped by room
    assert how == "bucket" and drafts <= 1 and acc <= drafts and src == "mtp"
    assert replay.Mtp([], 2, 1.0).round({}, 2, 34, 4, 9, rng) == (2, 2, "mtp", "tier0")
    assert replay.Mtp([], 2, 1.0).round({}, 2, 34, 4, 2, rng) == (2, 1, "mtp", "tier0")   # the reply ends first


def test_the_corpus_is_causal_per_decoder_and_holds_only_foreground_replies(tmp_path):
    secret = [500 + i for i in range(30)]
    first = _trace([(i, 1, 0, "") for i in range(1, 31)], rid="first", order=1, out=[1] + secret, prompt=[7, 8, 9])
    reply = [2, 3] + secret[:20] + [4]
    early = _trace([(i, 1, 0, "") for i in range(1, len(reply))], rid="early", order=2, after=0, out=reply,
                   prompt=[secret[0], 6, 6])                 # joined before ``first`` finished
    late = dict(early, rid="late", after=1)                  # joined after
    elsewhere = dict(late, process="q")                      # another decoder: its corpus never held ``first``
    hidden = dict(first, background=True)
    args = _args(anchors=[4], windows=[8])
    for traces, helps in (([first, early], False), ([first, late], True), ([first, elsewhere], False),
                          ([hidden, late], False)):
        for f in tmp_path.iterdir():
            f.unlink()
        for i, tr in enumerate(traces):
            (tmp_path / f"{i}.json").write_text(json.dumps(tr))
        result = replay.run(replay.load([str(tmp_path)]), args)
        session = result["results"]["session-k4-w8"]["all"]
        corpus = result["results"]["corpus-k4-w8"]["all"]
        assert (corpus["suffix_tokens"] > session["suffix_tokens"]) is helps, traces[1]["rid"]
        assert corpus["tokens"] == session["tokens"]


def test_corpus_segments_go_oldest_first_past_the_cap():
    corpus = replay.Corpus([2], 5, 3)
    corpus.add([1, 2, 3])
    corpus.add([4, 5, 6])                                    # a new segment; 6 tokens > 5: the oldest goes
    assert corpus.size == 3 and len(corpus.segments) == 1
    assert corpus.match(2, [9, 4, 5], 3, 32, 64)[0] == 2 and corpus.match(2, [9, 1, 2], 3, 32, 64)[0] == 0
    corpus.add([7, 8])                                       # the open segment is full (3 tokens): a new one
    assert corpus.size == 5 and [s[0] for s in corpus.segments] == [[3], [2]]


def test_load_rejects_incomplete_traces(tmp_path):
    tr = _trace(SERIAL)
    del tr["decoder"]["copy_match"]
    (tmp_path / "a.json").write_text(json.dumps(tr))
    with pytest.raises(ValueError, match="version 1"):
        replay.load([str(tmp_path)])


def test_run_reports_parity_and_skips_serial_requests(tmp_path):
    want = [(i, 1, 0, "") for i in range(1, 11)] + [(11, 5, 4, "copy"), (16, 5, 4, "copy"), (21, 5, 2, "copy"),
                                                     (24, 1, 0, ""), (25, 1, 0, "")]
    (tmp_path / "a.json").write_text(json.dumps(_trace(want, copy=True)))
    serial = dict(_trace(SERIAL, rid="b", order=2), draft=False)
    (tmp_path / "b.json").write_text(json.dumps(serial))
    result = replay.run(replay.load([str(tmp_path)]), _args())
    assert result["parity"] == {"checked": 1, "equal": 1, "differ": []}
    assert result["results"]["recorded"]["all"]["requests"] == 2
    assert result["results"]["copy8"]["all"]["requests"] == 1
    assert result["results"]["copy8"]["all"]["tokens"] == len(Y) - 1
    assert "copy8 parity with recorded rounds: 1/1" in replay.table(result)


def test_prefix_reads_like_a_copied_slice():
    seq = list(range(50))
    view = replay.Prefix(seq, 30)
    assert len(view) == 30 and view[-1] == 29 and view[-8:] == seq[22:30] and view[:5] == seq[:5]
    assert view[3:40] == seq[3:30] and tuple(view[10:18]) == tuple(seq[10:18])
    with pytest.raises(IndexError):
        view[30]


def test_dropped_traces_add_up_across_decoders(tmp_path):
    for i, (process, dropped) in enumerate((("p", 1), ("p", 2), ("q", 3))):
        tr = dict(_trace(SERIAL, rid=f"r{i}", order=i + 1, process=process), dropped=dropped)
        (tmp_path / f"{i}.json").write_text(json.dumps(tr))
    result = replay.run(replay.load([str(tmp_path)]), _args(windows=[5], anchors=[4]))
    assert result["traces_dropped"] == 5 and "dropped 5 traces" in replay.table(result)


COPY_ROUNDS = [(i, 1, 0, "") for i in range(1, 11)] + [(11, 5, 4, "copy"), (16, 5, 4, "copy"), (21, 5, 2, "copy"),
                                                        (24, 1, 0, ""), (25, 1, 0, "")]


def _recorded_suffix(rounds, rule="old"):
    tr = _trace([(n, rows, acc, "suffix" if src == "copy" else src) for n, rows, acc, src in rounds], copy=True)
    tr["text"] = "suffix"
    for r in tr["rounds"]:
        r["rule"] = rule
    tr["decoder"].update(anchor=4, recent=32, longest=64)
    return tr


def test_suffix_old_replays_the_servers_session_drafts_and_checks_parity(tmp_path):
    """The server's session drafts under the A/B baseline's rule: whole continuations of matches of 8+ tokens (the
    4-token anchor first matches at i = 7 with 4 tokens: left to MTP), so this reply repeats copy8's rounds."""

    (tmp_path / "a.json").write_text(json.dumps(_recorded_suffix(COPY_ROUNDS)))
    wrong = _recorded_suffix(COPY_ROUNDS[:10] + [(11, 4, 3, "copy")] + COPY_ROUNDS[11:])
    (tmp_path / "b.json").write_text(json.dumps(dict(wrong, rid="b", order=2)))
    newer = _recorded_suffix(COPY_ROUNDS, rule="new")               # drafted by the cost-aware rule: not checked
    (tmp_path / "c.json").write_text(json.dumps(dict(newer, rid="c", order=3)))
    result = replay.run(replay.load([str(tmp_path)]), _args())
    assert result["parity_suffix"] == {"checked": 2, "equal": 1, "differ": ["b"]}
    assert result["parity"] == {"checked": 0, "equal": 0, "differ": []}       # copy8 checks copy traces only
    st = result["results"]["suffix-old"]["all"]
    assert st["suffix_tokens"] == 3 * (5 + 5 + 3) and st["tokens"] == 3 * (len(Y) - 1)        # three traces
    assert "suffix-old parity with recorded rounds: 1/2" in replay.table(result)


def test_suffix_new_takes_what_pays_under_the_prior_or_fitted_chances(tmp_path):
    tr = _recorded_suffix(COPY_ROUNDS)
    (tmp_path / "a.json").write_text(json.dumps(tr))
    result = replay.run(replay.load([str(tmp_path)]), _args(windows=[5]))
    for name in ("suffix-new-w5", "copy8-new"):                       # the prior: whole windows, as the baseline
        assert result["results"][name]["all"]["rounds"] == result["results"]["suffix-old"]["all"]["rounds"], name
    rule = draft_rule.Predictor.prior(0.5).as_dict()
    rule["source"] = {"tool": "test"}
    for policy in ("greedy", "sampled"):     # matches of 4+: 2 tokens pay (.9, .81 >= .303), a third (.243) not;
        rule["tables"]["suffix"][policy] = {"feature": "L", "depth_edges": [0, 2], "score_edges": [3.5],   # an
                                            "table": [[0.0, 0.9], [0.0, 0.3]], "abar": [0.9, 0.3]}  # MTP chain
        rule["tables"]["mtp"][policy] = {"feature": "p", "depth_edges": [0], "score_edges": [], "table": [[0.5]],
                                         "abar": [0.5]}                                             # is worth .197
    tables = tmp_path / "rule.json"
    tables.write_text(json.dumps(rule))
    result = replay.run(replay.load([str(tmp_path / "a.json")]), _args(windows=[5], tables=str(tables)))
    st = result["results"]["suffix-new-w5"]["all"]
    # recorded MTP rounds to i = 6; text rounds of two drafts from i = 7 (the 4-token anchor) to 22, where the quote
    # ends after one; then the recorded rounds at 24 and 25
    assert (st["rounds"], st["drafted"], st["accepted"], st["suffix_tokens"]) == (6 + 6 + 2, 12, 11, 17)


def test_stop_rules_compare_on_labelled_probe_chains(tmp_path):
    """A probe chain labelled to its room: today's rule verifies the confident first draft and stops at the low
    second; the prior's rule decides the same; a table that trusts every draft verifies all four and keeps two."""

    def entry(p, label):
        return {"j": 0, "p": p, "m": 1.0, "e": None, "g": None, "z": None, "c": None, "L": None, "a0": None,
                "inc": False, "hit": label, "label": label}

    drafts = [entry(0.9, 1), entry(0.3, 1), entry(0.9, 0), entry(0.9, 0)]
    for j, d in enumerate(drafts):
        d["j"] = j
    stream = {"rid": "r", "ctx": 1000, "n": 3, "greedy": False, "src": "mtp", "probe": True, "join": False,
              "constrained": False, "verified": 1, "accepted": 1, "done": False, "drafts": drafts}
    lines = [{"streams": [stream]}, {"streams": [dict(stream, probe=False)]},
             {"streams": [dict(stream, drafts=[dict(d, label=None) for d in drafts])]},
             {"streams": [{k: v for k, v in stream.items() if k != "greedy"}]}]
    log = tmp_path / "drafts.jsonl"
    log.write_text("\n".join(json.dumps(line) for line in lines))
    rule = draft_rule.Predictor.prior(0.5).as_dict()
    rule["source"] = {"tool": "test"}
    for policy in ("greedy", "sampled"):
        rule["tables"]["mtp"][policy] = {"feature": "p", "depth_edges": [0], "score_edges": [], "table": [[0.9]],
                                         "abar": [0.9]}
    tables = tmp_path / "rule.json"
    tables.write_text(json.dumps(rule))
    got = replay.rules(replay.load_log([str(log)]), _args(tables=str(tables)))
    res = got["results"]
    assert (res["confidence"]["all"]["rounds"], res["confidence"]["all"]["rows"], res["confidence"]["all"]["tokens"]) \
        == (2, 2 + 2, 2 + 2)                                          # 1 draft, kept: 2 tokens (the stream without
    assert res["prior"]["all"]["rows"] == 2 and res["prior"]["all"]["tokens"] == 2      # "greedy": today's only)
    assert (res["fit"]["all"]["rows"], res["fit"]["all"]["accepted"], res["fit"]["all"]["tokens"]) == (5, 2, 3)
    assert res["confidence"]["all"]["ms"] == pytest.approx(2 * (34.6 + 6.5 + 2 * 1.6))
    assert res["fit"]["all"]["ms"] == pytest.approx(34.6 + 4 * 6.5 + 4 * 1.6)
    assert got["skipped"] == {"unlabelled": 1, "no_greedy_field": 1, "no_features": 0}
    # allocate on the prior's chances (.9, .18, .162, .146): one draft has the best expected tokens a modelled ms
    alloc = res["allocate"]["all"]
    assert (alloc["rounds"], alloc["rows"], alloc["tokens"]) == (1, 2, 2)
    assert alloc["ms"] == pytest.approx(34.6 + 6.5 + 4 * 1.6)


def test_the_automaton_finds_the_longest_earlier_match_after_every_token():
    """The S2 reference (design 4.2): after each token, the longest suffix that occurred before (brute force), and
    a real occurrence of it; S1 never finds a longer one."""

    rng = random.Random(5)
    for alphabet in (2, 3, 7, 50):
        automaton, seq = replay.Automaton(), []
        for step in range(700):
            c = rng.randrange(alphabet) if rng.random() < 0.7 or len(seq) < 10 else seq[rng.randrange(len(seq))]
            length, end = automaton.extend(c)
            seq.append(c)
            n, want = len(seq), 0
            for e in range(n - 1):
                k = 0
                while k <= e and seq[e - k] == seq[n - 1 - k]:
                    k += 1
                want = max(want, k)
            assert length == want, (alphabet, step)
            if length:
                assert end < n - 1 and seq[end - length + 1:end + 1] == seq[n - length:]
            index = replay.Index(2)
            index.extend(seq)
            assert replay.match(index, seq, n, n - 1, 32, 1 << 20)[0] <= max(length, 0)

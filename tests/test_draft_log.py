"""The draft log's labels, lines and trace files, driven in the decoder's hook order without a model."""

import json
import math
import os
import stat
import threading

import pytest

from tensorfold.cuda import draft_log
from tensorfold.cuda.draft_log import DraftWatch
from tensorfold.cuda.record_writer import JsonFiles, JsonLines, RecordWriter
from tensorfold.cuda.streams import Stream, accept
from tensorfold.engine.exact_sampling import Sampling

TRUTH = [900001, 900002, 900003, 900004, 900005, 900006, 900007, 900008, 900009, 900010]   # the reply
PROMPT = [700001, 700002, 700003]
BAD = 912345                                               # a wrong draft


def _drive(watch: DraftWatch, s: Stream, rounds: list[list[tuple]], eos: tuple = ()) -> None:
    """Decode ``s`` toward TRUTH in the decoder's hook order: the first token, then rounds. ``rounds[r]`` are the
    drafts (token, depth, p, scores, kept by the rule) proposed after round r (``rounds[0]``: as the prompt joins);
    each round's window is its pending token and the kept drafts, kept as ``accept`` keeps them against TRUTH."""

    s.sid = 0
    s.take([TRUTH[0]], eos)
    watch.begin()
    watch.joined(s)                                        # no drafts yet: joined ones get features below instead
    _propose(watch, s, rounds[0], len(s.out))
    r = 0
    while not s.done:                                      # after ``rounds``, rounds without drafts
        r += 1
        nxt = rounds[r] if r < len(rounds) else []
        watch.begin()
        window = [s.out[-1]] + s.drafts
        n = len(s.out)
        sampled = TRUTH[n:n + len(window)]
        path, end = accept(window, list(range(-1, len(window) - 1)), sampled, s.count - n, eos)
        watch.verified(s, len(PROMPT) + n - 1, len(window), len(path) - 1)
        new = [window[i] for i in path[1:]] + [end]
        last = len(s.out) + len(new) >= s.count or end in eos
        _propose(watch, s, [] if last else nxt, len(s.out) + len(new))
        s.take(new, eos)
        watch.round_end([s], rows=len(window), mixed=False, pass_rows=0, steps=1 + len(nxt))
    watch.finished(s)


def _propose(watch, s, items, base):
    s.drafts = [item[0] for item in items if item[4]]       # the rule's drafts: a prefix of the chain
    if items:
        watch.drafted(s, base, "mtp", [item[:4] for item in items], probe=not all(item[4] for item in items))


def _lines(watch):
    watch.close()
    assert watch.log.failed is None
    return [json.loads(line) for line in open(watch.log.path)]


def test_off_is_none(monkeypatch):
    for env in (draft_log.LOG_ENV, draft_log.PROBE_ENV, draft_log.TRACE_ENV):
        monkeypatch.delenv(env, raising=False)
    assert DraftWatch.from_env() is None
    monkeypatch.setenv(draft_log.PROBE_ENV, "0")
    assert DraftWatch.from_env() is None


def test_probes_need_the_log_and_a_probability(monkeypatch, tmp_path):
    monkeypatch.delenv(draft_log.LOG_ENV, raising=False)
    monkeypatch.setenv(draft_log.PROBE_ENV, "0.5")
    with pytest.raises(ValueError, match="TENSORFOLD_DRAFT_LOG"):
        DraftWatch.from_env()
    monkeypatch.setenv(draft_log.LOG_ENV, str(tmp_path / "log.jsonl"))
    for bad in ("1.5", "-0.1", "nan", "often"):
        monkeypatch.setenv(draft_log.PROBE_ENV, bad)
        with pytest.raises(ValueError, match="probability"):
            DraftWatch.from_env()
    monkeypatch.setenv(draft_log.PROBE_ENV, "1")
    watch = DraftWatch.from_env()
    assert watch.margins and all(watch.probing() for _ in range(20))
    watch.close()


def test_labels_follow_the_reply_and_lines_wait_for_them(tmp_path):
    watch = DraftWatch(log=str(tmp_path / "log.jsonl"))
    s = Stream(list(PROMPT), 10, Sampling(seed=3))
    feats = {"m": 1.5, "e": float("inf"), "g": 0.25, "z": 0.125}
    rounds = [
        [(TRUTH[1], 0, 0.9, feats, True), (BAD, 1, 0.8, feats, True)],          # joined: one hit, then a miss
        # round 1 keeps TRUTH[1] and emits TRUTH[2]; next: drafts for TRUTH[3..], the third a rule-dropped shadow
        [(TRUTH[3], 0, 0.9, feats, True), (TRUTH[4], 1, 0.7, feats, True), (TRUTH[5], 2, 0.2, feats, False)],
        [(BAD, 0, 0.6, feats, True)],
        [(TRUTH[7], 0, 0.9, feats, True), (TRUTH[8], 1, 0.9, None, True)],
    ]
    _drive(watch, s, rounds)
    assert s.out == TRUTH
    lines = _lines(watch)
    text = "".join(json.dumps(x) for x in lines)
    assert all(str(t) not in text for t in TRUTH + PROMPT + [BAD])         # no token ids
    streams = [line["streams"][0] for line in lines]
    assert len({st["rid"] for st in streams}) == 1 and all(line["dropped"] == 0 for line in lines)
    assert [line["joined"] for line in lines] == [1] + [0] * (len(lines) - 1)   # counted past the round's restart
    assert not any(st["constrained"] for st in streams)
    first = next(st for st in streams if st["n"] == 1)
    assert first["src"] == "mtp" and first["verified"] == 2 and first["accepted"] == 1
    assert [(d["hit"], d["label"], d["inc"]) for d in first["drafts"]] == [(1, 1, True), (0, 0, True)]
    assert first["drafts"][0]["m"] == 1.5 and first["drafts"][0]["e"] == "inf"
    second = next(st for st in streams if st["n"] == 3)
    assert second["probe"] and second["verified"] == 2 and second["accepted"] == 2
    assert [(d["hit"], d["label"], d["inc"]) for d in second["drafts"]] == [(1, 1, True), (1, 1, True),
                                                                           (1, 1, False)]
    third = next(st for st in streams if st["n"] == 6)
    assert [(d["hit"], d["label"]) for d in third["drafts"]] == [(0, 0)] and third["accepted"] == 0
    for st in streams:                                    # acceptance is the labelled prefix, room permitting
        lead = 0
        for d in st["drafts"]:
            if not d["inc"] or d["label"] != 1:
                break
            lead += 1
        assert st["accepted"] == lead or st["done"]


def test_a_reply_that_ends_first_leaves_its_labels_null_and_its_lines_written(tmp_path):
    watch = DraftWatch(log=str(tmp_path / "log.jsonl"))
    s = Stream(list(PROMPT), 10, None)
    rounds = [[(TRUTH[1], 0, 0.9, None, True)],
              [(TRUTH[3], 0, 0.9, None, True), (TRUTH[4], 1, 0.9, None, True), (TRUTH[5], 2, 0.9, None, True)],
              [(TRUTH[6], 0, 0.9, None, True)]]
    _drive(watch, s, rounds, eos=(TRUTH[4],))             # the reply ends with its end token at index 4
    assert s.out == TRUTH[:5] and s.done
    lines = _lines(watch)
    assert not watch.lines and not watch.chains and not watch.pending and not watch.rids
    last = lines[-1]["streams"][0]
    assert last["done"] and last["accepted"] == 1
    assert [(d["hit"], d["label"]) for d in last["drafts"]] == [(1, 1), (1, 1), (None, None)]


def test_unlabelled_lines_are_bounded(tmp_path, monkeypatch):
    monkeypatch.setattr(draft_log, "OPEN", 3)
    watch = DraftWatch(log=str(tmp_path / "log.jsonl"))
    s = Stream(list(PROMPT), 100, None)
    s.sid = 0
    s.take([TRUTH[0]])
    watch.joined(s)
    for _ in range(6):                                    # a stream that is verified but never reaches its drafts
        watch.begin()
        watch.drafted(s, 50, "mtp", [(1, 0, 0.5, None)])
        watch.verified(s, 10, 2, 0)
        watch.round_end([s], rows=2, mixed=False, pass_rows=0, steps=1)
    assert len(watch.lines) == 3 and watch.forced == 3
    watch.finished(s)
    lines = _lines(watch)
    assert len(lines) == 6 and max(line["forced"] for line in lines) == 3


def test_trace_files_hold_the_request_and_only_its_owner_reads_them(tmp_path):
    watch = DraftWatch(trace=str(tmp_path / "trace"), decoder={"depth": 3, "confidence": 0.5, "copy": True})
    assert not watch.margins and watch.log is None
    s = Stream(list(PROMPT), 10, Sampling(seed=9, top_k=20, top_p=0.95))
    rounds = [[(TRUTH[1], 0, None, None, True), (TRUTH[2], 1, None, None, True)], [(BAD, 0, None, None, True)]]
    _drive(watch, s, rounds)
    watch.close()
    files = os.listdir(tmp_path / "trace")
    assert len(files) == 1 and files[0].endswith(".json")
    path = tmp_path / "trace" / files[0]
    assert stat.S_IMODE(os.stat(path).st_mode) == 0o600
    doc = json.loads(path.read_text())
    assert doc["prompt"] == PROMPT and doc["out"] == TRUTH and doc["count"] == 10
    assert doc["sampling"] == {"seed": 9, "temperature": 1.0, "top_k": 20, "top_p": 0.95, "min_p": 0.0}
    assert doc["decoder"] == {"depth": 3, "confidence": 0.5, "copy": True} and doc["draft"] and not doc["constrained"]
    assert sum(r["acc"] + 1 for r in doc["rounds"]) == len(TRUTH) - 1
    assert doc["rounds"][0] == {"n": 1, "rows": 3, "acc": 2, "src": "mtp", "live": 1, "mixed": False,
                                "ms": doc["rounds"][0]["ms"], "rule": "old"}
    assert doc["text"] is None                           # neither a session index nor copies on the stream
    assert doc["start"] <= doc["end"] and doc["dropped"] == 0
    assert (doc["order"], doc["after"]) == (1, 0) and doc["process"] == watch.process


def test_yielded_or_dropped_streams_write_no_trace(tmp_path):
    watch = DraftWatch(trace=str(tmp_path / "trace"))
    s = Stream(list(PROMPT), 10, None)
    s.sid = 4
    s.take([TRUTH[0]])
    watch.joined(s)
    watch.finished(s)                                     # not done: yielded or dropped
    watch.close()
    assert not (tmp_path / "trace").exists() or not os.listdir(tmp_path / "trace")


def test_features_encode_infinity_and_nan_as_strings():
    assert draft_log._num(math.inf) == "inf" and draft_log._num(-math.inf) == "-inf"
    assert draft_log._num(math.nan) == "nan" and draft_log._num(None) is None and draft_log._num(0.123456) == 0.12346


class _Slow(RecordWriter):
    def __init__(self, **kw):
        self.taken, self.release, self.seen = threading.Event(), threading.Event(), []
        super().__init__(name="test-slow", **kw)

    def write(self, record):
        self.taken.set()
        assert self.release.wait(10)
        self.seen.append(record)


def test_a_full_queue_drops_the_oldest_and_counts_only_real_drops():
    writer = _Slow(queue_size=2)
    writer.put(1)
    assert writer.taken.wait(10)                          # the thread holds record 1
    for record in (2, 3, 4, 5):
        writer.put(record)
    assert writer.dropped == 2                            # 2 and 3 made room for 4 and 5
    writer.release.set()
    writer.close()
    assert writer.seen == [1, 4, 5] and writer.failed is None


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")    # the failure, by design
def test_a_failed_writer_reports_at_close_and_drops_what_follows(tmp_path):
    files = JsonFiles(str(tmp_path / "t"), name="test-files")
    assert stat.S_IMODE(os.stat(tmp_path / "t").st_mode) == 0o700     # made at startup, owner only
    files.put(("a.json", {"bad": {1, 2}}))                # sets do not serialize
    files.thread.join(10)
    assert isinstance(files.failed, TypeError) and os.listdir(tmp_path / "t") == []   # no partial file left
    files.put(("b.json", {}))
    assert files.dropped == 1
    with pytest.raises(RuntimeError, match="test-files stopped"):
        files.close()
    with pytest.raises(OSError):
        JsonLines(str(tmp_path / "missing" / "log.jsonl"), name="test-lines")       # a bad path fails at startup


def test_a_round_that_fails_leaves_no_entries_for_the_next(tmp_path):
    watch = DraftWatch(log=str(tmp_path / "log.jsonl"), trace=str(tmp_path / "trace"))
    s = Stream(list(PROMPT), 10, None)
    s.sid = 2
    s.take([TRUTH[0]])
    watch.joined(s)
    watch.begin()
    watch.verified(s, 3, 1, 0)                           # the round raises before round_end; the decoder drops s
    watch.finished(s)
    other = Stream(list(PROMPT), 10, None)
    other.sid = 3
    other.take([TRUTH[0]])
    watch.joined(other)
    watch.begin()                                        # the next round
    watch.verified(other, 3, 1, 0)
    other.take([TRUTH[1]])
    watch.round_end([other], rows=1, mixed=False, pass_rows=0, steps=0)
    watch.finished(other)
    lines = _lines(watch)
    assert [len(line["streams"]) for line in lines] == [1] and lines[0]["joined"] == 2
    assert not watch.traces and not watch.rids and not os.listdir(tmp_path / "trace")   # neither finished done


@pytest.mark.filterwarnings("ignore::pytest.PytestUnhandledThreadExceptionWarning")    # the failure, by design
def test_a_failed_writer_does_not_keep_the_others_open_or_fail_a_retried_close(tmp_path):
    pytest.importorskip("torch")
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder

    watch = DraftWatch(log=str(tmp_path / "log.jsonl"), trace=str(tmp_path / "trace"))
    watch.log.put({"bad": {1}})                           # kills the log writer
    watch.log.thread.join(10)
    watch.trace.put(("a.json", {"kept": True}))
    dec = object.__new__(MultiDecoder)
    dec.prof, dec.watch = None, watch
    with pytest.raises(RuntimeError, match="tensorfold-draft-log stopped"):
        dec.close_telemetry()
    assert not watch.trace.thread.is_alive() and os.listdir(tmp_path / "trace") == ["a.json"]   # still written
    assert dec.watch is None
    dec.close_telemetry()                                 # a retried shutdown: nothing left to report

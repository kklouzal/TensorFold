"""Serving-matrix benchmark contracts against a local fake OpenAI server; no GPU or model is needed."""
import hashlib
import importlib.util
import json
import math
import statistics
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest

from test_deepseek_v41_serving import app_for, model_dir as deepseek_model_dir  # noqa: F401

TOOLS = Path(__file__).resolve().parents[1] / "tools"


@pytest.fixture(scope="module")
def mb():
    spec = importlib.util.spec_from_file_location("matrix_benchmark_under_test", TOOLS / "matrix_benchmark.py")
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


CORPUS = "".join(f"line {i} of the frozen corpus\n" for i in range(2000))


def make_plan(mb, tmp_path, lengths, levels, reps):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text(CORPUS, encoding="utf-8")
    prompts = {}
    for mode, arrival, c, length, rep, slot in mb.plan_keys(["greedy", "sampled"], ["sync"], levels, lengths, reps,
                                                            True):
        key = mb.prompt_key(mode, arrival, c, length, rep, slot)
        start = (int(hashlib.sha256(key.encode()).hexdigest(), 16) % 1000) * 10
        text = mb.message(key, CORPUS[start:start + length])
        prompts[key] = {"start": start, "end": start + length, "tokens": len(text),
                        "sha256": mb.sha256(text.encode("utf-8"))}
    plan = tmp_path / "plan.json"
    plan.write_text(json.dumps({"schema": mb.PLAN_SCHEMA, "version": 1, "corpus_sha256": mb.sha256(CORPUS.encode()),
                                "prompts": prompts}))
    return plan, corpus


class Fake(BaseHTTPRequestHandler):
    """Prompt tokens = message characters; one SSE piece per token; optional prefix-cache report."""
    cached = 0
    seen: ClassVar[list] = []

    def log_message(self, *args):
        pass

    def do_GET(self):
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "m", "max_model_len": 4096}]}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_error(404)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Fake.seen.append(request)
        prompt = len(request["messages"][0]["content"])
        n = request["max_tokens"]
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        for i in range(n):
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {"content": f"t{i} "}}]}).encode() +
                             b"\n\n")
            self.wfile.flush()
        usage = {"prompt_tokens": prompt, "completion_tokens": n,
                 "prompt_tokens_details": {"cached_tokens": Fake.cached}}
        self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": usage}).encode() + b"\n\ndata: [DONE]\n\n")


@pytest.fixture
def server():
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Fake)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    Fake.cached, Fake.seen = 0, []
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    httpd.server_close()


def run_args(mb, url, plan, corpus, out, *extra):
    argv = ["matrix_benchmark.py", "run", "--url", url, "--model", "m", "--plan", str(plan), "--corpus", str(corpus),
            "--out", str(out), "--label", "test", "--lanes", "2", "--concurrency", "1,2,4", "--lengths", "100,4090",
            "--reps", "2", "--output-tokens", "8", "--warmup-lengths", "100", *extra]
    old = sys.argv
    sys.argv = argv
    try:
        return mb.main()
    finally:
        sys.argv = old


@pytest.mark.parametrize(("command", "arguments"), [
    ("plan", ["--tokenizer", "model-dir", "--template", "chat.jinja"]),
    ("run", ["--plan", "plan.json", "--model", "m", "--label", "candidate"]),
])
@pytest.mark.parametrize("override", [False, True])
def test_plan_and_run_share_default_matrix_and_allow_explicit_overrides(mb, monkeypatch, command, arguments, override):
    argv = ["matrix_benchmark.py", command, "--corpus", "corpus.txt", "--out", "out.json", *arguments]
    if override:
        argv.extend(["--concurrency", "1,32", "--lengths", "4096,65536"])
    monkeypatch.setattr(sys, "argv", argv)
    parsed = []
    monkeypatch.setattr(mb, "build_plan" if command == "plan" else "run", parsed.append)
    mb.main()
    args, = parsed
    assert args.concurrency == ([1, 32] if override else [1, 2, 4, 8, 16])
    assert args.lengths == ([4096, 65536] if override else [8192])
    assert args.modes == ["greedy", "sampled"]


def test_deepseek_plan_counts_the_standard_cuda_render(mb, deepseek_model_dir):
    model_dir = deepseek_model_dir
    app = app_for(model_dir, default_thinking=True)
    text = "Keep literal text <｜DSML｜ calls> in a quoted example.\nSummarize the rest."
    args = SimpleNamespace(deepseek_v41=True, tokenizer=model_dir)
    count, offsets, metadata = mb._plan_tokenization(args, text)
    expected = app.tokenize({"messages": [{"role": "user", "content": text}],
                             "chat_template_kwargs": {"enable_thinking": False}})
    assert count(text) == expected["count"]
    assert offsets == [start for start, _ in app.tok.encode(text, add_special_tokens=False).offsets]
    assert metadata["template_sha256"] == expected["chat_template_sha256"]


def test_run_measures_supported_cells_and_records_gated_ones(mb, server, tmp_path):
    plan, corpus = make_plan(mb, tmp_path, [100, 4090], [1, 2, 4], 2)
    out = tmp_path / "result.json"
    assert run_args(mb, server, plan, corpus, out) == 0
    data = json.loads(out.read_text())
    measured = {(c["mode"], c["c"], c["length"], c["rep"]) for c in data["cells"]}
    assert measured == {(m, c, 100, r) for m in ("greedy", "sampled") for c in (1, 2) for r in (0, 1)}
    reasons = {(g["c"], g["length"]): g["status"] for g in data["gated"] if g["mode"] == "greedy"}
    assert reasons == {(1, 4090): "infeasible", (2, 4090): "infeasible", (4, 100): "pending",
                       (4, 4090): "pending"}
    for cell in data["cells"]:
        assert cell["status"] == "measured" and cell["metrics"]["requests"] == cell["c"]
        assert cell["metrics"]["prefill_agg_tps"] > 0 and cell["metrics"]["decode_median_tps"] > 0
        assert cell["metrics"]["decode_agg_tps"] > 0
    keys = [r["key"] for c in data["cells"] + data["warmup"] for r in c["requests"]]
    assert len(keys) == len(set(keys))  # every request has its own prompt
    greedy = [r for r in Fake.seen if r.get("temperature") == 0]
    sampled = [r for r in Fake.seen if "temperature" not in r]
    assert greedy and sampled and all("seed" in r and "top_p" not in r for r in sampled)


def test_prefix_cache_hits_invalidate_the_cell(mb, server, tmp_path):
    plan, corpus = make_plan(mb, tmp_path, [100, 4090], [1, 2, 4], 2)
    Fake.cached = 3
    out = tmp_path / "result.json"
    assert run_args(mb, server, plan, corpus, out, "--modes", "greedy") == 1
    data = json.loads(out.read_text())
    assert all(c["status"] == "invalid" for c in data["cells"])
    assert "prefix cache" in data["cells"][0]["problems"][0]


def arm(tmp_path, mb, name, values, session=None, plan="p"):
    """values: {(c, length): [(prefill, decode) per repeat]}; ``session``: the server instance id (else none)."""
    cells = []
    for (c, length), reps in values.items():
        for rep, (prefill, decode) in enumerate(reps):
            cells.append({"mode": "greedy", "arrival": "sync", "c": c, "length": length, "rep": rep,
                          "status": "measured", "problems": [],
                          "metrics": {"prefill_agg_tps": prefill, "decode_median_tps": decode}})
    path = tmp_path / name
    path.write_text(json.dumps({"schema": mb.SCHEMA, "label": name, "plan_sha256": plan, "output_tokens": 512,
                                "session": session and {"instance_id": session}, "cells": cells, "gated": []}))
    return str(path)


def compare(mb, tmp_path, ref, cand, *calibration, aggregate=False):
    out = tmp_path / "cmp.json"
    old = sys.argv
    sys.argv = ["matrix_benchmark.py", "compare", ref, cand, "--json", str(out)]
    if aggregate:
        sys.argv.append("--aggregate-decode")
    for group in calibration:
        sys.argv += ["--calibration", group]
    try:
        assert mb.main() == 0
    finally:
        sys.argv = old
    return json.loads(out.read_text())


def test_aggregate_decode_uses_the_whole_cohort_interval(mb):
    requests = [
        {"sent": 100.0, "arrivals_s": [1, 2, 3], "usage": {"completion_tokens": 13}},
        {"sent": 103.0, "arrivals_s": [1, 2, 3], "usage": {"completion_tokens": 19}},
    ]
    assert mb.aggregate_decode_tps(requests) == 6.0  # 30 tokens over 101–106, including the gap
    assert mb.aggregate_decode_tps(requests[:1]) == 6.0
    assert mb.aggregate_decode_tps([]) is None


@pytest.mark.parametrize("record", [
    {},
    {"sent": "100", "arrivals_s": [1, 2], "usage": {"completion_tokens": 2}},
    {"sent": 100, "arrivals_s": [1, 3, 2], "usage": {"completion_tokens": 2}},
    {"sent": 100, "arrivals_s": [1, 2], "usage": {"completion_tokens": True}},
    {"sent": 100, "arrivals_s": [1, float("nan"), 2], "usage": {"completion_tokens": 2}},
])
def test_aggregate_decode_requires_complete_ordered_arrivals_and_usage(mb, record):
    complete = {"sent": 100.0, "arrivals_s": [1, 2, 3], "usage": {"completion_tokens": 13}}
    assert mb.aggregate_decode_tps([complete, record]) is None


def test_opt_in_comparison_recovers_raw_aggregate_decode_and_requires_coverage(mb, tmp_path):
    ref = arm(tmp_path, mb, "ref.json", {(1, 4096): [(5000, 230)]})
    cand = arm(tmp_path, mb, "cand.json", {(1, 4096): [(5000, 240)]})
    key = "v1/greedy/sync/c1/l4096"
    missing = compare(mb, tmp_path, ref, cand, aggregate=True)
    assert missing["coverage"]["missing_aggregate_decode"] == [key]
    assert missing["verdict"].startswith("incomplete")

    for path, span in ((ref, 4.0), (cand, 2.0)):
        file = Path(path)
        data = json.loads(file.read_text())
        data["cells"][0]["requests"] = [
            {"sent": 10.0, "arrivals_s": [1, 1 + span], "usage": {"completion_tokens": 512}}
        ]
        file.write_text(json.dumps(data))
    result = compare(mb, tmp_path, ref, cand, aggregate=True)
    aggregate = result["aggregate"]["decode_agg_tps"]
    assert aggregate["geomean_ratio"] == pytest.approx(2.0) and aggregate["df"] == 0
    assert result["coverage"]["missing_aggregate_decode"] == []
    assert result["verdict"].startswith("uncertain")


def test_aggregate_comparison_refuses_partial_cohorts_and_disjoint_metric_repeats(mb, tmp_path):
    values = {(2, 4096): [(5000, 230), (5100, 240)]}
    ref = arm(tmp_path, mb, "ref.json", values)
    cand = arm(tmp_path, mb, "cand.json", values)
    complete = [{"sent": 10.0, "arrivals_s": [1, 2], "usage": {"completion_tokens": 512}}] * 2
    for path, valid_repeat in ((ref, 0), (cand, 1)):
        file = Path(path)
        data = json.loads(file.read_text())
        for cell in data["cells"]:
            cell["requests"] = complete if cell["rep"] == valid_repeat else complete[:1]
        file.write_text(json.dumps(data))
    result = compare(mb, tmp_path, ref, cand, aggregate=True)
    row, = result["rows"]
    assert row["decode_median_tps"]["repeats"] == [0, 1]
    assert row["decode_agg_tps"] is None and "decode_agg_tps" not in result["aggregate"]
    assert result["coverage"]["missing_aggregate_decode"] == ["v1/greedy/sync/c2/l4096"]
    assert result["verdict"].startswith("incomplete")


BASE = {(1, 4096): [(5000, 230), (5010, 200), (4990, 250)], (1, 65536): [(3800, 180), (3900, 170), (3850, 190)]}


def sessions(tmp_path, mb, prefix, values, shifts):
    """One result file per server session; session i scales every value by shifts[i] (between-process shift)."""
    return ",".join(arm(tmp_path, mb, f"{prefix}{i}.json", {key: [(p * f, d * f) for p, d in reps]
                                                              for key, reps in values.items()}, f"{prefix}{i}")
                    for i, f in enumerate(shifts))


def scaled(values, prefill, decode, cell=None):
    return {key: [(p * prefill, d * decode) for p, d in reps] if cell in (None, key) else reps
            for key, reps in values.items()}


def test_one_regressed_cell_rejects_despite_aggregate_gain(mb, tmp_path):
    ref = sessions(tmp_path, mb, "ref", BASE, (1.0, 1.002, 0.999))
    better = scaled(scaled(BASE, 1.2, 1.2), 0.98 / 1.2, 1.0, (1, 65536))
    result = compare(mb, tmp_path, ref, sessions(tmp_path, mb, "cand", better, (1.001, 0.999, 1.0)))
    assert result["verdict"].startswith("regression")
    assert result["aggregate"]["prefill_agg_tps"]["geomean_ratio"] > 1
    row = next(r for r in result["rows"] if r["cell"].endswith("l65536"))
    assert row["prefill_agg_tps"]["design"] == "session" and row["prefill_agg_tps"]["class"] == "regression"
    assert row["prefill_agg_tps"]["sessions"] == [3, 3] and row["prefill_agg_tps"]["df"] == 4


def test_session_is_the_unit_and_missing_cells_are_incomplete(mb, tmp_path):
    ref = sessions(tmp_path, mb, "ref", BASE, (1.0, 1.002))
    gain = scaled(BASE, 1.01, 1.01)
    assert compare(mb, tmp_path, ref, sessions(tmp_path, mb, "cand", gain, (1.0, 0.999)))["verdict"].startswith(
        "superior")
    # One session per arm without calibration: repeat noise alone cannot bound process-to-process variation.
    single = compare(mb, tmp_path, arm(tmp_path, mb, "r.json", BASE, "x"), arm(tmp_path, mb, "c.json", gain, "y"))
    assert single["verdict"].startswith("uncertain") and "no replicated server sessions" in single["verdict"]
    # The same single sessions with an A/A calibration group become decidable.
    calibrated = compare(mb, tmp_path, arm(tmp_path, mb, "r.json", BASE, "x"), arm(tmp_path, mb, "c.json", gain, "y"),
                         sessions(tmp_path, mb, "aa", BASE, (1.0, 1.001, 0.999)))
    assert calibrated["verdict"].startswith("superior") and calibrated["aggregate"]["decode_median_tps"]["df"] == 2
    partial = sessions(tmp_path, mb, "part", {(1, 4096): gain[(1, 4096)]}, (1.0, 0.999))
    result = compare(mb, tmp_path, ref, partial)
    assert result["verdict"].startswith("incomplete")
    assert result["coverage"]["missing_in_candidate"] == ["v1/greedy/sync/c1/l65536"]


def test_files_of_one_server_process_merge_and_a_session_cannot_be_reused(mb, tmp_path):
    first = {key: reps[:2] for key, reps in BASE.items()}
    arm_files = arm(tmp_path, mb, "a.json", first, "same") + "," + arm(tmp_path, mb, "b.json", first, "same")
    with pytest.raises(ValueError, match="duplicate repeat"):
        mb.load_arm(arm_files)
    assert len(mb.load_arm(arm(tmp_path, mb, "a.json", BASE, "one") + "," +
                           arm(tmp_path, mb, "b.json", BASE, "two"))["sessions"]) == 2
    ref = sessions(tmp_path, mb, "ref", BASE, (1.0, 1.001))
    with pytest.raises(ValueError, match="appears in two"):
        compare(mb, tmp_path, ref, sessions(tmp_path, mb, "cand", BASE, (1.0, 1.0)), ref)


def test_cross_process_aa_is_not_a_regression(mb, tmp_path):
    """Regression test: two server processes of one identical Gemma 4 SM70 configuration (rows-5 sessions s and r)
    differ by about 1% in prefill. Repeat-paired intervals inside one process called that a familywise regression
    (aggregate 0.989, 95% 0.981-0.997); with the session as the unit and the other identical-configuration session
    groups as calibration it is uncertain, while the real decode gain of 4 draft rows over 5 stays established."""
    data = json.loads((Path(__file__).parent / "fixtures" / "matrix_draft_rows_sessions.json").read_text())

    def files(*names):
        return ",".join(arm(tmp_path, mb, name + ".json",
                            {(1, int(length)): [tuple(r) for r in reps]
                             for length, reps in data["sessions"][name]["cells"].items()},
                            data["sessions"][name]["instance_id"], data["plan_sha256"]) for name in names)

    s, r = data["sessions"]["s"]["cells"], data["sessions"]["r"]["cells"]
    # The former repeat-paired aggregate on this pair excluded 1: the fixture still holds the false positive.
    diffs = [[math.log(b[0] / a[0]) for a, b in zip(s[length], r[length])] for length in s]
    mean = statistics.fmean(statistics.fmean(d) for d in diffs)
    se = math.sqrt(sum(statistics.variance(d) / len(d) for d in diffs)) / len(diffs)
    assert math.exp(mean + mb.t_half(0.95, 20) * se) < 1

    result = compare(mb, tmp_path, files("s"), files("r"), files("baseline", "aa"), files("rows4-s", "rows4-r"),
                     files("rows3-s", "rows3-r"))
    prefill = result["aggregate"]["prefill_agg_tps"]
    assert 0.985 < prefill["geomean_ratio"] < 0.993 and prefill["ci95"][0] < 1 < prefill["ci95"][1]
    assert prefill["df"] == 3 and prefill["regressions"] == 0 and result["verdict"].startswith("uncertain")

    gain = compare(mb, tmp_path, files("s", "r"), files("rows4-s", "rows4-r"), files("baseline", "aa"),
                   files("rows3-s", "rows3-r"))
    assert gain["aggregate"]["decode_median_tps"]["ci95"][0] > 1
    assert gain["aggregate"]["decode_median_tps"]["gains"] == 5

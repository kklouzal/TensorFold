"""Serving-matrix benchmark contracts against a local fake OpenAI server; no GPU or model is needed."""
import hashlib
import importlib.util
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import ClassVar

import pytest

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


def arm(tmp_path, mb, name, values):
    """values: {(c, length): [(prefill, decode) per repeat]}"""
    cells = []
    for (c, length), reps in values.items():
        for rep, (prefill, decode) in enumerate(reps):
            cells.append({"mode": "greedy", "arrival": "sync", "c": c, "length": length, "rep": rep,
                          "status": "measured", "problems": [],
                          "metrics": {"prefill_agg_tps": prefill, "decode_median_tps": decode}})
    path = tmp_path / name
    path.write_text(json.dumps({"schema": mb.SCHEMA, "label": name, "plan_sha256": "p", "output_tokens": 512,
                                "cells": cells, "gated": []}))
    return str(path)


def compare(mb, tmp_path, ref, cand):
    out = tmp_path / "cmp.json"
    old = sys.argv
    sys.argv = ["matrix_benchmark.py", "compare", ref, cand, "--json", str(out)]
    try:
        assert mb.main() == 0
    finally:
        sys.argv = old
    return json.loads(out.read_text())


BASE = {(1, 4096): [(5000, 230), (5010, 200), (4990, 250)], (1, 65536): [(3800, 180), (3900, 170), (3850, 190)]}


def test_one_regressed_cell_rejects_despite_aggregate_gain(mb, tmp_path):
    ref = arm(tmp_path, mb, "ref.json", BASE)
    better = {key: [(p * 1.2, d * 1.2) for p, d in reps] for key, reps in BASE.items()}
    better[(1, 65536)] = [(p * 0.98, d * 1.2) for p, d in BASE[(1, 65536)]]
    result = compare(mb, tmp_path, ref, arm(tmp_path, mb, "cand.json", better))
    assert result["verdict"].startswith("regression")
    assert result["aggregate"]["prefill_agg_tps"]["geomean_ratio"] > 1
    row = next(r for r in result["rows"] if r["cell"].endswith("l65536"))
    assert row["prefill_agg_tps"]["design"] == "paired" and row["prefill_agg_tps"]["class"] == "regression"


def test_paired_repeats_resolve_content_noise_and_missing_cells_are_incomplete(mb, tmp_path):
    ref = arm(tmp_path, mb, "ref.json", BASE)
    gain = {key: [(p * 1.01, d * 1.01) for p, d in reps] for key, reps in BASE.items()}
    assert compare(mb, tmp_path, ref, arm(tmp_path, mb, "cand.json", gain))["verdict"].startswith("superior")
    partial = {(1, 4096): gain[(1, 4096)]}
    result = compare(mb, tmp_path, ref, arm(tmp_path, mb, "part.json", partial))
    assert result["verdict"].startswith("incomplete")
    assert result["coverage"]["missing_in_candidate"] == ["v1/greedy/sync/c1/l65536"]

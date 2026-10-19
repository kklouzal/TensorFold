"""Campaign measurement tools: interval and ratio statistics, run records over a loopback server, the tool-call schema
checks, needle placement and the teacher-forced KL bound. No model or GPU."""

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import itertools
import json
import math
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def load(name):
    spec = importlib.util.spec_from_file_location("campaign_" + name, ROOT / "tools" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "tools"))
    try:
        spec.loader.exec_module(module)
    finally:
        sys.path.pop(0)
    return module


bench = load("campaign_bench")
gates = load("campaign_gates")
kl = load("campaign_kl")
trace = load("campaign_trace")


# --- statistics -----------------------------------------------------------------------------------------------------

def test_t_quantiles_match_the_table_and_the_large_sample_limit():
    assert bench.t95(1) == 12.706 and bench.t95(9) == 2.262 and bench.t95(30) == 2.042
    assert abs(bench.t95(60) - 2.000) < 1e-3 and abs(bench.t95(120) - 1.980) < 1e-3
    with pytest.raises(ValueError):
        bench.t95(0)


def test_interval_is_the_two_sided_t_half_width_and_absent_for_one_unit():
    s = bench.interval([1.0, 2.0, 3.0])
    assert s["mean"] == 2.0 and math.isclose(s["half95"], 4.303 * 1.0 / math.sqrt(3))
    assert bench.interval([5.0])["half95"] is None


def _record(prompt, step, sampling, seed, tps, prefill_s, ids):
    n = len(ids)
    return {"prompt": prompt, "depth": prompt.split("-")[0], "task": "t", "thinking": False, "step": step,
            "sampling": sampling, "seed": seed, "draft": True, "phase": "cold" if step == 0 else "warm",
            "expected_prompt_tokens": 100, "usage": {"prompt_tokens": 100, "completion_tokens": n},
            "stats": {"token_sha": str(hash(tuple(ids)))}, "token_ids": ids, "mem_min_available_mib": 1000,
            "metrics": {"decode_tps": tps, "decode_tps_wall": tps, "accept": 0.5, "tokens_per_round": 2.0,
                        "queue_s": 0.0, "prefill_tps": 100 / prefill_s if step == 0 else None,
                        "ttft_s": prefill_s if step == 0 else None, "ttft_client_s": None, "cached": 0}}


def test_prefill_cells_take_only_cold_requests_that_were_mostly_uncached():
    fresh = _record("4k-x", 0, "greedy", None, 50.0, 2.0, [1, 2])
    resumed = _record("4k-y", 0, "greedy", None, 50.0, 0.1, [1, 2])
    resumed["metrics"]["cached"] = 97              # the other thinking mode's prompt-end state covered 97 of 100
    cells = bench.cell_values({"records": [fresh, resumed]})
    assert cells[("prefill", "4k", "nothink")]["prefill_tps"] == 50.0
    assert cells[("decode", "4k", "nothink", "greedy")]["decode_tps"] == 50.0


def test_abba_ratios_cancel_linear_drift_and_identity_reports_divergence(tmp_path):
    # base decodes at 50 + drift, cand at 55 + drift (ratio 1.1 at zero drift); positions 1..4 per cycle ABBA
    for cycle in (1, 2, 3):
        for position, variant in enumerate("ABBA", start=1):
            drift = (cycle - 1) * 4 + position
            tps = (50.0 if variant == "A" else 55.0) + drift
            ids = [1, 2, 3] if variant == "A" or cycle < 3 else [1, 9, 3]
            doc = {"records": [_record("4k-x", 0, "greedy", None, tps, 2.0, ids),
                               _record("4k-x", 1, "card", 1001, tps, 2.0, [4, 5])],
                   "profile": {"max_tokens": 3}, "tool_sha256": "a" * 64, "host_after": {}}
            run = tmp_path / f"ab-c{cycle}-{position}-{'base' if variant == 'A' else 'cand'}"
            run.mkdir()
            (run / "bench.json").write_text(json.dumps(doc))
    result = bench.analyze(str(tmp_path), "ab", "base")
    ratio = result["ratios"]["decode/4k/nothink/greedy"]["cand"]["decode_tps"]
    expected = [(57.5 + 4 * c) / (52.5 + 4 * c) for c in (0, 1, 2)]     # equal mean positions within each cycle
    assert ratio["n"] == 3 and all(math.isclose(a, b, rel_tol=1e-4) for a, b in zip(ratio["ratios"], expected))
    absolute = result["absolute"]["prefill/4k/nothink"]["base"]["prefill_tps"]
    assert absolute["n"] == 6 and absolute["mean"] == 50.0
    ident = result["identity"]
    assert ident["keys"] == 2 and ident["identical"] == 1
    assert {d["first_divergence"] for d in ident["differing"][0]["against_first"]} == {None, 1}
    assert "decode tok/s" in bench.markdown(result)
    # the card request decoded 2 of the profile's 3 tokens in every run; every run finished; one tool version
    assert len(result["replies_off_schedule"]) == 12 and result["incomplete_runs"] == []
    assert result["tool_versions"] == ["a" * 64]
    with pytest.raises(ValueError, match="base variant"):
        bench.analyze(str(tmp_path), "ab", "typo")


def test_analysis_lists_incomplete_runs_and_mixed_tool_versions(tmp_path):
    for position, (sha, finished) in enumerate((("a" * 64, True), ("b" * 64, False)), start=1):
        doc = {"records": [_record("4k-x", 0, "greedy", None, 50.0, 2.0, [1, 2])], "tool_sha256": sha}
        if finished:
            doc["host_after"] = {}
        if position == 2:
            doc["held"] = ["32k-x"]                # a prompt the long-prefill hold kept out of this run
        run = tmp_path / f"s-c1-{position}-base"
        run.mkdir()
        (run / "bench.json").write_text(json.dumps(doc))
    result = bench.analyze(str(tmp_path), "s", None)
    assert result["incomplete_runs"] == ["c1-2-base"] and len(result["tool_versions"]) == 2
    assert result["held"] == ["32k-x"]
    text = bench.markdown(result)
    assert "Incomplete runs" in text and "2 campaign_bench.py versions" in text
    assert "long-prefill hold (not measured): 32k-x" in text


def test_size_prompt_lands_within_tolerance_below_the_target():
    text = "word " * 200000

    def count(messages):          # a deterministic stand-in tokenizer: 1 token per 5 characters plus a template
        return 20 + len(messages[0]["content"]) // 5

    messages, n = bench.size_prompt(count, "4k", "doc-review", text, 4096)
    assert 4096 - 32 <= n <= 4096 and count(messages) == n


# --- one position over a loopback server ---------------------------------------------------------------------------

@contextlib.contextmanager
def fake_server(seen=None, ids=True, cache=True):
    """A streaming chat/completions server whose reply ids depend only on max_tokens (omitted unless ``ids``);
    ``seen`` collects the request bodies. Like Gemma4's one-entry cache (with ``cache``), a prompt that starts with
    the previous prompt resumes all of it. /v1/tokenize counts 20 template tokens plus one per 4 characters of the
    first message and returns fake_ids() as the tokens."""

    last = [None]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def _json(self, value):
            body = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._json({"server": {"source_revision": "test"}})

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            if self.path.endswith("/v1/tokenize") and request.get("return_tokens"):
                # a production image's Harness bridge: ids only up to 16,384 tokens; the gates must not depend on it
                self.send_response(400)
                self.end_headers()
                return
            if self.path.endswith("/tokenize"):        # /v1/tokenize (counts) and the native /tokenize (ids)
                return self._json({"count": count_tokens(request["messages"]), "tokens": fake_ids(request["messages"])})
            if seen is not None:
                seen.append(request)
            prompt = request["prompt"] if "prompt" in request else fake_ids(request["messages"])
            cached = len(last[0]) if cache and last[0] is not None and prompt[:len(last[0])] == last[0] else 0
            last[0] = prompt
            n = request["max_tokens"]
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(n):
                piece = {"text": f"t{i} "} if "prompt" in request else {"delta": {"content": f"t{i} "}}
                self.wfile.write(b"data: " + json.dumps({"choices": [piece]}).encode() + b"\n\n")
            stats = {"stats_version": 2, "prefill_s": 0.5, "decode_s": 1.0, "decode_tps": (n - 1) / 1.0,
                     "ttft_s": 0.6, "queue_s": 0.0, "cached": cached, "rounds": n // 2, "drafted": n,
                     "accepted": n // 2, "token_sha": "abc", **({"token_ids": list(range(10, 10 + n))} if ids else {})}
            usage = {"prompt_tokens": len(prompt), "completion_tokens": n}
            self.wfile.write(b"data: " + json.dumps({"choices": [{"delta": {}, "finish_reason": "length"}],
                                                     "tensorfold": stats}).encode() + b"\n\n")
            self.wfile.write(b"data: " + json.dumps({"choices": [], "usage": usage, "tensorfold": stats}).encode()
                             + b"\n\ndata: [DONE]\n\n")

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def fake_ids(messages):
    """A chat prompt's fake token ids: one per character of the joined messages plus a generation-prompt token, so a
    next chat turn is not an extension of the previous prompt (as with Gemma4's template without thinking)."""

    return [100 + ord(c) % 50 for c in "\n".join(m["content"] for m in messages)] + [1]


def count_tokens(messages):
    return 20 + len(messages[0]["content"]) // 4


PROFILE = {"model": "m", "context": 1000, "max_tokens": 8, "eos_token_ids": [12], "warmup_tokens": 4,
           "depths": {"short": 0}, "workloads": [["short", "code"]], "thinking": [False],
           "card_sampling": {"nothink": {"temperature": 1.0, "top_k": 20}},
           "schedule": [{"sampling": "greedy"}, {"sampling": "card", "seed": 5}]}
PROMPTS = {"items": [{"id": "short-code-nothink", "depth": "short", "task": "code", "thinking": False,
                      "messages": [{"role": "user", "content": "x"}], "prompt_tokens": 7}]}


def _write_inputs(tmp_path):
    (tmp_path / "profile.json").write_text(json.dumps(PROFILE))
    (tmp_path / "prompts.json").write_text(json.dumps(PROMPTS))


def test_run_records_every_scheduled_request_with_its_metrics(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    seen = []
    phases = []
    original = bench.Progress.update

    def recorded(phase, **fields):
        phases.append(phase)
        original(phase, **fields)

    with fake_server(seen) as url:
        bench.Progress.configure(str(tmp_path / "progress.json"))
        monkeypatch.setattr(bench.Progress, "update", recorded)
        try:
            bench.run(url, str(tmp_path / "profile.json"), str(tmp_path / "prompts.json"), str(tmp_path / "out.json"),
                      meta=None, label="t", select={"depths": [], "tasks": [], "thinking": None})
        finally:
            bench.Progress.configure(None)
    beat = json.loads((tmp_path / "progress.json").read_text())
    # the watchdog heartbeat: each request prefills, then decodes (the role-free text chunks), then is idle
    assert phases[:3] == ["prefill", "decode", "decode"] and phases.count("prefill") == 3 and beat["phase"] == "idle"
    assert bench.expected_tokens({"prompt": [1, 2, 3]}) == 3 and bench.expected_tokens({"prompt": "abcd"}) == 2
    common = {"model": "m", "messages": [{"role": "user", "content": "x"}], "max_tokens": 8,
              "chat_template_kwargs": {"enable_thinking": False}, "return_token_ids": True, "ignore_eos": True,
              "stream": True, "stream_options": {"include_usage": True}}
    assert seen[1:] == [{**common, "temperature": 0}, {**common, "temperature": 1.0, "top_k": 20, "seed": 5}]
    doc = json.loads((tmp_path / "out.json").read_text())
    assert [r["sampling"] for r in doc["records"]] == ["greedy", "card"] and doc["records"][1]["seed"] == 5
    first = doc["records"][0]
    assert first["phase"] == "cold" and first["eos_at"] == 2 and first["token_ids"][:3] == [10, 11, 12]
    m = first["metrics"]
    assert m["decode_tps"] == 7.0 and m["prefill_tps"] == 4.0 and m["accept"] == 0.5 and m["tokens_per_round"] == 1.75
    assert "token_ids" not in first["stats"] and doc["warmup"]["usage"]["completion_tokens"] == 4


# --- gates ---------------------------------------------------------------------------------------------------------

def test_determinism_and_identity_gates_over_a_loopback_server(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    seen = []
    results = {}
    with fake_server(seen) as url:
        for gate in ("determinism", "identity"):
            out = tmp_path / f"{gate}.json"
            monkeypatch.setattr(sys, "argv", ["campaign_gates.py", gate, "--url", url, "--out", str(out),
                                              "--profile", str(tmp_path / "profile.json"),
                                              "--prompts", str(tmp_path / "prompts.json"),
                                              "--gate-prompts", "short-code-nothink", "--max-tokens", "6"])
            assert gates.main() == 0
            results[gate] = json.loads(out.read_text())
            assert results[gate]["source_revision"] == "test" and len(results[gate]["profile_sha256"]) == 64
    determinism = results["determinism"]["gates"]["determinism"]
    identity = results["identity"]["gates"]["identity"]
    assert determinism["status"] == "pass" and determinism["verdict"] is True and determinism["cases"] == 2
    assert determinism["exercised"] == {"warm": 2, "recold": 2, "extended": 2} and determinism["identical"] == 2
    assert determinism["next_turn_exercised"] == 0                 # a next chat turn does not extend the prompt
    extension = [r for r in seen if "prompt" in r]
    assert len(extension) == 4 and extension[0]["prompt"][-6:] == list(range(10, 16))   # prompt + its own reply
    assert identity["status"] == "pass" and identity["greedy_cases"] == identity["greedy_identical"] == 1
    assert identity["greedy_drafted"] == 1
    measured = [r for r in seen if r["max_tokens"] == 6 and "messages" in r]
    assert all(r["return_token_ids"] and r["ignore_eos"] and r["chat_template_kwargs"] == {"enable_thinking": False}
               for r in measured)
    assert [r.get("draft") for r in measured[-4:]] == [None, False, None, False]
    assert measured[-1]["seed"] == 2024 and measured[-1]["temperature"] == 1.0


def test_unexercised_cache_paths_make_determinism_incomplete_not_pass(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    out = tmp_path / "determinism.json"
    with fake_server(cache=False) as url:
        monkeypatch.setattr(sys, "argv", ["campaign_gates.py", "determinism", "--url", url, "--out", str(out),
                                          "--profile", str(tmp_path / "profile.json"),
                                          "--prompts", str(tmp_path / "prompts.json"),
                                          "--gate-prompts", "short-code-nothink", "--max-tokens", "6"])
        assert gates.main() == 3
    result = json.loads(out.read_text())
    gate = result["gates"]["determinism"]
    assert result["status"] == gate["status"] == "incomplete" and gate["verdict"] is None
    assert gate["identical"] == 2 and gate["exercised"] == {"warm": 0, "recold": 2, "extended": 0}


def test_gates_without_cases_are_incomplete_not_pass(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    with fake_server() as url:                             # the only gate prompt is a thinking-off one
        for gate in ("determinism", "identity"):
            out = tmp_path / f"{gate}.json"
            monkeypatch.setattr(sys, "argv", ["campaign_gates.py", gate, "--url", url, "--out", str(out),
                                              "--profile", str(tmp_path / "profile.json"),
                                              "--prompts", str(tmp_path / "prompts.json"),
                                              "--gate-prompts", "short-code-nothink", "--thinking", "on"])
            assert gates.main() == 3
            result = json.loads(out.read_text())
            assert result["status"] == result["gates"][gate]["status"] == "incomplete"
            assert result["gates"][gate]["verdict"] is None


def test_gates_refuse_replies_without_token_ids(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    with fake_server(ids=False) as url:
        monkeypatch.setattr(sys, "argv", ["campaign_gates.py", "identity", "--url", url, "--out", str(tmp_path / "o"),
                                          "--profile", str(tmp_path / "profile.json"),
                                          "--prompts", str(tmp_path / "prompts.json"),
                                          "--gate-prompts", "short-code-nothink", "--max-tokens", "6"])
        assert gates.main() == 4                       # a refusal is an error result, never a fail or a pass
    result = json.loads((tmp_path / "o").read_text())
    assert result["status"] == "error" and "return_token_ids" in result["error"] and result["gates"] == {}
    monkeypatch.setattr(sys, "argv", ["campaign_gates.py", "identity", "--url", "http://127.0.0.1:9", "--out",
                                      str(tmp_path / "down"), "--profile", str(tmp_path / "profile.json"),
                                      "--prompts", str(tmp_path / "prompts.json"), "--gate-prompts", "x"])
    assert gates.main() == 4                           # no server: an error, not a verdict
    assert json.loads((tmp_path / "down").read_text())["status"] == "error"
    for profile in (tmp_path / "missing.json", tmp_path / "empty.json"):     # an unreadable or empty profile too
        (tmp_path / "empty.json").write_text("{}")
        monkeypatch.setattr(sys, "argv", ["campaign_gates.py", "identity", "--url", "http://127.0.0.1:9", "--out",
                                          str(tmp_path / "bad"), "--profile", str(profile),
                                          "--prompts", str(tmp_path / "prompts.json"), "--gate-prompts", "x"])
        assert gates.main() == 4
        assert json.loads((tmp_path / "bad").read_text())["status"] == "error"


def test_full_depth_needle_leaves_room_for_the_thinking_reply(tmp_path):
    import argparse

    (tmp_path / "corpus.txt").write_text("".join(f"line {i} of the haystack\n" for i in range(4000)))
    seen = []
    args = argparse.Namespace(corpus=str(tmp_path / "corpus.txt"), seed=1, needle_fractions="0.2,0.8",
                              needle_depths="full", needle_tokens=64, needle_thinking="on", min_needle_recall=None)
    with fake_server(seen) as url:
        result = gates.gate_needle(url, {**PROFILE, "context": 4000, "full_margin": 16}, args)
    (body,) = seen
    assert body["max_tokens"] == 64 * 8 and result["rows"][0]["target"] == 4000 - 64 * 8 - 16
    assert count_tokens(body["messages"]) + body["max_tokens"] <= 4000


LONG = {"id": "long-code-nothink", "depth": "32k", "task": "doc-code", "thinking": False,
        "messages": [{"role": "user", "content": "y" * 40}], "prompt_tokens": bench.LONG_PREFILL_TOKENS + 1}


def test_long_prefill_hold_keeps_long_prompts_out_and_reports_them(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    (tmp_path / "prompts.json").write_text(json.dumps({"items": [*PROMPTS["items"], LONG]}))
    for allow in (False, True):
        seen = []
        with fake_server(seen) as url:
            bench.run(url, str(tmp_path / "profile.json"), str(tmp_path / "prompts.json"), str(tmp_path / "out.json"),
                      meta=None, label="t", select={"depths": [], "tasks": [], "thinking": None, "allow_long": allow})
        doc = json.loads((tmp_path / "out.json").read_text())
        asked = {r["messages"][0]["content"] for r in seen if "messages" in r}
        assert doc["held"] == ([] if allow else ["long-code-nothink"]) and ("y" * 40 in asked) == allow
    out = tmp_path / "determinism.json"
    with fake_server() as url:
        monkeypatch.setattr(sys, "argv", ["campaign_gates.py", "determinism", "--url", url, "--out", str(out),
                                          "--profile", str(tmp_path / "profile.json"),
                                          "--prompts", str(tmp_path / "prompts.json"), "--max-tokens", "6",
                                          "--gate-prompts", "short-code-nothink,long-code-nothink"])
        assert gates.main() == 3                       # the short prompt passes; the held one is coverage not run
    gate = json.loads(out.read_text())["gates"]["determinism"]
    assert gate["status"] == "incomplete" and gate["verdict"] is None and gate["cases"] == 2   # greedy + card
    assert gate["held"] == [{"prompt": "long-code-nothink", "prompt_tokens": bench.LONG_PREFILL_TOKENS + 1,
                             "longest_prompt_tokens": bench.LONG_PREFILL_TOKENS + 7, "status": "incomplete (hold)"}]


def test_determinism_holds_a_prompt_whose_extension_or_next_turn_exceeds_the_limit(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    monkeypatch.setattr(gates, "LONG_PREFILL_TOKENS", 30)          # the limit scaled to the fake server's prompts
    out = tmp_path / "determinism.json"

    def determinism(max_tokens):
        seen = []
        with fake_server(seen) as url:
            monkeypatch.setattr(sys, "argv", ["campaign_gates.py", "determinism", "--url", url, "--out", str(out),
                                              "--profile", str(tmp_path / "profile.json"),
                                              "--prompts", str(tmp_path / "prompts.json"),
                                              "--gate-prompts", "short-code-nothink", "--max-tokens", str(max_tokens)])
            code = gates.main()
        return code, json.loads(out.read_text())["gates"]["determinism"], seen

    # 7 prompt tokens plus a 24-token reply, re-sent as one prompt (the extension), exceed 30: held before any request
    code, gate, seen = determinism(24)
    assert (code, gate["cases"], seen) == (3, 0, [])
    assert gate["held"] == [{"prompt": "short-code-nothink", "prompt_tokens": 7, "longest_prompt_tokens": 31,
                             "status": "incomplete (hold)"}]
    # 7 + 6 fit, but the next chat turn (the reply's text and "Continue.") renders 31 ids: held, never sent
    code, gate, seen = determinism(6)
    assert code == 3 and gate["status"] == "incomplete" and gate["cases"] == gate["identical"] == 2
    assert gate["exercised"] == {"warm": 2, "recold": 2, "extended": 2}
    assert gate["held"] == [{"prompt": "short-code-nothink", "sampling": mode, "next_turn_prompt_tokens": 31,
                             "status": "incomplete (hold)"} for mode in ("greedy", "card")]
    assert not any("Continue." in json.dumps(r.get("messages")) for r in seen)
    assert all(r["cached"]["next_turn"] is None and not r["next_turn_exercised"] for r in gate["rows"])


def test_needle_holds_depths_beyond_the_long_prefill_limit(tmp_path):
    import argparse

    (tmp_path / "corpus.txt").write_text("".join(f"line {i} of the haystack\n" for i in range(4000)))
    args = argparse.Namespace(corpus=str(tmp_path / "corpus.txt"), seed=1, needle_fractions="0.5",
                              needle_depths=f"512,{bench.LONG_PREFILL_TOKENS + 1}", needle_tokens=16,
                              needle_thinking="off", min_needle_recall=0.0, allow_long_prefill=False)
    with fake_server() as url:
        result = gates.gate_needle(url, {**PROFILE, "context": 40000}, args)
    assert [r["target"] for r in result["rows"]] == [512] and result["status"] == "incomplete"
    assert result["held"] == [{"depth": str(bench.LONG_PREFILL_TOKENS + 1), "target": bench.LONG_PREFILL_TOKENS + 1,
                               "thinking": False, "status": "incomplete (hold)"}] and result["verdict"] is None


def test_kl_dump_refuses_held_items_before_the_engine_loads(tmp_path, monkeypatch):
    loads = []
    monkeypatch.setattr(kl, "with_app", lambda serve, port, work: loads.append(serve))
    items = [{"name": "short", "prompt": [1] * 10, "answer": [2]},
             {"name": "long", "prompt": [1] * (bench.LONG_PREFILL_TOKENS + 1), "answer": [2]}]
    (tmp_path / "eval.json").write_text(json.dumps({"version": 1, "items": items}))
    (tmp_path / "corpus.txt").write_text("text")
    serve = ["--", "serve", "m"]

    def dump(*flags):
        monkeypatch.setattr(sys, "argv", ["campaign_kl.py", "dump", "--out", str(tmp_path / "x.npz"), *flags, *serve])
        return kl.main()

    with pytest.raises(SystemExit, match=r"eval items \['long'\] exceed .*long-prefill hold"):
        dump("--eval", str(tmp_path / "eval.json"))
    missing = str(tmp_path / "new-eval.json")
    with pytest.raises(SystemExit, match="long-prefill hold"):           # --build would write a 128k item
        dump("--eval", missing, "--build", "--corpus", str(tmp_path / "corpus.txt"), "--long-depths", "131072")
    with pytest.raises(SystemExit, match="does not exist"):
        dump("--eval", missing)
    with pytest.raises(SystemExit, match="long-prefill hold"):           # a mid item above the limit
        dump("--eval", missing, "--build", "--corpus", str(tmp_path / "corpus.txt"), "--long-depths", "",
             "--mid-depths", f"4096,{bench.LONG_PREFILL_TOKENS + 1}")
    with pytest.raises(SystemExit, match="needs --corpus"):
        dump("--eval", missing, "--build", "--mid-depths", "4096")
    with pytest.raises(SystemExit, match=r"unknown --short sets \['extras'\]"):    # a typo builds nothing silently
        dump("--eval", missing, "--build", "--short", "v1,extras")
    assert loads == []                                                    # no refusal paid for a model load
    dump("--eval", str(tmp_path / "eval.json"), "--allow-long-prefill")
    dump("--eval", missing, "--build", "--corpus", str(tmp_path / "corpus.txt"), "--long-depths", "8192")
    assert loads == [["serve", "m"]] * 2
    dump("--eval", str(tmp_path / "eval.json"), "--allow-long-prefill", "--card", "{}", "--profile",
         str(tmp_path / "no-profile.json"))                     # --card overrides: the profile is not read
    assert len(loads) == 3


class FakeApp:
    """campaign_kl.build_eval's view of the App: a template of 10 tokens (12 with thinking) plus one per 4 characters
    of the user message, and an engine that samples ``answer_tokens`` ids recording each request's sampling."""

    def __init__(self):
        self.sampled = []
        self.engine = self

    def prepare(self, body, chat):
        content = body["messages"][0]["content"]
        thinking = body["chat_template_kwargs"]["enable_thinking"]
        return SimpleNamespace(prompt=[7] * (10 + 2 * thinking + len(content) // 4), sampling=dict(body))

    def generate(self, prompt, n, sampling, callback):
        self.sampled.append(sampling)
        callback(list(range(n + 3)))                   # more than asked: the eval keeps answer_tokens


def _build_args(tmp_path, **overrides):
    (tmp_path / "corpus.txt").write_text("".join(f"line {i:06d} of the corpus\n" for i in range(20000)))
    fields = {"corpus": str(tmp_path / "corpus.txt"), "short": "v1", "mid_depths": "", "long_depths": "",
              "corpus_start": 0, "allow_long_prefill": False, "answer_tokens": 5, "serve": ["serve", "/m"],
              "cards": {"nothink": {"temperature": 0.7}, "think": {"temperature": 1.0}}}
    return SimpleNamespace(**{**fields, **overrides})


def test_build_eval_mid_items_fill_their_depths_on_disjoint_windows_in_both_thinking_modes(tmp_path):
    app = FakeApp()
    args = _build_args(tmp_path, short="", mid_depths="400,800,1200,1600", corpus_start=1000)
    data = kl.build_eval(app, args)
    names = [it["name"] for it in data["items"]]
    assert names == ["mid-400-doc-review-nothink", "mid-800-doc-code-think", "mid-1200-doc-review-think",
                     "mid-1600-doc-code-nothink"]
    windows = [it["document"] for it in data["items"]]
    assert windows[0][0] == 1000 and all(a[1] == b[0] for a, b in itertools.pairwise(windows))
    corpus = (tmp_path / "corpus.txt").read_text()
    for it, depth in zip(data["items"], (400, 800, 1200, 1600), strict=True):
        start, end = it["document"]
        assert corpus[start:end] in it["messages"][0]["content"]
        assert depth - 64 // 4 - 1 <= len(it["prompt"]) <= depth       # the bisection resolves 64 characters
        thinking = it["name"].endswith("-think")
        assert it["sampling"]["chat_template_kwargs"] == {"enable_thinking": thinking}
        assert it["sampling"]["temperature"] == (1.0 if thinking else 0.7) and len(it["answer"]) == 5
        assert "thinking" not in it
    assert data["built_by"]["corpus_sha256"] == kl.file_sha256(str(tmp_path / "corpus.txt"))
    assert [s["seed"] for s in app.sampled] == [7000, 7001, 7002, 7003]


def test_build_eval_keeps_the_v1_short_and_long_items(tmp_path):
    data = kl.build_eval(FakeApp(), _build_args(tmp_path, long_depths="900"))
    assert [it["name"] for it in data["items"]] == [f"short-{i}" for i in range(len(kl.PROMPTS))] + ["long-900"]
    long = data["items"][-1]
    corpus = (tmp_path / "corpus.txt").read_text()
    assert long["messages"][0]["content"] == ("<document>\n" + corpus[:long["document"][1]] + "\n</document>\n\n"
                                              + kl.LONG_ASK)
    assert all(it["sampling"]["chat_template_kwargs"] == {"enable_thinking": False} for it in data["items"])
    # eval-v2's extra prompts alone (to append): numbered after v1's, each in its own thinking mode
    extra = kl.build_eval(FakeApp(), _build_args(tmp_path, short="extra"))["items"]
    first = len(kl.PROMPTS)
    assert [it["name"] for it in extra] == [f"short-{first + i}" for i in range(len(kl.EXTRA_PROMPTS))]
    assert [it["sampling"]["chat_template_kwargs"]["enable_thinking"] for it in extra] == [
        thinking for _, thinking in kl.EXTRA_PROMPTS]
    both = kl.build_eval(FakeApp(), _build_args(tmp_path, short="v1,extra"))["items"]
    assert [it["name"] for it in both] == [f"short-{i}" for i in range(first + len(kl.EXTRA_PROMPTS))]


def _golden_part(tmp_path, name, items, lp, **meta):
    np = pytest.importorskip("numpy")
    path = tmp_path / name
    path.mkdir()
    (path / "eval.json").write_text(json.dumps({"version": 1, "items": items}))
    sizes = [len(it["answer"]) for it in items]
    offsets = tuple(int(x) for x in np.concatenate([[0], np.cumsum(sizes)[:-1]]))
    contract = {"source": "s@1", "architecture": "A", "config_sha256": "c", "versions": {"torch": "t"},
                "chunk": 2048, "precision": {"weights": "bf16", "compute": "fp32"}, "model_dir": "/golden"}
    _dump(path / "golden.npz", lp, 8, golden=True, offsets=offsets, names=[it["name"] for it in items],
          eval_sha256=kl.file_sha256(str(path / "eval.json")), **{**contract, **meta})
    return str(path / "golden.npz"), str(path / "eval.json")


def test_append_extends_a_golden_reference_without_rescoring_and_records_its_parts(tmp_path):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(9)
    lp = _log_softmax(np, rng.normal(0, 3, (40, 32)))
    old = [{"name": "short-0", "prompt": [1] * 10, "answer": [2] * 10},
           {"name": "long-32768", "prompt": [1] * 99, "answer": [2] * 6}]
    new = [{"name": "mid-400", "prompt": [1] * 40, "answer": [2] * 14},
           {"name": "mid-800", "prompt": [1] * 80, "answer": [2] * 10}]
    base_golden, base_eval = _golden_part(tmp_path, "base", old, lp[:16], tool_sha256="v1",
                                          precision={"weights": "bf16", "compute": "fp32", "tf32": False})
    new_golden, new_eval = _golden_part(tmp_path, "new", new, lp[16:], tool_sha256="v2")
    result = kl.append(base_golden, base_eval, new_golden, new_eval, str(tmp_path / "v2"))
    joined = np.load(tmp_path / "v2" / "golden" / "kl.npz")
    meta = json.loads(str(joined["meta"]))
    assert result["items"] == meta["names"] == ["short-0", "long-32768", "mid-400", "mid-800"]
    assert list(joined["offsets"]) == [0, 10, 16, 30] and result["positions"] == 40
    for key in ("ids", "lp", "tail_lp", "forced", "entropy"):         # rows copied, never recomputed
        assert np.array_equal(joined[key], np.concatenate([np.load(base_golden)[key], np.load(new_golden)[key]]))
    v2 = json.loads((tmp_path / "v2" / "eval.json").read_text())
    assert [it["name"] for it in v2["items"]] == meta["names"] and meta["golden"]
    assert meta["eval_sha256"] == kl.file_sha256(str(tmp_path / "v2" / "eval.json"))
    assert [p["tool_sha256"] for p in v2["parts"]] == ["v1", "v2"] and v2["parts"] == meta["parts"]
    assert v2["parts"][0]["golden_sha256"] == kl.file_sha256(base_golden) and "tool_sha256" not in meta
    assert "golden_dir" in json.loads((tmp_path / "v2" / "golden" / "meta.json").read_text())
    cut = kl.subset(str(tmp_path / "v2" / "golden" / "kl.npz"), str(tmp_path / "v2" / "eval.json"),
                    str(tmp_path / "v2s"), max_prompt_tokens=90)
    assert cut["items"] == ["short-0", "mid-400", "mid-800"] and cut["positions"] == 34
    again = kl.append(str(tmp_path / "v2" / "golden" / "kl.npz"), str(tmp_path / "v2" / "eval.json"),
                      *_golden_part(tmp_path, "later", [{"name": "long-131072", "prompt": [1] * 9, "answer": [2] * 4}],
                                    lp[:4], tool_sha256="v3"), str(tmp_path / "v3"))
    assert [p["items"] for p in again["parts"]] == [["short-0", "long-32768"], ["mid-400", "mid-800"],
                                                    ["long-131072"]]


def test_append_refuses_parts_that_are_not_one_reference(tmp_path):
    np = pytest.importorskip("numpy")
    lp = _log_softmax(np, np.random.default_rng(3).normal(0, 3, (8, 32)))
    item = {"name": "a", "prompt": [1], "answer": [2] * 4}
    other = {**item, "name": "b"}
    base = _golden_part(tmp_path, "base", [item], lp[:4])
    cases = {"other": (other, {"source": "s@2"}, "differ in \\['source'"),
             "compute": (other, {"precision": {"weights": "bf16", "compute": "bf16"}}, "precision.compute"),
             "same": (item, {}, "names repeat")}
    for name, (new_item, meta, message) in cases.items():
        new = _golden_part(tmp_path, name, [new_item], lp[4:], **meta)
        with pytest.raises(ValueError, match=message):
            kl.append(*base, *new, str(tmp_path / f"out-{name}"))
    np.savez(tmp_path / "bare.npz", **{k: v for k, v in np.load(base[0]).items() if k != "entropy"})
    with pytest.raises(ValueError, match="hold different arrays"):
        kl.append(str(tmp_path / "bare.npz"), base[1], *_golden_part(tmp_path, "c", [other], lp[4:]),
                  str(tmp_path / "out-arrays"))
    kl.subset(*base, str(tmp_path / "cut"), max_prompt_tokens=5)
    with pytest.raises(ValueError, match="subset"):
        kl.append(str(tmp_path / "cut" / "golden" / "kl.npz"), str(tmp_path / "cut" / "eval.json"),
                  *_golden_part(tmp_path, "b", [other], lp[4:]),
                  str(tmp_path / "out-subset"))


def test_schema_errors_cover_type_enum_required_extra_and_items():
    schema = gates.EVENT["function"]["parameters"]
    good = {"title": "t", "start": "2026-11-03T14:00", "duration_minutes": 45,
            "attendees": [{"name": "A", "email": "a@x"}]}
    assert gates.schema_errors(good, schema) == []
    bad = {"title": 1, "start": "s", "duration_minutes": 45.0, "attendees": [{"name": "A"}], "extra": True}
    errors = gates.schema_errors(bad, schema)
    assert any("title: expected string" in e for e in errors) and any("duration_minutes" in e for e in errors)
    assert any("missing email" in e for e in errors) and any("unexpected extra" in e for e in errors)
    assert gates.schema_errors("x", {"type": "string", "enum": ["y"]}) == ["$: 'x' not in ['y']"]


def test_tool_case_expectations_accept_the_intended_call_only():
    expect = gates.TOOL_CASES["weather"][3]
    assert expect([("get_weather", {"city": "Paris, France", "unit": "celsius"})], "")
    assert not expect([("get_weather", {"city": "Paris", "unit": "fahrenheit"})], "")
    assert gates.TOOL_CASES["no-call"][3]([], "Leaves fall") and not gates.TOOL_CASES["no-call"][3]([("x", {})], "")
    long_expect = gates.TOOL_CASES["long-context"][3]
    assert long_expect([("send_email", {"to": gates.DIRECTORY_EMAIL, "subject": "Welcome", "body": "Hi"})], "")
    assert gates.DIRECTORY_EMAIL in gates.DIRECTORY and gates.DIRECTORY_NAME in gates.DIRECTORY


def test_needle_prompt_places_every_code_once_in_order():
    text = "".join(f"line {i}\n" for i in range(5000))
    codes = {"AMBER": "111111", "BIRCH": "222222", "CEDAR": "333333"}
    content = gates.needle_prompt(text, len(text), [0.1, 0.5, 0.9], codes)[0]["content"]
    positions = [content.index(code) for code in codes.values()]
    assert positions == sorted(positions) and all(content.count(code) == 1 for code in codes.values())
    assert 0.05 < positions[0] / len(content) < 0.15 and 0.85 < positions[2] / len(content) < 0.95


# --- KL ------------------------------------------------------------------------------------------------------------

def _outside(np, lp_full, ids):
    """float32 log-sum-exp of each row outside ``ids``, as ``campaign_kl._tail``; -inf for an empty complement."""

    rest = lp_full.astype(np.float32).copy()
    np.put_along_axis(rest, ids, -np.inf, -1)
    top = rest.max(-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        lse = top + np.log(np.exp(rest - top).sum(-1, keepdims=True, dtype=np.float32))
    return np.where(np.isneginf(top), -np.inf, lse).ravel().astype(np.float64)


def _dump(path, lp_full, top_k, reference=None, golden=False, offsets=(0,), **meta_fields):
    """A dump of full log-probability rows (items starting at ``offsets``; ``meta_fields`` override the meta); with
    ``reference`` (a dump's path) a candidate dump against it."""

    np = pytest.importorskip("numpy")
    order = np.argsort(-lp_full, axis=-1)[:, :top_k]
    arrays = {"ids": order.astype(np.int32), "lp": np.take_along_axis(lp_full, order, -1).astype(np.float32),
              "tail_lp": _outside(np, lp_full, order), "forced": lp_full[:, 0],
              "entropy": -(np.exp(lp_full) * lp_full).sum(-1), "offsets": np.array(offsets)}
    meta = {"eval_sha256": "e", "top_k": top_k, "logits_dtype": "torch.float32", "names": ["all"], "golden": golden,
            **meta_fields}
    if reference is not None:
        ids = np.load(reference)["ids"].astype(np.int64)
        arrays["at_ref"] = np.take_along_axis(lp_full, ids, -1).astype(np.float32)
        arrays["at_ref_tail_lp"] = _outside(np, lp_full, ids)
        meta["reference_sha256"] = kl.file_sha256(str(reference))
    np.savez(path, meta=json.dumps(meta), **arrays)
    return order


def _log_softmax(np, x):
    x = x - x.max(-1, keepdims=True)
    return x - np.log(np.exp(x).sum(-1, keepdims=True))


def test_kl_is_exact_with_the_full_vocabulary_and_a_lower_bound_with_top_k(tmp_path):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(0)
    p = _log_softmax(np, rng.normal(0, 3, (50, 64)))
    q = _log_softmax(np, p + rng.normal(0, 0.3, p.shape))
    exact = (np.exp(p) * (p - q)).sum(-1).mean()
    for k, check in ((64, lambda got: math.isclose(got, exact, rel_tol=1e-3)),   # float32 storage
                     (8, lambda got: got <= exact + 1e-6)):
        _dump(tmp_path / f"ref{k}.npz", p, k)
        _dump(tmp_path / f"cand{k}.npz", q, k, reference=tmp_path / f"ref{k}.npz")
        result = kl.compare(str(tmp_path / f"ref{k}.npz"), str(tmp_path / f"cand{k}.npz"))
        assert check(result["kl"]["mean"]), (k, result["kl"]["mean"], exact)
        assert result["kl"]["kind"] == "lower bound" and not result["identical_logprobs"]
    _dump(tmp_path / "ref.npz", p, 8)
    same = _dump(tmp_path / "same.npz", p, 8, reference=tmp_path / "ref.npz")
    result = kl.compare(str(tmp_path / "ref.npz"), str(tmp_path / "same.npz"))
    assert result["kl"]["max"] < 1e-6 and result["top1_agreement"] == 1.0 and result["identical_logprobs"]
    assert same.shape == (50, 8)
    _dump(tmp_path / "other.npz", q, 8)
    with pytest.raises(ValueError, match="another reference"):          # same eval and shapes, other reference
        kl.compare(str(tmp_path / "other.npz"), str(tmp_path / "same.npz"))


def test_kl_tail_stays_a_bound_when_the_tail_is_below_float32_resolution(tmp_path):
    # confident rows whose mass outside the top-K (~1e-7) is the size of float32 rounding of a top-K sum: one minus
    # that sum overshot the exact KL by up to 7e-5 nats in 68 of these 400 rows; log-sum-exp tails stay within 2.5e-7
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    x = rng.normal(0, 1, (400, 4096))
    x[:, :4] += np.array([24.0, 23.0, 22.5, 22.0])
    y = x + rng.normal(0, 1e-3, x.shape)

    def log_softmax32(v):                      # the dump's float32 log-softmax
        v = v.astype(np.float32)
        m = v.max(-1, keepdims=True)
        return v - m - np.log(np.exp(v - m).sum(-1, keepdims=True, dtype=np.float32))

    exact = (np.exp(_log_softmax(np, x)) * (_log_softmax(np, x) - _log_softmax(np, y))).sum(-1)
    _dump(tmp_path / "ref.npz", log_softmax32(x), 64)
    _dump(tmp_path / "cand.npz", log_softmax32(y), 64, reference=tmp_path / "ref.npz")
    result = kl.compare(str(tmp_path / "ref.npz"), str(tmp_path / "cand.npz"))
    assert result["kl"]["max"] <= exact.max() + 5e-7 and abs(result["kl"]["mean"] - exact.mean()) < 1e-7
    assert 0 < result["tail_mass_ref"]["mean"] < 1e-6


def test_capture_records_forced_positions_and_refuses_uncovered_paths():
    torch = pytest.importorskip("torch")
    capture = kl.Capture(torch, 3)
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0], [3.0, 2.0, 1.0, 0.0]])
    capture.begin(10, [3, 0], reference_ids=[[3, 2, 1], [0, 1, 2]])
    assert capture(logits[:1], [10], None) == [3] and capture(logits[1:], [11], None) == [0]
    rows = capture.end()
    assert rows[0]["ids"].tolist() == [3, 2, 1] and math.isclose(rows[0]["forced"], float(rows[0]["lp"][0]))
    assert torch.allclose(rows[1]["at_ref"], rows[1]["lp"])
    full = torch.log_softmax(logits, -1)
    assert math.isclose(rows[0]["tail_lp"], float(full[0, 0]), rel_tol=1e-6)        # the one id outside the top 3
    assert math.isclose(rows[1]["at_ref_tail_lp"], float(full[1, 3]), rel_tol=1e-6)
    capture.begin(10, [3, 0])
    with pytest.raises(RuntimeError, match="uncovered engine path"):
        capture(logits, [11, 12], None)


def test_sampler_replaced_swaps_every_tensorfold_reference_and_restores_it(monkeypatch):
    import types

    def original(*args):
        return "original"

    def other(*args):
        return "other"

    sampling = types.ModuleType("tensorfold.cuda.sampling")
    sampling.sample_rows = original
    cuda = types.ModuleType("tensorfold.cuda")
    cuda.sampling = sampling
    root = types.ModuleType("tensorfold")
    root.cuda = cuda
    user, unrelated, outside = (types.ModuleType(name) for name in ("tensorfold.engine", "tensorfold.x", "elsewhere"))
    user.sample_rows, unrelated.sample_rows, outside.sample_rows = original, other, original
    for module in (root, cuda, sampling, user, unrelated, outside):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    with pytest.raises(KeyError):
        with kl.sampler_replaced("capture"):
            assert sampling.sample_rows == user.sample_rows == "capture"
            assert unrelated.sample_rows is other and outside.sample_rows is original
            raise KeyError("restored on failure too")
    assert sampling.sample_rows is user.sample_rows is outside.sample_rows is original
    assert unrelated.sample_rows is other


# --- timeline summary ----------------------------------------------------------------------------------------------

def test_trace_summary_counts_kernel_union_launches_and_syncs_per_round():
    def x(name, ts, dur=0, cat="cpu_op"):
        return {"ph": "X", "name": name, "ts": ts, "dur": dur, "cat": cat}

    events = [x("tf.sample", 0), x("tf.sample", 30), x("tf.sample", 60),
              # round 1: overlapping kernels (0-10, 5-15), one nested (2-8) and one past the round end (25-35)
              x("k1", 0, 10, "kernel"), x("k2", 5, 10, "kernel"), x("k3", 2, 6, "kernel"), x("k1", 25, 10, "kernel"),
              x("cudaLaunchKernel", 1, cat="cuda_runtime"), x("cudaGraphLaunch", 3, cat="cuda_runtime"),
              x("cudaStreamSynchronize", 20, cat="cuda_runtime"),
              x("Memcpy DtoH (Device -> Pinned)", 21, 1, "gpu_memcpy"),
              # round 2: the clipped tail of 25-35 only
              x("cuLaunchKernel", 40, cat="cuda_driver")]
    s = trace.summarize({"traceEvents": events})
    first, second = s["per_round"]
    assert s["rounds"] == 2
    assert first == {"wall_us": 30, "gpu_busy_us": 20, "launches": 2, "syncs": 1, "d2h_copies": 1}
    assert second == {"wall_us": 30, "gpu_busy_us": 5, "launches": 1, "syncs": 0, "d2h_copies": 0}
    assert s["top_kernels"][0]["name"] == "k1" and s["top_kernels"][0]["calls"] == 2


def test_block_bootstrap_keeps_constant_means_and_never_spans_items():
    np = pytest.importorskip("numpy")
    values = np.stack([np.full(300, 2.5), np.r_[np.zeros(100), np.ones(200)]], 1)
    boot = kl.block_bootstrap(values, [0, 100], block=64, resamples=200, seed=1)
    assert boot.shape == (200, 2) and np.allclose(boot[:, 0], 2.5)
    assert 0.4 < boot[:, 1].mean() < 0.9                       # mixes both items around the 2/3 position mean


def test_block_bootstrap_intervals_centre_on_the_mean_wherever_the_change_sits():
    # a change in each reply's first 32 positions, and one confined to a short reply: moving blocks that do not wrap
    # centred these at a third and at a fiftieth of the sample mean (the intervals missed it)
    np = pytest.importorskip("numpy")
    front = np.zeros(13 * 256)
    for item in range(13):
        front[item * 256:item * 256 + 32] = 1.0
    short = np.r_[np.ones(40), np.zeros(12 * 256)]
    for values, offsets in ((front, [i * 256 for i in range(13)]), (short, [0] + [40 + i * 256 for i in range(12)])):
        boot = kl.block_bootstrap(values[:, None], offsets, block=64, resamples=4000, seed=0)[:, 0]
        low, high = np.quantile(boot, [0.025, 0.975])
        assert abs(boot.mean() - values.mean()) < 0.05 * values.mean() and low < values.mean() < high
    with pytest.raises(ValueError):
        kl.block_bootstrap(front[:, None], [0], block=0, resamples=10, seed=0)


def test_golden_gate_is_non_inferiority_with_margins_calibrated_from_reorder_variants(tmp_path):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    golden = _log_softmax(np, rng.normal(0, 3, (3000, 64)))
    old = golden + rng.normal(0, 0.3, golden.shape)

    def near(base, scale):                       # a reorder-like perturbation of a build's log-probabilities
        return _log_softmax(np, base + rng.normal(0, scale, base.shape))

    _dump(tmp_path / "golden.npz", golden, 16, golden=True)
    rows = {"old": _log_softmax(np, old), "same": near(old, 0.02), "worse": near(old, 0.5),
            **{f"reorder{i}": near(old, 0.05) for i in range(4)}}
    for name, lp in rows.items():
        _dump(tmp_path / f"{name}.npz", lp, 16, reference=tmp_path / "golden.npz")
    _dump(tmp_path / "engine.npz", golden, 16)                      # an engine dump is no golden reference
    _dump(tmp_path / "old-vs-engine.npz", rows["old"], 16, reference=tmp_path / "engine.npz")
    with pytest.raises(ValueError, match="not a campaign_golden reference"):
        kl.gate(str(tmp_path / "engine.npz"), str(tmp_path / "old-vs-engine.npz"), str(tmp_path / "old-vs-engine.npz"))
    def path(name):
        return str(tmp_path / f"{name}.npz")

    variants = {f"r{i}": path(f"reorder{i}") for i in range(4)}
    record = kl.calibrate(path("golden"), path("old"), variants, resamples=300)

    def largest_bound(rows, key):
        return max(abs(b) for v in rows.values() for b in v[key]["ci95"])

    # the margins are the largest deviation (95% bound) a reorder-only build shows, fixed before any candidate
    assert record["margin_method"] == "ci_bound" and len(record["variants"]) == 4
    assert record["delta_kl"] == largest_bound(record["variants"], "kl_delta") > 0
    assert record["delta_agreement"] == largest_bound(record["variants"], "agreement_delta")
    assert record["delta_kl_pct_of_base_kl"] == pytest.approx(100 * record["delta_kl"]
                                                              / record["base_values"]["kl_old"]["mean"])
    held_out = record["leave_one_out"]["variants"]               # each variant judged with the others' margins
    for name, check in held_out.items():
        others = {k: v for k, v in record["variants"].items() if k != name}
        assert check["delta_kl"] == largest_bound(others, "kl_delta")
        assert check["delta_agreement"] == largest_bound(others, "agreement_delta")
        row = record["variants"][name]
        assert (check["status"], check["measures"]) == kl.verdict(row["kl_delta"]["ci95"],
                                                                  row["agreement_delta"]["ci95"],
                                                                  check["delta_kl"], check["delta_agreement"])
    assert record["leave_one_out"]["false_alarms"] == sorted(n for n, c in held_out.items() if c["status"] != "pass")
    assert record["variants"]["r0"]["new_sha256"] == kl.file_sha256(path("reorder0"))
    _dump(tmp_path / "exact.npz", golden, 16, reference=tmp_path / "golden.npz")      # a base whose KL is exactly 0
    exact = kl.calibrate(path("golden"), path("exact"), variants, resamples=300)
    assert exact["base_values"]["kl_old"]["mean"] == 0 and exact["delta_kl_pct_of_base_kl"] is None   # no percentage
    assert kl.gate(path("golden"), path("exact"), path("same"), calibration=exact,
                   resamples=300)["margins"]["delta_kl_pct_of_base_kl"] is None
    with pytest.raises(ValueError, match="4 to 6 reorder-only variants, not 3"):
        kl.calibrate(path("golden"), path("old"), dict(list(variants.items())[:3]), resamples=300)
    assert kl.gate(path("golden"), path("old"), path("same"), resamples=300)["status"] == "uncalibrated"
    passed = kl.gate(path("golden"), path("old"), path("same"), calibration=record, resamples=300)
    failed = kl.gate(path("golden"), path("old"), path("worse"), calibration=record, resamples=300)
    assert passed["status"] == "pass" and passed["kl_delta"]["ci95"][1] <= record["delta_kl"]
    assert passed["margins"]["delta_kl_pct_of_base_kl"] == record["delta_kl_pct_of_base_kl"]
    assert failed["status"] == "fail" and "kl" in failed["failed"]        # wholly beyond the margin: a regression
    assert failed["kl_delta"]["ci95"][0] > record["delta_kl"]
    old_rule = {k: v for k, v in record.items() if k != "margin_method"}  # a record made under another margin rule
    with pytest.raises(ValueError, match="recalibrate"):
        kl.gate(path("golden"), path("old"), path("same"), calibration=old_rule, resamples=300)
    with pytest.raises(ValueError, match="does not match"):
        kl.gate(path("golden"), path("old"), path("same"), calibration=dict(record, eval_sha256="x"), resamples=300)
    with pytest.raises(ValueError, match="bootstrap resamples"):
        kl.gate(path("golden"), path("old"), path("same"), calibration=record, resamples=200)
    _dump(tmp_path / "copy.npz", rows["old"], 16, reference=tmp_path / "golden.npz")    # a patch that did not load
    with pytest.raises(ValueError, match="bit-identical to the base"):
        kl.calibrate(path("golden"), path("old"), dict(variants, r3=path("copy")), resamples=300)
    broken = rows["same"].copy()
    broken[[5, 900]] = np.nan                                       # a kernel that returned NaN logits twice
    _dump(tmp_path / "broken.npz", broken, 16, reference=tmp_path / "golden.npz")
    with pytest.raises(ValueError, match="not finite at 2 positions"):      # NaN bounds would compare as a pass
        kl.gate(path("golden"), path("old"), path("broken"), calibration=record, resamples=300)


def test_calibrated_gate_verdict_order_and_exit_status(tmp_path, monkeypatch):
    """fail when an interval lies wholly beyond its margin (a regression is established); pass when the KL upper
    bound and the agreement lower bound lie within their margins (at any width: improvements never fail); otherwise
    incomplete (an interval straddles its margin). No verdict never exits 0."""

    boot = {"block": 64, "resamples": 4000, "seed": 0}
    record = {"golden_sha256": "g", "eval_sha256": "e", "positions": 10, "bootstrap": boot, "base": "old.npz",
              "margin_method": "ci_bound", "delta_kl": 1.0, "delta_kl_pct_of_base_kl": 50.0, "delta_agreement": 0.1,
              "variants": {f"r{i}": {} for i in range(4)}, "leave_one_out": {"false_alarms": []}}

    def paired_with(kl_ci, agree_ci):
        return lambda *args, **kwargs: {
            "golden_sha256": "g", "eval_sha256": "e", "positions": 10, "bootstrap": dict(boot),
            "kl_delta": {"mean": sum(kl_ci) / 2, "ci95": list(kl_ci)},
            "agreement_delta": {"mean": sum(agree_ci) / 2, "ci95": list(agree_ci)}}

    cases = (((1.2, 1.4), (-0.01, 0.01), "fail", ["kl"]),               # KL wholly beyond its margin
             ((1.5, 9.0), (-0.01, 0.01), "fail", ["kl"]),               # ... at any width
             ((-0.2, 0.2), (-0.5, -0.2), "fail", ["agreement"]),        # agreement wholly beyond
             ((1.2, 1.4), (-0.5, -0.2), "fail", ["kl", "agreement"]),
             ((-0.5, 0.9), (-0.01, 0.01), "pass", []),                  # within the margins: no width rule
             ((-9.0, -1.5), (0.2, 0.4), "pass", []),                    # an improvement never fails
             ((0.2, 1.0), (-0.1, 0.0), "pass", []),                     # bounds on the margins count as within
             ((-0.4, 0.5), (-0.05, 0.03), "pass", []),
             ((0.5, 1.2), (-0.01, 0.01), "incomplete", ["kl"]),         # straddles the KL margin: uncertainty
             ((-0.2, 0.2), (-0.12, 0.0), "incomplete", ["agreement"]),
             ((-0.02, 0.02), (-0.11, -0.05), "incomplete", ["agreement"]),
             ((0.5, 1.2), (-0.12, 0.0), "incomplete", ["kl", "agreement"]))
    for kl_ci, agree_ci, status, measures in cases:
        monkeypatch.setattr(kl, "paired", paired_with(kl_ci, agree_ci))
        result = kl.gate("g", "o", "n", calibration=record)
        assert (result["status"], result["failed"] or result["straddling"]) == (status, measures), (kl_ci, agree_ci)
        assert result["margins"]["delta_kl_pct_of_base_kl"] == 50.0
    (tmp_path / "calibration.json").write_text(json.dumps(record))
    for kl_ci, code in (((-0.4, 0.5), 0), ((1.2, 1.4), 1), ((0.5, 1.2), 3), (None, 3)):    # None: uncalibrated
        monkeypatch.setattr(kl, "paired", paired_with(kl_ci or (0.0, 0.0), (-0.01, 0.01)))
        margins = ["--calibration", str(tmp_path / "calibration.json")] if kl_ci else []
        monkeypatch.setattr(sys, "argv", ["campaign_kl.py", "gate", "g", "o", "n", *margins])
        assert kl.main() == code
    monkeypatch.setattr(kl, "paired", paired_with((4.0, 6.0), (-0.01, 0.01)))        # a gross KL regression
    out = tmp_path / "gate.json"
    monkeypatch.setattr(sys, "argv", ["campaign_kl.py", "gate", "g", "o", "n", "--calibration",
                                      str(tmp_path / "calibration.json"), "--out", str(out)])
    # json writes and reads NaN and Infinity, margins that would pass any candidate; negative or null: no margin
    refusals = [({**record, field: bad}, "not finite non-negative numbers")
                for field, bad in (("delta_kl", math.nan), ("delta_kl", math.inf), ("delta_agreement", math.nan),
                                   ("delta_agreement", -0.1), ("delta_kl", None))]
    refusals += [({k: v for k, v in record.items() if k != "margin_method"}, "recalibrate"),
                 (dict(record, golden_sha256="other"), "golden_sha256 does not match"),
                 (dict(record, bootstrap=dict(boot, seed=1)), "bootstrap seed")]
    for calibration, reason in refusals:       # each refusal kind, over a gross regression: status error, exit 4
        (tmp_path / "calibration.json").write_text(json.dumps(calibration))
        assert kl.main() == 4, reason
        result = json.loads(out.read_text())
        assert result["status"] == "error" and reason in result["error"], (reason, result)

    def refuse(*args, **kwargs):
        raise ValueError("KL(golden || n) is not finite at 2 positions")

    monkeypatch.setattr(kl, "paired", refuse)                   # comparison refusals (non-finite KL, foreign dumps)
    assert kl.main() == 4 and "not finite" in json.loads(out.read_text())["error"]
    monkeypatch.setattr(sys, "argv", ["campaign_kl.py", "gate", "g", "o", "n", "--calibration",
                                      str(tmp_path / "missing.json"), "--out", str(out)])
    assert kl.main() == 4 and "FileNotFoundError" in json.loads(out.read_text())["error"]
    monkeypatch.setattr(kl, "paired", paired_with((-0.4, 0.5), (-0.01, 0.01)))      # a pass that cannot be recorded
    (tmp_path / "calibration.json").write_text(json.dumps(record))
    monkeypatch.setattr(sys, "argv", ["campaign_kl.py", "gate", "g", "o", "n", "--calibration",
                                      str(tmp_path / "calibration.json"), "--out", str(tmp_path / "no-dir" / "g.json")])
    assert kl.main() == 4
    for bad in (["r0"], ["=x.npz"], ["r0=a.npz", "r0=b.npz"]):            # NAME=DUMP, distinct names
        monkeypatch.setattr(sys, "argv", ["campaign_kl.py", "calibrate", "g", "b", *bad])
        with pytest.raises(SystemExit) as stop:
            kl.main()
        assert stop.value.code == 2


def test_golden_float32_layers_compute_in_float32_and_restore_weight_dtypes():
    torch = pytest.importorskip("torch")
    golden = load("campaign_golden")
    seen = []

    class TinyDecoderLayer(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(4, 4, dtype=torch.bfloat16)
            self.register_buffer("scale", torch.ones(1, dtype=torch.bfloat16))

        def forward(self, x):
            seen.append((self.proj.weight.dtype, self.scale.dtype, x.dtype))
            return self.proj(x) * self.scale

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(10, 4, dtype=torch.bfloat16)
            self.layers = torch.nn.ModuleList([TinyDecoderLayer(), TinyDecoderLayer()])
            self.vision_tower = torch.nn.Linear(4, 4, dtype=torch.bfloat16)

        def forward(self, ids):
            x = self.embed(ids)
            for layer in self.layers:
                x = layer(x)
            return x

    model = Tiny()
    bf16_weight = model.layers[0].proj.weight.detach().clone()
    counts = golden.float32_layers(torch, model)
    out = model(torch.tensor([[1, 2, 3]]))
    assert counts == {"layers": 2, "calls": 2} and out.dtype == torch.float32
    assert seen == [(torch.float32, torch.float32, torch.float32)] * 2
    assert model.layers[0].proj.weight.dtype == torch.bfloat16 and torch.equal(model.layers[0].proj.weight, bf16_weight)
    assert model.layers[1].scale.dtype == torch.bfloat16 and model.embed.weight.dtype == torch.float32
    assert model.vision_tower.weight.dtype == torch.bfloat16


def test_golden_loader_keeps_float32_constants_and_matches_a_float32_model(tmp_path):
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    if not hasattr(transformers, "Gemma4ForConditionalGeneration"):
        pytest.skip("transformers without Gemma4")
    golden = load("campaign_golden")
    text = dict(num_hidden_layers=2, hidden_size=72, intermediate_size=96, moe_intermediate_size=32, num_experts=4,
                top_k_experts=2, num_attention_heads=4, num_key_value_heads=2, num_global_key_value_heads=1,
                head_dim=16, global_head_dim=32, vocab_size=128, sliding_window=8, enable_moe_block=True,
                layer_types=["sliding_attention", "full_attention"])
    vision = dict(num_hidden_layers=1, hidden_size=32, intermediate_size=32, num_attention_heads=2)
    torch.manual_seed(0)
    model = transformers.Gemma4ForConditionalGeneration(
        transformers.Gemma4Config(text_config=text, vision_config=vision)).to(torch.bfloat16)
    model.save_pretrained(tmp_path)
    loaded, report = golden.load_bf16(torch, transformers, transformers.Gemma4ForConditionalGeneration, tmp_path,
                                      "cpu")
    assert report["parameter_dtypes"] == ["torch.bfloat16"]
    scale = loaded.model.language_model.embed_tokens.embed_scale
    assert scale.dtype == torch.float32 and abs(float(scale) - 72 ** 0.5) < 1e-6      # bf16 would give 8.5
    golden.float32_layers(torch, loaded)
    reference = transformers.Gemma4ForConditionalGeneration.from_pretrained(tmp_path, dtype=torch.float32).eval()
    ids = torch.tensor([[3, 17, 5, 99, 42, 7, 8, 64, 1, 2, 3]])
    with torch.no_grad():
        assert torch.equal(loaded(input_ids=ids).logits.float(), reference(input_ids=ids).logits.float())
    rows, dtype = golden.score(torch, loaded, [3, 17, 5, 99], [42, 7, 8, 64, 1, 2, 3], chunk=2, top_k=8)
    with torch.no_grad():
        full = torch.log_softmax(reference(input_ids=ids).logits[0, 3:-1].float(), -1)
    forced = torch.tensor([r["forced"] for r in rows])
    assert dtype == "torch.float32" and torch.allclose(forced, full[torch.arange(7), ids[0, 4:]], atol=1e-5)
    # a checkpoint without one persistent buffer (layer_scalar multiplies a layer's output) fails instead of loading
    # the buffer's constructor value
    from safetensors.torch import load_file, save_file
    partial = tmp_path / "partial"
    partial.mkdir()
    (partial / "config.json").write_text((tmp_path / "config.json").read_text())
    tensors = load_file(str(tmp_path / "model.safetensors"))
    dropped = next(key for key in tensors if key.endswith("layer_scalar"))
    save_file({k: v for k, v in tensors.items() if k != dropped}, str(partial / "model.safetensors"))
    with pytest.raises(ValueError, match="persistent buffers were not in the checkpoint"):
        golden.load_bf16(torch, transformers, transformers.Gemma4ForConditionalGeneration, partial, "cpu")
    # a checkpoint that keeps a module under another path (tvall43's Qwen3.6 vision tower) loads only through an
    # explicit rename, into the same values; a rename that matches nothing fails
    tower_tensors = sum(k.startswith("model.vision_tower.") for k in tensors)
    assert tower_tensors
    moved = tmp_path / "moved"
    moved.mkdir()
    (moved / "config.json").write_text((tmp_path / "config.json").read_text())
    save_file({k.replace("model.vision_tower.", "model.language_model.vision_tower.", 1): v
               for k, v in tensors.items()}, str(moved / "model.safetensors"))
    with pytest.raises(ValueError, match="have no model tensor"):
        golden.load_bf16(torch, transformers, transformers.Gemma4ForConditionalGeneration, moved, "cpu")
    renamed, report = golden.load_bf16(torch, transformers, transformers.Gemma4ForConditionalGeneration, moved, "cpu",
                                       {"model.language_model.vision_tower.": "model.vision_tower."})
    assert report["renamed"] == {"model.language_model.vision_tower.": tower_tensors}
    original = dict(loaded.named_parameters())
    assert all(torch.equal(p.to(original[n].dtype), original[n]) for n, p in renamed.named_parameters()
               if n.startswith("model.vision_tower."))
    with pytest.raises(ValueError, match="match no checkpoint tensor"):
        golden.load_bf16(torch, transformers, transformers.Gemma4ForConditionalGeneration, tmp_path, "cpu",
                         {"model.nothing.": "model.vision_tower."})


def test_golden_refuses_held_items_before_the_model_loads(tmp_path):
    items = [{"name": "short", "prompt": [1] * 10, "answer": [2]},
             {"name": "long", "prompt": [1] * (bench.LONG_PREFILL_TOKENS + 1), "answer": [2]}]
    (tmp_path / "eval.json").write_text(json.dumps({"version": 1, "items": items}))
    run = subprocess.run([sys.executable, str(ROOT / "tools" / "campaign_golden.py"), "--model", str(tmp_path / "none"),
                          "--eval", str(tmp_path / "eval.json"), "--out", str(tmp_path / "g.npz"), "--source", "s@1"],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 2 and "['long'] exceed" in run.stderr and "long-prefill hold" in run.stderr
    (tmp_path / "short.json").write_text(json.dumps({"version": 1, "items": items[:1]}))
    run = subprocess.run([sys.executable, str(ROOT / "tools" / "campaign_golden.py"), "--model", str(tmp_path / "none"),
                          "--eval", str(tmp_path / "short.json"), "--out", str(tmp_path / "g.npz"), "--source", "s@1",
                          "--rename", "model.a.=model.x.", "--rename", "model.a.b.=model.y."],
                         capture_output=True, text=True, check=False)
    assert run.returncode == 2 and "overlap" in run.stderr       # nested prefixes would depend on argument order


def test_heartbeat_ignores_the_role_chunk_and_keepalives_and_counts_call_deltas(tmp_path, monkeypatch):
    frames = [b": keepalive\n\n",
              b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n',
              b": keepalive\n\n",
              b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"name": "f"}}]}}]}\n\n',
              (b'data: {"choices": [{"delta": {}, "finish_reason": "tool_calls"}], "usage": {"prompt_tokens": 3, '
               b'"completion_tokens": 2}, "tensorfold": {"token_ids": [5, 6]}}\n\ndata: [DONE]\n\n')]

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for frame in frames:
                self.wfile.write(frame)
                self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    states = []
    original = bench.Progress.update
    monkeypatch.setattr(bench.Progress, "update", lambda phase, **fields: (
        original(phase, **fields), bench.Progress.path and states.append(dict(bench.Progress.state))))
    try:
        bench.Progress.configure(str(tmp_path / "beat.json"))
        reply = bench.stream_chat(f"http://127.0.0.1:{server.server_port}", {"messages": []}, prompt_tokens=3,
                                  label="probe")
    finally:
        server.shutdown()
        server.server_close()
        thread.join()
        bench.Progress.configure(None)
    assert reply["tool_call_deltas"] and reply["first"] is None          # no text: client TTFT stays undefined
    assert [s["phase"] for s in states] == ["idle", "prefill", "decode", "idle"]
    assert states[2]["request"] == "probe" and states[2]["prompt_tokens"] == 3        # identity kept into decode
    assert "request" not in states[3]                                                  # idle drops it


def test_subset_cuts_a_golden_reference_to_short_items_and_records_its_origin(tmp_path):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(5)
    lp = _log_softmax(np, rng.normal(0, 3, (30, 32)))
    items = [{"name": "short", "prompt": list(range(10)), "answer": list(range(10))},
             {"name": "long", "prompt": list(range(100)), "answer": list(range(20))}]
    (tmp_path / "eval.json").write_text(json.dumps({"version": 1, "items": items}))
    order = _dump(tmp_path / "golden.npz", lp, 8, golden=True, offsets=(0, 10), names=["short", "long"],
                  eval_sha256=kl.file_sha256(str(tmp_path / "eval.json")), source="s@1", model_dir="/golden")
    result = kl.subset(str(tmp_path / "golden.npz"), str(tmp_path / "eval.json"), str(tmp_path / "out"),
                       max_prompt_tokens=50)
    cut = np.load(tmp_path / "out" / "golden" / "kl.npz")
    cut_meta = json.loads(str(cut["meta"]))
    assert result["items"] == ["short"] and result["positions"] == 10 and list(cut["offsets"]) == [0]
    assert np.array_equal(cut["lp"], np.take_along_axis(lp, order, -1).astype(np.float32)[:10])
    assert cut_meta["eval_sha256"] == kl.file_sha256(str(tmp_path / "out" / "eval.json")) and cut_meta["golden"]
    assert cut_meta["subset_of"]["golden_sha256"] == kl.file_sha256(str(tmp_path / "golden.npz"))
    assert json.loads((tmp_path / "out" / "eval.json").read_text())["subset_of"]["items"] == ["short"]
    assert "golden_dir" in json.loads((tmp_path / "out" / "golden" / "meta.json").read_text())
    # candidates dumped from the subset's eval set are gated with the full set named; full-set dumps do not mix in
    cut_lp = lp[:10]
    for name in ("old", "new"):
        _dump(tmp_path / f"{name}.npz", cut_lp, 8, reference=tmp_path / "out" / "golden" / "kl.npz",
              eval_sha256=cut_meta["eval_sha256"])
    pair = kl.paired(str(tmp_path / "out" / "golden" / "kl.npz"), str(tmp_path / "old.npz"), str(tmp_path / "new.npz"),
                     block=64, resamples=50, seed=0)
    assert pair["golden_subset_of"] == cut_meta["subset_of"] and pair["positions"] == 10
    _dump(tmp_path / "full.npz", lp, 8, reference=tmp_path / "golden.npz",
          eval_sha256=cut_meta["subset_of"]["eval_sha256"])
    with pytest.raises(ValueError, match="different eval sets"):
        kl.paired(str(tmp_path / "out" / "golden" / "kl.npz"), str(tmp_path / "old.npz"), str(tmp_path / "full.npz"),
                  block=64, resamples=50, seed=0)
    full_eval = kl.file_sha256(str(tmp_path / "eval.json"))
    with pytest.raises(ValueError, match="would replace"):          # never written over the files it is cut from
        kl.subset(str(tmp_path / "golden.npz"), str(tmp_path / "eval.json"), str(tmp_path), max_prompt_tokens=50)
    assert kl.file_sha256(str(tmp_path / "eval.json")) == full_eval
    cli = subprocess.run([sys.executable, str(ROOT / "tools" / "campaign_kl.py"), "subset",
                          str(tmp_path / "golden.npz"), str(tmp_path / "eval.json"), str(tmp_path / "cli"),
                          "--max-prompt-tokens", "50"],
                         capture_output=True, check=False)
    assert cli.returncode == 0 and (tmp_path / "cli" / "golden" / "kl.npz").exists(), cli.stderr

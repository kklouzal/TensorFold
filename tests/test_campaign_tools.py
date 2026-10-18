"""Campaign measurement tools: interval and ratio statistics, run records over a loopback server, the tool-call schema
checks, needle placement and the teacher-forced KL bound. No model or GPU."""

import contextlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import json
import math
from pathlib import Path
import sys
import threading

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
        run = tmp_path / f"s-c1-{position}-base"
        run.mkdir()
        (run / "bench.json").write_text(json.dumps(doc))
    result = bench.analyze(str(tmp_path), "s", None)
    assert result["incomplete_runs"] == ["c1-2-base"] and len(result["tool_versions"]) == 2
    text = bench.markdown(result)
    assert "Incomplete runs" in text and "2 campaign_bench.py versions" in text


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
            if self.path.endswith("/v1/tokenize"):
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


def test_run_records_every_scheduled_request_with_its_metrics(tmp_path):
    _write_inputs(tmp_path)
    seen = []
    with fake_server(seen) as url:
        bench.run(url, str(tmp_path / "profile.json"), str(tmp_path / "prompts.json"), str(tmp_path / "out.json"),
                  meta=None, label="t", select={"depths": [], "tasks": [], "thinking": None})
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


def test_gates_refuse_replies_without_token_ids(tmp_path, monkeypatch):
    _write_inputs(tmp_path)
    with fake_server(ids=False) as url:
        monkeypatch.setattr(sys, "argv", ["campaign_gates.py", "identity", "--url", url, "--out", str(tmp_path / "o"),
                                          "--profile", str(tmp_path / "profile.json"),
                                          "--prompts", str(tmp_path / "prompts.json"),
                                          "--gate-prompts", "short-code-nothink", "--max-tokens", "6"])
        with pytest.raises(ValueError, match="return_token_ids"):
            gates.main()


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
    """float32 log-sum-exp of each row outside ``ids``, as ``Capture._tail``; -inf for an empty complement."""

    rest = lp_full.astype(np.float32).copy()
    np.put_along_axis(rest, ids, -np.inf, -1)
    top = rest.max(-1, keepdims=True)
    with np.errstate(invalid="ignore"):
        lse = top + np.log(np.exp(rest - top).sum(-1, keepdims=True, dtype=np.float32))
    return np.where(np.isneginf(top), -np.inf, lse).ravel().astype(np.float64)


def _dump(path, lp_full, top_k, reference=None):
    np = pytest.importorskip("numpy")
    order = np.argsort(-lp_full, axis=-1)[:, :top_k]
    arrays = {"ids": order.astype(np.int32), "lp": np.take_along_axis(lp_full, order, -1).astype(np.float32),
              "tail_lp": _outside(np, lp_full, order), "forced": lp_full[:, 0],
              "entropy": -(np.exp(lp_full) * lp_full).sum(-1), "offsets": np.array([0])}
    if reference is not None:
        arrays["at_ref"] = np.take_along_axis(lp_full, reference, -1).astype(np.float32)
        arrays["at_ref_tail_lp"] = _outside(np, lp_full, reference)
    meta = {"eval_sha256": "e", "top_k": top_k, "logits_dtype": "torch.float32", "names": ["all"]}
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
        ref = _dump(tmp_path / f"ref{k}.npz", p, k)
        _dump(tmp_path / f"cand{k}.npz", q, k, reference=ref)
        result = kl.compare(str(tmp_path / f"ref{k}.npz"), str(tmp_path / f"cand{k}.npz"))
        assert check(result["kl"]["mean"]), (k, result["kl"]["mean"], exact)
        assert result["kl"]["kind"] == "lower bound" and not result["identical_logprobs"]
    same = _dump(tmp_path / "same.npz", p, 8, reference=_dump(tmp_path / "ref.npz", p, 8))
    result = kl.compare(str(tmp_path / "ref.npz"), str(tmp_path / "same.npz"))
    assert result["kl"]["max"] < 1e-6 and result["top1_agreement"] == 1.0 and result["identical_logprobs"]
    assert same.shape == (50, 8)


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
    ref = _dump(tmp_path / "ref.npz", log_softmax32(x), 64)
    _dump(tmp_path / "cand.npz", log_softmax32(y), 64, reference=ref)
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


def test_golden_gate_passes_an_equal_error_change_and_fails_a_worse_one(tmp_path):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    golden = _log_softmax(np, rng.normal(0, 3, (600, 64)))
    old = _log_softmax(np, golden + rng.normal(0, 0.3, golden.shape))
    same = _log_softmax(np, golden + rng.normal(0, 0.3, golden.shape))
    worse = _log_softmax(np, golden + rng.normal(0, 0.6, golden.shape))
    ref = _dump(tmp_path / "golden.npz", golden, 16)
    for name, lp in (("old", old), ("same", same), ("worse", worse)):
        _dump(tmp_path / f"{name}.npz", lp, 16, reference=ref)
    passed = kl.gate(str(tmp_path / "golden.npz"), str(tmp_path / "old.npz"), str(tmp_path / "same.npz"),
                     resamples=500)
    failed = kl.gate(str(tmp_path / "golden.npz"), str(tmp_path / "old.npz"), str(tmp_path / "worse.npz"),
                     resamples=500)
    assert passed["status"] == "pass" and passed["kl_delta"]["ci95"][0] <= 0 <= passed["kl_delta"]["ci95"][1]
    assert failed["status"] == "fail" and "kl" in failed["failed"] and failed["kl_delta"]["ci95"][0] > 0
    assert failed["kl_new"]["mean"] > failed["kl_old"]["mean"] > 0


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

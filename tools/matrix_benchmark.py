"""Serving matrix benchmark: prefill and decode tok/s over concurrency x prompt length, and its scoring rule.

Subcommands (see docs/benchmarking.md, "Serving matrix"):

  corpus   Freeze the prompt corpus from a pinned git commit (README.md, docs/**/*.md, src/**/*.py), with the
           tokenizer's special-token strings neutralized. Standard library; needs git.
  plan     Cut one unique, exactly sized chat prompt per (mode, arrival, C, L, repeat, slot) from the corpus and
           record its character span, token count and SHA-256. Needs ``transformers`` (run it in the serving image).
  run      Stream the planned requests at an OpenAI-compatible server: C requests per cell start together (or
           staggered), each with a fixed output length and ignore_eos. Records every request's raw timings, usage
           and the server's per-request statistics where it reports them. Standard library.
  summary  Per-cell table of one arm (one or more result files).
  compare  Per-cell candidate/reference ratios with uncertainty and the predeclared whole-matrix verdict.

Per request: TTFT = first streamed text piece - send; prefill tok/s = prompt tokens / TTFT; decode tok/s =
(completion tokens - 1) / (last piece - first piece). SSE pieces can carry several tokens (DFlash rounds), so the
client decode rate is a text-arrival estimate; the TensorFold per-request ``tensorfold`` object (engine
``prefill_s``/``decode_s``) is recorded beside it. Per cell repeat: prefill_agg = all prompt tokens / (last
request's first piece - first send); decode_agg = committed tokens after excluding one first token per request,
divided by the time between the first and last text arrivals across the cohort; decode_median = median request
decode tok/s; total = (prompt + completion tokens) / makespan.
"""

import argparse
import hashlib
import json
from itertools import pairwise
import math
import platform
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

if __package__:
    from . import openai_protocol as protocol
    from .benchmark_output import write_json
else:
    import openai_protocol as protocol
    from benchmark_output import write_json

SCHEMA = "tensorfold.matrix-benchmark"
SCHEMA_VERSION = 1
PLAN_SCHEMA = "tensorfold.matrix-plan"
CORPUS_SUFFIXES = (("README.md", ""), ("docs/", ".md"), ("src/", ".py"))
MODES = ("greedy", "sampled")
SCORED = ("prefill_agg_tps", "decode_median_tps")
AGGREGATE_DECODE = "decode_agg_tps"
HEADER = ("Request {key}.\nBelow is an excerpt from a software project's documentation and source code.\n\n")
FOOTER = ("\n\nWrite a detailed technical summary of the excerpt above: its purpose, its main components and "
          "any notable design decisions.")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def int_list(text: str) -> list[int]:
    values = [int(part) for part in text.split(",") if part]
    if not values or any(value <= 0 for value in values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError("comma-separated distinct positive integers required")
    return values


def word_list(allowed):
    def parse(text: str) -> list[str]:
        values = [part for part in text.split(",") if part]
        if not values or any(value not in allowed for value in values) or len(set(values)) != len(values):
            raise argparse.ArgumentTypeError(f"comma-separated distinct values from {', '.join(allowed)}")
        return values
    return parse


def arrival_name(stagger_ms: float) -> str:
    if not math.isfinite(stagger_ms) or stagger_ms < 0:
        raise ValueError("stagger milliseconds must be finite and nonnegative")
    return "sync" if stagger_ms == 0 else f"stagger{stagger_ms:g}ms"


def prompt_key(mode: str, arrival: str, c: int, length: int, rep, slot: int) -> str:
    return f"v1/{mode}/{arrival}/c{c}/l{length}/r{rep}/s{slot}"


def cell_key(mode: str, arrival: str, c: int, length: int) -> str:
    return f"v1/{mode}/{arrival}/c{c}/l{length}"


def message(key: str, excerpt: str) -> str:
    return HEADER.format(key=key) + excerpt + FOOTER


# ---------------------------------------------------------------- corpus

def build_corpus(args) -> int:
    commit = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "--verify", args.commit + "^{commit}"],
                                     text=True).strip()
    names = subprocess.check_output(["git", "-C", str(args.repo), "ls-tree", "-r", "-z", "--name-only", commit])
    paths = sorted(name for name in names.decode().split("\0")
                   if any(name.startswith(prefix) and name.endswith(suffix) for prefix, suffix in CORPUS_SUFFIXES))
    tokenizer = json.loads((args.tokenizer / "tokenizer.json").read_text(encoding="utf-8"))
    specials = sorted({token["content"] for token in tokenizer["added_tokens"] if token.get("special")}, key=len,
                      reverse=True)
    if any(not special.startswith("<") for special in specials):
        raise ValueError("special-token neutralization assumes each special token starts with '<'")
    parts, skipped, replaced = [], [], 0
    for path in paths:
        blob = subprocess.check_output(["git", "-C", str(args.repo), "show", f"{commit}:{path}"])
        try:
            text = blob.decode("utf-8")
        except UnicodeDecodeError:
            skipped.append(path)
            continue
        for special in specials:
            count = text.count(special)
            if count:
                replaced += count
                text = text.replace(special, "‹" + special[1:])
        parts.append(f"===== {path} =====\n\n{text.rstrip()}\n\n")
    corpus = "".join(parts).encode("utf-8")
    args.out.write_bytes(corpus)
    manifest = {"schema": "tensorfold.matrix-corpus", "commit": commit, "files": len(parts), "skipped": skipped,
                "special_tokens_neutralized": replaced, "bytes": len(corpus), "sha256": sha256(corpus),
                "tokenizer_sha256": sha256((args.tokenizer / "tokenizer.json").read_bytes())}
    write_json(Path(str(args.out) + ".manifest.json"), manifest)
    print(json.dumps(manifest))
    return 0


def load_corpus(path: Path, expected: str | None) -> str:
    data = path.read_bytes()
    digest = sha256(data)
    if expected is not None and digest != expected:
        raise ValueError(f"corpus {path} has SHA-256 {digest}, the plan expects {expected}")
    return data.decode("utf-8")


# ---------------------------------------------------------------- plan

def plan_keys(modes, arrivals, levels, lengths, reps, warmup):
    for mode in modes:
        if warmup:
            for length in lengths:
                yield mode, "warmup", 1, length, "w", 0
        for arrival in arrivals:
            for c in levels:
                for length in lengths:
                    for rep in range(reps):
                        for slot in range(c):
                            yield mode, arrival, c, length, rep, slot


_PLAN = {}  # the forked plan workers' read-only corpus, token starts and tokenizer


def _count(text: str) -> int:
    ids = _PLAN["tok"].apply_chat_template([{"role": "user", "content": text}], tokenize=True,
                                           add_generation_prompt=True, chat_template=_PLAN["template"],
                                           enable_thinking=False)
    if hasattr(ids, "keys"):
        ids = ids["input_ids"]
    return len(ids)


def _cut(item):
    """One exact-length prompt: a key-hashed line start, then token-budget corrections until the render fits.
    A merge at the cut can make the count step over the target; the next hashed start is then tried."""
    key, length = item
    corpus, starts, room = _PLAN["corpus"], _PLAN["starts"], _PLAN["room"][length]
    tried = []
    for attempt in range(8):
        budget = length - _count(message(key, ""))
        digest = hashlib.sha256(f"{key}#{attempt}".encode() if attempt else key.encode()).digest()
        start = room[int.from_bytes(digest[:8], "big") % len(room)]
        seen = set()
        while 0 < budget < length and start + budget < len(starts) and budget not in seen:
            seen.add(budget)
            end = starts[start + budget]
            text = message(key, corpus[starts[start]:end])
            n = _count(text)
            tried.append((attempt, budget, n))
            if n == length:
                return key, {"start": starts[start], "end": end, "tokens": n, "attempt": attempt,
                             "sha256": sha256(text.encode("utf-8"))}
            budget += length - n
    raise RuntimeError(f"{key}: could not cut an exact {length}-token prompt; tried {tried}")


def build_plan(args) -> int:
    import multiprocessing

    import transformers  # plan only: the serving image provides it

    corpus = load_corpus(args.corpus, None)
    tok = transformers.AutoTokenizer.from_pretrained(str(args.tokenizer), local_files_only=True)
    template = args.template.read_text(encoding="utf-8")
    starts = [span[0] for span in tok(corpus, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]]
    line_starts = [i for i, c in enumerate(starts) if c == 0 or corpus[c - 1] == "\n"]
    _PLAN.update(tok=tok, template=template, corpus=corpus, starts=starts,
                 room={length: [i for i in line_starts if i + length + 64 < len(starts)] for length in args.lengths})
    arrivals = [arrival_name(ms) for ms in args.stagger_ms]
    items = list(dict.fromkeys((prompt_key(*spec), spec[3]) for spec in plan_keys(
        args.modes, arrivals, args.concurrency, args.lengths, args.reps, True)))
    items.sort(key=lambda item: -item[1])  # longest first balances the pool
    with multiprocessing.get_context("fork").Pool(args.jobs) as pool:
        entries = dict(pool.imap_unordered(_cut, items, chunksize=4))
    plan = {"schema": PLAN_SCHEMA, "version": 1, "corpus_sha256": sha256(corpus.encode("utf-8")),
            "corpus_path": str(args.corpus), "tokenizer": str(args.tokenizer),
            "tokenizer_sha256": sha256((args.tokenizer / "tokenizer.json").read_bytes()),
            "template_sha256": sha256(template.encode("utf-8")),
            "render": "transformers apply_chat_template, add_generation_prompt, enable_thinking=False",
            "transformers": transformers.__version__, "header": HEADER, "footer": FOOTER,
            "prompts": {key: entries[key] for key, _ in sorted(items)}}
    write_json(args.out, plan, indent=None)
    print(json.dumps({"prompts": len(entries), "plan_sha256": sha256(args.out.read_bytes())}))
    return 0


# ---------------------------------------------------------------- run

def http_json(url: str, timeout: float = 10.0):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        if error.code == 404:
            return None
        raise


def server_stats(base: str):
    stats = http_json(base + "/stats")
    if stats is not None and stats.get("object") != "tensorfold.stats":
        return None
    return stats


def server_session(stats) -> dict | None:
    """The server process identity that makes one result file one scoring unit (``compare``)."""
    server = (stats or {}).get("server") or {}
    if not server.get("instance_id"):
        return None
    return {"instance_id": server["instance_id"], "started_unix_s": server.get("started_unix_s")}


def source_digest(root: Path) -> dict:
    digest, files = hashlib.sha256(), 0
    for path in sorted(p for p in root.rglob("*") if p.is_file() and "__pycache__" not in p.parts):
        digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes() + b"\0")
        files += 1
    return {"path": str(root), "files": files, "sha256": digest.hexdigest()}


def environment(args) -> dict:
    env = {"host": platform.node(), "kernel": platform.release(), "python": sys.version.split()[0],
           "tool_sha256": sha256(Path(__file__).read_bytes())}
    try:
        env["cpu_governor"] = Path("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor").read_text().strip()
    except OSError as error:
        env["cpu_governor"] = f"unreadable: {error}"
    try:
        env["gpus"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,power.limit,clocks.max.sm,memory.used",
             "--format=csv,noheader"], capture_output=True, text=True, timeout=30, check=True).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        env["gpus"] = f"unavailable: {error}"
    if args.source_dir:
        env["source"] = source_digest(args.source_dir)
    return env


def request_body(args, mode: str, key: str, text: str) -> dict:
    body = {"model": args.model, "messages": [{"role": "user", "content": text}], "max_tokens": args.output_tokens,
            "stream": True, "stream_options": {"include_usage": True}, "ignore_eos": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    if mode == "greedy":
        body["temperature"] = 0
    else:  # the server's default sampling parameters, reproducible per request
        body["seed"] = int.from_bytes(hashlib.sha256(key.encode()).digest()[:4], "big") & 0x7FFFFFFF
    return body


def stream_request(base: str, body: dict, gate: threading.Barrier, delay: float, args, limits) -> dict:
    data = json.dumps(body).encode()
    request = urllib.request.Request(base + "/v1/chat/completions", data=data,
                                     headers={"Content-Type": "application/json"})
    gate.wait(timeout=120)
    if delay:
        time.sleep(delay)
    sent = time.perf_counter()
    pieces, usage, runtime, texts, refusals = [], None, None, [], 0
    while True:  # --retry-overloaded: a capacity refusal (503) is resent; its time counts from the first send
        try:
            response = urllib.request.urlopen(request, timeout=args.timeout)
            break
        except urllib.error.HTTPError as error:
            if error.code != 503 or refusals >= args.retry_overloaded:
                raise
            refusals += 1
            time.sleep(args.retry_delay)
    with response:
        protocol.response_type(response, "text/event-stream")
        for chunk in protocol.sse_objects(response, limits=limits):
            now = time.perf_counter()
            if now - sent > args.deadline:
                raise TimeoutError("whole-request deadline exceeded")
            new_usage, new_runtime, new_pieces = protocol.chunk_fields(chunk)
            usage = new_usage if new_usage is not None else usage
            runtime = new_runtime if new_runtime is not None else runtime
            for piece in new_pieces:
                if piece:
                    pieces.append(now - sent)
                    texts.append(piece)
    ended = time.perf_counter() - sent
    if usage is None:
        raise ValueError("stream ended without usage")
    protocol.completion_usage(usage, args.output_tokens, prompt=True)
    output = "".join(texts)
    return {"sent": sent, "end_s": ended, "arrivals_s": pieces, "usage": usage, "server": runtime, "refusals": refusals,
            "output_chars": len(output), "output_sha256": sha256(output.encode("utf-8"))}


def request_metrics(record: dict) -> dict:
    arrivals, usage = record["arrivals_s"], record["usage"]
    n, prompt = usage["completion_tokens"], usage["prompt_tokens"]
    out = {"ttft_s": None, "prefill_tps": None, "decode_tps": None, "e2e_s": record["end_s"]}
    if arrivals:
        first, last = arrivals[0], arrivals[-1]
        out.update(ttft_s=first, prefill_tps=prompt / first if first > 0 else None)
        if n > 1 and last > first and len(arrivals) >= 2:
            out["decode_tps"] = (n - 1) / (last - first)
    server = record.get("server") or {}
    for key in ("prefill_s", "decode_s", "queue_s", "ttft_s", "decode_tps", "rounds", "drafted", "accepted",
                "cached"):
        if key in server:
            out["server_" + key] = server[key]
    return out


def finite_measurement(value) -> bool:
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def aggregate_decode_tps(requests: list[dict]) -> float | None:
    """Client-observed cohort decode, preserving stagger and time between streams."""
    if not requests:
        return None
    windows = []
    completion = 0
    for request in requests:
        if type(request) is not dict:
            return None
        sent, arrivals, usage = request.get("sent"), request.get("arrivals_s"), request.get("usage")
        tokens = usage.get("completion_tokens") if type(usage) is dict else None
        if (not finite_measurement(sent) or type(arrivals) is not list or len(arrivals) < 2
                or type(tokens) is not int or tokens < 2 or not finite_measurement(tokens)
                or any(not finite_measurement(t) or t < 0 for t in arrivals)
                or any(a > b for a, b in pairwise(arrivals)) or arrivals[-1] <= arrivals[0]):
            return None
        windows.append((float(sent), float(arrivals[0]), float(arrivals[-1])))
        completion += tokens - 1
    origin = min(sent for sent, _, _ in windows)
    first = min(sent - origin + start for sent, start, _ in windows)
    last = max(sent - origin + end for sent, _, end in windows)
    interval = last - first
    if not math.isfinite(interval) or interval <= 0:
        return None
    try:
        rate = completion / interval
    except OverflowError:
        return None
    return rate if math.isfinite(rate) and rate > 0 else None


def rep_metrics(requests: list[dict]) -> dict:
    start = min(r["sent"] for r in requests)
    firsts = [r["sent"] - start + r["arrivals_s"][0] for r in requests if r["arrivals_s"]]
    ends = [r["sent"] - start + r["end_s"] for r in requests]
    prompt = sum(r["usage"]["prompt_tokens"] for r in requests)
    completion = sum(r["usage"]["completion_tokens"] for r in requests)
    per = [r["metrics"] for r in requests]
    decode = [m["decode_tps"] for m in per if m["decode_tps"] is not None]
    ttft = [m["ttft_s"] for m in per if m["ttft_s"] is not None]
    makespan = max(ends)
    out = {"requests": len(requests), "prompt_tokens": prompt, "completion_tokens": completion,
           "makespan_s": makespan,
           "prefill_agg_tps": prompt / max(firsts) if len(firsts) == len(requests) and max(firsts) > 0 else None,
           AGGREGATE_DECODE: aggregate_decode_tps(requests),
           "decode_median_tps": statistics.median(decode) if len(decode) == len(requests) else None,
           "decode_min_tps": min(decode) if len(decode) == len(requests) else None,
           "ttft_median_s": statistics.median(ttft) if ttft else None, "ttft_max_s": max(ttft) if ttft else None,
           "output_tps": completion / makespan, "total_tps": (prompt + completion) / makespan,
           "refusals": sum(r.get("refusals", 0) for r in requests)}
    server_decode = [m["server_decode_tps"] for m in per if m.get("server_decode_tps") is not None]
    if len(server_decode) == len(requests):
        out["server_decode_median_tps"] = statistics.median(server_decode)
    server_prefill = [m["server_prefill_s"] for m in per if m.get("server_prefill_s") is not None]
    if len(server_prefill) == len(requests):
        out["server_prefill_s_sum"] = sum(server_prefill)
    return out


TOTAL_KEYS = ("requests", "prompt_tokens", "cached_prompt_tokens", "generated_tokens", "prefill_seconds",
              "decode_seconds", "queue_seconds", "rounds", "drafted", "accepted", "failed", "cancelled",
              "memory_waits", "memory_stops", "preemptions")


def stats_delta(before, after) -> dict | None:
    if before is None or after is None:
        return None
    a, b = before.get("totals") or {}, after.get("totals") or {}
    return {key: (b.get(key) - a.get(key)) if isinstance(a.get(key), (int, float)) and
            isinstance(b.get(key), (int, float)) else None for key in TOTAL_KEYS}


def settle_stats(base: str, before, expected: int):
    if before is None:
        return None
    target = before["totals"]["requests"] + expected
    stop = time.monotonic() + 15
    while True:
        after = server_stats(base)
        done = after["totals"]["requests"]
        if done > target:
            raise RuntimeError("another client's request finished during the cell")
        if done == target and after["load"]["requests_in_flight"] == 0:
            return after
        if time.monotonic() >= stop:
            raise TimeoutError("server totals did not settle after the cell")
        time.sleep(0.05)


def run_cell(base: str, args, plan: dict, corpus: str, mode: str, arrival: str, stagger: float, c: int, length: int,
             rep, limits, session) -> dict:
    keys = [prompt_key(mode, arrival, c, length, rep, slot) for slot in range(c)]
    bodies = []
    for key in keys:
        entry = plan["prompts"].get(key)
        if entry is None:
            raise KeyError(f"plan has no prompt {key}; extend the plan")
        text = message(key, corpus[entry["start"]:entry["end"]])
        if sha256(text.encode("utf-8")) != entry["sha256"]:
            raise ValueError(f"{key}: regenerated prompt does not match the plan hash")
        bodies.append(request_body(args, mode, key, text))
    before = server_stats(base)
    if before is not None and before["load"]["requests_in_flight"] != 0:
        raise RuntimeError("server is not idle before the cell")
    gate = threading.Barrier(c)
    results: list = [None] * c

    def worker(i: int):
        try:
            results[i] = stream_request(base, bodies[i], gate, i * stagger / 1e3, args, limits)
        except BaseException as error:  # noqa: BLE001 - reported in the cell, never counted as a measurement
            gate.abort()
            results[i] = {"error": f"{type(error).__name__}: {error}"}

    threads = [threading.Thread(target=worker, args=(i,), name=f"matrix-{i}") for i in range(c)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    errors = [r["error"] for r in results if "error" in r]
    cell = {"key": cell_key(mode, arrival, c, length), "mode": mode, "arrival": arrival, "c": c, "length": length,
            "rep": rep, "prompt_keys": keys, "requests": results, "problems": []}
    if errors:
        cell["status"] = "failed"
        cell["problems"] = errors
        return cell
    problems = []
    for key, record in zip(keys, results):
        record["key"] = key
        record["metrics"] = request_metrics(record)
        usage = record["usage"]
        if usage["prompt_tokens"] != plan["prompts"][key]["tokens"]:
            problems.append(f"{key}: prompt tokens {usage['prompt_tokens']} != planned {plan['prompts'][key]['tokens']}")
        if usage["completion_tokens"] != args.output_tokens:
            problems.append(f"{key}: completion tokens {usage['completion_tokens']} != {args.output_tokens}")
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
        if cached:
            problems.append(f"{key}: {cached} prompt tokens came from a prefix cache")
        if record["metrics"]["decode_tps"] is None:
            problems.append(f"{key}: no decode interval (fewer than two text pieces)")
    try:
        after = settle_stats(base, before, c)
    except (RuntimeError, TimeoutError) as error:
        after = None
        problems.append(f"server statistics: {error}")
    for stats in (before, after):
        if stats is not None and server_session(stats) != session:
            problems.append("the server process changed during the run: one result file must be one server session")
            break
    delta = stats_delta(before, after)
    if delta is not None:
        if delta["prompt_tokens"] != sum(r["usage"]["prompt_tokens"] for r in results) or \
                delta["generated_tokens"] != sum(r["usage"]["completion_tokens"] for r in results):
            problems.append("server token totals do not describe exactly this cell's requests")
        if delta["cached_prompt_tokens"]:
            problems.append("server reports cached prompt tokens")
    cell["server_totals_delta"] = delta
    cell["load_after"] = None if after is None else after.get("load")
    token_mismatch_only = all("prompt tokens" in p and "!= planned" in p for p in problems)
    if problems and not (token_mismatch_only and args.allow_prompt_token_drift):
        cell["status"] = "invalid"
    else:
        cell["status"] = "measured"
    cell["problems"] = problems
    cell["metrics"] = rep_metrics(results)
    return cell


def cell_gate(args, info: dict, c: int, length: int) -> str | None:
    if c > info["lanes"]:
        return f"pending: server runs {info['lanes']} request lane(s)"
    if info["context"] is not None and length + args.output_tokens > info["context"]:
        return f"infeasible: {length}+{args.output_tokens} tokens exceed the served {info['context']}-token context"
    if args.kv_tokens is not None and c * (length + args.output_tokens) > args.kv_tokens:
        return f"infeasible: {c} x {length + args.output_tokens} tokens exceed the {args.kv_tokens}-token KV budget"
    return None


def run(args) -> int:
    base = args.url.rstrip("/")
    plan_bytes = args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    if plan.get("schema") != PLAN_SCHEMA:
        raise ValueError("not a matrix plan")
    corpus = load_corpus(args.corpus, plan["corpus_sha256"])
    limits = protocol.from_arguments(args)
    models = http_json(base + "/v1/models")
    model = next((m for m in models["data"] if m["id"] == args.model), None)
    if model is None:
        raise ValueError(f"server does not serve {args.model}")
    stats = server_stats(base)
    lanes = args.lanes if args.lanes is not None else ((stats or {}).get("limits") or {}).get("lanes")
    if lanes is None:
        raise ValueError("the server reports no lane count; pass --lanes")
    context = model.get("max_model_len") or ((stats or {}).get("model") or {}).get("context_length")
    info = {"lanes": lanes, "context": context}
    arrival = arrival_name(args.stagger_ms)
    report = {"schema": SCHEMA, "version": SCHEMA_VERSION, "label": args.label, "note": args.note,
              "created_unix_s": time.time(), "argv": sys.argv, "url": base, "model": args.model,
              "output_tokens": args.output_tokens, "arrival": arrival, "stagger_ms": args.stagger_ms,
              "plan_sha256": sha256(plan_bytes), "corpus_sha256": plan["corpus_sha256"],
              "environment": environment(args), "server_models": models, "server_lanes": lanes,
              "server_context": context, "kv_tokens": args.kv_tokens,
              "session": server_session(stats), "server_stats_start": stats, "warmup": [], "cells": [],
              "gated": []}
    for mode in args.modes:
        for c in args.concurrency:
            for length in args.lengths:
                reason = cell_gate(args, info, c, length)
                if reason:
                    report["gated"].append({"mode": mode, "arrival": arrival, "c": c, "length": length,
                                            "status": reason.split(":")[0], "reason": reason})
    write_json(args.out, report)
    failures = 0
    for mode in args.modes:
        for length in args.warmup_lengths if args.warmup_lengths is not None else args.lengths:
            if cell_gate(args, info, 1, length):
                continue
            cell = run_cell(base, args, plan, corpus, mode, "warmup", 0.0, 1, length, "w", limits, report["session"])
            report["warmup"].append(cell)
            failures += cell["status"] == "failed"  # an invalid warmup (e.g. token drift) is still a warmup
            print(json.dumps({"warmup": cell["key"], "status": cell["status"], "problems": cell["problems"]}),
                  flush=True)
            write_json(args.out, report)
    for rep in range(args.first_rep, args.first_rep + args.reps):
        for mode in args.modes:
            for c in args.concurrency:
                for length in args.lengths:
                    if cell_gate(args, info, c, length):
                        continue
                    cell = run_cell(base, args, plan, corpus, mode, arrival, args.stagger_ms, c, length, rep, limits,
                                    report["session"])
                    failures += cell["status"] != "measured"
                    report["cells"].append(cell)
                    metrics = cell.get("metrics") or {}
                    print(json.dumps({"cell": cell["key"], "status": cell["status"],
                                      "prefill_agg_tps": metrics.get("prefill_agg_tps"),
                                      "decode_median_tps": metrics.get("decode_median_tps"),
                                      "problems": cell["problems"][:3]}), flush=True)
                    write_json(args.out, report)
                    if failures and args.fail_fast:
                        return 1
    report["server_stats_end"] = server_stats(base)
    report["completed_unix_s"] = time.time()
    write_json(args.out, report)
    return 1 if failures else 0


# ---------------------------------------------------------------- scoring

FAMILY_ALPHA = 0.05  # familywise two-sided error over all compared (cell, metric) pairs (Bonferroni)


def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction of the regularized incomplete beta function (modified Lentz)."""
    tiny = 1e-300
    c, d = 1.0, 1.0 - (a + b) * x / (a + 1.0)
    d = 1.0 / (d if abs(d) > tiny else tiny)
    h = d
    for m in range(1, 400):
        for numerator in (m * (b - m) * x / ((a + 2 * m - 1) * (a + 2 * m)),
                          -(a + m) * (a + b + m) * x / ((a + 2 * m) * (a + 2 * m + 1))):
            d = 1.0 + numerator * d
            d = 1.0 / (d if abs(d) > tiny else tiny)
            c = 1.0 + numerator / c
            c = c if abs(c) > tiny else tiny
            h *= d * c
        if abs(d * c - 1.0) < 1e-15:
            break
    return h


def t_cdf(t: float, df: float) -> float:
    """Student t cumulative distribution."""
    x = df / (df + t * t)
    a, b = df / 2.0, 0.5
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b) + a * math.log(x) + b * math.log1p(-x))
    ibeta = front * _betacf(a, b, x) / a if x < (a + 1) / (a + b + 2) else 1.0 - front * _betacf(b, a, 1 - x) / b
    return 1.0 - 0.5 * ibeta if t >= 0 else 0.5 * ibeta


def t_half(level: float, df: float) -> float:
    """Two-sided quantile: P(|T| <= q) = level."""
    if not df >= 1:
        return math.inf
    target = 0.5 + level / 2.0
    lo, hi = 0.0, 1.0
    while t_cdf(hi, df) < target:
        hi *= 2.0
    for _ in range(200):
        mid = (lo + hi) / 2.0
        lo, hi = (mid, hi) if t_cdf(mid, df) < target else (lo, mid)
    return hi


def session_of(data: dict, name: str) -> str:
    """The server process that produced a result: its ``/stats`` instance id, else the file itself."""
    session = data.get("session") or ((data.get("server_stats_start") or {}).get("server") or {})
    return session.get("instance_id") or f"file:{name}"


def load_arm(spec: str) -> dict:
    """One arm's result files grouped by server session: session -> cell key -> {repeat: cell}. Files of the same
    server process merge; a repeat measured twice in one session is an error."""
    arm = {"files": [], "sessions": {}, "gated": {}, "plan": set(), "output_tokens": set()}
    for name in spec.split(","):
        data = json.loads(Path(name).read_text(encoding="utf-8"))
        if data.get("schema") != SCHEMA:
            raise ValueError(f"{name}: not a matrix benchmark result")
        session_id = session_of(data, name)
        arm["files"].append({"path": name, "label": data["label"], "session": session_id,
                             "sha256": sha256(Path(name).read_bytes())})
        arm["plan"].add(data["plan_sha256"])
        arm["output_tokens"].add(data["output_tokens"])
        session = arm["sessions"].setdefault(session_id, {})
        for cell in data["cells"]:
            key = cell_key(cell["mode"], cell["arrival"], cell["c"], cell["length"])
            if cell["status"] != "measured":
                arm["gated"].setdefault(key, f"{cell['status']}: {'; '.join(cell['problems'][:2])}")
                continue
            reps = session.setdefault(key, {})
            if cell["rep"] in reps:
                raise ValueError(f"{name}: duplicate repeat {cell['rep']} of {key} in server session {session_id}")
            reps[cell["rep"]] = cell
        for gated in data.get("gated", []):
            arm["gated"].setdefault(cell_key(gated["mode"], gated["arrival"], gated["c"], gated["length"]),
                                    gated["reason"])
    arm["cells"] = {key for session in arm["sessions"].values() for key in session}
    for key in arm["cells"]:
        arm["gated"].pop(key, None)
    if len(arm["plan"]) != 1 or len(arm["output_tokens"]) != 1:
        raise ValueError(f"arm {spec} mixes plans or output lengths")
    return arm


def rep_values(reps: dict, metric: str) -> dict:
    values = {}
    for rep, cell in reps.items():
        value = cell["metrics"].get(metric)
        if metric == AGGREGATE_DECODE:
            requests = cell.get("requests")
            if metric not in cell["metrics"] and type(requests) is list and len(requests) == cell["c"]:
                value = aggregate_decode_tps(requests)
            if not finite_measurement(value) or value <= 0:
                continue
        if value:
            values[rep] = value
    return values


def describe(sessions: list[dict]) -> dict:
    """Geometric mean of one arm's cell. Each element maps repeat -> value for one server session. With two or more
    sessions the 95% interval is over session means (the session is the unit); with one it is over repeats and
    omits between-process variation."""
    means = [statistics.fmean(math.log(v) for v in values.values()) for values in sessions]
    values = [v for session in sessions for v in session.values()]
    if len(sessions) > 1:
        units, unit = means, "session"
    else:
        units, unit = [math.log(v) for v in values], "repeat"
    mean, n = statistics.fmean(units), len(units)
    half = t_half(0.95, n - 1) * math.sqrt(statistics.variance(units) / n) if n > 1 else math.inf
    return {"n": len(values), "sessions": len(sessions), "unit": unit, "geomean": math.exp(statistics.fmean(means)),
            "ci95": [math.exp(mean - half), math.exp(mean + half)],
            "cv": statistics.stdev(values) / statistics.fmean(values) if len(values) > 1 else None}


def session_ratio(groups: list[list[float]], family: int) -> dict:
    """Candidate/reference ratio with the server session as the unit. ``groups`` holds per-session mean log values:
    [reference sessions, candidate sessions, *calibration groups]. Every group is one identical configuration, so
    the pooled within-group variance of session means (tau^2, sum(m_g - 1) df) carries both between-process and
    within-session repeat noise; se^2 = tau^2 (1/m_ref + 1/m_cand). ``ci95`` is a screen; ``ci_family`` (level
    1 - 0.05 / family) decides the class. Without replicated sessions (df 0) the interval is unbounded."""
    ref, cand = groups[0], groups[1]
    d = statistics.fmean(cand) - statistics.fmean(ref)
    df = sum(len(g) - 1 for g in groups)
    tau2 = sum((x - statistics.fmean(g)) ** 2 for g in groups for x in g) / df if df else math.inf
    se2 = tau2 * (1 / len(ref) + 1 / len(cand))

    def interval(level):
        if not math.isfinite(se2):
            return [0.0, math.inf]
        half = t_half(level, df) * math.sqrt(se2)
        return [math.exp(d - half), math.exp(d + half)]

    def klass(ci):
        return "gain" if ci[0] > 1 else "regression" if ci[1] < 1 else "uncertain"
    screen, decide = interval(0.95), interval(1 - FAMILY_ALPHA / family)
    return {"ratio": math.exp(d), "ci95": screen, "ci_family": decide, "family": family, "log": d,
            "sessions": [len(ref), len(cand)], "tau2": tau2, "se2": se2, "df": df, "design": "session",
            "class": klass(decide), "screen": klass(screen)}


def sort_key(key: str):
    _, mode, arrival, c, length = key.split("/")
    return mode, arrival, int(c[1:]), int(length[1:])


def grid(rows: list[dict], cell, metrics) -> None:
    """Print one Markdown table per (mode, arrival, metric): concurrency rows by prompt-length columns."""
    groups = {}
    for row in rows:
        _, mode, arrival, c, length = row["cell"].split("/")
        groups.setdefault((mode, arrival), (set(), set()))
        groups[(mode, arrival)][0].add(int(c[1:]))
        groups[(mode, arrival)][1].add(int(length[1:]))
    for (mode, arrival), (levels, lengths) in sorted(groups.items()):
        for metric in metrics:
            print(f"\n{mode} / {arrival} / {metric}\n")
            print("| C | " + " | ".join(f"{length // 1024}K" for length in sorted(lengths)) + " |")
            print("| --- |" + " --- |" * len(lengths))
            for c in sorted(levels):
                print(f"| {c} | " + " | ".join(cell(cell_key(mode, arrival, c, length), metric)
                                              for length in sorted(lengths)) + " |")


def summary(args) -> int:
    arm = load_arm(args.arm)
    rows = []
    for key in sorted(arm["cells"] | set(arm["gated"]), key=sort_key):
        if key not in arm["cells"]:
            rows.append({"cell": key, "status": arm["gated"][key]})
            continue
        sessions = [s[key] for s in arm["sessions"].values() if key in s]
        row = {"cell": key, "status": "measured", "reps": sum(len(s) for s in sessions), "sessions": len(sessions)}
        for metric in (*SCORED, AGGREGATE_DECODE, "total_tps", "output_tps", "ttft_max_s", "server_decode_median_tps"):
            values = [rep_values(s, metric) for s in sessions]
            if all(len(v) == len(s) for v, s in zip(values, sessions)):
                row[metric] = describe(values)
        rows.append(row)
    if args.json:
        write_json(args.json, {"schema": SCHEMA + ".summary", "arm": arm["files"], "rows": rows})
    by_key = {row["cell"]: row for row in rows}

    def cell(key, metric):
        row = by_key.get(key)
        if row is None:
            return "not run"
        if row["status"] != "measured":
            return row["status"].split(":")[0]
        value = row.get(metric)
        if value is None:
            return "n/a"
        lo, hi = value["ci95"]
        return f"{value['geomean']:.1f} ({lo:.1f}-{hi:.1f}) n={value['n']}/{value['sessions']}s"
    grid(rows, cell, (*SCORED, AGGREGATE_DECODE))
    return 0


def json_safe(value):
    """Unbounded interval ends and variances (no replicated sessions) are written as null."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    return value


RULE = ("unit: one server session (process). Per (cell, metric) and session: mean log value over the repeats common "
        "to every session scored for that cell (identical prompts). Cell ratio = exp(mean candidate session mean - "
        "mean reference session mean); t interval with se^2 = tau^2 (1/m_ref + 1/m_cand), tau^2 the pooled "
        "within-group variance of session means over the reference arm, the candidate arm and each calibration "
        "group (identical-configuration A/A sessions), df = sum(m_g - 1); deciding level familywise 1 - 0.05/k over "
        "the k compared (cell, metric) pairs (Bonferroni): gain if its lower bound > 1, regression if its upper "
        "bound < 1; 95% per-cell screens are reported, not decisive; df 0 leaves every interval unbounded. "
        "Aggregate per metric: per-session score = mean over the compared cells of its cell means (sessions that "
        "ran every compared cell), same pooling, 95% interval decides. Verdict: any cell or aggregate regression "
        "rejects; superior needs a cell or aggregate gain, no regression and full coverage of reference cells.")


def compare(args) -> int:
    scored = (AGGREGATE_DECODE, *SCORED) if args.aggregate_decode else SCORED
    ref, cand = load_arm(args.reference), load_arm(args.candidate)
    calibration = [load_arm(spec) for spec in args.calibration]
    if ref["plan"] != cand["plan"] or ref["output_tokens"] != cand["output_tokens"]:
        raise ValueError("arms use different prompt plans or output lengths: not comparable")
    for group in calibration:
        if group["plan"] != ref["plan"] or group["output_tokens"] != ref["output_tokens"]:
            raise ValueError(f"calibration group {group['files'][0]['path']} uses another plan or output length")
    arms = [ref, cand, *calibration]
    seen = {}
    for index, group in enumerate(arms):
        for session in group["sessions"]:
            if seen.setdefault(session, index) != index:
                raise ValueError(f"server session {session} appears in two arms/groups: a session is one unit")
    common = sorted(ref["cells"] & cand["cells"], key=sort_key)
    missing = sorted(ref["cells"] - cand["cells"], key=sort_key)
    extra = sorted(cand["cells"] - ref["cells"], key=sort_key)

    def session_means(key, metric):
        """[(group, session, values)] for every session scoring this cell, cut to the repeats they all ran."""
        found = [(g, sid, rep_values(cells[key], metric)) for g, group in enumerate(arms)
                 for sid, cells in group["sessions"].items() if key in cells]
        found = [item for item in found if item[2]]
        if not any(g == 0 for g, _, _ in found) or not any(g == 1 for g, _, _ in found):
            return None, []
        shared = set.intersection(*(set(values) for _, _, values in found))
        if not shared:
            if metric == AGGREGATE_DECODE:
                return None, []
            raise ValueError(f"{key} {metric}: the sessions share no repeat (run every session on the same repeats)")
        return sorted(shared), [(g, sid, statistics.fmean(math.log(values[r]) for r in shared))
                                for g, sid, values in found]

    family = max(1, sum(1 for key in common for metric in scored if session_means(key, metric)[0]))
    rows, per_session = [], {metric: {} for metric in scored}
    for key in common:
        row = {"cell": key}
        for metric in scored:
            shared, means = session_means(key, metric)
            if not shared:
                row[metric] = None
                continue
            groups = [[m for g, _, m in means if g == index] for index in range(len(arms))]
            row[metric] = session_ratio(groups[:2] + [g for g in groups[2:] if g], family)
            row[metric].update(reference=math.exp(statistics.fmean(groups[0])),
                               candidate=math.exp(statistics.fmean(groups[1])), repeats=shared)
            for g, sid, m in means:
                per_session[metric].setdefault((g, sid), {})[key] = m
        rows.append(row)
    aggregate = {}
    for metric in scored:
        cells = [row[metric] for row in rows if row[metric]]
        if not cells:
            continue
        keys = [row["cell"] for row in rows if row[metric]]
        groups = [[] for _ in arms]
        for (g, _), means in per_session[metric].items():
            if all(key in means for key in keys):
                groups[g].append(statistics.fmean(means[key] for key in keys))
        if groups[0] and groups[1]:
            value = session_ratio(groups[:2] + [g for g in groups[2:] if g], 1)
            sessions, ratio_, ci, df = value["sessions"], value["ratio"], value["ci95"], value["df"]
        else:
            sessions, ratio_, ci, df = [len(groups[0]), len(groups[1])], math.exp(
                statistics.fmean(c["log"] for c in cells)), [0.0, math.inf], 0
        worst = min(cells, key=lambda c: c["ratio"])
        aggregate[metric] = {"cells": len(cells), "geomean_ratio": ratio_, "ci95": ci, "sessions": sessions, "df": df,
                             "worst_cell_ratio": worst["ratio"], "worst_cell_ci95": worst["ci95"],
                             "screen_flags": sum(c["screen"] != "uncertain" and c["class"] == "uncertain"
                                                 for c in cells),
                             "gains": sum(c["class"] == "gain" for c in cells),
                             "regressions": sum(c["class"] == "regression" for c in cells),
                             "uncertain": sum(c["class"] == "uncertain" for c in cells)}
    regressions = [(row["cell"], m) for row in rows for m in scored if row[m] and row[m]["class"] == "regression"]
    gains = [(row["cell"], m) for row in rows for m in scored if row[m] and row[m]["class"] == "gain"]
    regressions += [("aggregate", m) for m, value in aggregate.items() if value["ci95"][1] < 1]
    gains += [("aggregate", m) for m, value in aggregate.items() if value["ci95"][0] > 1]
    missing_aggregate = []
    if args.aggregate_decode:
        for row in rows:
            key = row["cell"]
            if row[AGGREGATE_DECODE] is None or any(
                    len(rep_values(session[key], AGGREGATE_DECODE)) != len(session[key])
                    for arm in arms for session in arm["sessions"].values() if key in session):
                missing_aggregate.append(key)
    if missing:
        verdict = "incomplete: the candidate lacks reference-measured cells"
    elif missing_aggregate:
        verdict = "incomplete: aggregate decode is unavailable for some measured repeats"
    elif regressions:
        verdict = ("regression: " + ", ".join(f"{cell} {metric}" for cell, metric in regressions) +
                   " (a gain elsewhere cannot offset it)")
    elif gains:
        verdict = "superior: " + ", ".join(f"{cell} {metric}" for cell, metric in gains) + " gained; no regression"
    else:
        verdict = "uncertain: no established gain or regression"
        if all(value["df"] == 0 for value in aggregate.values()):
            verdict += " (no replicated server sessions: run two or more per arm or pass --calibration)"
    if regressions and gains:
        verdict += "; mixed with gains: adopt only with a REGION-specific selection decision"
    result = {"schema": SCHEMA + ".comparison", "reference": ref["files"], "candidate": cand["files"],
              "calibration": [group["files"] for group in calibration], "rule": RULE, "family_size": family,
              "verdict": verdict, "aggregate": aggregate, "rows": rows,
              "coverage": {"common": len(common), "missing_in_candidate": missing, "only_in_candidate": extra,
                           "missing_aggregate_decode": missing_aggregate,
                           "reference_gated": ref["gated"], "candidate_gated": cand["gated"]}}
    if args.json:
        write_json(args.json, json_safe(result))

    by_key = {row["cell"]: row for row in rows}

    def cell(key, metric):
        row = by_key.get(key)
        if row is None:
            if key in missing:
                return "missing in candidate"
            if key in extra:
                return "candidate only"
            return "not compared"
        value = row[metric]
        if value is None:
            return "n/a"
        mark = {"gain": "+", "regression": "-", "uncertain": "?"}[value["class"]]
        if value["class"] == "uncertain" and value["screen"] != "uncertain":
            mark += "!"  # the 95% screen alone flags it: more sessions needed
        lo, hi = value["ci_family"]
        return f"{value['ratio']:.3f} ({lo:.3f}-{hi:.3f}) {mark}"
    grid([{"cell": key} for key in set(common) | set(missing) | set(extra)], cell, scored)
    print(f"sessions: reference {len(ref['sessions'])}, candidate {len(cand['sessions'])}, calibration groups "
          + (", ".join(str(len(group["sessions"])) for group in calibration) or "none"))
    for metric, value in aggregate.items():
        print(f"{metric}: geomean {value['geomean_ratio']:.3f} ({value['ci95'][0]:.3f}-{value['ci95'][1]:.3f}) over "
              f"{value['cells']} cells, sessions {value['sessions'][0]}/{value['sessions'][1]}, df {value['df']}; "
              f"worst {value['worst_cell_ratio']:.3f}; screen-only flags {value['screen_flags']}; gains "
              f"{value['gains']}, regressions {value['regressions']}, uncertain {value['uncertain']}")
    print(f"coverage: {len(common)} common, {len(missing)} missing in candidate, {len(extra)} only in candidate")
    for key in missing:
        print(f"  missing in candidate: {key} ({cand['gated'].get(key, 'not run')})")
    for key in extra:
        print(f"  only in candidate: {key} ({ref['gated'].get(key, 'not run')})")
    print(f"cell intervals: familywise {1 - FAMILY_ALPHA / family:.5f} over {family} (cell, metric) pairs; "
          "+ gain, - regression, ? uncertain, ! flagged by the per-cell 95% screen only")
    print("verdict:", verdict)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("corpus", help="freeze the prompt corpus from a pinned commit")
    p.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--commit", required=True)
    p.add_argument("--tokenizer", type=Path, required=True, help="model directory holding tokenizer.json")
    p.add_argument("--out", type=Path, required=True)

    p = sub.add_parser("plan", help="cut exact-length unique prompts (needs transformers)")
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--tokenizer", type=Path, required=True)
    p.add_argument("--template", type=Path, required=True, help="the served chat template")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--modes", type=word_list(MODES), default=list(MODES))
    p.add_argument("--concurrency", type=int_list, default=[1, 2, 4, 8, 16, 24, 32])
    p.add_argument("--lengths", type=int_list, default=[4096, 8192, 16384, 32768, 65536])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--stagger-ms", type=float, nargs="*", default=[0.0])
    p.add_argument("--jobs", type=int, default=8, help="tokenizer worker processes")

    p = sub.add_parser("run", help="measure the matrix against a live server")
    p.add_argument("--url", default="http://127.0.0.1:8000")
    p.add_argument("--model", required=True)
    p.add_argument("--plan", type=Path, required=True)
    p.add_argument("--corpus", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--label", required=True, help="engine/source/configuration identity of this arm")
    p.add_argument("--note", default="", help="free text: launch command, image, flags")
    p.add_argument("--source-dir", type=Path, help="served source tree to digest into the result")
    p.add_argument("--modes", type=word_list(MODES), default=list(MODES))
    p.add_argument("--concurrency", type=int_list, default=[1, 2, 4, 8, 16, 24, 32])
    p.add_argument("--lengths", type=int_list, default=[4096, 8192, 16384, 32768, 65536])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--first-rep", type=int, default=0, help="first repeat index (continue a server session in another file; every scored session "
                        "must run the same repeats)")
    p.add_argument("--warmup-lengths", type=int_list, help="C=1 warmup requests per mode (default: every length)")
    p.add_argument("--output-tokens", type=int, default=512)
    p.add_argument("--stagger-ms", type=float, default=0.0, help="start request i at i x this many ms")
    p.add_argument("--lanes", type=int, help="concurrent request slots when the server has no /stats")
    p.add_argument("--kv-tokens", type=int, help="KV token budget: gate cells whose C x (L + output) exceeds it")
    p.add_argument("--allow-prompt-token-drift", action="store_true",
                   help="accept a server whose prompt rendering differs from the plan's token count (recorded)")
    p.add_argument("--timeout", type=float, default=900, help="socket inactivity timeout, seconds")
    p.add_argument("--deadline", type=float, default=3600, help="whole-request deadline, seconds")
    p.add_argument("--fail-fast", action="store_true")
    p.add_argument("--retry-overloaded", type=int, default=0, metavar="N",
                   help="resend a request refused with HTTP 503 up to N times (time counts from the first send; "
                        "each request records its refusals)")
    p.add_argument("--retry-delay", type=float, default=0.05, help="seconds between --retry-overloaded resends")
    protocol.add_arguments(p)

    p = sub.add_parser("summary", help="per-cell table of one arm")
    p.add_argument("arm", help="result file(s), comma-separated")
    p.add_argument("--json", type=Path)

    p = sub.add_parser("compare", help="score a candidate arm against a reference arm")
    p.add_argument("reference", help="reference result file(s), comma-separated")
    p.add_argument("candidate", help="candidate result file(s), comma-separated")
    p.add_argument("--calibration", action="append", default=[], metavar="FILES",
                   help="comma-separated result files of one identical-configuration A/A group (repeatable): extra "
                        "server sessions for the between-session variance")
    p.add_argument("--aggregate-decode", action="store_true",
                   help="also score aggregate decode from each repeat's client-observed cohort interval")
    p.add_argument("--json", type=Path)

    args = parser.parse_args()
    if args.command == "run" and (args.reps < 1 or args.first_rep < 0 or args.output_tokens < 2):
        parser.error("run needs --reps >= 1, --first-rep >= 0 and --output-tokens >= 2")
    return {"corpus": build_corpus, "plan": build_plan, "run": run, "summary": summary, "compare": compare}[
        args.command](args)


if __name__ == "__main__":
    sys.exit(main())

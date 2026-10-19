"""Campaign benchmark of a TensorFold OpenAI server: fixed prompt depths, model-card sampling and greedy, per-request
engine stats, and interleaved A/B analysis with drift-cancelling per-cycle ratios. Standard library only.

  corpus   OUT SOURCE_DIR...             the fixed text corpus (.py and .md files, sources interleaved) + its sha256
  prompts  URL PROFILE CORPUS OUT        size one prompt per (depth, task, thinking) with the server's /v1/tokenize
  run      URL PROFILE PROMPTS OUT       one position: each prompt cold, then warm repeats; raw records + environment
  analyze  RUNS_DIR PREFIX [--base A]    per-cell summaries and per-cycle ratios against the base variant

A PROFILE (JSON) carries the model's facts: served name, context, reply budget, end ids, depths, workloads, thinking
modes, model-card sampling per thinking mode and the per-prompt request schedule. Run directories are named
PREFIX-c<cycle>-<position>-<variant>/bench.json (the GB10 runner's layout). Every summary interval is a two-sided 95%
Student t interval over independent units (positions for absolute values, cycles for ratios); a cell with one unit
reports no interval. The analysis lists runs that stopped mid-schedule, replies whose token count differs from the
profile's max_tokens, and the campaign_bench.py versions (sha256, recorded per run) a series was measured with.

Metrics per request: engine decode tok/s (the ``tensorfold`` block's decode_tps = (tokens - 1) / decode_s), wall
decode tok/s (SSE arrival of the first and last text pieces; a piece can carry several tokens), prefill tok/s
((prompt - cached) / prefill_s, cold requests), server TTFT (arrival to first token) and client TTFT (send to first
piece), queue_s, DFlash acceptance (accepted / drafted) and tokens per round, minimum host MemAvailable while the
request ran (sampled every 0.25 s from /proc/meminfo; unified memory on the GB10 includes GPU allocations).
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import threading
import time
from typing import ClassVar
import urllib.error
import urllib.request

if sys.version_info < (3, 11):
    print(f"Python 3.11+ is required (this is {sys.version.split()[0]}); "
          "TensorFold's tools use PEP 604 unions. Try: python3.12 tools/campaign_bench.py ...", file=sys.stderr)
    sys.exit(4)                     # an error, never a verdict (campaign_gates and campaign_kl import this)

if __package__:
    from . import openai_protocol as protocol
    from .benchmark_output import write_json
else:
    import openai_protocol as protocol
    from benchmark_output import write_json

VERSION = 1
LIMITS = protocol.Limits(response_bytes=256 << 20, line_bytes=64 << 20)   # token ids ride the final frame
TIMEOUT_S = 7200                                                            # a cold full-context prefill takes minutes

# Fixed task instructions: part of the prompt-set identity (their text is hashed into every prompt's sha256).
TASKS = {
    "code": "Write a complete, production-quality Python module that implements a thread-safe LRU cache with "
            "per-entry TTL expiry, size accounting, type hints and docstrings, followed by a thorough pytest test "
            "suite. Output only code.",
    "prose": "Write a long, detailed essay of at least 1500 words about the history of the printing press and its "
             "effects on European society, science and religion.",
    "doc-code": "Using the files above as a style reference, write a new, complete Python module of at least 400 "
                "lines that implements a command-line tool for analysing such a source tree: count definitions per "
                "file, build a call graph, and print a report. Output only code.",
    "doc-review": "Write a detailed technical review of the files above: for each file in order, explain its "
                  "purpose, its main classes and functions, and any bugs or risky constructs you notice. Be "
                  "exhaustive.",
}

WARMUP = "Warm-up request, not measured. " + TASKS["code"]

# The campaign's hold on sustained long prefills (after the 2026-10-18 GB10 hang at the start of a 32k prefill;
# lifted to 65,536 tokens on 2026-10-19 once the 32k controls passed): a prompt above this many tokens (the 128k and
# full-context items) runs only with --allow-long-prefill; held work is reported as held, never dropped silently.
LONG_PREFILL_TOKENS = 65536

T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
       2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042)


def t95(df: int) -> float:
    """Two-sided 95% Student t quantile; beyond the table, the Cornish-Fisher expansion (error < 1e-3)."""

    if df < 1:
        raise ValueError("a t interval needs at least two units")
    if df <= len(T95):
        return T95[df - 1]
    z = 1.959964
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df ** 2)


def interval(values: list[float]) -> dict:
    """Mean and the two-sided 95% t half-width over independent units (no half-width for one unit)."""

    n = len(values)
    mean = statistics.fmean(values)
    half = t95(n - 1) * statistics.stdev(values) / math.sqrt(n) if n > 1 else None
    return {"n": n, "mean": mean, "half95": half, "min": min(values), "max": max(values)}


# ---------------------------------------------------------------------------------------------------------------
# HTTP


def post_json(url: str, body: dict, *, timeout: float = 600) -> dict:
    """POST a JSON body; an HTTP error raises with the server's error message (it says why, e.g. a refusal)."""

    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            if resp.status != 200:
                raise ValueError(f"HTTP {resp.status} from {url}")
            return protocol.object_json(resp.read())
    except urllib.error.HTTPError as error:
        detail = error.read(4096).decode("utf-8", "replace")
        raise ValueError(f"HTTP {error.code} from {url}: {detail}") from error


def get_json(url: str, *, timeout: float = 30) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as resp:
        return json.loads(resp.read())


class Progress:
    """The heartbeat a supervisor (the GB10 runner's watchdog) reads while a tool talks to a server: one small JSON
    file, replaced atomically, holding the current phase ("prefill" until the first generated text, reasoning or
    tool-call delta of a stream, "decode" after it, "wait" for a non-streamed reply, "idle" between requests), when
    that phase started, the last event's time, the request in flight (``request``, ``prompt_tokens``, and for "wait"
    ``max_tokens``; kept across that request's phases) and its chunk count. Facts only: the supervisor owns the
    stall policy. Writes are rate-limited to one per second except at phase changes. One per process (tools here are
    single-threaded CLIs): ``configure`` sets the path and resets the state, ``configure(None)`` turns it off again;
    unconfigured, every call is a no-op."""

    path: ClassVar[str | None] = None
    state: ClassVar[dict] = {}
    written: ClassVar[float] = 0.0
    CARRIED: ClassVar[tuple[str, ...]] = ("request", "prompt_tokens", "max_tokens")   # one request's identity

    @classmethod
    def configure(cls, path: str | None) -> None:
        cls.path, cls.state, cls.written = path, {}, 0.0
        cls.update("idle", force=True)

    @classmethod
    def update(cls, phase: str, *, force: bool = False, **fields) -> None:
        if cls.path is None:
            return
        now = time.time()
        if phase != cls.state.get("phase"):
            carried = {k: cls.state[k] for k in cls.CARRIED if k in cls.state and phase != "idle"}
            cls.state = {"phase": phase, "phase_since": now, **carried}
            force = True
        cls.state.update(fields, last_event=now, pid=os.getpid())
        if force or now - cls.written >= 1.0:
            temporary = f"{cls.path}.{os.getpid()}.tmp"
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(cls.state, f)
            os.replace(temporary, cls.path)
            cls.written = now


def expected_tokens(body: dict) -> int:
    """A request's prompt size for the watchdog's prefill allowance: token ids as given, else a generous estimate of
    one token per two characters of its messages or prompt text."""

    prompt = body.get("prompt")
    if isinstance(prompt, list):
        return len(prompt)
    text = prompt if isinstance(prompt, str) else json.dumps(body.get("messages", ""))
    return len(text) // 2


def stream_chat(base: str, body: dict, *, limits: protocol.Limits = LIMITS, timeout: float = TIMEOUT_S,
                route: str = "/v1/chat/completions", prompt_tokens: int | None = None, label: str = "") -> dict:
    """One streamed completion (chat by default; ``route="/v1/completions"`` for a text or token-id ``prompt``):
    client timings, text, reasoning, the final usage and ``tensorfold`` block. Reports its phase to ``Progress``
    (``label`` names the request; ``prompt_tokens``, when known, sizes the supervisor's prefill allowance, else
    ``expected_tokens``); the heartbeat is written before the client clock starts."""

    body = {**body, "stream": True, "stream_options": {"include_usage": True}}
    req = urllib.request.Request(base + route, data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    Progress.update("prefill", request=label or route,
                    prompt_tokens=prompt_tokens if prompt_tokens is not None else expected_tokens(body), chunks=0)
    sent = time.perf_counter()
    try:
        return _read_stream(req, timeout, limits, sent)
    finally:
        Progress.update("idle", force=True)


def _read_stream(req, timeout: float, limits: protocol.Limits, sent: float) -> dict:
    """stream_chat's reading and timing of one open request (``sent``: the client clock's start)."""

    first = last = None
    usage = runtime = finish = None
    content, reasoning, calls = [], [], []
    pieces = 0
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        protocol.response_type(resp, "text/event-stream")
        headers = time.perf_counter()
        for chunk in protocol.sse_objects(resp, limits=limits):
            new_usage, new_runtime, texts = protocol.chunk_fields(chunk)
            usage = new_usage if new_usage is not None else usage
            runtime = new_runtime if new_runtime is not None else runtime
            called = False
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                finish = choice.get("finish_reason") or finish
                if text := delta.get("content") or choice.get("text"):
                    content.append(text)
                if thought := delta.get("reasoning_content") or delta.get("reasoning"):
                    reasoning.append(thought)
                if delta.get("tool_calls"):
                    calls.extend(delta["tool_calls"])
                    called = True
            if any(texts):
                now = time.perf_counter()
                first = now if first is None else first
                last = now
                pieces += sum(1 for t in texts if t)
            # progress for the watchdog: generated text, reasoning or call deltas (the role chunk opens the stream
            # before the prefill and is no progress)
            if any(texts) or called:
                Progress.update("decode", chunks=Progress.state.get("chunks", 0) + 1)
    done = time.perf_counter()
    if usage is None or runtime is None:
        raise ValueError("stream ended without usage and the tensorfold block (include_usage)")
    return {"sent": sent, "headers": headers, "first": first, "last": last, "done": done, "pieces": pieces,
            "usage": usage, "stats": runtime, "finish": finish, "content": "".join(content),
            "reasoning": "".join(reasoning), "tool_call_deltas": calls}


# ---------------------------------------------------------------------------------------------------------------
# Host observation (GB10 host side; every read is optional and recorded as None when absent)


def _read(path: str) -> str | None:
    try:
        with open(path, encoding="ascii") as f:
            return f.read()
    except OSError:
        return None


def mem_available_mib() -> int | None:
    text = _read("/proc/meminfo")
    match = re.search(r"^MemAvailable:\s+(\d+) kB", text or "", re.M)
    return int(match.group(1)) // 1024 if match else None


GPU_FIELDS = ("temperature.gpu,power.draw,clocks.sm,clocks.mem,utilization.gpu,clocks_event_reasons.active,"
              "clocks_event_reasons.hw_thermal_slowdown,clocks_event_reasons.sw_thermal_slowdown,"
              "clocks_event_reasons.hw_power_brake_slowdown,clocks_event_reasons.sw_power_cap")


def host_snapshot(*, gpu: bool = False) -> dict:
    """Load, memory, thermal zones and (between requests only: it spawns a process) nvidia-smi's GPU state."""

    zones = {}
    for zone in sorted(glob.glob("/sys/class/thermal/thermal_zone*/temp")):
        value = _read(zone)
        if value and value.strip().lstrip("-").isdigit():
            zones[zone.split("/")[-2]] = int(value) / 1000
    out = {"unix_s": round(time.time(), 3), "loadavg": (_read("/proc/loadavg") or "").split()[:3],
           "mem_available_mib": mem_available_mib(), "thermal_c": zones}
    if gpu:
        try:
            query = subprocess.run(["nvidia-smi", f"--query-gpu={GPU_FIELDS}", "--format=csv,noheader"],
                                   capture_output=True, text=True, timeout=20, check=False)
            out["nvidia_smi"] = query.stdout.strip() if query.returncode == 0 else f"exit {query.returncode}"
            out["nvidia_smi_fields"] = GPU_FIELDS
        except (OSError, subprocess.SubprocessError) as error:
            out["nvidia_smi"] = f"unavailable: {type(error).__name__}"
    return out


class MemSampler:
    """Minimum MemAvailable while a request runs, read every ``period`` seconds on a daemon thread."""

    def __init__(self, period: float = 0.25):
        self.period, self.low, self._stop = period, None, threading.Event()
        self._thread = threading.Thread(target=self._loop, name="memsampler", daemon=True)

    def _loop(self) -> None:
        while not self._stop.is_set():
            value = mem_available_mib()
            if value is not None:
                self.low = value if self.low is None else min(self.low, value)
            self._stop.wait(self.period)

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()


# ---------------------------------------------------------------------------------------------------------------
# corpus


def corpus(out: str, sources: list[str], max_bytes: int) -> None:
    """Interleave each source directory's .py/.md files (sorted relative paths), one file at a time per source."""

    lists = []
    for source in sources:
        root = Path(source)
        files = sorted(p for p in root.rglob("*") if p.suffix in (".py", ".md") and p.is_file()
                       and not any(part in ("test", "tests", "idlelib", "site-packages", "__pycache__")
                                   for part in p.relative_to(root).parts))
        lists.append((root, files))
    parts, size, index = [], 0, 0
    while size < max_bytes and any(index < len(files) for _, files in lists):
        for root, files in lists:
            if index < len(files):
                text = files[index].read_text(encoding="utf-8", errors="replace")
                block = f"### file: {root.name}/{files[index].relative_to(root)}\n{text}\n"
                parts.append(block)
                size += len(block.encode())
        index += 1
    data = "".join(parts).encode()[:max_bytes].decode("utf-8", errors="ignore")
    Path(out).write_text(data, encoding="utf-8")
    digest = hashlib.sha256(data.encode()).hexdigest()
    Path(out + ".sha256").write_text(f"{digest}  {Path(out).name}\n")
    print(json.dumps({"corpus": out, "bytes": len(data.encode()), "sha256": digest}))


def corpus_text(path: str) -> tuple[str, str]:
    data = Path(path).read_text(encoding="utf-8")
    return data, hashlib.sha256(data.encode()).hexdigest()


def file_sha256(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def reply_token_ids(record: dict) -> list[int]:
    """A streamed reply's ids from its ``tensorfold`` block (the request sent ``return_token_ids``). Missing ids or
    a count other than ``usage.completion_tokens`` is an error, never an empty list that compares equal to another."""

    ids, n = record["stats"].get("token_ids"), record["usage"].get("completion_tokens")
    if not isinstance(ids, list) or len(ids) != n or not all(type(t) is int for t in ids):
        raise ValueError(f"reply token ids ({'absent' if ids is None else f'{len(ids)} values'}) do not match "
                         f"{n} completion tokens; the server must honour return_token_ids")
    return ids


# ---------------------------------------------------------------------------------------------------------------
# prompts


def _messages(depth: str, task: str, document: str | None) -> list[dict]:
    if document is None:
        return [{"role": "user", "content": TASKS[task]}]
    return [{"role": "user", "content": f"Benchmark session {depth}-{task}.\n<document>\n{document}\n</document>\n\n"
                                        f"{TASKS[task]}"}]


def depth_target(profile: dict, depth: str) -> int:
    value = profile["depths"][depth]
    if value == "full":                     # the advertised context less the reply budget and a template margin
        return int(profile["context"]) - int(profile["max_tokens"]) - int(profile.get("full_margin", 64))
    return int(value)


def size_prompt(count, depth: str, task: str, text: str, target: int, tolerance: int = 32) -> tuple[list, int]:
    """The most corpus characters whose rendered prompt stays within ``target`` tokens (to ``tolerance``)."""

    lo, hi = 0, min(len(text), target * 8)
    lo_count = count(_messages(depth, task, ""))
    if lo_count > target:
        raise ValueError(f"{depth}/{task}: the empty document already renders {lo_count} > {target} tokens")
    guess = min(hi, max(1, int(target * 3.5)))
    best = (0, lo_count)
    for _ in range(40):
        n = count(_messages(depth, task, text[:guess]))
        if n <= target:
            best = max(best, (guess, n))
            lo = guess
            if target - n <= tolerance:
                break
        else:
            hi = guess - 1
        if lo >= hi:
            break
        # secant step through the best point below and the current point, kept inside (lo, hi]
        rate = max(1e-3, (n - lo_count) / max(1, guess))
        step = int(guess + (target - tolerance / 2 - n) / rate)
        guess = min(hi, max(lo + 1, step))
    chars, n = best
    return _messages(depth, task, text[:chars]), n


def prompts(url: str, profile_path: str, corpus_path: str, out: str) -> None:
    profile = json.loads(Path(profile_path).read_text())
    text, digest = corpus_text(corpus_path)
    stats = get_json(url + "/v1/stats")
    items = []
    for depth, task in profile["workloads"]:
        for thinking in profile["thinking"]:
            kwargs = {"enable_thinking": bool(thinking)}

            def count(messages):
                return post_json(url + "/v1/tokenize", {"model": profile["model"], "messages": messages,
                                                        "chat_template_kwargs": kwargs})["count"]

            target = depth_target(profile, depth)
            if target == 0:
                messages = _messages(depth, task, None)
                n = count(messages)
            else:
                messages, n = size_prompt(count, depth, task, text, target)
            key = json.dumps({"messages": messages, "chat_template_kwargs": kwargs}, sort_keys=True)
            item = {"id": f"{depth}-{task}-{'think' if thinking else 'nothink'}", "depth": depth,
                    "target": target, "task": task, "thinking": bool(thinking),
                    "messages": messages, "prompt_tokens": n, "sha256": hashlib.sha256(key.encode()).hexdigest()}
            items.append(item)
            print(json.dumps({k: item[k] for k in ("id", "target", "prompt_tokens")}), flush=True)
    write_json(out, {"version": VERSION, "model": profile["model"], "corpus_sha256": digest,
                     "profile_sha256": file_sha256(profile_path),
                     "server_source_revision": stats.get("server", {}).get("source_revision"),
                     "tasks": TASKS, "items": items})


# ---------------------------------------------------------------------------------------------------------------
# run


def chat_request(profile: dict, item: dict, sampling: dict, max_tokens: int, **extra) -> dict:
    """A prompt-set item's chat request with token ids (bench and gates); ``sampling`` then ``extra`` fields last."""

    return {"model": profile["model"], "messages": item["messages"], "max_tokens": max_tokens,
            "chat_template_kwargs": {"enable_thinking": item["thinking"]}, "return_token_ids": True,
            "ignore_eos": bool(profile.get("ignore_eos", True)), **sampling, **extra}


def request_body(profile: dict, item: dict, step: dict) -> dict:
    """A schedule step's request: greedy, or the model card's sampling for the prompt's thinking mode plus a seed."""

    if step["sampling"] == "greedy":
        sampling = {"temperature": 0}
    else:
        sampling = {**profile["card_sampling"]["think" if item["thinking"] else "nothink"], "seed": int(step["seed"])}
    extra = {"draft": bool(step["draft"])} if "draft" in step else {}
    return chat_request(profile, item, sampling, int(profile["max_tokens"]), **extra)


def derived(record: dict) -> dict:
    """The per-request metrics the analysis reads, from the raw stream record."""

    s, usage = record["stats"], record["usage"]
    n, prompt = usage.get("completion_tokens", 0), usage.get("prompt_tokens", 0)
    cached = s.get("cached") or 0
    return {"tokens": n, "prompt_tokens": prompt, "cached": cached, "queue_s": s.get("queue_s"),
            "decode_tps": s.get("decode_tps"), "ttft_s": s.get("ttft_s"),
            "ttft_client_s": record["first"] - record["sent"] if record["first"] is not None else None,
            "decode_tps_wall": (n - 1) / (record["last"] - record["first"])
            if record["first"] is not None and record["last"] > record["first"] and n > 1 else None,
            "prefill_tps": (prompt - cached) / s["prefill_s"] if s.get("prefill_s") and prompt > cached else None,
            "accept": s["accepted"] / s["drafted"] if s.get("drafted") else None,
            "tokens_per_round": (n - 1) / s["rounds"] if s.get("rounds") else None}


def run(url: str, profile_path: str, prompts_path: str, out: str, *, meta: str | None, label: str,
        select: dict) -> None:
    profile = json.loads(Path(profile_path).read_text())
    prompt_set = json.loads(Path(prompts_path).read_text())
    items = [it for it in prompt_set["items"]
             if (not select["depths"] or it["depth"] in select["depths"])
             and (not select["tasks"] or it["task"] in select["tasks"])
             and (select["thinking"] is None or it["thinking"] == select["thinking"])]
    held = [it["id"] for it in items if it["prompt_tokens"] > LONG_PREFILL_TOKENS and not select.get("allow_long")]
    items = [it for it in items if it["id"] not in held]
    for name in held:
        print(json.dumps({"prompt": name, "held": "long-prefill hold"}), flush=True)
    if not items:
        raise ValueError("the selection leaves no prompts" + (" (all held by the long-prefill hold)" if held else ""))
    eos = set(profile.get("eos_token_ids", ()))
    doc = {"version": VERSION, "label": label, "url": url, "model": profile["model"],
           "tool_sha256": file_sha256(__file__), "held": held, "profile": profile,
           "profile_sha256": file_sha256(profile_path),
           "prompts_file_sha256": file_sha256(prompts_path),
           "corpus_sha256": prompt_set.get("corpus_sha256"),
           "meta": json.loads(Path(meta).read_text()) if meta else None,
           "server_stats_before": get_json(url + "/v1/stats"), "host_before": host_snapshot(gpu=True),
           "records": []}
    # warm-up past the first KV growth step and through every verify width, so one-time kernel compilation and
    # allocation land outside the measured requests; its own text, so no measured prompt starts with it (cache)
    warm = stream_chat(url, {"model": profile["model"], "messages": [{"role": "user", "content": WARMUP}],
                             "max_tokens": int(profile.get("warmup_tokens", 640)), "temperature": 0,
                             "ignore_eos": True, "chat_template_kwargs": {"enable_thinking": False}})
    doc["warmup"] = {"stats": warm["stats"], "usage": warm["usage"]}
    for item in items:
        for index, step in enumerate(profile["schedule"]):
            body = request_body(profile, item, step)
            before = host_snapshot()
            with MemSampler() as sampler:
                record = stream_chat(url, body, prompt_tokens=item["prompt_tokens"],
                                     label=f"bench {item['id']} step {index}")
            ids = reply_token_ids(record)
            row = {"prompt": item["id"], "depth": item["depth"], "task": item["task"], "thinking": item["thinking"],
                   "step": index, "sampling": step["sampling"], "seed": body.get("seed"),
                   "draft": body.get("draft", True), "phase": "cold" if index == 0 else "warm",
                   "expected_prompt_tokens": item["prompt_tokens"], "finish": record["finish"],
                   "eos_at": next((i for i, t in enumerate(ids) if t in eos), None),
                   "usage": record["usage"], "stats": {k: v for k, v in record["stats"].items() if k != "token_ids"},
                   "token_ids": ids, "client": {k: record[k] - record["sent"] if record[k] is not None else None
                                                for k in ("headers", "first", "last", "done")},
                   "pieces": record["pieces"], "mem_min_available_mib": sampler.low,
                   "host": {"loadavg": before["loadavg"], "mem_available_mib": before["mem_available_mib"],
                            "thermal_c": before["thermal_c"]},
                   "metrics": derived(record)}
            doc["records"].append(row)
            m = row["metrics"]
            print(json.dumps({"prompt": item["id"], "step": index, "sampling": step["sampling"],
                              "prompt_tokens": m["prompt_tokens"], "cached": m["cached"], "tokens": m["tokens"],
                              "decode_tps": m["decode_tps"],
                              "prefill_tps": m["prefill_tps"] and round(m["prefill_tps"], 1),
                              "ttft_s": m["ttft_s"], "accept": m["accept"] and round(m["accept"], 3),
                              "sha": record["stats"].get("token_sha")}), flush=True)
            write_json(out, doc)                                   # each request kept as it lands
    doc["server_stats_after"] = get_json(url + "/v1/stats")
    doc["host_after"] = host_snapshot(gpu=True)
    write_json(out, doc)


# ---------------------------------------------------------------------------------------------------------------
# analyze

# A cold request counts for prefill/TTFT only when at most this share of its prompt was resumed from the prefix
# cache (a prompt-end state of the other thinking mode's prompt can cover all but its last few tokens).
COLD_CACHED_SHARE = 0.1
DECODE_METRICS = ("decode_tps", "decode_tps_wall", "accept", "tokens_per_round", "queue_s")
PREFILL_METRICS = ("prefill_tps", "ttft_s", "ttft_client_s")


def load_runs(runs: str, prefix: str) -> list[dict]:
    """Every PREFIX-c<cycle>-<position>-<variant>/bench.json, with its cycle, position and variant."""

    out = []
    pattern = re.compile(rf"^{re.escape(prefix)}-c(\d+)-(\d+)-(.+)$")
    for path in sorted(glob.glob(os.path.join(runs, prefix + "-c*-*", "bench.json"))):
        match = pattern.match(os.path.basename(os.path.dirname(path)))
        if not match:
            continue
        doc = json.loads(Path(path).read_text())
        out.append({"cycle": int(match.group(1)), "position": int(match.group(2)), "variant": match.group(3),
                    "path": path, "doc": doc})
    return out


def cell_values(doc: dict) -> dict:
    """Per position: the mean of each metric per cell. Decode cells are (depth, thinking, sampling) over every
    request (workload cells split them by task); prefill cells are (depth, thinking) over cold requests (see
    COLD_CACHED_SHARE). A cell value is the mean over its requests (memory cells: the minimum)."""

    cells: dict[tuple, dict[str, list[float]]] = {}
    for r in doc["records"]:
        think = "think" if r["thinking"] else "nothink"
        groups = [(("decode", r["depth"], think, r["sampling"]), DECODE_METRICS),
                  (("workload", r["depth"], r["task"], think, r["sampling"]), DECODE_METRICS)]
        prompt = r["usage"].get("prompt_tokens") or 0
        if r["phase"] == "cold" and prompt and r["metrics"]["cached"] <= COLD_CACHED_SHARE * prompt:
            groups.append((("prefill", r["depth"], think), PREFILL_METRICS))
        groups.append((("memory", r["depth"], think), ("mem_min_available_mib",)))
        for key, metrics in groups:
            bucket = cells.setdefault(key, {})
            for metric in metrics:
                value = r["metrics"].get(metric) if metric != "mem_min_available_mib" else r[metric]
                if value is not None:
                    bucket.setdefault(metric, []).append(float(value))
    # memory cells keep the position's lowest reading (the safety margin), every other metric its mean
    return {key: {m: min(v) if m == "mem_min_available_mib" else statistics.fmean(v) for m, v in metrics.items()}
            for key, metrics in cells.items()}


def first_difference(a: list[int], b: list[int]) -> int | None:
    """None for identical token sequences, else the first differing index (the shorter length for a prefix)."""

    if a == b:
        return None
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


def run_name(run: dict) -> str:
    return f"c{run['cycle']}-{run['position']}-{run['variant']}"


def identity(runs: list[dict]) -> dict:
    """Token identity of every repeated request (same prompt, sampling, seed, draft) across all positions, and the
    first diverging index of any differing pair."""

    seen: dict[tuple, list] = {}
    for run in runs:
        for r in run["doc"]["records"]:
            key = (r["prompt"], r["sampling"], r["seed"], r.get("draft", True))
            seen.setdefault(key, []).append((run["variant"], run["cycle"], run["position"], r["step"],
                                             r["stats"].get("token_sha"), r["token_ids"]))
    report = {"keys": len(seen), "identical": 0, "differing": []}
    for key, rows in sorted(seen.items(), key=lambda kv: str(kv[0])):
        shas = {row[4] for row in rows}
        if len({tuple(row[5]) for row in rows}) == 1 and len(shas) == 1:      # the ids decide, not the hash alone
            report["identical"] += 1
            continue
        ref = rows[0][5]
        divergence = [{"variant": row[0], "cycle": row[1], "position": row[2], "step": row[3],
                       "first_divergence": first_difference(ref, row[5])} for row in rows[1:]]
        report["differing"].append({"prompt": key[0], "sampling": key[1], "seed": key[2], "draft": key[3],
                                    "shas": sorted(s or "" for s in shas), "against_first": divergence})
    return report


def analyze(runs_dir: str, prefix: str, base: str | None) -> dict:
    runs = load_runs(runs_dir, prefix)
    if not runs:
        raise ValueError(f"no {prefix}-c*-*/bench.json under {runs_dir}")
    variants = sorted({r["variant"] for r in runs})
    base = base or ("base" if "base" in variants else variants[0])
    if base not in variants:
        raise ValueError(f"base variant {base!r} has no runs (variants: {', '.join(variants)})")
    values = [(r, cell_values(r["doc"])) for r in runs]
    keys = sorted({k for _, cells in values for k in cells}, key=str)
    cycles = sorted({r["cycle"] for r in runs})

    def units(key, metric, variant, cycle=None) -> list[float]:
        """The cell values of ``variant`` (in ``cycle`` when given), one per position that has the metric."""

        return [c[key][metric] for r, c in values if r["variant"] == variant
                and (cycle is None or r["cycle"] == cycle) and metric in c.get(key, {})]

    absolute, ratios = {}, {}
    for key in keys:
        name = "/".join(key)
        key_metrics = sorted({m for _, c in values for m in c.get(key, {})})
        for variant in variants:
            metrics = {metric: interval(xs) for metric in key_metrics if (xs := units(key, metric, variant))}
            if metrics:
                absolute.setdefault(name, {})[variant] = metrics
        for variant in variants:
            if variant == base:
                continue
            for metric in key_metrics:
                per_cycle = []
                for cycle in cycles:
                    a, b = units(key, metric, base, cycle), units(key, metric, variant, cycle)
                    if a and b and statistics.fmean(a) != 0:
                        per_cycle.append(statistics.fmean(b) / statistics.fmean(a))
                if per_cycle:
                    summary = interval(per_cycle)
                    summary["geomean"] = math.exp(statistics.fmean(math.log(x) for x in per_cycle)) \
                        if all(x > 0 for x in per_cycle) else None
                    summary["ratios"] = [round(x, 5) for x in per_cycle]
                    summary["below_1"] = sum(x < 1 for x in per_cycle)
                    ratios.setdefault(name, {}).setdefault(variant, {})[metric] = summary
    meta = {}
    for r in runs:
        doc = r["doc"]
        meta[run_name(r)] = {
            "source_revision": doc.get("server_stats_before", {}).get("server", {}).get("source_revision"),
            "image": (doc.get("meta") or {}).get("image_id"), "args": (doc.get("meta") or {}).get("args"),
            "records": len(doc["records"]), "complete": "host_after" in doc, "tool_sha256": doc.get("tool_sha256"),
            "corpus_sha256": doc.get("corpus_sha256"), "prompts_file_sha256": doc.get("prompts_file_sha256")}
    incomplete = sorted(name for name, m in meta.items() if not m["complete"])     # the run stopped mid-schedule
    tool_versions = sorted({str(m["tool_sha256"]) for m in meta.values()})
    off_schedule = [{"run": run_name(r), "prompt": x["prompt"], "step": x["step"],
                     "tokens": x["usage"].get("completion_tokens"), "scheduled": r["doc"]["profile"]["max_tokens"]}
                    for r in runs if "profile" in r["doc"] for x in r["doc"]["records"]
                    if x["usage"].get("completion_tokens") != r["doc"]["profile"]["max_tokens"]]
    prompt_check = [{"run": run_name(r), "prompt": x["prompt"],
                     "expected": x["expected_prompt_tokens"], "served": x["usage"].get("prompt_tokens")}
                    for r in runs for x in r["doc"]["records"]
                    if x["usage"].get("prompt_tokens") != x["expected_prompt_tokens"]]
    cold_hits = [{"run": run_name(r), "prompt": x["prompt"], "cached": x["metrics"]["cached"]}
                 for r in runs for x in r["doc"]["records"] if x["phase"] == "cold" and x["metrics"]["cached"] > 0]
    return {"prefix": prefix, "base": base, "variants": variants, "runs": meta, "absolute": absolute,
            "ratios": ratios, "identity": identity(runs), "prompt_token_mismatches": prompt_check,
            "cold_requests_with_cache_hits": cold_hits, "incomplete_runs": incomplete,
            "replies_off_schedule": off_schedule, "tool_versions": tool_versions,
            "held": sorted({name for r in runs for name in r["doc"].get("held", [])})}


def _fmt(summary: dict, digits: int = 2) -> str:
    if summary["half95"] is None:
        return f"{summary['mean']:.{digits}f}"
    return f"{summary['mean']:.{digits}f} ± {summary['half95']:.{digits}f}"


def markdown(result: dict) -> str:
    lines = [f"# {result['prefix']} (base {result['base']}; variants {', '.join(result['variants'])})", "",
             ("| cell | variant | decode tok/s (engine) | wall | accept | tok/round | prefill tok/s | TTFT s | "
              "min MemAvailable GiB |"), "|---|---|---|---|---|---|---|---|---|"]
    for name, by_variant in result["absolute"].items():
        if not name.startswith("decode/"):
            continue
        _, depth, think, sampling = name.split("/")
        prefill = result["absolute"].get(f"prefill/{depth}/{think}", {})
        memory = result["absolute"].get(f"memory/{depth}/{think}", {})
        for variant, m in by_variant.items():
            p, low = prefill.get(variant, {}), memory.get(variant, {}).get("mem_min_available_mib")
            cols = [_fmt(m[k]) if k in m else "-" for k in ("decode_tps", "decode_tps_wall")]
            cols.append(_fmt(m["accept"], 3) if "accept" in m else "-")
            cols.append(_fmt(m["tokens_per_round"]) if "tokens_per_round" in m else "-")
            cols.append(_fmt(p["prefill_tps"], 0) if "prefill_tps" in p else "-")
            cols.append(_fmt(p["ttft_s"]) if "ttft_s" in p else "-")
            cols.append(f"{low['min'] / 1024:.1f}" if low else "-")      # lowest over positions
            lines.append(f"| {depth} {think} {sampling} | {variant} | " + " | ".join(cols) + " |")
    if result["ratios"]:
        lines += ["", "| cell | variant | metric | ratio vs base (95% t) | geomean | below 1 | per cycle |",
                  "|---|---|---|---|---|---|---|"]
        for name, by_variant in result["ratios"].items():
            for variant, metrics in by_variant.items():
                for metric, s in metrics.items():
                    if metric in ("decode_tps", "prefill_tps", "accept", "ttft_s"):
                        lines.append(f"| {name} | {variant} | {metric} | {_fmt(s, 4)} | "
                                     f"{s['geomean'] and round(s['geomean'], 4)} | {s['below_1']}/{s['n']} | "
                                     f"{s['ratios']} |")
    ident = result["identity"]
    lines += ["", (f"Token identity: {ident['identical']}/{ident['keys']} repeated requests identical across all "
                   f"positions; {len(ident['differing'])} differ.")]
    if result["incomplete_runs"]:
        lines.append("Incomplete runs (stopped mid-schedule; their cells hold only the requests that finished): "
                     + ", ".join(result["incomplete_runs"]))
    if result["held"]:
        lines.append("Prompts held by the long-prefill hold (not measured): " + ", ".join(result["held"]))
    if result["replies_off_schedule"]:
        lines.append(f"Replies with a token count other than the profile's max_tokens: "
                     f"{len(result['replies_off_schedule'])}")
    if len(result["tool_versions"]) > 1:
        lines.append(f"Runs measured with {len(result['tool_versions'])} campaign_bench.py versions: "
                     + ", ".join(v[:12] for v in result["tool_versions"]))
    if result["prompt_token_mismatches"]:
        lines.append(f"Prompt token mismatches: {len(result['prompt_token_mismatches'])} (template or tokenizer "
                     "changed between prompt sizing and serving)")
    if result["cold_requests_with_cache_hits"]:
        hits = result["cold_requests_with_cache_hits"]
        lines.append("Cold requests that resumed cached prompt tokens (prefill counts only uncached tokens): "
                     + ", ".join(f"{h['run']}/{h['prompt']}={h['cached']}" for h in hits))
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    c = commands.add_parser("corpus")
    c.add_argument("out")
    c.add_argument("sources", nargs="+")
    c.add_argument("--max-mib", type=float, default=4.0)
    p = commands.add_parser("prompts")
    for name in ("url", "profile", "corpus", "out"):
        p.add_argument(name)
    r = commands.add_parser("run")
    for name in ("url", "profile", "prompts", "out"):
        r.add_argument(name)
    r.add_argument("--meta", help="JSON written by the runner: image id, args, env, cache dir, container")
    r.add_argument("--progress", help="heartbeat file for a watchdog (see Progress)")
    r.add_argument("--allow-long-prefill", action="store_true",
                   help=f"run prompts above {LONG_PREFILL_TOKENS} tokens; without it they are held (recorded in the "
                        "run's 'held' list and listed by analyze)")
    r.add_argument("--label", default="")
    r.add_argument("--depths", default="", help="comma-separated depth names (default: all)")
    r.add_argument("--tasks", default="", help="comma-separated task names (default: all)")
    r.add_argument("--thinking", choices=("on", "off", "both"), default="both")
    a = commands.add_parser("analyze")
    a.add_argument("runs")
    a.add_argument("prefix")
    a.add_argument("--base")
    a.add_argument("--out", help="analysis JSON (the markdown table goes to stdout)")
    args = parser.parse_args()
    if args.command == "corpus":
        corpus(args.out, args.sources, int(args.max_mib * (1 << 20)))
    elif args.command == "prompts":
        prompts(args.url.rstrip("/"), args.profile, args.corpus, args.out)
    elif args.command == "run":
        Progress.configure(args.progress)
        select = {"depths": [d for d in args.depths.split(",") if d], "tasks": [t for t in args.tasks.split(",") if t],
                  "thinking": None if args.thinking == "both" else args.thinking == "on",
                  "allow_long": args.allow_long_prefill}
        run(args.url.rstrip("/"), args.profile, args.prompts, args.out, meta=args.meta, label=args.label,
            select=select)
    else:
        result = analyze(args.runs, args.prefix, args.base)
        if args.out:
            write_json(args.out, result)
        print(markdown(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

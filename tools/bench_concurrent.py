"""Decode speed of an OpenAI-compatible server under concurrent requests (bench_openai's prompts and settings).

Default: four cells (code and chat prompts, sampled and greedy). In each cell N requests of the same prompt start
at once, stream i with seed ``seed + i`` (greedy: all ``seed``), each ``--tokens`` long with ignore_eos. Per stream: decode tok/s =
(tokens - 1) / (last token - first token) and time to first token. Aggregate tok/s = the streams' tokens after
their first / (last token of any stream - first token of any stream). Medians over ``--reps``.

``--mixed``: request i takes prompt i % 2 and seed seed + i // 2; also reports the steady aggregate, over the time
every stream is live (tokens split by the characters each stream delivered inside that window).
``--alone``: runs every request alone first and checks each concurrent reply's token-id hash (``token_sha``)
against it; ``--serial`` checks the alone runs against ``"draft": false``. ``--no-seed``: sampled requests carry
no seed (mlx_lm's server batches only those), for measuring a standard; no hash checks then. Run a standard with its
prompt cache off (mlx_lm: ``--prompt-cache-size 0``): a cached reply can arrive as one text piece. Tokens come from
the reply's usage. A request that fails to connect or errors counts in ``failed``; one whose text came in fewer than
``MIN_PIECES`` pieces counts in ``unmeasured``; neither enters the rates. ``--stagger-ms`` spaces the starts (a
server with a small listen backlog resets simultaneous connections; those resets are reported, not hidden).
Standard library only.

Missing token hashes are unverified, never equal. ``--strict`` requires hashes
for measured replies and verifies all requested comparisons; failed/unmeasured
cells or unverified comparisons exit nonzero. Ordinary servers may omit hashes
without strict mode. SSE arrivals measure text pieces, not individual tokens.
Transport errors and unequal requested comparisons always exit nonzero.

  python3 tools/bench_concurrent.py http://127.0.0.1:8080 MODEL --levels 1,2,4,8 --alone --output out.json
"""

import argparse
import json
import math
import statistics
import threading
import time
import urllib.request

MIN_PIECES = 4           # text pieces a reply needs before its span is timed
MAX_SSE_BYTES = 16 * 1024**2


def compare_hash(left: dict, right: dict) -> bool | None:
    """Exact keyed comparison; None means failed or missing hash evidence."""
    a, b = left.get("token_sha"), right.get("token_sha")
    if left.get("error") or right.get("error") or not isinstance(a, str) or not a or not isinstance(b, str) or not b:
        return None
    return a == b

PROMPTS = [
    {"name": "code", "kind": "completion",
     "prompt": "Write a short Python function that computes the Fibonacci sequence and explain it."},
    {"name": "chat", "kind": "chat",
     "prompt": "Explain how matrix multiplication uses a GPU in plain English, then give a small numerical example."},
]


def stream(base: str, model: str, item: dict, tokens: int, temperature: float, seed: int | None,
           draft: bool = True, gate: threading.Barrier | None = None) -> dict:
    try:
        return _stream(base, model, item, tokens, temperature, seed, draft, gate)
    except Exception as exc:  # noqa: BLE001 - a failed request is reported, not fatal
        return {"prompt": item["name"], "seed": seed, "sent": None, "first": None, "last": None, "tokens": 0,
                "pieces": [], "ttft_s": None, "decode_tps": None, "token_sha": None,
                "error": f"{type(exc).__name__}: {exc}"}


def _stream(base: str, model: str, item: dict, tokens: int, temperature: float, seed: int | None, draft: bool,
            gate: threading.Barrier | None) -> dict:
    body = {"model": model, "max_tokens": tokens, "temperature": temperature, "stream": True,
            "stream_options": {"include_usage": True}, "ignore_eos": True}
    if seed is not None:
        body["seed"] = seed
    if not draft:
        body["draft"] = False
    if temperature > 0:
        body.update(top_k=20, top_p=0.95)
    if item["kind"] == "chat":
        url = base + "/v1/chat/completions"
        body["messages"] = [{"role": "user", "content": item["prompt"]}]
        body["chat_template_kwargs"] = {"enable_thinking": False}
    else:
        url = base + "/v1/completions"
        body["prompt"] = item["prompt"]
    req = urllib.request.Request(url, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    if gate is not None:
        gate.wait()
    sent = time.perf_counter()
    pieces: list[tuple[float, int]] = []           # (arrival, characters) of every text chunk
    usage: dict = {}
    runtime: dict = {}
    consumed, done = 0, False
    with urllib.request.urlopen(req, timeout=1800) as resp:
        for raw in resp:
            consumed += len(raw)
            if len(raw) > 65536 or consumed > MAX_SSE_BYTES:
                raise ValueError("SSE response exceeded bounded size")
            line = raw.decode().strip()
            if line == "data: [DONE]":
                done = True
                break
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:])
            if not isinstance(chunk, dict) or "error" in chunk:
                raise ValueError(f"SSE error or invalid object: {chunk!r}")
            usage = chunk.get("usage") or usage
            runtime = chunk.get("tensorfold") or runtime
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                # reasoning counts: its tokens are in completion_tokens, so the clock starts with them
                piece = choice.get("text") or delta.get("content") or delta.get("reasoning_content") or ""
                if piece:
                    pieces.append((time.perf_counter(), len(piece)))
    if not done:
        raise ValueError("SSE ended before [DONE]")
    n = usage.get("completion_tokens")
    if type(n) is not int or n <= 0 or n > tokens:
        raise ValueError("SSE missing bounded positive completion usage")
    out = {"prompt": item["name"], "seed": seed, "sent": sent, "first": None, "last": None, "tokens": n,
           "pieces": pieces, "ttft_s": None, "decode_tps": None, "token_sha": runtime.get("token_sha")}
    if pieces:
        first, last = pieces[0][0], pieces[-1][0]
        out.update(first=first, last=last, ttft_s=first - sent)
        if n > 1 and last > first and len(pieces) >= MIN_PIECES:
            out["decode_tps"] = (n - 1) / (last - first)
        else:
            out["unmeasured"] = True
    else:
        out["unmeasured"] = True
    return out


def together(base: str, model: str, specs: list[tuple[dict, int | None]], tokens: int, temperature: float,
             stagger_ms: float = 0.0) -> list[dict]:
    gate = threading.Barrier(len(specs))
    out: list[dict | None] = [None] * len(specs)

    def run(i: int) -> None:
        if stagger_ms:
            gate.wait()
            time.sleep(i * stagger_ms / 1e3)
            out[i] = stream(base, model, specs[i][0], tokens, temperature, specs[i][1])
        else:
            out[i] = stream(base, model, specs[i][0], tokens, temperature, specs[i][1], gate=gate)

    threads = [threading.Thread(target=run, args=(i,)) for i in range(len(specs))]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return [r for r in out if r is not None]


def aggregates(runs: list[dict]) -> dict:
    done = [r for r in runs if r["decode_tps"] is not None]
    out: dict = {"failed": sum(1 for r in runs if r.get("error")), "unmeasured": sum(1 for r in runs if r.get("unmeasured")),
                 "aggregate_tps": 0.0}
    if not done:
        return out
    first = min(r["first"] for r in done)
    last = max(r["last"] for r in done)
    if last > first:
        out["aggregate_tps"] = sum(r["tokens"] - 1 for r in done) / (last - first)
    lo, hi = max(r["first"] for r in done), min(r["last"] for r in done)       # every stream live
    if len(done) > 1 and hi > lo:
        share = 0.0
        for r in done:
            chars = sum(c for _, c in r["pieces"]) or 1
            share += r["tokens"] * sum(c for t, c in r["pieces"] if lo <= t <= hi) / chars
        out["steady_tps"] = share / (hi - lo)
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--levels", default="4", help="concurrent requests, a comma list")
    p.add_argument("--tokens", type=int, default=256)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--temperatures", default="1.0,0")
    p.add_argument("--mixed", action="store_true", help="request i takes prompt i %% 2 (sampled only)")
    p.add_argument("--alone", action="store_true", help="check each reply's token_sha against its request alone")
    p.add_argument("--serial", action="store_true", help="check the alone runs against \"draft\": false")
    p.add_argument("--no-seed", action="store_true", help="sampled requests without a seed (standards; no checks)")
    p.add_argument("--strict", action="store_true", help="fail on missing hashes, unverified comparisons or unmeasured cells")
    p.add_argument("--stagger-ms", type=float, default=0.0, help="start request i this many ms after request i - 1")
    p.add_argument("--label", default="")
    p.add_argument("--output")
    args = p.parse_args()
    if args.strict and args.no_seed:
        p.error("--strict requires keyed requests; omit --no-seed")
    if args.no_seed:
        args.alone = args.serial = False
    levels = [int(x) for x in args.levels.split(",")]
    temps = [1.0] if args.mixed else [float(t) for t in args.temperatures.split(",")]
    if any(n < 1 for n in levels) or args.tokens < 1 or args.reps < 1 or any(not math.isfinite(t) or t < 0 for t in temps):
        p.error("levels/tokens/reps must be positive and temperatures finite/nonnegative")
    cells = [None] if args.mixed else PROMPTS
    report: dict = {"label": args.label, "tokens": args.tokens, "mixed": args.mixed, "strict": args.strict,
                   "timing_unit": "SSE text-piece arrivals; not individual-token latency", "cells": []}
    failed = False
    for temp in temps:
        for item in cells:
            specs: list[tuple[dict, int | None]]
            if args.mixed or item is None:
                specs = [(PROMPTS[i % 2], args.seed + i // 2) for i in range(max(levels))]
            else:
                specs = [(item, args.seed + i if temp > 0 else args.seed) for i in range(max(levels))]
            if args.no_seed:
                specs = [(spec[0], None) for spec in specs]
            warmup = together(args.base, args.model, specs[:1], 16, temp)
            warmup_failed = any(r.get("error") for r in warmup)
            failed |= warmup_failed
            alone: dict[tuple[str, int | None], dict] = {}
            if args.alone or args.serial:
                for spec in specs:
                    key = (spec[0]["name"], spec[1])
                    if key not in alone:
                        alone[key] = stream(args.base, args.model, spec[0], args.tokens, temp, spec[1])
            if args.serial:
                for key, ref in alone.items():
                    serial = stream(args.base, args.model, next(q for q in PROMPTS if q["name"] == key[0]),
                                    args.tokens, temp, key[1], draft=False)
                    ref["serial_equal"] = compare_hash(serial, ref)
                    ref["serial_error"] = serial.get("error") or ref.get("error")
                    failed |= bool(ref["serial_error"])
            for n in levels:
                reps = []
                for _ in range(args.reps):
                    runs = together(args.base, args.model, specs[:n], args.tokens, temp, args.stagger_ms)
                    rep = aggregates(runs)
                    rep["per_stream_tps"] = [round(r["decode_tps"], 1) for r in runs if r["decode_tps"] is not None]
                    rep["ttft_s"] = [round(r["ttft_s"] or 0.0, 2) for r in runs]
                    rep["errors"] = [r["error"] for r in runs if r.get("error")]
                    rep["token_hashes"] = [r.get("token_sha") for r in runs]
                    rep["hashes_unverified"] = sum(not isinstance(r.get("token_sha"), str) or not r["token_sha"]
                                                  for r in runs)
                    if alone:
                        rep["equal_alone"] = [compare_hash(alone[(r["prompt"], r["seed"])], r)
                                              for r in runs]
                    reps.append(rep)
                cell = {"prompt": "mixed" if args.mixed or item is None else item["name"], "temperature": temp,
                        "streams": n, "failed": sum(r["failed"] for r in reps),
                        "unmeasured": sum(r["unmeasured"] for r in reps),
                        "hashes_unverified": sum(r["hashes_unverified"] for r in reps),
                        "warmup_failed": warmup_failed,
                        "aggregate_tps": round(statistics.median(r["aggregate_tps"] for r in reps), 1),
                        "per_stream_tps": round(statistics.median([v for r in reps for v in r["per_stream_tps"]] or [0.0]), 1),
                        "ttft_s_max": max(max(r["ttft_s"]) for r in reps)}
                if all("steady_tps" in r for r in reps):
                    cell["steady_tps"] = round(statistics.median(r["steady_tps"] for r in reps), 1)
                if alone:
                    checks = [v for r in reps for v in r["equal_alone"]]
                    cell["alone"] = {"equal": checks.count(True), "unequal": checks.count(False),
                                     "failed": checks.count(None), "unverified": checks.count(None)}
                if args.serial:
                    checks = [v.get("serial_equal") for v in alone.values()]
                    cell["serial"] = {"equal": checks.count(True), "unequal": checks.count(False),
                                      "failed": checks.count(None), "unverified": checks.count(None)}
                invalid = bool(cell["failed"] or warmup_failed or any(v.get("error") or v.get("serial_error") for v in alone.values()))
                invalid |= any(cell.get(name, {}).get("unequal", 0) > 0 for name in ("alone", "serial"))
                if args.strict:
                    invalid |= bool(cell["unmeasured"] or cell["hashes_unverified"] or
                                    any(cell.get(name, {}).get("unverified", 0) > 0 for name in ("alone", "serial")))
                cell["passed"] = not invalid
                failed |= invalid
                errors = sorted({e for r in reps for e in r["errors"]})
                if errors:
                    cell["errors"] = errors[:3]
                print(json.dumps({"label": args.label, **cell}), flush=True)
                report["cells"].append({**cell, "reps": reps})
    report["passed"] = not failed
    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=1)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

"""Verify a live V100 Q4 service against fresh serial replies from the same format.

This is a functional procedure, not a benchmark or an INT8 equivalence gate.
The twelve supplied requests retain their original generation budgets. --tokens
sets the nine seeded profiles and the four concurrent completion budgets. Every
HTTP exchange retains its request, raw response, metadata and parsed result.
This can investigate a candidate before production selection. Passing these
functional checks does not accept its separate numerical quality tradeoff.
Prefix reuse is required by default. Explicit --prefix-reuse cold instead
requires zero cached tokens in every completed drafted reply, including recovery.
"""

import argparse
from concurrent.futures import ThreadPoolExecutor
import copy
import hashlib
import json
from pathlib import Path
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

if __package__:
    from . import openai_protocol as protocol
    from .benchmark_output import write_json
else:
    import openai_protocol as protocol
    from benchmark_output import write_json


CASES = (
    "multiply", "modular", "reason", "json", "unicode", "python", "sql", "fix", "explain", "summarize", "france", "retrieve"
)
Q4_FORMATS = ("rotorquant-signed128-4", "rotorquant-signed64-norm4")
PROFILES = (
    ("code", "Write a Python function that merges two sorted lists without modifying them.", False),
    ("reason", "A train travels 90 kilometers in 75 minutes. What is its average speed in kilometers per hour? Explain your calculation.", True),
    ("chat", "Explain how a hash table handles a collision and when resizing helps.", False),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def integer(value, minimum=0):
    return type(value) is int and value >= minimum


def load_cases(directory):
    """Read request DATA only. Old response IDs are deliberately never opened."""
    directory = Path(directory)
    hashes = {}

    def read(name):
        path = directory / name
        require(not path.is_symlink() and path.is_file() and path.stat().st_size <= 1 << 20, "bounded case DATA file required")
        raw = path.read_bytes()
        hashes[name] = hashlib.sha256(raw).hexdigest()
        return protocol.object_json(raw)

    index = read("baseline-ready.json")
    require(index.get("passed") is True and index.get("cases") == list(CASES), "the complete twelve-case request index is required")
    cases = []
    for name in CASES:
        body = read(name + "-request.json")
        require(type(body.get("messages")) is list and bool(body["messages"]), "case messages are required")
        require(integer(body.get("max_tokens"), 1) and body["max_tokens"] <= 4096, "bounded case max_tokens required")
        require(not body.get("tools") and not body.get("functions") and body.get("n", 1) == 1, "single text/reasoning cases required")
        cases.append((name, body))
    return cases, hashes


def profiles(model, tokens):
    return [
        (f"{name}-s{seed}", dict(
            model=model, messages=[dict(role="user", content=prompt)], temperature=temperature,
            top_k=40 if temperature else 0, top_p=0.9 if temperature else 1, seed=seed,
            max_tokens=tokens, return_token_ids=True, chat_template_kwargs=dict(enable_thinking=thinking),
        ))
        for name, prompt, thinking in PROFILES for seed, temperature in ((17, 0), (19, 0.7), (47, 0.7))
    ]


def completion(response, maximum):
    protocol.chunk_fields(response)
    n = protocol.completion_usage(response.get("usage") or {}, maximum, prompt=True)
    require(type(response["usage"].get("total_tokens")) is int and
            response["usage"]["total_tokens"] == response["usage"]["prompt_tokens"] + n, "complete usage total must match prompt and completion")
    choices = response.get("choices")
    require(type(choices) is list and len(choices) == 1, "one complete choice is required")
    choice = choices[0]
    require(type(choice.get("index")) is int and choice["index"] == 0 and choice.get("finish_reason") in ("stop", "length"), "missing complete finish reason")
    message = choice.get("message")
    require(type(message) is dict and message.get("role") == "assistant", "complete assistant message required")
    require(set(message) <= {"role", "content", "reasoning_content"}, "unsupported message fields in text-only procedure")
    require("content" in message and (message["content"] is None or type(message["content"]) is str), "invalid message content")
    require("reasoning_content" not in message or type(message["reasoning_content"]) is str, "invalid reasoning content")
    runtime = response.get("tensorfold") or {}
    ids = runtime.get("token_ids")
    require(type(ids) is list and len(ids) == n and all(integer(token) for token in ids), "complete token IDs must match usage")
    return {"ids": ids, "message": message, "finish": choice["finish_reason"]}


def compare(reference, candidate, maximum):
    a, b = completion(reference, maximum), completion(candidate, maximum)
    return {"exact_ids": a["ids"] == b["ids"], "exact_message": a["message"] == b["message"],
            "exact_finish": a["finish"] == b["finish"], "tokens_reference": len(a["ids"]), "tokens_candidate": len(b["ids"]),
            "passed": a == b}


def prefix_receipt(response, mode, *, recovery=False):
    """Validate the declared prefix contract without changing any identity gate."""
    cached = (response.get("tensorfold") or {}).get("cached")
    if mode == "cold":
        require(type(cached) is int and cached == 0, "cold prefix mode requires cached==0 in every completed drafted reply")
    elif mode == "qualified":
        if recovery:
            require(integer(cached, 1), "recovered repeated prompt did not reuse its prefix")
    else:
        raise ValueError("unknown prefix acceptance mode")
    return {"prefix_reuse": mode, "cached": cached}


def capacity_proof(stats, *, parallel, kv_dtype, context=None):
    """Validate ordinary public allocation receipts, not inferred free VRAM."""
    require(stats.get("object") == "tensorfold.stats", "ordinary public stats required")
    model, plan = stats.get("model") or {}, stats.get("capacity_plan") or {}
    require(model.get("kv_dtype") == kv_dtype and kv_dtype in Q4_FORMATS, "service cache format differs from the selected Q4 format")
    selected = model.get("context_length")
    require(integer(selected, 1) and (context is None or selected == context), "selected service context mismatch")
    require(stats.get("limits", {}).get("lanes") == parallel, "service lane count differs from requested concurrency")
    windows = plan.get("kv_full_windows") or {}
    require(all(windows.get(key) == parallel for key in ("requested", "reserved", "stream_slots")), "every lane must reserve a complete equal window")
    require(windows.get("materialized") is True and windows.get("context_window") == selected, "windows must be physically materialized at the selected context")
    capacity = windows.get("cache_slots_each")
    require(integer(capacity, selected), "cache allocation does not cover the selected context")
    require(windows.get("startup_capacities") == [capacity] * parallel, "local startup capacities are not full and equal")
    require(integer(windows.get("workspace_bytes"), 2 << 30), "the existing 2GiB serving workspace must remain protected")
    ranks = windows.get("ranks")
    require(type(ranks) is list and len(ranks) == 8, "all eight physical rank receipts are required")
    cache_bytes = windows.get("allocated_cache_bytes")
    require(integer(cache_bytes, 1), "positive physical cache bytes required")
    for rank, row in enumerate(ranks):
        require(row.get("rank") == rank and row.get("materialized") is True, "missing or unordered physical rank")
        require(row.get("startup_capacities") == [capacity] * parallel, "rank cache capacities are not full and equal")
        require(row.get("allocated_cache_bytes") == cache_bytes, "unequal rank cache byte receipts")
        allocated, reserved = row.get("torch_allocated_bytes"), row.get("torch_reserved_bytes")
        require(integer(allocated, cache_bytes) and integer(reserved, allocated), "rank allocation cannot cover its materialized caches")
    require(plan.get("context_window") == selected, "admitted context and public model context disagree")
    return {"context": selected, "parallel": parallel, "kv_dtype": kv_dtype, "capacity_each": capacity,
            "allocated_cache_bytes_per_rank": cache_bytes, "ranks": ranks,
            "graphs": plan.get("multi_graphs"), "allocation_source": "public startup receipt; no new full-context payload test"}


class RecordedResponse:
    def __init__(self, response, stream, limit):
        self.response, self.stream, self.limit, self.consumed = response, stream, limit, 0
        self.status = response.status

    def getheader(self, *args):
        return self.response.getheader(*args)

    def read1(self, amount):
        raw = self.response.read1(min(amount, self.limit - self.consumed + 1))
        self.stream.write(raw)
        self.consumed += len(raw)
        require(self.consumed <= self.limit, "HTTP response exceeded bounded size")
        return raw


class Client:
    def __init__(self, url, output, *, timeout=900, limits=None):
        parsed = urllib.parse.urlsplit(url)
        require(parsed.scheme in ("http", "https") and parsed.netloc and not parsed.query and not parsed.fragment, "HTTP service URL required")
        self.url, self.output, self.timeout = url.rstrip("/"), Path(output), timeout
        self.limits = limits or protocol.Limits()

    def call(self, label, body=None, *, path=None, cancel=False):
        require(re.fullmatch(r"[a-zA-Z0-9_-]+", label) is not None, "safe unique exchange label required")
        directory = self.output / label
        directory.mkdir()
        path = path or ("/stats" if body is None else "/v1/chat/completions")
        write_json(directory / "request.json", {"url": self.url + path, "method": "GET" if body is None else "POST", "body": body})
        request = urllib.request.Request(self.url + path, data=None if body is None else json.dumps(body).encode(),
                                         headers={"Content-Type": "application/json"})
        metadata = {"complete": False, "cancelled_by_client": False}
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response, (directory / "response.raw").open("wb") as raw:
                metadata.update(status=response.status, headers=list(response.headers.items()))
                recorded = RecordedResponse(response, raw, self.limits.response_bytes)
                if body is not None and body.get("stream"):
                    protocol.response_type(recorded, "text/event-stream")
                    result = self.read_stream(recorded, body, cancel=cancel)
                    metadata.update(complete=not cancel, cancelled_by_client=cancel, done=not cancel)
                elif body is not None:
                    result = protocol.json_response(recorded, body["max_tokens"], limits=self.limits)
                    metadata["complete"] = True
                else:
                    protocol.response_type(recorded, "application/json")
                    chunks = []
                    while chunk := recorded.read1(protocol.READ_BYTES):
                        chunks.append(chunk)
                    result = protocol.object_json(b"".join(chunks))
                    metadata["complete"] = True
                write_json(directory / "result.json", result)
                return result
        except BaseException as error:
            if isinstance(error, urllib.error.HTTPError):
                metadata.update(status=error.code, headers=list(error.headers.items()))
                with error, (directory / "response.raw").open("wb") as raw:
                    raw.write(error.read(self.limits.response_bytes + 1))
            metadata["error"] = {"type": type(error).__name__, "message": str(error)}
            raise
        finally:
            write_json(directory / "exchange.json", metadata)

    def read_stream(self, response, body, *, cancel):
        content, reasoning, usage, runtime, finish, events = [], [], None, None, None, []
        for event in protocol.sse_objects(response, limits=self.limits):
            events.append(event)
            current_usage, current_runtime, pieces = protocol.chunk_fields(event)
            usage = current_usage if current_usage is not None else usage
            runtime = current_runtime if current_runtime is not None else runtime
            for choice in event.get("choices", []):
                require(choice.get("index") == 0, "unexpected SSE choice index")
                delta = choice.get("delta") or {}
                require(not delta.get("tool_calls"), "text-only SSE procedure")
                if delta.get("content"):
                    content.append(delta["content"])
                if delta.get("reasoning_content"):
                    reasoning.append(delta["reasoning_content"])
                finish = choice.get("finish_reason") or finish
            if cancel and any(pieces):
                require(finish is None and usage is None, "cancellation did not interrupt an active generation")
                return {"events": events, "intentionally_incomplete": True}
        require(not cancel, "stream completed before cancellation could interrupt generation")
        require(usage is not None and runtime is not None, "complete SSE requires usage and token IDs")
        message = {"role": "assistant", "content": "".join(content) or None}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        result = {"choices": [{"index": 0, "message": message, "finish_reason": finish}], "usage": usage, "tensorfold": runtime,
                  "sse_events": events, "sse_done": True}
        completion(result, body["max_tokens"])
        return result


def idle(stats):
    load = stats.get("load") or {}
    return all(type(load.get(name)) is int and load[name] == 0 for name in ("requests_in_flight", "requests_running", "requests_waiting"))


def run(args):
    cases, hashes = load_cases(args.cases_dir)
    output = args.output_dir
    require(not output.exists() or output.is_dir() and not any(output.iterdir()), "new or empty output directory required")
    output.mkdir(parents=True, exist_ok=True)
    client = Client(args.url, output, timeout=args.timeout, limits=protocol.from_arguments(args))
    summary = {"complete": False, "passed": False, "prefix_reuse": args.prefix_reuse,
               "scope": "same-format live functional checks; separate from INT8 numerical quality and performance",
               "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), "case_data_sha256": hashes, "checks": []}
    try:
        initial = client.call("stats-initial")
        summary["capacity"] = capacity_proof(initial, parallel=args.parallel, kv_dtype=args.kv_dtype, context=args.context)
        require(idle(initial), "qualification requires an initially idle service")
        model = args.model or initial["model"]["id"]
        require(model == initial["model"]["id"] or model in initial["model"].get("aliases", []), "model ID is not served")
        seeded = profiles(model, args.tokens)
        for group, rows in (("requests", cases), ("profiles", seeded)):
            for name, original in rows:
                body = {**copy.deepcopy(original), "model": model, "return_token_ids": True, "stream": False}
                reference = client.call(f"{group}-{name}-serial", {**body, "draft": False})
                candidate = client.call(f"{group}-{name}-draft", {**body, "draft": True})
                summary["checks"].append({"group": group, "case": name,
                                          **prefix_receipt(candidate, args.prefix_reuse),
                                          **compare(reference, candidate, body["max_tokens"])})
                write_json(output / "summary.json", summary)
        concurrent = [(name, {**body, "ignore_eos": True}) for name, body in seeded[:args.parallel]]
        require(len(concurrent) == args.parallel, "concurrency exceeds available independent seeded profiles")
        references = [client.call(f"concurrent-{index}-serial", {**body, "draft": False}) for index, (_, body) in enumerate(concurrent)]
        require(all(len(completion(result, args.tokens)["ids"]) == args.tokens and
                    result["choices"][0]["finish_reason"] == "length" for result in references),
                "ignore_eos concurrent references must complete their entire token budget")
        barrier = threading.Barrier(args.parallel)

        def one(index):
            barrier.wait(timeout=args.timeout)
            return client.call(f"concurrent-{index}-draft", {**concurrent[index][1], "draft": True})

        peak = 0
        poll = 0
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(one, index) for index in range(args.parallel)]
            while not all(future.done() for future in futures):
                stats = client.call(f"concurrent-stats-{poll}")
                peak = max(peak, stats.get("load", {}).get("requests_in_flight", 0))
                poll += 1
                time.sleep(0.1)
            results, errors = [], []
            for future in futures:
                try:
                    results.append(future.result())
                except Exception as error:
                    errors.append(error)
            if errors:
                raise ExceptionGroup("concurrent HTTP exchanges failed", errors)
        require(peak >= args.parallel, "public stats did not observe all concurrent requests in flight")
        summary["concurrency_peak_in_flight"] = peak
        for index, (reference, result) in enumerate(zip(references, results, strict=True)):
            summary["checks"].append({"group": "concurrent", "case": index,
                                      **prefix_receipt(result, args.prefix_reuse), **compare(reference, result, args.tokens)})
        body = concurrent[0][1]
        stream = client.call("complete-sse", {**body, "draft": True, "stream": True, "stream_options": {"include_usage": True}})
        summary["checks"].append({"group": "sse", "case": "complete",
                                  **prefix_receipt(stream, args.prefix_reuse), **compare(references[0], stream, args.tokens)})
        before = client.call("stats-before-cancel")
        client.call("cancelled-sse", {**body, "draft": True, "stream": True, "max_tokens": max(128, args.tokens),
                                      "stream_options": {"include_usage": True}}, cancel=True)
        deadline = time.monotonic() + args.cleanup_timeout
        attempt = 0
        while True:
            recovered = client.call(f"stats-cancel-recovery-{attempt}")
            if idle(recovered) and recovered.get("totals", {}).get("cancelled", 0) > before.get("totals", {}).get("cancelled", 0):
                break
            require(time.monotonic() < deadline, "cancelled request did not retire and increment cancellation counter")
            attempt += 1
            time.sleep(0.25)
        health = client.call("health-after-cancel", path="/health")
        require(health.get("ok") is True or health.get("status") == "ok", "service health failed after cancellation")
        for phase in ("prime", "reuse"):
            result = client.call("prefix-recovery-" + phase, {**body, "draft": True})
            summary["checks"].append({"group": "prefix-recovery", "case": phase,
                                      **prefix_receipt(result, args.prefix_reuse, recovery=phase == "reuse"),
                                      **compare(references[0], result, args.tokens)})
        final = client.call("stats-final")
        capacity_proof(final, parallel=args.parallel, kv_dtype=args.kv_dtype, context=summary["capacity"]["context"])
        require(final.get("server") == initial.get("server") and final["model"] == initial["model"], "service identity changed during qualification")
        require(idle(final), "service did not return to idle")
        require(final.get("totals", {}).get("failed") == initial.get("totals", {}).get("failed"), "service recorded a failed request")
        summary.update(complete=True, passed=all(check["passed"] for check in summary["checks"]), cancellation_cleanup=True,
                       prefix_recovered=True, prefix_reused=args.prefix_reuse == "qualified")
        return 0 if summary["passed"] else 2
    except BaseException as error:
        summary["error"] = {"type": type(error).__name__, "message": str(error)}
        return 130 if isinstance(error, KeyboardInterrupt) else 1
    finally:
        write_json(output / "summary.json", summary)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model")
    parser.add_argument("--cases-dir", type=Path, required=True, help="twelve request JSONs and baseline-ready.json DATA; old response IDs are never used")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parallel", type=int, default=4, help="complete concurrent requests and full resident windows (default4)")
    parser.add_argument("--tokens", type=int, default=128, help="seeded/concurrent token budgets; original twelve-case budgets stay unchanged")
    parser.add_argument("--context", type=int, help="expected public context; otherwise pin the initial selected context")
    parser.add_argument("--kv-dtype", choices=Q4_FORMATS, default=Q4_FORMATS[1])
    parser.add_argument("--prefix-reuse", choices=("qualified", "cold"), default="qualified",
                        help="qualified requires recovered prefix reuse (default); cold requires cached==0 in every completed drafted reply")
    parser.add_argument("--timeout", type=float, default=900, help="per-socket inactivity timeout; this is not a benchmark or whole-request deadline")
    parser.add_argument("--cleanup-timeout", type=float, default=60)
    protocol.add_arguments(parser)
    args = parser.parse_args(argv)
    require(2 <= args.parallel <= 8 and 1 <= args.tokens <= 4096, "parallel2..8 and tokens1..4096 required")
    require(args.timeout > 0 and args.cleanup_timeout > 0 and (args.context is None or args.context > 0), "positive context/timeouts required")
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())

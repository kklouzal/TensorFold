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
without strict mode. SSE arrivals measure text pieces, not individual tokens. Rates estimate tokens
from total usage and text arrivals; the first piece can contain multiple tokens.
Transport errors and unequal requested comparisons exit nonzero in benchmark modes.
With --repro-input, exact differences are diagnostic data and a complete procedure
exits zero; transport/protocol/configuration failures still exit nonzero.

  python3 tools/bench_concurrent.py http://127.0.0.1:8080 MODEL --levels 1,2,4,8 --alone --output out.json

Response budgets: --response-mib defaults to16MiB, --sse-line-kib to64KiB.
Raise them for larger valid replies or servers that batch a response into one
SSE line. Event storage shares the response budget. Limits concern client
measurement storage and never alter generation parameters. Inactivity timeouts
remain unchanged; these readers do not claim a whole-request wall deadline.
"""

import argparse
from functools import partial
import json
import math
import statistics
import sys
import threading
import hashlib
import itertools
from pathlib import Path
import uuid
import time
import urllib.request

if sys.version_info < (3, 11):
    sys.exit(f"Python 3.11+ is required (this is {sys.version.split()[0]}); "
             "TensorFold's tools use PEP 604 unions. Try: python3.12 tools/bench_concurrent.py ...")

if __package__:
    from . import openai_protocol as protocol
    from .benchmark_output import write_json
    from .worker_lifetime import Task, drain, raise_failures
else:
    import openai_protocol as protocol
    from benchmark_output import write_json
    from worker_lifetime import Task, drain, raise_failures

MIN_PIECES = 4           # text pieces a reply needs before its span is timed


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
           draft: bool = True, gate: threading.Barrier | None = None, *, limits=None) -> dict:
    try:
        if limits is None:
            return _stream(base, model, item, tokens, temperature, seed, draft, gate)
        return _stream(base, model, item, tokens, temperature, seed, draft, gate, limits=limits)
    except Exception as exc:  # noqa: BLE001 - a failed request is reported, not fatal
        return {"prompt": item["name"], "seed": seed, "sent": None, "first": None, "last": None, "tokens": 0,
                "pieces": [], "ttft_s": None, "decode_tps": None, "token_sha": None,
                "error": f"{type(exc).__name__}: {exc}"}


def completion_fields(chunk):
    # Responses has a nullable error member; completion SSE has no such field.
    if "error" in chunk:
        raise ValueError("SSE error or invalid completion event")
    return protocol.chunk_fields(chunk)


def _stream(base: str, model: str, item: dict, tokens: int, temperature: float, seed: int | None, draft: bool,
            gate: threading.Barrier | None, *, limits=None) -> dict:
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
    with urllib.request.urlopen(req, timeout=1800) as resp:
        protocol.response_type(resp, 'text/event-stream')
        for chunk in protocol.sse_objects(resp, limits=limits):
            new_usage, new_runtime, new_pieces = completion_fields(chunk)
            usage = new_usage if new_usage is not None else usage
            runtime = new_runtime if new_runtime is not None else runtime
            for piece in new_pieces:
                if piece:
                    pieces.append((time.perf_counter(), len(piece)))
    n = protocol.completion_usage(usage, tokens)
    out = {"prompt": item["name"], "seed": seed, "sent": sent, "first": None, "last": None, "tokens": n,
           "pieces": pieces, "ttft_s": None, "decode_tps": None, "token_sha": runtime.get("token_sha"),
           "timing_scope": "SSE text-arrival estimate; first piece may contain multiple tokens"}
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
             stagger_ms: float = 0.0, *, limits=None) -> list[dict]:
    if not math.isfinite(stagger_ms) or stagger_ms < 0:
        raise ValueError("stagger milliseconds must be finite and nonnegative")
    if not specs:
        return []
    gate = threading.Barrier(len(specs))
    callback_failures = [[] for _ in specs]
    callback_roots = [None] * len(specs)

    kwargs = {} if limits is None else {"limits": limits}

    def run(i: int) -> dict:
        try:
            if stagger_ms:
                gate.wait()
                time.sleep(i * stagger_ms / 1e3)
                result = stream(base, model, specs[i][0], tokens, temperature, specs[i][1], **kwargs)
            else:
                result = stream(base, model, specs[i][0], tokens, temperature, specs[i][1], gate=gate, **kwargs)
            if not isinstance(result, dict):
                raise TypeError("benchmark request must return a complete result object")
            if result.get("error"):
                while True:
                    try:
                        gate.abort()
                        break
                    except BaseException as cleanup:
                        callback_failures[i].append(cleanup)
                if callback_failures[i]:
                    raise callback_failures[i][0]
            return result
        except BaseException as error:
            callback_roots[i] = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error))
            while True:
                try:
                    gate.abort()
                    break
                except BaseException as cleanup:
                    callback_failures[i].append(cleanup)
            raise

    # Publish the unstarted owner before acquisition. Completion belongs to
    # the callback journal, so an interrupted Thread.join cannot release its
    # resources and a start return interrupted before publication loses no work.
    tasks, results, primary, failures = [], [None] * len(specs), None, []

    def execute(i):
        results[i] = run(i)

    try:
        try:
            for i in range(len(specs)):
                task = Task(partial(execute, i), name=f"benchmark-request-{i}")
                tasks.append(task)
                task.start()
        except BaseException as error:
            primary = error
            roots = (BaseException.__cause__.__get__(error), BaseException.__context__.__get__(error))
            while True:
                try:
                    gate.abort()
                    break
                except BaseException as cleanup:
                    failures.append(cleanup)
            failures.extend(root for root in roots if root is not None and root is not error)
    except BaseException as error:
        if primary is None:
            primary = error
        elif error is not primary:
            failures.append(error)
    finally:
        try:
            drain(tasks, primary)
        except BaseException as error:
            if primary is None:
                primary = error
            elif error is not primary:
                failures.append(error)
        for errors, roots in zip(callback_failures, callback_roots):
            failures.extend(errors)
            if roots is not None:
                failures.extend(root for root in roots if root is not None and root is not primary)
        if primary is not None:
            raise_failures(primary, failures)
    return results


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



STANDARD_LEVELS = (1, 2, 4, 8, 16, 24, 32)
PROFILE_CAPS = {"q8": 24, "q4": 32}
MATRIX_TEXT = "A careful engineer records observations, checks each result, and explains the next useful step. "


def matrix_ids(plan, request):
    prefix = request["prefix"]
    return prefix + list(itertools.islice(itertools.cycle(plan["filler_ids"]), request["input_tokens"] - len(prefix)))


def prepare_matrix(model_dir, model, profile, levels, inputs, tokens, reps):
    from tokenizers import Tokenizer
    path = Path(model_dir) / "tokenizer.json"
    tokenizer = Tokenizer.from_file(str(path))
    tokenizer.no_padding()
    tokenizer.no_truncation()
    run = uuid.uuid4().hex
    filler = tokenizer.encode(MATRIX_TEXT, add_special_tokens=False).ids
    if not filler:
        raise ValueError("benchmark natural text produced no tokens")
    plan = {"schema": "tensorfold.phase-matrix-plan-v1", "run_identity": run, "model": model,
            "profile": profile, "admin_cap": PROFILE_CAPS[profile], "levels": levels,
            "context": 262144, "tokens": tokens, "reps": reps, "inputs": inputs,
            "tokenizer_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "filler_ids": filler,
            "input_scope": "pretokenized natural completion text; model-core phases and separate HTTP wall time",
            "cohorts": [], "excluded_levels": [n for n in levels if n > PROFILE_CAPS[profile]]}
    for size in inputs:
        for level, repetition, warmup in [(1, 0, True)] + [(n, rep, False) for n in levels if n <= plan["admin_cap"] for rep in range(reps)]:
            cohort = {"id": len(plan["cohorts"]), "level": level, "input_tokens": size,
                      "tokens": min(tokens, 16) if warmup else tokens, "repetition": repetition, "warmup": warmup, "requests": []}
            for index in range(level):
                text = f"Run {run} case {cohort['id']}/{index}. Continue: "
                prefix = tokenizer.encode(text, add_special_tokens=False).ids
                if len(prefix) >= size or size + cohort["tokens"] > plan["context"]:
                    raise ValueError("input must fit the natural identity prefix and complete reply within262144")
                row = {"prefix": prefix, "input_tokens": size, "index": index}
                ids = matrix_ids(plan, row)
                row["prompt_sha256"] = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
                cohort["requests"].append(row)
            plan["cohorts"].append(cohort)
    hashes = [r["prompt_sha256"] for c in plan["cohorts"] for r in c["requests"]]
    if len(set(hashes)) != len(hashes):
        raise ValueError("native tokenizer collapsed benchmark request identities")
    return plan


def load_matrix(path):
    from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
    with Path(path).open("rb") as source:
        raw = source.read(MAX_MESSAGE_BYTES + 1)
    if len(raw) > MAX_MESSAGE_BYTES:
        raise ValueError("matrix plan exceeds existing96MiB data envelope")
    plan = json.loads(raw)
    if (plan.get("schema") != "tensorfold.phase-matrix-plan-v1" or plan.get("profile") not in PROFILE_CAPS
            or plan.get("admin_cap") != PROFILE_CAPS[plan["profile"]] or plan.get("context") != 262144
            or type(plan.get("run_identity")) is not str or len(plan["run_identity"]) != 32
            or any(char not in "0123456789abcdef" for char in plan["run_identity"])
            or not plan.get("cohorts") or not plan.get("filler_ids")
            or any(type(token) is not int or not 0 <= token < (1 << 31) for token in plan["filler_ids"])):
        raise ValueError("invalid private benchmark plan")
    seen = set()
    for cohort in plan["cohorts"]:
        if (any(type(cohort[key]) is not int or not 0 < cohort[key] <= (1 << 31) for key in ("level", "tokens", "input_tokens"))
                or type(cohort["id"]) is not int or cohort["id"] < 0
                or not 0 < cohort["level"] <= plan["admin_cap"]
                or cohort["input_tokens"] + cohort["tokens"] > plan["context"]
                or len(cohort["requests"]) != cohort["level"] or cohort["id"] in seen):
            raise ValueError("invalid cohort cardinality or identity")
        seen.add(cohort["id"])
        for row in cohort["requests"]:
            if (type(row.get("prefix")) is not list or not row["prefix"]
                    or len(row["prefix"]) >= cohort["input_tokens"]
                    or any(type(token) is not int or not 0 <= token < (1 << 31) for token in row["prefix"])
                    or row.get("input_tokens") != cohort["input_tokens"]):
                raise ValueError("invalid benchmark natural prefix")
            ids = matrix_ids(plan, row)
            if (len(ids) != cohort["input_tokens"] or len(ids) + cohort["tokens"] > plan["context"]
                    or any(type(token) is not int or token < 0 for token in ids)
                    or hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() != row["prompt_sha256"]):
                raise ValueError("matrix prompt geometry/hash mismatch")
    return plan


def _run_matrix(args, plan, report):
    if __package__:
        from . import v100_rotorquant_live as live
        from . import v100_adaptive_qualification as qualification
    else:
        import v100_rotorquant_live as live
        import v100_adaptive_qualification as qualification
    output = Path(args.output)
    if output.exists():
        raise ValueError("matrix output must be fresh")
    raw = output.with_suffix(output.suffix + ".raw")
    raw.mkdir()
    client = live.Client(args.base, raw, timeout=args.timeout, limits=protocol.from_arguments(args))
    serial = itertools.count()
    def read(path):
        return client.call(f"control-{next(serial)}", path=path, object_response=True)
    report.update(schema="tensorfold.phase-matrix-result-v1", plan=plan, cohorts=[], passed=False)
    write_json(output, report)
    initial = read("/stats")
    effective = qualification.profile(initial)
    materialized = qualification.materialization(initial, effective)
    qualification.require_idle(initial)
    report["materialization"] = materialized
    instance = initial["server"]["instance_id"]
    if args.model != plan["model"]:
        raise ValueError("benchmark model differs from its immutable input plan")
    expected_kv = "int8" if plan["profile"] == "q8" else "rotorquant-iso4-sm70-v1"
    if initial["model"].get("kv_dtype") != expected_kv or initial["model"].get("context_length") != plan["context"]:
        raise ValueError("service differs from the selected cache/full-context profile")
    session = initial["capacity_plan"]["session_capacity"]
    capacity = session["resident_capacity"]
    if session.get("configured_ceiling") != plan["admin_cap"] or capacity > plan["admin_cap"]:
        raise ValueError("service does not use the selected administrator concurrency ceiling")
    report.update(instance_id=instance, actual_resident_capacity=capacity,
                  excluded_by_admin_cap=plan["excluded_levels"],
                  scope="one loaded immutable service; overlapping inclusive GPU-stream envelopes/host gaps, not kernel-only")
    def publish():
        write_json(output, report)
    publish()
    for cohort in plan["cohorts"]:
        row = {"cohort": cohort["id"], "level": cohort["level"], "input_tokens": cohort["input_tokens"],
               "repetition": cohort["repetition"], "warmup": cohort["warmup"], "requests": [], "passed": False}
        report["cohorts"].append(row)
        if cohort["level"] > capacity:
            row.update(status="unsupported-resident-capacity", rates=None)
            publish()
            continue
        stats = read("/stats")
        if stats["server"]["instance_id"] != instance or stats["load"]["requests_in_flight"]:
            raise ValueError("benchmark requires the same idle service and no unrelated jobs")
        gate = threading.Barrier(cohort["level"])
        results, errors = [None] * cohort["level"], [None] * cohort["level"]
        row.update(requests=results, errors=errors)
        def request(index):
            spec = cohort["requests"][index]
            body = {"model": args.model, "prompt": matrix_ids(plan, spec), "max_tokens": cohort["tokens"],
                    "temperature": 0, "draft": True, "ignore_eos": True, "stream": True,
                    "stream_options": {"include_usage": True}, "return_token_ids": True}
            try:
                gate.wait()
                started = time.perf_counter()
                pieces = []
                reply = client.call(f"cohort-{cohort['id']}-request-{index}", body, path="/v1/completions",
                                    observer=lambda event, text: (completion_fields(event), pieces.extend(text)))
                reply["choices"][0]["message"]["content"] = "".join(pieces) or None
                finished = time.perf_counter()
                completion = live.completion(reply, cohort["tokens"])
                usage, native = reply["usage"], reply["tensorfold"]
                token_sha = hashlib.sha256(",".join(map(str, completion["ids"])).encode()).hexdigest()[:12]
                if native.get("token_sha") != token_sha:
                    raise ValueError("independent full token-ID SHA mismatch")
                if (usage["prompt_tokens"] != spec["input_tokens"] or native.get("cached") != 0
                        or usage.get("prompt_tokens_details", {}).get("cached_tokens") != 0
                        or len(completion["ids"]) != cohort["tokens"] or completion["finish"] != "length"):
                    raise ValueError("cold full-work benchmark request did not conform")
                phases = {}
                for name, numerator in (("prefill", usage["prompt_tokens"]), ("decode", max(0, len(completion["ids"]) - 1))):
                    seconds = native.get(name + "_s")
                    phases[name + "_reported_host_tps"] = numerator / seconds if type(seconds) in (int, float) and seconds > 0 else None
                results[index] = {"reply": reply, "sent": started, "finished": finished,
                                  "prompt_sha256": spec["prompt_sha256"], **phases}
            except BaseException as error:
                errors[index] = {"type": type(error).__name__, "message": str(error)}
                gate.abort()
        tasks = []
        primary = None
        try:
            for index in range(cohort["level"]):
                task = Task(partial(request, index), name=f"phase-matrix-{index}")
                tasks.append(task)
                task.start()
        except BaseException as error:
            primary = error
            gate.abort()
        finally:
            drain(tasks, primary)
        row.update(requests=results, errors=errors)
        if any(errors):
            row["status"] = "request-failed"
            publish()
            break
        final = read("/stats")
        if final["server"]["instance_id"] != instance or final["load"]["requests_in_flight"]:
            raise ValueError("cohort did not retire on the same service")
        deadline = time.monotonic() + args.timeout
        while True:
            snapshot = read("/v1/runtime")["execution"]["work_observations"].get("benchmark")
            pending = None if not snapshot else snapshot.get("cohorts", {}).get(str(cohort["id"]))
            if pending and pending.get("events_ready") or time.monotonic() >= deadline:
                break
            time.sleep(.01)  # Only post-cohort event readiness; no model/per-step fence.
        if (not snapshot or snapshot["run_identity"] != plan["run_identity"]
                or snapshot.get("server_instance_id") != instance):
            raise ValueError("the required private worker observer is absent or belongs to another plan")
        evidence = snapshot["cohorts"].get(str(cohort["id"]))
        row["backend"] = evidence
        if not evidence or evidence.get("completed") != cohort["level"] or not evidence.get("events_ready"):
            row["status"] = "unmeasured-worker-envelope"
            publish()
            break
        if evidence["prompt_hashes"] != sorted(r["prompt_sha256"] for r in cohort["requests"]):
            raise ValueError("worker prompt identity differs from the actual benchmark cohort")
        input_count = cohort["level"] * cohort["input_tokens"]
        decode_count = sum(max(len(live.completion(r["reply"], cohort["tokens"])["ids"]) - 1, 0) for r in results)
        prefill_seconds, decode_seconds = evidence["prefill_ms"] / 1000, evidence["decode_ms"] / 1000
        row["rates"] = {"prefill_envelope_tps": input_count / prefill_seconds if prefill_seconds > 0 else None,
                        "decode_envelope_tps": decode_count / decode_seconds if decode_seconds > 0 else None,
                        "cohort_http_wall_tps": sum(r["reply"]["usage"]["completion_tokens"] for r in results)
                            / (max(r["finished"] for r in results) - min(r["sent"] for r in results))}
        row["passed"] = evidence["active_owner_highwater"] >= cohort["level"] and all(value is not None for value in row["rates"].values())
        row["status"] = "measured" if row["passed"] else "unmeasured-phase-or-overlap"
        publish()
    report["unrun_cohorts"] = [c["id"] for c in plan["cohorts"] if c["id"] not in {r["cohort"] for r in report["cohorts"]}]
    report["passed"] = not report["unrun_cohorts"] and all(row["passed"] for row in report["cohorts"])
    publish()
    return 0 if report["passed"] else 2


def run_matrix(args, plan):
    output = Path(args.output)
    if output.exists() or output.with_suffix(output.suffix + ".raw").exists():
        raise ValueError("matrix output and raw directory must be fresh")
    report = {"schema": "tensorfold.phase-matrix-result-v1", "plan": plan, "cohorts": [], "passed": False}
    try:
        return _run_matrix(args, plan, report)
    except BaseException as error:
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        report["passed"] = False
        report["unrun_cohorts"] = [c["id"] for c in plan["cohorts"]
                                  if c["id"] not in {r["cohort"] for r in report["cohorts"]}]
        try:
            write_json(output, report)
        except BaseException as publication:
            raise_failures(error, (publication,))
        raise


def _repro_reply(live, reply, body):
    completion = live.completion(reply, 128)
    runtime, usage = reply["tensorfold"], reply["usage"]
    digest = hashlib.sha256(",".join(map(str, completion["ids"])).encode()).hexdigest()[:12]
    if (runtime.get("token_sha") != digest or len(completion["ids"]) != 128
            or completion["finish"] != "length" or usage["prompt_tokens"] != 64
            or type(runtime.get("cached")) is not int or runtime["cached"] != 0
            or type(usage.get("prompt_tokens_details", {}).get("cached_tokens")) is not int
            or usage["prompt_tokens_details"]["cached_tokens"] != 0
            or runtime.get("drafts") is not body["draft"]):
        raise ValueError("repro requires complete hashed cold64/128 replies and actual dispatch")
    for key in ("drafted", "accepted"):
        value = runtime.get(key)
        if type(value) is not int or value < 0 or (not body["draft"] and value != 0):
            raise ValueError("invalid actual repro proposal counters")
    if runtime["accepted"] > runtime["drafted"]:
        raise ValueError("repro accepted counter exceeds proposals")
    return completion


def load_repro(path, model):
    with path.open("rb") as source:
        raw = source.read((16 << 20) + 1)
    if len(raw) > 16 << 20:
        raise ValueError("repro DATA exceeds16MiB")
    data = protocol.object_json(raw)
    data["_source_sha256"] = hashlib.sha256(raw).hexdigest()
    if data.get("schema") != "tensorfold.raw-completion-repro-input-v1" or len(data.get("cases", [])) != 8:
        raise ValueError("repro requires eight preserved completion cases")
    for index, case in enumerate(data["cases"]):
        body = case["body"]
        ids = body.get("prompt")
        if (type(case.get("index")) is not int or case["index"] != index or type(ids) is not list or len(ids) != 64
                or any(type(token) is not int or not 0 <= token < 2**31 for token in ids)
                or body.get("model") != model or type(body.get("max_tokens")) is not int or body["max_tokens"] != 128
                or type(body.get("temperature")) not in (int, float) or body["temperature"] != 0
                or any(body.get(key) is not True for key in ("draft", "ignore_eos", "stream", "return_token_ids"))
                or body.get("stream_options") != {"include_usage": True}):
            raise ValueError("repro requires original indexed greedy64/128 completion bodies")
        digest = hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        if digest != case.get("body_sha256"):
            raise ValueError("repro body provenance differs")
    configuration = data.get("model_configuration")
    if (type(configuration) is not dict or configuration.get("id") != model
            or configuration.get("kv_dtype") != "rotorquant-iso4-sm70-v1"
            or configuration.get("context_length") != 262144 or configuration.get("mtp_drafts") != 4):
        raise ValueError("recorded full-context ISO4 MTP4 repro model configuration required")
    return data


def run_repro(args, data):
    if __package__:
        from . import v100_rotorquant_live as live
        from . import v100_adaptive_qualification as qualification
    else:
        import v100_rotorquant_live as live
        import v100_adaptive_qualification as qualification
    output = Path(args.output)
    raw = output.with_suffix(output.suffix + ".raw")
    if output.exists() or raw.exists():
        raise ValueError("repro output and raw directory must be fresh")
    for case in data["cases"]:
        _repro_reply(live, case["reference"], case["body"])
    raw.mkdir(parents=True)
    client = live.Client(args.base, raw, timeout=args.timeout, limits=protocol.from_arguments(args))
    report = {"schema": "tensorfold.raw-completion-repro-result-v1", "diagnostic_only": True,
              "performance_claim": False, "input_sha256": data["_source_sha256"],
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "stages": [], "completed": False, "differences": [], "conditional_repeat_required": None}
    control = itertools.count()
    initial = None
    def idle():
        stats = client.call(f"control-{next(control)}", path="/stats", object_response=True)
        effective = qualification.profile(stats)
        qualification.require_idle(stats)
        expert = stats.get("capacity_plan", {}).get("expert_parallel", {})
        if any(expert.get(key) != value for key, value in
               (("tp", 8), ("pp", 1), ("etp", 4), ("ep", 2), ("implementation", "owned-etp4-native-slots-v1"))):
            raise ValueError("repro requires the unchanged TP8/ETP4 owned-native implementation")
        effective["expert_parallel"] = expert
        if (effective["model"] != data["model_configuration"] or effective["context_tokens"] != 262144
                or effective["resident_capacity"] != 32 or effective["session_capacity"].get("configured_ceiling") != 32):
            raise ValueError("service differs from original full-context cap32 repro configuration")
        if initial is not None and effective != initial:
            raise ValueError("repro service instance/configuration changed")
        return stats, effective
    def stage(name, indices, draft):
        if getattr(args, "router_row_stage", None) is not None:
            write_json(args.router_row_stage, {"schema": "tensorfold.router-row-stage-v1",
                "run_identity": args.router_row_run_id, "stage": name})
        before, _ = idle()
        row = {"name": name, "indices": indices, "draft": draft, "requests": [None] * len(indices),
               "errors": [None] * len(indices), "stats_before": before}
        report["stages"].append(row)
        write_json(output, report)
        gate = threading.Barrier(len(indices))
        def request(position, index):
            released = False
            try:
                body = dict(data["cases"][index]["body"])
                if not draft:
                    body["draft"] = False  # The only permitted serial variant.
                gate.wait()
                released = True
                pieces = []
                reply = client.call(f"{name}-{index}", body, path="/v1/completions",
                    observer=lambda event, text: (completion_fields(event), pieces.extend(text)))
                reply["choices"][0]["message"]["content"] = "".join(pieces) or None
                _repro_reply(live, reply, body)
                row["requests"][position] = {"index": index, "body": body, "reply": reply,
                    "reference_comparison": live.compare(data["cases"][index]["reference"], reply, 128)}
            except BaseException as error:  # Task/drain retains the authentic primary after all borrowers stop.
                try:
                    row["errors"][position] = {"type": type(error).__name__, "message": str(error)}
                    if not released:
                        gate.abort()
                except BaseException as recording:
                    raise_failures(error, [recording])
                raise
        tasks, primary = [], None
        try:
            for position, index in enumerate(indices):
                task = Task(partial(request, position, index), name=f"repro-{name}-{index}")
                tasks.append(task)
                task.start()
        except BaseException as error:
            primary = error
            gate.abort()
        finally:
            drain(tasks, primary)
        row["stats_after"], _ = idle()
        write_json(output, report)
        return row
    primary = None
    try:
        stats, initial = idle()
        report["initial_stats"] = stats
        report["materialization"] = qualification.materialization(stats, initial)
        first = stage("baseline-c8-mtp", list(range(8)), True)
        report["conditional_repeat_required"] = any(not item["reference_comparison"]["passed"] for item in first["requests"])
        if report["conditional_repeat_required"]:
            stage("baseline-c8-mtp-repeat", list(range(8)), True)
        for index in (2, 7):
            stage(f"alone-{index}-mtp", [index], True)
            stage(f"alone-{index}-serial", [index], False)
        stage("baseline-c8-serial", list(range(8)), False)
        by_index = {index: [] for index in range(8)}
        for row in report["stages"]:
            for item in row["requests"]:
                if item is not None:
                    by_index[item["index"]].append((row["name"], item["reply"]))
        for index, replies in by_index.items():
            for left, right in itertools.combinations(replies, 2):
                comparison = live.compare(left[1], right[1], 128)
                report["differences"].append({"index": index, "left": left[0], "right": right[0], **comparison})
        report["completed"] = True
        report["all_exact"] = (all(item["reference_comparison"]["passed"] for row in report["stages"]
                                  for item in row["requests"])
                               and all(item["passed"] for item in report["differences"]))
        return 0  # Exact differences are diagnostic data, not a numerical acceptance gate.
    except BaseException as error:
        primary = error
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        cleanup = []
        planned = ["baseline-c8-mtp", "alone-2-mtp", "alone-2-serial", "alone-7-mtp", "alone-7-serial", "baseline-c8-serial"]
        if report["conditional_repeat_required"]:
            planned.insert(1, "baseline-c8-mtp-repeat")
        report["unrun_stages"] = [name for name in planned if name not in {row["name"] for row in report["stages"]}]
        try:
            report["final_stats"], _ = idle()
        except BaseException as error:
            report["completed"] = False
            report["final_stats_error"] = {"type": type(error).__name__, "message": str(error)}
            if primary is None:
                report["error"] = report["final_stats_error"]
            cleanup.append(error)
        try:
            write_json(output, report)
        except BaseException as error:
            cleanup.append(error)
        raise_failures(primary, cleanup)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("base")
    p.add_argument("model")
    p.add_argument("--levels", default="1,2,4,8,16,24,32", help="concurrent requests, a comma list")
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
    p.add_argument("--router-row-stage", type=Path, help="private atomic phase marker for router capture")
    p.add_argument("--router-row-run-id", help="fresh private router-capture run identity")
    p.add_argument("--repro-input", type=Path, help="diagnostic replay of eight preserved raw completion bodies/references; differences are data")
    p.add_argument("--prepare-matrix", type=Path)
    p.add_argument("--matrix-plan", type=Path)
    p.add_argument("--profile", choices=tuple(PROFILE_CAPS), default="q8")
    p.add_argument("--input-tokens", default="64")
    p.add_argument("--timeout", type=float, default=1800)
    protocol.add_arguments(p)
    args = p.parse_args()
    limits = protocol.from_arguments(args)
    if args.router_row_stage is not None or args.router_row_run_id is not None:
        if (args.repro_input is None or args.router_row_stage is None or type(args.router_row_run_id) is not str
                or len(args.router_row_run_id) != 32 or any(c not in "0123456789abcdef" for c in args.router_row_run_id)
                or args.router_row_stage.exists()):
            p.error("router marker requires repro, a fresh path and a lowercase32hex run identity")
    if args.repro_input is not None:
        if not args.output or args.prepare_matrix is not None or args.matrix_plan is not None:
            p.error("repro requires fresh --output and cannot be mixed with matrix modes")
        return run_repro(args, load_repro(args.repro_input, args.model))
    if args.prepare_matrix is not None or args.matrix_plan is not None:
        if not args.output or args.prepare_matrix is not None and args.matrix_plan is not None:
            p.error("matrix requires one preparation/run mode and a fresh --output")
        if args.prepare_matrix is not None:
            levels = sorted(set(int(value) for value in args.levels.split(",")))
            inputs = sorted(set(int(value) for value in args.input_tokens.split(",")))
            if any(value <= 0 for value in [*levels, *inputs, args.tokens, args.reps]):
                p.error("matrix sizes/levels/tokens/reps must be positive")
            if Path(args.output).exists():
                p.error("matrix plan output must be fresh")
            write_json(args.output, prepare_matrix(args.prepare_matrix, args.model, args.profile, levels, inputs, args.tokens, args.reps))
            return 0
        return run_matrix(args, load_matrix(args.matrix_plan))

    if args.strict and args.no_seed:
        p.error("--strict requires keyed requests; omit --no-seed")
    if args.no_seed:
        args.alone = args.serial = False
    levels = [int(x) for x in args.levels.split(",")]
    temps = [1.0] if args.mixed else [float(t) for t in args.temperatures.split(",")]
    if (any(n < 1 for n in levels) or args.tokens < 1 or args.reps < 1
            or any(not math.isfinite(t) or t < 0 for t in temps)
            or not math.isfinite(args.stagger_ms) or args.stagger_ms < 0):
        p.error("levels/tokens/reps must be positive; temperatures/stagger must be finite/nonnegative")
    cells = [None] if args.mixed else PROMPTS
    report: dict = {"label": args.label, "tokens": args.tokens, "mixed": args.mixed, "strict": args.strict,
                   "timing_unit": "SSE text-piece arrivals; not individual-token latency",
                   "response_budget_bytes": limits.response_bytes, "sse_line_budget_bytes": limits.line_bytes, "cells": []}
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
            warmup = together(args.base, args.model, specs[:1], 16, temp, limits=limits)
            warmup_failed = any(r.get("error") for r in warmup)
            failed |= warmup_failed
            alone: dict[tuple[str, int | None], dict] = {}
            if args.alone or args.serial:
                for spec in specs:
                    key = (spec[0]["name"], spec[1])
                    if key not in alone:
                        alone[key] = stream(args.base, args.model, spec[0], args.tokens, temp, spec[1], limits=limits)
            if args.serial:
                for key, ref in alone.items():
                    serial = stream(args.base, args.model, next(q for q in PROMPTS if q["name"] == key[0]),
                                    args.tokens, temp, key[1], draft=False, limits=limits)
                    ref["serial_equal"] = compare_hash(serial, ref)
                    ref["serial_error"] = serial.get("error") or ref.get("error")
                    failed |= bool(ref["serial_error"])
            for n in levels:
                reps = []
                for _ in range(args.reps):
                    runs = together(args.base, args.model, specs[:n], args.tokens, temp, args.stagger_ms, limits=limits)
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
        write_json(args.output, report)
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())

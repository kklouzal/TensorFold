"""Measure one production CUDA startup and explicit cold/warm text requests.

  python tools/model_phase_bench.py --plan PLAN.json --out NEW_DIRECTORY

The schema1 plan binds artifact_id, source_sha256 (tool and installed modules),
input_files (path/sha256 records), serve_args, port, cache_state and ordered
cases. Each case declares id, phase, body, prompt_tokens and cached_tokens.
The operator supplies exact cache expectations: dense Qwen normally retains
one fewer prompt token, while Gemma can retain the complete prompt. The tool
does not mutate engine caches. A fresh process is required for every startup
trial; file/compiler cache state is an operator-recorded experimental input.

Startup runs from immediately before the production CLI call until its App is
ready, including engine loading and CLI preparation but excluding this tool's
imports/input validation. Request timing consumes complete engine callbacks and
settles native work; it excludes HTTP/stream parsing. The CLI owns engine close.
Raw failed/partial outcomes remain in report.json and errors propagate.
Paired analysis must require identical prompt_sha256, response_tokens and full
token_ids for each corresponding case before attributing any speed difference.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

if __package__:
    from .benchmark_output import _compact_errors, write_json
    from .campaign_bench import HotGate, Progress, host_snapshot
    from .campaign_kl import with_app
else:
    from benchmark_output import _compact_errors, write_json
    from campaign_bench import HotGate, Progress, host_snapshot
    from campaign_kl import with_app


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read(path, limit=64 << 20):
    with Path(path).open("rb") as stream:
        raw = stream.read(limit + 1)
    if len(raw) > limit:
        raise ValueError(f"bounded benchmark input exceeds {limit} bytes: {path}")
    return raw


def _hash(value):
    return type(value) is str and re.fullmatch("[0-9a-f]{64}", value) is not None


def load_plan(path):
    path = Path(path)
    raw = _read(path, 1 << 20)
    plan = json.loads(raw)
    fields = {"schema", "artifact_id", "source_sha256", "input_files", "serve_args", "port", "cache_state", "cases"}
    if not isinstance(plan, dict) or set(plan) != fields or type(plan["schema"]) is not int or plan["schema"] != 1:
        raise ValueError("exact schema1 phase-benchmark fields required")
    if (type(plan["artifact_id"]) is not str or not plan["artifact_id"].startswith("sha256:")
            or not _hash(plan["artifact_id"][7:])):
        raise ValueError("operator-verified immutable image SHA256 required")
    sources = plan["source_sha256"]
    if (not isinstance(sources, dict) or "tool" not in sources or len(sources) < 2
            or any(not _hash(digest) for digest in sources.values())
            or any(name != "tool" and (type(name) is not str
                   or re.fullmatch(r"tensorfold(?:\.[a-zA-Z_]\w*)+", name) is None)
                   for name in sources)):
        raise ValueError("tool and affected installed TensorFold module hashes required")
    if sources["tool"] != _sha(_read(__file__)):
        raise ValueError("tool source differs from the sealed plan")
    serve = plan["serve_args"]
    if (not isinstance(serve, list) or len(serve) < 2 or serve[0] != "serve"
            or any(type(value) is not str for value in serve)
            or any(value.split("=", 1)[0] in {"--api-key", "--api-key-file"} for value in serve)):
        raise ValueError("nonsecret production serve arguments required")
    if type(plan["port"]) is not int or not 1 <= plan["port"] <= 65535:
        raise ValueError("loopback port must be an integer in1..65535")
    if type(plan["cache_state"]) is not str or not 1 <= len(plan["cache_state"]) <= 512:
        raise ValueError("an explicit file/compiler cache-state description is required")
    inputs = plan["input_files"]
    if not isinstance(inputs, list) or not inputs or len(inputs) > 64:
        raise ValueError("one through64 sealed ancillary or checkpoint-manifest files required")
    bound = []
    for entry in inputs:
        if (not isinstance(entry, dict) or set(entry) != {"path", "sha256"}
                or type(entry["path"]) is not str or not entry["path"] or not _hash(entry["sha256"])):
            raise ValueError("every input file needs its path and SHA256")
        target = Path(entry["path"])
        target = target if target.is_absolute() else path.parent / target
        digest = _sha(_read(target))
        if digest != entry["sha256"]:
            raise ValueError(f"input file hash changed: {target}")
        bound.append({"path": str(target.resolve()), "sha256": digest})
    cases = plan["cases"]
    if not isinstance(cases, list) or not 1 <= len(cases) <= 256:
        raise ValueError("one through256 explicit request cases required")
    names = set()
    for case in cases:
        if (not isinstance(case, dict) or set(case) != {"id", "phase", "body", "prompt_tokens", "cached_tokens"}
                or type(case["id"]) is not str or not case["id"] or len(case["id"]) > 128 or case["id"] in names
                or case["phase"] not in ("cold", "warm") or type(case["prompt_tokens"]) is not int
                or case["prompt_tokens"] < 1 or type(case["cached_tokens"]) is not int
                or not 0 <= case["cached_tokens"] <= case["prompt_tokens"]
                or (case["phase"] == "cold") != (case["cached_tokens"] == 0)):
            raise ValueError("unique cases need typed positive prompt counts and exact cold/warm cache expectations")
        body = case["body"]
        if (not isinstance(body, dict) or type(body.get("max_tokens")) is not int or body["max_tokens"] < 2
                or body.get("ignore_eos") is not True or type(body.get("draft", True)) is not bool
                or "messages" not in body or type(body.get("stream", True)) is not bool):
            raise ValueError("chat requests require at least2 tokens, ignore_eos=true and a boolean draft policy")
        # Strict outputs, media, logprobs and server-inserted stop/thought-budget work
        # belong to App.run; the direct engine lane here must never discard them.
        if any(key in body for key in ("tools", "response_format", "guided_json", "guided_regex", "guided_choice",
                                        "structured_outputs", "logprobs", "top_logprobs", "stop", "thinking_budget")):
            raise ValueError("phase requests support plain text generation only")
        messages = body["messages"]
        if (not isinstance(messages, list) or not messages or any(not isinstance(item, dict)
                or set(item) - {"role", "content"} or type(item.get("content")) is not str for item in messages)):
            raise ValueError("plain text role/content messages required")
        names.add(case["id"])
    if cases[0]["phase"] != "cold":
        raise ValueError("the first request of a fresh startup must be cold")
    return plan, raw, bound


def _failure(error):
    return {"type": type(error).__name__, "message": str(error)[:4096]}


def _settle(engine, torch):
    from tensorfold.cleanup import raise_failures

    primary, errors, count = None, [None, None], 0
    prior_cause = prior_context = None
    prior_suppress = False
    try:
        drain = getattr(engine, "drain", None)
        if drain is not None:
            drain()
    except BaseException as error:  # noqa: BLE001 - completion failure must not suppress GPU retirement
        primary = error
        prior_cause = BaseException.__cause__.__get__(error)
        prior_context = BaseException.__context__.__get__(error)
        prior_suppress = BaseException.__suppress_context__.__get__(error)
        errors[count] = error
        count += 1
    try:
        torch.cuda.synchronize()  # only this process's current CUDA device, under the experiment's exclusive guard
    except BaseException as error:  # noqa: BLE001 - report deferred native errors
        errors[count] = error
        count += 1
    while len(errors) > count:
        errors.pop()
    if primary is not None:
        BaseException.__cause__.__set__(primary, prior_cause)
        BaseException.__context__.__set__(primary, prior_context)
        BaseException.__suppress_context__.__set__(primary, prior_suppress)
    raise_failures(primary, errors)


def _attempt(app, case, row, report, output, torch):
    from tensorfold.cleanup import raise_failures

    engine = app.engine
    row["status"] = "RUNNING"
    write_json(output, report)
    ids, first, last = [], None, None
    primary, errors, assembly_errors, error_count = None, [None] * 5, [None], 0
    prior_cause = prior_context = None
    prior_suppress = False
    began = returned = settled = prepare_started = prepared_at = None
    prepared = None
    retirement_attempted = retired = False
    invalid_stats = False
    # Retain partial IDs even if later diagnostic preparation cannot allocate.
    row.update(token_ids=ids, native_settled=False, diagnostic_failed=False)

    def receive(tokens):
        nonlocal first, last
        now = time.perf_counter()
        if not isinstance(tokens, (list, tuple)) or any(type(token) is not int or token < 0 for token in tokens):
            raise ValueError("engine callbacks must carry nonnegative integer token IDs")
        if len(ids) + len(tokens) > case["body"]["max_tokens"]:
            raise ValueError("engine callback exceeded the declared output budget")
        if tokens:
            first = now if first is None else first
            last = now
            ids.extend(tokens)
            Progress.update("decode", request=case["id"], chunks=len(ids))
        return False

    try:
        row["hot_wait_s"] = HotGate.wait()
        _settle(engine, torch)
        row["host_before"] = host_snapshot()
        Progress.update("prefill", request=case["id"], prompt_tokens=case["prompt_tokens"],
                        max_tokens=case["body"]["max_tokens"], force=True)
        prepare_started = time.perf_counter()
        prepared = app.prepare(case["body"], True)
        prepared_at = time.perf_counter()
        if (len(prepared.prompt) != case["prompt_tokens"] or prepared.max_tokens != case["body"]["max_tokens"]
                or not prepared.ignore_eos or prepared.vision is not None or prepared.grammar is not None
                or prepared.calls is not None or prepared.stop or prepared.think_budget):
            raise ValueError("prepared request changed its declared counts or needs server-owned transformations")
        kwargs = {"draft": case["body"].get("draft", True), "stop_eos": False}
        if prepared.penalties is not None:
            kwargs["penalties"] = prepared.penalties
        began = time.perf_counter()
        stats = engine.generate(prepared.prompt, prepared.max_tokens, prepared.sampling, receive, **kwargs)
        returned = time.perf_counter()
        retirement_attempted = True
        _settle(engine, torch)
        settled, retired = time.perf_counter(), True
        try:
            json.dumps(stats, allow_nan=False)
        except (TypeError, ValueError):
            invalid_stats = True
            raise
        row["stats"] = stats
        if not isinstance(stats, dict):
            raise TypeError("engine statistics must be an object")
        for key in ("cached", "rounds", "drafted", "accepted", "min_rows"):
            if key in stats and (type(stats[key]) is not int or stats[key] < 0):
                raise ValueError(f"engine {key} must be a nonnegative integer")
        for key in ("prefill_s", "decode_s"):
            if type(stats.get(key)) not in (int, float) or not math.isfinite(stats[key]) or stats[key] < 0:
                raise ValueError(f"engine {key} must be a finite nonnegative duration")
        if stats.get("cached") != case["cached_tokens"]:
            raise ValueError("request did not exercise its declared prefix-cache state")
        if len(ids) != prepared.max_tokens:
            raise ValueError("request returned fewer than its complete declared token count")
    except BaseException as error:  # noqa: BLE001 - partial inference/interrupt evidence survives
        primary = error
        prior_cause = BaseException.__cause__.__get__(error)
        prior_context = BaseException.__context__.__get__(error)
        prior_suppress = BaseException.__suppress_context__.__get__(error)
    if not retirement_attempted:
        try:
            _settle(engine, torch)
            retired = True
        except BaseException as error:  # noqa: BLE001 - preserve native retirement alongside the primary failure
            errors[error_count] = error
            error_count += 1
    row["native_settled"] = retired
    row["status"] = "FAILED" if primary is not None or error_count else "COMPLETE"
    try:
        settled = time.perf_counter() if settled is None else settled
        if prepared is not None:
            row["prompt_sha256"] = _sha(json.dumps(prepared.prompt, separators=(",", ":")).encode())
        row.update(response_tokens=len(ids),
                   token_ids_sha256=_sha(json.dumps(ids, separators=(",", ":")).encode()),
                   first_s=None if first is None or began is None else first - began,
                   last_s=None if last is None or began is None else last - began,
                   generated_s=None if returned is None else returned - began,
                   settled_wall_s=None if began is None else settled - began,
                   prepare_s=None if prepared_at is None else prepared_at - prepare_started,
                   prepared_request_wall_s=None if prepare_started is None else settled - prepare_started,
                   first_from_prepare_s=None if first is None else first - prepare_started,
                   deferred_tail_s=None if returned is None else settled - returned,
                   retirement_errors=[_failure(errors[index]) for index in range(error_count)],
                   host_after=host_snapshot())
        if invalid_stats:
            row["invalid_stats_repr"] = repr(stats)[:4096]
        if primary is not None:
            row["error"] = _failure(primary)
    except BaseException as error:  # noqa: BLE001 - formatting must not bypass heartbeat reset or publication
        errors[error_count] = error
        error_count += 1
        row["status"] = "FAILED"
        row["diagnostic_failed"] = True
    try:
        Progress.update("idle", force=True)
    except BaseException as error:  # noqa: BLE001 - heartbeat failure cannot replace inference or suppress its report
        errors[error_count] = error
        error_count += 1
        row["status"] = "FAILED"
        try:
            row["progress_error"] = _failure(error)
        except BaseException as diagnostic:  # noqa: BLE001 - retain both heartbeat and formatter failures
            errors[error_count] = diagnostic
            error_count += 1
            row["diagnostic_failed"] = True
    try:
        write_json(output, report)
    except BaseException as error:  # noqa: BLE001 - publication cannot replace an inference failure
        errors[error_count] = error
        error_count += 1
        row["status"] = "FAILED"
    if primary is None and not error_count:
        return
    if primary is None:
        primary = errors[0]
        prior_cause = BaseException.__cause__.__get__(primary)
        prior_context = BaseException.__context__.__get__(primary)
        prior_suppress = BaseException.__suppress_context__.__get__(primary)
    BaseException.__cause__.__set__(primary, prior_cause)
    BaseException.__context__.__set__(primary, prior_context)
    BaseException.__suppress_context__.__set__(primary, prior_suppress)
    try:
        if error_count:
            _compact_errors(errors, error_count)
            raise_failures(primary, errors)
        else:
            raise_failures(primary, ())
    except BaseException as assembly:  # Final status assembly cannot replace inference or cleanup roots.
        if assembly is primary:
            raise
        assembly_errors[0] = assembly
        BaseException.__cause__.__set__(primary, prior_cause)
        BaseException.__context__.__set__(primary, prior_context)
        BaseException.__suppress_context__.__set__(primary, prior_suppress)
        raise_failures(primary, assembly_errors)


def run(plan_path, destination):
    from tensorfold.cleanup import raise_failures

    plan, raw, bound = load_plan(plan_path)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    output = destination / "report.json"
    (destination / "plan.json").write_bytes(raw)
    report = {"schema": 1, "status": "STARTING", "diagnostic_failed": False,
              "plan_sha256": _sha(raw), "plan": plan,
              "verified_inputs": bound, "tool_sha256": _sha(_read(__file__)), "host_before": host_snapshot(),
              "startup_scope": "production CLI entry to ready App; tool imports and input validation excluded",
              "records": [{"id": case["id"], "phase": case["phase"], "status": "NOT_RUN"}
                          for case in plan["cases"]]}
    write_json(output, report)
    primary, errors, error_count = None, [None] * 4, 0
    prior_cause = prior_context = None
    prior_suppress = False
    started = None

    def work(app):
        report["startup_s"] = time.perf_counter() - started
        import torch

        sources = {"tool": report["tool_sha256"]}
        paths = {}
        for name in plan["source_sha256"]:
            if name != "tool":
                module = importlib.import_module(name)
                paths[name] = module.__file__
                sources[name] = _sha(_read(module.__file__))
        report.update(source_sha256=sources, source_paths=paths, host_ready=host_snapshot(), status="RUNNING",
                      runtime={"python": sys.version, "torch": torch.__version__, "cuda": torch.version.cuda,
                               "device": torch.cuda.get_device_name(), "capability": torch.cuda.get_device_capability(),
                               "affinity": sorted(os.sched_getaffinity(0)), "engine": type(app.engine).__module__,
                               "context": app.engine.context_window, "served": app.served})
        if sources != plan["source_sha256"]:
            raise ValueError("installed source differs from the sealed plan")
        write_json(output, report)
        for case, row in zip(plan["cases"], report["records"]):
            _attempt(app, case, row, report, output, torch)

    try:
        HotGate.configure(None)
        Progress.configure(str(destination / "progress.json"))
        report["startup_hot_wait_s"] = HotGate.wait()
        Progress.update("startup", force=True)
        started = time.perf_counter()
        with_app(plan["serve_args"], plan["port"], work)
        report["status"] = "COMPLETE"
    except BaseException as error:  # noqa: BLE001 - CLI startup and close failures retain their own outcome
        primary = error
        prior_cause = BaseException.__cause__.__get__(error)
        prior_context = BaseException.__context__.__get__(error)
        prior_suppress = BaseException.__suppress_context__.__get__(error)
        report["status"] = "FAILED"
    try:
        report["cli_lifetime_s"] = None if started is None else time.perf_counter() - started
        report["host_after"] = host_snapshot()
        if primary is not None:
            report["error"] = _failure(primary)
    except BaseException as error:  # noqa: BLE001 - CLI has closed; preserve diagnostics independently
        errors[error_count] = error
        error_count += 1
        report["status"] = "FAILED"
        report["diagnostic_failed"] = True
    try:
        Progress.update("idle", force=True)
    except BaseException as error:  # noqa: BLE001 - retain progress and CLI failure independently
        errors[error_count] = error
        error_count += 1
        report["status"] = "FAILED"
        try:
            report["progress_error"] = _failure(error)
        except BaseException as diagnostic:  # noqa: BLE001 - publication still runs after formatter failure
            errors[error_count] = diagnostic
            error_count += 1
            report["diagnostic_failed"] = True
    try:
        write_json(output, report)
    except BaseException as error:  # noqa: BLE001 - retain CLI and publication failures
        errors[error_count] = error
        error_count += 1
        report["status"] = "FAILED"
    while len(errors) > error_count:
        errors.pop()
    if primary is not None:
        BaseException.__cause__.__set__(primary, prior_cause)
        BaseException.__context__.__set__(primary, prior_context)
        BaseException.__suppress_context__.__set__(primary, prior_suppress)
    raise_failures(primary, errors)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True, help="new report directory")
    args = parser.parse_args()
    run(args.plan, args.out)


if __name__ == "__main__":
    main()

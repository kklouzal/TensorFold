"""ABA complete-configuration indexer benchmark; standard-library HTTP only.

Run RAM4-A1, GPU3-B, RAM4-A2 as separate phases on the SAME immutable installed
image, INT8/stock MTP depth3/optimized/context262144. This compares deployable
configurations, not a storage-only effect. The primary capacity-first objective
is four vs three materialized full windows. Serial cold/warm decode, prefill,
TTFT and end-to-end timing are reported independently, including regressions.

Default two repetition blocks give twelve timed requests/phase. Four distinct
one-token fillers precede each block, outside timing. Every measured body must
complete512 tokens; cold requires cached0, its immediate warm repeat cached127.
Same-condition IDs/messages/finish must match within and across all three arms.
Cold-vs-warm identity is separately visible, never assumed or hidden.

The fixed small sample is descriptive only: no equivalence, statistically
verified speedup or global winner is claimed. No arbitrary gain floor applies.
Up to eight blocks may be declared before running. Analysis retains every cell,
sample and range; it does not pretend that request repeats measure deployment
drift. HTTP wall includes the identical Client recording work in each arm.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import time
import uuid

if __package__:
    from . import v100_rotorquant_live as live
    from . import v100_indexer_archive_live as archive_live
    from .worker_lifetime import raise_failures
else:
    import v100_rotorquant_live as live
    import v100_indexer_archive_live as archive_live
    from worker_lifetime import raise_failures

PHASES = {"ram4-a1": ("ram", 4), "gpu3-b": ("gpu", 3), "ram4-a2": ("ram", 4)}
METRICS = ("decode_tps", "decode_s", "prefill_s", "ttft_s", "wall_s")


def load_bodies(directory=None, named=()):
    paths = {}
    if directory is not None:
        for path in sorted(Path(directory).glob("*-p128-*-o512-request.json")):
            paths[path.stem.removesuffix("-request")] = path
    for item in named:
        name, separator, path = item.partition("=")
        live.require(separator and re.fullmatch(r"[A-Za-z0-9_-]+", name), "named request uses NAME=PATH")
        live.require(name not in paths, "duplicate request name")
        paths[name] = Path(path)
    live.require(len(paths) == 3, "exactly three frozen512-output request DATA files required")
    rows, hashes, identities = [], {}, set()
    for name, path in sorted(paths.items()):
        live.require(path.is_file() and not path.is_symlink() and path.stat().st_size <= 1 << 20,
                     "bounded regular body DATA required")
        raw = path.read_bytes()
        body = live.protocol.object_json(raw)
        live.require(body.get("max_tokens") == 512 and body.get("temperature") == 0
                     and body.get("top_p", 1) == 1 and body.get("draft", True) is True
                     and body.get("return_token_ids") is True and not body.get("stream")
                     and body.get("chat_template_kwargs", {}).get("enable_thinking") is False
                     and type(body.get("messages")) is list and body["messages"]
                     and not body.get("tools") and body.get("n", 1) == 1,
                     "frozen greedy text512 body changed")
        identity = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
        live.require(identity not in identities, "three distinct request bodies required")
        identities.add(identity)
        hashes[name] = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()}
        rows.append((name, body))
    return rows, hashes


def capacity(stats, mode, slots):
    model, plan = stats.get("model", {}), stats.get("capacity_plan", {})
    live.require(model.get("kv_dtype") == "int8" and model.get("mtp_drafts") == 3
                 and model.get("context_length") == 262144 and plan.get("context_window") == 262144,
                 "benchmark requires INT8/MTPD3/context262144")
    live.require(stats.get("limits", {}).get("lanes") == slots and live.idle(stats), "idle matching-lane service required")
    if mode == "ram":
        return archive_live.capacity_proof(stats, parallel=slots, context=262144, kv_dtype="int8")
    window = plan.get("kv_full_windows", {})
    live.require(all(window.get(key) == slots for key in ("requested", "reserved", "stream_slots"))
                 and window.get("materialized") is True, "GPU baseline needs complete materialized windows")
    each, ranks = window.get("cache_slots_each"), window.get("ranks")
    live.require(live.integer(each, 262144) and window.get("startup_capacities") == [each] * slots
                 and type(ranks) is list and len(ranks) == 8, "all eight GPU cache receipts required")
    live.require(window.get("workspace_bytes", 0) >= 2 << 30, "existing2GiB workspace must remain protected")
    live.require(live.integer(window.get("allocated_cache_bytes"), 1), "positive GPU cache allocation required")
    for index, row in enumerate(ranks):
        live.require(row.get("rank") == index and row.get("materialized") is True
                     and row.get("startup_capacities") == [each] * slots
                     and row.get("allocated_cache_bytes") == window.get("allocated_cache_bytes"),
                     "GPU baseline rank materialization differs")
        live.require(live.integer(row.get("allocated_cache_bytes"), 1)
                     and live.integer(row.get("torch_allocated_bytes"), row["allocated_cache_bytes"])
                     and live.integer(row.get("torch_reserved_bytes"), row["torch_allocated_bytes"]),
                     "positive actual GPU rank allocation required")
    live.require((plan.get("indexer_history") or {}).get("mode", "gpu") == "gpu", "GPU-history arm required")
    return {"history": "gpu", "context": 262144, "parallel": slots, "windows": window}


def static_identity(stats):
    """Immutable deployment identity, excluding counters/allocator growth."""
    server, model, plan = stats.get("server"), stats.get("model"), stats.get("capacity_plan")
    live.require(type(server) is dict and type(model) is dict and type(plan) is dict
                 and type(server.get("instance_id")) is str and bool(server["instance_id"])
                 and type(server.get("backend")) is str and bool(server["backend"])
                 and type(model.get("id")) is str and bool(model["id"]), "complete static server/model identity required")
    slots = stats.get("limits", {}).get("lanes")
    live.require(live.integer(slots, 1) and live.integer(model.get("context_length"), 1)
                 and plan.get("context_window") == model["context_length"], "static lane/context identity required")
    return {"server": copy.deepcopy(server), "model": copy.deepcopy(model), "slots": slots, "context": plan["context_window"],
            "history": (plan.get("indexer_history") or {}).get("mode", "gpu")}


def verify_static(stats, expected):
    live.require(static_identity(stats) == expected, "server/model/history/slots/context changed during benchmark")


def measurement(result, cache, wall):
    complete = live.completion(result, 512)
    runtime = result["tensorfold"]
    live.require(len(complete["ids"]) == 512 and complete["finish"] == "length"
                 and result["usage"]["prompt_tokens"] == 128, "registered128/512 workload did not finish")
    live.require(runtime.get("drafts") is True and type(runtime.get("cached")) is int
                 and runtime["cached"] == (0 if cache == "cold" else 127),
                 "actual prefix condition differs from registered cold0/warm127")
    row = {"wall_s": wall}
    for key in METRICS[:-1]:
        value = runtime.get(key)
        live.require(type(value) in (int, float) and math.isfinite(value) and value > 0,
                     "positive finite server timing required: " + key)
        row[key] = value
    return {**row, "completion": complete, "runtime": runtime, "usage": result["usage"]}


def run_phase(args):
    bodies, hashes = load_bodies(args.bodies_dir, args.request_json)
    live.require(1 <= args.reps <= 8, "declare1..8 repetition blocks before measurement")
    live.require(re.fullmatch(r"sha256:[0-9a-f]{64}", args.image or ""), "immutable deployment image declaration required")
    live.require(not args.out.exists(), "fresh benchmark output required")
    args.out.mkdir(parents=True)
    client = live.Client(args.url, args.out, timeout=args.timeout, limits=live.protocol.from_arguments(args))
    mode, slots = PHASES[args.phase]
    report = {"complete": False, "passed": False, "phase": args.phase, "image": args.image,
              "reps": args.reps, "load": "serial closed-loop; one outstanding generation, no throughput pacing",
              "scope": "complete RAM4/GPU3 deployable configurations; not storage-only or full262K payload test",
              "objective": "feasible full-window capacity first; independent cold/warm serial performance reported",
              "uncertainty": "descriptive samples/ranges only; insufficient deployment replication for equivalence or verified speedup",
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "bodies": hashes, "rows": [], "attempts": [], "pressure": []}
    references = {}
    live.write_json(args.out / "registration.json", report)
    primary = None
    try:
        initial = client.call("stats-initial")
        report["capacity"] = capacity(initial, mode, slots)
        identity = static_identity(initial)
        report["static_identity"] = identity
        report["residency_scope"] = "public physical startup GPU/cache/NUMA archive receipts and kv snapshots; not whole-processRSS"
        for block in range(args.reps):
            for index in range(4):
                filler = {"model": bodies[0][1]["model"], "messages": [{"role": "user", "content":
                          f"Cache isolation marker {args.phase}-{block}-{index}-{uuid.uuid4().hex}. Reply only OK."}],
                          "max_tokens": 1, "temperature": 0, "top_p": 1, "draft": True,
                          "return_token_ids": True, "chat_template_kwargs": {"enable_thinking": False}}
                result = client.call(f"pressure-{block}-{index}", filler)
                report["pressure"].append({"block": block, "index": index, "runtime": result.get("tensorfold")})
            for name, body in bodies:
                for cache in ("cold", "warm"):
                    label = f"b{block}-{name}-{cache}"
                    before = client.call(label + "-stats-before")
                    verify_static(before, identity)
                    live.require(live.idle(before), "unregistered concurrent load before timed request")
                    arrival = time.time()
                    started = time.perf_counter()
                    result = client.call(label, copy.deepcopy(body))
                    wall = time.perf_counter() - started
                    report["attempts"].append({"label": label, "body": name, "cache": cache,
                                                "block": block, "wall_s": wall})
                    row = measurement(result, cache, wall)
                    key = (name, cache)
                    if key in references:
                        live.require(row["completion"] == references[key], "same-condition output changed within phase")
                    references.setdefault(key, row["completion"])
                    report["rows"].append({"body": name, "cache": cache, "block": block,
                                           "arrival_unix_s": arrival, **row})
                    verify_static(client.call(label + "-stats-after"), identity)
                    live.write_json(args.out / "summary.json", report)
        report["complete"] = report["passed"] = True
    except BaseException as error:
        primary = error
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        try:
            live.write_json(args.out / "summary.json", report)
        except BaseException as cleanup:
            raise_failures(primary, [cleanup])
    return report


def analyze(paths):
    reports = [json.loads(Path(path).read_text()) for path in paths]
    live.require([row.get("phase") for row in reports] == list(PHASES), "fixed RAM4-A1/GPU3-B/RAM4-A2 order required")
    live.require(all(row.get("complete") is True and row.get("passed") is True for row in reports), "all complete phases required")
    for field in ("image", "reps", "tool_sha256", "bodies"):
        live.require(all(row[field] == reports[0][field] for row in reports), "phase provenance changed: " + field)
    cells = {}
    for report in reports:
        for row in report["rows"]:
            key = (row["body"], row["cache"])
            values = cells.setdefault(key, {})
            values.setdefault(report["phase"], []).append(row)
    live.require(set(cells) == {(name, cache) for name in reports[0]["bodies"] for cache in ("cold", "warm")},
                 "all three bodies and both prefix conditions required")
    output = []
    for (body, cache), arms in sorted(cells.items()):
        live.require(set(arms) == set(PHASES) and all(len(v) == reports[0]["reps"] for v in arms.values()),
                     "complete registered performance cells required")
        live.require(all({row["block"] for row in rows} == set(range(reports[0]["reps"])) for rows in arms.values()),
                     "one sample per registered repetition block required")
        expected = arms["gpu3-b"][0]["completion"]
        live.require(all(row["completion"] == expected for arm in arms.values() for row in arm),
                     "complete IDs/messages/finish differ across deployable arms")
        distributions = {phase: {metric: {"all": [r[metric] for r in rows],
                         "median": statistics.median(r[metric] for r in rows),
                         "min": min(r[metric] for r in rows), "max": max(r[metric] for r in rows)}
                         for metric in METRICS} for phase, rows in arms.items()}
        center = {phase: statistics.mean(math.log(row["decode_tps"]) for row in rows) for phase, rows in arms.items()}
        ratio = math.exp((center["ram4-a1"] + center["ram4-a2"]) / 2 - center["gpu3-b"])
        output.append({"body": body, "cache": cache, "exact_output": True, "distributions": distributions,
                       "descriptive_bracket_ram4_over_gpu3_decode_ratio": ratio})
    cache_identity = {report["phase"]: {name: next(row["completion"] for row in report["rows"]
                      if row["body"] == name and row["cache"] == "cold") ==
                      next(row["completion"] for row in report["rows"] if row["body"] == name and row["cache"] == "warm")
                      for name in report["bodies"]} for report in reports}
    return {"complete": True, "scope": "capacity-first4vs3; descriptiveABA, not isolated-storage or verified speedup/equivalence",
            "cold_vs_warm_identity_separate": cache_identity,
            "cells": output, "phase_paths": list(map(str, paths))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=(*PHASES, "analyze"), required=True)
    parser.add_argument("--url", default="http://127.0.0.1:8001")
    parser.add_argument("--bodies-dir", type=Path)
    parser.add_argument("--request-json", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--phase-summary", action="append", type=Path, default=[])
    parser.add_argument("--timeout", type=float, default=900)
    live.protocol.add_arguments(parser)
    args = parser.parse_args()
    result = analyze(args.phase_summary) if args.phase == "analyze" else run_phase(args)
    if args.phase == "analyze":
        live.write_json(args.out, result)
    print(json.dumps({key: value for key, value in result.items() if key not in ("rows", "cells")}, indent=2))


if __name__ == "__main__":
    main()

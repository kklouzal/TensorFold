"""Capture platform facts and exercise the adaptive service with existing HTTP tools.

Inventory and a generated test matrix are not qualification. The benchmark
command measures short-prompt SSE workloads only; it preserves failures and
lists the matrix gates it has not exercised. Run transfer probes separately
from serving measurements. No command changes GPU clocks, power, or placement.
"""
from __future__ import annotations

import argparse
import hashlib
import math
import os
from pathlib import Path
import platform
import selectors
import signal
import subprocess
import sys
import time

if __package__:
    from . import bench_concurrent as bench
    from .benchmark_output import write_json
    from . import openai_protocol as protocol
    from .worker_lifetime import raise_failures
else:
    import bench_concurrent as bench
    from benchmark_output import write_json
    import openai_protocol as protocol
    from worker_lifetime import raise_failures


def positive(value, name):
    if type(value) is not int or value <= 0:
        raise ValueError(name + " must be a positive integer")
    return value


def capture(argv, *, timeout=15.0, limit=1 << 20):
    """Bound retained bytes and wall time; retire the command's whole process group."""
    if not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("command timeout must be finite and positive")
    positive(limit, "command output limit")
    started, output, status, process = time.monotonic(), bytearray(), "ok", None
    result = {"argv": list(argv), "status": "unavailable", "exit_code": None}
    try:
        process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, start_new_session=True)
    except OSError as error:
        result["error"] = {"type": type(error).__name__, "message": str(error)}
        return result
    primary = None
    try:
        with selectors.DefaultSelector() as ready:
            ready.register(process.stdout, selectors.EVENT_READ)
            while True:
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    status = "timeout"
                    break
                events = ready.select(remaining)
                if not events:
                    status = "timeout"
                    break
                chunk = os.read(process.stdout.fileno(), min(65536, limit - len(output) + 1))
                if not chunk:
                    # Keep the leader unreaped until group retirement, so its
                    # PID/PGID cannot be recycled before killpg. stdout may
                    # close before normal process completion.
                    while os.waitid(os.P_PID, process.pid, os.WEXITED | os.WNOWAIT | os.WNOHANG) is None:
                        remaining = timeout - (time.monotonic() - started)
                        if remaining <= 0:
                            status = "timeout"
                            break
                        time.sleep(min(.01, remaining))
                    break
                output.extend(chunk)
                if len(output) > limit:
                    del output[limit:]
                    status = "output-limit"
                    break
    except BaseException as error:
        primary = error
        raise
    finally:
        # A descendant can retain the pipe after its parent exits. The process
        # group remains the cleanup owner even if poll() reports parent exit.
        errors = []
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        except BaseException as cleanup:
            errors.append(cleanup)
        try:
            process.wait(timeout=5)
        except BaseException as cleanup:
            errors.append(cleanup)
        try:
            process.stdout.close()
        except BaseException as cleanup:
            errors.append(cleanup)
        if errors:
            raise_failures(primary, errors)
    code = process.returncode
    result.update(status=status if status != "ok" else ("ok" if code == 0 else "failed"),
                  exit_code=code, elapsed_s=time.monotonic() - started,
                  output_utf8=output.decode("utf-8", errors="replace"),
                  output_sha256=hashlib.sha256(output).hexdigest())
    return result


GPU_FIELDS = "index,uuid,name,pci.bus_id,memory.total,ecc.mode.current,driver_version"
COMMANDS = {
    "gpus": ["nvidia-smi", "--query-gpu=" + GPU_FIELDS, "--format=csv,noheader,nounits"],
    "gpu_state": ["nvidia-smi", "-q", "-x"],
    "topology": ["nvidia-smi", "topo", "-m"],
    "peer_access_read": ["nvidia-smi", "topo", "-p2p", "r"],
    "peer_access_write": ["nvidia-smi", "topo", "-p2p", "w"],
    "nvlink_state": ["nvidia-smi", "nvlink", "-s"],
    "nvlink_errors": ["nvidia-smi", "nvlink", "-e"],
    "pci_tree": ["lspci", "-Dtv"],
    "pci_devices": ["lspci", "-Dnn"],
    "cpu": ["lscpu", "--json"],
    "numa": ["numactl", "--hardware"],
    "dimm_population": ["dmidecode", "--type", "17"],
    "cuda_compiler": ["nvcc", "--version"],
    "host_compiler": ["c++", "--version"],
    "nsight_systems": ["nsys", "--version"],
    "nsight_compute": ["ncu", "--version"],
}


def inventory(*, rank_devices=(), timeout=15.0, command_limit=1 << 20):
    facts = {name: capture(argv, timeout=timeout, limit=command_limit) for name, argv in COMMANDS.items()}
    framework = """import importlib.metadata as metadata
import json
result = {'packages': {}, 'runtime': {}}
for package in ('torch', 'triton', 'nvidia-nccl-cu12', 'nvidia-cudnn-cu12'):
    try:
        result['packages'][package] = metadata.version(package)
    except metadata.PackageNotFoundError:
        result['packages'][package] = None
try:
    import torch
except ImportError as error:
    result['runtime']['status'] = 'unavailable'
    result['runtime']['error'] = type(error).__name__ + ': ' + str(error)
else:
    result['runtime'].update(status='available', torch_file=torch.__file__, cuda_runtime=torch.version.cuda)
    for name, query in (('cuda_available', torch.cuda.is_available),
                        ('nccl', torch.cuda.nccl.version), ('cudnn', torch.backends.cudnn.version)):
        try:
            result['runtime'][name] = query()
        except (RuntimeError, AttributeError) as error:
            result['runtime'][name] = {'status': 'unavailable', 'error': type(error).__name__ + ': ' + str(error)}
    result['runtime']['compiled_arches'] = torch.cuda.get_arch_list() if torch.cuda.is_available() else None
print(json.dumps(result))
"""
    facts["framework"] = capture([sys.executable, "-c", framework], timeout=timeout, limit=command_limit)
    source = capture(["git", "rev-parse", "HEAD"], timeout=timeout, limit=4096)
    devices = []
    if facts["gpus"]["status"] == "ok":
        import csv

        for fields in csv.reader(facts["gpus"]["output_utf8"].splitlines(), skipinitialspace=True):
            if len(fields) != 7:
                raise ValueError("GPU inventory does not match the requested seven-field schema")
            index, uuid, name, bus, memory, ecc, driver = fields
            devices.append({"device_index": int(index), "uuid": uuid, "name": name, "pci_bus_id": bus,
                            "memory_mib": int(memory), "ecc_current": ecc, "driver": driver,
                            "logical_rank": None})
    if rank_devices:
        if len(set(rank_devices)) != len(rank_devices):
            raise ValueError("rank devices must be unique physical indexes or UUIDs")
        matched = []
        for rank, identity in enumerate(rank_devices):
            candidates = [d for d in devices if identity in (str(d["device_index"]), d["uuid"])]
            if len(candidates) != 1 or candidates[0]["uuid"] in matched:
                raise ValueError("rank mapping must name each inventoried physical GPU at most once")
            candidates[0]["logical_rank"] = rank
            matched.append(candidates[0]["uuid"])
    files = {}
    for name in ("/proc/meminfo", "/proc/sys/kernel/numa_balancing"):
        try:
            with open(name, "rb") as source_file:
                data = source_file.read(command_limit + 1)
            if len(data) > command_limit:
                raise ValueError("inventory file exceeds its byte limit")
            files[name] = {"status": "ok", "output_utf8": data.decode("utf-8")}
        except OSError as error:
            files[name] = {"status": "unavailable", "error": type(error).__name__ + ": " + str(error)}
    return {"schema": "tensorfold.v100-platform", "schema_version": 1, "evidence": "inventory-only",
            "captured_unix_s": time.time(), "host": platform.node(), "platform": platform.platform(),
            "python": sys.version, "source": source, "devices": devices, "commands": facts, "files": files,
            "source_worktree": capture(["git", "status", "--porcelain"], timeout=timeout, limit=65536),
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "rank_mapping_source": "explicit" if rank_devices else "unassigned",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "pending": ["pairwise-transfer-bandwidth", "collective-latency-bandwidth",
                        "sustained-clock-thermal-and-link-error-sampling", "native-sm70-kernel-qualification"]}


def profile(stats):
    if not isinstance(stats, dict) or stats.get("object") != "tensorfold.stats":
        raise ValueError("ordinary public TensorFold stats required")
    plan = stats.get("capacity_plan") or {}
    solved = plan.get("session_capacity") or {}
    capacity = positive(solved.get("resident_capacity"), "computed resident capacity")
    context = positive(solved.get("context_tokens"), "effective context")
    model = stats.get("model") or {}
    commitments = solved.get("full_context_commitments")
    if type(commitments) is not int or commitments != capacity or model.get("context_length") != context:
        raise ValueError("equal full-context commitments and matching public context required")
    if not isinstance(model.get("id"), str) or not model["id"]:
        raise ValueError("served model identity required")
    instance = (stats.get("server") or {}).get("instance_id")
    if not isinstance(instance, str) or not instance:
        raise ValueError("server instance identity required")
    return {"resident_capacity": capacity, "context_tokens": context,
            "model": model, "instance_id": instance, "session_capacity": solved}


MATRIX_AXES = {
    "retention": ["one-resident-one-active", "all-resident-one-active", "all-resident-all-active",
                  "partly-filled-histories", "near-full-histories"],
    "workload": ["prefill-only", "decode-only", "mixed-long-prefill-decode",
                 "heterogeneous-acceptance", "routed-expert-skew"],
    "transition": ["degree-1-to-2", "degree-2-to-1", "degree-1-to-4", "degree-4-to-1",
                   "degree-2-to-4", "degree-4-to-2", "partial-migration", "threshold-oscillation",
                   "arrivals-during-migration", "burst-one-to-capacity"],
    "lifecycle": ["completion", "eos", "stop-string", "context-exhaustion", "cancellation",
                  "slot-reuse", "admission-limit-decrease", "admission-limit-increase", "pending-profile"],
    "speculation": ["all-accept", "early-rejection", "late-rejection", "zero-accept",
                    "zero-proposal-warm", "final-anchor-bonus", "accepted-prefix-recovery"],
    "sustained": ["fill-drain-cycles", "graph-replacement", "allocator-fragmentation", "bounded-migration"],
}


def matrix(stats, *, max_cases=100000):
    effective = profile(stats)
    capacity, context = effective["resident_capacity"], effective["context_tokens"]
    positive(max_cases, "matrix case limit")
    if capacity > max_cases:
        raise ValueError("computed capacity exceeds matrix case budget; raise --max-cases explicitly")
    # Every step is represented; thresholds never truncate a non-threshold or
    # above-32 endpoint. Other axes are explicit gates, not a Cartesian-product
    # claim that a short-prompt HTTP run exercised state migration or retention.
    return {"schema": "tensorfold.v100-adaptive-matrix", "schema_version": 1,
            "evidence": "test-plan-unrun", "profile": effective,
            "concurrency": list(range(1, capacity + 1)),
            "contexts": sorted({min(context, 4096), min(context, 65536), context}),
            "axes": MATRIX_AXES, "numerics": ["same-codec-owner-equivalence", "codec-quality-reference",
                                                "target-causal-future-token-leakage", "rng-stream-ownership"],
            "capacity_candidates": {"q8_mtp_sessions": 16, "rotorquant_q4_mtp_sessions": 24,
                                    "status": "design-planning-candidates-not-measured-maxima"}}


def load_json(path, *, limit=16 << 20):
    with Path(path).open("rb") as source:
        data = source.read(limit + 1)
    if len(data) > limit:
        raise ValueError("input JSON exceeds byte limit")
    return protocol.object_json(data)


def transfers(*, payload_bytes=16 << 20, reps=16):
    """Measure sequential copy paths; this is a diagnostic, never a serving rate."""
    positive(payload_bytes, "transfer payload bytes")
    positive(reps, "transfer repetitions")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("transfer measurement requires the actual CUDA host")
    count = torch.cuda.device_count()
    if not count:
        raise RuntimeError("no CUDA device is visible")
    for device in range(count):
        free, _total = torch.cuda.mem_get_info(device)
        if payload_bytes > free // 4:
            raise ValueError("transfer buffers exceed one quarter of current free device memory")
    rows = []
    host = torch.empty(payload_bytes, dtype=torch.uint8, pin_memory=True)
    host.fill_(29)

    def measure(source, destination, touched, kind, source_device, destination_device, peer):
        for _ in range(2):
            destination.copy_(source, non_blocking=True)
        for device in touched:
            torch.cuda.synchronize(device)
        started = time.perf_counter()
        for _ in range(reps):
            destination.copy_(source, non_blocking=True)
        for device in touched:
            torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - started
        rows.append({"kind": kind, "source_device": source_device, "destination_device": destination_device,
                     "peer_access": peer, "payload_bytes": payload_bytes, "repetitions": reps,
                     "elapsed_s": elapsed, "bytes_per_s": payload_bytes * reps / elapsed,
                     "timing_scope": "host-wall repeated copy plus terminal device synchronization"})

    for source_device in range(count):
        source = torch.empty(payload_bytes, dtype=torch.uint8, device=f"cuda:{source_device}")
        measure(host, source, [source_device], "host-to-device", None, source_device, None)
        measure(source, host, [source_device], "device-to-host", source_device, None, None)
        for destination_device in range(count):
            if source_device == destination_device:
                continue
            destination = torch.empty(payload_bytes, dtype=torch.uint8, device=f"cuda:{destination_device}")
            peer = torch.cuda.can_device_access_peer(source_device, destination_device)
            measure(source, destination, [source_device, destination_device], "device-to-device",
                    source_device, destination_device, peer)
            del destination
        del source
    return {"schema": "tensorfold.v100-transfer-diagnostic", "schema_version": 1,
            "evidence": "isolated-copy-measurement", "captured_unix_s": time.time(),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "visible_devices": count, "rows": rows,
            "limitations": ["not a serving throughput measurement", "not a concurrent shared-uplink measurement",
                            "collective timing remains separate", "device indexes are process-visible ordinals"]}


def counter_deltas(before, after):
    result = {}
    for key in ("prompt_tokens", "cached_prompt_tokens", "generated_tokens"):
        start, end = ((body.get("totals") or {}).get(key) for body in (before, after))
        if type(start) is not int or type(end) is not int or not 0 <= start <= end:
            raise ValueError("missing or regressing server token counter: " + key)
        result[key] = end - start
    if result["cached_prompt_tokens"] > result["prompt_tokens"]:
        raise ValueError("cached prompt delta exceeds total prompt delta")
    return result


def summarize(runs):
    failures = [i for i, run in enumerate(runs) if run.get("error")]
    finished = [run for run in runs if not run.get("error")]
    return {"offered": len(runs), "completed": len(finished), "failed_indexes": failures,
            "unmeasured": sum(bool(run.get("unmeasured")) for run in runs),
            "aggregate": bench.aggregates(runs),
            "ttft_s": [run["ttft_s"] for run in finished],
            "decode_tps": [run["decode_tps"] for run in finished],
            "inter_piece_s": [[b[0] - a[0] for a, b in zip(run["pieces"], run["pieces"][1:])]
                              for run in finished],
            "timing_scope": "SSE text-piece arrivals; pieces may contain several tokens; not exact TPOT"}


def require_idle(stats):
    load = stats.get("load") or {}
    if any(type(load.get(key)) is not int or load[key] != 0
           for key in ("requests_in_flight", "requests_running", "requests_waiting")):
        raise ValueError("exclusive idle service required outside each measured request batch")


def materialization(stats, effective):
    """Require actual TP8 startup ownership, separately from the header solver."""
    windows = (stats.get("capacity_plan") or {}).get("kv_full_windows") or {}
    capacity, context = effective["resident_capacity"], effective["context_tokens"]
    if windows.get("materialized") is not True:
        raise ValueError("live benchmark requires materialized full windows, not a header-only plan")
    if any(type(windows.get(key)) is not int or windows[key] != capacity
           for key in ("reserved", "stream_slots")):
        raise ValueError("materialized slot count differs from computed resident capacity")
    ranks = windows.get("ranks")
    if type(ranks) is not list or len(ranks) != 8:
        raise ValueError("all eight physical rank allocation receipts required")
    for rank, row in enumerate(ranks):
        if row.get("rank") != rank or row.get("materialized") is not True:
            raise ValueError("unordered or unmaterialized physical rank receipt")
        lengths = row.get("startup_capacities")
        if type(lengths) is not list or len(lengths) != capacity or any(
                type(length) is not int or length < context for length in lengths):
            raise ValueError("rank does not own every promised full-window slot")
        cache, allocated, reserved = (row.get(key) for key in (
            "allocated_cache_bytes", "torch_allocated_bytes", "torch_reserved_bytes"))
        if any(type(value) is not int for value in (cache, allocated, reserved)) or not 0 < cache <= allocated <= reserved:
            raise ValueError("rank physical allocator readings do not cover owned cache bytes")
    return windows


def benchmark(args):
    from urllib.request import urlopen

    positive(args.reps, "repetitions")
    positive(args.tokens, "generated token budget")
    positive(args.max_inflight, "benchmark worker budget")
    positive(args.batch_response_mib, "batch response MiB")
    if type(args.warmup) is not int or args.warmup < 0:
        raise ValueError("warmup must be a nonnegative integer")
    output = args.out
    if output.exists():
        raise ValueError("fresh benchmark output directory required")
    output.mkdir(parents=True)
    limits = protocol.from_arguments(args)

    def read_stats():
        with urlopen(args.url.rstrip("/") + "/stats", timeout=30) as response:
            protocol.response_type(response, "application/json")
            raw = response.read(limits.response_bytes + 1)
        if len(raw) > limits.response_bytes:
            raise ValueError("stats response exceeds byte limit")
        return protocol.object_json(raw)

    report = {"schema": "tensorfold.v100-adaptive-benchmark", "schema_version": 1,
              "evidence": "short-prompt-sse-workload", "complete": False, "passed": False,
              "cells": [], "warmup_requests": [],
              "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "http_tool_sha256": hashlib.sha256(Path(bench.__file__).read_bytes()).hexdigest(),
              "protocol_sha256": hashlib.sha256(Path(protocol.__file__).read_bytes()).hexdigest(),
              "workload": {"prompts": bench.PROMPTS, "tokens": args.tokens, "temperature": args.temperature,
                           "seed": args.seed, "repetitions": args.reps, "warmup_batches_per_level": args.warmup,
                           "ignore_eos": True},
              "tool_budgets": {"maximum_inflight": args.max_inflight,
                               "batch_response_bytes": args.batch_response_mib << 20,
                               "response_bytes_each": limits.response_bytes, "sse_line_bytes": limits.line_bytes},
              "limitations": ["no near-full-history payload qualification", "no retention or transition proof",
                              "no quality equivalence proof", "no sustained hardware telemetry",
                              "no statistical speedup or optimum claim"]}
    primary = active_cell = None
    try:
        initial = read_stats()
        require_idle(initial)
        required = matrix(initial, max_cases=args.max_cases)
        effective = required["profile"]
        report["startup_materialization"] = materialization(initial, effective)
        if args.tokens >= effective["context_tokens"]:
            raise ValueError("output budget leaves no context for the benchmark prompts")
        report.update(initial_stats=initial, matrix=required)
        model, capacity = effective["model"]["id"], effective["resident_capacity"]
        if capacity > args.max_inflight:
            raise ValueError("capacity exceeds benchmark worker budget; raise --max-inflight explicitly")
        if capacity * (args.reps + args.warmup) > args.max_cases:
            raise ValueError("total timed and warmup batches exceed --max-cases manifest budget")
        if capacity * limits.response_bytes > args.batch_response_mib << 20:
            raise ValueError("sum of request response budgets exceeds --batch-response-mib; "
                             "reduce --response-mib or explicitly raise the batch budget")
        report["pending_axes"] = required["axes"]
        for level in required["concurrency"]:
            specs = [(bench.PROMPTS[i % len(bench.PROMPTS)], args.seed + i) for i in range(level)]
            for warmup in range(args.warmup):
                runs = bench.together(args.url.rstrip("/"), model, specs, args.tokens, args.temperature,
                                      limits=limits)
                warmup_row = {"active_requests": level, "warmup": warmup, "runs": runs}
                warmup_receipt = {"active_requests": level, "warmup": warmup,
                                  "artifact": f"warmup-{len(report['warmup_requests']):06d}.json"}
                report["warmup_requests"].append(warmup_receipt)
                write_json(output / warmup_receipt["artifact"], warmup_row)
                write_json(output / "report.json", report)
                if any(run.get("error") for run in runs):
                    raise ValueError("warmup request failed; preserved in report")
                del warmup_row, runs
            for repetition in range(args.reps):
                before = read_stats()
                require_idle(before)
                if profile(before) != effective:
                    raise ValueError("server instance/model/context/capacity changed during benchmark")
                started = time.perf_counter()
                runs = bench.together(args.url.rstrip("/"), model, specs, args.tokens, args.temperature,
                                      limits=limits)
                elapsed = time.perf_counter() - started
                row = {"active_requests": level, "repetition": repetition, "wall_s": elapsed,
                       "runs": runs, "summary": summarize(runs), "stats_before": before}
                receipt = {"active_requests": level, "repetition": repetition,
                           "artifact": f"cell-{len(report['cells']):06d}.json", "validated": False, "passed": False}
                report["cells"].append(receipt)
                active_cell = receipt, row
                write_json(output / receipt["artifact"], row)
                write_json(output / "report.json", report)
                after = read_stats()
                row["stats_after"] = after
                require_idle(after)
                if profile(after) != effective:
                    raise ValueError("server instance/model/context/capacity changed during benchmark")
                delta = counter_deltas(before, after)
                completed_tokens = sum(run["tokens"] for run in runs if not run.get("error"))
                if delta["generated_tokens"] != completed_tokens:
                    raise ValueError("server generated-token delta differs from measured completed replies")
                row.update(server_counter_deltas=delta,
                           useful_prompt_tokens_per_wall_s=(delta["prompt_tokens"] - delta["cached_prompt_tokens"]) / elapsed,
                           committed_decode_tokens_per_wall_s=delta["generated_tokens"] / elapsed)
                receipt.update(validated=True, passed=not row["summary"]["failed_indexes"]
                               and not row["summary"]["unmeasured"]
                               and row["summary"]["completed"] == level
                               and all(run["tokens"] == args.tokens for run in runs))
                write_json(output / receipt["artifact"], row)
                write_json(output / "report.json", report)
                active_cell = None
                del row, runs
        report["complete"] = True
        report["passed"] = all(cell["passed"] for cell in report["cells"])
        report["final_stats"] = read_stats()
        return 0 if report["passed"] else 2
    except BaseException as error:
        primary = error
        report["error"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        errors = []
        if active_cell is not None:
            receipt, row = active_cell
            try:
                write_json(output / receipt["artifact"], row)
            except BaseException as publication:
                errors.append(publication)
        try:
            write_json(output / "report.json", report)
        except BaseException as publication:
            errors.append(publication)
        if errors:
            raise_failures(primary, errors)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_subparsers(dest="mode", required=True)
    inv = modes.add_parser("inventory", help="bounded read-only hardware/runtime snapshot")
    inv.add_argument("--out", type=Path, required=True)
    inv.add_argument("--rank-devices", default="", help="logical-rank order of explicit GPU indexes or full UUIDs")
    inv.add_argument("--timeout", type=float, default=15)
    inv.add_argument("--command-mib", type=int, default=1)
    plan = modes.add_parser("matrix", help="derive all concurrency endpoints and pending qualification axes")
    plan.add_argument("--stats", type=Path, required=True)
    plan.add_argument("--out", type=Path, required=True)
    plan.add_argument("--max-cases", type=int, default=100000, help="tool-only case budget; never a serving capacity cap")
    transfer = modes.add_parser("transfers", help="isolated pairwise and pinned-host copies; run outside serving benchmarks")
    transfer.add_argument("--out", type=Path, required=True)
    transfer.add_argument("--payload-mib", type=int, default=16)
    transfer.add_argument("--reps", type=int, default=16)
    run = modes.add_parser("benchmark", help="measure existing SSE concurrency workloads, preserving every request")
    run.add_argument("--url", required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--reps", type=int, default=3)
    run.add_argument("--warmup", type=int, default=1)
    run.add_argument("--max-inflight", type=int, default=256, help="tool-only worker budget, explicitly raised for larger capacity")
    run.add_argument("--batch-response-mib", type=int, default=256, help="cumulative per-request response budgets in one batch")
    run.add_argument("--tokens", type=int, default=256)
    run.add_argument("--seed", type=int, default=1234)
    run.add_argument("--temperature", type=float, default=0)
    run.add_argument("--max-cases", type=int, default=100000)
    protocol.add_arguments(run)
    run.set_defaults(response_mib=1)
    args = parser.parse_args(argv)
    if args.mode == "benchmark":
        if not math.isfinite(args.temperature) or args.temperature < 0:
            parser.error("temperature must be finite and nonnegative")
        return benchmark(args)
    if args.mode == "inventory":
        positive(args.command_mib, "command MiB")
        value = inventory(rank_devices=args.rank_devices.split(",") if args.rank_devices else (),
                          timeout=args.timeout, command_limit=args.command_mib << 20)
    elif args.mode == "matrix":
        value = matrix(load_json(args.stats), max_cases=args.max_cases)
    else:
        positive(args.payload_mib, "transfer payload MiB")
        value = transfers(payload_bytes=args.payload_mib << 20, reps=args.reps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

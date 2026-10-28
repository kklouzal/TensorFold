"""One-rank CPU operator/nearest-project rescan of an already qualified capture.

Inclusive operator/API wall durations, assigned once at clipped event start.
Nested ATen intervals can overlap: never sum them into CPU busy or Python time.
No GPU imports, kernels retained, new profile, all-rank scan, or source overlay.
"""

import argparse
from collections import defaultdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import resource
import time

from tensorfold.cleanup import raise_failures

ANALYZER_SHA = "780c7d40cc90b4226212d4e1e0dcccaebeaad01e9c4148beaa15b790bda58df1"
CURRENT_APP = "4b0ea6b51047f93b2b973ed5a467d8c279b94f6d85c683c4258a712ff6edd99f"
MEMORY_LIMIT = 8 << 30  # Existing conservative6GiB Stream envelope +2GiB bounded tables/index.
MAX_ROWS = 1_000_000
MAX_RANGES = 200_000
MAX_NAMES = 8192
MAX_NAME_CHARS = 512
MAX_KEYS = 16_384
REPORT_BYTES = 2 << 20


def bounded_json(path):
    if path.stat().st_size > 4 << 20:
        raise ValueError("metadata exceeds4MiB")
    return json.loads(path.read_text())


def trusted_helpers(path):
    if hashlib.sha256(path.read_bytes()).hexdigest() != ANALYZER_SHA:
        raise ValueError("closed streaming/timeline helper source differs")
    spec = importlib.util.spec_from_file_location("closed_cpu_parent_stream_helpers", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def capture_context(root):
    state = bounded_json(root / "controller-progress.json")
    if state["status"] != "terminal" or state["docker_state"]["Running"] or state["docker_state"]["ExitCode"] != 0:
        raise ValueError("normal terminal0 capture required")
    candidate = bounded_json(root / "candidate.json")
    if candidate["wheel_sha256"] != CURRENT_APP or candidate["profile_concurrency"] != 24:
        raise ValueError("only current4b0 count24 capture is admitted")
    folder = root / "results/run"
    definition = bounded_json(root / "analysis-concurrent-v1/window-definition.json")
    summary = bounded_json(root / "analysis-concurrent-v1/all8-summary.json")
    case = bounded_json(folder / "fast-mtp-target-n24.json")
    if (
        not summary["complete"]
        or summary["offered"] != 24
        or definition["offered"] != 24
        or len(summary["trace_identities"]) != 8
        or {row["rank"] for row in summary["trace_identities"]} != set(range(8))
        or len(summary["windows"]) != 8
        or {row["rank"] for row in summary["windows"]} != set(range(8))
    ):
        raise ValueError("existing closed all8 count24 analysis required")
    if definition["frozen_screen"] != case["screen"]:
        raise ValueError("clock definition differs from actual frozen window")
    screen = case["screen"]
    if (
        not screen["measurement_valid"]
        or not screen["all_workers_permits_backend_owners_drained"]
        or not screen["native_commit_trace_complete"]
        or screen["duration_ns"] != 10_000_000_000
    ):
        raise ValueError("valid complete10s native window/drain required")
    if definition["translation"]["observed_hull_width_ns"] > 10_000_000:
        raise ValueError("closed clock hull exceeds sensitivity")
    for rank in range(8):
        profile = bounded_json(folder / f"rank-{rank}-profile.json")
        final = bounded_json(folder / f"rank-{rank}-final.json")
        if not profile["complete"] or not final["complete"] or not profile["engine_transport_retired"]:
            raise ValueError("all8 complete source/capture/retirement required")
        for boundary in ("before", "after"):
            if (
                profile["source_" + boundary]["status"] != "verified-source-software"
                or profile["cupti_" + boundary]["API"] != 28
            ):
                raise ValueError("all8 source/CUPTI28 guard required")
        identity = next(row for row in summary["trace_identities"] if row["rank"] == rank)
        if identity["sha256"] != profile["trace"]["sha256"] or identity["bytes"] != profile["trace"]["bytes"]:
            raise ValueError("closed all8 raw identity differs from capture")
    (reply,) = definition["replies"]
    start, end = reply["first_commit_upper_us"], reply["decode_core_end_us"]
    if not 0 < start < end or not math.isfinite(start + end) or end - start > 10_000_000:
        raise ValueError("invalid bounded decode core")
    for row in summary["windows"]:
        if row["window"]["start_us"] != start or row["window"]["end_us"] != end:
            raise ValueError("rescan core differs from closed all8 analysis")
    rank0 = bounded_json(folder / "rank-0-profile.json")
    return folder / "rank-0-trace.json", rank0, definition, start, end


def scan(path, helpers, start, end):
    """Owned close, one record->one parent; raw event ordering need not be sorted."""
    primary, close_failures = None, [None]
    ranges, events, names, runtime_ids = defaultdict(list), [], {}, set()
    header, range_count = {}, 0
    # All setup and the owned file slot exist before acquisition. Constructor
    # failures after opening are inside the same primary-preserving close.
    stream = object.__new__(helpers.Stream)
    stream.file = None
    try:
        helpers.Stream.__init__(stream, path)

        def name_key(value):
            if type(value) is not str or len(value) > MAX_NAME_CHARS:
                raise ValueError("selected CPU/project name exceeds512characters")
            if value not in names:
                if len(names) >= MAX_NAMES:
                    raise ValueError("distinct selected names exceed8192")
                names[value] = value
            return names[value]

        for event in stream.events(header):
            if event.get("ph") != "X":
                continue
            category = event.get("cat", "")
            if category not in ("cpu_op", "user_annotation", "cuda_runtime", "cuda_driver"):
                continue
            ts, duration = event.get("ts"), event.get("dur")
            if (
                type(ts) not in (int, float)
                or type(duration) not in (int, float)
                or not math.isfinite(ts + duration)
                or duration < 0
            ):
                raise ValueError("invalid timed selected CPU event")
            first, last = max(ts, start), min(ts + duration, end)
            if last <= first:
                continue
            tid = event.get("tid")
            if type(tid) is not int:
                raise ValueError("selected Kineto CPU thread must have integer identity")
            name = event.get("name", "")
            if type(name) is not str:
                raise ValueError("selected CPU name must be text")
            if category == "user_annotation":
                if name.startswith(("tensorfold/cpu/", "tensorfold/body/", "tensorfold/host/")):
                    range_count += 1
                    if range_count > MAX_RANGES:
                        raise ValueError("project ranges exceed200000")
                    ranges[tid].append((ts, ts + duration, name_key(name)))
                continue
            if len(events) >= MAX_ROWS:
                raise ValueError("selected CPU/API events exceed1000000")
            if category in ("cuda_runtime", "cuda_driver"):
                correlation = event.get("args", {}).get("correlation")
                if type(correlation) is int:
                    if correlation in runtime_ids:
                        raise ValueError("ambiguous duplicate runtime correlation")
                    runtime_ids.add(correlation)
            events.append((first, last - first, tid, name_key(name), category))
        index, table = helpers.timelines(ranges), {}
        for first, duration, tid, name, category in events:
            phase = helpers.phase_at(index, tid, first)
            key = (phase, category, name)
            if key not in table:
                if len(table) >= MAX_KEYS:
                    raise ValueError("operator/parent groups exceed16384")
                table[key] = [0, 0.0, 0.0]
            row = table[key]
            row[0] += 1
            row[1] += duration
            row[2] = max(row[2], duration)
        result = {
            "header": header,
            "project_ranges": range_count,
            "selected_events": len(events),
            "groups": [
                {
                    "parent": parent,
                    "category": category,
                    "name": name,
                    "count": values[0],
                    "inclusive_us": values[1],
                    "maximum_us": values[2],
                }
                for (parent, category, name), values in sorted(table.items(), key=lambda row: row[1][1], reverse=True)
            ],
        }
    except BaseException as error:
        primary = error
    finally:
        try:
            if stream.file is not None:
                stream.file.close()
        except BaseException as error:
            close_failures[0] = error
    finish(primary, close_failures)
    return result, stream


def finish(primary, failures):
    if failures[0] is not None:
        raise_failures(primary, failures)
    if primary is not None:
        raise primary


def write_result(path, encoded):
    primary, failures = None, [None]
    output = path.open("x")
    try:
        output.write(encoded)
    except BaseException as error:
        primary = error
    finally:
        try:
            output.close()
        except BaseException as error:
            failures[0] = error
    finish(primary, failures)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--rank", type=int, choices=(0,), default=0)
    parser.add_argument("--root-cpu-lease", action="store_true", required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError("fresh CPU-parent report required")
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    cap = MEMORY_LIMIT if hard == resource.RLIM_INFINITY else min(MEMORY_LIMIT, hard)
    if soft != resource.RLIM_INFINITY:
        cap = min(cap, soft)
    resource.setrlimit(resource.RLIMIT_AS, (cap, hard))
    helpers = trusted_helpers(args.root / "analysis-tools/analyze_traces.py")
    path, profile, definition, start, end = capture_context(args.root)
    before, begun = path.stat(), time.monotonic()
    report, stream = scan(path, helpers, start, end)
    after = path.stat()
    if (before.st_ino, before.st_size, before.st_mtime_ns) != (after.st_ino, after.st_size, after.st_mtime_ns):
        raise ValueError("raw trace changed during selected rescan")
    if stream.bytes != before.st_size or stream.digest.hexdigest() != profile["trace"]["sha256"]:
        raise ValueError("raw trace full-byte SHA/size mismatch")
    if report.pop("header").get("baseTimeNanoseconds") != definition["baseTimeNanoseconds"]:
        raise ValueError("clock trace base differs")
    report.update(
        schema="tensorfold.current4b0-rank0-CPU-parent-v1",
        complete=True,
        rank=0,
        core_start_us=start,
        core_end_us=end,
        raw_sha256=stream.digest.hexdigest(),
        raw_bytes=stream.bytes,
        elapsed_seconds=time.monotonic() - begun,
        memory_limit_bytes=cap,
        observed_maxrss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256=ANALYZER_SHA,
        scope="Each intersecting CPU/API event assigned exactly once by deepest project at clipped event start; per-event duration clipped to core. Inclusive ATen/API intervals can nest/overlap; not exclusive CPU busy, Python residual, model speed or quality. GPU kernels ignored. Current4b0 only; raw/reports remain immutable.",
    )
    encoded = json.dumps(report, indent=2, allow_nan=False) + "\n"
    if len(encoded.encode()) > REPORT_BYTES:
        raise ValueError("CPU-parent report exceeds2MiB; no truncation")
    write_result(args.out, encoded)
    print(
        json.dumps(
            {
                "complete": True,
                "rank": 0,
                "output": str(args.out),
                "raw_sha256": report["raw_sha256"],
                "raw_bytes": report["raw_bytes"],
                "seconds": report["elapsed_seconds"],
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()

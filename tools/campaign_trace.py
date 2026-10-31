"""A decode timeline of a TensorFold CUDA configuration under torch.profiler (Kineto/CUPTI), with a per-round summary.

  campaign_trace.py --out DIR [--prompts PROMPTS.json --prompt ID] [--tokens 48] [--warm 256] [--serial]
                    [--temperature 0 --top-k K --top-p P --seed N] -- serve MODEL [production flags]

The engine and App are built by the production CLI (``campaign_kl.with_app``). The chosen bench prompt is generated
once unprofiled (prefill, kernel compilation, prefix cache), then generated again under the profiler, so the trace
holds a cache-hit prefill and up to ``--tokens`` decode tokens of the same request. The policy prepared by the App
is used unchanged for warmup and profiling. CUDA sample_rows/effective_rows calls mark round boundaries (``tf.sample``
ranges); scoped stage markers identify proposal logits, packet/readback, target probabilities, rejection and append.
DIR must be fresh and receives trace.json (open in Perfetto or
chrome://tracing) and summary.json: per round its CPU annotation wall time, the union of GPU kernel time (busy fraction), kernel
launches, synchronizing runtime calls and device-to-host copies; and the top kernels by GPU time. Timings under the
profiler carry its overhead: they diagnose where a round's time goes, they are not speed claims. A trace without
actual CUDA kernels or complete sampler/engine round coverage fails explicitly; partial evidence is retained.
CPU stages use user_annotation ranges; gpu_user_annotation ranges are reported separately as inclusive GPU ranges,
which can overlap and are not kernel busy time or CPU wall time.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import statistics
import sys
import time
import traceback
from contextlib import contextmanager
from functools import wraps
from pathlib import Path

if __package__:
    from .benchmark_output import write_json
    from .campaign_kl import HotGate, with_app
else:
    from benchmark_output import write_json
    from campaign_kl import HotGate, with_app

LAUNCH = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx", "cudaGraphLaunch")
SYNC = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy", "cuStreamSynchronize",
        "cuCtxSynchronize")


def summarize(trace: dict, top: int = 25) -> dict:
    """Rounds between CPU ``tf.sample`` annotations; keep CPU/GPU stage range domains separate."""

    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    marks = sorted(e["ts"] for e in events if e.get("cat") == "user_annotation" and e.get("name") == "tf.sample")
    kernels = sorted((e["ts"], e["ts"] + e["dur"], e["name"]) for e in events
                     if e.get("cat") == "kernel" and e.get("dur", 0) > 0)
    runtime = [(e["ts"], e["name"]) for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver")]
    copies = [e["ts"] for e in events if e.get("cat") == "gpu_memcpy" and "DtoH" in e.get("name", "")]
    rounds = []
    for start, end in itertools.pairwise(marks):
        busy, cursor = 0.0, start
        for k0, k1, _ in kernels:
            if k1 <= start or k0 >= end:
                continue
            k0, k1 = max(k0, cursor, start), min(k1, end)
            if k1 > k0:
                busy += k1 - k0
                cursor = k1
        calls = [name for ts, name in runtime if start <= ts < end]
        rounds.append({"wall_us": end - start, "gpu_busy_us": busy,
                       "launches": sum(name in LAUNCH for name in calls), "syncs": sum(name in SYNC for name in calls),
                       "d2h_copies": sum(start <= ts < end for ts in copies)})
    totals: dict[str, list[float]] = {}
    for k0, k1, name in kernels:
        totals.setdefault(name, []).append(k1 - k0)
    gpu_total = sum(sum(v) for v in totals.values()) or 1.0
    ranked = sorted(totals.items(), key=lambda kv: -sum(kv[1]))[:top]

    def med(key):
        return statistics.median(r[key] for r in rounds) if rounds else None

    stages, gpu_stages = {}, {}
    for event in events:
        name = event.get("name", "")
        if name.startswith("tf.") and name != "tf.sample":
            if event.get("cat") == "user_annotation":
                table, duration = stages, "cpu_wall_us_inclusive"
            elif event.get("cat") == "gpu_user_annotation":
                table, duration = gpu_stages, "gpu_range_us_inclusive"
            else:
                continue
            row = table.setdefault(name, {"calls": 0, duration: 0.0})
            row["calls"] += 1
            row[duration] += event.get("dur", 0)
    return {"rounds": len(rounds), "sampler_markers": len(marks), "gpu_kernel_events": len(kernels),
            "stages": stages, "gpu_stages": gpu_stages,
            "round_wall_ms_median": med("wall_us") and med("wall_us") / 1000,
            "gpu_busy_fraction_median": statistics.median(r["gpu_busy_us"] / r["wall_us"] for r in rounds)
            if rounds else None,
            "launches_per_round_median": med("launches"), "syncs_per_round_median": med("syncs"),
            "d2h_copies_per_round_median": med("d2h_copies"),
            "top_kernels": [{"name": name[:160], "calls": len(v), "total_ms": round(sum(v) / 1000, 3),
                             "share": round(sum(v) / gpu_total, 4)} for name, v in ranked],
            "per_round": rounds}


@contextmanager
def stage_markers(engine, record_function):
    """Temporary diagnostics in this isolated producer; restore exact module/class/instance bindings."""
    from tensorfold.cleanup import raise_failures
    from tensorfold.cuda import sampling, speculative
    from tensorfold.engine import rejection

    patches, counts = [], {}
    primary = None

    def replace(owner, name, label, *, boundary=False):
        original = getattr(owner, name)
        own = vars(owner)
        previous = (name in own, own.get(name))

        @wraps(original)
        def marked(*args, **kwargs):
            counts[label] = counts.get(label, 0) + 1
            if boundary:
                with record_function("tf.sample"), record_function(label):
                    return original(*args, **kwargs)
            with record_function(label):
                return original(*args, **kwargs)

        patches.append((owner, name, *previous))
        setattr(owner, name, marked)
        counts.setdefault(label, 0)

    try:
        # Like the maintained sampler hook, cover already imported aliases as
        # well as Gemma's per-generate imports. Snapshot before replacing any.
        functions = ((sampling, "sample_rows", "tf.target.sample_rows", True),
                     (sampling, "effective_rows", "tf.target.effective_rows", True),
                     (rejection, "verify", "tf.rejection.verify", False))
        aliases = []
        for source, name, label, boundary in functions:
            original = getattr(source, name)
            for module_name, module in list(sys.modules.items()):
                if module_name.startswith("tensorfold") and getattr(module, name, None) is original:
                    aliases.append((module, name, label, boundary))
        for owner, name, label, boundary in aliases:
            replace(owner, name, label, boundary=boundary)
        for name, label in (("__init__", "tf.q.packet"), ("_readback", "tf.q.readback"),
                            ("sample", "tf.q.sample"), ("finish", "tf.q.finish")):
            replace(speculative.Chains, name, label)
        head = getattr(engine, "head", None)
        if head is not None:
            for name in ("logits", "append"):
                if hasattr(head, name):
                    replace(head, name, "tf.draft." + name)
        yield counts
    except BaseException as error:  # noqa: BLE001 - restore bindings before rethrowing interrupts/primary.
        primary = error
    finally:
        errors = []
        for owner, name, had_own, original in reversed(patches):
            try:
                if had_own:
                    setattr(owner, name, original)
                elif name in vars(owner):
                    delattr(owner, name)
            except BaseException as error:  # noqa: BLE001 - restore every binding; raise_failures retains each error.
                errors.append(error)
        patches.clear()
        raise_failures(primary, errors)


def trace(app, args) -> None:
    import torch

    from tensorfold.cleanup import raise_failures

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=False)
    item = None
    if args.prompts:
        items = {it["id"]: it for it in json.loads(Path(args.prompts).read_text())["items"]}
        if args.prompt not in items:
            raise SystemExit(f"prompt {args.prompt!r} is not in {args.prompts} ({', '.join(items)})")
        item = items[args.prompt]
    body = {"messages": item["messages"] if item else [{"role": "user", "content": "Write a long essay about rivers."}],
            "chat_template_kwargs": {"enable_thinking": bool(item and item["thinking"])},
            "temperature": args.temperature}
    for field in ("top_k", "top_p", "seed"):
        if getattr(args, field) is not None:
            body[field] = getattr(args, field)
    prepared = app.prepare(body, True)
    engine = app.engine
    out: list[int] = []
    summary = {"status": "FAIL", "prompt": args.prompt, "prompt_tokens": len(prepared.prompt),
               "serial": args.serial, "serve_args": args.serve, "request_policy": {k: body[k] for k in
                   ("temperature", "top_k", "top_p", "seed") if k in body},
               "prepared_policy": None if prepared.sampling is None else vars(prepared.sampling),
               "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "note": "profiled timings include CUPTI overhead; stages are inclusive CPU annotation wall time; "
                       "gpu_stages are inclusive GPU annotation ranges, not kernel busy time; diagnostic only"}
    write_json(outdir / "intent.json", summary)
    primary, errors, prof, entered = None, [], None, False
    counts = {}
    try:
        HotGate.wait()
        started = time.perf_counter()
        engine.generate(prepared.prompt, args.warm, prepared.sampling, lambda new: out.extend(new) or False,
                        draft=not args.serial)
        torch.cuda.synchronize()                       # no deferred warm append can leak into this trace
        summary["warm_s"] = round(time.perf_counter() - started, 4)
        out.clear()
        HotGate.wait()
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        prof = torch.profiler.profile(activities=activities, record_shapes=False)
        prof.__enter__()
        entered = True
        with stage_markers(engine, torch.profiler.record_function) as counts:
            began = time.perf_counter()
            summary["engine_stats"] = engine.generate(prepared.prompt, args.tokens, prepared.sampling,
                                                       lambda new: out.extend(new) or False, draft=not args.serial)
            returned = time.perf_counter()
            with torch.profiler.record_function("tf.deferred_tail"):
                torch.cuda.synchronize()
            settled = time.perf_counter()
            summary.update(generate_wall_s=returned - began, deferred_tail_s=settled - returned,
                           settled_wall_s=settled - began)
    except BaseException as error:  # noqa: BLE001 - exit/export the profiler before rethrowing the primary.
        primary = error
    finally:
        if entered:
            try:
                prof.__exit__(type(primary) if primary else None, primary,
                              primary.__traceback__ if primary else None)
            except BaseException as error:  # noqa: BLE001 - retain profiler-exit failure alongside the primary.
                errors.append(error)
        summary.update(tokens=len(out), marker_calls=dict(counts))
        if prof is not None and entered:
            try:
                prof.export_chrome_trace(str(outdir / "trace.json"))
                summary.update(summarize(json.loads((outdir / "trace.json").read_text())))
                if primary is None and not errors:
                    rounds = summary.get("engine_stats", {}).get("rounds")
                    if not summary["gpu_kernel_events"]:
                        raise RuntimeError("profiler trace has no actual CUDA kernel events")
                    if type(rounds) is not int or rounds < 1 or summary["rounds"] != rounds:
                        raise RuntimeError("profiler sampler markers do not cover the engine's decode rounds")
                    calls = counts.get("tf.target.sample_rows", 0) + counts.get("tf.target.effective_rows", 0)
                    if calls != rounds + 1 or summary["sampler_markers"] != calls:
                        raise RuntimeError("profiler trace omitted a required sampler boundary")
            except BaseException as error:  # noqa: BLE001 - retain export/validation errors through raise_failures.
                errors.append(error)
        if primary is None and not errors:
            summary["status"] = "PASS"
        else:
            summary["errors"] = [{"type": type(error).__name__, "message": str(error),
                                  "traceback": "".join(traceback.format_exception(error))[:65536]}
                                 for error in ([primary] if primary else []) + errors]
        try:
            write_json(outdir / "summary.json", summary)
        except BaseException as error:  # noqa: BLE001 - retain publication failure through raise_failures.
            errors.append(error)
        raise_failures(primary, errors)
    print(json.dumps({k: summary[k] for k in ("rounds", "round_wall_ms_median", "gpu_busy_fraction_median",
                                               "launches_per_round_median", "syncs_per_round_median",
                                               "d2h_copies_per_round_median")}))
    for k in summary["top_kernels"][:12]:
        print(json.dumps(k))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompts")
    ap.add_argument("--prompt", default="4k-doc-review-nothink")
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--warm", type=int, default=256)
    ap.add_argument("--serial", action="store_true", help='profile "draft": false')
    ap.add_argument("--temperature", type=float, default=0.0, help="request temperature; default0 preserves greedy")
    ap.add_argument("--top-k", type=int, help="request top-k; omitted uses the served model's default")
    ap.add_argument("--top-p", type=float, help="request top-p; omitted uses the served model's default")
    ap.add_argument("--seed", type=int, help="request seed; omitted uses the production prompt-keyed default")
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--hot-flag", help="the guard's hot-state file (campaign_bench.HotGate; default $TFC_HOT_FLAG)")
    ap.add_argument("serve", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if (args.tokens < 1 or args.warm < 1 or not math.isfinite(args.temperature)
            or (args.top_k is not None and args.top_k < 0)
            or (args.top_p is not None and (not math.isfinite(args.top_p) or not 0 <= args.top_p <= 1))):
        ap.error("positive token limits, finite temperature, nonnegative top-k and top-p in[0,1] required")
    HotGate.configure(args.hot_flag)
    if args.serve and args.serve[0] == "--":
        args.serve = args.serve[1:]
    if not args.serve or args.serve[0] != "serve":
        ap.error("needs -- serve MODEL [flags]")
    with_app(args.serve, args.port, lambda app: trace(app, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

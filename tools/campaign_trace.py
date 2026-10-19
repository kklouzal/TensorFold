"""A decode timeline of a TensorFold CUDA configuration under torch.profiler (Kineto/CUPTI), with a per-round summary.

  campaign_trace.py --out DIR [--prompts PROMPTS.json --prompt ID] [--tokens 48] [--warm 256] [--serial]
                    -- serve MODEL [production flags]

The engine and App are built by the production CLI (``campaign_kl.with_app``). The chosen bench prompt is generated
once unprofiled (prefill, kernel compilation, prefix cache), then generated again under the profiler, so the trace
holds a cache-hit prefill and ``--tokens`` decode tokens of the same request. Each call of the CUDA target sampler
marks a round boundary (``tf.sample`` ranges in the trace). DIR receives trace.json (open in Perfetto or
chrome://tracing) and summary.json: per round its wall time, the union of GPU kernel time (busy fraction), kernel
launches, synchronizing runtime calls and device-to-host copies; and the top kernels by GPU time. Timings under the
profiler carry its overhead: they diagnose where a round's time goes, they are not speed claims.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
import statistics
import time

if __package__:
    from .campaign_kl import HotGate, sampler_replaced, with_app
else:
    from campaign_kl import HotGate, sampler_replaced, with_app

LAUNCH = ("cudaLaunchKernel", "cudaLaunchKernelExC", "cuLaunchKernel", "cuLaunchKernelEx", "cudaGraphLaunch")
SYNC = ("cudaStreamSynchronize", "cudaDeviceSynchronize", "cudaEventSynchronize", "cudaMemcpy", "cuStreamSynchronize",
        "cuCtxSynchronize")


def summarize(trace: dict, top: int = 25) -> dict:
    """Rounds between successive ``tf.sample`` ranges: wall, GPU busy (kernel-interval union), launches, syncs."""

    events = [e for e in trace.get("traceEvents", []) if e.get("ph") == "X"]
    marks = sorted(e["ts"] for e in events if e.get("name") == "tf.sample")
    kernels = sorted(((e["ts"], e["ts"] + e.get("dur", 0), e["name"]) for e in events if e.get("cat") == "kernel"))
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

    return {"rounds": len(rounds), "round_wall_ms_median": med("wall_us") and med("wall_us") / 1000,
            "gpu_busy_fraction_median": statistics.median(r["gpu_busy_us"] / r["wall_us"] for r in rounds)
            if rounds else None,
            "launches_per_round_median": med("launches"), "syncs_per_round_median": med("syncs"),
            "d2h_copies_per_round_median": med("d2h_copies"),
            "top_kernels": [{"name": name[:160], "calls": len(v), "total_ms": round(sum(v) / 1000, 3),
                             "share": round(sum(v) / gpu_total, 4)} for name, v in ranked],
            "per_round": rounds}


def trace(app, args) -> None:
    import torch
    import tensorfold.cuda.sampling as sampling

    item = None
    if args.prompts:
        items = {it["id"]: it for it in json.loads(Path(args.prompts).read_text())["items"]}
        if args.prompt not in items:
            raise SystemExit(f"prompt {args.prompt!r} is not in {args.prompts} ({', '.join(items)})")
        item = items[args.prompt]
    body = {"messages": item["messages"] if item else [{"role": "user", "content": "Write a long essay about rivers."}],
            "chat_template_kwargs": {"enable_thinking": bool(item and item["thinking"])}, "temperature": 0}
    prepared = app.prepare(body, True)
    engine = app.engine
    out: list[int] = []
    HotGate.wait()                                      # a new prompt only from a cool host
    started = time.perf_counter()
    engine.generate(prepared.prompt, args.warm, None, lambda new: out.extend(new) or False, draft=not args.serial)
    warm_s = time.perf_counter() - started
    original = sampling.sample_rows

    def marked(*a, **kw):
        with torch.profiler.record_function("tf.sample"):
            return original(*a, **kw)

    out.clear()
    HotGate.wait()
    activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    with sampler_replaced(marked), torch.profiler.profile(activities=activities, record_shapes=False) as prof:
        stats = engine.generate(prepared.prompt, args.tokens, None, lambda new: out.extend(new) or False,
                                draft=not args.serial)
        torch.cuda.synchronize()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(outdir / "trace.json"))
    summary = summarize(json.loads((outdir / "trace.json").read_text()))
    summary.update(prompt=args.prompt, prompt_tokens=len(prepared.prompt), tokens=len(out), serial=args.serial,
                   warm_s=round(warm_s, 2), engine_stats=stats, serve_args=args.serve,
                   note="profiled timings include CUPTI overhead; diagnostic only")
    (outdir / "summary.json").write_text(json.dumps(summary, indent=1) + "\n")
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
    ap.add_argument("--port", type=int, default=8899)
    ap.add_argument("--hot-flag", help="the guard's hot-state file (campaign_bench.HotGate; default $TFC_HOT_FLAG)")
    ap.add_argument("serve", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    HotGate.configure(args.hot_flag)
    if args.serve and args.serve[0] == "--":
        args.serve = args.serve[1:]
    if not args.serve or args.serve[0] != "serve":
        ap.error("needs -- serve MODEL [flags]")
    with_app(args.serve, args.port, lambda app: trace(app, args))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

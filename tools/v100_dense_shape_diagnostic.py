#!/usr/bin/env python3
"""Root-only vendor-BLAS shape experiment. No application/backend mutation."""
from dataclasses import dataclass
from pathlib import Path
import argparse
import gc
import hashlib
import json
import statistics
import time


@dataclass(frozen=True)
class Geometry:
    role: str
    n: int
    k: int
    output: str
    rows: tuple


GEOMETRIES = (
    Geometry("hc_down", 320, 10240, "float32", (1, 2, 3)),
    Geometry("hc_inject", 4, 10240, "float32", (1, 2, 3)),
    Geometry("hc_up", 10240, 320, "float16", (1, 2, 3)),
    Geometry("vocab_head_rank", 31040, 2560, "float32", (1, 2, 3)),
    Geometry("attention_q_part", 1536, 2560, "float16", (1, 2, 3)),
    Geometry("gdn_projection", 2060, 2560, "float16", (1, 2, 3)),
    Geometry("mtp_fc_h_physical", 2560, 2560, "float16", (4, 8, 12)),
)
GPU_LIMIT = 2 << 30
REPORT_LIMIT = 8 << 20
SAMPLES = 3
REPLAYS = 8
UNROLL = 4


def shapes(rows):
    if type(rows) is not int or not 1 <= rows <= 32:
        raise ValueError("logical rows must be1..32")
    return (rows, *[m for m in (4, 8, 16, 32) if m > rows])


def payload_bound(g):
    # Two rotating weights, two inputs/outputs, baseline result, pad/result
    # buffers and four unrolled graph results. Library/allocator slack separate.
    return 2 * g.n * g.k * 2 + 16 * (32 * g.k * 2 + 32 * g.n * 4) + (64 << 20)


def write(path, value):
    raw = json.dumps(value, indent=1, allow_nan=False).encode() + b"\n"
    if len(raw) > REPORT_LIMIT:
        raise ValueError("diagnostic report exceeds fixed8MiB envelope")
    path.write_bytes(raw)


def metrics(torch, actual, baseline):
    a, b = actual.detach().cpu(), baseline.detach().cpu()
    if not torch.isfinite(a).all().item() or not torch.isfinite(b).all().item():
        raise ValueError("finite diagnostic inputs produced nonfinite output")
    x, y = a.double(), b.double()
    delta, norm = x - y, torch.linalg.vector_norm(y).item()
    return {"bits_exact": torch.equal(a.view(torch.uint8), b.view(torch.uint8)),
            "different_elements": int((a != b).sum()),
            "different_bit_elements": int((a.view(torch.int16 if a.dtype == torch.float16 else torch.int32)
                != b.view(torch.int16 if b.dtype == torch.float16 else torch.int32)).sum()),
            "max_abs": delta.abs().max().item(),
            "rms": delta.square().mean().sqrt().item(),
            "relative_L2": torch.linalg.vector_norm(delta).item() / norm if norm else None,
            "scope": "empirical output error, not model-quality or universal-bound acceptance"}


class Variant:
    def __init__(self, torch, native, x, weight, output, padded, *, prepared=False):
        self.torch, self.native, self.x, self.weight = torch, native, x, weight
        self.out = torch.empty((x.shape[0], weight.shape[0]), dtype=output, device=x.device)
        self.prepared = prepared
        self.pad = None if native is not None or padded == x.shape[0] else torch.zeros((padded, x.shape[1]), dtype=x.dtype, device=x.device)
        if self.pad is not None:
            self.pad[:x.shape[0]].copy_(x)
        self.result = None

    def call(self):
        if self.native is not None:
            self.native(self.x, self.weight, self.out)
            return self.out
        if self.pad is not None and not self.prepared:
            self.pad.zero_()
            self.pad[:self.x.shape[0]].copy_(self.x)
        value = self.x if self.pad is None else self.pad
        product = (self.torch.mm(value, self.weight.T, out_dtype=self.torch.float32)
                   if self.out.dtype == self.torch.float32 else self.torch.mm(value, self.weight.T))
        self.result = product  # Keep latest captured result/storage owned.
        if not self.prepared:
            self.out.copy_(product[:self.x.shape[0]])
            return self.out
        return product[:self.x.shape[0]]


def timed(torch, variants):
    graphs = []
    for variant in variants:
        variant.call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(UNROLL):
                variant.call()
        graphs.append(graph)
    for graph in graphs:
        graph.replay()
    torch.cuda.synchronize()
    cpu, gpu = [], []
    for _ in range(SAMPLES):
        start, stop = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        before = time.perf_counter_ns()
        for index in range(REPLAYS):
            graphs[index % len(graphs)].replay()
        elapsed = time.perf_counter_ns() - before
        stop.record()
        stop.synchronize()
        cpu.append(elapsed / (REPLAYS * UNROLL))
        gpu.append(start.elapsed_time(stop) * 1e6 / (REPLAYS * UNROLL))
    return {"graph_cpu_enqueue_ns_per_projection": cpu,
            "graph_gpu_stream_ns_per_projection": gpu,
            "cpu_median_ns": statistics.median(cpu), "gpu_median_ns": statistics.median(gpu),
            "samples": SAMPLES, "replays": REPLAYS, "unrolled_operations_per_graph": UNROLL,
            "scope": "GPU stream elapsed includes graph work and possible CPU enqueue starvation; not isolated kernel-only time"}


def run(args):
    import torch
    from tensorfold.cuda.sm70 import gguf_ops
    from tensorfold.cuda.sm70.execution_policy import _family_runtime
    _family_runtime(torch)  # Assert actual flags; never set them.
    if not args.root_gpu_lease:
        raise ValueError("Root exclusive GPU lease flag required")
    if args.out.exists():
        raise FileExistsError("fresh diagnostic directory required")
    args.out.mkdir()
    module = gguf_ops.extension()
    if module.prepare_dense32() != 4:
        raise RuntimeError("nativeM32code residency differs")
    props = torch.cuda.get_device_properties(0)
    report = {"schema": "tensorfold.vendor-BLAS-N1-shapes-v1", "complete": False, "results": [],
        "torch": str(torch.__version__), "torch_git": torch.version.git_version, "cuda": torch.version.cuda,
        "device": {"name": props.name, "capability": [props.major, props.minor], "L2_bytes": getattr(props, "L2_cache_size", None)},
        "math_flags": {"reduced_fp16_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
                       "fp16_accumulation": torch.backends.cuda.matmul.allow_fp16_accumulation,
                       "preferred_BLAS": str(torch.backends.cuda.preferred_blas_library())},
        "input_weight_dtype": "float16 unchanged", "GPU_limit_bytes": GPU_LIMIT,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "synthetic exact-geometry microdiagnostic; no model speed/quality/capacity claim; no weights/backend converted",
        "cache_scope": "one repeated weight versus alternating two graph/weight addresses; no claimed L2 flush/cold cache. Footprints explicit, not whole-decoder reuse.",
        "first_call_scope": "first synchronized call after allocation, dependent on prior shape/order/cache; not comparable cold-start benchmark"}
    from tensorfold.cleanup import raise_failures
    diagnostic_failures = [None]
    try:
        for gi, g in enumerate(GEOMETRIES):
            if payload_bound(g) > GPU_LIMIT:
                raise MemoryError("geometry exceeds prepaid2GiB project envelope")
            generator = torch.Generator(device="cuda").manual_seed(17100 + gi)
            weights = [torch.randn((g.n, g.k), generator=generator, dtype=torch.float16, device="cuda") * .01 for _ in range(2)]
            for rows in g.rows:
                x = torch.randn((rows, g.k), generator=generator, dtype=torch.float16, device="cuda") * .1
                dtype = getattr(torch, g.output)
                baselines = [Variant(torch, gguf_ops.dense32, x, w, dtype, 32) for w in weights]
                before = time.perf_counter_ns()
                references = [v.call().clone() for v in baselines]
                torch.cuda.synchronize()
                first_baseline_ns = time.perf_counter_ns() - before
                for padded in (None, *shapes(rows)):
                    for prepared in ((False,) if padded is None else (False, True)):
                        variants = baselines if padded is None else [Variant(torch, None, x, w, dtype, padded, prepared=prepared) for w in weights]
                        label = "native_current_M32" if padded is None else f"vendor_M{padded}_" + ("prepared_math" if prepared else "complete_pad_copy")
                        before = time.perf_counter_ns()
                        outputs = [v.call().clone() for v in variants]
                        torch.cuda.synchronize()
                        first_ns = time.perf_counter_ns() - before
                        differences = [metrics(torch, output, baseline) for output, baseline in zip(outputs, references, strict=True)]
                        hot = timed(torch, variants[:1])
                        rotating = timed(torch, variants)
                        captured_differences = [metrics(torch, (v.result[:rows] if v.prepared else v.out), output)
                            for v, output in zip(variants, outputs, strict=True)]
                        peak = torch.cuda.memory_reserved()
                        if peak > GPU_LIMIT:
                            raise MemoryError("observed allocator reservation exceeded diagnostic2GiB limit")
                        row = {"role": g.role, "N": g.n, "K": g.k, "logical_rows": rows, "padded_rows": 32 if padded is None else padded,
                            "output_dtype": g.output, "variant": label, "prepared_input": prepared,
                            "one_weight_bytes": g.n * g.k * 2, "rotating_weight_bytes": 2 * g.n * g.k * 2,
                            "payload_operating_bound": payload_bound(g), "observed_reserved_bytes": peak,
                            "first_two_weight_calls_wall_ns": first_ns, "baseline_initial_two_calls_wall_ns": first_baseline_ns,
                            "output_differences": differences, "captured_vs_own_eager_output": captured_differences, "hot_single_weight": hot, "alternating_two_weights": rotating}
                        report["results"].append(row)
                        write(args.out / "progress.json", report)
                        if padded is not None:
                            del variants, outputs
                        gc.collect()
                del baselines, references, x
            del weights
            torch.cuda.synchronize()
            torch.cuda.empty_cache()  # Between independent geometries only; not a device cache flush.
        report["complete"] = True
    except BaseException as error:
        try:
            report["failure"] = {"type": type(error).__name__, "message": str(error)}
            write(args.out / "failure.json", report)
        except BaseException as diagnostic:
            diagnostic_failures[0] = diagnostic
        raise_failures(error, diagnostic_failures)
    write(args.out / "result.json", report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--root-gpu-lease", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()

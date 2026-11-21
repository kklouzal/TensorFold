"""dmv13 dense projections vs the FP16-decoded torch.matmul path on real DeepSeek-V4.1 rank-3 slices (one V100).

Usage: python ds41_dense_bench.py [out.json]. Reports per projection: plane vs FP16 bytes, numerics (max abs / rel
error vs the FP64 product of the ggml FP32 weights, for dmv13 and for the FP16 path, plus dmv13 vs FP16), decode
times M = 1..8, 16, 24, 32 (CUDA-graph replay, LAYERS distinct weight copies cycled so weights stream from HBM),
and prefill times M = 64..8192 (restore + cuBLAS, the FP16 GEMM, dense_rows in 32-row chunks up to 256 rows).
Also checks the BF16 router gate: fp32_projection with BF16 vs FP32 coefficients (bit equality) when the lane
kernels are importable.
"""

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

from tensorfold.cuda.sm70 import ds41_dense as dense
from tensorfold.families.deepseek_v4.cuda.weights import TP, Loader, Shards

sys.path.insert(0, str(Path(__file__).parent))
from test_ds41_dense import ggml_fp32  # noqa: E402

RANK, LAYERS = 3, 6
MODEL = Path("/home/kklouzal/models/ds41-tf/config.json")
DECODE = (1, 2, 3, 4, 5, 6, 7, 8, 16, 24, 32)
PREFILL = (64, 128, 256, 512, 2048, 8192)
# name -> (tensors, slice kind) ; same-input tensors form one group (one launch)
L6 = tuple(range(LAYERS))
SPECS = {
    "qa_kv": (("attn_q_a", "attn_kv"), None, L6),
    "wq_b": (("attn_q_b",), "column", L6),
    "wo_a": (("attn_output_a",), "wo_a", L6),
    "wo_b": (("attn_output_b",), "row", L6),
    "sh_gu": (("ffn_gate_shexp", "ffn_up_shexp"), "column", L6),
    "sh_down": (("ffn_down_shexp",), "row", L6),
    "engram_wkv": (("engram_wkv",), "column", (1, 14)),
    "head": (("output",), "column", (None,)),
}


def tensor(name, layer):
    return f"{name}.weight" if layer is None else f"blk.{layer}.{name}.weight"


def load(loader, shards, name, kind, layer):
    full = tensor(name, layer)
    t = shards.tensors[full]
    if kind is None:
        return loader.planes(full), loader.dense(full), (0, t.rows), (0, t.shape[-1])
    if kind == "column":
        n = t.rows // TP
        rows = (RANK * n, n)
        return loader.planes(full, rows=rows), loader.dense(full, rows=rows), rows, (0, t.shape[-1])
    if kind == "wo_a":
        rows = (RANK * 1024, 1024)
        return loader.planes(full, rows=rows), loader.dense(full, rows=rows), rows, (0, t.shape[-1])
    n = t.shape[-1] // TP
    cols = (RANK * n, n)
    return loader.planes(full, cols=cols), loader.dense(full, cols=cols), (0, t.rows), cols


def fp32_weight(shards, full, rows, cols):
    t, data = shards.rows(full)
    raw = np.array(data[rows[0]:rows[0] + rows[1]])
    return torch.from_numpy(ggml_fp32(raw, t.ggml_type, t.shape[-1])[:, cols[0]:cols[0] + cols[1]].copy())


def timed(fn, reps=20, calls=24):
    """Median microseconds per call of ``fn(i)`` replayed from one CUDA graph of ``calls`` calls."""
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for i in range(3):
            fn(i)
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for i in range(calls):
            fn(i)
    times = []
    for _ in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000 / calls)
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def eager(fn, reps=10):
    for i in range(2):
        fn(i)
    torch.cuda.synchronize()
    times = []
    for r in range(reps):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        fn(r)
        end.record()
        end.synchronize()
        times.append(start.elapsed_time(end) * 1000)
    return float(np.median(times)), float(np.min(times)), float(np.max(times))


def errors(got, ref):
    diff = (got.double() - ref).abs()
    return {"max_abs": float(diff.max()), "max_rel": float((diff / ref.abs().clamp_min(1e-3)).max()),
            "rel_l2": float(diff.norm() / ref.norm())}


def main(out_path):
    config = json.loads(MODEL.read_text())
    shards = Shards(Path(config["gguf_file"]), Path(config["engram_file"]))
    device = torch.device("cuda")
    loader = Loader(shards, RANK, device)
    scratch = dense.Scratch(device)
    report = {"rank": RANK, "layers_cycled": LAYERS, "device": torch.cuda.get_device_name(), "ops": {}}
    torch.manual_seed(0)
    for op, (names, kind, layers) in SPECS.items():
        groups, fp16 = [], []
        copies = len(layers)
        for layer in layers:
            loaded = [load(loader, shards, name, kind, layer) for name in names]
            groups.append(dense.Group([p for p, _, _, _ in loaded], scratch))
            fp16.append([w for _, w, _, _ in loaded])
        k = groups[0].k
        entry = {"k": k, "n": groups[0].widths, "plane_bytes": groups[0].nbytes,
                 "fp16_bytes": sum(w.numel() * 2 for w in fp16[0])}
        # numerics: activations with the lane's scale (RMS-normalized, unit variance), layer 0
        x = torch.randn(32, k, dtype=torch.float16, device=device)
        refs = []
        for (planes, _, rows, cols), name in zip([load(loader, shards, n, kind, layers[0]) for n in names], names):
            w32 = fp32_weight(shards, tensor(name, layers[0]), rows, cols).to(device)
            refs.append(x.double() @ w32.double().t())
        got_d = groups[0](x)
        got_b = groups[0](x[:8])
        got_g = groups[0](torch.cat([x] * 2 + [x[:16]]))     # 80 rows: restore + cuBLAS band
        num = []
        for i, ref in enumerate(refs):
            f16 = torch.matmul(x, fp16[0][i].t())
            num.append({"dmv13_vs_fp64": errors(got_d[i], ref), "fp16_path_vs_fp64": errors(f16, ref),
                        "dmv13_vs_fp16_path": errors(got_d[i], f16.double()),
                        "gemm_band_vs_fp64": errors(got_g[i][:32], ref),
                        "m8_equals_m32_rows": bool(torch.equal(got_b[i], got_d[i][:8]))})
        entry["numerics"] = num
        # decode timing (graph replay, layers cycled)
        decode = {}
        for m in DECODE:
            xm = torch.randn(m, k, dtype=torch.float16, device=device)
            outs = [[torch.empty(m, n, dtype=torch.float16, device=device) for n in groups[0].widths]
                    for _ in range(copies)]
            planes_t = timed(lambda i: groups[i % copies](xm, outs[i % copies]))
            fp16_t = timed(lambda i: [torch.matmul(xm, w.t()) for w in fp16[i % copies]])
            decode[m] = {"dmv13_us": planes_t, "fp16_matmul_us": fp16_t}
        entry["decode"] = decode
        prefill = {}
        for m in (PREFILL if op != "head" else ()):          # logits rows stay in the decode band
            xm = torch.randn(m, k, dtype=torch.float16, device=device)
            row = {"restore_gemm_us": eager(lambda i: groups[i % copies](xm)),
                   "fp16_matmul_us": eager(lambda i: [torch.matmul(xm, w.t()) for w in fp16[i % copies]])}
            if m <= 256:
                outs = [torch.empty(m, n, dtype=torch.float16, device=device) for n in groups[0].widths]
                row["rows_chunked_us"] = eager(lambda i: groups[i % copies]._rows(xm, outs))
            prefill[m] = row
        entry["prefill"] = prefill
        report["ops"][op] = entry
        print(op, json.dumps({k2: v for k2, v in entry.items() if k2 not in ("decode", "prefill")}), flush=True)
        for m, v in decode.items():
            print(f"  M={m:3d} dmv13 {v['dmv13_us'][0]:8.1f} us  fp16 {v['fp16_matmul_us'][0]:8.1f} us", flush=True)
        for m, v in prefill.items():
            print(f"  M={m:5d} " + "  ".join(f"{a} {b[0]:9.1f}" for a, b in v.items()), flush=True)
        del groups, fp16
        torch.cuda.empty_cache()
    report["gate_bf16"] = gate_check(loader, device)
    print("gate", report["gate_bf16"], flush=True)
    Path(out_path).write_text(json.dumps(report, indent=1))


def gate_check(loader, device):
    try:
        from tensorfold.families.deepseek_v4.cuda.kernels.mhc import fp32_projection
    except ImportError as error:
        return {"skipped": str(error)}
    result = {}
    for layer in (0, 20, 39):
        name = f"blk.{layer}.ffn_gate_inp.weight"
        w32 = loader.floating(name, torch.float32)
        w16 = loader.router_gate(name)
        same = {"dtype": str(w16.dtype), "exact_values": bool(torch.equal(w16.float(), w32))}
        for m in (1, 2, 4, 6, 8, 64):
            x = torch.randn(m, 5120, dtype=torch.float16, device=device)
            same[f"M{m}"] = bool(torch.equal(fp32_projection(x, w16), fp32_projection(x, w32)))
        same["bytes_fp32"], same["bytes_now"] = w32.numel() * 4, w16.numel() * w16.element_size()
        result[layer] = same
    return result


if __name__ == "__main__":
    t0 = time.time()
    main(sys.argv[1] if len(sys.argv) > 1 else "ds41_dense_bench.json")
    print(f"done in {time.time() - t0:.0f} s")

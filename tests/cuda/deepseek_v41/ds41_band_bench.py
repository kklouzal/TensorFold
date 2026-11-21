"""DeepSeek-V4.1 decode-band pieces on one V100: exactness and CUDA-graph timings (development bench).

Usage: python ds41_band_bench.py [out.json]. Synthetic inputs at the lane's shapes (hidden 5120, 384 experts, BF16
gate, HC 4 x 5120, mHC fn [24, 20480]).

Exactness (bitwise, ``torch.equal``):
  * band.route_local == route_topk + Target.moe's former torch EP mapping (both EP groups, M 1..64 and 8192);
  * band.add_halves == a.float() + b.float();
  * batch invariance of the patch-89 band: short_rms_norm / fp32_projection (gate, mHC) / hc_transition over M
    rows == the same rows computed one at a time (M 7..64);
Bands: short_rms_norm M <= 64, the gate projection M <= 12, the HC transition M <= 16 (band.py).
Numerics: max relative error vs FP64 of the band kernels and of the former M > 6 paths (torch norm, F.linear FP32,
windowed per-token HC transition).
Timings: us per call inside a CUDA graph of 40 calls (distinct weights per call for the gate), M 1..32: former vs
band paths.
"""

import json
import sys

import torch

from tensorfold.families.deepseek_v4.cuda.kernels import band, mhc, mhc_fused, norm, router
from tensorfold.families.deepseek_v4.cuda.kernels.mhc import fp32_projection
from tensorfold.families.deepseek_v4.cuda.kernels.mhc_fused import hc_transition

dev = torch.device("cuda")
H, E = 5120, 384
out = {"exact": {}, "error": {}, "us": {}}
torch.manual_seed(0)


def graph_us(fn, reps=40, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for i in range(2):
            fn(0)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            fn(i)
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (iters * reps)


def rel(x, ref):
    return float(((x.double() - ref).abs().max() / ref.abs().max().clamp_min(1e-30)).item())


# ---------------------------------------------------------------- route_local / add_halves exactness
def old_route(logits, bias, first, count):
    p, ids = router.route_topk(logits.float(), bias, 6, 1.5)
    local = ids.to(torch.int32) - first
    mine = (local >= 0) & (local < count)
    local = torch.where(mine, local, torch.zeros_like(local)).contiguous()
    w = torch.where(mine, p.float(), torch.zeros_like(p, dtype=torch.float32)).contiguous()
    return local, w


bias = torch.randn(E, device=dev) * 0.1
ok = True
for m in (1, 2, 6, 7, 8, 16, 24, 32, 64, 8192):
    logits = torch.randn(m, E, device=dev) * 3
    logits[0, :8] = logits[0, 0]          # ties
    for first in (0, 192):
        a = old_route(logits, bias, first, 192)
        b = band.route_local(logits, bias, 6, 1.5, first, 192)
        ok &= torch.equal(a[0], b[0]) and torch.equal(a[1], b[1])
out["exact"]["route_local"] = bool(ok)
ok = True
for m in (1, 8, 32, 8192):
    x, y = torch.randn(m, H, device=dev).half(), torch.randn(m, H, device=dev).half()
    ok &= torch.equal(band.add_halves(x, y), x.float() + y.float())
out["exact"]["add_halves"] = bool(ok)

# ---------------------------------------------------------------- band invariance
inv = {}
for width in (128, 512, 1280, 5120):
    w = torch.rand(width, device=dev) + 0.5
    good = True
    for m in (7, 8, 16, 24, 32, 64):
        x = (torch.randn(m, width, device=dev) * 2).half()
        y = norm.short_rms_norm(x, w, 1e-20)
        good &= all(torch.equal(y[i:i + 1], norm.short_rms_norm(x[i:i + 1], w, 1e-20)) for i in range(m))
    inv[f"rms{width}"] = bool(good)
gate = (torch.randn(E, H, device=dev) * 0.02).bfloat16()
fn = torch.randn(24, 4 * H, device=dev) * 0.01
good_g = good_h = True
for m in (7, 8, 12):
    x = torch.randn(m, H, device=dev).half()
    y = fp32_projection(x, gate)
    good_g &= all(torch.equal(y[i:i + 1], fp32_projection(x[i:i + 1], gate)) for i in range(m))
    r = torch.randn(m, 4 * H, device=dev)
    z = fp32_projection(r, fn, normalized=True)
    good_h &= m > 6 or all(torch.equal(z[i:i + 1], fp32_projection(r[i:i + 1], fn, normalized=True)) for i in range(m))
inv["gate"], inv["mhc"] = bool(good_g), bool(good_h)


def hc_inputs(m):
    residual = torch.randn(m, 4, H, device=dev)
    y = torch.randn(m, H, device=dev).half()
    post = torch.rand(m, 4, device=dev) * 2
    comb = torch.rand(m, 4, 4, device=dev)
    pre = torch.rand(m, 4, device=dev)
    nw = torch.rand(H, device=dev) + 0.5
    scale = torch.rand(3, device=dev)
    base = torch.randn(24, device=dev) * 0.1
    return residual, y, post, comb, pre, nw, scale, base


good = True
for m in (7, 8, 12, 16):
    residual, y, post, comb, pre, nw, scale, base = hc_inputs(m)
    for mix in (True, False):
        args = (fn, scale, base) if mix else ()
        r1 = residual.clone()
        o, mx = hc_transition(r1, y, post, comb, pre, nw, *args)
        for i in range(m):
            ri = residual[i:i + 1].clone()
            oi, mi = hc_transition(ri, y[i:i + 1], post[i:i + 1], comb[i:i + 1], pre[i:i + 1], nw, *args)
            good &= torch.equal(oi, o[i:i + 1]) and torch.equal(ri, r1[i:i + 1])
            if mix:
                good &= all(torch.equal(a[i:i + 1], b) for a, b in zip(mx, mi))
inv["hc_transition"] = bool(good)
out["exact"]["band_invariance"] = inv

# ---------------------------------------------------------------- numerics vs FP64 (band vs former M > 6 paths)
m = 32
x = (torch.randn(m, 1280, device=dev) * 2).half()
w = torch.rand(1280, device=dev) + 0.5
ref = x.double() * torch.rsqrt(x.double().square().mean(-1, keepdim=True) + 1e-20) * w.double()
old = (x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + 1e-20) * w).half()
out["error"]["rms1280_M32"] = {"band": rel(norm.short_rms_norm(x, w, 1e-20), ref), "former": rel(old, ref)}
x = torch.randn(m, H, device=dev).half()
ref = x.double() @ gate.double().t()
out["error"]["gate_M32"] = {"band": rel(fp32_projection(x, gate), ref),
                            "former": rel(torch.nn.functional.linear(x.float(), gate.float()), ref)}
residual, y, post, comb, pre, nw, scale, base = hc_inputs(m)
r1, r2 = residual.clone(), residual.clone()
o1, mx1 = hc_transition(r1, y, post, comb, pre, nw, fn, scale, base)
saved = mhc_fused.HC_ROWS
mhc_fused.HC_ROWS = 8
o2, mx2 = hc_transition(r2, y, post, comb, pre, nw, fn, scale, base)
mhc_fused.HC_ROWS = saved
out["error"]["hc_transition_M32_band_vs_windowed"] = {
    "residual_equal": bool(torch.equal(r1, r2)), "inputs_max_abs": float((o1.float() - o2.float()).abs().max()),
    "pre_max_abs": float((mx1[0] - mx2[0]).abs().max()), "comb_max_abs": float((mx1[2] - mx2[2]).abs().max())}

# ---------------------------------------------------------------- timings (us per call, graph of 40 calls)
gates = [(torch.randn(E, H, device=dev) * 0.02).bfloat16() for _ in range(40)]
for m in (1, 2, 4, 6, 7, 8, 12, 16, 24, 32, 48, 64):
    x = torch.randn(m, H, device=dev).half()
    row = {}
    row["gate_former_sgemm"] = graph_us(lambda i: torch.nn.functional.linear(x.float(), gates[i].float()))
    row["gate_row_projection"] = graph_us(lambda i: fp32_projection(x, gates[i]))
    logits = torch.randn(m, E, device=dev)
    row["route_former"] = graph_us(lambda i: old_route(logits, bias, 192, 192))
    row["route_local"] = graph_us(lambda i: band.route_local(logits, bias, 6, 1.5, 192, 192))
    a, b = x.clone(), x.clone()
    row["add_former"] = graph_us(lambda i: a.float() + b.float())
    row["add_halves"] = graph_us(lambda i: band.add_halves(a, b))
    xn = (torch.randn(m, 1280, device=dev)).half()
    wn = torch.rand(1280, device=dev) + 0.5
    row["rms1280_former"] = graph_us(
        lambda i: (xn.float() * torch.rsqrt(xn.float().square().mean(-1, keepdim=True) + 1e-20) * wn).half())
    row["rms1280_band"] = graph_us(lambda i: norm.short_rms_norm(xn, wn, 1e-20))
    residual, y, post, comb, pre, nw, scale, base = hc_inputs(m)
    row["hc_band"] = graph_us(lambda i: hc_transition(residual, y, post, comb, pre, nw, fn, scale, base))
    mhc_fused.HC_ROWS = 8
    row["hc_former"] = graph_us(lambda i: hc_transition(residual, y, post, comb, pre, nw, fn, scale, base))
    mhc_fused.HC_ROWS = saved
    out["us"][m] = {k: round(v, 2) for k, v in row.items()}
    print(m, out["us"][m], flush=True)

print(json.dumps(out["exact"]), json.dumps(out["error"]), flush=True)
if len(sys.argv) > 1:
    json.dump(out, open(sys.argv[1], "w"), indent=1)

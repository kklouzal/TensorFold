"""One V100: the V3 HC transition with the row-blocked first kernel (hc_rows, ROWS_PER tokens a program) vs the
vendored per-token V3 and the windowed per-token kernels: bitwise equality and CUDA-graph us per call, M 1..64."""
import torch
from tensorfold.families.deepseek_v4.cuda.kernels import mhc_fused as F

dev, H = torch.device("cuda"), 5120
torch.manual_seed(0)
fn = torch.randn(24, 4 * H, device=dev) * 0.01


def graph_us(call, reps=40, iters=20):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        call(0); call(0)
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for i in range(reps):
            call(i)
    g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (iters * reps)


def hc_inputs(m):
    residual = torch.randn(m, 4, H, device=dev)
    y = torch.randn(m, H, device=dev).half()
    post = torch.rand(m, 4, device=dev) * 2
    comb = torch.rand(m, 4, 4, device=dev) * 0.25
    pre = torch.rand(m, 4, device=dev)
    nw = torch.rand(H, device=dev) + 0.5
    scale = torch.rand(3, device=dev)
    base = torch.randn(24, device=dev) * 0.1
    return residual, y, post, comb, pre, nw, scale, base


ok = True
for m in (1, 2, 4, 8, 12, 16, 24, 32, 48, 64):
    residual, y, post, comb, pre, nw, scale, base = hc_inputs(m)
    ref_r = residual.clone()
    ref = F._hc_transition3(ref_r, y, post, comb, pre, nw, fn, scale, base, 1e-20, 1e-6, 20, rows_per=1)
    row = {}
    for rp in (2, 4, 8, 16):
        r2 = residual.clone()
        got = F._hc_transition3(r2, y, post, comb, pre, nw, fn, scale, base, 1e-20, 1e-6, 20, rows_per=rp)
        same = torch.equal(r2, ref_r) and torch.equal(got[0], ref[0]) and all(torch.equal(a, b) for a, b in zip(got[1], ref[1]))
        ok &= same
        row[f"v3_rows{rp}"] = round(graph_us(lambda i: F._hc_transition3(residual, y, post, comb, pre, nw, fn, scale, base,
                                                                        1e-20, 1e-6, 20, rows_per=rp)), 2)
    row["v3_rows1"] = round(graph_us(lambda i: F._hc_transition3(residual, y, post, comb, pre, nw, fn, scale, base,
                                                                1e-20, 1e-6, 20, rows_per=1)), 2)
    saved = F.HC_ROWS
    F.HC_ROWS = 0
    row["windowed"] = round(graph_us(lambda i: F.hc_transition(residual, y, post, comb, pre, nw, fn, scale, base)), 2)
    F.HC_ROWS = saved
    print(m, row, flush=True)
print("bitwise equal (rows_per 2..16 vs 1):", ok)

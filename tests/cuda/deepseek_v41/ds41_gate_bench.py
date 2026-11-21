"""One V100: router gate projection above the row band: FP32 GEMM with BF16 widening (former) vs band.gate_projection
(Triton FMA-dot tiles reading BF16 bits): CUDA-graph us per call (40 distinct gate copies) and error vs FP64."""
import torch
from tensorfold.families.deepseek_v4.cuda.kernels import band
from tensorfold.families.deepseek_v4.cuda.kernels.mhc import fp32_projection


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

dev = torch.device("cuda")
torch.manual_seed(1)
gates = [(torch.randn(384, 5120, device=dev) * 0.02).bfloat16() for _ in range(40)]
for m in (1, 8, 12, 13, 16, 24, 32, 48, 64):
    x = torch.randn(m, 5120, device=dev).half()
    ref = x.double() @ gates[0].double().t()
    scale = ref.abs().max()
    former = torch.nn.functional.linear(x.float(), gates[0].float())
    new = band.gate_projection(x, gates[0])
    row = fp32_projection(x, gates[0]) if m <= band.GATE_ROWS else None
    err = lambda v: (float((v.double() - ref).abs().max() / scale), float((v.double() - ref).abs().mean() / scale))
    print(m, "us former %.1f gate_projection %.1f%s" % (
        graph_us(lambda i: torch.nn.functional.linear(x.float(), gates[i].float())),
        graph_us(lambda i: band.gate_projection(x, gates[i])),
        "" if row is None else " row %.1f" % graph_us(lambda i: fp32_projection(x, gates[i]))),
        "| rel err max/mean former %.2e/%.2e new %.2e/%.2e" % (*err(former), *err(new)),
        "" if row is None else "row %.2e/%.2e" % err(row),
        "| rows invariant", bool(all(torch.equal(band.gate_projection(x[i:i + 1].contiguous(), gates[0])[0], new[i]) for i in range(m))),
        flush=True)

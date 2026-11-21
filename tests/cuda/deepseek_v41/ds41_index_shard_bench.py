"""One-GPU timing of the K4 pieces at decode shapes (CUDA graph replays): per rank, the head-sharded scores
(index_scores, 4 heads, FP32 out) + FP16 rounding + index_topk vs the position-sharded scores (_shard_scores rank 0)
+ select_gathered (position-order copy + mu_topk). Collectives are not included (the FP32 [rows, width] all-reduce vs the FP16 [rows, width / 8]
all-gather + the query gather): see the 8-GPU A/B."""
import sys
import torch
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from test_ds41_index_shard import build, BLOCK  # noqa: E402
from tensorfold.families.deepseek_v4.cuda.kernels import index_shard as ish, mu_packed as pk
from tensorfold.families.deepseek_v4.cuda.kernels.mu_topk import topk as index_topk
from tensorfold.families.deepseek_v4.cuda.kernels.triton_utils import triton


def graph_us(fn, reps=8, iters=10):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn(); fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(reps):
            fn()
    g.replay(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        g.replay()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) * 1e3 / (reps * iters)


for rows, length, width, RATIO in ((1, 16384, 16384, 2), (32, 16384, 16384, 2), (1, 16384, 32768, 1),
                                   (8, 16384, 32768, 1), (32, 16384, 32768, 1), (32, 4096, 8192, 1)):
    layout, cache, table, q, w, pos, valid, req = build([length - r for r in range(rows)], width, 1, RATIO)
    part = torch.empty(rows, width, dtype=torch.float32, device="cuda")
    sel = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
    q4, w4 = q[:, :4].contiguous(), w[:, :4].contiguous()

    def old():
        pk.index_scores(q4, w4, cache, table, pos, valid, req, layout=layout, layer=0, out=part)
        index_topk(sel, part.half(), pos, valid, RATIO, 512)

    shards = torch.empty(8, rows, width // 8, dtype=torch.float16, device="cuda")
    blocks = width // 8 // ish.BN

    def new():
        ish._shard_scores[(rows, triton.cdiv(blocks, 4))](
            q, w, cache, table, pos, valid, req, shards[0], width // 8, table.stride(0), 0, PAGE=layout.page_bytes,
            BLOCK_SIZE=BLOCK, RATIO=RATIO, OFFSET=layout.regions[0][3], H=32, BN_=ish.BN, BLOCKS=4, WORLD_=8,
            num_warps=8)
        ish.select_gathered(shards, pos, valid, RATIO, out=sel)
    print(f"rows {rows} length {length} width {width}: head-sharded scores+topk {graph_us(old):.1f} us, "
          f"position-sharded {graph_us(new):.1f} us", flush=True)

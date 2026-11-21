"""K4 position-sharded decode indexer (index_shard) vs the head-sharded TP8 path, 8 ranks simulated on one device
(CUDA, or CPU under TRITON_INTERPRET=1).

Reference: per rank r, mu_packed.index_scores with heads [4 r, 4 r + 4) into FP32, the 8 partials summed in rank
order (FP32), rounded once to FP16, index_topk. Sharded: per rank r, _shard_scores (32 heads, key blocks b % 8 ==
r), the shards stacked as the all-gather lays them out, select_gathered. The FP16 scores may differ only by the FP32
summation order of the 32 head terms (asserted: <= 1 FP16 ulp where they differ, and rarely); the selections must
equal index_topk run on the sharded path's own scores (exact selection contract), and agree with the reference
selection except where its scores differ."""

import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels import index_shard as ish
from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
from tensorfold.families.deepseek_v4.cuda.kernels.mu_topk import topk as index_topk
from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import attention_qdq
from tensorfold.families.deepseek_v4.cuda.kernels.triton_utils import INTERPRET, triton

DEVICE = "cpu" if INTERPRET else "cuda"
pytestmark = pytest.mark.skipif(not INTERPRET and not torch.cuda.is_available(), reason="CUDA or TRITON_INTERPRET")
BLOCK = 64


def build(lengths, width, seed, RATIO=4):
    torch.manual_seed(seed)
    layout = pk.PackedLayout({0: RATIO}, BLOCK)
    pages_per = width * RATIO // BLOCK
    rows = len(lengths)
    cache = torch.zeros(rows * pages_per + 1, layout.page_bytes, dtype=torch.uint8, device=DEVICE)
    table = torch.arange(rows * pages_per, dtype=torch.int32, device=DEVICE).view(rows, pages_per)
    for r, length in enumerate(lengths):        # every compressed index row of request r up to its length
        positions = torch.arange(RATIO - 1, length, RATIO, device=DEVICE)
        keys = (torch.randn(positions.numel(), 128, device=DEVICE) * 0.5).half()
        req = torch.full((positions.numel(),), r, dtype=torch.int32, device=DEVICE)
        valid = torch.ones(positions.numel(), dtype=torch.bool, device=DEVICE)
        pk.insert_rows(cache, keys, positions, valid, req, table, layout=layout, layer=0, compressed=False)
    q = attention_qdq((torch.randn(rows * 32, 128, device=DEVICE) * 0.5).half(), fp4=True, group=32).view(rows, 32, 128)
    w = (torch.randn(rows, 32, device=DEVICE) * 0.2).half()
    pos = torch.tensor([length - 1 for length in lengths], dtype=torch.int64, device=DEVICE)
    valid = torch.ones(rows, dtype=torch.bool, device=DEVICE)
    req = torch.arange(rows, dtype=torch.int32, device=DEVICE)
    return layout, cache, table, q, w, pos, valid, req


@pytest.mark.parametrize("lengths,width,RATIO", [((16384, 9000, 2049, 300, 64), 8192, 4), ((4096, 4093, 2052), 2048, 4),
                                                ((16384, 12000, 600), 32768, 1), ((16384, 16000), 16384, 2)])
def test_sharded_indexer(lengths, width, RATIO):
    layout, cache, table, q, w, pos, valid, req = build(lengths, width, len(lengths), RATIO)
    rows = len(lengths)
    partial = torch.zeros(rows, width, dtype=torch.float32, device=DEVICE)
    for r in range(8):
        part = torch.zeros(rows, width, dtype=torch.float32, device=DEVICE)
        pk.index_scores(q[:, 4 * r:4 * r + 4].contiguous(), w[:, 4 * r:4 * r + 4].contiguous(), cache, table, pos,
                        valid, req, layout=layout, layer=0, out=part)
        partial += part
    ref_scores = partial.half()
    ref = torch.empty(rows, 512, dtype=torch.int32, device=DEVICE)
    index_topk(ref, ref_scores, pos, valid, RATIO, 512)

    part = width // 8
    shards = torch.zeros(8, rows, part, dtype=torch.float16, device=DEVICE)
    blocks = part // ish.BN
    for r in range(8):
        ish._shard_scores[(rows, triton.cdiv(blocks, 4))](
            q, w, cache, table, pos, valid, req, shards[r], part, table.stride(0), r, PAGE=layout.page_bytes,
            BLOCK_SIZE=BLOCK, RATIO=RATIO, OFFSET=layout.regions[0][3], H=32, BN_=ish.BN, BLOCKS=4, WORLD_=8,
            num_warps=8)
    got = ish.select_gathered(shards, pos, valid, RATIO)
    # the sharded scores in global order
    flat = shards.view(8, rows, blocks, ish.BN).permute(1, 2, 0, 3).reshape(rows, width)
    n = torch.clamp((pos + 1) // RATIO, max=width)
    live = torch.arange(width, device=DEVICE)[None, :] < n[:, None]
    differ = live & (flat.view(torch.int16) != ref_scores.view(torch.int16))
    ulps = (flat.float().view(torch.int32) - ref_scores.float().view(torch.int32)).abs()
    assert int(differ.sum()) <= max(1, int(live.sum()) // 100), "head-sum reorder changes more than 1% of scores"
    assert bool((torch.where(differ, ulps, 0) <= (1 << 13)).all()), "differences above one FP16 ulp"
    own = torch.empty(rows, 512, dtype=torch.int32, device=DEVICE)
    index_topk(own, torch.where(live, flat, torch.zeros_like(flat)).contiguous(), pos, valid, RATIO, 512)
    assert torch.equal(got, own)
    if not bool(differ.any()):
        assert torch.equal(got, ref)
    print("differing scores", int(differ.sum()), "of", int(live.sum()),
          "selection equal to reference:", bool(torch.equal(got, ref)))

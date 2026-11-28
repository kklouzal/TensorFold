"""Candidate-pool consumer layers: scoring only the pool's slots selects what full-row scoring then compaction does."""

from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
if not torch.cuda.is_available():
    pytest.skip("CUDA required", allow_module_level=True)

from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
from tensorfold.families.deepseek_v4.cuda.kernels.mu_reindex import select_ranked

PAGE, BLOCK_SIZE = 68 * 64, 64


def fixture(width, tokens, heads, seed):
    g = torch.Generator(device="cuda").manual_seed(seed)
    pages = width // BLOCK_SIZE + 1
    cache = torch.randint(0, 256, (pages, PAGE), generator=g, device="cuda", dtype=torch.uint8)
    cache.view(pages, 64, 68)[:, :, 64:] = torch.randint(118, 134, (pages, 64, 4), generator=g, device="cuda",
                                                          dtype=torch.uint8)
    table = torch.arange(pages, dtype=torch.int32, device="cuda").repeat(tokens, 1).contiguous()
    q = (torch.randn((tokens, heads, 128), generator=g, device="cuda") * 0.3).half()
    w = (torch.randn((tokens, heads), generator=g, device="cuda") * 0.2).half()
    layout = SimpleNamespace(page_bytes=PAGE, block_size=BLOCK_SIZE, regions={7: (1, 64, 0, 0)})
    return g, cache, table, q, w, layout


@pytest.mark.parametrize("score_dtype", [torch.float16, torch.float32])
def test_candidate_rows_select_like_full_scoring(score_dtype):
    width, tokens, heads, blocks, block = 65536, 3, 4, 2048, 8
    g, cache, table, q, w, layout = fixture(width, tokens, heads, 3)
    positions = torch.tensor([width - 5, 40000, 17000], device="cuda")
    valid = torch.ones(tokens, dtype=torch.bool, device="cuda")
    req = torch.arange(tokens, dtype=torch.int32, device="cuda")
    lengths = positions + 1
    # a layer-20-like pool: random distinct reachable blocks ascending, the newest block pinned, -1 padded
    cand = torch.full((tokens, blocks), -1, dtype=torch.int32, device="cuda")
    for t in range(tokens):
        reach = int((lengths[t] + block - 1) // block)
        pick = torch.randperm(reach - 1, generator=torch.Generator().manual_seed(t))[:blocks - 1 - 7 * t]
        chosen = torch.cat([pick, torch.tensor([reach - 1])]).sort().values
        cand[t, :len(chosen)] = chosen.to(torch.int32).cuda()
    identity = lambda scores: scores
    full = torch.empty((tokens, width), dtype=score_dtype, device="cuda")
    pk.index_scores(q, w, cache, table, positions, valid, req, layout=layout, layer=7, out=full)
    want = torch.empty((tokens, 512), dtype=torch.int32, device="cuda")
    select_ranked(want, identity(full).to(torch.float16), positions, valid, 1, pk._topk,
                  candidates=cand, block=block, publish=False)
    got = torch.empty_like(want)
    pk._candidate_rows(got, q, w, positions, valid, req, cache, table, cand, block, layout=layout, source=7, ratio=1,
                       width=width, all_reduce=identity, score_dtype=score_dtype)
    assert torch.equal(got, want)

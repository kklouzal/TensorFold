"""Captured batched score packets keep the original aligned row-one rule."""

from types import SimpleNamespace

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.forward import candidates
from tensorfold.families.qwen4_exp.cuda.state import CAND

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA candidate reduction geometry")


@pytest.mark.parametrize("width", [1, 31, 9948, 9949, 31040])
@pytest.mark.parametrize("rows", [1, 6, 8])
def test_captured_packets_match_original_row_one_scores_ids_and_lse(width, rows):
    source = torch.empty((rows, width), dtype=torch.float32, device="cuda")
    mapping = torch.arange(width, dtype=torch.int64, device="cuda").flip(0) + 513
    b = SimpleNamespace(cand=torch.empty((rows, 2*CAND+1), device="cuda"),
                        cand_all=torch.empty(rows*(2*CAND+1), device="cuda"), world=1)
    w = SimpleNamespace(meta={"execution_provider": "sm70"},
                        comm=SimpleNamespace(all_gather=lambda src, dst: dst.copy_(src.flatten())))
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        source.zero_()
        candidates(w, b, source, rows, id_map=mapping)
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=side):
        candidates(w, b, source, rows, id_map=mapping)
    for seed in (17, 38, 101):
        values = torch.randn(source.shape, generator=torch.Generator(device="cuda").manual_seed(seed),
                             device="cuda") * 4 - 8
        if seed == 101:
            values[:, :min(width, 40)] = 5  # ambiguous top32 still uses the serial producer's exact words
        source.copy_(values)
        graph.replay()
        expected = torch.empty_like(b.cand)
        kept = min(CAND, width)
        for row in range(rows):
            original = values[row:row+1].clone()
            scores, columns = torch.topk(original, kept, dim=-1, sorted=False)
            expected[row:row+1, :kept] = scores
            expected[row:row+1, CAND:CAND+kept] = mapping[columns].to(torch.int32).view(torch.float32)
            expected[row:row+1, kept:CAND] = -float("inf")
            expected[row:row+1, CAND+kept:2*CAND].view(torch.int32).fill_(-1)
            expected[row:row+1, -1:] = torch.logsumexp(original, dim=-1, keepdim=True)
        torch.testing.assert_close(b.cand_all.view(rows, -1).view(torch.int32),
                                   expected.view(torch.int32), rtol=0, atol=0)
    graph.reset()

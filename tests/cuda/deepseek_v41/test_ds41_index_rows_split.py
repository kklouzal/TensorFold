"""Row-split prefill indexer (patch 180 at TP8, attention_ops.select_rows_split): the eight simulated ranks' merged
selections equal index_topk over the TP8 reference scores (per-rank FP16 4-head sums of relu(q.k) * w, then the
FP32 rank-ordered sum rounded to FP16). Runs on CUDA, or on CPU under TRITON_INTERPRET=1."""

import os

import pytest
import torch

import tensorfold.families.deepseek_v4.cuda.kernels.attention_ops as ops

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
pytestmark = pytest.mark.skipif(DEVICE == "cpu" and os.environ.get("TRITON_INTERPRET") != "1",
                                reason="CUDA or the Triton interpreter required")
WORLD = 8


def _reference(q, w, keys, positions, valid, ratio):
    tokens = q.shape[0]
    scores = torch.mm(q.reshape(-1, 128), keys.T).view(tokens, 32, -1)
    scores = scores.relu().mul(w[:, :, None])
    total = None
    for g in range(WORLD):           # one rank's Tensor.sum(1) over its 4 heads, FP16; FP32 sum in rank order
        part = scores[:, 4 * g:4 * g + 4].sum(1).float()
        total = part if total is None else total + part
    out = torch.empty(tokens, 512, dtype=torch.int32, device=q.device)
    return ops.index_topk(out, total.half().contiguous(), positions, valid, ratio, 512)


@pytest.mark.parametrize("tokens,first,ratio", [(300, 2000, 4), (256, 2100, 4), (700, 1500, 2)])
def test_row_split_selections_equal_the_tp8_reference(tokens, first, ratio):
    g = torch.Generator(device="cpu").manual_seed(tokens)
    # FP4-like grid values: every FP32 dot is exact (as for the model's FP4-QDQ'd queries and keys), so scores do
    # not depend on the GEMM's blocking
    q = (torch.randint(-6, 7, (tokens, 32, 128), generator=g) * 0.25).half().to(DEVICE)
    w = (torch.randn(tokens, 32, generator=g) * 0.2).half().to(DEVICE)
    end = (first + tokens) // ratio
    keys = (torch.randint(-6, 7, (max(end, 512), 128), generator=g) * 0.25).half().to(DEVICE)
    positions = torch.arange(first, first + tokens, device=DEVICE)
    valid = torch.ones(tokens, dtype=torch.bool, device=DEVICE)
    want = _reference(q, w, keys, positions, valid, ratio)

    picks = {}

    def run(rank, phase):
        def all_gather(send, recv):
            calls.append(send.shape)
            if len(calls) <= 2:       # queries, then weights: every rank's heads of every row
                full = q if len(calls) == 1 else w
                for src in range(WORLD):
                    part = full[:, 4 * src:4 * src + 4]
                    recv[src][:tokens].copy_(part)
                    recv[src][tokens:].zero_()
            else:                     # selections: the other ranks' picks from the first phase
                picks[rank] = send.clone()
                for src in range(WORLD):
                    recv[src].copy_(picks.get(src, send) if phase else send)
        calls = []
        return ops.select_rows_split(q[:, 4 * rank:4 * rank + 4].contiguous(), w[:, 4 * rank:4 * rank + 4].contiguous(),
                                     keys, positions, valid, ratio, first_position=first, rank=rank, world=WORLD,
                                     all_gather=all_gather)

    for rank in range(WORLD):
        run(rank, 0)
    for rank in range(WORLD):
        got = run(rank, 1)
        assert torch.equal(got, want), rank

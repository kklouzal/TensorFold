"""DeepSeek-V4.1 lane collectives (``families/deepseek_v4/cuda/collectives.py``) on CPU.

``rank_sum`` (the gather path's consumer) under TRITON_INTERPRET=1: bitwise the rank-ordered FP32 adds rounded once
(what the LL and bulk row all-reduces produce), including signed zeros, infinities, NaN and subnormals. Path
selection: a table entry is used only where the communicator's mailbox admits the shape (else NCCL), and the LL /
bulk entries fall back to NCCL when TF_SMALL_ALLREDUCE=0 disabled them. The 8-GPU in-graph comparison against
NCCL and the gathered rank-order oracle is the K3 microbenchmark (ledger).
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET", "0") != "1" or torch.cuda.is_available(),
    reason="CPU Triton-interpreter test (TRITON_INTERPRET=1, no CUDA)",
)

if os.environ.get("TRITON_INTERPRET", "0") == "1":
    from tensorfold.families.deepseek_v4.cuda import collectives as C


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("shape", [(1, 5120), (3, 136), (2, 1500)])
def test_rank_sum_is_rank_ordered_fp32_rounded_once(dtype, shape):
    generator = torch.Generator().manual_seed(41)
    parts = torch.randn((8,) + shape, generator=generator) * torch.logspace(-30, 4, 8)[:, None, None]
    flat = parts.view(8, -1)
    flat[:, 0] = torch.tensor([0.0, -0.0, -0.0, 0.0, -0.0, -0.0, -0.0, -0.0])
    flat[:, 1] = -0.0
    flat[:, 2] = torch.tensor([1.0, float("inf"), 2.0, -float("inf"), 0, 0, 0, 0])
    flat[:, 3] = torch.tensor([1.0, float("nan")] + [0.0] * 6)
    flat[:, 4] = 1e-45                                   # FP32 subnormals
    flat[:, 5] = torch.tensor([65504.0, 16.0] + [0.0] * 6)  # the FP16 overflow midpoint (rounds to inf)
    expected = parts[0].clone()
    for r in range(1, 8):
        expected = expected + parts[r]
    expected = expected.to(dtype)
    out = torch.empty(shape, dtype=dtype)
    C.rank_sum(parts.contiguous(), out)
    assert torch.equal(out.view(torch.int16 if dtype == torch.float16 else torch.int32),
                       expected.view(torch.int16 if dtype == torch.float16 else torch.int32))


class _Small:
    def reduce_lines(self, rows, dims):
        return 1 if rows * dims <= 4 * 5120 else 0

    def bulk_chunk_rows(self, rows, dims, f32):
        return 1 if rows * dims <= 32 * 5120 else 0


class _Comm:
    small_gather = _Small()
    small_gather_bytes = 2 * 5120 * 4


def test_path_selection_respects_admission(monkeypatch):
    monkeypatch.setattr(C, "_PATHS", {5120: {1: "gather", 3: "gather", 4: "ll", 8: "ll", 16: "bulk", 64: "bulk"}})
    c = C.Collectives(_Comm())
    assert [c.path(r, 5120) for r in (1, 3, 4, 8, 16, 64, 2)] == ["gather", "nccl", "ll", "nccl", "bulk", "nccl", "nccl"]
    assert c.path(1, 4096) == "nccl"
    c.reduce = False                                     # TF_SMALL_ALLREDUCE=0
    assert [c.path(r, 5120) for r in (1, 4, 16)] == ["gather", "nccl", "nccl"]
    c = C.Collectives(object())                          # no small collectives: NCCL everywhere
    assert c.path(1, 5120) == "nccl"

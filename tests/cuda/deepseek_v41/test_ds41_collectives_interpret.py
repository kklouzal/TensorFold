"""DeepSeek-V4.1 lane collectives (``families/deepseek_v4/cuda/collectives.py``) on CPU.

``rank_sum`` (the gather path's consumer) under TRITON_INTERPRET=1: bitwise the rank-ordered FP32 adds rounded once
(what the LL and bulk row all-reduces produce), including signed zeros, infinities, NaN and subnormals. Path
selection: by elements per rank, each path only where the communicator's mailbox admits the shape (else NCCL), and
no LL row all-reduce when TF_SMALL_ALLREDUCE=0 disabled it. The 8-GPU in-graph comparison against
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
        return 1 if rows * dims <= 8 * 5120 else 0


class _Comm:
    small_gather = _Small()
    small_gather_bytes = 5120 * 4


def test_path_selection_by_elements_and_admission():
    c = C.Collectives(_Comm())
    shapes = [(1, 5120), (2, 5120), (8, 5120), (9, 5120), (11, 5120), (12, 5120), (1, 32768), (1, 65536), (2, 2560)]
    assert [c.path(r * d, d) for r, d in shapes] == ["gather", "ll", "ll", "nccl", "nccl", "nccl", "ll", "nccl", "gather"]
    c.reduce = False                                     # TF_SMALL_ALLREDUCE=0
    assert [c.path(r * 5120, 5120) for r in (1, 2)] == ["gather", "nccl"]
    c = C.Collectives(object())                          # no small collectives: NCCL everywhere
    assert c.path(5120, 5120) == "nccl"

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f benchmarks/deepseek_v41/tests/test_topk_par.py (blob 1f0b6191a5215e5e27b333279462f11724c2f922).
# TensorFold changes: imports; the script body is one pytest function asserting its own "ALL OK" criterion;
# its CUDA-event timing loop is dropped (benchmark output, not a check).
import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels.index_topk import index_topk, index_topk_parallel

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_index_topk_parallel_equals_single():
    torch.manual_seed(0)
    ok = True
    for rows, width, start in ((6, 524288, 400000), (6, 524288, 300), (3, 131072, 100000), (6, 65536, 65000)):
        sc = torch.randn(rows, width, device="cuda").half()
        sc[:, ::7] = 0.5  # ties
        pos = torch.arange(start, start + rows, device="cuda")
        valid = torch.ones(rows, dtype=torch.bool, device="cuda")
        a = torch.empty(rows, 512, dtype=torch.int32, device="cuda")
        b = torch.empty_like(a)
        index_topk(a, sc, pos, valid, 1, 512)
        index_topk_parallel(b, sc, pos, valid, 1, 512)
        same = torch.equal(a, b)
        ok &= same
        print(f"rows {rows} width {width} start {start} equal={same}")
    assert ok

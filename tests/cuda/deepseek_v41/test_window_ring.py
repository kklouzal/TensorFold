# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f benchmarks/deepseek_v41/tests/test_window_ring.py (blob 39a73b301495729defe144d136f2e2566c2de2f9).
# TensorFold changes: imports; the script body is one pytest function asserting its own "ALL OK" criterion;
# its CUDA-event timing loop is dropped (benchmark output, not a check).
"""G4 window ring vs paged window cache: identical attention over chunked prefill + decode."""

import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels.attention_ops import (
    WINDOW_RING,
    attend_paged_latents,
    insert_latents,
    insert_window_ring,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_window_ring_matches_paged_window():
    torch.manual_seed(0)
    dev = "cuda"
    H, BS = 16, 64
    total = 1200
    cache = torch.zeros(total // BS + 2, BS, 512, dtype=torch.float16, device=dev)
    table = torch.arange(cache.shape[0], dtype=torch.int32, device=dev)[None]
    ring = torch.zeros(WINDOW_RING, 512, dtype=torch.float16, device=dev)
    sink = torch.randn(H, device=dev)
    ok, start = True, 0
    for T in (300, 200, 6, 6, 3, 400, 1, 5):
        pos = torch.arange(start, start + T, device=dev)
        valid = torch.ones(T, dtype=torch.bool, device=dev)
        values = (torch.randn(T, 512, device=dev) * 0.3).half()
        q = (torch.randn(T, H, 512, device=dev) * 0.3).half()
        insert_latents(cache, values, pos, valid, table, block_size=BS)
        old = attend_paged_latents(q, pos, valid, cache, table, sink, window_block=BS)
        new = attend_paged_latents(
            q, pos, valid, None, table, sink, window_block=BS, window_ring=ring, window_values=values
        )
        insert_window_ring(ring, values, pos, valid)
        same = torch.equal(old, new)
        ok &= same
        print(f"chunk T={T} start={start} equal={same} maxdiff={(old.float() - new.float()).abs().max().item():.3g}")
        start += T
    assert ok

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f benchmarks/deepseek_v41/tests/test_fused_sparse.py (blob efbfdf0350f3d00fe187dee96a025e225eefe4b4).
# TensorFold changes: imports; the script body is one pytest function asserting its own "ALL OK" criterion;
# its CUDA-event timing loop is dropped (benchmark output, not a check).
"""Fused sparse attention vs the gather + bmm path (ring window + compressed top-k)."""

import pytest
import torch

import tensorfold.families.deepseek_v4.cuda.kernels.attention_ops as ops

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_fused_sparse_matches_gather_bmm(monkeypatch):
    torch.manual_seed(0)
    dev = "cuda"
    H, BS, R = 16, 64, 2
    ok = True
    for T, start, comp_rows in ((300, 0, 0), (512, 700, 600), (2048, 5000, 2600)):
        total = start + T
        ring = (torch.randn(ops.WINDOW_RING, 512, device=dev) * 0.3).half()
        values = (torch.randn(T, 512, device=dev) * 0.3).half()
        q = (torch.randn(T, H, 512, device=dev) * 0.3).half()
        pos = torch.arange(start, total, device=dev)
        valid = torch.ones(T, dtype=torch.bool, device=dev)
        valid[5] = False
        sink = torch.randn(H, device=dev)
        table = torch.arange(total // BS + 2, dtype=torch.int32, device=dev)[None]
        kwargs = {"window_block": BS, "window_ring": ring, "window_values": values}
        if comp_rows:
            ccache = (torch.randn(total // BS + 2, BS // R, 512, device=dev) * 0.3).half()
            n_c = (pos + 1) // R
            topk = (torch.rand(T, 512, device=dev) * n_c[:, None]).long()
            topk = torch.where(torch.rand(T, 512, device=dev) < 0.05, -1, topk).int()
            kwargs.update(
                compressed_cache=ccache, compressed_table=table, compressed_block=BS, ratio=R, topk=topk.contiguous()
            )
        monkeypatch.setattr(ops, "_FUSED_SPARSE", False)
        ref = ops.attend_paged_latents(q, pos, valid, None, table, sink, **kwargs)
        monkeypatch.setattr(ops, "_FUSED_SPARSE", True)
        new = ops.attend_paged_latents(q, pos, valid, None, table, sink, **kwargs)
        diff = (new.float() - ref.float()).abs()
        rel = (diff.norm() / ref.float().norm()).item()
        good = torch.isfinite(new).all().item() and rel < 5e-3
        ok &= good
        print(f"T={T} rel={rel:.2e} max={diff.max().item():.3g} ok={good}")
    assert ok

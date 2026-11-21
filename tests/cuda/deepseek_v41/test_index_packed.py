# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f benchmarks/deepseek_v41/tests/test_index_packed.py
# (blob aaf0a25c759ec59d3cecb58d0390d8d30a2af47c).
# TensorFold changes: imports; the VLLM_V41K_LIB native-library default is dropped (that CUDA decode-scores path is
# not vendored, so the Triton kernel always runs); timing is dropped. Upstream prints the top-512 overlap of the
# packed path against the FP16 gather + mm path and states no pass threshold, so this port asserts only the
# output contract (shape, ascending valid indices below each row's live length, -1 padding) and prints the
# overlap; an agreement threshold remains unspecified.
"""Packed fused indexer vs the gather + mm path: top-512 agreement."""

import pytest
import torch

import tensorfold.families.deepseek_v4.cuda.kernels.attention_ops as ops
from tensorfold.families.deepseek_v4.cuda.kernels.kv_pack import pack_rows
from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import attention_qdq

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_packed_indexer_contract_and_overlap():
    torch.manual_seed(0)
    H = 8
    ident = lambda x: x
    for T, start, ratio, rows in (
        (6, 9000, 2, 8192),
        (6, 15000, 1, 16384),
        (512, 3000, 2, 8192),
        (2048, 12000, 1, 16384),
        (6, 400000, 1, 524288),
    ):
        raw = (torch.randn(rows, 128, device="cuda") * 0.5).half()
        ref_keys = attention_qdq(raw, fp4=True, group=32)
        packed = pack_rows(raw, False)
        q = attention_qdq(
            (torch.randn(T, H, 128, device="cuda") * 0.5).half().reshape(T * H, 128).contiguous(),
            fp4=True,
            group=32,
        ).view(T, H, 128)
        w = (torch.randn(T, H, device="cuda") * 0.2).half()
        pos = torch.arange(start, start + T, device="cuda")
        valid = torch.ones(T, dtype=torch.bool, device="cuda")
        old = ops.select_compressed_positions(q, w, ref_keys, pos, valid, ratio, ident) if rows <= 16384 else None
        new = ops.select_packed_positions(q, w, packed, pos, valid, ratio, ident)
        assert new.shape == (T, 512) and new.dtype == torch.int32
        n = torch.clamp((pos + 1) // ratio, max=rows)
        for r in range(T):
            live = new[r][new[r] >= 0]
            assert int(live.numel()) == min(512, int(n[r]))
            assert bool((live < n[r]).all()) and bool((live[1:] > live[:-1]).all())
        if old is not None:
            agree = []
            for r in range(T):
                a, b = set(old[r].tolist()) - {-1}, set(new[r].tolist()) - {-1}
                agree.append(len(a & b) / max(1, len(a)))
            exact = sum(1 for x in agree if x == 1.0)
            print(
                f"T={T} start={start} ratio={ratio}: top-512 overlap min {min(agree):.4f} "
                f"mean {sum(agree) / T:.5f} exact rows {exact}/{T}"
            )

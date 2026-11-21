# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f benchmarks/deepseek_v41/tests/test_kv_pack.py (blob 3ff092055173db89c095e285c884a0c47fc13113).
# TensorFold changes: imports; the script body is one pytest function asserting its own "ALL OK" criterion;
# its CUDA-event timing loop is dropped (benchmark output, not a check).
"""Packed KV rows reproduce attention_qdq FP16 values bit for bit."""

import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels.kv_pack import pack_rows, unpack_rows
from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import attention_qdq

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_kv_pack_bit_exact():
    torch.manual_seed(0)
    ok = True
    for scale in (1e-6, 1e-3, 0.05, 1.0, 30.0, 3000.0):
        for compressed in (True, False):
            dim = 512 if compressed else 128
            x = (torch.randn(4096, dim, device="cuda") * scale).half()
            x[0].zero_()
            x[1, :17] = 0  # zero groups
            ref = attention_qdq(x, fp4=True, group=16 if compressed else 32, e4m3_scale=compressed)
            got = unpack_rows(pack_rows(x, compressed), compressed)
            same = torch.equal(ref.view(torch.int16), got.view(torch.int16))
            ok &= same
            mismatches = int((ref.view(torch.int16) != got.view(torch.int16)).sum())
            print(f"scale {scale:g} compressed={compressed} bit-exact={same} mismatches={mismatches}")
    assert ok

# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Ported from 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f benchmarks/deepseek_v41/tests/test_packed_attn.py
# (blob 695c9b3f3276d0bc15168e577ec0a95e4f02c893).
# TensorFold changes: imports (_fill_selected from the vendored kv_store kernels); the script body is one pytest
# function asserting its own "ALL OK" criterion.
"""Packed compressed rows (hot ring + staging / selected buffer) vs the FP16 paged cache."""

from types import SimpleNamespace

import pytest
import torch

import tensorfold.families.deepseek_v4.cuda.kernels.attention_ops as ops
from tensorfold.families.deepseek_v4.cuda.kernels.kv_pack import pack_rows
from tensorfold.families.deepseek_v4.cuda.kernels.kv_store import _fill_selected
from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import attention_qdq

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_packed_attention_bit_exact_vs_paged_fp16():
    torch.manual_seed(0)
    dev = "cuda"
    H, BS, R = 16, 64, 2
    ok = True
    for T, start, hot_rows in ((300, 5000, 1024), (6, 9000, 1024), (6, 3000, 4096), (2048, 12000, 2048)):
        total = start + T
        rows = total // R + 1
        raw = (torch.randn(rows, 512, device=dev) * 0.3).half()
        fp16 = attention_qdq(raw, fp4=True, group=16, e4m3_scale=True)
        packed = pack_rows(raw, True)
        # paged FP16 reference cache: identity block table
        ccache = torch.zeros(rows // (BS // R) + 2, BS // R, 512, dtype=torch.float16, device=dev)
        ccache.view(-1, 512)[:rows] = fp16
        table = torch.arange(ccache.shape[0], dtype=torch.int32, device=dev)[None]
        ring = (torch.randn(ops.WINDOW_RING, 512, device=dev) * 0.3).half()
        values = (torch.randn(T, 512, device=dev) * 0.3).half()
        q = (torch.randn(T, H, 512, device=dev) * 0.3).half()
        pos = torch.arange(start, total, device=dev)
        valid = torch.ones(T, dtype=torch.bool, device=dev)
        sink = torch.randn(H, device=dev)
        n_c = (pos + 1) // R
        topk = (torch.rand(T, 512, device=dev) * n_c[:, None]).long()
        topk = torch.where(torch.rand(T, 512, device=dev) < 0.05, -1, topk).int().contiguous()
        ref = ops.attend_paged_latents(
            q, pos, valid, None, table, sink, window_block=BS, window_ring=ring, window_values=values,
            compressed_cache=ccache, compressed_table=table, compressed_block=BS, ratio=R, topk=topk,
        )
        newest = int(total) // R - 1
        hot = torch.zeros(hot_rows, 288, dtype=torch.uint8, device=dev)
        idx = torch.arange(max(0, newest - hot_rows + 1), newest + 1, device=dev)
        hot[idx % hot_rows] = packed[idx]
        old = newest - hot_rows + 1
        staging = torch.zeros(max(rows, 1), 288, dtype=torch.uint8, device=dev)
        if old > 0:
            staging[:old] = packed[:old]
        selected = torch.zeros(8, 512, 288, dtype=torch.uint8, device=dev)
        store = SimpleNamespace(hot={0: hot}, staging=staging, hot_rows={0: hot_rows}, selected=selected)
        if T <= 8:
            _fill_selected[(T, 8)](topk, pos, hot, packed, selected, T, hot_rows, RATIO=R, TOPK=512, BLOCK=64)
        new = ops.attend_paged_latents(
            q, pos, valid, None, table, sink, window_block=BS, window_ring=ring, window_values=values,
            ratio=R, topk=topk, packed=(store, 0),
        )
        same = torch.equal(ref, new)
        ok &= same
        mode = "selected" if T <= 8 else "hot+staging"
        maxdiff = (ref.float() - new.float()).abs().max().item()
        print(f"T={T} start={start} hot_rows={hot_rows} mode={mode} bit-exact={same} maxdiff={maxdiff:.3g}")
    assert ok


@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability() != (7, 0), reason="SM70 native")
@pytest.mark.parametrize("tokens,ratio", [(9, 2), (137, 1), (137, 2)])
def test_legacy_cold_working_pages_match_normal_attention(tokens, ratio):
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
    from tensorfold.families.deepseek_v4.cuda.target import Target

    source = 7
    dev = torch.device("cuda")
    generator = torch.Generator(device=dev).manual_seed(53)
    positions = torch.arange(tokens, device=dev)
    valid = torch.ones(tokens, device=dev, dtype=torch.bool)
    request = torch.zeros(tokens, device=dev, dtype=torch.int32)
    boundaries = torch.tensor([0, tokens], device=dev, dtype=torch.int32)
    slot = torch.tensor([3], device=dev, dtype=torch.int32)
    step = SimpleNamespace(valid=valid, token_req=request, qsl=boundaries, req_slot=slot)
    count = -(-tokens // 64)
    logical_topk = torch.arange(512, device=dev, dtype=torch.int32)[None, :]
    topk = torch.where(logical_topk < (positions[:, None] + 1) // ratio, logical_topk, -1)
    query = torch.randn(tokens, 8, 512, dtype=torch.float16, device=dev, generator=generator)
    window = torch.randn(tokens, 512, dtype=torch.float16, device=dev, generator=generator)
    compact = torch.randn(tokens, 512, dtype=torch.float16, device=dev, generator=generator)
    sink = torch.randn(8, device=dev, generator=generator)

    def attend(sources, table, ring):
        layout = pk.PackedLayout(sources, 64)
        cache = torch.full((count, layout.page_bytes), 0xA5, device=dev, dtype=torch.uint8)
        pk.insert_rows(cache, compact, positions, valid, request, table, layout=layout, layer=source, compressed=True)
        return Target._attend(None, query, step, ring, window, sink, cache=cache, block_table=table,
                              layout=layout, layer=source, topk=topk)

    identity = torch.arange(count, device=dev, dtype=torch.int32)[None, :]
    standard = attend({1: 1, source: ratio}, identity.flip(1), torch.zeros(5, pk.RING, 512, device=dev,
                                                                        dtype=torch.float16))
    poisoned = torch.full((1, pk.RING, 512), float("nan"), device=dev, dtype=torch.float16)
    working = attend({source: ratio}, identity, poisoned)
    assert torch.isfinite(working).all()
    assert torch.equal(working.view(torch.int16), standard.view(torch.int16))


@pytest.mark.skipif(torch.cuda.is_available() and torch.cuda.get_device_capability() != (7, 0), reason="SM70 codec")
def test_rotorquant_window_and_compressed_gather_match_native_codec():
    from tensorfold.cuda.sm70 import rotorquant_q4_native as native
    from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
    from tensorfold.families.deepseek_v4.cuda.kernels.kv_pack import I_ROW, ROTOR_ROW

    device = "cuda"
    gen = torch.Generator(device=device).manual_seed(11)
    raw = torch.randn((136, 1, 512), generator=gen, device=device, dtype=torch.float16)
    raw[..., 64:128] *= 11
    raw[..., 128:192] = 0
    codes = torch.empty((136, 1, 256), dtype=torch.uint8, device=device)
    scales = torch.empty((136, 1, 4), dtype=torch.float32, device=device)
    status = torch.tensor([0, -2], dtype=torch.int32, device=device)
    native.encode(raw, codes, scales, codec=11, status=status)
    packed = torch.cat((codes, scales.view(torch.uint8)), dim=-1).squeeze(1)
    decoded = native.decode(codes, scales, torch.empty_like(raw), codec=11, status=status).squeeze(1)
    assert status.tolist() == [0, -2]

    layout = pk.PackedLayout({3: 2, 7: 2}, 64, rotorquant=True)
    assert layout.compressed_row_bytes == ROTOR_ROW == 272
    assert layout.regions == {3: (2, 32, 0, 32 * ROTOR_ROW),
                              7: (2, 32, 32 * (ROTOR_ROW + I_ROW), 32 * (2 * ROTOR_ROW + I_ROW))}
    cache = torch.full((3, layout.page_bytes), 0xA5, dtype=torch.uint8, device=device)
    table = torch.tensor([[2, 0, -1], [1, -1, -1]], dtype=torch.int32, device=device)
    ring = torch.zeros((3, pk.RING, ROTOR_ROW), dtype=torch.uint8, device=device)

    history = torch.arange(132, dtype=torch.int32, device=device)
    history_req = torch.zeros_like(history)
    history_live = torch.ones_like(history, dtype=torch.bool)
    history_qsl = torch.tensor([0, 132], dtype=torch.int32, device=device)
    slots = torch.tensor([1, 0], dtype=torch.int32, device=device)
    pk.insert_rows(cache, packed[:132], history, history_live, history_req, table, layout=layout, layer=3,
                   compressed=True)
    pk.insert_ring(ring, packed[:132], history, history_live, history_req, history_qsl, slots)
    assert torch.equal(ring[1], torch.cat((packed[128:132], packed[4:128])))
    assert torch.count_nonzero(ring[[0, 2]]).item() == 0
    gathered = pk.gather_request_rows(cache, table[0], 66, layout=layout, layer=3, compressed=True)
    assert torch.equal(gathered[:64], packed[1:128:2])
    assert torch.count_nonzero(gathered[64:]).item() == 0
    assert torch.all(cache[:, 32 * ROTOR_ROW:] == 0xA5).item()
    assert torch.all(cache[1] == 0xA5).item()

    positions = torch.tensor([132, 133, 0, -1], dtype=torch.int32, device=device)
    requests = torch.tensor([0, 0, 1, -1], dtype=torch.int32, device=device)
    live = torch.tensor([True, True, True, False], device=device)
    qsl = torch.tensor([0, 2, 3], dtype=torch.int32, device=device)
    topk = torch.tensor([[0, 31, 32, 63, 64, 65, 66, -1]] * 4, dtype=torch.int32, device=device)
    keys = torch.empty((4, pk.WINDOW + 8, 512), dtype=torch.float16, device=device)
    unused = torch.empty(keys.shape[:-1], dtype=torch.bool, device=device)
    pk.gather_attention(keys, unused, 0, 4, positions, live, requests, qsl, slots, ring, packed[132:],
                        cache=cache, block_table=table, layout=layout, layer=3, topk=topk)

    expected = torch.zeros_like(keys)
    expected[0, :128], expected[1, :128] = decoded[5:133], decoded[6:134]
    expected[:2, 128:132] = decoded[[1, 63, 65, 127]]
    expected[2, 127] = decoded[134]
    expected_unused = torch.ones_like(unused)
    expected_unused[:2, :132] = False
    expected_unused[2, 127] = False
    assert torch.equal(unused, expected_unused)
    assert torch.equal(keys.view(torch.int16), expected.view(torch.int16))

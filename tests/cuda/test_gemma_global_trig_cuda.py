"""Exact CT global trig and old-consumer oracles; actual CUDA execution required."""
import pytest
import torch
import triton
import triton.language as tl

from tensorfold.families.gemma4.cuda import attention


@triton.jit
def _single_position_oracle(FREQ, COS, SIN, POS):
    # Independent one-dimensional launch evaluates the old consumer expression.
    key = tl.program_id(0) + POS
    col = tl.arange(0, 64)
    angle = key.to(tl.float32) * tl.load(FREQ + col)
    tl.store(COS + tl.program_id(0) * 64 + col, tl.cos(angle))
    tl.store(SIN + tl.program_id(0) * 64 + col, tl.sin(angle))


def frequency():
    inv = torch.pow(1e6, -torch.arange(0, 512, 2, dtype=torch.float32, device='cuda') / 512)
    inv[64:] = 0
    return inv


@pytest.mark.parametrize('context', [33, 513, 131072, 262144])
def test_table_exact_fp32_words_at_position_boundaries(context):
    inv = frequency()
    table = attention.GlobalTrig(inv, context)
    assert table.nbytes == context * 64 * 8
    if context >= 131072:
        # Cover every supported absolute coordinate with an independent 1D
        # program per position, rather than sampling the production 2D layout.
        cos = torch.empty((context, 64), device='cuda', dtype=torch.float32)
        sin = torch.empty_like(cos)
        _single_position_oracle[(context,)](inv, cos, sin, 0, num_warps=4, enable_fp_fusion=False)
        assert torch.equal(cos.view(torch.int32), table.cos.view(torch.int32))
        assert torch.equal(sin.view(torch.int32), table.sin.view(torch.int32))
        return
    for start in sorted({0, min(31, context - 1), min(511, context - 1), context - 1}):
        cos = torch.empty((1, 64), device='cuda', dtype=torch.float32)
        sin = torch.empty_like(cos)
        _single_position_oracle[(1,)](inv, cos, sin, start, num_warps=4, enable_fp_fusion=False)
        assert torch.equal(cos.view(torch.int32), table.cos[start:start + 1].view(torch.int32))
        assert torch.equal(sin.view(torch.int32), table.sin[start:start + 1].view(torch.int32))


@pytest.mark.parametrize('start,rows', [(0, 1), (31, 5), (511, 5), (512, 17), (65535, 5), (131071, 1)])
def test_global_output_exact_against_original_consumer(start, rows):
    torch.manual_seed(812 + start)
    context = start + rows
    inv_freq = frequency()
    table = attention.GlobalTrig(inv_freq, context)
    raw = torch.randn((context, 2, 512), device='cuda', dtype=torch.bfloat16)
    inv = torch.rsqrt(raw.float().square().mean(-1) + 1e-6)
    norm = torch.randn(512, device='cuda', dtype=torch.bfloat16)
    q = torch.randn((rows, 16, 512), device='cuda', dtype=torch.bfloat16)
    pos = torch.tensor([start], device='cuda', dtype=torch.int32)
    kw = dict(chunks=-(-context // 512), round_normalized=False)
    for sequential in (False, True):
        expected = attention.full(q, raw, inv, norm, inv_freq, pos, sequential=sequential, **kw)
        actual = attention.full(q, raw, inv, norm, inv_freq, pos, sequential=sequential, trig=table, **kw)
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))
    with pytest.raises(ValueError, match='frequency/context/arithmetic'):
        attention.full(q, raw, inv, norm, inv_freq.clone(), pos, trig=table, **kw)
    with pytest.raises(ValueError, match='frequency/context/arithmetic'):
        attention.full(q, raw[:-1], inv[:-1], norm, inv_freq, pos, trig=table, **kw)
    with pytest.raises(ValueError, match='frequency/context/arithmetic'):
        attention.full(q, raw, inv, norm, inv_freq, pos, trig=table, **{**kw, 'round_normalized': True})


def test_append_graph_replay_and_rollback_keep_exact_bits():
    torch.manual_seed(451)
    context, rows = 1025, 5
    freq = frequency()
    table = attention.GlobalTrig(freq, context)
    raw = torch.randn((context, 2, 512), device='cuda', dtype=torch.bfloat16)
    inv = torch.rsqrt(raw.float().square().mean(-1) + 1e-6)
    kr = torch.randn((rows, 1024), device='cuda', dtype=torch.bfloat16)
    qr = torch.randn((rows, 8192), device='cuda', dtype=torch.bfloat16)
    norm = torch.randn(512, device='cuda', dtype=torch.bfloat16)
    pos = torch.zeros(1, device='cuda', dtype=torch.int32)
    pointers = table.cos.data_ptr(), table.sin.data_ptr()

    def run(trig):
        q = attention.prep_full(qr, kr, norm, freq, pos, raw, inv, eps=1e-6, round_normalized=False)
        return attention.full(q, raw, inv, norm, freq, pos, chunks=3, round_normalized=False, trig=trig)

    run(table)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = run(table)
    for start in (0, 31, 511, 1020, 512, 0):
        pos.fill_(start)
        # Change both staged inputs before replay, including rollback/overwrite.
        qr.mul_(.5)
        kr.neg_()
        graph.replay()
        expected = run(None)
        assert torch.equal(captured.view(torch.int16), expected.view(torch.int16))
        assert pointers == (table.cos.data_ptr(), table.sin.data_ptr())

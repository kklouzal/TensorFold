"""Root-only actual GPU bit checks; raw global kernel is the unchanged oracle.

This is a private investigation candidate, not selected production arithmetic.
No fitted tolerance: prepared and raw attention outputs must be bit-identical.
"""
import pytest
import torch

from tensorfold.families.gemma4.cuda import attention


@pytest.mark.parametrize('start,rows',
    [(start, rows) for start in (0, 1, 511, 512, 513)
     for rows in (1, 2, 3, 4, 5, 6, 7, 8, 17, 33, 1024)]
    + [(start, rows) for start in (65536, 262111) for rows in (1, 3, 17, 33)])
def test_raw_prepared_global_bits_and_scratch_tail(start, rows):
    torch.manual_seed(173 + start + rows)
    assert start + rows <= 262144
    length, capacity, hk, heads, dim = start + rows, start + rows + 7, 2, 16, 512
    raw = (torch.randn((capacity, hk, dim), device='cuda') * .15).to(torch.bfloat16)
    inv = torch.rsqrt(raw.float().square().mean(-1) + 1e-6)
    norm = (torch.randn(dim, device='cuda') * .1 + 1).to(torch.bfloat16)
    freq = torch.pow(1e6, -torch.arange(0, dim, 2, device='cuda', dtype=torch.float32) / dim)
    freq[64:] = 0
    q = (torch.randn((rows, heads, dim), device='cuda') * .15).to(torch.bfloat16)
    pos = torch.tensor(start, device='cuda', dtype=torch.int32)
    keys = torch.full_like(raw, float('nan'))
    values = torch.full_like(raw, float('nan'))
    chunks = -(-length // attention.GLOBAL_CHUNK)
    reference = attention.full(q, raw, inv, norm, freq, pos, chunks=chunks, round_normalized=False)
    actual = attention.full_prepared(q, raw, inv, norm, freq, pos, chunks=chunks, length=length,
                                    round_normalized=False, scratch=(keys, values))
    assert torch.equal(actual, reference)
    assert torch.isfinite(keys[:length]).all() and torch.isfinite(values[:length]).all()
    assert torch.isnan(keys[length:]).all() and torch.isnan(values[length:]).all()
    # New raw owner/layer data must overwrite the same fixed scratch addresses.
    pointers = keys.data_ptr(), values.data_ptr()
    raw.mul_(.5)
    inv.copy_(torch.rsqrt(raw.float().square().mean(-1) + 1e-6))
    reference2 = attention.full(q, raw, inv, norm, freq, pos, chunks=chunks, round_normalized=False)
    actual2 = attention.full_prepared(q, raw, inv, norm, freq, pos, chunks=chunks, length=length,
                                     round_normalized=False, scratch=(keys, values))
    assert torch.equal(actual2, reference2)
    assert (keys.data_ptr(), values.data_ptr()) == pointers

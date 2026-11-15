"""Exact attention mapping differential and ring-cache known answers.

An explicit arange position map exercises the complete physical-position scan;
it must agree with implicit linear positions. Retain this oracle for traversal
changes without duplicating the attention algorithm. The existing Gemma kernel
suite also retains its FP64 numerical attention oracle.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="needs an SM70 GPU",
)


def _run(ext, q, k, v, positions, base, heads, window, noncausal, splits, dtype):
    rows, dim = q.shape
    slots = k.shape[0]
    chunk = 64 if rows <= 16 else 256
    padded = (rows + 15) // 16 * 16
    po = torch.full((splits, padded, dim), 123.25, device="cuda")
    pm = torch.full((splits, padded), 124.25, device="cuda")
    pl = torch.full_like(pm, 125.25)
    guarded = torch.full((rows + 2, dim), 126.25, dtype=dtype, device="cuda")
    out = guarded[1:-1]
    ext.attn(q, rows, heads, dim, k, v, positions, slots, base, window, int(noncausal),
             dim ** -0.5, slots // chunk, splits, po, pm, pl, padded, out)
    assert torch.all(guarded[0] == 126.25) and torch.all(guarded[-1] == 126.25)
    return out, po, pm, pl


def _exact(a, b):
    assert torch.equal(torch.isnan(a), torch.isnan(b))
    present = ~torch.isnan(a)
    bits = torch.int16 if a.dtype == torch.float16 else torch.int32
    assert torch.equal(a.view(bits)[present], b.view(bits)[present])


@pytest.mark.parametrize("rows,heads,dim,slots,base,window,noncausal,half", [
    (1, 2, 512, 32768, 0, 0, False, False),
    (5, 2, 512, 32768, 63, 0, False, False),
    (8, 2, 512, 32768, 64, 0, False, False),
    (16, 2, 512, 32768, 255, 0, False, False),
    (40, 2, 512, 8192, 1023, 0, False, True),
    (4, 4, 128, 32768, 4095, 1024, False, False),
    (4, 4, 128, 32768, 4095, 1024, True, False),
    (5, 4, 128, 32768, 4095, 1024, False, False),
    (5, 4, 128, 32768, 4095, 1024, True, False),
    (2, 4, 128, 4096, 62, 2, False, False),
    (3, 4, 128, 4096, 64, 2, True, False),
    (16, 4, 128, 32768, 8191, 1024, True, False),
    (16, 4, 128, 4096, 255, 2, False, False),
    (1, 2, 128, 262144, 300, 0, False, False),
    (1, 2, 256, 512, 511, 1024, False, False),
    # No key in the window: all splits remain empty and the original merge's
    # zero-likelihood output is NaN. Do not conceal it with a fabricated value.
    (1, 2, 256, 512, 1024, 2, False, False),
])
def test_linear_attention_matches_explicit_position_scan(rows, heads, dim, slots, base, window, noncausal, half):
    from tensorfold.cuda.sm70.gemma4_native import extension

    torch.manual_seed(20261115)
    q = (torch.randn(rows * heads, dim, device="cuda") * 0.2).half()
    k = (torch.randn(slots, dim, device="cuda") * 0.2).half()
    v = torch.randn(slots, dim, device="cuda").half()
    positions = torch.arange(slots, dtype=torch.int32, device="cuda")
    start = torch.tensor([base], dtype=torch.int32, device="cuda")
    before = (q.clone(), k.clone(), v.clone(), positions.clone(), start.clone())
    chunks = slots // (64 if rows * heads <= 16 else 256)
    dtype = torch.float16 if half else torch.float32
    for splits in sorted({1, min(chunks, 7), min(chunks, 160)}):
        bounded = _run(extension(), q, k, v, None, start, heads, window, noncausal, splits, dtype)
        scan = _run(extension(), q, k, v, positions, start, heads, window, noncausal, splits, dtype)
        for actual, expected in zip(bounded, scan):
            _exact(actual, expected)
    for original, saved in zip((q, k, v, positions, start), before):
        if original.is_floating_point():
            _exact(original, saved)
        else:
            assert torch.equal(original, saved)


@pytest.mark.parametrize("rows,dim", [(1, 256), (5, 128), (16, 512)])
def test_ring_attention_keeps_visible_key_at_unrelated_physical_index(rows, dim):
    from tensorfold.cuda.sm70.gemma4_native import extension

    torch.manual_seed(20261116)
    heads, slots, base, window = 2, 2048, 2, 2
    q = torch.randn(rows * heads, dim, device="cuda").half()
    k = (torch.randn(slots, dim, device="cuda") * 0.2).half()
    v = torch.randn(slots, dim, device="cuda").half()
    positions = torch.full((slots,), -1, dtype=torch.int32, device="cuda")
    query_position = base + rows - 1
    positions[-1] = query_position
    positions[0] = query_position - window  # outside the strict lower boundary
    positions[1] = query_position + 1       # future
    start = torch.tensor([base], dtype=torch.int32, device="cuda")
    chunks = slots // (64 if rows * heads <= 16 else 256)
    for splits in (1, min(7, chunks)):
        out, _, _, _ = _run(extension(), q, k, v, positions, start, heads, window, True, splits, torch.float32)
        _exact(out, v[-1].float().expand_as(out).contiguous())

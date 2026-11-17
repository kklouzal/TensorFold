"""One-element-per-thread family kernels cover every row of a large prompt piece.

normed/mix/gate launch one thread per element without a grid-stride loop; a grid clamped to 65535 blocks of
256 threads left every element past 16,776,960 unwritten (SC117 HC normalization above 1638 prompt rows). Each
operation is row independent, so a 2048-row call must equal the same rows computed in two halves, bitwise.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason='SM70 native family')

ROWS, DIMS, STREAMS = 2048, 2560, 4  # ROWS * STREAMS * DIMS = 20,971,520 > 65535 * 256


@pytest.fixture(scope='module')
def native():
    from tensorfold.cuda.sm70.family_native import extension

    return extension()


def halves(run, rows):
    """run(first, last) -> tensors for those rows; a full call and two half calls."""
    full = run(0, rows)
    split = [torch.cat(parts) for parts in zip(run(0, rows // 2), run(rows // 2, rows))]
    return full, split


@torch.no_grad()
def test_hc_normed_writes_every_row(native):
    torch.manual_seed(0)
    wide = STREAMS * DIMS
    h = torch.randn((ROWS, wide), device='cuda').half()
    pss = torch.rand((ROWS * wide // 256,), device='cuda') * 64 + 1
    scale = torch.rand((wide,), device='cuda') + 0.5

    def run(first, last):
        out = torch.full((last - first, wide), float('nan'), device='cuda').half()
        xs = torch.full(((last - first) * wide // 32,), float('nan'), device='cuda')
        native.normed(h[first:last], pss[first * wide // 256:last * wide // 256], scale, out, xs, STREAMS, 1e-6)
        return out, xs
    (out, xs), (out2, xs2) = halves(run, ROWS)
    assert not out.isnan().any() and not xs.isnan().any()
    assert torch.equal(out, out2) and torch.equal(xs, xs2)


@torch.no_grad()
def test_hc_mix_and_attention_gate_write_every_row(native):
    torch.manual_seed(1)
    rows = 8192  # mix: rows * DIMS > 65535 * 256 past 6553 rows
    up = torch.randn((rows, STREAMS * DIMS), device='cuda').half()
    normed = torch.randn((rows, STREAMS * DIMS), device='cuda').half()

    def mix(first, last):
        out = torch.full((last - first, DIMS), float('nan'), device='cuda').half()
        xs = torch.full(((last - first) * DIMS // 32,), float('nan'), device='cuda')
        native.mix(up[first:last], normed[first:last], out, xs, STREAMS)
        return out, xs
    (out, xs), (out2, xs2) = halves(mix, rows)
    assert not out.isnan().any() and not xs.isnan().any()
    assert torch.equal(out, out2) and torch.equal(xs, xs2)

    heads, dim = 16, 256  # gate: rows * heads * dim > 65535 * 256 past 4095 rows
    value = torch.randn((rows, heads, dim), device='cuda').half()
    projection = torch.randn((rows, 2 * heads * dim), device='cuda').half()

    def gate(first, last):
        out = torch.full((last - first, heads * dim), float('nan'), device='cuda').half()
        xs = torch.full(((last - first) * heads * dim // 32,), float('nan'), device='cuda')
        native.gate(value[first:last], projection[first:last], out, xs, heads, dim)
        return out, xs
    (out, xs), (out2, xs2) = halves(gate, rows)
    assert not out.isnan().any() and not xs.isnan().any()
    assert torch.equal(out, out2) and torch.equal(xs, xs2)

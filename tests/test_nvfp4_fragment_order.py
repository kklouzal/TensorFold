"""FP8 byte layout against independent scalar-address decoding, on CPU and CUDA."""

import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda.nvfp4 import linear


def _source_addresses(n, k, npad):
    """Decode each output lane/byte's original address; zero denotes a padded row.

    The output ABI is [rowtile64, ktile64, rowgroup8, lane32, khalf2,
    byte8]. A lane owns its row modulo8 and two adjacent K inputs. The byte
    selects the 16- and8-input groups and the final input within that pair.
    Distinct one-based source addresses also distinguish padding from row0.
    """
    address = np.arange(npad * k, dtype=np.int64)
    byte, address = address % 8, address // 8
    half, address = address % 2, address // 2
    lane, address = address % 32, address // 32
    rowgroup, address = address % 8, address // 8
    ktile, rowtile = address % max(1, k // 64), address // max(1, k // 64)
    row = rowtile * 64 + rowgroup * 8 + lane // 4
    col = ktile * 64 + half * 32 + (byte // 4) * 16 + ((byte // 2) % 2) * 8 + (lane % 4) * 2 + byte % 2
    return np.where(row < n, row * k + col + 1, 0)


SHAPES = [(0, 64, 0), (0, 64, 128), (17, 0, 128), (1, 64, 128), (63, 128, 128), (64, 64, 128),
          (65, 192, 128), (127, 256, 128), (128, 64, 128), (129, 128, 256), (257, 192, 384),
          (4095, 64, 4096), (4096, 128, 4096), (4097, 64, 4224), (8192, 64, 8192),
          (8256, 128, 8256), (12288, 64, 12288), (8193, 64, 8320)]


def _layout(codes, layout):
    n, k = codes.shape
    if layout == "contiguous":
        return codes
    if layout == "column":
        view = torch.empty((n, k * 2), dtype=torch.uint8, device=codes.device)[:, ::2]
    elif layout == "row":
        view = torch.empty((n * 2, k), dtype=torch.uint8, device=codes.device)[::2]
    elif layout == "transpose":
        view = torch.empty((k, n), dtype=torch.uint8, device=codes.device).t()
    elif layout == "offset":
        view = torch.empty((n + 2, k + 2), dtype=torch.uint8, device=codes.device)[1:n + 1, 1:k + 1]
    else:
        raise ValueError(layout)
    view.copy_(codes)
    return view


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("layout", ["contiguous", "column", "row", "transpose", "offset"])
@pytest.mark.parametrize("n,k,npad", SHAPES)
def test_fragment_bytes_encode_every_original_address(n, k, npad, layout, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    expected = _source_addresses(n, k, npad)
    source = torch.arange(1, n * k + 1, dtype=torch.int64, device=device).reshape(n, k)
    planes = max(1, math.ceil(max(1, (n * k).bit_length()) / 8))
    # Encode distinct source addresses across byte planes. A periodic byte
    # pattern alone could conceal swapped rows or 256-byte source offsets.
    for plane in range(planes):
        codes = _layout(((source >> (plane * 8)) & 255).to(torch.uint8), layout)
        before = codes.clone()
        actual = linear._fragment_order(codes, npad)
        want = torch.from_numpy(((expected >> (plane * 8)) & 255).astype(np.uint8)).to(device)
        assert actual.dtype == torch.uint8 and actual.device == codes.device
        assert actual.is_contiguous() and actual.shape == (npad * k,)
        assert torch.equal(actual, want)
        assert torch.equal(codes, before)
        codes.zero_()
        assert torch.equal(actual, want)  # output owns its bytes after the source changes


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("n,k,npad", [(128, 192, 128), (129, 128, 256), (4097, 64, 4224), (8192, 64, 8192)])
@pytest.mark.parametrize("layout", ["overlap", "broadcast_rows", "broadcast_columns"])
def test_fragment_reads_overlapping_and_broadcast_views_without_writing_them(n, k, npad, layout, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if layout == "overlap":
        values = np.arange(1, n + k, dtype=np.int64)
        logical = np.arange(n)[:, None] + np.arange(k)[None, :] + 1
        strides = (1, 1)
    elif layout == "broadcast_rows":
        values = np.arange(1, k + 1, dtype=np.int64)
        logical = np.broadcast_to(values[None, :], (n, k))
        strides = (0, 1)
    else:
        values = np.arange(1, n + 1, dtype=np.int64)
        logical = np.broadcast_to(values[:, None], (n, k))
        strides = (1, 0)
    addresses = _source_addresses(n, k, npad)
    lookup = np.concatenate((np.zeros(1, dtype=np.int64), logical.reshape(-1)))
    for plane in range(max(1, math.ceil(int(values[-1]).bit_length() / 8))):
        backing = torch.from_numpy(((values >> (plane * 8)) & 255).astype(np.uint8)).to(device)
        codes = backing.as_strided((n, k), strides)
        before = backing.clone()
        want = torch.from_numpy(((lookup[addresses] >> (plane * 8)) & 255).astype(np.uint8)).to(device)
        actual = linear._fragment_order(codes, npad)
        assert torch.equal(actual, want)
        assert torch.equal(backing, before)
        backing.zero_()
        assert torch.equal(actual, want)


@pytest.mark.parametrize("rows", [64, 128, 192, 4096])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_fragment_partial_chunks_preserve_independent_layout(monkeypatch, rows, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    n, k, npad = 450, 192, 512
    expected = _source_addresses(n, k, npad)
    rng = np.random.default_rng(1701)
    values = rng.integers(0, 256, size=n * k + 1, dtype=np.uint8)
    values[0] = 0
    codes = torch.from_numpy(values[1:].reshape(n, k)).to(device)
    want = torch.from_numpy(values[expected]).to(device)
    monkeypatch.setattr(linear, "FRAGMENT_ROWS", rows)
    assert torch.equal(linear._fragment_order(codes, npad), want)

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
          (4095, 64, 4096), (4096, 128, 4096), (4097, 64, 4224), (8193, 64, 8320)]


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("n,k,npad", SHAPES)
def test_fragment_bytes_encode_every_original_address(n, k, npad, strided, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    expected = _source_addresses(n, k, npad)
    source = torch.arange(1, n * k + 1, dtype=torch.int64, device=device).reshape(n, k)
    planes = max(1, math.ceil(max(1, (n * k).bit_length()) / 8))
    # Encode distinct source addresses across byte planes. A periodic byte
    # pattern alone could conceal swapped rows or 256-byte source offsets.
    for plane in range(planes):
        codes = ((source >> (plane * 8)) & 255).to(torch.uint8)
        if strided:
            backing = torch.empty((n, k * 2), dtype=torch.uint8, device=device)
            backing[:, ::2] = codes
            codes = backing[:, ::2]
        before = codes.clone()
        actual = linear._fragment_order(codes, npad)
        want = torch.from_numpy(((expected >> (plane * 8)) & 255).astype(np.uint8)).to(device)
        assert actual.dtype == torch.uint8 and actual.device == codes.device
        assert actual.is_contiguous() and actual.shape == (npad * k,)
        assert torch.equal(actual, want)
        assert torch.equal(codes, before)
        codes.zero_()
        assert torch.equal(actual, want)  # output owns its bytes after the source changes


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

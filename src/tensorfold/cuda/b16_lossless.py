"""Private, unconnected BF16 weight codec with exact reconstructed operands.

Each64-value block retains96 payload bytes plus4 metadata bytes. A block whose
exponent span exceeds15 additionally owns128 raw bytes; its unused96-byte slots
remain charged. This codec changes no BF16 bits, including signed zero/NaN bits.
The caller exclusively owns tensors and must not mutate them until all GPU readers
retire. Frozen fields do not make PyTorch storage immutable. No serialization API.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

_BLOCK = 64
_COMPRESSED = (1 << 24) - 1
_MAX_DIM = (1 << 31) - 257


@dataclass(frozen=True)
class PackedBF16:
    low: torch.Tensor
    delta: torch.Tensor
    metadata: torch.Tensor
    raw: torch.Tensor
    shape: tuple[int, int]

    @property
    def nbytes(self):
        return sum(x.numel() * x.element_size() for x in (self.low, self.delta, self.metadata, self.raw))

    def to(self, device):
        """Synchronous owned transfer; source owners remain valid for the copy."""
        return PackedBF16(*(x.to(device=device) for x in (self.low, self.delta, self.metadata, self.raw)), self.shape)


def _blocks(weight):
    words = weight.view(torch.int16).reshape(-1, _BLOCK).to(torch.int32) & 65535
    exponent = (words >> 7) & 255
    maximum = exponent.amax(dim=1)
    escapes = maximum - exponent.amin(dim=1) > 15
    return words, exponent, maximum, escapes


def pack(weight: torch.Tensor, *, chunk_rows=32) -> PackedBF16:
    """Two bounded CPU-row passes; preserves original CPU weight/source lifetime.

    Temporary work is bounded by chunk_rows*K. Persistent compact raw pool is
    counted before allocation. Preparation does not retain the source or a full
    decoded mirror; caller decides when its original source may retire.
    """
    if (weight.device.type != 'cpu' or weight.dtype != torch.bfloat16 or weight.ndim != 2
            or not weight.is_contiguous()):
        raise ValueError('pack requires an owned contiguous CPU BF16 [N,K] tensor')
    n, k = weight.shape
    if not (0 < n <= _MAX_DIM and 0 < k <= _MAX_DIM and k % _BLOCK == 0):
        raise ValueError('positive signed32 dimensions and K multiple64 required')
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('positive integer chunk_rows required')
    blocks = n * (k // _BLOCK)
    if blocks > (1 << 63) // 128:
        raise ValueError('encoded byte product exceeds signed64')
    raw_count = 0
    for start in range(0, n, chunk_rows):
        raw_count += int(_blocks(weight[start:start + chunk_rows])[3].sum())
    if raw_count > _COMPRESSED:
        raise ValueError('raw escape count exceeds24-bit block index capacity')
    low = torch.empty((blocks, 64), dtype=torch.uint8)
    delta = torch.empty((blocks, 32), dtype=torch.uint8)
    metadata = torch.empty(blocks, dtype=torch.int32)
    raw = torch.empty((raw_count, 64), dtype=torch.int16)
    cursor = 0
    for start in range(0, n, chunk_rows):
        words, exponents, maximum, escapes = _blocks(weight[start:start + chunk_rows])
        count = len(words)
        begin = start * (k // 64)
        sign_mantissa = ((words >> 8) & 128) | (words & 127)
        differences = torch.where(escapes[:, None], 0, maximum[:, None] - exponents)
        sign_mantissa = torch.where(escapes[:, None], 0, sign_mantissa)
        low[begin:begin + count].copy_(sign_mantissa.to(torch.uint8))
        delta[begin:begin + count].copy_((differences[:, 0::2] | (differences[:, 1::2] << 4)).to(torch.uint8))
        indices = torch.full((count,), _COMPRESSED, dtype=torch.int64)
        raw_here = int(escapes.sum())
        if raw_here:
            indices[escapes] = torch.arange(cursor, cursor + raw_here)
            raw[cursor:cursor + raw_here].copy_(words[escapes].to(torch.int16))
        metadata[begin:begin + count].copy_(((indices << 8) | maximum).to(torch.int32))
        cursor += raw_here
    if cursor != raw_count:
        raise RuntimeError('source changed during two-pass encoding')
    return PackedBF16(low, delta, metadata, raw, (n, k))


def unpack_cpu(weight: PackedBF16, *, chunk_blocks=4096):
    """Independent CPU bit decoder, used as a codec oracle before GPU execution."""
    if type(chunk_blocks) is not int or chunk_blocks <= 0:
        raise ValueError('positive integer chunk_blocks required')
    _validate(weight, cuda=False)
    out = torch.empty(weight.shape, dtype=torch.int16)
    flat = out.reshape(-1, 64)
    for start in range(0, len(weight.metadata), chunk_blocks):
        meta = weight.metadata[start:start + chunk_blocks].to(torch.int64) & 0xffffffff
        indices = meta >> 8
        lows = weight.low[start:start + chunk_blocks].to(torch.int64)
        packed = weight.delta[start:start + chunk_blocks].to(torch.int64)
        differences = torch.stack((packed & 15, packed >> 4), dim=-1).flatten(1)
        exponent = (meta & 255)[:, None] - differences
        compressed = indices == _COMPRESSED
        if bool(((exponent < 0) & compressed[:, None]).any()):
            raise ValueError('invalid compressed exponent delta')
        decoded = ((lows & 128) << 8) | (exponent << 7) | (lows & 127)
        if bool((~compressed).any()):
            selected = indices[~compressed]
            if int(selected.max()) >= len(weight.raw):
                raise ValueError('raw escape index outside owned pool')
            decoded[~compressed] = weight.raw.index_select(0, selected).to(torch.int64) & 65535
        flat[start:start + len(meta)].copy_(decoded.to(torch.int16))
    return out.view(torch.bfloat16)


def _validate(weight, *, cuda):
    if not isinstance(weight, PackedBF16) or type(weight.shape) is not tuple or len(weight.shape) != 2:
        raise ValueError('owned PackedBF16 required')
    n, k = weight.shape
    if not (type(n) is int and type(k) is int and 0 < n <= _MAX_DIM and 0 < k <= _MAX_DIM and k % 64 == 0):
        raise ValueError('invalid encoded geometry')
    blocks = n * (k // 64)
    tensors = (weight.low, weight.delta, weight.metadata, weight.raw)
    device = tensors[0].device
    if cuda != (device.type == 'cuda') or any(x.device != device or not x.is_contiguous() for x in tensors):
        raise ValueError('encoded tensors require contiguous storage on the same expected device')
    if (weight.low.dtype != torch.uint8 or weight.low.shape != (blocks, 64)
            or weight.delta.dtype != torch.uint8 or weight.delta.shape != (blocks, 32)
            or weight.metadata.dtype != torch.int32 or weight.metadata.shape != (blocks,)
            or weight.raw.dtype != torch.int16 or weight.raw.ndim != 2 or weight.raw.shape[1] != 64
            or len(weight.raw) > _COMPRESSED):
        raise ValueError('encoded tensor schema differs')


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load
    root = Path(__file__).parent
    return load(name='tensorfold_b16_lossless_v1', sources=[str(root / 'b16_lossless.cpp'), str(root / 'b16_lossless.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)


def unpack(weight: PackedBF16):
    """GPU diagnostic full reconstruction; not retained by fused projection."""
    _validate(weight, cuda=True)
    return _ext().unpack(weight.low, weight.delta, weight.metadata, weight.raw, *weight.shape)


def matmul_rows(x, weight: PackedBF16, rows=8):
    """Fused lossless operand reconstruction + existing b16 FP32 FMA/XOR order.

    No whole-weight decode owner. Explicit rows1/2/4/8 selects reuse only, retaining
    each row's reduction/rounding. Comparison with native BLAS remains separate.
    """
    _validate(weight, cuda=True)
    if type(rows) is not int or rows not in (1, 2, 4, 8):
        raise ValueError('reuse rows must be1/2/4/8')
    return _ext().matmul_rows(x, weight.low, weight.delta, weight.metadata, weight.raw, *weight.shape, rows)

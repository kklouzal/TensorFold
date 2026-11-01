"""Unconnected Sym8 v1: exact BF16 words, direct GEMV and row lookup.

Palette8*u16 + packed32*4bit indices + metadatau32 =36 bytes/group;
ineligible groups additionally retain all32 raw words (64 bytes). Every base
slot remains allocated on escape. Fields are borrowed immutable storage:
callers serialize pack/transfer/readers and retain owners until CUDA retirement.
If captured, packed fields and static input/output owners outlive the complete
CUDA graph and every future replay; capture/replay qualification is pending.
No loader, serving, precision mode, full-GPU decode or speed claim is provided.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import torch

SCHEMA = 'tensorfold.bf16.sym8.v1'
SENTINEL = 0xffffff
MAX_DIM = (1 << 31) - 257
MAX_I64 = (1 << 63) - 1


@dataclass(frozen=True, init=False)
class PackedSym8:
    palette: torch.Tensor
    codes: torch.Tensor
    metadata: torch.Tensor
    raw: torch.Tensor
    shape: tuple[int, int]
    schema: str = SCHEMA

    def __init__(self, palette, codes, metadata, raw, shape, schema=SCHEMA):
        # Descriptor publication is fallible too. Clear both partially stored
        # fields and our borrowed formals if construction cannot complete.
        try:
            object.__setattr__(self, 'palette', palette)
            object.__setattr__(self, 'codes', codes)
            object.__setattr__(self, 'metadata', metadata)
            object.__setattr__(self, 'raw', raw)
            object.__setattr__(self, 'shape', shape)
            object.__setattr__(self, 'schema', schema)
        except BaseException as primary:
            _native_cause = BaseException.__cause__.__get__(primary)
            _native_context = BaseException.__context__.__get__(primary)
            try:
                dict.clear(object.__getattribute__(self, '__dict__'))
            except BaseException as cleanup:
                # A failed release remains explicit and trace-owned; never
                # replace the original failure while preparing diagnostics.
                try:
                    from tensorfold.cleanup import raise_failures
                    raise_failures(primary, [cleanup])
                except BaseException as transport:
                    if transport is not primary:
                        raise primary from transport
                    raise
            raise
        finally:
            palette = codes = metadata = raw = None

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.palette, self.codes, self.metadata, self.raw))


@dataclass
class TransferStatus:
    """Preallocate before CUDA. Unsafe outcome is process-terminal, no retry.

    Strong owners survive failed exception annotation/return publication. A
    caller must retain this status/primary until process teardown on unsafe
    failure. Successful result owns GPU fields; CPU sources are then released.
    """
    started: bool = False
    safe_to_release: bool = True
    transferred: bool = False
    source: PackedSym8 | None = None
    fields: list = field(default_factory=lambda: [None] * 4)
    result: PackedSym8 | None = None


def _geometry(shape):
    if (type(shape) is not tuple or len(shape) != 2 or any(type(v) is not int for v in shape)
            or not 0 < shape[0] <= MAX_DIM or not 0 < shape[1] <= MAX_DIM or shape[1] % 32):
        raise ValueError('positive signed32 dimensions and K multiple32 required')
    n, k = shape
    if n * k > MAX_I64 // 2:
        raise ValueError('original BF16 byte count exceeds signed64')
    return n * (k // 32)



def _allocation_bytes(groups, raw_count):
    if (type(groups) is not int or groups <= 0 or type(raw_count) is not int
            or not 0 <= raw_count <= min(groups, SENTINEL)):
        raise ValueError('valid group count and24-bit raw pool required')
    size = 36 * groups + 64 * raw_count
    if size > MAX_I64:
        raise ValueError('encoded byte count exceeds signed64')
    return size


def _validate(packed, *, cuda):
    if type(packed) is not PackedSym8 or packed.schema != SCHEMA:
        raise ValueError('exact owned Sym8 v1 descriptor required')
    groups = _geometry(packed.shape)
    fields = packed.palette, packed.codes, packed.metadata, packed.raw
    if any(type(t) is not torch.Tensor for t in fields):
        raise ValueError('four real tensor fields required')
    dev = fields[0].device
    if (dev.type != ('cuda' if cuda else 'cpu') or any(t.device != dev or not t.is_contiguous() for t in fields)
            or packed.palette.dtype != torch.int16 or packed.palette.shape != (groups, 8)
            or packed.codes.dtype != torch.uint8 or packed.codes.shape != (groups, 16)
            or packed.metadata.dtype != torch.int32 or packed.metadata.shape != (groups,)
            or packed.raw.dtype != torch.int16 or packed.raw.ndim != 2 or packed.raw.shape[1] != 32
            or not 0 <= len(packed.raw) <= SENTINEL):
        raise ValueError('Sym8 fields disagree with contiguous same-device storage schema')
    _allocation_bytes(groups, len(packed.raw))


def _plan(words):
    try:
        import numpy as np
        mag = words & 0x7fff
        ordered = np.sort(words, axis=1)
        wc = 1 + np.count_nonzero(ordered[:, 1:] != ordered[:, :-1], axis=1)
        sorted_mag = np.sort(mag, axis=1)
        unique = np.ones(sorted_mag.shape, dtype=np.bool_)
        unique[:, 1:] = sorted_mag[:, 1:] != sorted_mag[:, :-1]
        unique &= sorted_mag != 0
        count = np.count_nonzero(unique, axis=1)
        positive_zero = (words == 0).any(axis=1)
        negative_zero = (words == 0x8000).any(axis=1)
        valid = (wc <= 16) & (count <= 8) & ~negative_zero
        return mag, sorted_mag, unique, count, positive_zero, valid
    finally:
        words = None


def _encode(words):
    try:
        import numpy as np
        mag, sorted_mag, unique, count, zero, valid = _plan(words)
        groups = words.shape[0]
        palette = np.zeros((groups, 8), dtype=np.uint16)
        rank = np.cumsum(unique, axis=1) - 1
        rr, cc = np.nonzero(unique & (rank < 8))
        palette[rr, rank[rr, cc]] = sorted_mag[rr, cc]
        extra_sign = np.zeros(groups, dtype=np.bool_)
        need_last = valid & zero & (count == 8)
        if np.any(need_last):
            single = np.zeros((groups, 8), dtype=np.bool_)
            negative = np.zeros((groups, 8), dtype=np.bool_)
            for slot in range(8):
                value = palette[:, slot, None]
                pos = (words == value).any(axis=1) & (value[:, 0] != 0)
                neg = (words == (value | 0x8000)).any(axis=1) & (value[:, 0] != 0)
                single[:, slot] = pos ^ neg
                negative[:, slot] = neg
            chosen = np.where(single, np.arange(8), -1).max(axis=1)
            if np.any(need_last & (chosen < 0)):
                raise RuntimeError('eligible zero+8 palette has no single-sign magnitude')
            rows = np.nonzero(need_last)[0]
            selected = chosen[rows]
            held = palette[rows, 7].copy()
            palette[rows, 7] = palette[rows, selected]
            palette[rows, selected] = held
            extra_sign[rows] = negative[rows, selected]
        code = np.zeros(words.shape, dtype=np.uint8)
        sign = ((words >> 15) * 8).astype(np.uint8)
        for slot in range(7):
            match = (mag == palette[:, slot, None]) & (mag != 0)
            code[match] = np.broadcast_to((slot + 1) + sign, words.shape)[match]
        match = (mag == palette[:, 7, None]) & (mag != 0)
        last_code = np.where(zero[:, None], 8, sign).astype(np.uint8)
        code[match] = np.broadcast_to(last_code, words.shape)[match]
        palette[~valid] = 0
        code[~valid] = 0
        flags = zero.astype(np.uint32) | (extra_sign.astype(np.uint32) << 1) | (count.astype(np.uint32) << 2)
        flags[~valid] = 0
        return valid, palette, code[:, 0::2] | (code[:, 1::2] << 4), flags
    finally:
        words = None


def pack(weight, *, chunk_groups=4096):
    """Two bounded CPU passes. Caller holds the original immutable throughout.

    Temporaries are bounded by chunk_groups*32 words, independent of full N/K.
    All original bits (including NaNs and signed zeros) survive via code or raw.
    """
    words = palette = codes = metadata = raw = None
    po = co = mo = ro = original = valid = pal = code = flags = indices = None
    try:
        import numpy as np
        if (not isinstance(weight, torch.Tensor) or weight.device.type != 'cpu'
                or weight.dtype != torch.bfloat16 or weight.ndim != 2 or not weight.is_contiguous()):
            raise ValueError('original contiguous CPU BF16[N,K] required')
        groups = _geometry(tuple(weight.shape))
        if type(chunk_groups) is not int or chunk_groups <= 0:
            raise ValueError('positive integer chunk_groups required')
        words = weight.detach().view(torch.int16).numpy().view(np.uint16).reshape(-1, 32)
        raw_count = 0
        for start in range(0, groups, chunk_groups):
            raw_count += int(np.count_nonzero(~_plan(words[start:start + chunk_groups])[-1]))
        _allocation_bytes(groups, raw_count)
        palette = torch.empty((groups, 8), dtype=torch.int16, device='cpu')
        codes = torch.empty((groups, 16), dtype=torch.uint8, device='cpu')
        metadata = torch.empty(groups, dtype=torch.int32, device='cpu')
        raw = torch.empty((raw_count, 32), dtype=torch.int16, device='cpu')
        po = palette.numpy().view(np.uint16)
        co, mo, ro = codes.numpy(), metadata.numpy().view(np.uint32), raw.numpy().view(np.uint16)
        cursor = 0
        for start in range(0, groups, chunk_groups):
            original = words[start:start + chunk_groups]
            valid, pal, code, flags = _encode(original)
            count = len(original)
            indices = np.full(count, SENTINEL, dtype=np.uint32)
            escaped = int(np.count_nonzero(~valid))
            if cursor + escaped > raw_count:
                raise RuntimeError('source changed between encoding passes')
            if escaped:
                indices[~valid] = np.arange(cursor, cursor + escaped, dtype=np.uint32)
                ro[cursor:cursor + escaped] = original[~valid]
            po[start:start + count], co[start:start + count] = pal, code
            mo[start:start + count] = (indices << 8) | flags
            cursor += escaped
        if cursor != raw_count:
            raise RuntimeError('source changed between encoding passes')
        return PackedSym8(palette, codes, metadata, raw, tuple(weight.shape))
    finally:
        weight = words = palette = codes = metadata = raw = None
        po = co = mo = ro = original = valid = pal = code = flags = indices = None


def decode_cpu_chunks(packed, *, chunk_groups=4096):
    """Independent bit decoder yielding (flat word offset, owned BF16 chunk).

    Explicit oracle only. No full decoded owner is retained by the descriptor.
    Close the iterator if stopping early; its input borrow lasts until close.
    """
    meta = index = flags = palette = held = code = out = None
    try:
        import numpy as np
        _validate(packed, cuda=False)
        if type(chunk_groups) is not int or chunk_groups <= 0:
            raise ValueError('positive integer chunk_groups required')
        groups = _geometry(packed.shape)
        for start in range(0, groups, chunk_groups):
            stop = min(groups, start + chunk_groups)
            meta = packed.metadata[start:stop].numpy().view(np.uint32)
            index, flags = meta >> 8, meta & 255
            palette = packed.palette[start:stop].numpy().view(np.uint16)
            held = packed.codes[start:stop].numpy()
            code = np.stack((held & 15, held >> 4), axis=-1).reshape(-1, 32)
            out = np.empty(code.shape, dtype=np.uint16)
            raw_rows = index != SENTINEL
            if np.any(raw_rows):
                if np.any(flags[raw_rows] != 0) or np.any(index[raw_rows] >= len(packed.raw)):
                    raise ValueError('raw index/metadata outside owned pool')
                out[raw_rows] = packed.raw.numpy().view(np.uint16)[index[raw_rows]]
            compressed = ~raw_rows
            if np.any(compressed):
                # Decode tags/symbols by direct bit fields, independent of the
                # encoder's sorted-magnitude membership/sign-selection algorithm.
                value = flags[compressed]
                zero, extra, count = (value & 1) != 0, (value & 2) != 0, value >> 2
                if np.any((value & 0xc0) != 0) or np.any(count > 8) or np.any((~zero) & (count == 0)):
                    raise ValueError('invalid compressed metadata flags')
                if np.any(extra & ((~zero) | (count != 8))):
                    raise ValueError('invalid compressed extra-sign flag')
                symbols = code[compressed]
                zero_word = zero[:, None] & (symbols == 0)
                special = (symbols == 0) | (symbols == 8)
                slot = np.where(special, 7, (symbols & 7).astype(np.int32) - 1)
                if np.any(special & ~zero_word & (count[:, None] != 8)):
                    raise ValueError('unused last-magnitude code')
                if np.any((~special) & (slot >= np.minimum(count, 7)[:, None])):
                    raise ValueError('code addresses unused palette slot')
                magnitude = np.take_along_axis(palette[compressed], slot, axis=1)
                if np.any((~zero_word) & ((magnitude == 0) | (magnitude >= 0x8000))):
                    raise ValueError('palette entry is not nonzero absolute BF16 bits')
                sign = np.where(special & zero[:, None], extra[:, None], symbols >= 8)
                out[compressed] = np.where(zero_word, 0, magnitude | (sign.astype(np.uint16) << 15))
            yield start * 32, torch.from_numpy(out.view(np.int16)).view(torch.bfloat16).reshape(-1)
    finally:
        packed = meta = index = flags = palette = held = code = out = None


def to_cuda(packed, *, device='cuda', status):
    """Synchronous upload. Any CUDA failure is terminal; never retries a drain."""
    _validate(packed, cuda=False)
    if (type(status) is not TransferStatus or status.started or not status.safe_to_release
            or status.transferred or status.source is not None or status.result is not None
            or type(status.fields) is not list or len(status.fields) != 4 or any(t is not None for t in status.fields)):
        raise ValueError('fresh preallocated TransferStatus required')
    dev = torch.device(device)
    if dev.type != 'cuda':
        raise ValueError('to_cuda requires a CUDA device')
    status.source = packed
    status.safe_to_release = False
    status.started = True
    try:
        for slot, tensor in enumerate((packed.palette, packed.codes, packed.metadata, packed.raw)):
            status.fields[slot] = tensor.to(device=dev, non_blocking=True)
        torch.cuda.synchronize(dev)  # one explicit success fence; none after any failed copy
        status.safe_to_release = True
        status.result = PackedSym8(*status.fields, packed.shape)
        status.transferred = True
        status.source = None
        for slot in range(4):
            status.fields[slot] = None
        return status.result
    except BaseException as primary:
        # Preallocated status retains all acquired owners even if transport
        # annotation/preparation cannot allocate. No further CUDA or cleanup.
        _native_cause = BaseException.__cause__.__get__(primary)
        _native_context = BaseException.__context__.__get__(primary)
        raise


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load
    root = Path(__file__).parent
    return load(name='tensorfold_b16_sym8_v1', sources=[str(root / 'b16_sym8.cpp'), str(root / 'b16_sym8.cu')],
                extra_cuda_cflags=['-O3', '-DNDEBUG'], verbose=False)


def matmul(x, packed):
    """Direct BF16 GEMV/smallM<=8 with original b16 FP32 FMA/XOR order."""
    _validate(packed, cuda=True)
    if (not isinstance(x, torch.Tensor) or x.dtype != torch.bfloat16 or x.ndim != 2
            or x.device != packed.palette.device or not x.is_contiguous()
            or not 1 <= x.shape[0] <= 8 or x.shape[1] != packed.shape[1]):
        raise ValueError('contiguous same-device BF16[M1..8,K] required')
    return _ext().matmul(x, packed.palette, packed.codes, packed.metadata, packed.raw, *packed.shape)


def lookup(ids, packed, *, max_rows):
    """Exact raw BF16 rows; arbitrary ID shape including empty/scalar.

    CPU metadata is validated before allocation. A device out-of-domain ID is
    a required fatal device assertion; caller must terminate, not reuse CUDA.
    No host .item()/synchronization or clamp/masked fallback conceals bad IDs.
    """
    _validate(packed, cuda=True)
    if (type(max_rows) is not int or not 0 <= max_rows <= MAX_I64 // packed.shape[1] // 2
            or not isinstance(ids, torch.Tensor)
            or ids.dtype not in (torch.int32, torch.int64) or ids.device != packed.palette.device
            or not ids.is_contiguous() or ids.numel() > max_rows):
        raise ValueError('bounded same-device contiguous int32/int64 IDs required')
    return _ext().lookup(ids, packed.palette, packed.codes, packed.metadata, packed.raw, *packed.shape)

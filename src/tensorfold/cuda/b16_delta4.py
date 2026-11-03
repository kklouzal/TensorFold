"""Owned experimental Delta4 BF16 storage, CPU encoder and transfer authority.

Only the required CPU format logic is derived from the private b16_lossless
prototype (63e8cf7e); that protected source remains an independent historical
oracle, never an imported runtime dependency. One64-word block stores64 sign/
mantissa bytes,32 exponent-delta bytes and one32-bit metadata word. A span>15
additionally owns64 raw16-bit words; base slots remain charged on escape.

Fields preserve every original BF16 bit. They are borrowed immutable storage:
the exclusive owner must retain inputs/fields through queued GPU consumers and
graph retirement. Frozen descriptors do not make Tensor storage immutable.
This has no serialization/cache format API or default production selection.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

SCHEMA = 'tensorfold.bf16.delta4.v1'
SENTINEL = (1 << 24) - 1
MAX_DIM = (1 << 31) - 257
MAX_I64 = (1 << 63) - 1
PACK_CHUNK_ROWS = 512


def geometry(shape):
    if (type(shape) is not tuple or len(shape) != 2 or any(type(value) is not int for value in shape)
            or not 0 < shape[0] <= MAX_DIM or not 0 < shape[1] <= MAX_DIM or shape[1] % 64):
        raise ValueError('positive signed32 dimensions and K multiple64 required')
    if shape[0] > MAX_I64 // shape[1] // 2:
        raise ValueError('original BF16 byte product exceeds signed64')
    return shape[0] * (shape[1] // 64)


def allocation_bytes(groups, raw_count):
    if (type(groups) is not int or groups <= 0 or type(raw_count) is not int
            or not 0 <= raw_count <= min(groups, SENTINEL)):
        raise ValueError('valid Delta4 group count and24-bit raw pool required')
    size = 100 * groups + 128 * raw_count
    if size > MAX_I64:
        raise ValueError('encoded byte product exceeds signed64')
    return size


def header_bounds(shape, *, chunk_rows=PACK_CHUNK_ROWS):
    groups = geometry(shape)
    if type(chunk_rows) is not int or chunk_rows <= 0:
        raise ValueError('positive CPU pack chunk_rows required')
    words = min(shape[0], chunk_rows) * shape[1]
    chunk_groups = words // 64
    # Sum named Tensor outputs/expression temporaries even across nonoverlap;
    # no allocator/runtime or released-buffer reuse credit is assumed.
    inventory = {
        'word_cast_and_mask_i32': 8 * words,
        'exponent_shift_and_mask_i32': 8 * words,
        'maximum_minimum_span_i32': 12 * chunk_groups,
        'escape_mask_bool': chunk_groups,
        'sign_shift_masks_and_or_i32': 16 * words,
        'exponent_subtract_and_where_i32': 8 * words,
        'escape_zeroed_sign_i32': 4 * words,
        'low_cast_u8': words,
        'delta_shift_or_i32_and_cast_u8': 4 * words + words // 2,
        'selected_escape_words_i32_and_i16': 6 * words,
        'indices_arange_and_mask_coordinates_i64': 24 * chunk_groups,
        'metadata_shift_or_i64_and_cast_i32': 20 * chunk_groups,
        'previous_encode_chunk_owners': 16 * words + 13 * chunk_groups,
    }
    return {'schema': SCHEMA, 'groups': groups, 'raw_source_bytes': shape[0] * shape[1] * 2,
            'max_valid_encoded_bytes': allocation_bytes(groups, min(groups, SENTINEL)),
            'pack_chunk_rows': min(shape[0], chunk_rows), 'pack_temporary_inventory': inventory,
            'pack_temporary_bytes': sum(inventory.values()),
            'scope': 'logical Tensor owners; reader/pageable-driver/allocator/runtime overhead admitted separately'}


@dataclass(frozen=True, init=False)
class PackedDelta4:
    low: torch.Tensor
    delta: torch.Tensor
    metadata: torch.Tensor
    raw: torch.Tensor
    shape: tuple[int, int]
    schema: str = SCHEMA

    def __init__(self, low, delta, metadata, raw, shape, schema=SCHEMA):
        try:
            object.__setattr__(self, 'low', low)
            object.__setattr__(self, 'delta', delta)
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
                try:
                    from tensorfold.cleanup import raise_failures
                    raise_failures(primary, [cleanup])
                except BaseException as transport:
                    if transport is not primary:
                        raise primary from transport
                    raise
            raise
        finally:
            low = delta = metadata = raw = None

    @property
    def nbytes(self):
        return sum(value.numel() * value.element_size() for value in (self.low, self.delta, self.metadata, self.raw))


def _validate(packed, *, cuda):
    if type(packed) is not PackedDelta4 or packed.schema != SCHEMA:
        raise ValueError('exact owned Delta4 descriptor required')
    groups = geometry(packed.shape)
    fields = packed.low, packed.delta, packed.metadata, packed.raw
    if any(type(value) is not torch.Tensor for value in fields):
        raise ValueError('four real Tensor fields required')
    device = fields[0].device
    if (device.type != ('cuda' if cuda else 'cpu')
            or any(value.device != device or not value.is_contiguous() for value in fields)
            or packed.low.dtype != torch.uint8 or packed.low.shape != (groups, 64)
            or packed.delta.dtype != torch.uint8 or packed.delta.shape != (groups, 32)
            or packed.metadata.dtype != torch.int32 or packed.metadata.shape != (groups,)
            or packed.raw.dtype != torch.int16 or packed.raw.ndim != 2 or packed.raw.shape[1] != 64):
        raise ValueError('Delta4 fields require the exact contiguous same-device schema')
    allocation_bytes(groups, len(packed.raw))


def _blocks(weight):
    words = exponent = maximum = escapes = None
    try:
        words = weight.view(torch.int16).reshape(-1, 64).to(torch.int32) & 65535
        exponent = (words >> 7) & 255
        maximum = exponent.amax(dim=1)
        escapes = maximum - exponent.amin(dim=1) > 15
        return words, exponent, maximum, escapes
    finally:
        # CPU work is synchronous; returned tuple owns successful outputs.
        weight = words = exponent = maximum = escapes = None


def pack(weight, *, chunk_rows=PACK_CHUNK_ROWS):
    """Two bounded CPU passes over one exclusively owned immutable BF16 table."""
    low = delta = metadata = raw = words = exponent = maximum = escapes = None
    sign_mantissa = differences = indices = None
    try:
        if (type(weight) is not torch.Tensor or weight.device.type != 'cpu' or weight.dtype != torch.bfloat16
                or weight.ndim != 2 or not weight.is_contiguous()):
            raise ValueError('pack requires original contiguous CPU BF16[N,K]')
        shape = tuple(weight.shape)
        groups = geometry(shape)
        if type(chunk_rows) is not int or chunk_rows <= 0:
            raise ValueError('positive integer chunk_rows required')
        n, k = shape
        raw_count = 0
        for start in range(0, n, chunk_rows):
            raw_count += int(_blocks(weight[start:start + chunk_rows])[3].sum())
        allocation_bytes(groups, raw_count)
        low = torch.empty((groups, 64), dtype=torch.uint8, device='cpu')
        delta = torch.empty((groups, 32), dtype=torch.uint8, device='cpu')
        metadata = torch.empty(groups, dtype=torch.int32, device='cpu')
        raw = torch.empty((raw_count, 64), dtype=torch.int16, device='cpu')
        cursor = 0
        for start in range(0, n, chunk_rows):
            words, exponent, maximum, escapes = _blocks(weight[start:start + chunk_rows])
            count = len(words)
            begin = start * (k // 64)
            sign_mantissa = ((words >> 8) & 128) | (words & 127)
            differences = torch.where(escapes[:, None], 0, maximum[:, None] - exponent)
            sign_mantissa = torch.where(escapes[:, None], 0, sign_mantissa)
            low[begin:begin + count].copy_(sign_mantissa.to(torch.uint8))
            delta[begin:begin + count].copy_((differences[:, 0::2] | (differences[:, 1::2] << 4)).to(torch.uint8))
            indices = torch.full((count,), SENTINEL, dtype=torch.int64, device='cpu')
            escaped = int(escapes.sum())
            if cursor + escaped > raw_count:
                raise RuntimeError('source changed between Delta4 encoding passes')
            if escaped:
                indices[escapes] = torch.arange(cursor, cursor + escaped, device='cpu')
                raw[cursor:cursor + escaped].copy_(words[escapes].to(torch.int16))
            metadata[begin:begin + count].copy_(((indices << 8) | maximum).to(torch.int32))
            cursor += escaped
        if cursor != raw_count:
            raise RuntimeError('source changed between Delta4 encoding passes')
        return PackedDelta4(low, delta, metadata, raw, shape)
    finally:
        # CPU-only failures need no borrowed full source or last-chunk owners
        # retained in their native traceback. GPU transfer has a separate,
        # caller-owned status and is deliberately not cleaned up here.
        weight = low = delta = metadata = raw = words = exponent = maximum = escapes = None
        sign_mantissa = differences = indices = None


@dataclass
class TransferStatus:
    """Caller preallocates and retains this owner through terminal failure."""
    started: bool = False
    safe_to_release: bool = True
    transferred: bool = False
    source: PackedDelta4 | None = None
    fields: list = field(default_factory=lambda: [None] * 4)
    result: PackedDelta4 | None = None


def to_cuda(packed, device='cuda', *, status):
    """One owned transfer; failed copy/fence never retries or releases owners."""
    if (type(status) is not TransferStatus or status.started or status.transferred
            or not status.safe_to_release or status.source is not None or status.result is not None
            or type(status.fields) is not list or len(status.fields) != 4
            or any(value is not None for value in status.fields)):
        raise ValueError('fresh caller-owned Delta4 transfer status required')
    _validate(packed, cuda=False)
    device = torch.device(device)
    if device.type != 'cuda':
        raise ValueError('Delta4 transfer destination must be CUDA')
    status.source = packed
    status.started = True
    status.safe_to_release = False
    for index, source in enumerate((packed.low, packed.delta, packed.metadata, packed.raw)):
        status.fields[index] = source.to(device=device)
    torch.cuda.synchronize(device)
    status.safe_to_release = True
    status.result = PackedDelta4(*status.fields, packed.shape)
    status.transferred = True
    status.source = None
    return status.result

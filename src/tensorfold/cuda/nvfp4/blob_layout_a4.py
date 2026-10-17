"""Private single-blob candidate geometry, with unchanged hot-cell byte budget."""

from dataclasses import dataclass
from math import prod

from .host_layout import authority_fields_a4


@dataclass(frozen=True)
class BlobField:
    shape: tuple[int, ...]
    dtype: str
    offset: int
    size: int


@dataclass(frozen=True)
class BlobGeometry:
    entry_bytes: int
    fields: tuple[BlobField, ...]
    padding: tuple[tuple[int, int], ...]


def blob_geometry_a4(width, dims):
    """Same16-byte field offsets as six-field HostExpertCache, in one row."""
    fields, padding, cursor = [], [], 0
    for shape, dtype in authority_fields_a4(1, width, dims):
        offset = -(-cursor // 16) * 16
        if offset != cursor:
            padding.append((cursor, offset))
        size = prod(shape) * (1 if dtype == "U8" else 4)
        fields.append(BlobField(shape[1:], dtype, offset, size))
        cursor = offset + size
    entry = -(-cursor // 16) * 16
    if entry != cursor:
        padding.append((cursor, entry))
    return BlobGeometry(entry, tuple(fields), tuple(padding))

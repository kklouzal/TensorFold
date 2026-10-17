"""Header-only host backing bound for the existing direct-read protocol.

The pinned runtime's CachingHostAllocator rounds each Reader allocation to a
power-of-two bucket. The two 64 MiB + 3-page blocks have a 256 MiB backing bound,
including alignment; this excludes OS/driver metadata. Source controls tie it to
direct_read.Reader; this module does not configure the reader or import CUDA.
"""

import os


def pinned_backing_bytes(payload_bytes: int) -> int:
    """Pinned allocator bucket bound, including uncached smaller allocations."""
    if type(payload_bytes) is not int or payload_bytes < 0:
        raise ValueError("pinned payload bound must be nonnegative integer bytes")
    return 0 if payload_bytes == 0 else 1 << (payload_bytes - 1).bit_length()


READER_HOST_RING_BYTES = 2 * pinned_backing_bytes((64 << 20) + 3 * 4096)


def reader_host_bytes(buffered_bytes: int, publication_bytes: int = 0, *, staged: bool | None = None) -> int:
    """Rings coexist with either a buffered read or later D2H publication.

    Native direct/buffered reads and later D2H payloads are successive phases.
    On Windows, blocking GPU uploads use pinned buffered reads instead: their
    cached power-of-two size classes sum to less than twice the largest bucket
    and remain resident alongside publication. Authority/read-ahead is separate.
    """
    if any(type(n) is not int or n < 0 for n in (buffered_bytes, publication_bytes)):
        raise ValueError("reader host payload bounds must be nonnegative integer bytes")
    if staged is None:
        staged = os.name == "nt"
    if type(staged) is not bool:
        raise ValueError("staged reader policy must be a bool")
    if staged:
        return 2 * pinned_backing_bytes(buffered_bytes) + publication_bytes
    return READER_HOST_RING_BYTES + max(buffered_bytes, publication_bytes)

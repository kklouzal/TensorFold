"""Header-only host backing bound for the existing direct-read protocol.

The pinned runtime's CachingHostAllocator rounds each Reader allocation to its
power-of-two bucket. The two64MiB+3page blocks have a256MiB backing bound, including
alignment; this is not OS/driver metadata accounting. Source controls tie it to direct_read.Reader; this
module neither configures that reader nor imports an accelerator runtime.
"""

READER_HOST_RING_BYTES = 2 * (1 << (((64 << 20) + 3 * 4096 - 1).bit_length()))


def reader_host_bytes(buffered_bytes: int, publication_bytes: int = 0) -> int:
    """Rings coexist with either a buffered read or later D2H publication.

    Those two temporary payloads belong to successive loading phases, so their
    maximum is charged once; authority and asynchronous read-ahead are separate.
    """
    if any(type(n) is not int or n < 0 for n in (buffered_bytes, publication_bytes)):
        raise ValueError("reader host payload bounds must be nonnegative integer bytes")
    return READER_HOST_RING_BYTES + max(buffered_bytes, publication_bytes)

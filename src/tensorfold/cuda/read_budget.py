"""Header-only host backing bound for the existing direct-read protocol.

The pinned runtime's CachingHostAllocator rounds each Reader allocation to a
power-of-two bucket. The two 64 MiB + 3-page blocks have a 256 MiB backing bound,
including alignment; this excludes OS/driver metadata. Source controls tie it to
direct_read.Reader; this module does not configure the reader or import CUDA.
"""

import os
import sys


def pinned_backing_bytes(payload_bytes: int) -> int:
    """Pinned allocator bucket bound, including uncached smaller allocations."""
    if type(payload_bytes) is not int or payload_bytes < 0:
        raise ValueError("pinned payload bound must be nonnegative integer bytes")
    return 0 if payload_bytes == 0 else 1 << (payload_bytes - 1).bit_length()


READER_PIECE_BYTES = 64 << 20
READER_HOST_RING_BYTES = 2 * pinned_backing_bytes(READER_PIECE_BYTES + 3 * 4096)


def f32_scalar_batch_host_bytes(count: int, name_bytes: int) -> int:
    """Additional host envelope for SafeTensors.get_f32_scalars' owned plan.

    Singleton containers conservatively charge each plan entry as its own
    dict/list/set, including their growth slack. Offset integers and paths are
    borrowed; enumerate's index and the offset/index tuple are owned. The name
    tuple and generated strings belong to the caller. Charge both the seen-set
    planning phase and sort/order-list phase, although they do not coexist.
    The byte-row reorder owns F32[N], int64[K] and raw U8[4K] with K <= N,
    including the Reader's direct-I/O alignment. It retains no per-field native
    Tensor/Storage objects. Existing per-projection scalar reads already cover
    the fixed number of native records/views; this adds their payload and all
    variable Python metadata explicitly, without consuming packing slack.
    """
    if any(type(n) is not int or n < 0 for n in (count, name_bytes)):
        raise ValueError("scalar batch count and name bytes must be nonnegative integers")
    if not count:
        return 0
    pointer = sys.getsizeof((None,)) - sys.getsizeof(())
    appended = []
    appended.append(None)
    # A separate singleton list per entry also covers the temporary position
    # list. Four pointer planes cover key-sort and merge temporary storage.
    per_entry = (sys.getsizeof({None: None}) + 2 * sys.getsizeof(appended)
                 + sys.getsizeof({None}) + sys.getsizeof((None, None))
                 + sys.getsizeof(count) + 4 * pointer)
    metadata = name_bytes + sys.getsizeof(()) + count * (pointer + per_entry)
    return metadata + 16 * count + 3 * 4096


def reader_host_bytes(buffered_bytes: int, publication_bytes: int = 0, *, staged: bool | None = None) -> int:
    """Rings coexist with a read or caller-owned CPU preparation/publication.

    Native reads and later CPU preparation/publication are successive phases.
    On Windows, blocking GPU uploads use pinned buffered reads instead: their
    cached power-of-two size classes sum to less than twice the largest bucket
    and remain resident alongside CPU temporaries. Authority/read-ahead is
    otherwise separate; callers include overlapping authority in a phase bound.
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

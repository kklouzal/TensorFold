"""Execution-family names shared by preparation, profile scope and TP plans.

Availability means that the native owners and buffers were prepared at startup;
it is not hardware/quality/performance qualification. Production selection still
requires the content-verified profile evidence contract.
"""

from types import MappingProxyType

from tensorfold.cuda.sm70.paged_attention import KERNEL_FAMILY

REFERENCE_KERNEL = "torch-reference"
UNCAPTURED = "uncaptured"
GRAPH_FAMILY = "owner-attention-v1"
FAMILY_FIELDS = frozenset({"kernel_family", "graph_family"})
REFERENCE = MappingProxyType({"kernel_family": REFERENCE_KERNEL, "graph_family": UNCAPTURED})


def row_buckets(rows, requested=None):
    """A bounded padded-operation policy shared by eager and captured native work."""
    if type(rows) is not int or rows < 1:
        raise ValueError("owner execution rows must be a positive admitted bound")
    if requested is None:
        return tuple(1 << power for power in range(rows.bit_length()) if 1 << power < rows) + (rows,)
    if (not isinstance(requested, (list, tuple)) or not requested
            or any(type(value) is not int or not 1 <= value <= rows for value in requested)
            or any(left >= right for left, right in zip(requested, requested[1:]))
            or requested[-1] != rows):
        raise ValueError("owner buckets must be distinct increasing rows covering the exact endpoint")
    return tuple(requested)

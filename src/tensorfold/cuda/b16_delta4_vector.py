"""Experimental vector-load projection/lookup over owned Delta4 v1 storage.

This experiment depends on the owned ``b16_delta4`` CPU format authority.
R1/2/4/8/16 changes reuse only;
each row keeps the raw B16 ordered FP32 FMA/XOR/BF16-rounding contract.

The caller owns all packed fields and input storage, keeps them unchanged
until every queued reader retires, and establishes any cross-stream handoff.
Results are asynchronous on the current CUDA stream. Unknown CUDA failure is
terminal for the isolated worker: retain actual owners/errors, never retry a
fence. Numerical, resource and integrated speed qualification remain required.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tensorfold.cuda.b16_delta4 import PackedDelta4


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    root = Path(__file__).parent
    return load(name='tensorfold_b16_delta4_vector_v1',
                sources=[str(root / 'b16_delta4_vector.cpp'), str(root / 'b16_delta4_vector.cu')],
                extra_cuda_cflags=['-O3'], verbose=False)


def unpack(weight: PackedDelta4):
    """Diagnostic complete bit decoder using the same eight-word load helper."""
    from tensorfold.cuda.b16_delta4 import _validate

    _validate(weight, cuda=True)
    return _ext().unpack(weight.low, weight.delta, weight.metadata, weight.raw, *weight.shape)


def matmul_rows(x, weight: PackedDelta4, rows=8):
    """Original operands, one ordered FP32 chain per row; no decoded mirror.

    Input uses bounded1/2/4/8/16-row native tiles, including17+ callers.
    Aligned vector accesses require low8/delta4/
    metadata4/raw16/input16-byte base alignment; the native boundary rejects
    misaligned contiguous views before allocation or GPU work. No different
    prompt GEMM arithmetic contract is selected.
    """
    if type(rows) is not int or rows not in (1, 2, 4, 8, 16):
        raise ValueError('reuse rows must be1/2/4/8/16')
    from tensorfold.cuda.b16_delta4 import _validate

    _validate(weight, cuda=True)
    return _ext().matmul_rows(x, weight.low, weight.delta, weight.metadata, weight.raw, *weight.shape, rows)


def lookup(ids, weight: PackedDelta4, *, max_rows):
    """Return unscaled original BF16 rows; CUDA rejects invalid IDs fatally.

    Any rank of contiguous CUDA INT32/INT64 IDs is supported, including empty
    and scalar input. The admitted caller bound applies to cumulative ID count.
    No full-table decode or sqrt(hidden) scaling is performed here.
    """
    from tensorfold.cuda.b16_delta4 import MAX_I64, _validate

    _validate(weight, cuda=True)
    if type(max_rows) is not int or not 0 < max_rows <= MAX_I64 // weight.shape[1] // 2:
        raise ValueError('positive admitted lookup row bound required')
    return _ext().lookup(ids, weight.low, weight.delta, weight.metadata, weight.raw, *weight.shape, max_rows)

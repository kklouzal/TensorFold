"""Prepared exact row reuse for the GB10 Gemma CT BF16 vocabulary head.

The qualified GB10 region selects exact R16 reuse at preparation. Other
regions retain their original head. Both share the same original BF16 parameter;
there is no new weight authority, quantization, scratch cache or runtime flag.
"""

from __future__ import annotations

from functools import partial
from dataclasses import dataclass

import torch

from tensorfold.families.qwen3_5.cuda.b16 import matmul_rows
from tensorfold.families.qwen3_5.cuda.weights import Plain


class _RowsHead(Plain):
    """One immutable prepared policy with the named incumbent kept as reference."""

    def __init__(self, incumbent: Plain, reuse_rows: int):
        super().__init__(incumbent.weight)
        self.incumbent = incumbent
        self.reuse_rows = reuse_rows
        self._multiply = partial(matmul_rows, rows=reuse_rows)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # A singleton cannot share weight loads. Keep its original launch;
        # every multi-row call uses the same selected exact reuse geometry.
        if x.shape[0] == 1:
            return self.incumbent(x)
        return self._multiply(x, self.weight)


def _qualified(incumbent: Plain) -> bool:
    """Stable selection discriminants, resolved once during preparation."""
    if type(incumbent) is not Plain:
        return False
    weight = incumbent.weight
    return (weight.dtype == torch.bfloat16 and tuple(weight.shape) == (262144, 2816)
            and weight.device.type == "cuda" and weight.is_contiguous()
            and torch.cuda.get_device_capability(weight.device) == (12, 1))


def select_head(incumbent: Plain) -> Plain:
    """Bind the measured exact R16 policy for original CT Gemma on GB10SM12.1.

    Code/chat holdout establishes a strict gain over the original head. Ranking
    R8 against R16 remains unresolved; this does not assert a unique optimum.
    Other architectures, shapes and parameter representations retain their
    original implementation. Call once while constructing the CT model, before
    target/DFlash borrowers are created; no hot capability discovery occurs.
    """
    return _RowsHead(incumbent, 16) if _qualified(incumbent) else incumbent


def prepare_head(incumbent: Plain, *, reuse_rows: int) -> Plain:
    """Prepare an explicitly chosen exact-reuse candidate before DFlash binds.

    Investigation is confined to original CT BF16 Gemma[262144,2816] on the
    current GB10SM12.1 region. Unsupported explicit requests fail at preparation.
    This function does not install the candidate or select a measured winner.
    The returned head retains Plain's public shape/storage/prefill contracts;
    target and DFlash must both call its __call__ through project_logits, whose
    final softcap remains outside the head and is applied exactly once.
    """
    if type(reuse_rows) is not int or reuse_rows not in (2, 4, 8, 16):
        raise ValueError("Gemma exact head reuse_rows must be2,4,8 or16")
    if not _qualified(incumbent):
        raise ValueError("Gemma exact head reuse requires original contiguous CUDA BF16[262144,2816] on GB10SM12.1")
    return _RowsHead(incumbent, reuse_rows)


@dataclass(frozen=True, init=False)
class Delta4Table:
    """Experimental exact tied lookup/head authority, with no raw .weight.

    Reuse is a bounded investigation choice, provisionally16; no performance
    policy is selected. Packed storage is immutable for all graph borrowers.
    Long-row projections use the same native ordered1/2/4/8/16-row tiles,
    preserving the17+ direct-head contract without a second GEMM arithmetic.
    """

    packed: object
    lookup_limit: int
    reuse_rows: int
    layout: str

    def __init__(self, packed, lookup_limit, *, reuse_rows=16):
        try:
            from tensorfold.cuda.b16_delta4 import MAX_I64, SCHEMA, _validate

            _validate(packed, cuda=True)
            if type(lookup_limit) is not int or not 0 < lookup_limit <= MAX_I64 // packed.shape[1] // 2:
                raise ValueError('positive admitted Delta4 lookup row bound required')
            if type(reuse_rows) is not int or reuse_rows not in (1, 2, 4, 8, 16):
                raise ValueError('Delta4 reuse rows must be1/2/4/8/16')
            object.__setattr__(self, 'packed', packed)
            object.__setattr__(self, 'lookup_limit', lookup_limit)
            object.__setattr__(self, 'reuse_rows', reuse_rows)
            object.__setattr__(self, 'layout', SCHEMA)
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
            packed = None

    @property
    def n(self):
        return self.packed.shape[0]

    @property
    def k(self):
        return self.packed.shape[1]

    @property
    def device(self):
        return self.packed.low.device

    def nbytes(self):
        return self.packed.nbytes

    def embedding(self, ids):
        from tensorfold.cuda.b16_delta4_vector import lookup

        return lookup(ids.contiguous(), self.packed, max_rows=self.lookup_limit)

    def __call__(self, hidden):
        from tensorfold.cuda.b16_delta4_vector import matmul_rows

        return matmul_rows(hidden.to(dtype=torch.bfloat16).contiguous(), self.packed, rows=self.reuse_rows)

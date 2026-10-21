"""Prepared exact row reuse for the GB10 Gemma CT BF16 vocabulary head.

The qualified GB10 region selects exact R16 reuse at preparation. Other
regions retain their original head. Both share the same original BF16 parameter;
there is no new weight authority, quantization, scratch cache or runtime flag.
"""

from __future__ import annotations

from functools import partial

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


def original_bf16_region(weight: torch.Tensor) -> bool:
    """The original contiguous BF16 Gemma vocabulary weight on GB10SM12.1."""
    return (weight.dtype == torch.bfloat16 and tuple(weight.shape) == (262144, 2816)
            and weight.device.type == "cuda" and weight.is_contiguous()
            and torch.cuda.get_device_capability(weight.device) == (12, 1))


def _qualified(incumbent: Plain) -> bool:
    """Stable selection discriminants, resolved once during preparation."""
    return type(incumbent) is Plain and original_bf16_region(incumbent.weight)


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

"""Prepared exact row reuse for the GB10 Gemma CT BF16 vocabulary head.

The candidate GB10 policy uses existing R4/R8/R16 kernels by row count. Other
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


class _AdaptiveRowsHead(Plain):
    """One prepared shape policy; the fixed R16 head remains a comparator."""

    def __init__(self, incumbent: Plain):
        super().__init__(incumbent.weight)
        self.incumbent = incumbent
        self._small = partial(matmul_rows, rows=4)
        self._medium = partial(matmul_rows, rows=8)
        self._large = partial(matmul_rows, rows=16)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        rows = x.shape[0]
        if rows == 1:
            return self.incumbent(x)
        # Zero rows retain the fixed R16 native validation/error path.
        multiply = self._small if 1 < rows <= 4 else self._medium if 4 < rows <= 8 else self._large
        return multiply(x, self.weight)


def _qualified(incumbent: Plain) -> bool:
    """Stable selection discriminants, resolved once during preparation."""
    if type(incumbent) is not Plain:
        return False
    weight = incumbent.weight
    return (weight.dtype == torch.bfloat16 and tuple(weight.shape) == (262144, 2816)
            and weight.device.type == "cuda" and weight.is_contiguous()
            and torch.cuda.get_device_capability(weight.device) == (12, 1))


def select_head(incumbent: Plain) -> Plain:
    """Bind the adaptive-row candidate for original CT Gemma on GB10SM12.1.

    R4 through four rows, R8 through eight, R16 above eight, with the original
    singleton. The prior fixed R16 gain does not qualify this candidate's cost.
    Other architectures, shapes and parameter representations retain their
    original implementation. Call once while constructing the CT model, before
    target/DFlash borrowers are created; no hot capability discovery occurs.
    """
    return _AdaptiveRowsHead(incumbent) if _qualified(incumbent) else incumbent


def prepare_head(incumbent: Plain, *, reuse_rows: int) -> Plain:
    """Prepare an explicitly chosen exact-reuse candidate before DFlash binds.

    Investigation is confined to original CT BF16 Gemma[262144,2816] on the
    current GB10SM12.1 region. Unsupported explicit requests fail at preparation.
    This function does not install the candidate or select a measured winner.
    The returned head retains Plain's public shape/storage/prefill contracts.
    The target applies its softcap once in project_logits. BF16 DFlash borrows
    that projector; bits8 DFlash copies the original weight into its private
    FP8 Head and bypasses this BF16 callable.
    """
    if type(reuse_rows) is not int or reuse_rows not in (2, 4, 8, 16):
        raise ValueError("Gemma exact head reuse_rows must be2,4,8 or16")
    if not _qualified(incumbent):
        raise ValueError("Gemma exact head reuse requires original contiguous CUDA BF16[262144,2816] on GB10SM12.1")
    return _RowsHead(incumbent, reuse_rows)

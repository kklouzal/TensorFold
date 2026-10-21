"""Prepared exact row reuse for the GB10 Gemma CT BF16 vocabulary head.

The qualified GB10 region selects exact R16 reuse at preparation. Other
regions retain their original head. Both share the same original BF16 parameter;
there is no new weight authority, quantization, scratch cache or runtime flag.
"""

from __future__ import annotations

from functools import lru_cache, partial
from pathlib import Path

import torch

from tensorfold.families.qwen3_5.cuda import b16
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


def owns_logits_result(head: Plain) -> bool:
    """Prove this immutable projector returns fresh storage, at preparation.

    Both native b16 entry points allocate their result with at::empty. A
    weight's shape/device alone does not establish that contract for a foreign
    projector. The row-reuse owner must retain its original Plain singleton
    projector, shared weight and constructor's native matmul binding.
    """
    if type(head) is Plain:
        return True
    return (type(head) is _RowsHead and type(head.incumbent) is Plain
            and head.weight is head.incumbent.weight and head.reuse_rows in (2, 4, 8, 16)
            and isinstance(head._multiply, partial) and head._multiply.func is matmul_rows
            and not head._multiply.args and head._multiply.keywords == {"rows": head.reuse_rows})


def _cap30_build_options() -> dict:
    """Canonical recipe for the separate unselected Gemma epilogue module."""
    here = Path(b16.__file__).parent
    return {"name": "tensorfold_gemma_b16_cap30_v1", "sources": [str(here / "b16.cpp"), str(here / "b16.cu")],
            "extra_cflags": ["-DTENSORFOLD_B16_CAP30=1"],
            "extra_cuda_cflags": ["-O3", "-DTENSORFOLD_B16_CAP30=1", "--ftz=false"], "verbose": False,
            "prepared": True}


@lru_cache(maxsize=1)
def _cap30_extension():
    """Bind through the managed build boundary; raw module code stays unchanged.

    A distinct module adds cold compilation/loading unless compatible prepared
    artifacts are present. All loading/residency costs remain selection gates.
    """
    from tensorfold.cuda.build import load

    return load(**_cap30_build_options())


def prepare_cap30(head: Plain):
    """Prepare a capped projector without changing the public raw head.

    Accept only the proved native ownership/arithmetic policies. Bind the
    authoritative parameter, reuse policy and loaded native entry once; there
    is no HOT capability discovery or extra softcap launch. The caller owns
    model authority, cap/region qualification and graph-bank replacement.
    """
    if not owns_logits_result(head):
        raise ValueError("Gemma cap30 epilogue requires its exact Plain or prepared row head")
    weight = head.weight
    rows = 0 if type(head) is Plain else head.reuse_rows
    native = _cap30_extension().b16_linear_cap30

    def project(hidden):
        # Plain casts to its weight and makes input contiguous at this boundary.
        if hidden.dtype != weight.dtype:
            hidden = hidden.to(weight.dtype)
        return native(hidden.contiguous(), weight, rows)

    return project


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

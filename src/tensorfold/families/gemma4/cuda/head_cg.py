"""Cold-bound Gemma-only weight-cache policy; generic B16 imports stay unchanged."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch

from tensorfold.families.qwen3_5.cuda import b16


@lru_cache(maxsize=1)
def _ext():
    from tensorfold.cuda.build import load

    here = Path(b16.__file__).parent
    # Direct authority bytes and the private macro are Torch-versioned inputs.
    return load(name="tensorfold_gemma_b16_cg_v1",
                sources=[str(here / "gemma_head_cg.cpp"), str(here / "b16.cu")],
                extra_cuda_cflags=["-O3", "-DTENSORFOLD_GEMMA_HEAD_CG"], verbose=False)


def matmul(x: torch.Tensor, w: torch.Tensor, rows: int) -> torch.Tensor:
    """Original row math with weight-only CG; publish/retain inputs to completion."""
    if x.dtype != w.dtype:
        x = x.to(w.dtype)
    return _ext().gemma_b16_head_cg(x.contiguous(), w.contiguous(), rows)

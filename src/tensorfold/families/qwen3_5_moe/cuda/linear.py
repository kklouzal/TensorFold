"""Qwen3.6 BF16 projections on one row-invariant tensor-core arithmetic for every row count.

The CT checkpoint stores its GDN projections and lm_head in BF16, and the DFlash drafter is BF16. Every row of
these projections -- prompt chunk, verify window, serial step, drafter block or drafter context -- runs the bf16 mma
prompt kernel's arithmetic (``b16.prompt``; ``b16.rows16`` issues the same mma chain from a narrower tile for up to
16 rows): one FP32 chain over K per (row, column) in fixed 16-wide k steps. A row's bits therefore depend on neither
the row count, the chunk, the tile height nor the other rows, and prompt and decode rows share them. Up to 16 rows
stream the weight once per launch (the warp-per-column ``b16.matmul`` streams it once per 8 or 16 rows and re-loads
the rows per column); more rows stream it once per prompt row tile. Weight values and precision are unchanged; a
layer with ``--prefill-fp8`` copies keeps its objects (``prepare``).
"""
from __future__ import annotations

import torch

from tensorfold.families.qwen3_5.cuda import b16
from tensorfold.families.qwen3_5.cuda.weights import Plain


class MmaLinear:
    """A stored BF16 [N, K] projection whose rows all take ``b16.prompt``'s arithmetic.

    Not a ``Plain``: the shared forward pairs two ``Plain`` gates on the warp-per-column kernel, which would give
    those rows another arithmetic. The weight is validated once here, so a call never reaches ``b16.prompt``'s
    FMA fallback (non-BF16 or K not a multiple of 64).
    """

    layout = "b16"
    __slots__ = ("weight",)

    def __init__(self, weight: torch.Tensor) -> None:
        if (not isinstance(weight, torch.Tensor) or weight.ndim != 2 or weight.dtype != torch.bfloat16
                or weight.device.type != "cuda" or not weight.is_contiguous() or weight.shape[0] < 1
                or weight.shape[1] < 64 or weight.shape[1] % 64 or weight.data_ptr() % 16):
            raise ValueError("MMA projections take contiguous 16-byte aligned CUDA BF16 [N, K] weights "
                             "with K a positive multiple of 64")
        self.weight = weight

    @property
    def n(self) -> int:
        return int(self.weight.shape[0])

    @property
    def k(self) -> int:
        return int(self.weight.shape[1])

    def nbytes(self) -> int:
        return self.weight.numel() * self.weight.element_size()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        # Up to 16 rows take the narrow-tile kernel, more the prompt kernel: the same mma chain, the same bits.
        return b16.rows16(x, self.weight) if x.shape[0] <= 16 else b16.prompt(x, self.weight)

    def prefill(self, x: torch.Tensor) -> torch.Tensor:
        return b16.prompt(x, self.weight)


def _bf16(projection) -> bool:
    return type(projection) is Plain and projection.weight.dtype == torch.bfloat16


def prepare(weights) -> None:
    """Bind each GDN layer stored wholly in BF16, and a BF16 head, to ``MmaLinear`` in place, before DFlash borrows it.

    Such a layer also gets ``zba`` = [z | b | a] stacked once (one launch for the three; each output column keeps its
    own chain), which replaces their storage: z, b and a stay as row views of it, so ``Weights.nbytes`` counts it
    once. A layer with any projection in another format, or a ``Plain8`` one (with an FP8 prompt copy), keeps its
    objects.
    """
    for layer in weights.layers:
        gdn = layer.gdn
        if gdn is None or not all(_bf16(p) for p in (gdn.qkv, gdn.z, gdn.b, gdn.a, gdn.out)):
            continue
        stacked = torch.cat([gdn.z.weight, gdn.b.weight, gdn.a.weight])
        z, b = gdn.z.n, gdn.b.n
        gdn.qkv, gdn.out = MmaLinear(gdn.qkv.weight), MmaLinear(gdn.out.weight)
        gdn.zba = MmaLinear(stacked)
        gdn.z, gdn.b, gdn.a = MmaLinear(stacked[:z]), MmaLinear(stacked[z:z + b]), MmaLinear(stacked[z + b:])
    if _bf16(weights.head):
        weights.head = MmaLinear(weights.head.weight)

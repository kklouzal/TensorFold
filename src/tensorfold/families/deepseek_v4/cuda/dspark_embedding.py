"""Graph-capturable proposal embeddings owned by one installed DSpark head."""

from __future__ import annotations

import torch

from . import dspark_weights as dw
from .kernels.triton_utils import jit, tl, triton


@jit
def _assemble_rows(anchors, noise, out, HIDDEN: tl.constexpr, TOKENS: tl.constexpr, BLOCK: tl.constexpr):
    request = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    anchor = tl.load(anchors + request * HIDDEN + columns)
    fixed = tl.load(noise + columns)
    dst = out + request * TOKENS * HIDDEN + columns
    tl.store(dst, anchor)
    for row in tl.static_range(1, TOKENS):
        tl.store(dst + row * HIDDEN, fixed)


class ProposalEmbedding:
    """Retain the installed target's single FP16 NOISE row for all graph replays.

    Prepare on the lane stream before graph capture. Every rank must install the same policy.
    The target returns owned rows. The head owner must keep this object alive until all
    captures and queued reads retire and order other streams after preparation.
    """

    def __init__(self, target_embed, device: torch.device):
        if device.type != "cuda":
            raise ValueError("installed DSpark proposal embeddings require a CUDA target")
        with torch.inference_mode():
            noise = target_embed(torch.full((1,), dw.NOISE, device=device, dtype=torch.int64))
        self._check_rows(noise, 1, device)
        self._target_embed = target_embed
        self._noise = noise.view(torch.int16)

    @staticmethod
    def _check_rows(rows, count, device):
        if (rows.dtype != torch.float16 or rows.device != device or rows.shape != (count, dw.HIDDEN)
                or not rows.is_contiguous()):
            raise ValueError("the installed target embedding must return contiguous CUDA FP16 [T, 5120]")

    def __call__(self, anchors):
        if (anchors.dtype != torch.int64 or anchors.ndim != 1 or not anchors.is_contiguous()
                or anchors.device != self._noise.device or not 1 <= anchors.numel() <= 65535 // dw.BLOCK):
            raise ValueError("DSpark proposal anchors must be a nonempty contiguous CUDA int64 vector on the head device")
        count = anchors.numel()
        rows = self._target_embed(anchors)
        self._check_rows(rows, count, anchors.device)
        out = torch.empty((count * dw.BLOCK, dw.HIDDEN), device=anchors.device, dtype=torch.float16)
        block = 512
        # Copy raw words so the assembler preserves signed zeros and every other FP16 code.
        _assemble_rows[(count, triton.cdiv(dw.HIDDEN, block))](
            rows.view(torch.int16), self._noise, out.view(torch.int16), dw.HIDDEN, dw.BLOCK, block, num_warps=4)
        return out

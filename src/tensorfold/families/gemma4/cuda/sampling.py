"""Gemma-only greedy verification of immutable identity-vocabulary score rows."""

from __future__ import annotations

import numpy as np
import torch

from tensorfold.cuda.sampling import validate_rows


def greedy_rows(logits: torch.Tensor, positions, sampling) -> list[int]:
    """Return first-maximum columns without a separate vocabulary-validity scan.

    The engine selects this greedy policy once, after the initial legacy draw
    has validated the request. Identity token IDs need no map. Keep dynamic
    row/position validation and the legacy whole-packet invalid-score errors.

    Reduction uses the original FP16/BF16/FP32/FP64 score dtype. Widen only the
    returned maxima to FP64, then transport their bits beside int64 columns.
    This preserves finite FP64 extremes and avoids another vocabulary pass.
    Adopted max/index semantics propagate NaN and select the lowest tied index.

    Scores remain immutable on the caller's current CUDA stream until the
    synchronous readback returns. Local references retain all borrowed graph
    outputs and packet storage through completion; returned IDs own no tensors.
    """
    validate_rows(logits, positions, sampling)
    if logits.shape[0] == 0:
        return []
    maximum, columns = logits.max(dim=-1)
    words = maximum.to(torch.float64).contiguous().view(torch.int64)
    packet = torch.stack((columns, words), dim=1).cpu().numpy()
    maxima = packet[:, 1].view(np.float64)  # Same8-byte view permits stride16; packet owns this live storage.
    # The legacy sampler gives any NaN/+inf row precedence over all-masked rows
    # anywhere in the same packet. Validate every row before publishing IDs.
    if np.isnan(maxima).any() or np.isposinf(maxima).any():
        raise ValueError("sampling scores contain NaN or positive infinity")
    if not np.isfinite(maxima).all():
        raise ValueError("a complete vocabulary row needs at least one finite score")
    return packet[:, 0].tolist()

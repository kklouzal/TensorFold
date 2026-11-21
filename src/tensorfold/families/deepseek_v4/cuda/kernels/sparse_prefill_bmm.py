# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Provenance: 1Cat-vLLM 4d57473db67eb77a13018d179e4d95169e82044f
#   vllm/models/deepseek_v4/common/ops/sparse_prefill_bmm.py (blob 9b2be25acea5e8ea9fb746021da5ed87a1cb16a9)
# TensorFold changes:
# - Only attend_gathered_keys (verbatim) is vendored. The V4 index-based sparse_attn_prefill_bmm, its workspace
#   specs, _fit and MAX_TOKENS_PER_PASS are V4-only and unused by V4.1.
# - At the friend's production constants (VLLM_V41_ATTN_FUSED=1) the V4.1 attention paths use
#   attention_ops._attend_fused instead; this is the upstream unfused PV/softmax boundary, kept as requested
#   for the multi-request port and as a reference.
"""Sparse MLA attention over gathered keys as two batched matmuls (V4.1 unfused reference)."""

import torch


def attend_gathered_keys(
    q: torch.Tensor,
    keys: torch.Tensor,
    unused: torch.Tensor,
    scale: float,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    scores: torch.Tensor,
    logits: torch.Tensor,
    probs: torch.Tensor,
) -> None:
    """Attention of q [T, H, D] over its own gathered keys [T, W, D]; unused
    [T, W] marks slots that hold no key. Buffers are exactly sized."""
    width = keys.shape[1]
    # An unused slot holds whatever the gather read for it, which need not be
    # a written key: a short prompt has no compressed entries yet, and what
    # the workspace held before may be NaN. Its weight is zero below, but
    # 0 * NaN is NaN.
    keys.masked_fill_(unused[:, :, None], 0)
    # Preserve the existing FP32 QK score boundary before softmax. Rounding
    # scores to FP16 here can overflow even when the final attention is finite.
    torch.bmm(q, keys.transpose(1, 2), out=scores, out_dtype=torch.float32)
    scores.mul_(scale)
    logits[..., :width].copy_(scores)
    logits[..., :width].masked_fill_(unused[:, None, :], float("-inf"))
    logits[..., width] = attn_sink
    torch.softmax(logits, dim=-1, out=probs)
    # Existing HMMA attention also converts probabilities to the key dtype
    # before PV; keep that precision boundary rather than narrowing QK.
    probabilities = probs[..., :width].to(keys.dtype)
    torch.bmm(probabilities, keys, out=output)

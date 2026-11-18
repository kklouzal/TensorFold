"""Greedy SM70 tensor-parallel draft choice evaluated on the device.

Equivalent to ``decode.choose_gathered(..., with_prob=True)`` for a greedy
policy whose gathered packet and shard plan are usable (the caller's
``_gathered_fits`` and ``graph_ids_safe`` conditions). The family kernel
evaluates the host chooser's packet checks and its greedy rule (maximum real
FP32 score, lowest global ID among exact ties) per row and writes small words:
flags, token, chosen score and each rank's FP32 log-sum-exp. ``issue`` queues
the kernel and the words' copy into pinned memory right after the producing
MTP step, ahead of the step's status readback wait, so the pick needs no
readback of its own. ``take`` raises the original errors in their original
precedence across rows, returns ``None`` for a possibly truncated tie (the
caller then runs the original chooser and its full-source repair), and
computes the confidence with the original FP64 operations (bitwise equal).
"""

from __future__ import annotations

import numpy as np
import torch

from .state import CAND

PADDING, SCORES, NORM_INVALID, NORM_NONE, AMBIGUOUS, BEST, DISTINCT = 1, 2, 4, 8, 16, 32, 64


class DraftPick:
    """One decoder's fixed kernel operands; used only on its coordinator stream."""

    def __init__(self, w, plan, rows: int) -> None:
        from tensorfold.cuda.sm70.family_native import extension

        world = int(w.meta["world"])
        if plan is None or len(plan.widths) != world or not 1 <= world <= 8 or type(rows) is not int or rows < 1:
            raise ValueError("device draft choice requires one shard plan per rank and a bounded row envelope")
        widths = [int(size) for size in plan.widths]
        self.world, self.rows, self.plan = world, rows, plan
        self.kept = torch.tensor([min(CAND, size) for size in widths], dtype=torch.int64, device=w.device)
        self.incomplete = torch.tensor([size > CAND for size in widths], dtype=torch.uint8, device=w.device)
        self.words = torch.empty(rows * (3 + world), dtype=torch.int64, device=w.device)
        self.host = torch.empty(rows * (3 + world), dtype=torch.int64, pin_memory=True)
        self.host_words = self.host.numpy()
        self.done = torch.cuda.Event()
        self.native = extension()
        self.pending = None                              # (rows, packet pointer) of the queued, unconsumed pick

    def issue(self, cand_all: torch.Tensor, rows: int) -> None:
        """Queue the pick over the step's packet and its words' host copy on the current stream (no wait)."""
        self.pending = None
        world, width = self.world, 2 * CAND + 1
        if (type(rows) is not int or not 0 < rows <= self.rows or cand_all.dtype != torch.float32
                or not cand_all.is_cuda or not cand_all.is_contiguous() or cand_all.numel() < world * rows * width):
            raise ValueError("device draft choice requires the complete CUDA packet within its row envelope")
        count = rows * (3 + world)
        self.native.draft_pick(cand_all, self.kept, self.incomplete, self.words, world, rows, CAND)
        self.host[:count].copy_(self.words[:count], non_blocking=True)
        self.done.record(torch.cuda.current_stream(cand_all.device))
        self.pending = rows, cand_all.data_ptr()

    def take(self, cand_all: torch.Tensor, rows: int):
        """The queued pick for exactly this packet and row count, or ``False`` when none is queued for it."""
        pending, self.pending = self.pending, None
        if pending != (rows, cand_all.data_ptr()):
            return False
        self.done.synchronize()
        words = self.host_words[:rows * (3 + self.world)].reshape(rows, 3 + self.world)
        flags = int(np.bitwise_or.reduce(words[:, 0]))
        if flags & PADDING:
            raise ValueError("graph-gathered padding differs from the immutable vocabulary shard widths")
        if flags & SCORES:
            raise ValueError("sampling needs finite scores or -inf masks, with one finite score per complete row")
        if flags & NORM_INVALID:
            raise ValueError("a rank's complete sampling scores contain NaN or positive infinity")
        if flags & NORM_NONE:
            raise ValueError("a complete vocabulary row needs at least one finite score")
        if flags & AMBIGUOUS:
            return None
        if flags & BEST:
            raise ValueError("sampling needs finite scores or -inf masks, with one finite score per complete row")
        if flags & DISTINCT:
            raise ValueError("gathered global token IDs must be distinct")
        best = words[:, 2].astype(np.int32).view(np.float32)
        norms = np.ascontiguousarray(words[:, 3:].T).astype(np.int32).view(np.float32)
        # decode._gathered_probabilities' FP64 operations on the same (world, rows) layout.
        lse = norms.astype(np.float64)
        top = lse.max(axis=0)
        total = top + np.log(np.exp(lse - top).sum(axis=0))
        return [(int(words[row, 1]), float(np.exp(float(best[row]) - total[row]))) for row in range(rows)]

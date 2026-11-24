"""One explicitly prepared DeepSeek-V4.1 draft/verify replay shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from .dspark_head import RING
from .dspark_weights import BLOCK as DRAFT_BLOCK, HEAD_DIM, TAPS
from .graphs import shape_keys
from .round_sampling import compact_packet
from .target import BLOCK, Step


@dataclass(frozen=True)
class VerifyCommit:
    """Host owner's resolved preceding block, consumed once before the next use of its slot."""

    position: int
    rows: int
    accepted: int


@dataclass(frozen=True)
class VerifyEntry:
    slot: int
    position: int
    anchor: int
    pending: VerifyCommit | None = None


class VerifyOutput(NamedTuple):
    compact: torch.Tensor
    drafts: torch.Tensor
    local_logits: torch.Tensor


class SpecShape:
    """One (bucket, depth, index context) for identical rank-ordered calls.

    The caller owns pending blocks across all shapes. Supply a pending commit
    exactly once for a still-owned previous block; settle ending/canceled
    slots before any read, detach, reset or reuse. The slot-page table, target
    state, head weights and draft ring must retain their storage for this
    shape's entire graph lifetime. Do not run two shapes concurrently on
    those owners. Device results are stream-ordered and valid through the
    next run's submission; a consumer must finish its reads before reuse.
    After a failed submission retire the cohort using the engine's failure
    policy; do not retry a frame that may already have consumed its receipt.

    Depth zero here refreshes draft context. The normal no-drafter decode
    graph remains a separate owner.
    """

    def __init__(self, engine, head, spec, bucket: int, depth: int, index_context: int):
        target = engine.target
        if (type(bucket) is not int or type(depth) is not int or type(index_context) is not int
                or (bucket, index_context) not in shape_keys(engine.streams, engine.context)
                or not 0 <= depth <= DRAFT_BLOCK):
            raise ValueError("DSpark requires an admitted bucket, depth and index class")
        rows = depth + 1
        tokens = bucket * rows
        if tokens > min(RING, target.hash_state.max_tokens):
            raise ValueError("DSpark verify rows exceed the shape's scratch ring or target budget")
        self.engine, self.target, self.head, self.spec = engine, target, head, spec
        self.bucket, self.depth, self.index_context = bucket, depth, index_context
        self.rows = rows
        self.scratch = engine.slots - 1
        self._page_table = engine.slot_pages.table
        device = target.device
        if (spec.slots != self.scratch
                or self._page_table.shape != (engine.slots, engine.slot_pages.width)
                or self._page_table.dtype != torch.int32 or self._page_table.device != device
                or not self._page_table.is_contiguous()
                or head.ring.shape != (engine.slots, len(TAPS), RING, HEAD_DIM)
                or head.ring.dtype != torch.float16 or head.ring.device != device or not head.ring.is_contiguous()):
            raise ValueError("DSpark requires contiguous device page storage and the prepared draft ring")
        self._host = np.empty((5, bucket), dtype=np.int64)
        self._padding = np.array([engine.geometry.bos, 0, self.scratch, -1, self.scratch], dtype=np.int64)[:, None]
        self._staged = torch.empty((5, bucket), dtype=torch.int64, device=device)
        self.commit_slots = torch.full((bucket,), self.scratch, dtype=torch.int32, device=device)
        self.offsets = torch.arange(rows, dtype=torch.int64, device=device)
        self.row_indices = torch.arange(tokens, dtype=torch.int64, device=device)
        self.taps = torch.empty((tokens, len(TAPS) * target.g.hidden), dtype=torch.float16, device=device)
        self.empty_drafts = torch.empty((bucket, 0), dtype=torch.int64, device=device)
        self.compact = torch.empty((engine.world, tokens, 3), dtype=torch.float32, device=device)
        self.step = Step(
            ids=torch.empty(tokens, dtype=torch.int64, device=device),
            positions=torch.empty(tokens, dtype=torch.int64, device=device),
            valid=torch.empty(tokens, dtype=torch.bool, device=device),
            token_req=torch.arange(bucket, dtype=torch.int32, device=device).repeat_interleave(rows),
            qsl=torch.arange(bucket + 1, dtype=torch.int32, device=device) * rows,
            qsl_cpu=list(range(0, tokens + 1, rows)),
            num_reqs=bucket,
            req_slot=torch.full((bucket + 8,), self.scratch, dtype=torch.int32, device=device),
            block_table=torch.empty((bucket, engine.slot_pages.width), dtype=torch.int32, device=device),
            prefill=[],
            decode_index=None,
            single_token_slots=depth == 0,
            spec=spec if depth else None,
        )
        spec.validate_step(target, self.step)  # even depth zero may settle a prior verify block
        self.graph = None
        self._graph_out = None

    def _stage(self, entries):
        if len(entries) > self.bucket:
            raise ValueError("DSpark frame exceeds its prepared bucket")
        seen = set()
        owned = self.target.pool.owned
        for entry in entries:
            if (type(entry) is not VerifyEntry or any(type(v) is not int for v in (
                    entry.slot, entry.position, entry.anchor))):
                raise ValueError("DSpark verify entries require integer slot, position and token")
            slot, pos = entry.slot, entry.position
            end = pos + self.rows
            pages = owned.get(slot, ())
            if (slot in seen or not 0 <= slot < self.scratch or not 0 <= pos
                    or end > self.index_context or end > len(pages) * BLOCK
                    or not 0 <= entry.anchor < self.engine.geometry.vocab):
                raise ValueError("DSpark frame has an invalid or duplicate live slot, position or token")
            seen.add(slot)
            pending = entry.pending
            if pending is not None and (
                type(pending) is not VerifyCommit
                or any(type(v) is not int for v in (pending.position, pending.rows, pending.accepted))
                or not 0 <= pending.position
                or not 2 <= pending.rows <= DRAFT_BLOCK + 1
                or not 0 <= pending.accepted <= pending.rows
                or pending.position + pending.rows > min(self.engine.context, len(pages) * BLOCK)
                or pending.position + pending.accepted != pos
            ):
                raise ValueError("DSpark pending verify count does not lead to the next anchor")
        host = self._host
        host[:] = self._padding
        pending_any = False
        for i, entry in enumerate(entries):
            host[:3, i] = entry.anchor, entry.position, entry.slot
            if entry.pending is not None:
                host[3:, i] = entry.pending.accepted, entry.slot
                pending_any = True
        # Like SlotPages.set, the pageable input is staged before the host can rewrite it.
        self._staged.copy_(torch.from_numpy(host), non_blocking=True)
        return pending_any

    def _head_logits(self, value):
        return self.target._linear(value, self.target.w.head).float()

    def _body(self):
        target, step = self.target, self.step
        anchors, positions, slots = self._staged[:3]
        step.req_slot[:self.bucket].copy_(slots)
        step.valid.copy_((slots != self.scratch).repeat_interleave(self.rows))
        if self.depth:
            drafts = self.head.draft(anchors, positions, slots, target.embed, self._head_logits, depth=self.depth)
            step.ids.copy_(torch.cat((anchors[:, None], drafts), 1).reshape(-1))
        else:
            drafts = self.empty_drafts
            step.ids.copy_(anchors)
        step.positions.copy_((positions[:, None] + self.offsets).reshape(-1))
        torch.index_select(self._page_table, 0, step.req_slot[:self.bucket], out=step.block_table)

        prior = target.index_context
        target.index_context = self.index_context
        try:
            state = target.begin(step)
            target.layers_with_draft_taps(state, target.units, self.row_indices, self.taps)
            logits = target.end(state, self.row_indices)
        finally:
            target.index_context = prior
        # Only live slots can expose draft context. Scratch rows use distinct
        # physical destinations and retain the staged rotary positions.
        self.head.insert_context(slots.repeat_interleave(self.rows), self.taps, step.positions,
                                 padding=(self.scratch, self.row_indices))
        packet = compact_packet(logits, self.engine.rank)
        self.engine.comm.all_gather(packet, self.compact)
        return VerifyOutput(self.compact, drafts, logits)

    def _commit_staged(self) -> None:
        self.commit_slots.copy_(self._staged[4])
        self.spec.commit(self.commit_slots, self._staged[3])

    def run(self, entries) -> VerifyOutput:
        """Commit the staged owners once, then enqueue this shape on the current stream."""
        if self._stage(entries):
            self._commit_staged()
        if self.graph is None:
            return self._body()
        self.graph.replay()
        return self._graph_out

    @torch.inference_mode()
    def capture(self, pool, stream) -> None:
        """Capture a scratch-only frame using the target's warmed inference workspaces."""
        if self.graph is not None:
            raise RuntimeError("DSpark verify shape is already captured")
        self._stage(())
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            # Use the live commit buffers, with every slot masked to scratch.
            # Prepare this separately so the first accepted block cannot JIT
            # compile a restore, and no commit is added to graph replay.
            self._commit_staged()
            self._body()
        torch.cuda.current_stream().wait_stream(stream)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            self._graph_out = self._body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = graph

"""One explicitly prepared DeepSeek-V4.1 draft/verify replay shape."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import numpy as np
import torch

from .dspark_weights import BLOCK as DRAFT_BLOCK, RING as DRAFT_RING
from .graphs import shape_keys
from .round_sampling import compact_packet
from .target import BLOCK, DRAFT_TAP_LAYERS, Step


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
    full_logits: torch.Tensor | None = None
    sampled: torch.Tensor | None = None      # rank 0: device-sampled lanes' keyed tokens, their rows in vector order


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

    def __init__(self, engine, head, spec, bucket: int, depth: int, index_context: int, *,
                 use_prepared_embedding: bool = False):
        target = engine.target
        if (type(bucket) is not int or type(depth) is not int or type(index_context) is not int
                or (bucket, index_context) not in shape_keys(engine.streams, engine.context)
                or not 0 <= depth <= DRAFT_BLOCK):
            raise ValueError("DSpark requires an admitted bucket, depth and index class")
        rows = depth + 1
        tokens = bucket * rows
        if tokens > min(DRAFT_RING, target.hash_state.max_tokens):
            raise ValueError("DSpark verify rows exceed the shape's scratch ring or target budget")
        self.engine, self.target, self.head, self.spec = engine, target, head, spec
        self._embed = None if use_prepared_embedding or depth == 0 else target.embed
        self.bucket, self.depth, self.index_context = bucket, depth, index_context
        self.rows = rows
        self.scratch = engine.slots - 1
        self._page_table = engine.slot_pages.table
        device = target.device
        if spec._target is not target:
            raise ValueError("DSpark verify state belongs to another target")
        if (spec.slots != self.scratch
                or self._page_table.shape != (engine.slots, engine.slot_pages.width)
                or self._page_table.dtype != torch.int32 or self._page_table.device != device
                or not self._page_table.is_contiguous()
                or head.ring.shape != (engine.slots, len(DRAFT_TAP_LAYERS), DRAFT_RING, target.g.head_dim)
                or head.ring.dtype != torch.float16 or head.ring.device != device or not head.ring.is_contiguous()):
            raise ValueError("DSpark requires contiguous device page storage and the prepared draft ring")
        self._host = np.empty((5, bucket), dtype=np.int64)
        self._padding = np.array([engine.geometry.bos, 0, self.scratch, -1, self.scratch], dtype=np.int64)[:, None]
        self._staged = torch.empty((5, bucket), dtype=torch.int64, device=device)
        self.commit_slots = torch.full((bucket,), self.scratch, dtype=torch.int32, device=device)
        self.offsets = torch.arange(rows, dtype=torch.int64, device=device)
        self.row_indices = torch.arange(tokens, dtype=torch.int64, device=device)
        self.taps = torch.empty((tokens, len(DRAFT_TAP_LAYERS) * target.g.hidden), dtype=torch.float16, device=device)
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
            drafts = self.head.draft(anchors, positions, slots, self._embed, self._head_logits, depth=self.depth)
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

    def _commit_staged(self, *, scratch_only=False) -> None:
        self.commit_slots.copy_(self._staged[4])
        if scratch_only:
            self.spec.commit(self.commit_slots, self._staged[3])
        else:
            self.commit_checked(self.commit_slots, self._staged[3])

    def commit_checked(self, slots: torch.Tensor, counts: torch.Tensor) -> None:
        """Fence this rank before writes and before its commit may be acknowledged."""
        status = (self.target.hash_state.err, *(state.err for state in self.target.c2.values()))
        before = torch.cat(status).tolist()
        if any(before):
            raise RuntimeError(f"DeepSeek DSpark refused a poisoned verification state: {before}")
        self.spec.commit(slots, counts)
        after = torch.cat(status).tolist()
        if any(after):
            raise RuntimeError(f"DeepSeek DSpark commit failed: {after}")

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
            self._commit_staged(scratch_only=True)
            self._body()
        torch.cuda.current_stream().wait_stream(stream)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, pool=pool, stream=stream):
            self._graph_out = self._body()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = graph


class FixedVerifyOwner:
    """One unresolved verification per rank across prepared vector shapes.

    The future engine transport must submit identical frames on every rank
    and join the commit acknowledgments before publishing output or reusing
    state. A failed CUDA submission poisons the cohort.
    """

    def __init__(self, shape: SpecShape, *, narrower: tuple[SpecShape, ...] = ()):
        if not shape.depth:
            raise ValueError("fixed DSpark verification needs a positive depth")
        self.shape = shape
        self.bucket, self.depth, self.head = shape.bucket, shape.depth, shape.head
        self.shapes = (*narrower, shape)
        if narrower and (tuple(s.bucket for s in self.shapes) != tuple(range(1, shape.bucket + 1))
                         or any(s.spec is not shape.spec or s.head is not shape.head
                                or s.depth != shape.depth or s.index_context != shape.index_context
                                for s in narrower)):
            raise ValueError("DSpark catalogue requires consecutive widths and shared device state")
        self.next_sequence = 0
        self.pending: tuple[int, tuple[int, ...]] | None = None
        self._host_slots = np.full(shape.bucket, shape.scratch, dtype=np.int32)
        self._host_counts = np.full(shape.bucket, -1, dtype=np.int64)
        self._slots = torch.from_numpy(self._host_slots).to(shape.target.device, copy=True)
        self._counts = torch.from_numpy(self._host_counts).to(shape.target.device, copy=True)
        # Warm the separately aligned serving ABI with no live destination.
        shape.spec.commit(self._slots, self._counts)

    def release_graphs(self) -> None:
        """Release drained captures while retaining per-slot draft context."""
        if self.pending is not None:
            raise RuntimeError("cannot release DSpark graphs with a pending verification")
        for shape in self.shapes:
            shape.graph = shape._graph_out = None

    def capture(self) -> None:
        stream = torch.cuda.Stream()
        for shape in self.shapes:
            # Serving can replay the widths in any order. Each graph owns
            # its private temporary storage, while its inputs share owners.
            shape.capture(torch.cuda.graph_pool_handle(), stream)
        torch.cuda.synchronize()

    def verify_batch(self, sequence: int, entries: tuple[VerifyEntry, ...]) -> VerifyOutput:
        count = len(entries)
        if (self.pending is not None or type(sequence) is not int or sequence != self.next_sequence
                or not 1 <= count <= self.bucket
                or any(type(entry) is not VerifyEntry or entry.pending is not None for entry in entries)):
            raise ValueError("invalid fixed DSpark verify sequence or owner count")
        shape = self.shapes[count - 1] if len(self.shapes) > 1 else self.shape
        output = shape.run(entries)
        self.pending = sequence, tuple(entry.slot for entry in entries)
        self.next_sequence += 1
        return output

    def commit_batch(self, sequence: int, decisions: tuple[tuple[int, int], ...]) -> None:
        if (self.pending is None or type(sequence) is not int or len(decisions) != len(self.pending[1])
                or any(type(slot) is not int or type(count) is not int or not 0 <= count <= self.shape.rows
                       for slot, count in decisions)):
            raise ValueError("invalid fixed DSpark commit owners or consumed counts")
        slots, counts = zip(*decisions, strict=True)
        if self.pending != (sequence, slots):
            raise ValueError("fixed DSpark commit must settle the pending slots in frame order")
        n = len(slots)
        self._host_slots.fill(self.shape.scratch)
        self._host_counts.fill(-1)
        self._host_slots[:n] = slots
        self._host_counts[:n] = counts
        self._slots.copy_(torch.from_numpy(self._host_slots), non_blocking=True)
        self._counts.copy_(torch.from_numpy(self._host_counts), non_blocking=True)
        self.shape.commit_checked(self._slots, self._counts)
        self.pending = None

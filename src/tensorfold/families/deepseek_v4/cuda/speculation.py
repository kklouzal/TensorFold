"""Target-side DeepSeek-V4.1 state for a DSpark verify block.

Keep the anchor and accepted draft inputs by committing their count. Engram and
C2 compute per-count candidates without overwriting the committed state.
The window ring restores the overwritten rows for every rejected input, even
when a verify block wraps. Compressed and index rows beyond the accepted end
remain unreachable by causal selection. The next forward writes those rows
before their first causal read, including ratio-2 rows completed from the
accepted C2 partial.

The engine serializes snapshot, layer backup/write and commit on the same
stream. Each live slot occurs once within a verify or commit batch. The
engine's pending-verify state chooses at most one nonnegative commit per
block, before any consumer or slot reuse. Zero rejects the entire block, and
a negative count skips the slot. The owner and its original ring allocations
outlive all graph replays that use their addresses.
"""

from __future__ import annotations

import torch

from .kernels.kv_pack import C_DIM, ROTOR_ROW
from .kernels.mu_packed import RING
from .kernels.mu_state import MuSpeculation, _mu_c2_candidates, _mu_history_candidates, _mu_snapshot_lengths
from .kernels.triton_utils import jit, tl


MAX_ROWS = 8  # anchor and up to five drafts, with room for the power-of-two verify block
_WORD_BYTES = 4


@jit
def _ring_backup(ring, backup, saved, positions, valid, token_req, qsl, req_slot, S: tl.constexpr,
                 REQUESTS: tl.constexpr, K: tl.constexpr, RING_ROWS: tl.constexpr, ROW_WORDS: tl.constexpr,
                 BLOCK: tl.constexpr, RECORD_POSITIONS: tl.constexpr):
    t = tl.program_id(0)
    r = tl.load(token_req + t).to(tl.int64)
    request = (r >= 0) & (r < REQUESTS)
    start = tl.load(qsl + r, request, 0).to(tl.int64)
    slot = tl.load(req_slot + r, request, -1).to(tl.int64)
    position = tl.load(positions + t).to(tl.int64)
    j = t - start
    owned = request & (slot >= 0) & (slot < S) & (j >= 0) & (j < K)
    live = owned & (position >= 0) & tl.load(valid + t)
    if RECORD_POSITIONS:
        tl.store(saved + slot * K + j, tl.where(live, position, -1), owned)
    col = tl.arange(0, BLOCK)
    copy = live & (col < ROW_WORDS)
    data = tl.load(ring + (slot * RING_ROWS + position % RING_ROWS) * ROW_WORDS + col, copy, 0)
    tl.store(backup + (slot * K + j) * ROW_WORDS + col, data, copy)


@jit
def _ring_restore(addresses, backup, saved, lengths, slots, counts, S: tl.constexpr,
                  K: tl.constexpr, RING_ROWS: tl.constexpr, ROW_WORDS: tl.constexpr, BLOCK: tl.constexpr):
    request, layer, j = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    slot = tl.load(slots + request).to(tl.int64)
    count = tl.load(counts + request).to(tl.int64)
    live = (slot >= 0) & (slot < S) & (count >= 0)
    n = tl.load(lengths + slot, live, 0)
    rejected = live & (j >= count) & (j < n) & (n <= K)
    position = tl.load(saved + slot * K + j, rejected, -1)
    if position >= 0:
        ring = tl.multiple_of(tl.load(addresses + layer).to(tl.pointer_type(tl.int32)), 16)
        col = tl.arange(0, BLOCK)
        row = (layer * (S + 1) + slot) * K + j
        data = tl.load(backup + row * ROW_WORDS + col, col < ROW_WORDS, 0)
        tl.store(ring + (slot * RING_ROWS + position % RING_ROWS) * ROW_WORDS + col, data, col < ROW_WORDS)


class Speculation(MuSpeculation):
    """Per-slot verify candidates and backups of the target's separately owned window rings.

    Each ring is a zero-copy view of its original storage. Its address is
    recorded once after the backing tensors are allocated. The pool may clear
    and write those tensors in place, and may not replace or resize them while
    this owner is live. Forward calls back up layer zero before other layers.
    """

    def __init__(self, target):
        hs, c2 = target.hash_state, target.c2
        rings = tuple(target.pool.rings)
        if not rings:
            raise ValueError("DSpark requires one window ring per target layer")
        first = rings[0]
        dev = hs.history.device
        width = ROTOR_ROW if first.dtype == torch.uint8 else C_DIM
        if (dev.type != "cuda" or first.dtype not in (torch.float16, torch.uint8)
                or any(ring.device != dev or ring.dtype != first.dtype or not ring.is_contiguous()
                       or ring.shape != (hs.slots + 1, RING, width) for ring in rings)):
            raise ValueError("DSpark requires matching contiguous FP16 or H64 window rings on the target device")
        addresses = tuple(ring.data_ptr() for ring in rings)
        ordered = sorted(addresses)
        if any(addr <= 0 or addr % 16 for addr in addresses) or any(
            left + first.nbytes > right for left, right in zip(ordered, ordered[1:])
        ):
            raise ValueError("DSpark window rings must be aligned and disjoint")

        super().__init__(hs, c2.values(), slots=hs.slots, max_inputs=MAX_ROWS)
        self._c2_indices = {layer: index for index, layer in enumerate(c2)}
        # Copy storage words so FP16 values and the H64 scale tail retain their original bits.
        self._rings = tuple(ring.view(torch.int32) for ring in rings)
        self._row_words = self._rings[0].shape[-1]
        self._block = 1 << (self._row_words - 1).bit_length()
        self._addresses = torch.tensor(addresses, dtype=torch.int64, device=dev)
        self._saved = torch.full((self.slots + 1, MAX_ROWS), -1, dtype=torch.int64, device=dev)
        self._backup = torch.empty(len(rings), self.slots + 1, MAX_ROWS, self._row_words,
                                   dtype=torch.int32, device=dev)

    @staticmethod
    def state_bytes(slots: int, layers: int, c2_layers: int, row_bytes: int) -> int:
        """Exact device payload for `slots` real slots plus the scratch slot."""
        candidates = MAX_ROWS + 2
        per_slot = 8 + candidates * (4 * 8 + c2_layers * (2 * 512 * 4 + 8))
        per_slot += MAX_ROWS * (8 + layers * row_bytes)
        return (slots + 1) * per_slot + layers * 8

    @property
    def nbytes(self) -> int:
        return self.state_bytes(self.slots, len(self._rings), len(self._c2_indices), self._row_words * _WORD_BYTES)

    def snapshot_hash(self, step) -> None:
        if not step.num_reqs:
            return
        hs = self.hash_state
        _mu_snapshot_lengths[(step.num_reqs,)](step.qsl, step.req_slot, step.token_req, self.cand_n, S=self.slots)
        _mu_history_candidates[(step.num_reqs, self.J)](
            step.ids, step.positions, step.valid, step.qsl, step.req_slot, step.token_req, hs.token_map, hs.history,
            self.history, S=self.slots, VOCAB=hs.token_map.numel(), J=self.J)

    def snapshot_c2(self, layer: int, values, scores, step) -> None:
        if not step.num_reqs:
            return
        index = self._c2_indices[layer]
        _mu_c2_candidates[(step.num_reqs, self.J)](
            values.contiguous(), scores.contiguous(), step.positions, step.qsl, step.req_slot, step.token_req,
            self.c2_states[index].partial, self.partials[index], self.positions[index],
            S=self.slots, J=self.J, D=C_DIM, BLOCK=C_DIM)

    def backup_ring(self, layer: int, step) -> None:
        """Copy the complete stored rows before this layer inserts its verify window."""
        tokens = step.positions.shape[0]
        if tokens:
            _ring_backup[(tokens,)](
                self._rings[layer], self._backup[layer], self._saved, step.positions, step.valid,
                step.token_req, step.qsl, step.req_slot, S=self.slots, REQUESTS=step.num_reqs,
                K=MAX_ROWS, RING_ROWS=RING, ROW_WORDS=self._row_words, BLOCK=self._block,
                RECORD_POSITIONS=layer == 0)

    def commit(self, slots: torch.Tensor, counts: torch.Tensor) -> None:
        """Commit accepted input counts for the slots' pending verify blocks."""
        e = slots.shape[0]
        if not e:
            return
        super().commit(slots, counts, e)
        _ring_restore[(e, len(self._rings), MAX_ROWS)](
            self._addresses, self._backup, self._saved, self.cand_n, slots, counts, S=self.slots,
            K=MAX_ROWS, RING_ROWS=RING, ROW_WORDS=self._row_words, BLOCK=self._block)

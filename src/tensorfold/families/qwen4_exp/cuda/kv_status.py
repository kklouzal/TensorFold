"""Bounded status collection for a tensor-parallel FP16 decode cohort."""

from __future__ import annotations

import torch
import numpy as np

from .state import CAND, KVNumericError

_CANDIDATE_OWNER_HEADER_BYTES = np.empty((0, 0, 0), dtype=np.float32).__sizeof__()


def workspace_bytes(slots: int, world: int, *, candidate_rows: int = 0, numeric_words: int = 0) -> tuple[int, int]:
    """Device and host commitment, including one escaped owned candidate copy."""

    if type(slots) is not int or slots < 1 or type(world) is not int or world < 2:
        raise ValueError("KV status workspace requires positive slots and at least two ranks")
    if type(candidate_rows) is not int or candidate_rows < 0:
        raise ValueError("KV candidate workspace requires nonnegative admitted rows")
    if type(numeric_words) is not int or numeric_words not in (0, 2):
        raise ValueError("numeric status header must contain exactly zero or two words")
    candidates = 4 * world * candidate_rows * (2 * CAND + 1)
    owner = candidates + _CANDIDATE_OWNER_HEADER_BYTES if candidate_rows else 0
    return (8 * slots * (world + 1) + candidates + 4 * numeric_words * (world + 1),
            8 * slots * world + candidates + owner + 4 * numeric_words * world)


class KVStatusCohort:
    """One decoder worker owns the buffers through each matched gather/readback.

    The rank-identical sequence order comes from the existing TP scheduler.
    Every rank completes the same gather before numeric errors are latched.
    Local failures retain precedence, followed by sequence/rank peer order,
    exactly as the ordinary per-sequence validation path.
    """

    def __init__(self, slots: int, comm, device, *, candidate_rows: int = 0, numeric=None) -> None:
        self.numeric, self.numeric_words = numeric, 0 if numeric is None else 2
        self.comm = comm
        device_bytes, host_bytes = workspace_bytes(slots, comm.world, candidate_rows=candidate_rows, numeric_words=self.numeric_words)
        if type(comm.rank) is not int or not 0 <= comm.rank < comm.world:
            raise ValueError("KV status workspace requires an admitted rank")
        self.slots, self.world, self.rank = slots, comm.world, comm.rank
        self.candidate_rows = candidate_rows
        self.device = torch.device(device)
        self.send = torch.empty(slots * 2 + self.numeric_words, dtype=torch.int32, device=self.device)
        self.device = self.send.device
        words = (slots * 2 + self.numeric_words) * comm.world + comm.world * candidate_rows * (2 * CAND + 1)
        self.receive = torch.empty(words, dtype=torch.int32, device=self.device)
        self.host = torch.empty(words, dtype=torch.int32,
                                pin_memory=self.device.type == "cuda")
        self.device_bytes, self.host_bytes = device_bytes, host_bytes
        self.pinned_host_bytes = self.host.nbytes
        self.owned_candidate_bytes = host_bytes - self.pinned_host_bytes
        self._readback_error = None

    def _gather(self, states, comm):
        if self._readback_error is not None:
            raise RuntimeError("failed KV readback owner requires retirement") from self._readback_error
        if self.numeric is not None and (comm is not self.comm or self.numeric.comm is not comm
                                         or self.numeric.cohort is not self or self.numeric_words != 2):
            raise ValueError("numeric frame requires its original admitted cohort/communicator")
        n = len(states)
        if not 1 <= n <= self.slots or comm is None or (comm.rank, comm.world) != (self.rank, self.world):
            raise ValueError("KV status cohort differs from its admitted slots/rank/world")
        if any(getattr(st, "execution_dtype", None) != torch.float16 or st.kv_status is None
               or st.kv_status.shape != (2,) or st.kv_status.dtype != torch.int32
               or st.kv_status.device != self.device for st in states):
            raise ValueError("KV status cohort requires matching FP16 sequence status buffers")
        try:
            for row, st in enumerate(states):
                self.send[row * 2:row * 2 + 2].copy_(st.kv_status)
            count = n * 2 + self.numeric_words
            if self.numeric is not None:
                self.numeric.check_alive()
                for role, value in enumerate(self.numeric.status_words):
                    self.send[n * 2 + role:n * 2 + role + 1].copy_(value)
            words = self.receive[:self.world * count]
            source = self.send[:count] if self.numeric_words else self.send[:count].view(n, 2)
            comm.all_gather(source, words)
        except BaseException as error:
            self._readback_error = error
            if self.numeric is not None:
                self.numeric.failure = error
            raise
        return words.numel()

    def _readback(self, count):
        # A failed enqueue/wait does not prove that pinned or device storage
        # may be reused. Keep the original failure and its owner until teardown.
        try:
            self.host[:count].copy_(self.receive[:count], non_blocking=self.device.type == "cuda")
            if self.device.type == "cuda":
                torch.cuda.current_stream(self.device).synchronize()
        except BaseException as error:
            self._readback_error = error
            if self.numeric is not None:
                self.numeric.failure = error
            raise

    def check(self, states, *, comm) -> None:
        count = self._gather(states, comm)
        self._readback(count)
        self._check_packet(states, count)

    def read_feedback(self, states, feedback, *, comm):
        """One status-first fence for compact owned int32 target decisions."""
        if (not isinstance(feedback, torch.Tensor) or feedback.dtype != torch.int32
                or feedback.device != self.device or not feedback.is_contiguous() or feedback.ndim != 1
                or not 0 < feedback.numel() <= self.world * self.candidate_rows * (2 * CAND + 1)):
            self.check(states, comm=comm)
            raise ValueError("target feedback differs from its admitted int32 readback envelope")
        status = self._gather(states, comm)
        count = feedback.numel()
        try:
            self.receive[status:status + count].copy_(feedback)
        except BaseException as error:
            self._readback_error = error
            if self.numeric is not None:
                self.numeric.failure = error
            raise
        self._readback(status + count)
        self._check_packet(states, status)
        return self.host[status:status + count].numpy().copy()

    def _check_packet(self, states, count):
        n = len(states)
        blocks = self.host[:count].view(self.world, n * 2 + self.numeric_words)
        peers = blocks[:, :n * 2].reshape(self.world, n, 2).tolist()
        numeric_error = None
        if self.numeric is not None:
            try:
                self.numeric.check_received(blocks[:, n * 2:].tolist())
            except BaseException as error:
                numeric_error = error
        # Preserve existing KV precedence while retaining any numeric poison.
        try:
            self.check_received(states, peers)
        except BaseException as primary:
            if numeric_error is not None:
                from tensorfold.cleanup import raise_failures
                raise_failures(primary, [numeric_error])
            raise
        if numeric_error is not None:
            raise numeric_error

    def check_received(self, states, peers) -> None:
        """Validate an owned, completed frame packet without another readback."""
        if len(peers) != self.world or any(len(row) != len(states) for row in peers):
            raise ValueError("completed KV status packet differs from the cohort's rank/sequence geometry")
        first = None
        for st, (bits, layer) in zip(states, peers[self.rank]):
            try:
                st.kv_check_words(bits, layer)
            except KVNumericError as error:
                if first is None:
                    first = error
        for row, st in enumerate(states):
            for rank, rank_words in enumerate(peers):
                bits, layer = rank_words[row]
                if bits:
                    if first is None:
                        first = KVNumericError(f"K={st.kv_key_dtype}, V={st.kv_value_dtype} numeric validation failed on rank {rank}: "
                                               f"bits={bits}, layer={'MTP' if layer == -1 else layer}, pos={st.pos}; "
                                               "this sequence requires reset")
                    st._kv_error = first
                    break
        if first is not None:
            raise first

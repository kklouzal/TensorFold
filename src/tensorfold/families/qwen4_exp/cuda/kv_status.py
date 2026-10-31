"""Bounded status collection for a tensor-parallel FP16 decode cohort."""

from __future__ import annotations

import torch

from .state import KVNumericError


def workspace_bytes(slots: int, world: int) -> tuple[int, int]:
    """Retained device and pinned host bytes, independent of context capacity."""

    if type(slots) is not int or slots < 1 or type(world) is not int or world < 2:
        raise ValueError("KV status workspace requires positive slots and at least two ranks")
    return 8 * slots * (world + 1), 8 * slots * world


class KVStatusCohort:
    """One decoder worker owns the buffers through each matched gather/readback.

    The rank-identical sequence order comes from the existing TP scheduler.
    Every rank completes the same gather before numeric errors are latched.
    Local failures retain precedence, followed by sequence/rank peer order,
    exactly as the ordinary per-sequence validation path.
    """

    def __init__(self, slots: int, comm, device) -> None:
        device_bytes, host_bytes = workspace_bytes(slots, comm.world)
        if type(comm.rank) is not int or not 0 <= comm.rank < comm.world:
            raise ValueError("KV status workspace requires an admitted rank")
        self.slots, self.world, self.rank = slots, comm.world, comm.rank
        self.device = torch.device(device)
        self.send = torch.empty((slots, 2), dtype=torch.int32, device=self.device)
        self.device = self.send.device
        self.receive = torch.empty(slots * comm.world * 2, dtype=torch.int32, device=self.device)
        self.host = torch.empty(slots * comm.world * 2, dtype=torch.int32,
                                pin_memory=self.device.type == "cuda")
        self.device_bytes, self.host_bytes = device_bytes, host_bytes

    def check(self, states, *, comm) -> None:
        n = len(states)
        if not 1 <= n <= self.slots or comm is None or (comm.rank, comm.world) != (self.rank, self.world):
            raise ValueError("KV status cohort differs from its admitted slots/rank/world")
        if any(getattr(st, "execution_dtype", None) != torch.float16 or st.kv_status is None
               or st.kv_status.shape != (2,) or st.kv_status.dtype != torch.int32
               or st.kv_status.device != self.device for st in states):
            raise ValueError("KV status cohort requires matching FP16 sequence status buffers")
        for row, st in enumerate(states):
            self.send[row].copy_(st.kv_status)
        words = self.receive[:self.world * n * 2]
        comm.all_gather(self.send[:n], words)
        host = self.host[:words.numel()]
        host.copy_(words, non_blocking=self.device.type == "cuda")
        if self.device.type == "cuda":
            torch.cuda.current_stream(self.device).synchronize()
        peers = host.view(self.world, n, 2).tolist()
        self.check_received(states, peers)

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

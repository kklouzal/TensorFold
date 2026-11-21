# SPDX-License-Identifier: Apache-2.0
# Provenance: friend overlay patch 85 (SPDX Apache-2.0), snapshot lp-snap-c1es8-numa-admit2-imgtext-share2-fix5-ram4-stepbud-ram3-memfix-ctx1m-ppovl-opt2-pfx-1-eng-pfxfix4,
#   patches/ds41mu_files/local_mu_model.py (sha256 b5ee701db3a4d421374d76b99f2ca7c5b9956fab211ec5fa0b30e763690b52bd).
#   (restore_slot: patch 90's MuContext.restore_slot in the same file.)
# TensorFold changes:
# - Only the pure torch helpers are vendored: SlotTable, MuStep, split_requests, and MuContext.restore_slot
#   restated as a function over explicit owners (hash_state: MuEngramHashState or None; c2_states: the
#   MuC2State list). Logic verbatim.
# - VLLM_LOCAL_DS41_MU_DECODE_MAX at its production value (unset: 8).
# - MuStep.__slots__ listed in sorted order (lint only).
# - Not vendored (vLLM glue): enabled, admission_errors, MuContext (bind/begin/hash/postload/engram stats,
#   pinned host ring), build_metadata_view, _compress, _prepare_keys, _select, _attend, attention_forward.
"""Multi-request DeepSeek-V4.1 slot bookkeeping (patch 85, torch only).

A request owns one state slot in [0, capacity) from its first schedule until it finishes or is preempted;
slot `capacity` is the scratch row every padding / dummy row maps to. MuStep is the per-forward batch view
every layer shares; split_requests picks the requests that take the single-request (prefill) indexer path.
"""

import heapq

import torch

LOG = "LOCAL-PATCH 85"
DECODE_MAX = 8


class SlotTable:
    """req_id -> slot in [0, capacity); `capacity` itself is the scratch slot.

    The lowest free slot is assigned first (deterministic across PP stages and
    restarts). A slot is released when its request finishes or is preempted;
    a request that is merely not scheduled this step (PP micro-batches,
    budget) keeps it.
    """

    def __init__(self, capacity):
        if capacity < 1:
            raise ValueError(f"{LOG}: slot capacity must be >= 1")
        self.capacity = capacity
        self.scratch = capacity
        self._free = list(range(capacity))
        heapq.heapify(self._free)
        self._slots = {}

    def __len__(self):
        return len(self._slots)

    def get(self, req_id):
        return self._slots.get(req_id)

    def assign(self, req_id):
        """(slot, fresh): fresh is True when the slot was newly taken."""
        slot = self._slots.get(req_id)
        if slot is not None:
            return slot, False
        if not self._free:
            raise RuntimeError(
                f"{LOG}: more than {self.capacity} live V4.1 requests "
                "(slot table full; max_num_seqs mismatch?)"
            )
        slot = heapq.heappop(self._free)
        self._slots[req_id] = slot
        return slot, True

    def release(self, req_id):
        slot = self._slots.pop(req_id, None)
        if slot is not None:
            heapq.heappush(self._free, slot)
        return slot


class MuStep:
    """Per-forward batch view shared by every layer (set by MuContext.begin)."""

    __slots__ = ("decode_index", "num_reqs", "prefill", "qsl", "qsl_cpu", "token_req", "valid")

    def __init__(self, token_req, qsl, qsl_cpu, num_reqs, valid, prefill, decode_index):
        self.token_req, self.qsl, self.qsl_cpu = token_req, qsl, qsl_cpu
        self.num_reqs, self.valid = num_reqs, valid
        # prefill: [(r, s0, s1)] requests on the single-request indexer path;
        # decode_index: device rows of the other tokens (None = all rows).
        self.prefill, self.decode_index = prefill, decode_index


def split_requests(qsl_cpu, num_reqs, decode_max=DECODE_MAX):
    """Requests with more than decode_max scheduled tokens: [(r, s0, s1)]."""
    q = qsl_cpu[: num_reqs + 1].tolist()
    return [(r, q[r], q[r + 1]) for r in range(num_reqs) if q[r + 1] - q[r] > decode_max]


def restore_slot(hash_state, c2_states, slot, cached, token_ids):
    """Slot state after `cached` accepted tokens of `token_ids` (patch 90).

    Engram history: the last three compressed ids before `cached` (-1
    before the sequence start) and the committed position, exactly what
    _mu_commit_history leaves; C2: an even `cached` closes every group,
    so the partial row is zero and the committed position is `cached`."""
    if cached % 2:
        raise RuntimeError(f"{LOG}: prefix hit at odd position {cached} splits a C2 group")
    if len(token_ids) < cached:
        raise RuntimeError(f"{LOG}: prefix hit {cached} beyond {len(token_ids)} known tokens")
    if hash_state is not None:
        history = hash_state.history
        tail = [token_ids[i] if i >= 0 else -1 for i in range(cached - 3, cached)]
        ids = torch.tensor([max(t, 0) for t in tail], dtype=torch.long, device=history.device)
        mapped = hash_state.token_map[ids].long()
        known = torch.tensor([t >= 0 for t in tail], device=history.device)
        history[slot, :3] = torch.where(known, mapped, torch.full_like(mapped, -1))
        history[slot, 3] = cached
    for state in c2_states:
        state.partial[slot].zero_()
        state.committed[slot] = cached

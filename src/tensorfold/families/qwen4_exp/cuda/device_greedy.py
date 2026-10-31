"""Qualified single-stream greedy MTP handoff through existing bound graphs.

The capability is off unless prepared weights explicitly qualify it. Original
sampling and every nonzero-confidence or concurrent path remain the authority.
No proposal or provisional MTP position is published before all frame words
have completed their one readback and passed ordinary numeric validation.
"""

from __future__ import annotations

import copy

import numpy as np
import torch

from tensorfold.cuda.sampling import valid_scores, validate_policy

from .state import CAND, KVNumericError


WORDS = 5  # FP32 max bits, exact global ID, FP32 full-domain LSE bits, codec bits, layer


def workspace_bytes(width: int, depth: int, world: int) -> tuple[int, int]:
    """Conservative retained tensor bytes, excluding a borrowed immutable ID map."""
    if any(type(value) is not int or value < 1 for value in (width, depth, world)):
        raise ValueError("device greedy workspace requires positive width/depth/world")
    words = depth * (world * WORDS + 1)
    # IDs, equal-score mask and tied IDs; rank score/mask/ID scratch; scalars.
    device = width * (8 + 1 + 8) + world * (4 + 1 + 8) + words * 8 + WORDS * 8 + 256
    return device, words * 8


def recorded_token(frame: np.ndarray, observed: int) -> int:
    """Check a direct original-shard maximum without any top-k truncation."""
    if frame.ndim != 2 or frame.shape[1] != WORDS or frame.dtype != np.int64:
        raise ValueError("complete rank-ordered greedy frame words required")
    scores = frame[:, 0].astype(np.int32).view(np.float32)
    ids = frame[:, 1]
    norms = frame[:, 2].astype(np.int32).view(np.float32)
    valid_scores(scores[None])
    if np.isnan(norms).any() or np.isposinf(norms).any() or not np.isfinite(norms).any():
        raise ValueError("a complete draft vocabulary needs finite scores without NaN or positive infinity")
    if (ids < 0).any() or len(np.unique(ids)) != len(ids):
        raise ValueError("greedy maxima require distinct nonnegative global token IDs")
    order = np.lexsort((ids, -scores))
    expected = int(ids[order[0]])
    if int(observed) != expected:
        raise ValueError("device greedy token differs from its original-shard score/global-ID packet")
    return expected


class GreedyHandoff:
    """One graph owner retains fixed buffers until its decoder worker retires."""

    def __init__(self, bound):
        self.bound, self.dec = bound, bound.dec
        self.w, self.plan = self.dec.w, self.dec._draft_plan
        self.depth, self.world = self.dec.depth, self.w.comm.world
        self.busy, self.error = False, None
        self.completed, self.steps, self.readbacks = 0, 0, 0
        self.native = None
        if getattr(self.w, "meta", {}).get("device_greedy_native_ready", False):
            from tensorfold.cuda.sm70.greedy_native import extension

            self.native = extension()
        width, device = self.plan.width, self.w.device
        self.device_bytes, self.host_bytes = workspace_bytes(width, self.depth, self.world)
        self.ids = (self.plan.device_map if self.plan.device_map is not None else
                    torch.arange(width, dtype=torch.int64, device=device) + self.plan.offset)
        self.equal = torch.empty(width, dtype=torch.bool, device=device)
        self.tied = torch.empty(width, dtype=torch.int64, device=device)
        self.maximum = torch.empty(1, dtype=torch.float32, device=device)
        self.column = torch.empty(1, dtype=torch.int64, device=device)
        self.local_id = torch.empty(1, dtype=torch.int64, device=device)
        self.sentinel = torch.full((), torch.iinfo(torch.int64).max, dtype=torch.int64, device=device)
        self.zero = torch.zeros((), dtype=torch.int64, device=device)
        self.rank_words = torch.empty(self.world, dtype=torch.int32, device=device)
        self.rank_scores = self.rank_words.view(torch.float32)
        self.rank_equal = torch.empty(self.world, dtype=torch.bool, device=device)
        self.rank_ids = torch.empty(self.world, dtype=torch.int64, device=device)
        self.input_token = torch.empty(1, dtype=torch.int32, device=device)
        self.safe_id = torch.empty(1, dtype=torch.int64, device=device)
        self.low, self.high = (torch.empty(1, dtype=torch.bool, device=device) for _ in range(2))
        self.send = torch.empty(WORDS, dtype=torch.int64, device=device)
        size = self.depth * (self.world * WORDS + 1)
        self.storage = torch.empty(size, dtype=torch.int64, device=device)
        self.frames = self.storage[:self.depth * self.world * WORDS].view(self.depth, self.world, WORDS)
        self.tokens = self.storage[self.depth * self.world * WORDS:]
        self.host = torch.empty(size, dtype=torch.int64, pin_memory=True)

    def eligible(self, streams, new):
        dec = self.dec
        if (self.error is not None or len(streams) != 1 or dec.confidence != 0 or self.depth < 1
                or dec.w.mtp is None or dec.w.mtp.layer.linear or dec.kv_cohort is None):
            return None
        stream, at, keep = streams[0]
        state, sampling = stream.st, stream.sampling
        count = min(self.depth, stream.count - len(stream.out) - len(keep))
        if count < 1:
            return None
        try:
            validate_policy([state.pos + 1 + step for step in range(count)], sampling)
        except (TypeError, ValueError, OverflowError):
            # The ordinary path remains the validation authority, including
            # its codec-status precedence on a failed producing MTP step.
            return None
        if (getattr(stream, "copies", None) is not None or getattr(stream, "constraint", None) is not None
                or sampling is not None and sampling.temperature > 0
                or not self.plan.graph_ids_safe or state._kv_error is not None or state._resize_error is not None
                or state._kv_pending or dec._rotor_lookup(stream)):
            return None
        start = state.mtp_len - state.mtp_drafted
        if not self.bound.mtp_ready(state, len(keep), start):
            return None
        if any(not self.bound.mtp_ready(state, 1, start + len(keep) + step - 1) for step in range(1, count)):
            return None
        return stream, at, keep, count, start

    def gather(self, step, logits, state):
        self.plan.check_model(self.w, logits)
        if self.native is not None:
            packet = 2 * CAND + 1
            norm = self.dec.mbuf.cand_all[:self.world * packet].view(self.world, 1, packet)[self.w.comm.rank, 0, -1:]
            frame = self.frames[step]
            self.native.local(logits, self.ids, norm, state.kv_status, self.send)
            self.w.comm.all_gather(self.send, frame.view(-1))
            self.native.global_winner(frame, self.tokens[step:step+1], self.input_token, self.w.cfg.vocab)
            return
        torch.max(logits, dim=1, out=(self.maximum, self.column))
        torch.eq(logits[0], self.maximum, out=self.equal)
        torch.where(self.equal, self.ids, self.sentinel, out=self.tied)
        torch.amin(self.tied, dim=0, keepdim=True, out=self.local_id)
        packet = 2 * CAND + 1
        norm = self.dec.mbuf.cand_all[:self.world * packet].view(self.world, 1, packet)[self.w.comm.rank, 0, -1:]
        self.send[0:1].copy_(self.maximum.view(torch.int32))
        self.send[1:2].copy_(self.local_id)
        self.send[2:3].copy_(norm.view(torch.int32))
        self.send[3:5].copy_(state.kv_status)
        frame = self.frames[step]
        self.w.comm.all_gather(self.send, frame.view(-1))
        self.rank_words.copy_(frame[:, 0])
        torch.max(self.rank_scores, dim=0, keepdim=True, out=(self.maximum, self.column))
        torch.eq(self.rank_scores, self.maximum, out=self.rank_equal)
        torch.where(self.rank_equal, frame[:, 1], self.sentinel, out=self.rank_ids)
        torch.amin(self.rank_ids, dim=0, keepdim=True, out=self.tokens[step:step+1])
        # An invalid score frame must not turn its sentinel ID into an unsafe
        # embedding access in a later, unpublished step. The CPU rejects it.
        torch.ge(self.tokens[step:step+1], 0, out=self.low)
        torch.lt(self.tokens[step:step+1], self.w.cfg.vocab, out=self.high)
        torch.logical_and(self.low, self.high, out=self.low)
        torch.where(self.low, self.tokens[step:step+1], self.zero, out=self.safe_id)
        self.input_token.copy_(self.safe_id)

    def finish(self, state, count, start, absorbed):
        self.host.copy_(self.storage, non_blocking=True)
        torch.cuda.current_stream(self.w.device).synchronize()
        self.readbacks = getattr(self, "readbacks", 0) + 1
        host = self.host.numpy()
        frames = host[:self.depth * self.world * WORDS].reshape(self.depth, self.world, WORDS)
        tokens = host[self.depth * self.world * WORDS:]
        result = []
        for step in range(count):
            # The same status checker handles each historical frame. Its view
            # has the correct diagnostic position while real metadata stays
            # committed until every frame has passed.
            view = copy.copy(state)
            view.mtp_len = start if step == 0 else start + absorbed + step - 1
            view._kv_pending = True
            try:
                self.dec.kv_cohort.check_received([view], frames[step, :, 3:5][:, None, :].tolist())
            except KVNumericError:
                state._kv_error = view._kv_error
                raise
            result.append(recorded_token(frames[step], int(tokens[step])))
        state._kv_pending = False
        state.set_mtp_len(start + absorbed + count - 1)
        state.mtp_drafted = count - 1
        return result

    def try_draft(self, streams, new):
        if self.busy:
            raise RuntimeError("a device MTP handoff cannot overlap another use of its buffers")
        selected = self.eligible(streams, new)
        if selected is None:
            return False
        from .mtp import mtp_stage

        stream, at, keep, count, start = selected
        state = stream.st
        self.busy = True
        try:
            stream.drafts = []
            state.set_mtp_len(start)
            state.mtp_drafted = 0
            self.storage.zero_()
            segs = mtp_stage(self.dec.mbuf, [(state, keep, self.dec.buf.streams[at:at+len(keep)])])
            for step in range(count):
                position = start if step == 0 else start + len(keep) + step - 1
                logits = self.dec.graphs.mtp_step(segs, mtp_positions=[position],
                                                device_tokens=self.input_token if step else None)
                if logits is None:
                    raise RuntimeError("an admitted device MTP key disappeared during its operation")
                if self.dec.controller is not None:
                    self.dec.controller.work_observations.mtp(
                        segs, role="refresh" if step == 0 else "proposal", position=position)
                self.gather(step, logits, state)
                if step + 1 < count:
                    row = len(keep) - 1 if step == 0 else 0
                    segs = mtp_stage(self.dec.mbuf, [(state, [0], self.dec.mbuf.streams[row:row+1])])
            stream.drafts = self.finish(state, count, start, len(keep))
            self.completed += 1
            self.steps += count
            return True
        except BaseException as error:
            self.error = error
            if state._kv_error is None:
                state._kv_error = KVNumericError("unvalidated device MTP writes require this sequence to be reset")
            if self.dec.controller is not None:
                self.dec.controller.work_observations.incomplete(error)
            raise
        finally:
            self.busy = False

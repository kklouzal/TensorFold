"""Engine-private DFlash body graphs over one exclusive request lane.

One entry per actual width retains only its current full-attention grid. A
public native MemPool owns each generation: after completed consumers, retire
the graph and output before releasing that pool owner. Implicit dead graph
pools are merely reclaimable and are not used as a retention bound here.

The ordinary DFlash.logits API remains eager and fresh. This private proposer
finishes Chains' existing synchronous readback before returning it, so static
graph logits never escape to an unfinished consumer. No sampling algorithm,
accepted-taps append, attention grid or numerical kernel is replaced.
"""
from __future__ import annotations

import gc
from dataclasses import dataclass

import torch

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.speculative import Chains

from .dflash import FULL_CHUNK


def _propagate(primary, prior, errors, count):
    """Transport only after independent cleanup, retaining raw roots on OOM."""
    if primary is None and count:
        primary = errors[0]
        prior[0] = BaseException.__cause__.__get__(primary)
        prior[1] = BaseException.__context__.__get__(primary)
        prior[2] = BaseException.__suppress_context__.__get__(primary)
    if primary is None:
        return
    BaseException.__cause__.__set__(primary, prior[0])
    BaseException.__context__.__set__(primary, prior[1])
    BaseException.__suppress_context__.__set__(primary, prior[2])
    try:
        del errors[count:]
        raise_failures(primary, errors)
    except BaseException as transport:
        if transport is primary:
            raise
        raise primary from transport


@dataclass(slots=True)
class _Entry:
    full_slots: int
    pool: object = None
    graph: object = None
    output: torch.Tensor | None = None
    output_initialized: bool = False

    def reset_graph(self):
        if self.graph is not None:
            self.graph.reset()
            self.graph = None

    def retire(self):
        # Only after the lane's consumers completed. reset is NOT a wait.
        self.reset_graph()
        self.output = None
        self.output_initialized = False
        # MemPool::~MemPool invokes pool-specific emptyCache. All captured
        # tensor aliases must be gone first; a bare pool-ID tuple is no owner.
        self.pool = None


class DraftBodyGraphs:
    """Private, early-bound proposer for a prepared FP8/row5 native lane.

    The engine serializes calls on ``stream``. Reset/restore update the same
    DraftState buffers in place; a new model/state needs a new owner. Captures
    write proposal noise beyond committed length and never advance it. Each
    normal call completes GPU readback before another entry can be retired.
    Capture/replay/readback failures poison this owner; they never use eager
    recovery. A failed entry stays owned through a successful drain and the
    closed model's lifetime because exception frames may retain its tensors.
    """

    def __init__(self, head, state, stream):
        if head.bits != 8 or head.block != 5:
            raise ValueError("private DFlash body graphs require prepared FP8 and five rows")
        self.head, self.state, self.stream = head, state, stream
        cfg = head.config
        # The S arena is lane-owned, not a snapshot/model-global buffer. Only
        # index zero changes; every kernel treats the mask tail as read-only.
        self.ids = torch.full((head.block,), head.mask_id, dtype=torch.int32, device=head.device)
        self.query = torch.empty((head.block, cfg.heads, cfg.head_dim), dtype=torch.bfloat16, device=head.device)
        self.inputs = tuple((self.ids[:rows], self.query[:rows]) for rows in range(2, head.block + 1))
        self.entries: list[_Entry | None] = [None] * (head.block - 1)
        self.failed = self.closed = False
        self._failed_entry = None
        self.captures = self.replays = self.retirements = 0

    def _capture(self, entry, run):
        # Direct public capture avoids the graph context manager's global
        # allocator flush. Its warmup and current nondefault stream were
        # explicitly drained by propose() before entry.
        graph = entry.graph
        begin, end = graph.capture_begin, graph.capture_end
        enabled, enable = gc.isenabled(), gc.enable
        primary, prior, errors, count = None, [None, None, False], [None, None], 0
        try:
            gc.disable()
            try:
                begin(pool=entry.pool, capture_error_mode="thread_local")
            except BaseException as error:  # noqa: BLE001 - partial acquisition still needs GC restoration.
                primary = error
                prior[0] = BaseException.__cause__.__get__(error)
                prior[1] = BaseException.__context__.__get__(error)
                prior[2] = BaseException.__suppress_context__.__get__(error)
            else:
                try:
                    entry.output = run()
                except BaseException as error:  # noqa: BLE001 - retain the body root before capture retirement.
                    primary = error
                    prior[0] = BaseException.__cause__.__get__(error)
                    prior[1] = BaseException.__context__.__get__(error)
                    prior[2] = BaseException.__suppress_context__.__get__(error)
                try:
                    end()
                except BaseException as error:  # noqa: BLE001 - retirement cannot replace the body failure.
                    if primary is None:
                        primary = error
                        prior[0] = BaseException.__cause__.__get__(error)
                        prior[1] = BaseException.__context__.__get__(error)
                        prior[2] = BaseException.__suppress_context__.__get__(error)
                    else:
                        errors[count], count = error, count + 1
        except BaseException as error:  # noqa: BLE001 - setup interruption still needs independent restoration.
            primary = error
            prior[0] = BaseException.__cause__.__get__(error)
            prior[1] = BaseException.__context__.__get__(error)
            prior[2] = BaseException.__suppress_context__.__get__(error)
        finally:
            if enabled:
                try:
                    enable()
                except BaseException as error:  # noqa: BLE001 - report GC restoration with the capture outcome.
                    if primary is None:
                        primary = error
                        prior[0] = BaseException.__cause__.__get__(error)
                        prior[1] = BaseException.__context__.__get__(error)
                        prior[2] = BaseException.__suppress_context__.__get__(error)
                    else:
                        errors[count], count = error, count + 1
        _propagate(primary, prior, errors, count)

    @torch.no_grad()
    def propose(self, state, anchor: int, count: int, history=None, *, sampled=False, sampling=None):
        if state is not self.state:
            raise ValueError("a private DFlash graph belongs to one model and DraftState")
        if self.closed or self.failed:
            raise RuntimeError("private DFlash graph owner is closed or poisoned; retire the model")
        head, cfg = self.head, self.head.config
        # Keep the public validation/no-op order: no anchor check or device
        # work when no block can be proposed.
        if type(count) is not int or not 0 <= count <= 127:
            raise ValueError("Gemma DFlash proposal count must be an integer from zero through127")
        length = min(head.block, count + 1, head.window - state.length)
        if length < 2 or state.length == 0:
            return None
        if type(anchor) is not int or not 0 <= anchor < cfg.vocab:
            raise ValueError("Gemma DFlash anchors must be native target-vocabulary token IDs")
        full_slots = (state.length + length - 1) // FULL_CHUNK + 1
        index = length - 2
        previous = self.entries[index]
        missing = previous is None or previous.full_slots != full_slots
        entry = _Entry(full_slots) if missing else previous
        ids, query = self.inputs[index]
        if missing:
            def run():
                return head._logits_body(state, head.raw_embed_tokens(ids), full_slots=full_slots, query=query)

        try:
            if previous is not None and missing:
                self._failed_entry = previous
                previous.retire()
                self.entries[index] = None
                self.retirements += 1
            self._failed_entry = entry
            if missing:
                # Publish owners before acquiring CUDA resources, so any
                # partially acquired entry survives an operation failure.
                self.entries[index] = entry
                entry.pool = torch.cuda.MemPool()
                entry.graph = torch.cuda.CUDAGraph()
            # These are dynamic inputs, never host constants frozen into a
            # captured fill/copy. KV and the position scalar keep fixed storage.
            ids[0] = anchor
            state.pos.fill_(state.length)
            if missing:
                # This is already the actual ordinary proposal for the real
                # prefix/anchor. Keep its default-pool Tensor strong through
                # capture and Chains readback. Capture records nodes; it does
                # not execute another body or initialize entry.output.
                logits = run()
                self.stream.synchronize()
                self._capture(entry, run)
                self.captures += 1
            else:
                entry.graph.replay()
                self.replays += 1
                logits = entry.output
            chains = Chains(logits, [history], length - 1, owners=entry,
                            sampled=sampled, sampling=sampling)
            # Engine callers immediately finish/sample this same policy and
            # history. Complete their existing readback here, before a static
            # result can escape or a later width/grid replaces its entry.
            chains._readback()
            if not missing:
                entry.output_initialized = True  # a completed first replay, never capture alone
        except BaseException:
            self.failed = True
            raise
        self._failed_entry = None
        return chains

    def close_after_drain(self):
        """Called under the engine run lock only after successful lane drain.

        Normal entries release graph, tensor and pool in that order. Failed
        entries keep their tensor/pool owners: raw failure frames can still
        alias those tensors. No new capture is allowed on a failed/closed
        owner, so this quarantine is bounded by the model's width cache and
        ends with model/failure retirement, rather than a retry queue.
        """
        if self.closed:
            return  # never repeat a failed graph teardown as an implicit retry
        self.closed = True
        primary, prior = None, [None, None, False]
        errors, count = [None] * len(self.entries), 0
        for index, entry in enumerate(self.entries):
            if entry is None:
                continue
            try:
                if entry is self._failed_entry:
                    entry.reset_graph()
                else:
                    entry.retire()
                    self.entries[index] = None
            except BaseException as error:  # noqa: BLE001 - attempt every entry and retain all cleanup roots.
                self.failed = True
                if primary is None:
                    primary = error
                    prior[0] = BaseException.__cause__.__get__(error)
                    prior[1] = BaseException.__context__.__get__(error)
                    prior[2] = BaseException.__suppress_context__.__get__(error)
                else:
                    errors[count], count = error, count + 1
        if not self.failed:
            self.inputs = ()
            self.ids = self.query = None
        _propagate(primary, prior, errors, count)

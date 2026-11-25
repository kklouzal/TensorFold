"""Sample routed expert loads and retire one asynchronous history write at a time."""

from concurrent.futures import ThreadPoolExecutor
from random import Random
from time import monotonic

import torch

from tensorfold.cleanup import finish
from .expert_usage import ExpertUsage


class ExpertObserver:
    def __init__(self, device, history, *, sample_every: int, minimum_routes: int):
        self.usage = ExpertUsage(device)
        self.history = history
        self.minimum_routes = minimum_routes
        self.sample_every = sample_every
        self.random = Random(0)
        self.eager_flags = torch.arange(3, dtype=torch.int64, device=device)
        self.worker = ThreadPoolExecutor(1, thread_name_prefix="ds41-expert-history")
        self.write = None
        self.active = False
        self.copy_pending = False
        self.next_poll = monotonic() + 60

    def sample(self, phase: int, *, may_copy: bool) -> int:
        if not self.active:
            return 0
        if may_copy and (now := monotonic()) >= self.next_poll:
            self._advance(now)
        return phase + 1 if self.random.randrange(self.sample_every) == 0 else 0

    def _advance(self, now: float) -> None:
        if self.write is not None:
            if not self.write.done():
                self.next_poll = now + 1
                return
            self.write.result()
            self.write = None
        if self.copy_pending:
            snapshot = self.usage.poll_snapshot()
            if snapshot is None:
                self.next_poll = now + 1
                return
            self.copy_pending = False
            self.write = self.worker.submit(self._record, snapshot)
            self.next_poll = now + 60
        else:
            if not self.usage.request_snapshot():
                raise RuntimeError("expert usage snapshot buffer is already owned")
            self.copy_pending = True
            self.next_poll = now + 1

    def _record(self, snapshot, *, closing=False) -> None:
        if self.history.record(snapshot.tolist()) or closing:
            self.history.flush()
            plan = self.history.recommendation(minimum_routes=self.minimum_routes)
            if plan is not None:
                self.history.publish_recommendation(plan)

    def finish(self) -> None:
        """Called after admission closes and the engine fences its last model work."""
        self.active = False
        finish((self._finish_pending, self.worker.shutdown, self.history.close))

    def _finish_pending(self) -> None:
        if self.write is not None:
            self.write.result()
        if self.copy_pending:
            snapshot = self.usage.poll_snapshot()
            if snapshot is None:
                raise RuntimeError("final expert usage snapshot did not finish behind the device fence")
            self._record(snapshot)
        if not self.usage.request_snapshot():
            raise RuntimeError("final expert usage snapshot buffer is already owned")
        torch.cuda.current_stream(self.usage.counts.device).synchronize()
        snapshot = self.usage.poll_snapshot()
        if snapshot is None:
            raise RuntimeError("final expert usage snapshot did not finish after synchronization")
        self._record(snapshot, closing=True)

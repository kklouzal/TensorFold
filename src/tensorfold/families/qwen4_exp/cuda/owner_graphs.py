"""Bounded captures of the canonical unique-owner attention operation.

Preparation happens before admission, on one GPU per process. Each graph reads
the same stable descriptor/input slabs as uncaptured native execution. Python
preparation/indexer work and target sampling remain outside these partial graphs.
The graph does not own or duplicate a session's history.
"""

from __future__ import annotations

from functools import partial
import time

import torch

from tensorfold.cleanup import error_name, raise_failures
from .owner_execution import GRAPH_FAMILY, KERNEL_FAMILY, row_buckets


class OwnerAttentionGraphs:
    """One serialized workspace and at most logarithmically many graph objects.

    The startup owner must retain this object, the workspace and communicator
    through final replay completion. Capture is never attempted on a serving
    thread or on a cache miss. Captured and uncaptured execution use native_run.
    """

    def __init__(self, workspace, *, rows, budget_bytes, comm, buckets=None):
        self.buckets = row_buckets(rows, buckets)
        if type(budget_bytes) is not int or budget_bytes < 1:
            raise ValueError("owner graph preparation requires a positive admitted graph budget")
        if comm is None or comm.world != 8:
            raise ValueError("owner attention graphs require the canonical TP8 communicator")
        self.workspace, self.comm = workspace, comm
        self.rows, self.budget = rows, budget_bytes
        self.device = workspace.queries.device
        self.side = self.pool = self.votes = None
        self.graphs = {}
        self.pending = []
        self.measurements = []
        self.room_bytes = self.capture_s = 0
        # Physical-only admission remains valid without relying on a particular
        # framework's private-pool snapshot ABI. Never credit unknown spare.
        self.private_spare_bytes = None
        self.allocator_record = {"accounting": "physical-only"}
        self.prepared = self.closed = False
        self.failure = None
        self.hits = self.eager_calls = 0
        self.side = torch.cuda.Stream(device=self.device)
        self.pool = torch.cuda.graph_pool_handle()
        self.votes = torch.empty((1 + comm.world,), dtype=torch.int64, device=self.device)

    def _agree(self, allowed):
        self.votes[:1].fill_(int(allowed))
        self.comm.all_gather(self.votes[:1], self.votes[1:])
        return bool(self.votes[1:].min().item())

    def _same(self, value):
        self.votes[:1].fill_(int(value))
        self.comm.all_gather(self.votes[:1], self.votes[1:])
        return int(self.votes[1:].min().item()) == value and int(self.votes[1:].max().item()) == value

    def bucket(self, actual_rows):
        if type(actual_rows) is not int or not 1 <= actual_rows <= self.rows:
            raise ValueError("owner attention rows exceed the prepared execution envelope")
        return next(bound for bound in self.buckets if actual_rows <= bound)

    def _free(self):
        return int(torch.cuda.mem_get_info(self.device)[0])

    def _capture(self, run):
        graph = torch.cuda.CUDAGraph()
        self.pending.append(graph)  # Journal before any fallible native capture operation.
        graph.capture_begin(pool=self.pool, capture_error_mode="global")
        primary = None
        try:
            run()
        except BaseException as error:
            primary = error
        errors = []
        try:
            graph.capture_end()
        except BaseException as error:
            errors.append(error)
        raise_failures(primary, errors)
        return graph

    def _poison(self, error):
        first = self.failure is None
        if first:
            self.failure = error
        if first:
            from tensorfold.cuda.cohort import notify_failure

            try:
                notify_failure()
            except BaseException as notification:
                raise_failures(error, [notification])

    def prepare(self, *, quiescent):
        """Capture the complete bounded catalogue in a process-wide idle startup scope.

        PyTorch 2.14 forbids unrelated CUDA work during capture and requires
        stable addresses. NCCL capture/replay participation is uniform across
        the whole communicator; the same row bucket is prepared on all ranks.
        """
        started = time.perf_counter()
        try:
            valid = not self.prepared and not self.closed and self.failure is None and callable(quiescent)
            if not self._agree(valid):
                raise RuntimeError("owner attention graph preparation is a one-time startup operation")
            if not self._agree(bool(quiescent())):
                raise RuntimeError("owner attention graphs require globally quiescent startup preparation")
            pair = self.workspace.native_pair
            # Different catalogues could otherwise agree for a common prefix
            # and then enter different collective counts or CUDA graph modes.
            signature = (self.rows, len(self.buckets), *self.buckets, self.workspace.heads, self.workspace.head_dim,
                         self.workspace.selection.shape[1], self.workspace.native_page_tokens,
                         pair.key.bits, pair.key.codec, pair.value.bits, pair.value.codec)
            for value in signature:
                if not self._same(value):
                    raise ValueError("owner attention graph geometry/codecs differ across tensor-parallel ranks")
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            beginning = self._free()
            for rows in self.buckets:
                # Inactive frames never dereference session pointers. Transport
                # inputs are initialized too; no uninitialized padding is sent.
                self.workspace.clear_native_inputs(rows)
                self.workspace.wait_native_inputs()
                run = partial(self.workspace.native_run, rows)
                run()  # Kernel/module/NCCL lazy initialization precedes capture.
                torch.cuda.synchronize(self.device)
                before = self._free()
                self.side.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(self.side):
                    graph = self._capture(run)
                torch.cuda.current_stream(self.device).wait_stream(self.side)
                torch.cuda.synchronize(self.device)
                after = self._free()
                self.graphs[rows] = graph
                self.pending.remove(graph)
                self.room_bytes = max(self.room_bytes, beginning - after, 0)
                self.measurements.append({"rows": rows, "physical_free_before": before,
                                          "physical_free_after": after, "raw_capture_bytes": before - after})
                if not self._agree(self.room_bytes <= self.budget):
                    raise MemoryError("owner attention captures exceed their admitted per-rank graph budget")
            torch.cuda.empty_cache()
            self.room_bytes = max(self.room_bytes, beginning - self._free(), 0)
            if not self._agree(self.room_bytes <= self.budget):
                raise MemoryError("settled owner attention graphs exceed their admitted budget")
            self.capture_s = time.perf_counter() - started
            self._account_pool()
            self.prepared = True
        except BaseException as error:
            # Another rank may be in a captured collective. Its ordinary ACK
            # cannot retire this failure; notify the existing cohort supervisor.
            self._poison(error)
            raise

    def _account_pool(self):
        """Use the selected framework's settled private-pool schema when available."""
        from .service_memory import private_pool_spare

        try:
            backend = torch.cuda.memory.get_allocator_backend()
        except AttributeError:
            return
        self.allocator_record = {"backend": backend, "accounting": "physical-only"}
        if backend != "native":
            return
        try:
            segments = torch.cuda.memory_snapshot(mempool_id=self.pool, include_traces=False)
        except (AttributeError, TypeError):
            # Unsupported snapshot ABI gives no evidence of zero private spare.
            return
        self.private_spare_bytes = private_pool_spare(segments, self.pool, self.device.index or 0)
        self.allocator_record = {"backend": backend, "accounting": "settled-native-private-pool",
                                 "private_reserved_bytes": sum(row["total_size"] for row in segments),
                                 "private_allocated_bytes": sum(row["allocated_size"] for row in segments)}

    def run(self, actual_rows, *, graph):
        """Execute the agreed plan and borrow output until the caller's last read.

        The adapter records mark_native_submitted only after copying/consuming
        this view, so its reuse event covers the actual final workspace reader.
        """
        if self.closed or self.failure is not None:
            raise RuntimeError("owner attention graph workspace is closed or poisoned") from self.failure
        try:
            rows = self.bucket(actual_rows)
            self.workspace.wait_native_inputs()
            if graph:
                if not self.prepared:
                    raise RuntimeError("owner attention graphs were not prepared before admission")
                self.graphs[rows].replay()
                self.hits += 1
            else:
                self.workspace.native_run(rows)
                self.eager_calls += 1
            return self.workspace.native_views(rows)["out"][:actual_rows]
        except BaseException as error:
            self._poison(error)
            raise

    def record(self):
        return {"kernel_family": KERNEL_FAMILY, "graph_family": GRAPH_FAMILY,
                "scope": "partial native owner attention; preparation/indexer/sampling remain outside",
                "prepared": self.prepared, "closed": self.closed,
                "failure_type": None if self.failure is None else error_name(self.failure),
                "rows": list(self.buckets), "graph_keys": len(self.graphs),
                "graph_budget_bytes": self.budget, "observed_preparation_residency_bytes": self.room_bytes,
                "observed_capture_charge_bytes": sum(max(0, row["raw_capture_bytes"]) for row in self.measurements),
                "allocator": dict(self.allocator_record), "private_pool_spare_bytes": self.private_spare_bytes,
                "vote_workspace_bytes": 0 if self.votes is None else self.votes.numel() * self.votes.element_size(),
                "capture_s": self.capture_s, "captures": list(self.measurements),
                "replays": self.hits, "uncaptured_calls": self.eager_calls,
                "qualification": "source candidate; native SM70 correctness and performance gates required"}

    def close(self):
        if self.closed:
            return
        if self.failure is not None:
            raise RuntimeError("failed owner capture retains its native borrowers for terminal cohort retirement") from self.failure
        try:
            torch.cuda.synchronize(self.device)
            errors = []
            for rows, graph in tuple(self.graphs.items()):
                try:
                    graph.reset()
                    del self.graphs[rows]
                except BaseException as error:
                    errors.append(error)
            raise_failures(None, errors)
            self.votes = self.side = self.pool = None
            self.workspace = self.comm = None
            self.closed = True
        except BaseException as error:
            self._poison(error)
            raise

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


def _outcome_slot(error=None):
    """Reserve cold diagnostic ownership before capture or inside its outcome guard."""
    return [error]


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
        self.replays = {}
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
        self.tensor_leases, self.comm_leases = {}, ()
        self._transport = None
        self._lease_resources()
        construction = getattr(comm, "_profile_construction", None)
        if construction is not None:
            construction.bind_protocol(self, "_capture", "owner/capture", refusal=())
        self.side = torch.cuda.Stream(device=self.device)
        self.pool = torch.cuda.graph_pool_handle()
        self.votes = torch.empty((1 + comm.world,), dtype=torch.int64, device=self.device)

    @staticmethod
    def _tensor_signature(tensor, storage):
        return (id(tensor), tensor.data_ptr(), storage.data_ptr(), storage.nbytes(), tuple(tensor.shape),
                tuple(tensor.stride()), tensor.storage_offset(), tensor.dtype, tensor.device)

    @staticmethod
    def _comm_signature(comm):
        if comm.closed or not comm.comm.value:
            raise RuntimeError("owner graphs require live initialized communicators")
        return (id(comm), comm.comm.value, comm.world, comm.rank, comm.device)

    def _lease_resources(self):
        # Retaining a Tensor alone does not retain its old storage after set_().
        # Lease both, and validate the no-resize/no-rebind contract at capture
        # and final model adoption. Addresses remain private runtime checks.
        for name, tensor in self.workspace.graph_tensors().items():
            storage = tensor.untyped_storage()
            self.tensor_leases[name] = (tensor, storage, self._tensor_signature(tensor, storage))
        self.comm_leases = tuple((owner, self._comm_signature(owner))
                                 for owner in (self.comm, self.workspace.native_comm))
        self._transport = self._execution_signature()

    def _execution_signature(self):
        owner, pair = self.workspace, self.workspace.native_pair
        return (owner.rows, owner.rank, owner.heads, owner.head_dim, owner.native_page_tokens,
                owner.native_buckets, pair.key.bits, pair.key.codec, pair.value.bits, pair.value.codec,
                owner.native_collective, owner.native_comm.world)

    def validate_adoption(self):
        """Startup boundary; no file/pointer checks added to each replay."""
        try:
            current = self.workspace.graph_tensors()
            if set(current) != set(self.tensor_leases):
                raise RuntimeError("owner graph tensor catalogue changed after capture")
            for name, (_, storage, signature) in self.tensor_leases.items():
                if self._tensor_signature(current[name], current[name].untyped_storage()) != signature:
                    raise RuntimeError("owner graph tensor/storage contract changed after capture: " + name)
            if (self._execution_signature() != self._transport
                    or self.workspace.native_comm is not self.comm_leases[1][0]
                    or self.workspace.comm is not self.comm_leases[0][0]
                    or any(self._comm_signature(owner) != signature for owner, signature in self.comm_leases)):
                raise RuntimeError("owner graph communicator contract changed after capture")
        except BaseException as error:
            self._poison(error)
            raise

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

    def _capture(self, graph, run):
        secondary = _outcome_slot()  # Fallible storage precedes native acquisition.
        graph.capture_begin(pool=self.pool, capture_error_mode="global")
        primary = None
        try:
            run()
        except BaseException as error:
            primary = error
        try:
            graph.capture_end()
        except BaseException as error:
            secondary[0] = error
        if primary is not None:
            raise_failures(primary, secondary)
        if secondary[0] is not None:
            raise secondary[0]
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
                try:
                    raise_failures(error, _outcome_slot(notification))
                except BaseException as secondary:
                    if secondary is error:
                        raise
                    # The secondary traceback/context owns both a failed
                    # notification and any outcome-preparation exhaustion.
                    # Preserve the primary's explicit cause and latched owner.
                    BaseException.__context__.__set__(error, secondary)
                    raise error

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
            self.validate_adoption()
            pair = self.workspace.native_pair
            # Different catalogues could otherwise agree for a common prefix
            # and then enter different collective counts or CUDA graph modes.
            signature = (self.rows, len(self.buckets), *self.buckets, self.workspace.heads, self.workspace.head_dim,
                         self.workspace.selection.shape[1], self.workspace.native_page_tokens,
                         pair.key.bits, pair.key.codec, pair.value.bits, pair.value.codec,
                         self.workspace.native_comm.world)
            for value in signature:
                if not self._same(value):
                    raise ValueError("owner attention graph geometry/codecs differ across tensor-parallel ranks")
            torch.cuda.synchronize(self.device)
            torch.cuda.empty_cache()
            beginning = self._free()
            for rows in self.buckets:
                # Inactive frames never dereference session pointers. Transport
                # inputs are initialized too; no uninitialized padding is sent.
                run = partial(self.workspace.native_run, rows)
                self.side.wait_stream(torch.cuda.current_stream(self.device))
                # PyTorch requires eager warmup on a side stream as well as
                # nondefault-stream capture. The existing device fence joins
                # this warmup before measuring or entering capture.
                with torch.cuda.stream(self.side):
                    self.workspace.clear_native_inputs(rows)
                    self.workspace.warm_native_transport(rows)
                    self.workspace.wait_native_inputs()
                    run()  # Kernel/module/NCCL lazy initialization precedes capture.
                torch.cuda.synchronize(self.device)
                before = self._free()
                graph = torch.cuda.CUDAGraph()
                self.pending.append(graph)  # Allocate/journal before the construction protocol GO.
                with torch.cuda.stream(self.side):
                    graph = self._capture(graph, run)
                torch.cuda.current_stream(self.device).wait_stream(self.side)
                torch.cuda.synchronize(self.device)
                after = self._free()
                self.graphs[rows] = graph
                self.replays[rows] = graph.replay
                construction = getattr(self.comm, "_profile_construction", None)
                if construction is not None:
                    construction.bind_mapping_submit(self.replays, rows, "owner/replay")
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
                self.replays[rows]()
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
            construction = getattr(self.comm, "_profile_construction", None)
            if construction is not None:
                construction.retire_bindings(self)
                construction.retire_bindings(self.replays)
            torch.cuda.synchronize(self.device)
            errors = []
            for rows, graph in tuple(self.graphs.items()):
                try:
                    graph.reset()
                    del self.graphs[rows]
                except BaseException as error:
                    errors.append(error)
            raise_failures(None, errors)
            self.replays.clear()
            self.tensor_leases, self.comm_leases = {}, ()
            self.votes = self.side = self.pool = None
            self.workspace = self.comm = None
            self.closed = True
        except BaseException as error:
            self._poison(error)
            raise

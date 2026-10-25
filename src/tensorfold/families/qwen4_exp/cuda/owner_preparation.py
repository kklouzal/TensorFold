"""Opt-in, pre-admission ownership of native attention workspaces and graphs.

Preparation is capability work, not release qualification. Tensor storage is
already included in the subsequent physical-free baseline and is adopted by
the real serving buffers, never allocated a second time by their geometry.
"""

from __future__ import annotations

from copy import deepcopy
from functools import partial
from tensorfold.cleanup import finish, raise_failures
from .owner_execution import FULL_COLLECTIVE, HEAD_COLLECTIVE, GRAPH_FAMILY, KERNEL_FAMILY, UNCAPTURED, row_buckets

_ROLES = frozenset({"decode", "mtp", "prefill"})
_CHANNEL = "tensorfold/owner-preparation/v1"


def parse_preparation(value):
    if value is None:
        return None
    required = {"kernel_family", "graph_family", "graph_budget_bytes", "trial"}
    if (not isinstance(value, dict) or not required <= value.keys()
            or set(value).difference(required | {"row_buckets", "collective_family"})
            or value.get("collective_family", FULL_COLLECTIVE) not in (FULL_COLLECTIVE, HEAD_COLLECTIVE)
            or value["kernel_family"] != KERNEL_FAMILY
            or value["graph_family"] not in (UNCAPTURED, GRAPH_FAMILY)
            or type(value["trial"]) is not bool or type(value["graph_budget_bytes"]) is not int
            or value["graph_budget_bytes"] < 0
            or (value["graph_family"] == GRAPH_FAMILY) != (value["graph_budget_bytes"] > 0)):
        raise ValueError("native preparation needs its exact family, explicit trial flag and bounded graph allowance")
    buckets = value.get("row_buckets", {})
    if not isinstance(buckets, dict) or set(buckets).difference(_ROLES):
        raise ValueError("native preparation row buckets must name supported execution roles")
    for rows in buckets.values():
        if not isinstance(rows, list) or not rows:
            raise ValueError("a native execution role requires a nonempty bounded bucket list")
        row_buckets(rows[-1], rows)
    return {"kernel_family": value["kernel_family"], "graph_family": value["graph_family"],
            "graph_budget_bytes": value["graph_budget_bytes"], "trial": value["trial"],
            "collective_family": value.get("collective_family", FULL_COLLECTIVE),
            "row_buckets": deepcopy(buckets)}


class OwnerPreparation:
    def __init__(self, specification, *, role_rows, manifest_sha256):
        self.specification = parse_preparation(specification)
        if self.specification is None or not isinstance(role_rows, dict) or not {"decode", "prefill"} <= role_rows.keys():
            raise ValueError("native preparation requires the actual decode and prefill envelopes")
        if set(role_rows).difference(_ROLES) or set(self.specification["row_buckets"]).difference(role_rows):
            raise ValueError("native preparation names a role absent from the selected method")
        self.role_rows = {role: role_rows[role] for role in ("decode", "mtp", "prefill") if role in role_rows}
        self.buckets = {role: row_buckets(rows, self.specification["row_buckets"].get(role))
                        for role, rows in self.role_rows.items()}
        self.manifest_sha256 = manifest_sha256
        self.workspaces, self.graphs = {}, {}
        self.head_comm = None
        self.comm = self._open_head = None
        self.source_foundation = None
        self.phases = []
        self.before = self.after = None
        self.graph_physical_bytes = 0
        self.prepared = self.closed = False
        self.allocations = {}
        self.private_spare_bytes = None
        self.outside_active_bytes = None

    @staticmethod
    def _memory(torch, device):
        return {"physical_free_bytes": int(torch.cuda.mem_get_info(device)[0]),
                "torch_allocated_bytes": int(torch.cuda.memory_allocated(device)),
                "torch_reserved_bytes": int(torch.cuda.memory_reserved(device))}

    def _phase(self, engine, name, operation, *, may_refuse=True):
        from .execution_identity import _phase

        self.phases.append(name)
        return _phase(engine, _CHANNEL, name, operation, may_refuse=may_refuse)

    def prepare(self, engine, text, pair, *, page_tokens, device):
        import torch
        from tensorfold.engine.model_identity import canonical_runtime_json
        from .execution_identity import _fatal, _retire_store
        from .history_owner import OwnerBuffers
        from .owner_graphs import OwnerAttentionGraphs

        if self.prepared or self.closed:
            raise RuntimeError("native owner preparation is a one-time startup operation")
        self.comm = engine.comm
        construction = getattr(engine.comm, "_profile_construction", None)
        contract = {"specification": self.specification, "manifest_sha256": self.manifest_sha256,
                    "roles": self.role_rows, "page_tokens": page_tokens, "pair": pair.identity,
                    "heads": text["num_attention_heads"], "kv_heads": text["num_key_value_heads"],
                    "head_dim": text.get("head_dim") or text["hidden_size"] // text["num_attention_heads"],
                    "index_budget": text.get("indexer_budget", 2048),
                    "index_ratio": text.get("indexer_compress_ratio", 4)}
        import json

        contract = json.loads(canonical_runtime_json(contract, max_bytes=4 << 20))
        primary = None
        try:
            def prepared_contract():
                from tensorfold.cuda.sm70.foundation_headers import selected_identity

                self.source_foundation = selected_identity()
                return {**contract, "source_foundation": self.source_foundation}

            peers = self._phase(engine, "contract", prepared_contract)
            if any(peer != peers[0] for peer in peers):
                raise ValueError("native owner preparation differs across TP8 ranks")
            torch.cuda.synchronize(device)
            torch.cuda.empty_cache()
            self.before = self._memory(torch, device)
            if self.specification["collective_family"] == HEAD_COLLECTIVE:
                def head_owner():
                    from tensorfold.cuda.comm import NCCL

                    self.head_comm = NCCL()  # Journal before fallible native initialization.
                    self._open_head = partial(self.head_comm.open, engine.rank % 4, 4, "", 0, store=engine.comm.store,
                                              key=_CHANNEL + "/head/" + str(engine.rank // 4))
                    if construction is not None:
                        construction.bind_protocol(self, "_open_head", "owner/head-init", refusal=())
                    return {"members": list(range(engine.rank // 4 * 4, engine.rank // 4 * 4 + 4))}

                # CPU owner/arguments/hooks finish on all8 before any raw init
                # GO. A post-operation phase vote alone cannot establish this.
                self._phase(engine, "head-owner", head_owner, may_refuse=False)
                def head_group():
                    self._open_head()
                    if construction is not None:
                        construction.bind_submit(self.head_comm, "all_gather", "owner/head-gather")
                    return {"members": list(range(engine.rank // 4 * 4, engine.rank // 4 * 4 + 4))}

                self._phase(engine, "head-group", head_group, may_refuse=False)
                # Every subgroup member has consumed its UID before this phase
                # completes. Fixed two keys cannot grow with sessions/buckets.
                def retire_uids():
                    if engine.rank == 0:
                        for group in range(2):
                            engine.comm.store.delete_key(_CHANNEL + "/head/" + str(group))
                    return {}

                self._phase(engine, "head-uid-retirement", retire_uids, may_refuse=False)
            for role, rows in self.role_rows.items():
                def allocate(role=role, rows=rows):
                    owner = OwnerBuffers.from_text(text, rows=rows, prefill=role == "prefill",
                        idw=contract["index_budget"] + contract["index_ratio"], world=engine.tp,
                        rank=engine.rank, device=device, comm=engine.comm,
                        register=lambda owner: self.workspaces.__setitem__(role, owner))
                    owner.prepare_native(pair=pair, page_tokens=page_tokens, buckets=self.buckets[role])
                    owner.configure_native_transport(self.specification["collective_family"],
                                                     self.head_comm or engine.comm)
                    return owner.allocation_record()

                records = self._phase(engine, "allocate-" + role, allocate, may_refuse=False)
                self.allocations[role] = records
                owner = self.workspaces[role]
                def reference_wire(owner=owner):
                    owner.warm_reference_transport()
                    return {"full8_reference_rows": owner.rows}

                self._phase(engine, "reference-wire-" + role, reference_wire, may_refuse=False)
                if self.specification["graph_family"] == GRAPH_FAMILY:
                    remaining = self.specification["graph_budget_bytes"] - self.graph_physical_bytes
                    before_graph = self._memory(torch, device)

                    def graph_owner(role=role, owner=owner, remaining=remaining):
                        if remaining <= 0:
                            raise ValueError("native owner graph preparation exhausted its aggregate allowance")
                        graphs = object.__new__(OwnerAttentionGraphs)
                        self.graphs[role] = graphs
                        graphs.__init__(owner, rows=owner.rows, budget_bytes=remaining, comm=engine.comm,
                                        buckets=self.buckets[role])
                        return {"rows": list(graphs.buckets)}

                    self._phase(engine, "graph-owner-" + role, graph_owner, may_refuse=False)
                    graphs = self.graphs[role]
                    graphs.prepare(quiescent=lambda: engine.multi is None and not engine._calls)
                    torch.cuda.synchronize(device)
                    charged = max(0, before_graph["physical_free_bytes"] - self._memory(torch, device)["physical_free_bytes"])
                    self.graph_physical_bytes += charged

                    def check_budget():
                        if self.graph_physical_bytes > self.specification["graph_budget_bytes"]:
                            raise ValueError("native graph roles exceed their aggregate physical preparation allowance")
                        return {"physical_preparation_bytes": charged, **graphs.record()}

                    self._phase(engine, "graph-budget-" + role, check_budget)
                else:
                    # Eager execution uses this same finite padded catalogue;
                    # eligible library/kernel shape residency precedes admission.
                    def warm(owner=owner, role=role):
                        for bucket in self.buckets[role]:
                            owner.clear_native_inputs(bucket)
                            owner.warm_native_transport(bucket)
                            owner.wait_native_inputs()
                            owner.native_run(bucket)
                            torch.cuda.synchronize(device)
                        return {"rows": list(self.buckets[role])}

                    self._phase(engine, "warm-" + role, warm, may_refuse=False)
            torch.cuda.empty_cache()
            self.after = self._memory(torch, device)
            def commitment():
                from tensorfold.cuda.capacity import cuda_limit_bytes

                limit, committed = cuda_limit_bytes(), self.committed_bytes()
                if limit is not None and committed is None:
                    raise ValueError("absolute CUDA cap requires settled private-pool and preparation residency accounting")
                if limit is not None and committed > limit:
                    raise ValueError("native owner preallocation exceeds the absolute CUDA cap")
                return {"proved_nonreusable_bytes": committed, "absolute_cap_bytes": limit}

            self._phase(engine, "commitment", commitment)
            private = [graph.private_spare_bytes for graph in self.graphs.values()]
            self.private_spare_bytes = None if any(value is None for value in private) else sum(private)
            committed = self.committed_bytes()
            if committed is not None:
                self.outside_active_bytes = committed - (
                    self.after["torch_allocated_bytes"] - self.before["torch_allocated_bytes"])
            self.prepared = True
        except BaseException as error:
            primary = error
            if not isinstance(error, ValueError) or getattr(engine, "_execution_manifest_uncertain", False):
                _fatal(engine, error)
        errors = []
        if not getattr(engine, "_execution_manifest_uncertain", False):
            try:
                _retire_store(engine, _CHANNEL, stages=tuple(self.phases))
            except BaseException as error:
                _fatal(engine, error)
                errors.append(error)
        raise_failures(primary, errors)

    def adopt(self, weights, decoder):
        if not self.prepared or self.closed:
            raise RuntimeError("native resources were not prepared before model admission")
        if weights.meta.get("owner_workspaces") is not self.workspaces:
            raise RuntimeError("model weights did not publish the exact prepared owner workspace catalogue")
        from tensorfold.cuda.sm70.foundation_headers import selected_identity

        if selected_identity() != self.source_foundation:
            raise ValueError("selected source foundation differs from native preparation")
        for graphs in self.graphs.values():
            graphs.validate_adoption()
        for owner in self.workspaces.values():
            owner.bind_native_states(decoder.states)
        for role, buffers in (("decode", decoder.buf), ("mtp", decoder.mbuf), ("prefill", decoder.pbuf)):
            if buffers is None:
                continue
            if buffers.history_owner is not self.workspaces[role]:
                raise RuntimeError("serving buffers did not adopt their preallocated owner workspace")
            buffers.owner_graphs = self.graphs.get(role)
        # Every owner frame is bounded by its adopted workspace, even when a
        # larger model batch is split into several such attention operations.
        graph_rows = decoder.buf.rows if self.graphs else 0
        decoder.controller.configure_execution_families(native=True, graph_rows=graph_rows,
            trial={"kernel_family": KERNEL_FAMILY, "graph_family": self.specification["graph_family"]}
                  if self.specification["trial"] else None)
        decoder.apply_execution_family()

    def committed_bytes(self):
        """Settled nonreusable preallocation, excluding ordinary reusable cache."""
        if self.after is None:
            return None
        if any(graph.private_spare_bytes is None for graph in self.graphs.values()):
            return None
        active = self.after["torch_allocated_bytes"] - self.before["torch_allocated_bytes"]
        reserved = self.after["torch_reserved_bytes"] - self.before["torch_reserved_bytes"]
        physical = self.before["physical_free_bytes"] - self.after["physical_free_bytes"]
        outside = physical - reserved
        if active < 0 or outside < 0:
            return None  # Competing/freeing activity cannot establish a positive owner charge.
        return active + outside + sum(graph.private_spare_bytes for graph in self.graphs.values())

    def live_bytes(self):
        """Credit ordinary spare only after excluding proved private-pool ownership."""
        import torch
        from tensorfold.cuda.capacity import available_bytes, cuda_limit_bytes
        from tensorfold.cuda.kv_arena import mapped_bytes

        active = int(torch.cuda.memory_allocated())
        spare = (max(0, int(torch.cuda.memory_reserved()) - active - self.private_spare_bytes)
                 if self.private_spare_bytes is not None else 0)
        free = int(available_bytes(torch)) + spare
        limit = cuda_limit_bytes()
        if limit is None:
            return free
        if self.outside_active_bytes is None:
            return 0
        return max(0, min(free, limit - active - mapped_bytes() - self.outside_active_bytes))

    def identity(self):
        # The explicit trial switch selects a candidate for measurement; it
        # changes neither representation nor resources. Excluding it permits
        # the resulting verified execution proof to be installed for release.
        return {"specification": {name: deepcopy(value) for name, value in self.specification.items() if name != "trial"},
                "source_foundation_headers": deepcopy(self.source_foundation),
                "roles": dict(self.role_rows),
                "row_buckets": {role: list(rows) for role, rows in self.buckets.items()},
                "allocations": deepcopy(self.allocations),
                "allocator_accounting": "exclude-proved-private-spare; physical-only-when-unavailable"}

    def record(self):
        before, after = self.before, self.after
        return {**self.identity(), "prepared": self.prepared, "closed": self.closed,
                "before": None if before is None else dict(before), "after": None if after is None else dict(after),
                "observed_physical_bytes": None if after is None else before["physical_free_bytes"] - after["physical_free_bytes"],
                "observed_torch_allocated_delta": None if after is None else after["torch_allocated_bytes"] - before["torch_allocated_bytes"],
                "observed_torch_reserved_delta": None if after is None else after["torch_reserved_bytes"] - before["torch_reserved_bytes"],
                "graph_physical_preparation_bytes": self.graph_physical_bytes,
                "proved_nonreusable_commitment_bytes": self.committed_bytes(),
                "private_pool_spare_bytes": self.private_spare_bytes,
                "outside_active_torch_commitment_bytes": self.outside_active_bytes,
                "graphs": {role: graph.record() for role, graph in self.graphs.items()},
                "qualification_trial": self.specification["trial"],
                "included_in_physical_free_baseline": True, "future_owner_persistent_bytes": 0,
                "qualification": "UNQUALIFIED: source-built foundation and real SM70 model/quality/performance gates required"}

    def close(self):
        if self.closed:
            return
        construction = None if self.comm is None else getattr(self.comm, "_profile_construction", None)
        if construction is not None:
            construction.retire_bindings(self)
        finish((graph.close for graph in self.graphs.values()))
        finish((owner.close for owner in self.workspaces.values()))
        if self.head_comm is not None:
            if construction is not None:
                construction.retire_bindings(self.head_comm)
            self.head_comm.close()
            self.head_comm = None
        self.comm = self._open_head = None
        self.graphs, self.workspaces = {}, {}
        self.closed = True

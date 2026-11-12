"""Private supervised SC117 owned-routed qualification over the ordinary CLI/API.

Parent: --gpus 0,1,2,3,4,5,6,7 --out NEW_DIR -- MODEL [ordinary serve options].
No public serving flag selects this policy. Runtime/device/model guards remain
in the existing constructor; this procedure records a bound candidate only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import stat
import sys
import uuid
import threading
import math
import resource
import time

from tensorfold.cleanup import raise_failures, rollback
from tensorfold.cuda.cohort import FAILURE_FD_ENV, notify_failure

if __package__:
    from .benchmark_output import write_json
else:
    from benchmark_output import write_json



def _prompt_digest(tokens):
    """Compact JSON integer-array SHA, borrowing IDs without a joined copy."""
    digest = hashlib.sha256(b"[")
    for index, token in enumerate(tokens):
        if type(token) is not int or not 0 <= token < (1 << 31):
            raise ValueError("attestation requires admitted bounded token IDs")
        if index:
            digest.update(b",")
        digest.update(str(token).encode("ascii"))
    digest.update(b"]")
    return digest.hexdigest()


def _history_input(path, capacity=None, context=None):
    # This deliberately small DATA schema carries no request bodies or gold.
    # Five names/scalars plus C fixed-width SHA strings fit this ASCII bound.
    from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
    limit = MAX_MESSAGE_BYTES if capacity is None else 1024 + 256 * capacity
    with Path(path).open("rb") as source:
        raw = source.read(limit + 1)
    if len(raw) > limit:
        raise ValueError("history attestation DATA exceeds its admitted-C byte bound")
    def pairs(entries):
        result = {}
        for key, value in entries:
            if key in result:
                raise ValueError("duplicate history attestation field")
            result[key] = value
        return result
    record = json.loads(raw, object_pairs_hook=pairs)
    if (type(record) is not dict or set(record) != {"C", "N", "R", "L", "prompt_token_ids_sha256"}
            or any(type(record[key]) is not int or not 0 < record[key] <= sys.maxsize for key in ("C", "N", "R", "L"))
            or capacity is not None and record["C"] != capacity
            or context is not None and record["L"] != context or record["N"] + record["R"] != record["L"]):
        raise ValueError("history attestation requires actual C/N/R/L geometry")
    capacity = record["C"]
    limit = 1024 + 256 * capacity
    if len(raw) > limit:
        raise ValueError("history attestation DATA exceeds its admitted-C byte bound")
    hashes = record["prompt_token_ids_sha256"]
    if (type(hashes) is not list or len(hashes) != capacity or any(type(value) is not str
            or len(value) != 64 or any(char not in "0123456789abcdef" for char in value) for value in hashes)
            or len(set(hashes)) != capacity):
        raise ValueError("history attestation requires C distinct lowercase token-ID SHA256 values")
    return record, hashlib.sha256(raw).hexdigest(), limit


def _history_resources(capacity, input_limit):
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        _json_payload_policy, _json_parse_payload, _json_tree_payload,
        _python_layout, _streamed_json_encoding_payload,
    )
    policy = _json_payload_policy()
    # Fixed identity/header/resource names plus C rows of <=10 bounded scalars
    # and one 64-byte SHA; ASCII indentation is included in these source bounds.
    receipt_limit = 4096 + 1024 * capacity
    # The extra input tree also pays the five fixed object_pairs_hook pairs
    # and partial result dict; the C hash list is borrowed, not copied by the hook.
    per_rank = (_json_parse_payload(input_limit, policy)["total_bytes"]
                + _json_tree_payload(input_limit, policy)
                + _json_tree_payload(receipt_limit, policy)
                + _streamed_json_encoding_payload(receipt_limit, _python_layout())["total_bytes"]
                + policy.frame_work)
    total = 8 * per_rank
    if total > sys.maxsize:
        raise ValueError("history qualification host obligation exceeds the host size domain")
    return {"global_host_obligation_bytes": total, "per_rank_payload_bytes": per_rank,
            "input_bytes_limit": input_limit, "receipt_bytes_limit": receipt_limit,
            "materialized_credit_bytes": 0, "runtime_policy": policy.identity,
            "scope": "source/object payload; native allocator/RSS separate; probe excluded from performance"}


def prepare_history(engine, prepared, output):
    data, input_sha, resources = prepared
    if (len(engine.multi.states) != data["C"] or engine.context_window != data["L"]
            or engine._owned_qualification_host != (resources["global_host_obligation_bytes"], input_sha)):
        raise ValueError("history qualification differs from actual C/context/paid obligation")
    catalog = engine.retention_catalog
    if catalog is None or catalog.host_resources is None:
        raise ValueError("history qualification requires the original immutable FIFO host baseline")
    if catalog.host_resources.baseline_host_bytes < resources["global_host_obligation_bytes"]:
        raise ValueError("history qualification obligation is absent from the host baseline")
    observer = _HistoryAttestation(engine, data, input_sha, output, resources)
    observer.install()
    return resources


def _router_input(path):
    if __package__:
        from .openai_protocol import object_json
    else:
        from openai_protocol import object_json
    with Path(path).open("rb") as source:
        raw = source.read(4097)
    if len(raw) > 4096:
        raise ValueError("router-row input exceeds4096bytes")
    data = object_json(raw)
    if (set(data) != {"schema", "run_identity", "prompt_token_ids_sha256"}
            or data["schema"] != "tensorfold.router-row-input-v1"
            or type(data["run_identity"]) is not str or len(data["run_identity"]) != 32
            or any(c not in "0123456789abcdef" for c in data["run_identity"])):
        raise ValueError("router-row input requires its fresh bounded run identity")
    hashes = data["prompt_token_ids_sha256"]
    if (type(hashes) is not list or len(hashes) != 2
            or any(type(h) is not str or len(h) != 64 or any(c not in "0123456789abcdef" for c in h) for h in hashes)
            or len(set(hashes)) != 2):
        raise ValueError("router-row input requires ordered prompt2/prompt7 SHA256")
    return data, hashlib.sha256(raw).hexdigest()


def _router_resources(output):
    import io
    import numpy as np
    import torch
    # Four frames of <=32 piece records plus four selected aliases/markers.
    # Each piece fits the existing <=1024-byte receipt-row bound.
    path_chars = len(json.dumps(str((Path(output) / "router-row-stage.json").absolute()), ensure_ascii=True))
    base = _history_resources(4 * (32 + 1) + 8 + (path_chars + 1023) // 1024, 4096)
    # One rank owns W and four x/logit pairs; NumPy views borrow the CPU tensors.
    payload = 513 * 2560 * 4 + 4 * (2560 * 2 + 513 * 4)
    wrappers = 9 * (sys.getsizeof(torch.empty(0)) + sys.getsizeof(np.empty((0, 0))))
    io_work = 2 * 65536 + io.DEFAULT_BUFFER_SIZE + 9 * 256
    return {**base, "global_host_obligation_bytes": base["global_host_obligation_bytes"] + payload + wrappers + io_work,
            "rank0_tensor_bytes": payload, "tensor_numpy_headers": wrappers, "file_hash_writer_bytes": io_work,
            "additional_device_bytes": 0, "maximum_capture_rows": 4,
            "copy_policy": "contiguous same-dtype CUDA view directly to CPU; no CUDA clone/contiguous conversion"}


def _save_router_tensor(path, tensor):
    import io
    import numpy as np
    import torch
    if (not tensor.is_cuda or not tensor.is_contiguous() or tensor.requires_grad
            or tensor.dtype not in (torch.float16, torch.float32) or tensor.ndim != 2
            or tensor.is_neg() or tensor.is_conj()):
        raise ValueError("router capture requires admitted contiguous CUDA Half/Float matrices")
    # Same dtype + contiguous source/destination takes ATen's direct D2H copy.
    # This synchronously observes the diagnostic stream, never a performance run.
    host = tensor.detach().to(device="cpu", copy=True)
    view = host.numpy()
    if not view.flags.c_contiguous or view.ctypes.data != host.data_ptr():
        raise ValueError("router CPU NumPy view did not borrow the copied storage")
    output = Path(path).open("xb", buffering=io.DEFAULT_BUFFER_SIZE)
    try:
        np.save(output, view, allow_pickle=False)
        output.flush()
        os.fchmod(output.fileno(), 0o444)
        os.fsync(output.fileno())
    except BaseException as primary:
        rollback(output, primary, output.close)
    else:
        output.close()
    directory = os.open(Path(path).parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    except BaseException as primary:
        rollback(directory, primary, lambda: os.close(directory))
    else:
        os.close(directory)
    before = Path(path).stat()
    if before.st_size > tensor.numel() * tensor.element_size() + 256:
        raise ValueError("router NPY header exceeds its funded bound")
    digest = hashlib.sha256()
    source = Path(path).open("rb", buffering=io.DEFAULT_BUFFER_SIZE)
    try:
        while chunk := source.read(65536):
            digest.update(chunk)
    except BaseException as primary:
        rollback(source, primary, source.close)
    else:
        source.close()
    after = Path(path).stat()
    def stamp(s):
        return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns]
    if stamp(before) != stamp(after):
        raise ValueError("router NPY changed during hashing")
    return {"file": Path(path).name, "sha256": digest.hexdigest(), "shape": list(tensor.shape),
            "dtype": view.dtype.str, "stat": stamp(after), "bytes": after.st_size}


class _RouterRows:
    """Default-off finite private observer; original math calls remain once."""
    def __init__(self, engine, data, input_sha, resources, output):
        self.engine, self.data, self.input_sha, self.resources = engine, data, input_sha, resources
        self.output, self.decoder = Path(output), engine.multi
        self.frame, self.pending, self.records, self.seen = None, [], [], set()
        self.bank = engine.w.layers[0].moe.experts
        self.weights = engine.w.layers[0].moe.router
        self.original_bank = self.bank.forward_partitioned_native_candidate
        self.original_prepare = self.decoder._prepare_pass
        self.observations = self.decoder.controller.work_observations
        self.original_target, self.original_round = self.observations.target, self.decoder.round
        self.weight_record = None

    def marker(self):
        if __package__:
            from .openai_protocol import object_json
        else:
            from openai_protocol import object_json
        path = self.output / "router-row-stage.json"
        with open(path, "rb", opener=lambda name, flags: os.open(name, flags | os.O_NOFOLLOW)) as source:
            before = os.fstat(source.fileno())
            if not stat.S_ISREG(before.st_mode):
                raise ValueError("router stage marker requires an ordinary file")
            raw = source.read(513)
            after = os.fstat(source.fileno())
        if (before.st_ino, before.st_dev, before.st_size, before.st_mtime_ns) != (
                after.st_ino, after.st_dev, after.st_size, after.st_mtime_ns):
            raise ValueError("router stage marker changed during reading")
        if len(raw) > 512:
            raise ValueError("router stage marker exceeds512bytes")
        marker = object_json(raw)
        if (set(marker) != {"schema", "run_identity", "stage"}
                or marker["schema"] != "tensorfold.router-row-stage-v1"
                or marker["run_identity"] != self.data["run_identity"]
                or marker["stage"] not in {"baseline-c8-mtp", "baseline-c8-mtp-repeat", "alone-2-mtp",
                                            "alone-2-serial", "alone-7-mtp", "alone-7-serial", "baseline-c8-serial"}):
            raise ValueError("router stage marker differs from its controlled run")
        marker.update(path=path.name, sha256=hashlib.sha256(raw).hexdigest(),
                      stat=[after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns])
        return marker

    def preparing(self, pieces):
        if len(pieces) > 32:
            raise ValueError("router metadata exceeds its prepared frame")
        self.frame = None
        marker = self.marker()
        stage = marker["stage"]
        allowed = ({0, 1} if stage == "baseline-c8-mtp" else {0} if stage == "alone-2-mtp"
                   else {1} if stage == "alone-7-mtp" else set())
        rows, records, targets, state_cuts = 0, [], [], []
        for stream, start, count in pieces:
            state = stream.st
            wire = self.engine.tp_decoder.wires.get(id(stream))
            digest = _prompt_digest(stream.prompt)
            policy = stream.sampling
            sampling = {key: None if policy is None else getattr(policy, key, None)
                        for key in ("temperature", "seed", "top_k", "top_p", "min_p")}
            if any(value is not None and (type(value) not in (int, float) or not math.isfinite(value)
                                         or abs(value) > sys.maxsize) for value in sampling.values()):
                raise ValueError("router sampling metadata exceeds bounded scalar domain")
            attention_layer = min(state.att_index)
            ordinal = state.att_index[attention_layer]
            record = {"wire": wire, "prompt_sha256": digest, "slot": state.history_slot,
                      "slot_generation": state.slot_generation, "ownership_generation": state.history_generation,
                      "prompt_start": start, "count": count, "first": rows, "last": rows + count,
                      "sampling": sampling, "draft": stream.draft,
                      "attention_model_layer": attention_layer, "attention_ordinal": ordinal,
                      "indexer_owners": [state.history_placement.indexer_owner(state.history_slot, ordinal)],
                      "page_owners": [state.history_placement.kv_owner(state.history_slot, ordinal, 0, page) for page in (0, 1)]}
            records.append(record)
            state_cuts.append((state, rows, rows + count))
            if digest in self.data["prompt_token_ids_sha256"]:
                ordinal = self.data["prompt_token_ids_sha256"].index(digest)
                key = (stage, ordinal)
                if ordinal in allowed and key not in self.seen and start + count == len(stream.prompt):
                    if len(stream.prompt) != 64 or stream.count != 128 or not stream.draft or wire is None:
                        raise ValueError("router selected request differs from original64/128MTP")
                    targets.append((key, rows + count - 1, record))
            rows += count
        if rows > self.decoder.pbuf.rows:
            raise ValueError("router metadata exceeds its prepared frame")
        result = self.original_prepare(pieces)
        self.frame = {"stage": stage, "marker": marker, "rows": rows, "pieces": records,
                      "state_cuts": state_cuts, "targets": targets, "prefix_operation": self.engine.tp_decoder.prefix_operation}
        return result

    def bank_call(self, x, picks, **kwargs):
        result = self.original_bank(x, picks, **kwargs)
        frame = self.frame
        if frame is None or not frame["targets"] or x.data_ptr() != self.decoder.pbuf.mixed.data_ptr():
            return result
        if x.shape != (frame["rows"], 2560) or len(self.records) + len(self.pending) + len(frame["targets"]) > 4:
            raise ValueError("router capture exceeds its four admitted rows")
        for key, offset, record in frame["targets"]:
            item = {"stage": frame["stage"], "target": key[1], "frame_rows": frame["rows"], "row_offset": offset,
                    "pieces": frame["pieces"], "marker": frame["marker"], "selected": record, "prefix_operation": frame["prefix_operation"]}
            if self.engine.rank == 0:
                label = f"router-{key[0]}-{key[1]}"
                item["input"] = _save_router_tensor(self.output / (label + "-input.npy"), x.narrow(0, offset, 1))
                item["logits"] = _save_router_tensor(self.output / (label + "-logits.npy"),
                                                       self.decoder.pbuf.moe.logits.narrow(0, offset, 1))
            self.pending.append(item)
            self.seen.add(key)
        frame["targets"] = []
        return result

    def targeted(self, segments, **kwargs):
        result = self.original_target(segments, **kwargs)
        if kwargs.get("prefill") and self.frame is not None:
            expected = [(r["first"], r["last"]) for r in self.frame["pieces"]]
            if ([(first, last) for _, first, last in segments] != expected
                    or any(actual is not wanted for (actual, _, _), (wanted, _, _) in
                           zip(segments, self.frame["state_cuts"]))):
                raise ValueError("router submitted segment cuts/States differ from captured frame")
            for item in self.pending:
                item["submitted"] = True
        return result

    def rounded(self):
        try:
            result = self.original_round()
            if self.pending:
                if any(not item.get("submitted") for item in self.pending):
                    raise ValueError("router capture lacks successful target submission")
                self.records.extend(self.pending)
                self.pending = []
                from tensorfold.server.stats import INSTANCE_ID
                receipt = {
                    "schema": "tensorfold.router-row-capture-v1", "diagnostic_only": True,
                    "rank": self.engine.rank, "run_identity": self.data["run_identity"], "server_instance": INSTANCE_ID,
                    "input_sha256": self.input_sha, "stage_path": str((self.output / "router-row-stage.json").absolute()),
                    "resources": self.resources, "weights": self.weight_record,
                    "records": self.records, "complete": len(self.records) == 4,
                    "scope": "successful worker submission/round; client outputs and operation ACKs remain required"}
                encoded_bytes = 0
                for chunk in json.JSONEncoder(indent=1, ensure_ascii=True, allow_nan=False).iterencode(receipt):
                    encoded_bytes += len(chunk)
                    if encoded_bytes + 1 > self.resources["receipt_bytes_limit"]:
                        raise ValueError("router receipt exceeds its funded metadata bound")
                write_json(self.output / f"router-rank-{self.engine.rank}.json", receipt)
            return result
        except BaseException as primary:
            _observer_failure(self.engine, primary)
            raise

    def install(self):
        if self.engine.rank == 0:
            self.weight_record = _save_router_tensor(self.output / "router-layer0-weights.npy", self.weights)
        self.decoder._prepare_pass = self.preparing
        self.bank.forward_partitioned_native_candidate = self.bank_call
        self.observations.target = self.targeted
        self.decoder.round = self.rounded


def prepare_router_rows(engine, prepared, output):
    data, input_sha, resources = prepared
    if len(engine.multi.states) != 32 or engine.context_window != 262144:
        raise ValueError("router capture requires its original32/full262144 profile")
    if engine._owned_qualification_host != (resources["global_host_obligation_bytes"], input_sha):
        raise ValueError("router row obligation differs from original admission")
    if engine.retention_catalog.host_resources.baseline_host_bytes < resources["global_host_obligation_bytes"]:
        raise ValueError("router row obligation missing from immutable host baseline")
    if engine.w.layers[0].moe.router.shape != (513, 2560):
        raise ValueError("router capture requires actual layer0 SC117 weights")
    observer = _RouterRows(engine, data, input_sha, resources, output)
    observer.install()
    return resources


def _prepared_feature_resources():
    # Three fixed role records of at most eight 1KiB schema units each, plus
    # fixed identity/grant headers. Pay both retained intent and final trees.
    base = _history_resources(32, 4096)
    source_bytes = Path(__file__).stat().st_size
    per_rank = 2 * base["per_rank_payload_bytes"] + source_bytes
    return {**base, "per_rank_payload_bytes": per_rank,
            "global_host_obligation_bytes": 8 * per_rank, "maximum_roles": 3,
            "observer_source_bytes": source_bytes, "additional_device_bytes": 0,
            "scope": "private fixed metadata/close observer; existing Graph counters only"}


def _feature_integer(value, name, *, optional=False):
    if optional and value is None:
        return None
    if type(value) is not int or not 0 <= value <= 2**63 - 1:
        raise ValueError("prepared feature metadata has invalid " + name)
    return value


def _feature_memory(value, keys):
    if type(value) is not dict or set(value) != set(keys):
        raise ValueError("prepared feature memory snapshot differs from its fixed schema")
    return {key: _feature_integer(value[key], key) for key in keys}


def _feature_digest(value):
    if type(value) is not str or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("prepared feature source requires SHA256")
    return value


class _PreparedFeatures:
    """Keep at most three already-owned Graph references through normal close.

    No request callback, CUDA probe or extra fence: Graph owners supply their
    existing counters and preparation records. Unresolved close keeps this
    journal and every reference; this observer never retries Engine.close.
    """
    def __init__(self, engine, output, resources, grouped_gate_up):
        if type(engine.rank) is not int or not 0 <= engine.rank < 8:
            raise ValueError("prepared feature receipts require an actual TP8 rank")
        self.engine, self.resources = engine, resources
        self.original_close = engine.close
        self.had_close = "close" in vars(engine)
        self.old_close_attribute = vars(engine).get("close")
        self.attempted = self.closed = False
        self.failure = None
        self.graphs = {}
        self.directory = Path(output) / f"prepared-features-rank-{engine.rank}"
        self.grouped_gate_up = _grouped_gate_up_receipt(engine, grouped_gate_up)
        preparation = getattr(engine, "_owner_preparation", None)
        request = None if preparation is None else preparation.reference_arithmetic_request
        source = None
        if request is not None:
            source = _feature_digest(preparation.reference_arithmetic_source)
            expected = {"decode": engine.multi.buf, "prefill": engine.multi.pbuf}
            if engine.w.mtp is not None:
                expected["mtp"] = engine.multi.mbuf
            if set(preparation.reference_arithmetic) != set(expected):
                raise ValueError("prepared feature Graph roles differ from actual buffers")
            for role, buffers in expected.items():
                graph = preparation.reference_arithmetic[role]
                owner = preparation.workspaces[role]
                if (buffers.history_owner is not owner or owner.reference_tiles.arithmetic is not graph
                        or any(graph is other for other in self.graphs.values())):
                    raise ValueError("prepared feature Graph lacks its exclusive journalled role")
                self.graphs[role] = graph
        elif preparation is not None and preparation.reference_arithmetic:
            raise ValueError("prepared feature Graph exists without its preparation request")
        if not self.graphs and grouped_gate_up is None:
            raise ValueError("prepared feature receipts require Graph or grouped gate/up selection")
        records = self._records(retired=False)
        if any(record["source_sha256"] != source or record["policy"] != request.policy
               for record in records.values()):
            raise ValueError("prepared feature Graph source differs from preparation")
        charged = sum(record["charged_preparation_bytes"] for record in records.values())
        aggregate = None if request is None else _feature_integer(request.graph_budget_bytes, "aggregate budget")
        if request is not None and (charged != _feature_integer(preparation.reference_arithmetic_bytes, "aggregate charge")
                                    or charged > aggregate):
            raise ValueError("prepared feature aggregate charge exceeds or differs from actual admission")
        self.preparation = {"graph_budget_bytes": aggregate, "charged_preparation_bytes": charged,
            "before": None if request is None else _feature_memory(preparation.before,
                ("physical_free_bytes", "torch_allocated_bytes", "torch_reserved_bytes")),
            "after": None if request is None else _feature_memory(preparation.after,
                ("physical_free_bytes", "torch_allocated_bytes", "torch_reserved_bytes")),
            "private_pool_spare_bytes": None if request is None else _feature_integer(
                preparation.private_spare_bytes, "preparation pool spare", optional=True),
            "source_sha256": source}
        self.marker = {"schema": "tensorfold.prepared-features-v1", "rank": engine.rank,
            "directory": self.directory.name, "expected_roles": list(self.graphs),
            "grouped_gate_up": self.grouped_gate_up, "preparation": self.preparation,
            "roles": records if self.graphs else None, "resources": dict(resources),
            "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        self._encoded(self.marker)

    def _encoded(self, value):
        data = json.dumps(value, sort_keys=True, ensure_ascii=True, allow_nan=False, indent=1).encode()
        if len(data) > self.resources["receipt_bytes_limit"]:
            raise ValueError("prepared feature receipt exceeds its paid JSON schema bound")
        return data

    def _records(self, *, retired):
        records = {}
        if len(self.graphs) > 3:
            raise ValueError("prepared feature Graph role bound exceeded")
        for role, graph in self.graphs.items():
            if (graph.prepared is not True or graph.closed is not retired or graph.failure is not None):
                raise RuntimeError("prepared feature Graph failed or did not prove retirement") from graph.failure
            record = graph.record()
            policy = record["policy"]
            if type(policy) is not str or not 0 < len(policy) <= 128:
                raise ValueError("prepared feature policy exceeds its schema")
            counts = record["replays"]
            if type(counts) is not dict or set(counts) != {"stats", "probability", "pv"}:
                raise ValueError("prepared feature Graph counter schema differs")
            counts = {key: _feature_integer(value, key) for key, value in counts.items()}
            if retired and any(value == 0 for value in counts.values()):
                raise RuntimeError("prepared feature Graph role lacks stats/probability/PV replay")
            result = {"policy": policy, "prepared": True, "closed": retired, "failed": False,
                "replays": counts, "source_sha256": _feature_digest(record["source_sha256"]),
                "before": _feature_memory(record["before"], ("free", "allocated", "reserved")),
                "after": _feature_memory(record["after"], ("free", "allocated", "reserved"))}
            for name in ("graph_budget_bytes", "charged_preparation_bytes", "retained_tensor_payload_bytes",
                         "private_pool_spare_bytes"):
                result[name] = _feature_integer(record[name], name, optional=name == "private_pool_spare_bytes")
            if result["charged_preparation_bytes"] > result["graph_budget_bytes"]:
                raise ValueError("prepared feature role exceeds its actual budget")
            records[role] = result
        return records

    def _restore(self):
        if self.engine.close != self.close_binding:
            raise RuntimeError("prepared feature close binding changed before retirement")
        if self.had_close:
            self.engine.close = self.old_close_attribute
        else:
            delattr(self.engine, "close")

    def _fail(self, primary, errors=()):
        self.failure = primary
        failures = list(errors)
        try:
            _observer_failure(self.engine, primary)
        except BaseException as notification:
            if notification is not primary:
                failures.append(notification)
        raise_failures(primary, failures)

    def close(self, *args, **kwargs):
        if self.attempted:
            if self.failure is not None:
                raise self.failure
            raise RuntimeError("prepared feature close observer already retired")
        self.attempted = True
        try:
            result = self.original_close(*args, **kwargs)
            if self.engine._closed is not True:
                raise RuntimeError("prepared feature Engine close did not prove retirement")
            if self.engine._abort_comm is not False:
                raise RuntimeError("prepared feature Engine retired with an aborted cohort")
        except BaseException as primary:
            self._fail(primary)  # Binding/Graph refs remain with unresolved owners.
        primary, errors, summary = None, [], None
        try:
            roles = self._records(retired=True)
            if any({key: value for key, value in record.items() if key not in ("closed", "replays")}
                   != {key: value for key, value in self.marker["roles"][role].items() if key not in ("closed", "replays")}
                   for role, record in roles.items()):
                raise ValueError("prepared feature Graph preparation/source changed during execution")
            summary = {"schema": self.marker["schema"], "rank": self.marker["rank"],
                "closed": True, "failed": False, "preparation": self.preparation,
                "grouped_gate_up": self.grouped_gate_up, "roles": roles if self.graphs else None,
                "tool_sha256": self.marker["tool_sha256"],
                "prepared_marker_sha256": hashlib.sha256(self._encoded(self.marker)).hexdigest()}
            self._encoded(summary)
        except BaseException as error:
            primary = error
        try:
            self._restore()
        except BaseException as error:
            errors.append(error)
        if primary is not None or errors:
            if primary is None:
                primary = errors.pop(0)
            self._fail(primary, errors)
        try:
            target = self.directory / "summary.json"
            if target.exists() or target.is_symlink():
                raise ValueError("prepared feature summary already exists")
            write_json(target, summary)
        except BaseException as primary:
            self._fail(primary)
        self.closed = True
        self.graphs.clear()
        self.engine = self.original_close = self.old_close_attribute = self.close_binding = None
        return result

    def install(self):
        self.directory.mkdir(mode=0o700)  # Exclusive namespace, never adopt an existing run.
        write_json(self.directory / "prepared.json", self.marker)
        directory = os.open(self.directory.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        except BaseException as primary:
            rollback(directory, primary, lambda: os.close(directory))
        else:
            os.close(directory)
        self.close_binding = self.close
        try:
            self.engine.close = self.close_binding
        except BaseException as primary:
            errors = []
            try:
                if self.had_close:
                    self.engine.close = self.old_close_attribute
                else:
                    vars(self.engine).pop("close", None)
            except BaseException as cleanup:
                errors.append(cleanup)
            self._fail(primary, errors)
        return self.marker


def prepare_prepared_features(engine, resources, output, grouped_gate_up):
    if (engine._owned_qualification_host != (resources["total_global_host_obligation_bytes"], resources["input_sha256"])
            or engine.retention_catalog.host_resources.baseline_host_bytes < resources["total_global_host_obligation_bytes"]):
        raise ValueError("prepared feature observer is absent from original host admission")
    observer = _PreparedFeatures(engine, output, resources, grouped_gate_up)
    return observer.install()


def _gdn_batch_preflight(args, cli):
    if (args.draft_method != 'none' or str(args.parallel).lower() == 'auto'
            or cli._parallel(args.parallel) != 24):
        raise ValueError('GDN parity requires explicit NONE and24 fixed resident slots before construction')


def _gdn_batch_resources(plan, output):
    from tensorfold.cuda.sm70 import gdn_native
    if not all(callable(getattr(gdn_native, name, None)) for name in ('batch_chain', 'chain')):
        raise ValueError('installed runtime lacks GDN batch/scalar witness before construction')
    if not any(c['level'] == 24 and not c['warmup'] for c in plan['cohorts']):
        raise ValueError('GDN witness requires an actual C24 benchmark cohort')
    input_rows = sum(c['level']*c['input_tokens'] for c in plan['cohorts'])
    output_rows = sum(c['level']*c['tokens'] for c in plan['cohorts'])
    journal_rows = 2*input_rows + output_rows
    base = _history_resources(2*journal_rows + 48 + 64, 4096)
    shapes = {'pre_rec': ((24, 6, 128, 128), 4), 'pre_conv': ((24, 3, 1280), 2),
              'pre_projection': ((24, 2060), 2), 'pre_keep': ((24,), 4),
              'pre_descriptor': ((8*48*24,), 8), 'pre_starts': ((25,), 4),
              'actual_input_rec': ((24, 6, 128, 128), 4), 'actual_conv': ((24, 3, 1280), 2),
              'actual_projection': ((24, 2060), 2)}
    for phase in ('expected', 'actual'):
        shapes.update({phase+'_rec': ((24, 6, 128, 128), 4), phase+'_k': ((24, 1, 2, 128), 4),
            phase+'_v': ((24, 1, 6, 128), 2), phase+'_g': ((24, 1, 6), 4),
            phase+'_b': ((24, 1, 6), 4), phase+'_out': ((24, 768), 2), phase+'_sums': ((24, 24), 4)})
    payload = sum(math.prod(shape)*width for shape, width in shapes.values())
    # Two actual/published bounded JSON trees and serializer are prepaid by
    # base; pinned tensors, native wrappers/streaming IO are additional.
    source_bytes = sum(Path(x).stat().st_size for x in (__file__, gdn_native.__file__))
    per_rank = base['per_rank_payload_bytes'] + payload + source_bytes + 2*65536 + len(shapes)*4096
    if 8*per_rank > sys.maxsize:
        raise ValueError('GDN witness host obligation exceeds size domain')
    return {**base, 'per_rank_payload_bytes': per_rank, 'global_host_obligation_bytes': 8*per_rank,
            'slabs': shapes, 'pinned_tensor_payload_bytes': payload, 'maximum_segment_rows': journal_rows,
            'maximum_segment_events': journal_rows, 'maximum_calls': 48*output_rows,
            'additional_device_bytes': 0, 'diagnostic_only': True, 'P3_valid': False}


def _gdn_witness_region(engine):
    import torch
    dec = engine.multi
    if (engine.w.device.type != 'cuda' or engine.w.mtp is not None or dec.speculation.method != 'none'
            or dec.speculation.maximum_proposals != 0 or len(dec.states) != 24 or not dec.full_resident
            or dec.buf.h.dtype != torch.float16 or (engine.w.cfg.nk, engine.w.cfg.nv) != (2, 6)
            or not dec.round_tables.native_half or dec.round_tables.slots != 24 or not 0 < dec.gdn.lin <= 48 or dec.held
            or engine.capacity_plan.get('native_gdn_batch', {}).get('enabled') is not True):
        raise ValueError('GDN witness requires the actual C24 Half/NONE native batch region')
    return dec


def _write_gdn_host_tensor(path, tensor):
    import numpy as np
    if __package__:
        from .gb10_io import opened_file, write_all
    else:
        from gb10_io import opened_file, write_all
    values = tensor.numpy()
    with opened_file(path, 'xb') as stream:
        np.lib.format.write_array_header_1_0(stream, np.lib.format.header_data_from_array_1_0(values))
        view = memoryview(values).cast('B')
        for begin in range(0, len(view), 65536):
            write_all(stream, view[begin:begin+65536])
        stream.flush()
        os.fsync(stream.fileno())
    size = path.stat().st_size
    return {'file': path.name, 'bytes': size, 'sha256': _source_sha(path, size)}


class _GDNBatchParity:
    """Current scalar BEFORE batch on identical live inputs; not a math oracle."""
    def __init__(self, engine, resources, output, *, module=None, allocator=None, writer=None):
        import torch
        from types import SimpleNamespace
        from tensorfold.cuda.sm70 import gdn_native
        self.engine, self.resources, self.rank = engine, resources, engine.rank
        self.module = gdn_native if module is None else module
        self.original, self.scalar = self.module.batch_chain, self.module.chain
        self.original_close = engine.close
        self.had_close, self.old_close = 'close' in vars(engine), vars(engine).get('close')
        self.status, self.transport = [None]*8, [None]
        self.status_order = (0, 1, 2, 3, 5, 7, 4, 6)  # Actual body/cleanup chronology.
        self.active = self.closed = self.failed = self.installed = False
        self.primary = None
        self.observer = None
        self.prefill_pending = None
        self.counts = [{'calls': 0, 'rows': 0, 'matched': 0} for _ in range(48)]
        self.calls = self.segment_rows = self.segment_count = self.maximum_states = 0
        self.segments = [None]*resources['maximum_segment_events']
        self.slabs = {}
        self.writer = _write_gdn_host_tensor if writer is None else writer
        self.decoder = _gdn_witness_region(engine)
        shared = self.decoder.gdn
        for value, dtype in ((shared.k, torch.float32), (shared.v, torch.float16), (shared.q, torch.float32),
                             (shared.g, torch.float32), (shared.beta, torch.float32)):
            if not value.is_contiguous() or value.device != engine.w.device or value.dtype != dtype:
                raise ValueError('GDN witness needs contiguous unused Half sharedscratch')
        if shared.k.numel() < 6*128*128 or shared.q.numel() < 2*128+24 or shared.v.numel() < 2*6*128:
            raise ValueError('GDN witness exceeds existing unused sharedscratch')
        self.final = shared.k.view(-1)[:6*128*128].view(6, 128, 128)
        self.shadow = SimpleNamespace(k=shared.q.view(-1)[:256].view(1, 2, 128),
            v=shared.v.view(-1)[:768].view(1, 6, 128),
            g=shared.g.view(-1)[:6].view(1, 6), b=shared.beta.view(-1)[:6].view(1, 6))
        self.out = shared.v.view(-1)[768:1536].view(1, 768)
        self.sums = shared.q.view(-1)[256:280].view(1, 24)
        allocate = torch.empty if allocator is None else allocator
        for name, (shape, width) in resources['slabs'].items():
            dtype = torch.int64 if width == 8 else torch.int32 if name in ('pre_keep', 'pre_starts') else torch.float32 if width == 4 else torch.float16
            self.slabs[name] = allocate(tuple(shape), dtype=dtype, device='cpu', pin_memory=True)
        self.sources = {name: _source_sha(path, Path(path).stat().st_size) for name, path in
                        [('tool', __file__), ('scalar_and_batch_wrapper', self.module.__file__)]}
        if module is None:
            native = self.module.extension()  # Already resident under actual enabled startup preparation.
            self.sources['native_module'] = native.__name__
            self.sources['native_elf'] = _source_sha(native.__file__, Path(native.__file__).stat().st_size)
        self.directory = Path(output)/f'gdn-batch-parity-rank-{engine.rank}'
        self.directory.mkdir(mode=0o700)
        self._publish()

    def _publish(self):
        self.published = {'schema': 'tensorfold.gdn-batch-parity-v1', 'rank': self.rank,
            'closed': self.closed, 'failed': self.failed, 'calls': self.calls, 'maximum_states': self.maximum_states,
            'counts': [dict(x) for x in self.counts], 'segment_count': self.segment_count,
            'segment_rows': self.segment_rows, 'sources': dict(self.sources),
            'scope': 'same-module scalar-vs-batch live-input witness; not independent mathematics/fullquality',
            'benchmark_plan_sha256': self.resources.get('input_sha256'),
            'diagnostic_only': True, 'P3_valid': False, 'saved_rows_compared_per_state': 1}

    def _first_failure(self, primary=None):
        index = 0
        while primary is None and index < 8:
            primary = self.status[self.status_order[index]]
            index += 1
        return primary

    def _raise(self, primary):
        primary = self._first_failure(primary)
        try:
            errors = [x for x in self.status if x is not None and x is not primary]
        except BaseException as error:
            self.transport[0] = error
            raise_failures(primary, self.transport)
        raise_failures(primary, errors)

    def release_prefill(self):
        self.prefill_pending = None

    def _segment_authority(self, state, stream):
        if stream is None or stream.st is not state or self.observer is None:
            raise ValueError('GDN segment lost benchmark request authority')
        cohort, request = self.observer._identity(stream)
        wire = self.engine.tp_decoder.wires.get(id(stream))
        if type(wire) is not int:
            raise ValueError('GDN segment lost its stable TP wire')
        return wire, cohort['id'], request['index'], request['prompt_sha256']

    def segment(self, kind, segments, pieces=None):
        if self.closed:
            raise RuntimeError('GDN segment journal retired')
        try:
            if not 0 < len(segments) <= 24:
                raise ValueError('GDN segment count exceeds actual State envelope')
            rows, pending = [], None
            if kind == 'prepare-prefill':
                if pieces is None or len(pieces) != len(segments) or self.prefill_pending is not None:
                    raise ValueError('GDN prefill authority is missing, mismatched or unconsumed')
                bound = []
                for index, ((state, begin, end), (stream, start, count)) in enumerate(zip(segments, pieces)):
                    if (type(start) is not int or type(count) is not int or start < 0 or count <= 0
                            or end-begin != count or start+count > len(stream.prompt)
                            or begin != (0 if index == 0 else segments[index-1][2])
                            or any(previous[0] is state for previous in bound)):
                        raise ValueError('GDN prepare segment differs from authoritative piece/order')
                    authority = self._segment_authority(state, stream)
                    generation = (state.history_slot, state.slot_generation, state.history_generation)
                    bound.append((state, stream, authority, start, count, begin, end, generation))
                pending = tuple(bound)  # Strong ownership, bounded to this single prepare pass.
                entries = pending
            elif kind == 'target-prefill':
                pending, self.prefill_pending = self.prefill_pending, None
                if pieces is not None or pending is None or len(pending) != len(segments):
                    raise ValueError('GDN target prefill has no matching prepared authority')
                for (state, begin, end), entry in zip(segments, pending):
                    prior, stream, authority, start, count, first, last, generation = entry
                    if (state is not prior or begin != first or end != last or end-begin != count
                            or (state.history_slot, state.slot_generation, state.history_generation) != generation
                            or self._segment_authority(state, stream) != authority):
                        raise ValueError('GDN target prefill changed State/generation/request/span/order')
                entries = pending
            elif kind in ('target-decode', 'mismatch-decode'):
                if pieces is not None:
                    raise ValueError('GDN decode cannot borrow prefill authority')
                entries = []
                for state, begin, end in segments:
                    matches = [s for s in self.decoder.streams.values() if s.st is state]
                    if len(matches) != 1 or end <= begin:
                        raise ValueError('GDN decode lost unique active-stream authority')
                    stream = matches[0]
                    authority = self._segment_authority(state, stream)
                    entries.append((state, stream, authority, state.pos, end-begin, begin, end,
                                    (state.history_slot, state.slot_generation, state.history_generation)))
            else:
                raise ValueError('unknown GDN segment journal transition')
            for state, stream, authority, start, count, begin, end, generation in entries:
                wire, cohort, request, prompt_sha = authority
                # pos is the actual observed position at this callback. The
                # input start/count are retained from prepare, not inferred
                # from a mutable State position after GPU work.
                rows.append([wire, cohort, request, *generation, state.pos, begin, end, start, prompt_sha])
            if (self.segment_count >= len(self.segments)
                    or self.segment_rows+len(rows) > self.resources['maximum_segment_rows']):
                raise ValueError('GDN segment journal exceeds finite planned work')
            self.segments[self.segment_count] = {'kind': kind, 'rows': rows}
            self.segment_count += 1
            self.segment_rows += len(rows)
            if kind == 'prepare-prefill':
                self.prefill_pending = pending
        except BaseException:
            self.release_prefill()
            raise

    def prefill_joined(self, pieces):
        if self.prefill_pending is not None:
            self.release_prefill()
            raise ValueError('GDN prefill joined before its target transition')

    def __call__(self, p, weight, decay, bias, norm, eps, tables, layer, output, sums, *, negative_decay=False):
        import torch
        if self.closed:
            raise RuntimeError('GDN parity callback retired')
        if self.active or self.failed:
            raise RuntimeError('GDN parity callback is active or failed')
        engine = self.engine
        self.active = True
        mismatch = None
        try:
            segs = tables.segs
            n = len(segs)
            if (tables.buffers is not self.decoder.buf or not 1 < n <= 24 or p.shape != (n, 2060)
                    or tables.rows != n or not 0 <= layer < min(48, self.decoder.gdn.lin)
                    or any(end-begin != 1 for _, begin, end in segs)
                    or self.calls >= self.resources['maximum_calls'] or self.decoder.held
                    or torch.cuda.is_current_stream_capturing()):
                raise ValueError('GDN parity call differs from admitted live NONE frame')
            self.calls += 1
            self.maximum_states = max(self.maximum_states, n)
            self.counts[layer]['calls'] += 1
            self.counts[layer]['rows'] += n
            parities = [state.cur[layer] for state, _, _ in segs]
            c = self.slabs
            c['pre_projection'][:n].copy_(p, non_blocking=True)
            descriptor_size = tables.static.i64.numel()
            c['pre_descriptor'][:descriptor_size].copy_(tables.static.i64, non_blocking=True)
            c['pre_starts'][:n+1].copy_(tables.starts, non_blocking=True)
            for i, (state, begin, end) in enumerate(segs):
                cur = state.cur[layer]
                scratch = state.scratch[layer]
                c['pre_rec'][i].copy_(state.rec[cur, layer], non_blocking=True)
                c['pre_conv'][i].copy_(state.conv[layer], non_blocking=True)
                keep = getattr(scratch, 'keep_rows', None)
                if keep is None:
                    c['pre_keep'][i] = -1
                else:
                    c['pre_keep'][i:i+1].copy_(keep, non_blocking=True)
                self.shadow.keep_rows = keep
                self.scalar(p[begin:end], state.conv[layer], weight, state.rec[cur, layer], decay,
                    bias, norm, eps, 1, self.shadow, self.final, self.out, self.sums,
                    negative_decay=negative_decay)
                c['expected_rec'][i].copy_(self.final, non_blocking=True)
                for name in ('k', 'v', 'g', 'b'):
                    c['expected_'+name][i].copy_(getattr(self.shadow, name), non_blocking=True)
                c['expected_out'][i].copy_(self.out[0], non_blocking=True)
                c['expected_sums'][i].copy_(self.sums[0], non_blocking=True)
            # Preserve actual candidate args, return value and real State writes.
            result = self.original(p, weight, decay, bias, norm, eps, tables, layer, output, sums,
                                   negative_decay=negative_decay)
            for i, (state, _, _) in enumerate(segs):
                c['actual_rec'][i].copy_(state.rec[1-parities[i], layer], non_blocking=True)
                c['actual_input_rec'][i].copy_(state.rec[parities[i], layer], non_blocking=True)
                c['actual_conv'][i].copy_(state.conv[layer], non_blocking=True)
                for name in ('k', 'v', 'g', 'b'):
                    c['actual_'+name][i].copy_(getattr(state.scratch[layer], name)[:1], non_blocking=True)
            c['actual_projection'][:n].copy_(p, non_blocking=True)
            c['actual_out'][:n].copy_(output, non_blocking=True)
            c['actual_sums'][:n].copy_(sums, non_blocking=True)
            torch.cuda.current_stream(engine.w.device).synchronize()  # One diagnostic fence; no new event/payload.
            if any(state.cur[layer] != parity for (state, _, _), parity in zip(segs, parities)):
                mismatch = RuntimeError('GDN batch changed input parity metadata')
            for name, actual_name, expected_name in (
                    ('rec', 'actual_rec', 'expected_rec'), ('k', 'actual_k', 'expected_k'),
                    ('v', 'actual_v', 'expected_v'), ('g', 'actual_g', 'expected_g'),
                    ('b', 'actual_b', 'expected_b'), ('out', 'actual_out', 'expected_out'),
                    ('sums', 'actual_sums', 'expected_sums'), ('input_rec', 'actual_input_rec', 'pre_rec'),
                    ('conv', 'actual_conv', 'pre_conv'), ('projection', 'actual_projection', 'pre_projection')):
                if mismatch is not None:
                    break
                a, b = c[actual_name][:n], c[expected_name][:n]
                if not torch.equal(a.view(torch.uint8), b.view(torch.uint8)):
                    mismatch = RuntimeError('GDN batch/scalar raw-bit mismatch: '+name)
                    break
            if mismatch is not None:
                # Construct numerical primary BEFORE any fallible metadata/IO.
                try:
                    files = {name: self.writer(self.directory/(name+'.npy'),
                        value[:descriptor_size] if name == 'pre_descriptor' else value[:n+1] if name == 'pre_starts' else value[:n])
                        for name, value in c.items()}
                    self.segment('mismatch-decode', segs)
                    target = self.directory/'mismatch.json'
                    if target.exists() or target.is_symlink():
                        raise ValueError('GDN mismatch evidence already exists')
                    write_json(target, {'layer': layer, 'rows': n,
                        'benchmark_segments': self.segments[self.segment_count-1],
                        'sources': self.sources, 'files': files, 'negative_decay': negative_decay,
                        'states': [[s.history_slot, s.slot_generation, s.history_generation, s.pos, s.cur[layer], a, z]
                                   for s, a, z in segs],
                        'keep_present': [getattr(s.scratch[layer], 'keep_rows', None) is not None for s, _, _ in segs]})
                except BaseException as error:
                    self.status[0] = error
                self._raise(mismatch)
            self.counts[layer]['matched'] += n
            self._publish()
            return result
        except BaseException as primary:
            if mismatch is not None and primary is not mismatch:
                self.status[7] = primary
                primary = mismatch
            self.failed = True
            self.release_prefill()
            if self.primary is None:
                self.primary = primary
            try:
                self._publish()
            except BaseException as error:
                self.status[1] = error
            try:
                _observer_failure(engine, primary)
            except BaseException as error:
                if error is not primary:
                    self.status[2] = error
            self._raise(primary)
        finally:
            self.active = False

    def install(self):
        def close(*args, **kwargs):
            if self.closed:
                raise RuntimeError('GDN parity close callback retired')
            engine = self.engine
            try:
                result = self.original_close(*args, **kwargs)
                if engine._closed is not True or self.active:
                    raise RuntimeError('GDN parity owner callbacks not retired')
                self.release_prefill()
            except BaseException as primary:
                try:
                    _observer_failure(engine, primary)
                except BaseException as notification:
                    if notification is not primary:
                        self.status[6] = notification
                self._raise(primary)
            try:
                if self.module.batch_chain is not self:
                    raise RuntimeError('GDN parity module binding changed')
                self.module.batch_chain = self.original
            except BaseException as error:
                self.status[3] = error
            try:
                if self.had_close:
                    engine.close = self.old_close
                else:
                    delattr(engine, 'close')
            except BaseException as error:
                self.status[5] = error
            self.failed = self.failed or getattr(engine, '_abort_comm', False) or self._first_failure() is not None
            if not self.failed:
                complete = self.maximum_states == 24 and self.observer is not None and self.observer.finished
                for layer in range(self.decoder.gdn.lin):
                    count = self.counts[layer]
                    complete = complete and count['calls'] > 0 and count['matched'] == count['rows']
                if not complete:
                    self.status[7] = ValueError('GDN parity lacks actual24-State/layer coverage or finite plan completion')
                    self.failed = True
            self.closed = True
            try:
                self._publish()
                target = self.directory/'summary.json'
                if target.exists() or target.is_symlink():
                    raise ValueError('GDN summary already exists')
                write_json(target, {**self.published,
                    'segments': self.segments[:self.segment_count]})
            except BaseException as error:
                self.status[4] = error
            self.slabs.clear()
            self.final = self.shadow = self.out = self.sums = None
            self.engine = self.decoder = self.observer = self.original_close = self.old_close = None
            self.original = self.scalar = self.writer = self.module = self.close_binding = None
            primary = self._first_failure(self.primary)
            if primary is not None:
                try:
                    _observer_failure(engine, primary)
                except BaseException as error:
                    if error is not primary:
                        self.status[6] = error
            self._raise(primary)
            return result
        self.close_binding = close
        try:
            self.module.batch_chain = self
            self.engine.close = close
            self.installed = True
        except BaseException as primary:
            try:
                self.module.batch_chain = self.original
            except BaseException as error:
                self.status[5] = error
            try:
                if self.had_close:
                    self.engine.close = self.old_close
                else:
                    vars(self.engine).pop('close', None)
            except BaseException as error:
                self.status[6] = error
            self._raise(primary)



def prepare_gdn_batch_parity(engine, resources, output):
    if (engine._owned_qualification_host != (resources['total_global_host_obligation_bytes'], resources['input_sha256'])
            or engine.retention_catalog.host_resources.baseline_host_bytes < resources['total_global_host_obligation_bytes']):
        raise ValueError('GDN parity obligation is absent from original admission')
    observer = object.__new__(_GDNBatchParity)
    engine._gdn_batch_parity = observer
    observer.__init__(engine, resources, output)
    observer.install()
    return observer


def _normalization_resources(output):
    # OwnerBuffers.from_text caps every prepared role at min(rows,256).
    rows = 256
    from tensorfold.cuda.sm70 import reference_owner_tiles
    if not all(callable(getattr(reference_owner_tiles, name, None)) for name in
               ('merge_global_statistics', 'global_normalization_workspace_bytes')):
        raise ValueError('installed runtime lacks normalization parity capability before construction')
    path_chars = len(json.dumps(str(Path(output).absolute()), ensure_ascii=True))
    base = _history_resources(3 * 49 + 16 + (path_chars + 1023) // 1024, 4096)
    # One same-dtype D2H file at a time; binary payload, streaming writer/hash,
    # bounded source-file identities and metadata, never JSON float arrays.
    from tensorfold.families.qwen4_exp.cuda import history_owner
    source_bytes = sum(Path(path).stat().st_size for path in
                       (__file__, reference_owner_tiles.__file__, history_owner.__file__))
    extra = 8 * rows * 12 * 2 * 4 + 2 * 65536 + source_bytes
    per_rank = base['per_rank_payload_bytes'] + extra
    return {**base, 'per_rank_payload_bytes': per_rank,
            'global_host_obligation_bytes': 8 * per_rank, 'maximum_rows': rows,
            'maximum_layer': 48, 'additional_device_bytes': 0,
            'diagnostic_only': True, 'comparison_fence_per_call': 1}


def _q8_parity_resources(plan, output):
    from tensorfold.families.qwen4_exp.cuda import history_owner
    if not callable(getattr(history_owner, '_prepare_reference', None)):
        raise ValueError('installed Q8 witness lacks the retained reference before construction')
    cohort = next((c for c in plan['cohorts'] if not c['warmup'] and c['level'] == 24), None)
    if plan['profile'] != 'q8' or cohort is None or cohort['tokens'] < 4:
        raise ValueError('Q8 witness requires its finite Q8/C24 cohort and at least four outputs')
    fields = {'q': ((1, 3, 256), 'float16'), 'iq': ((1, 4, 128), 'float16'),
              'k': ((1, 1, 256), 'int8'), 'v': ((1, 1, 256), 'int8'),
              'ks': ((1, 1, 8), 'float16'), 'vs': ((1, 1, 8), 'float16'), 'status': ((2,), 'int32')}
    slabs = {'projection': ((1, 2688), 'float16'), 'coordinates': ((1, 3), 'int64'),
             'q_scale': ((256,), 'float32'), 'k_scale': ((256,), 'float32'),
             'iq_scale': ((128,), 'float32'), 'frequency': ((64,), 'float32'),
             'initial_status': ((2,), 'int32'), 'generations': ((2,), 'int64'), 'flags': ((7,), 'bool')}
    slabs.update({side+'_'+key: value for side in ('expected', 'actual') for key, value in fields.items()})
    widths = {'float16': 2, 'float32': 4, 'int32': 4, 'int64': 8, 'int8': 1, 'bool': 1}
    payload = sum(math.prod(shape)*widths[dtype] for shape, dtype in slabs.values())
    base = _history_resources(288 + 24 + 48, 4096)
    source_bytes = Path(__file__).stat().st_size + Path(history_owner.__file__).stat().st_size
    per_rank = base['per_rank_payload_bytes'] + payload + source_bytes + len(slabs)*4096
    if 8*per_rank > sys.maxsize:
        raise ValueError('Q8 witness host obligation exceeds size domain')
    return {**base, 'per_rank_payload_bytes': per_rank, 'global_host_obligation_bytes': 8*per_rank,
            'slabs': slabs, 'fields': fields, 'pinned_tensor_payload_bytes': payload,
            'maximum_calls': 288, 'maximum_rounds': 1, 'cohort_id': cohort['id'],
            'additional_device_bytes': 0, 'diagnostic_only': True, 'P3_valid': False,
            'device_scope': 'borrow unused native-Half GDN q slab; original single-row reference temporaries use unchanged runtime workspace policy'}


def _q8_witness_region(engine):
    import torch
    decoder = engine.multi
    if (_owned_q8_preparation_receipt(engine) is None or engine.w.device.type != 'cuda'
            or engine.w.mtp is not None or decoder.speculation.method != 'none'
            or decoder.speculation.maximum_proposals != 0 or len(decoder.states) != 24
            or not decoder.full_resident or not decoder.round_tables.native_half
            or decoder.buf.h.dtype != torch.float16 or decoder.held
            or (engine.w.cfg.heads, engine.w.cfg.head_dim, engine.w.cfg.index_heads,
                engine.w.cfg.index_dim) != (3, 256, 4, 128)):
        raise ValueError('Q8 witness requires actual full24/NONE/Half owned-Q8 decode')
    return decoder


class _Q8PreparationParity:
    """Retained Torch writer oracle BEFORE one warmed24-State native round."""
    def _first_failure(self, primary=None):
        index = 0
        while primary is None and index < len(self.status_order):
            primary = self.status[self.status_order[index]]
            index += 1
        return primary

    _raise = _GDNBatchParity._raise

    def __init__(self, engine, resources, output, *, module=None, reference=None, allocator=None, fence=None):
        import torch
        from tensorfold.cuda.sm70 import family_ops
        from tensorfold.families.qwen4_exp.cuda import history_owner
        self.engine, self.resources, self.rank = engine, resources, engine.rank
        self.decoder = _q8_witness_region(engine)
        self.module = family_ops if module is None else module
        self.original = self.module.owned_q8_prepare
        if isinstance(self.original, _Q8PreparationParity):
            raise ValueError('Q8 writer already has an owned diagnostic binding')
        self.reference = history_owner._prepare_reference if reference is None else reference
        self.original_close = engine.close
        self.had_close, self.old_close = 'close' in vars(engine), vars(engine).get('close')
        self.status, self.transport = [None]*10, [None]
        self.status_order = (0, 1, 2, 3, 7, 4, 5, 6, 8, 9)
        self.primary = self.observer = self.call_owners = self.round_thread = None
        self.in_call = False
        self.concurrent_error = RuntimeError('Q8 witness is outside its owned round thread or is reentrant')
        self.active = self.selected = self.completed = self.closed = self.failed = self.installed = False
        self.notification_attempted = self.evidence_attempted = self.close_attempted = False
        self.retired_error = RuntimeError('Q8 witness is retired or close was already attempted')
        self.calls, self.matches = 0, 0
        self.counts = [0]*48
        self.seen, self.bindings = set(), ()
        self.mismatch = RuntimeError('Q8 preparation same-input raw-bit mismatch')
        self.layers = tuple(layer.index for layer in engine.w.layers if layer.attn is not None)
        if len(self.layers) != 12:
            raise ValueError('Q8 witness requires actual12 attention layers')
        self.workspace = self.decoder.gdn.q
        if (self.workspace.dtype != torch.float32 or not self.workspace.is_contiguous()
                or self.workspace.device != engine.w.device):
            raise ValueError('Q8 witness requires its unused native-Half shared q slab')
        self.views, self.slabs = {}, {}
        raw, cursor = self.workspace.view(torch.uint8).view(-1), 0
        layout = dict(resources['fields'])
        layout.update(initial_status=((2,), 'int32'), flags=((7,), 'bool'))
        for name, (shape, dtype) in layout.items():
            cursor = (cursor + 7)//8*8
            size = math.prod(shape)*torch.empty((), dtype=getattr(torch, dtype)).element_size()
            if size > raw.numel() - cursor:
                raise ValueError('Q8 witness exceeds its prepaid unused shared q slab')
            self.views[name] = raw[cursor:cursor+size].view(getattr(torch, dtype)).view(shape)
            cursor += size
        self.device_borrow_bytes = cursor
        allocate = torch.empty if allocator is None else allocator
        for name, (shape, dtype) in resources['slabs'].items():
            self.slabs[name] = allocate(shape, dtype=getattr(torch, dtype), device='cpu', pin_memory=True)
        self.fence = (lambda: torch.cuda.current_stream(engine.w.device).synchronize()) if fence is None else fence
        self.sources = {name: _source_sha(path, Path(path).stat().st_size) for name, path in
                        [('tool', __file__), ('reference_prepare', history_owner.__file__), ('writer_wrapper', self.module.__file__)]}
        native = self.decoder.buf.history_owner.owned_q8_preparation
        self.sources['native_module'] = native.__name__
        self.sources['native_elf'] = _source_sha(native.__file__, Path(native.__file__).stat().st_size)
        self.directory = Path(output)/f'q8-preparation-parity-rank-{self.rank}'
        self.directory.mkdir(mode=0o700)
        self._publish()

    def _publish(self):
        self.published = {'schema': 'tensorfold.q8-preparation-parity-v1', 'rank': self.rank,
            'selected': self.selected, 'completed': self.completed, 'closed': self.closed, 'failed': self.failed,
            'calls': self.calls, 'matched': self.matches, 'counts': list(self.counts),
            'cohort_id': self.resources['cohort_id'], 'benchmark_plan_sha256': self.resources.get('input_sha256'),
            'device_borrow_bytes': self.device_borrow_bytes, 'sources': dict(self.sources),
            'scope': 'retained Torch same-input preparation math/writer witness; skips shadow compact side effects; not reader/pool/physical-store/fullquality proof',
            'diagnostic_only': True, 'P3_valid': False}

    def begin_round(self, observer):
        if self.closed or self.active:
            raise RuntimeError('Q8 witness round ownership is retired/reentrant')
        if self.selected or observer.current != self.resources['cohort_id'] or self.decoder.filling:
            return
        streams = [s for s in self.decoder.streams.values() if not s.done and s.st is not None]
        if len(streams) != 24 or any(len(s.out) <= 1 for s in streams):
            return
        bindings = []
        for stream in streams:
            cohort, request = observer._identity(stream)
            state = stream.st
            wire = self.engine.tp_decoder.wires.get(id(stream))
            if (cohort['id'] != self.resources['cohort_id'] or type(wire) is not int
                    or any(row[0] is state for row in bindings)):
                raise ValueError('Q8 witness lost current24-State request authority')
            bindings.append((state, stream, wire, request['index'], request['prompt_sha256'],
                             state.slot_generation, state.history_generation, state.pos))
        self.bindings, self.observer = tuple(bindings), observer
        self.round_thread = threading.current_thread()
        self.selected = self.active = True

    def end_round(self, primary=None):
        if not self.active:
            return
        self.active = False
        if primary is not None:
            self.failed = True
            if self.primary is None:
                self.primary = primary
            return
        if self.calls != 288 or self.matches != 288 or any(self.counts[index] != 24 for index in self.layers):
            self._fail(ValueError('Q8 witness did not execute all24 States/all12 layers in its selected round'))
        self.completed = True
        self._publish()

    def _fail(self, primary):
        self.failed = True
        if self.primary is None:
            self.primary = primary
        if self.observer is not None:
            self.observer.failed = self.primary
        if not self.notification_attempted:
            self.notification_attempted = True
            try:
                _observer_failure(self.engine, primary)
            except BaseException as error:
                if error is not primary:
                    self.status[2] = error
        self._raise(self.primary)

    def _evidence(self, metadata, inputs, actual):
        if self.evidence_attempted:
            raise RuntimeError('Q8 mismatch evidence already attempted')
        self.evidence_attempted = True
        for name, value in inputs.items():
            self.slabs[name].copy_(value, non_blocking=True)
        for name, value in self.views.items():
            if name in self.resources['fields']:
                self.slabs['expected_'+name].copy_(value, non_blocking=True)
        for name, value in actual.items():
            self.slabs['actual_'+name].copy_(value, non_blocking=True)
        self.fence()
        names = [*inputs, *('expected_'+name for name in self.resources['fields']),
                 *('actual_'+name for name in actual), 'flags']
        rows = {name: _write_gdn_host_tensor(self.directory/(name+'.npy'), self.slabs[name]) for name in names}
        write_json(self.directory/'first-mismatch.json', {**metadata, 'files': rows, 'sources': self.sources,
            'scope': self.published['scope'], 'P3_valid': False})

    def __call__(self, module, projection, coordinates, weights, cache, state, layer, query, iq, *,
                 frequency, sections, amplitude, start):
        if self.closed:
            raise self.retired_error
        if not self.active:
            return self.original(module, projection, coordinates, weights, cache, state, layer, query, iq,
                                 frequency=frequency, sections=sections, amplitude=amplitude, start=start)
        import torch
        from types import SimpleNamespace as NS
        if threading.current_thread() is not self.round_thread or self.in_call:
            self._fail(self.concurrent_error)
        self.in_call = True
        try:
            binding = next((row for row in self.bindings if row[0] is state), None)
            tables = self.decoder.buf.gdn_tables
            row = next(((begin, end) for st, begin, end in tables.segs if st is state), None)
            if (binding is None or binding[1].st is not state or binding[1].done
                    or self.engine.tp_decoder.wires.get(id(binding[1])) != binding[2]
                    or (state.slot_generation, state.history_generation, state.pos) != binding[5:]
                    or row is None or row[1]-row[0] != 1 or layer.index not in self.layers
                    or weights is not self.engine.w or start != state.pos or self.calls >= 288
                    or (id(state), layer.index) in self.seen
                    or projection.data_ptr() != self.decoder.buf.pa[row[0]:row[1]].data_ptr()
                    or coordinates.data_ptr() != self.decoder.buf.target_indexer_coordinates[row[0]:row[1]].data_ptr()):
                raise ValueError('Q8 witness input is not its current issued State/layer/row')
            self.call_owners = (projection, coordinates, query, iq, cache.k, cache.v, cache.ks, cache.vs,
                                state.kv_status, state.history_generations, frequency, layer.attn)
            self.views['initial_status'].copy_(state.kv_status)
            self.views['status'].copy_(state.kv_status)
            appended = [False]
            def append(position, k, v, ks, vs):
                if position != start or appended[0]:
                    raise ValueError('shadow reference changed its single writer span')
                for name, value in (('k', k), ('v', v), ('ks', ks), ('vs', vs)):
                    self.views[name].copy_(value)
                appended[0] = True
            shadow_cache = NS(key_format=cache.key_format, value_format=cache.value_format, append=append)
            shadow_state = NS(pos=start, att_index={layer.index: 0}, kc=[shadow_cache],
                kv_status=self.views['status'], compact_indexer=object(), history_slot=state.history_slot,
                history_rank=state.history_rank,
                history_placement=NS(indexer_owner=lambda *a, **k: (state.history_rank+1)%8))
            shadow_buffers = NS(prefill=False, target_indexer_coordinates=coordinates,
                target_indexer_position=NS(inv_freq=frequency, sections=sections, amplitude=amplitude),
                q=self.views['q'], iq=self.views['iq'])
            self.reference(projection, shadow_state, layer, weights, shadow_buffers, 0, 1, mtp=False)
            if not appended[0]:
                raise ValueError('Q8 shadow reference failed to publish its encoded row')
            result = self.original(module, projection, coordinates, weights, cache, state, layer, query, iq,
                                   frequency=frequency, sections=sections, amplitude=amplitude, start=start)
            if result is None:
                raise ValueError('Q8 witness candidate declined its selected live row')
            physical = cache.offsets[start//cache.placement.page_tokens]
            actual = {'q': query, 'iq': iq, 'status': state.kv_status}
            if physical >= 0:
                physical += start % cache.placement.page_tokens
                actual.update({name: getattr(cache, name)[physical:physical+1] for name in ('k', 'v', 'ks', 'vs')})
            for index, name in enumerate(self.resources['fields']):
                if name not in actual:
                    self.views['flags'][index].fill_(True)
                else:
                    self.views['flags'][index].copy_(torch.eq(actual[name].view(torch.uint8),
                                                              self.views[name].view(torch.uint8)).all())
            self.slabs['flags'].copy_(self.views['flags'], non_blocking=True)
            self.fence()
            if not bool(self.slabs['flags'].all()):
                self.primary = self.mismatch  # Recognized numeric primary BEFORE fallible evidence/metadata.
                try:
                    metadata = {'rank': self.rank, 'layer': layer.index, 'slot': state.history_slot,
                        'generation': list(binding[5:7]), 'position': start, 'wire': binding[2],
                        'request_index': binding[3], 'prompt_sha256': binding[4], 'owned': physical >= 0,
                        'cohort_id': self.resources['cohort_id']}
                    inputs = {'projection': projection, 'coordinates': coordinates,
                        'q_scale': layer.attn.q_scale, 'k_scale': layer.attn.k_scale, 'iq_scale': layer.attn.iq_scale,
                        'frequency': frequency, 'initial_status': self.views['initial_status'],
                        'generations': state.history_generations}
                    self._evidence(metadata, inputs, actual)
                except BaseException as error:
                    self.status[1] = error
                self._fail(self.mismatch)
            self.seen.add((id(state), layer.index))
            self.calls += 1
            self.matches += 1
            self.counts[layer.index] += 1
            self.call_owners = None  # Comparison fence positively retired this borrow.
            return result
        except BaseException as primary:
            self._fail(primary)
        finally:
            self.in_call = False

    def install(self):
        def close(*args, **kwargs):
            if self.close_attempted:
                if self.primary is not None:
                    self._raise(self.primary)
                raise self.retired_error
            self.close_attempted = True
            engine = self.engine
            try:
                result = self.original_close(*args, **kwargs)
            except BaseException as primary:
                self.status[3] = primary
                self._fail(self.primary if self.primary is not None else primary)
            if not getattr(engine, '_closed', False) or self.active:
                self._fail(RuntimeError('Q8 witness lacks positive owner retirement'))
            if not self.failed and (not self.completed or self.observer is None or not self.observer.finished):
                self.status[7] = ValueError('Q8 witness lacks selected round/full finite benchmark completion')
                self.failed = True
            try:
                if self.module.owned_q8_prepare is not self:
                    raise RuntimeError('Q8 witness module binding changed')
                self.module.owned_q8_prepare = self.original
            except BaseException as error:
                self.status[4] = error
            try:
                if self.had_close:
                    engine.close = self.old_close
                else:
                    vars(engine).pop('close', None)
            except BaseException as error:
                self.status[5] = error
            self.failed = self.failed or any(error is not None for error in self.status)
            self.closed = True
            try:
                self._publish()
                target = self.directory/'summary.json'
                if target.exists() or target.is_symlink():
                    raise ValueError('Q8 witness summary already exists')
                write_json(target, self.published)
            except BaseException as error:
                self.status[6] = error
            # Positive original retirement precedes all view release. Attempt
            # both independent releases even if one diagnostic cleanup fails.
            try:
                self.call_owners = self.bindings = self.workspace = None
                self.views.clear()
            except BaseException as error:
                self.status[8] = error
            try:
                self.slabs.clear()
            except BaseException as error:
                self.status[9] = error
            primary = self._first_failure(self.primary)
            if primary is not None:
                self._fail(primary)
            self.engine = self.decoder = self.observer = self.original_close = self.old_close = None
            self.module = self.original = self.reference = self.fence = None
            return result
        try:
            self.module.owned_q8_prepare = self
            self.engine.close = close
            self.installed = True
        except BaseException as primary:
            try:
                if self.module.owned_q8_prepare is self:
                    self.module.owned_q8_prepare = self.original
            except BaseException as error:
                self.status[4] = error
            try:
                if self.had_close:
                    self.engine.close = self.old_close
                else:
                    vars(self.engine).pop('close', None)
            except BaseException as error:
                self.status[5] = error
            self._fail(primary)
        return self.published


def prepare_q8_preparation_parity(engine, resources, output):
    if (engine._owned_qualification_host != (resources['total_global_host_obligation_bytes'], resources['input_sha256'])
            or engine.retention_catalog.host_resources.baseline_host_bytes < resources['total_global_host_obligation_bytes']):
        raise ValueError('Q8 witness is absent from original host admission')
    observer = object.__new__(_Q8PreparationParity)
    engine._q8_preparation_parity = observer
    observer.__init__(engine, resources, output)
    observer.install()
    return observer


class _NormalizationParity:
    """Same-input all-rank diagnostic; never installed by ordinary serving."""
    def __init__(self, engine, output, resources, *, module=None, scalar=None, writer=None):
        import torch
        from tensorfold.cuda.sm70 import reference_owner_tiles
        from tensorfold.families.qwen4_exp.cuda.history_owner import merge_statistics
        self.engine, self.output, self.resources = engine, Path(output), resources
        self.rank = engine.rank
        self.module = reference_owner_tiles if module is None else module
        self.scalar = merge_statistics if scalar is None else scalar
        self.writer = _save_router_tensor if writer is None else writer
        self.original = self.module.merge_global_statistics
        self.observations = engine.multi.controller.work_observations
        self.original_record, self.original_close = self.observations.record, engine.close
        self.had_close = 'close' in vars(engine)
        self.old_close_attribute = vars(engine).get('close')
        self.active = self.failed = self.installed = self.closed = False
        self.counts, self.geometry, self.role_layouts, self.roles = {}, {}, {}, []
        self.sources = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest() for name, path in
                        [('candidate_file', self.module.__file__), ('tool_file', __file__)]}
        import inspect
        self.sources['scalar_body'] = hashlib.sha256(inspect.getsource(self.scalar).encode()).hexdigest()
        for role, name in [('decode', 'buf'), ('prefill', 'pbuf'), ('mtp', 'mbuf')]:
            buffer = getattr(engine.multi, name, None)
            if buffer is None:
                continue
            owner = buffer.history_owner
            maximum = owner.rows
            if (not 0 < maximum <= resources['maximum_rows'] or owner.global_stats is None
                    or owner.partial.dtype != torch.float32 or owner.partials.dtype != torch.float32
                    or owner.global_stats.dtype != torch.float32
                    or not owner.partial.is_contiguous() or not owner.partials.is_contiguous()
                    or not owner.global_stats.is_contiguous() or owner.partial.data_ptr() % 512
                    or owner.global_stats.shape != (maximum, 12, 2)
                    or owner.partial.device != owner.partials.device or owner.partial.device != owner.global_stats.device
                    or any(torch._C._overlaps(a, b) for a, b in
                           ((owner.partial, owner.partials), (owner.partial, owner.global_stats),
                            (owner.partials, owner.global_stats)))
                    or owner.reference_tiles.normalization_workspace is not owner.partial
                    or owner.reference_tiles.normalization_output is not owner.global_stats):
                raise ValueError('normalization parity lacks the actual prepared owner')
            raw = owner.partial.view(torch.uint8).view(-1)
            cursor = -(-reference_owner_tiles.global_normalization_workspace_bytes(maximum) // 512) * 512
            tail_start = cursor
            def take(count, dtype):
                nonlocal cursor
                size = 1 if dtype == torch.bool else 4
                cursor = -(-cursor // size) * size
                end = cursor + count * size
                if end > raw.numel():
                    raise ValueError('normalization parity exceeds the unused paid local tail')
                value = raw[cursor:end].view(dtype)
                cursor = end
                return value
            entry = {'role': role, 'owner': owner, 'maximum': maximum,
                     'snapshot': take(8 * maximum * 24, torch.float32),
                     'old': take(2 * maximum * 12, torch.float32),
                     'before': take(2 * maximum, torch.int32).view(maximum, 2),
                     'shadow': take(2 * maximum, torch.int32).view(maximum, 2),
                     'actual': take(2 * maximum, torch.int32).view(maximum, 2),
                     'input_equal': take(8 * maximum * 24, torch.bool),
                     'output_equal': take(2 * maximum * 12, torch.bool),
                     'status_equal': take(2 * maximum, torch.bool), 'flags': take(4, torch.bool)}
            self.roles.append(entry)
            self.role_layouts[role] = {'maximum_rows': maximum, 'tail_start_bytes': tail_start,
                                      'tail_bytes': cursor - tail_start, 'local_slab_bytes': raw.numel()}
        if not self.roles:
            raise ValueError('normalization parity has no prepared roles')
        self.directory = self.output / f'normalization-parity-rank-{engine.rank}'
        self.directory.mkdir(mode=0o700)
        self._publish()

    def _publish(self):
        self.published = {'schema': 'tensorfold.normalization-parity-v1', 'diagnostic_only': True,
            'rank': self.rank, 'sources': dict(self.sources), 'failed': self.failed,
            'resources': dict(self.resources), 'closed': self.closed,
            'role_layouts': dict(self.role_layouts), 'geometry': {k: dict(v) for k, v in self.geometry.items()},
            'performance_qualification': 'invalid: per-helper diagnostic comparison fence',
            'counts': {key: dict(value) for key, value in self.counts.items()}}

    def __call__(self, gathered, workspace, output, group_start, status_ranges, layer):
        import torch
        if self.closed:
            raise RuntimeError('normalization parity callback already retired')
        engine = self.engine
        entered = False
        try:
            if self.failed or self.active:
                raise RuntimeError('normalization parity is closed, failed or concurrently entered')
            entry = next((e for e in self.roles if e['owner'].partial is workspace
                          and e['owner'].global_stats is output), None)
            if entry is None or type(layer) is not int or not 0 <= layer <= self.resources['maximum_layer']:
                raise ValueError('normalization parity role/layer is outside its admitted source')
            rows = gathered.shape[1]
            if (gathered.shape != (8, rows, 12, 2) or not 0 < rows <= entry['maximum']
                    or gathered.dtype != torch.float32 or not gathered.is_contiguous()
                    or gathered.data_ptr() != entry['owner'].partials.data_ptr()
                    or group_start not in (0, 4) or type(status_ranges) not in (tuple, list)):
                raise ValueError('normalization parity input differs from its prepared gather')
            if gathered.is_cuda and torch.cuda.is_current_stream_capturing():
                raise ValueError('normalization parity cannot run in CUDA capture')
            unique, mapped, cursor = [], [], 0
            for status, first, last in status_ranges:
                if (type(first) is not int or type(last) is not int or first != cursor or not first < last <= rows
                        or status.dtype != torch.int32 or status.shape != (2,) or not status.is_contiguous()
                        or status.device != workspace.device
                        or any(torch._C._overlaps(status, t) for t in (gathered, workspace, output))):
                    raise ValueError('normalization parity status ranges are malformed or overlap scratch')
                index = next((i for i, old in enumerate(unique) if old.data_ptr() == status.data_ptr()), len(unique))
                if index == len(unique):
                    unique.append(status)
                mapped.append((index, first, last))
                cursor = last
            if cursor != rows:
                raise ValueError('normalization parity ranges do not cover the gather')
            self.active = entered = True
            snapshot = entry['snapshot'][:8 * rows * 24].view(8, rows, 12, 2)
            snapshot.copy_(gathered)
            for i, status in enumerate(unique):
                entry['before'][i].copy_(status)
                entry['shadow'][i].copy_(entry['before'][i])
            result = self.original(gathered, workspace, output, group_start, status_ranges, layer)
            old = entry['old'][:2 * rows * 12].view(2, rows, 12)
            for index, first, last in mapped:
                for row in range(first, last):
                    pieces = snapshot[group_start:group_start + 4, row]
                    maximum, denominator = self.scalar(pieces[..., 0], pieces[..., 1],
                        status=entry['shadow'][index], layer=layer)
                    old[0, row].copy_(maximum)
                    old[1, row].copy_(denominator)
            for i, status in enumerate(unique):
                entry['actual'][i].copy_(status)
            masks = entry['output_equal'][:2 * rows * 12].view(2, rows, 12)
            for i in range(2):
                if result[i].shape != (rows, 12) or result[i].dtype != torch.float32:
                    raise ValueError('normalization parity candidate returned an unexpected schema')
                torch.eq(result[i].view(torch.int32), old[i].view(torch.int32), out=masks[i])
            input_mask = entry['input_equal'][:gathered.numel()].view_as(gathered)
            torch.eq(gathered.view(torch.int32), snapshot.view(torch.int32), out=input_mask)
            status_mask = entry['status_equal'][:2 * len(unique)].view(len(unique), 2)
            torch.eq(entry['actual'][:len(unique)], entry['shadow'][:len(unique)], out=status_mask)
            flags = entry['flags']
            torch.all(input_mask.flatten(), dim=0, out=flags[0])
            torch.all(masks.flatten(), dim=0, out=flags[1])
            torch.all(status_mask.flatten(), dim=0, out=flags[2])
            torch.logical_and(flags[0], flags[1], out=flags[3])
            torch.logical_and(flags[3], flags[2], out=flags[3])
            matches = bool(flags[3].item())  # Exactly one success-path comparison fence.
            failure = None if matches else RuntimeError('same-input normalization raw-bit/status parity mismatch')
            metadata_errors = []
            try:
                key = f"{entry['role']}:{layer}"
                prior = self.counts.get(key, {'calls': 0, 'rows': 0, 'mismatches': 0})
                if prior['calls'] >= 2**63 - 1 or prior['rows'] > 2**63 - 1 - rows:
                    raise OverflowError('normalization parity counters exhausted')
                self.counts[key] = {'calls': prior['calls'] + 1, 'rows': prior['rows'] + rows,
                                    'mismatches': prior['mismatches'] + int(not matches)}
                prior_rows = self.geometry.get(entry['role'], {'min_rows': rows, 'max_rows': rows})
                self.geometry[entry['role']] = {'min_rows': min(rows, prior_rows['min_rows']),
                                                 'max_rows': max(rows, prior_rows['max_rows'])}
                self.failed = not matches
                self._publish()
            except BaseException as error:
                if failure is None:
                    raise
                metadata_errors.append(error)
            if failure is not None:
                try:
                    tensors = {'gathered-before': snapshot.view(-1, 2), 'gathered-after': gathered.view(-1, 2),
                               'new-max': result[0], 'new-den': result[1], 'old-max': old[0], 'old-den': old[1],
                               'status-before': entry['before'][:len(unique)].view(torch.float32),
                               'status-new': entry['actual'][:len(unique)].view(torch.float32),
                               'status-old': entry['shadow'][:len(unique)].view(torch.float32)}
                    files = {name: self.writer(self.directory / f'{name}.npy', tensor)
                             for name, tensor in tensors.items()}
                    target = self.directory / 'mismatch.json'
                    if target.exists() or target.is_symlink():
                        raise ValueError('normalization parity mismatch receipt already exists')
                    write_json(target, {'role': entry['role'], 'layer': layer,
                        'rows': rows, 'group_start': group_start, 'sources': self.sources, 'files': files,
                        'status_storage': 'raw int32 bits viewed as float32; no numeric conversion',
                        'status_ranges': mapped})
                except BaseException as cleanup:
                    metadata_errors.append(cleanup)
                raise_failures(failure, metadata_errors)
            return result  # Preserve the candidate's actual tuple and tensor identities.
        except BaseException as primary:
            self.failed = True
            errors = []
            try:
                self._publish()
            except BaseException as error:
                errors.append(error)
            try:
                _observer_failure(engine, primary)
            except BaseException as error:
                raise_failures(error, errors)
            raise_failures(primary, errors)
        finally:
            if entered:
                self.active = False

    def install(self):
        if self.installed:
            raise ValueError('normalization parity is already installed')
        original_record = self.original_record
        def record(*, credit=None):
            result = original_record(**({} if credit is None else {'credit': credit}))
            published = self.published
            if credit is not None:
                credit(published)
            return {**result, 'normalization_parity': published}
        def close(*args, **kwargs):
            if self.closed:
                raise RuntimeError('normalization parity lifecycle callback already retired')
            engine = self.engine
            try:
                result = self.original_close(*args, **kwargs)
                if engine._closed is not True or self.active:
                    raise RuntimeError('normalization parity callbacks did not prove retirement')
            except BaseException as primary:
                _observer_failure(engine, primary)
                raise  # Keep binding/views for unresolved owners; never retry here.
            errors = []
            for restore in (lambda: self._restore_module(),
                            lambda: setattr(self.observations, 'record', self.original_record),
                            lambda: setattr(engine, 'close', self.old_close_attribute) if self.had_close else delattr(engine, 'close')):
                try:
                    restore()
                except BaseException as error:
                    errors.append(error)
            self.closed = True
            self.roles.clear()  # Original close fenced owners and drained callbacks.
            published = False
            try:
                self._publish()
                published = True
            except BaseException as error:
                errors.append(error)
            if published:
                try:
                    target = self.directory / 'summary.json'
                    if target.exists() or target.is_symlink():
                        raise ValueError('normalization parity summary already exists')
                    write_json(target, self.published)
                except BaseException as error:
                    errors.append(error)
            self.engine = self.original_close = self.original_record = self.observations = None
            self.record_binding = self.close_binding = self.original = self.scalar = self.writer = self.module = None
            if errors:
                primary = errors[0]
                try:
                    _observer_failure(engine, primary)
                except BaseException as error:
                    raise_failures(error, errors[1:])
                raise_failures(primary, errors[1:])
            return result
        self.record_binding, self.close_binding = record, close
        try:
            self.module.merge_global_statistics = self
            self.observations.record = record
            self.engine.close = close
            self.installed = True
        except BaseException as primary:
            errors = []
            for restore in (lambda: self._restore_module(rollback=True),
                            lambda: setattr(self.observations, 'record', self.original_record),
                            lambda: setattr(self.engine, 'close', self.old_close_attribute) if self.had_close else vars(self.engine).pop('close', None)):
                try:
                    restore()
                except BaseException as error:
                    errors.append(error)
            raise_failures(primary, errors)

    def _restore_module(self, *, rollback=False):
        if rollback and self.module.merge_global_statistics is self.original:
            return
        if self.module.merge_global_statistics is not self:
            raise RuntimeError('normalization parity module binding changed before retirement')
        self.module.merge_global_statistics = self.original


def prepare_normalization_parity(engine, resources, output):
    if (engine._owned_qualification_host != (resources['total_global_host_obligation_bytes'], resources['input_sha256'])
            or engine.retention_catalog.host_resources.baseline_host_bytes < resources['total_global_host_obligation_bytes']):
        raise ValueError('normalization parity obligation is absent from original host admission')
    observer = _NormalizationParity(engine, output, resources)
    observer.install()
    return observer.published


def _observer_failure(engine, primary):
    engine._abort_comm = True
    try:
        if not notify_failure():
            raise RuntimeError("history attestation lost its TP supervisor capability")
    except BaseException as cleanup:
        raise_failures(primary, [cleanup])


def _commit_trace_spec(args):
    events = getattr(args, 'commit_trace_events', None)
    nvtx = getattr(args, 'commit_trace_nvtx', False)
    if events is None:
        if nvtx:
            raise ValueError('commit NVTX markers require an explicit finite trace')
        return None
    if any(getattr(args, name, None) for name in ('teacher_plan', 'benchmark_client',
            'router_row_input', 'history_input')):
        raise ValueError('commit trace is a separate finite diagnostic, not a closed teacher/history operation')
    if __package__:
        from .v100_commit_trace import resources
    else:
        from v100_commit_trace import resources
    plan = resources(events)
    plan['nvtx'] = bool(nvtx)
    plan['global_host_obligation_bytes'] = 8 * (
        plan['retained_event_bytes'] + plan['serialization_bytes'] + 8192)
    if plan['global_host_obligation_bytes'] > sys.maxsize:
        raise ValueError('commit trace global host obligation exceeds its size domain')
    return plan


def _finish_commit_trace(trace, engine, output, spec):
    report = trace.close()  # CLI has drained and retired the actual engine first.
    report.update(rank=engine.rank, world=engine.tp, requested=spec,
                  engine_closed=engine._closed is True, engine_aborted=engine._abort_comm is not False,
                  observer_source_sha256=_source_sha(Path(__file__)))
    if report['engine_closed'] is not True or report['engine_aborted']:
        report['complete'] = False
    if len(json.dumps(report, allow_nan=False).encode()) > spec['serialization_bytes']:
        raise ValueError('commit diagnostic exceeded its admitted encoding bound')
    target = Path(output) / f'commit-trace-rank-{engine.rank}.json'
    if target.exists() or target.is_symlink():
        raise ValueError('commit diagnostic receipt already exists')
    write_json(target, report)
    if report['complete'] is not True:
        raise RuntimeError('commit diagnostic incomplete; model outcome remains independent')


class _HistoryAttestation:
    """One private worker-boundary witness; never installed on normal serving."""
    def __init__(self, engine, data, input_sha, output, resources):
        self.engine, self.data, self.input_sha = engine, data, input_sha
        self.output, self.resources = Path(output), resources
        self.original = engine.multi.round
        self.installed = False

    def install(self):
        if self.installed:
            raise ValueError("history observer is already installed")
        self.engine.multi.round = self.round
        self.installed = True

    def snapshot(self):
        engine, data = self.engine, self.data
        decoder, tp = engine.multi, engine.tp_decoder
        streams = [stream for stream in decoder.streams.values() if not stream.done]
        if len(streams) != data["C"] or decoder.filling:
            return None
        if any(stream.st is None or stream.st.pos < data["N"] for stream in streams):
            return None
        if len({id(stream.st) for stream in streams}) != data["C"]:
            raise ValueError("history attestation found aliased active States")
        coordinate = engine.w.meta["coordinate_profile_identity"]
        numerical = ("owned-expert-native", engine._owned_expert_plan.identity)
        rows = []
        for stream in streams:
            state = stream.st
            if (not any(state is owned for owned in decoder.states) or getattr(state, "_history_error", None) is not None
                    or state.coordinate_profile_identity != coordinate
                    or numerical not in state.kv_identity
                    or ("coordinate-profile-v1", coordinate) not in state.kv_identity
                    or len(stream.prompt) != data["N"] or stream.count != data["R"]):
                raise ValueError("history attestation owner differs from its prepared identity/geometry")
            rows.append({"wire": tp.wires[id(stream)], "slot": state.history_slot,
                         "slot_generation": state.slot_generation, "ownership_generation": state.history_generation,
                         "cached_length": state.pos, "prompt_tokens": len(stream.prompt),
                         "prompt_token_ids_sha256": _prompt_digest(stream.prompt)})
        if ({row["prompt_token_ids_sha256"] for row in rows} != set(data["prompt_token_ids_sha256"])
                or len({row["wire"] for row in rows}) != data["C"]
                or len({row["slot"] for row in rows}) != data["C"]
                or any(type(row[key]) is not int or row[key] < 0 for row in rows
                       for key in ("wire", "slot", "slot_generation", "ownership_generation", "cached_length"))
                or type(tp.prefix_operation) is not int or tp.prefix_operation < 0):
            raise ValueError("history attestation corpus, wire, slot or generation mismatch")
        return {"schema": "tensorfold.private-owned-history-v1", "rank": engine.rank, "world": engine.tp,
                "prefix_sequence": tp.prefix_operation, "input_sha256": self.input_sha,
                "C": data["C"], "N": data["N"], "R": data["R"], "L": data["L"],
                "coordinate_profile_identity": coordinate, "plan_identity": list(engine._owned_expert_plan.identity),
                "owners": sorted(rows, key=lambda row: row["wire"]), "resources": self.resources,
                "scope": "coherent pre-round logical owner/device-stream order; not wall-clock GPU completion",
                "qualification": "requires all-eight common facts, successful TP ACKs and bound client outputs",
                "run_identity": engine._history_run_identity, "server_instance_id": engine._history_server_instance}

    def round(self):
        try:
            witness = self.snapshot()
        except BaseException as primary:
            _observer_failure(self.engine, primary)
            raise
        # A model failure retains its exact existing propagation and never publishes.
        result = self.original()
        if witness is not None:
            try:
                target = self.output / f"history-rank-{self.engine.rank}.json"
                if target.exists() or target.is_symlink():
                    raise ValueError("history attestation receipt already exists")
                encoded = sum(len(chunk) for chunk in json.JSONEncoder(indent=1, allow_nan=False).iterencode(witness)) + 1
                if encoded > self.resources["receipt_bytes_limit"]:
                    raise ValueError("history receipt exceeds its prepaid output bound")
                write_json(target, witness)
                self.engine.multi.round = self.original  # Restore before any subsequent owner operation.
            except BaseException as primary:
                _observer_failure(self.engine, primary)
                raise
        return result



def _trace_c24_only(plan):
    """Additional region checks AFTER canonical load_matrix validation."""
    cohorts = plan['cohorts']
    if not cohorts or any(c['level'] != 24 for c in cohorts):
        return False  # Mixed matrices retain ONLY the incumbent C1/C8 schedule.
    if (plan.get('profile') != 'q8' or plan.get('admin_cap') != 24
            or plan.get('context') != 262144
            or not any(not c['warmup'] for c in cohorts)
            or any(type(c['warmup']) is not bool or len(c['requests']) != 24
                   or (not c['warmup'] and c['tokens'] < 4) for c in cohorts)):
        raise ValueError('C24 trace requires a canonical Q8/full24 plan and at least four measured outputs')
    return True


def _trace_c24_preflight(args, cli):
    if (args.draft_method != 'none' or str(args.parallel).lower() == 'auto'
            or cli._parallel(args.parallel) != 24 or args.context != 262144
            or args.kv_dtype != 'int8'):
        raise ValueError('C24 trace requires explicit NONE/Q8/full24/context262144 before construction')


def _trace_prefill_selection(plan, start):
    if type(start) is not int or not 0 <= start <= sys.maxsize:
        raise ValueError('scheduler trace prefill start must be a nonnegative integer')
    if _trace_c24_only(plan):
        if start:
            raise ValueError('C24-only trace excludes a prefill selector')
        return None
    cohort = next((c for c in plan['cohorts'] if not c['warmup'] and c['level'] == 1), None)
    if cohort is None:
        if start:
            raise ValueError('late scheduler trace requires a nonwarmup C1 cohort')
        return None  # Preserve existing C8-only diagnostic plans.
    if start >= cohort['input_tokens']:
        raise ValueError('scheduler trace prefill start must precede the designated prompt end')
    request = cohort['requests'][0]
    return {'run_identity': plan['run_identity'], 'cohort_id': cohort['id'],
            'prompt_sha256': request['prompt_sha256'], 'input_tokens': cohort['input_tokens'],
            'requested_start': start, 'status': 'UNSUPPORTED_NOT_OBSERVED', 'reason': None, 'P3_valid': False,
            'boundary_start': None, 'actual_start': None, 'actual_count': None,
            'slot': None, 'slot_generation': None, 'state_pos': None, 'actual_state_pos': None, 'wire': None,
            'prefix_operation': None, 'prepared_calls': 0, 'prepared_returned': False,
            'scope': 'one rank0 TP adapter round; logical host geometry, diagnostic and P3-invalid'}


class _SchedulerTrace:
    """At most five incumbent calls OR two C24-only decode calls, rank zero."""
    def __init__(self, engine, output, *, factory=None, prefill_start=0, selection=None):
        if type(prefill_start) is not int or not 0 <= prefill_start <= sys.maxsize:
            raise ValueError('scheduler trace prefill start must be a nonnegative integer')
        self.engine, self.output, self.factory = engine, Path(output), factory
        self.counts, self.active = {}, False
        self.c24_only, self.region_bound, self.capture_limit = False, False, 5
        self.prefill_start, self.selection = prefill_start, selection
        self.plan_bound = selection is not None
        self.prefill_owner = self.prefill_state = self.profiler = self.failed = None
        self.capture_errors = [None] * 7  # Acquired before profiler/hook effects.
        self.capture_error_order = (0, 1, 2, 3, 6, 4, 5)  # Geometry precedes receipt/publication.
        self.notification_errors = [None] * 2
        self.transport = raise_failures
        self.headroom_check = None

    def label(self, observer):
        if observer.current is None:
            return None
        if not self.region_bound:
            self.c24_only = _trace_c24_only(observer.plan)
            self.capture_limit = 2 if self.c24_only else 5
            self.region_bound = True
        if not self.plan_bound:
            self.selection = _trace_prefill_selection(observer.plan, self.prefill_start)
            self.plan_bound = True
        cohort = next(c for c in observer.plan['cohorts'] if c['id'] == observer.current)
        if cohort['warmup'] or cohort['level'] not in ((24,) if self.c24_only else (1, 8)):
            return None
        live = [s for s in observer.decoder.streams.values() if not s.done and s.st is not None]
        if self.c24_only:
            # A count alone cannot identify the current cohort's actual streams.
            for stream in live:
                wire = self.engine.tp_decoder.wires.get(id(stream))
                identity = observer.identities.get(wire)
                if identity is None or identity[0]['id'] != cohort['id']:
                    return None
        selection = self.selection
        if (selection is not None and cohort['id'] == selection['cohort_id']
                and observer.decoder.filling and not self.counts.get('prefill')):
            filling = [s for s in observer.decoder.filling if not s.done]
            if len(filling) != 1 or live and any(s is not filling[0] for s in live):
                selection['reason'] = 'not_one_filling_owner'
                return None
            stream = filling[0]
            fill = observer.decoder.fills.get(stream.sid)
            state = stream.st
            wire = self.engine.tp_decoder.wires.get(id(stream))
            identity = observer.identities.get(wire)
            if (state is None or fill is None or len(fill) < 3 or type(fill[2]) is not int
                    or type(wire) is not int or identity is None or identity[0]['id'] != selection['cohort_id']
                    or identity[1]['prompt_sha256'] != selection['prompt_sha256']):
                selection['reason'] = 'missing_or_foreign_owner'
                return None
            cancel = getattr(stream, 'cancel', None)
            if cancel is not None and cancel.is_set():
                selection['reason'] = 'cancelled_before_selection'
                return None
            cursor = fill[2]
            if not self.prefill_start <= cursor < selection['input_tokens']:
                return None
            self.counts['prefill'] = 1
            self.prefill_owner, self.prefill_state = stream, state
            selection.update(status='SELECTED_UNVERIFIED', reason=None, boundary_start=cursor,
                slot=state.history_slot, slot_generation=state.slot_generation, state_pos=state.pos, wire=wire)
            return ('c1-cold-prefill-containing-round' if self.prefill_start == 0 else
                    f'c1-prefill-start-ge-{self.prefill_start}-round')
        key = str(cohort['level'])
        if (not observer.decoder.filling and len(live) == cohort['level']
                and all(len(s.out) > 1 for s in live) and self.counts.get(key, 0) < 2):
            self.counts[key] = self.counts.get(key, 0) + 1
            return f'c{key}-warmed-decode-round-{self.counts[key]}'
        return None

    def note_prepare(self, pieces):
        if self.prefill_owner is None:
            return
        selection = self.selection
        selection['prepared_calls'] += 1
        if len(pieces) != 1:
            selection['reason'] = 'not_one_actual_prefill_piece'
            return
        stream, start, count = pieces[0]
        if (stream is not self.prefill_owner or type(start) is not int or type(count) is not int or count <= 0
                or start != selection['boundary_start'] or start + count > selection['input_tokens']
                or stream.st is not self.prefill_state or stream.st.history_slot != selection['slot']
                or stream.st.slot_generation != selection['slot_generation']):
            selection['reason'] = 'actual_prefill_geometry_differed'
            return
        selection['actual_start'], selection['actual_count'] = start, count
        selection['actual_state_pos'] = stream.st.pos

    @staticmethod
    def process():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"io": Path("/proc/self/io").read_text(), "minor_faults": usage.ru_minflt,
                "major_faults": usage.ru_majflt, "maxrss_kib_cumulative": usage.ru_maxrss}

    def capture(self, label, callback):
        if threading.current_thread() is not self.engine.scheduler.thread:
            raise RuntimeError('profiler must be owned by the actual scheduler thread')
        if self.failed is not None:
            raise self.failed  # No retry after uncertain profiler retirement.
        if self.active:
            raise RuntimeError('scheduler profiler capture is reentrant')
        if self.headroom_check is not None:
            self.selection['profiler_entry_headroom'] = self.headroom_check()
        errors = self.capture_errors
        for index in range(len(errors)):
            errors[index] = None
        if self.factory is None:
            import torch
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False,
                with_stack=False)
        else:
            profiler = self.factory()
        self.profiler = profiler  # Journal before the foreign profiler can become active.
        primary, entered, result = None, False, None
        before = started = entered_at = work_end = stopped = None
        trace_bytes = None
        prefill = self.prefill_owner is not None
        try:
            before, started = self.process(), time.monotonic()
            profiler.__enter__()
            entered = True
            self.active = True
            entered_at = time.monotonic()
            result = callback()
        except BaseException as error:
            primary = error
        finally:
            self.active = False
            try:
                work_end = time.monotonic()
            except BaseException as error:
                errors[0] = error
            if entered:
                try:
                    profiler.__exit__(type(primary) if primary is not None else None, primary,
                                      primary.__traceback__ if primary is not None else None)
                except BaseException as error:
                    errors[1] = error
                try:
                    stopped = time.monotonic()
                except BaseException as error:
                    errors[2] = error
                try:
                    trace = self.output / (label + '.json')
                    profiler.export_chrome_trace(str(trace))
                    trace_bytes = trace.stat().st_size
                except BaseException as error:
                    errors[3] = error
                if prefill:
                    selection = self.selection
                    try:
                        cancel = getattr(self.prefill_owner, 'cancel', None)
                        if cancel is not None and cancel.is_set():
                            selection['reason'] = 'cancelled_during_selected_round'
                        observed = (primary is None and selection['prepared_calls'] == 1
                            and selection['prepared_returned'] and selection['actual_count'] is not None
                            and selection['reason'] is None and errors[0] is None and errors[1] is None
                            and errors[2] is None and errors[3] is None)
                        selection['status'] = ('FAILED' if primary is not None or errors[0] is not None
                            or errors[1] is not None or errors[2] is not None or errors[3] is not None else
                            'OBSERVED' if observed else 'UNSUPPORTED_NOT_OBSERVED')
                        selection['prefix_operation'] = getattr(self.engine.tp_decoder, 'prefix_operation', None)
                    except BaseException as error:
                        errors[6] = error
                        selection['status'] = 'FAILED'
                try:
                    write_json(self.output / (label + '-receipt.json'), {
                        'rank': self.engine.rank, 'thread': threading.current_thread().name,
                        'thread_ident': threading.get_ident(), 'label': label,
                        'profiler_enter_seconds': None if entered_at is None or started is None else entered_at - started,
                        'work_call_seconds': None if work_end is None or entered_at is None else work_end - entered_at,
                        'profiler_stop_seconds': None if stopped is None or work_end is None else stopped - work_end,
                        'export_seconds': None if stopped is None else time.monotonic() - stopped,
                        'process_before': before, 'process_after': self.process(),
                        'primary': None if primary is None else type.__dict__['__name__'].__get__(type(primary)),
                        'post_errors': [type.__dict__['__name__'].__get__(type(error)) for error in errors if error is not None],
                        'exported_trace_bytes': trace_bytes,
                        'prefill_selection': self.selection if prefill else None, 'P3_valid': False,
                        'trace_region': 'c24-only' if self.c24_only else 'incumbent-c1-c8',
                        'capture_limit': self.capture_limit,
                        'resource_scope': f'source bounds capture count to {self.capture_limit} calls; operator owns profiler memory and filesystem containment',
                        'scope': 'rank0 TP adapter round; inclusive prefill/decode/control, diagnostic and P3-invalid; peers untraced'})
                except BaseException as error:
                    errors[4] = error
            if prefill:
                if not entered:
                    self.selection['status'] = 'FAILED' if primary is not None else 'UNSUPPORTED_NOT_OBSERVED'
                try:
                    write_json(self.output / 'prefill-selection.json', self.selection)
                except BaseException as error:
                    errors[5] = error
        index = 0
        while primary is None and index < len(self.capture_error_order):
            primary = errors[self.capture_error_order[index]]
            index += 1
        if primary is not None:
            self.failed = primary  # Retain profiler/engine/borrowed owner on uncertainty.
            index = 0
            while index < len(errors):
                if errors[index] is None:
                    errors[index] = primary
                index += 1
            try:
                self.transport(primary, errors)
            except BaseException as failure:
                if failure is primary:
                    raise
                raise primary from failure
        self.profiler = self.prefill_owner = self.prefill_state = None
        return result

    def wrap(self, owner, name, label):
        original = getattr(owner, name)
        def wrapped(*args, **kwargs):
            if not self.active:
                return original(*args, **kwargs)
            import torch
            with torch.profiler.record_function(label):
                return original(*args, **kwargs)
        setattr(owner, name, wrapped)

    def install(self, observer):
        from tensorfold.families.qwen4_exp.cuda import history_owner
        from tensorfold.cuda.sm70 import family_ops

        adapter, decoder = self.engine.tp_decoder, observer.decoder
        for owner, name, label in ((adapter, "_invoke", "TP.publish_execute_vote_retire"),
                (adapter, "_execute", "TP.local_dispatch"),
                (decoder, "_read_ahead", "PLE.read_ahead"),
                (self.engine.w.meta["input_preparation_context"], "prepare", "input_prepare.agree_vote_retire"),
                # These regular functions consume the tile generator inside the
                # label. _attention_layer resolves their module globals at use.
                (history_owner, "partial_statistics", "attention.partial_statistics"),
                (history_owner, "merge_statistics", "attention.global_statistics"),
                (history_owner, "normalized_partial", "attention.normalized_partial"),
                # Half glue resolves these entrypoints dynamically. Wrapping
                # _torch_* instead would miss the cached _reference aliases.
                (family_ops, "hc_writeback", "HC.writeback"),
                (family_ops, "hc_normed", "HC.normed"),
                (family_ops, "hc_act", "HC.act"),
                (family_ops, "hc_mix", "HC.mix")):
            self.wrap(owner, name, label)
        prepare = decoder._prepare_pass
        def preparing(*args, **kwargs):
            if not self.active:
                return prepare(*args, **kwargs)
            import torch
            with torch.profiler.record_function('PLE.prepare_frame_stage'):
                self.note_prepare(args[0] if args else kwargs['pieces'])
                result = prepare(*args, **kwargs)
                if self.prefill_owner is not None:
                    self.selection['prepared_returned'] = True
                return result
        decoder._prepare_pass = preparing
        if self.selection is not None:
            write_json(self.output / 'prefill-selection.json', self.selection)
        original = adapter.round
        def failure(primary):
            self.failed = primary
            try:
                _observer_failure(self.engine, primary)
            except BaseException as notification:
                if notification is primary:
                    raise
                self.notification_errors[0], self.notification_errors[1] = notification, primary
                try:
                    self.transport(primary, self.notification_errors)
                except BaseException as failure:
                    if failure is primary:
                        raise
                    raise primary from failure
            raise primary
        def rounded():
            try:
                label = self.label(observer)
            except BaseException as primary:
                failure(primary)
            if label is None:
                return original()
            try:
                return self.capture(label, original)
            except BaseException as primary:
                failure(primary)
        adapter.round = rounded


def _trace_inventory(engine, output):
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat, GGUFPacked
    from tensorfold.cuda.sm70 import family_ops
    buffers = {}
    for name in ("buf", "pbuf", "mbuf"):
        buf = getattr(engine.multi, name)
        buffers[name] = None if buf is None else {"moe_y_dtype": str(buf.moe.y.dtype),
            "moe_y_shape": list(buf.moe.y.shape), "rows": buf.rows}
    matrices = []
    def matrix(name, value):
        if value is None:
            return
        if not isinstance(value, (GGUFDense, GGUFConcat, GGUFPacked)):
            raise ValueError("trace inventory requires an actual declared GGUF matrix")
        matrices.append({"role": name, "class": type(value).__name__,
            "implementation": None if isinstance(value, GGUFPacked) else value.implementation,
            "n": value.n, "k": value.k})
        if isinstance(value, GGUFConcat):
            for index, part in enumerate(value.parts):
                matrices.append({"role": f"{name}.part{index}", "class": type(part).__name__,
                    "implementation": None if isinstance(part, GGUFPacked) else part.implementation,
                    "n": part.n, "k": part.k})
    def layer_sample(name, layer):
        for role in ("attn_hc", "mlp_hc"):
            pair = getattr(layer, role)
            matrix(name + "." + role + ".down", pair.down)
            matrix(name + "." + role + ".up", pair.up)
        if layer.gdn is not None:
            matrix(name + ".gdn.proj", layer.gdn.proj)
            matrix(name + ".gdn.out", layer.gdn.out)
        if layer.attn is not None:
            matrix(name + ".attn.proj", layer.attn.proj)
            matrix(name + ".attn.o", layer.attn.o)
        if layer.ple is not None:
            matrix(name + ".ple.key", layer.ple.key)
            matrix(name + ".ple.value", layer.ple.value)
        for index, value in enumerate(layer.moe.experts.shared):
            matrix(f"{name}.shared{index}", value)
    matrix("head", engine.w.head)
    selected = {0}
    for role in ("gdn", "attn", "ple"):
        selected.add(next(index for index, layer in enumerate(engine.w.layers)
                          if getattr(layer, role) is not None))
    for index in sorted(selected):
        layer_sample(f"target-layer{index}", engine.w.layers[index])
    if engine.w.mtp is not None:
        matrix("mtp.fc_e", engine.w.mtp.fc_e)
        matrix("mtp.fc_h", engine.w.mtp.fc_h)
        layer_sample("mtp.layer", engine.w.mtp.layer)
    write_json(Path(output) / "inventory.json", {"rank": engine.rank, "buffers": buffers,
        "Dense_implementation": GGUFDense.implementation,
        "Concat_implementation": GGUFConcat.implementation,
        "sm70_profile": engine.w.meta.get("sm70_profile"),
        "family_native": family_ops._NATIVE_FAMILY,
        "actual_matrix_samples": matrices,
        "cold_scope": "cold request prefix after model/compiler warmup; no filesystem/cache-cold claim",
        "scope": "CPU metadata only; no tensor values copied; buf/pbuf/mbuf roles remain distinct"})


class _BenchmarkObserver:
    """Private current-stream inclusive envelopes; four events reused per cohort."""
    def __init__(self, engine, plan, *, events=None, timed=True, strict=False, segment_journal=None, preparation_witness=None, commit_trace=None):
        import torch
        from tensorfold.server.stats import INSTANCE_ID
        self.server_instance_id = INSTANCE_ID
        self.engine, self.plan = engine, plan
        self.decoder, self.observations = engine.multi, engine.multi.controller.work_observations
        self.timed, self.strict = timed, strict
        self.segment_journal = segment_journal
        self.preparation_witness = preparation_witness
        self.commit_trace = commit_trace
        self.finished, self.failed = False, None
        self.seen = set()
        self.failure_status = [None] * 2
        self.events = events if events is not None else ([torch.cuda.Event(enable_timing=True) for _ in range(4)] if timed else [])
        self.current = None
        self.records = {}
        self.identities = {}
        self.cases = {tuple(request["prefix"]): (cohort, request)
                      for cohort in plan["cohorts"] for request in cohort["requests"]}
        self.prefix_lengths = sorted({len(prefix) for prefix in self.cases})
        self.published = {"schema": "tensorfold.private-benchmark-v1", "run_identity": plan["run_identity"],
                          "server_instance_id": self.server_instance_id, "cohorts": {}}
        self.stream = None
        self._publish()

    def _invalid(self, message):
        error = ValueError(message)
        if self.strict:
            self.failed = error
            _owned_model_failure(self.engine, error, self.failure_status)
        _observer_failure(self.engine, error)
        raise error

    def _identity(self, stream):
        key = self.engine.tp_decoder.wires.get(id(stream))
        if type(key) is not int:
            self._invalid("benchmark request lost its stable TP wire identity")
        if key not in self.identities:
            match = next((self.cases[tuple(stream.prompt[:size])] for size in self.prefix_lengths
                          if tuple(stream.prompt[:size]) in self.cases), None)
            if match is None:
                if self.current is None and not self.strict:
                    return None  # Explicit pre-matrix unobserved smoke phase.
                self._invalid("request does not belong to the private benchmark plan")
            cohort, request = match
            marker = (cohort['id'], request['index'])
            if self.strict and marker in self.seen:
                self._invalid('closed benchmark repeated an already admitted request')
            if self.strict:
                self.seen.add(marker)
            if (len(stream.prompt) != cohort["input_tokens"] or stream.count != cohort["tokens"]
                    or _prompt_digest(stream.prompt) != request["prompt_sha256"]):
                self._invalid("actual benchmark prompt/reply differs from its plan")
            self.identities[key] = (cohort, request)
        return self.identities[key]

    def _cohort(self, stream):
        import torch
        identity = self._identity(stream)
        if identity is None:
            return None
        cohort, request = identity
        if self.current != cohort["id"]:
            if self.current is not None and not self.records[self.current].get("events_ready"):
                self._invalid("next benchmark cohort arrived before prior idle timer retirement")
            if self.strict and self.current is not None and len(self.records[self.current]['done']) != self.records[self.current]['level']:
                self._invalid('closed benchmark advanced before actual request completion')
            self.current = cohort["id"]
            self.records[self.current] = {"level": cohort["level"], "prefill_done": set(), "done": set(),
                "hashes": set(), "active_owner_highwater": 0, "prefill_batch_highwater": 0,
                "decode_batch_highwater": 0, "started": [False] * 4, "events_ready": not self.timed}
            self.stream = torch.cuda.current_stream(self.engine.w.device) if self.timed else None
        return cohort, request, self.records[self.current]

    def _record(self, index):
        if not self.timed:
            self.records[self.current]['started'][index] = True
            return
        import torch
        if torch.cuda.current_stream(self.engine.w.device) != self.stream:
            self._invalid("benchmark changed the actual critical CUDA stream")
        if self.records[self.current]["started"][index]:
            self._invalid("benchmark phase event was already recorded")
        try:
            self.events[index].record(self.stream)
        except BaseException as error:
            if self.strict:
                self.failed = error
                _owned_model_failure(self.engine, error, self.failure_status)
            _observer_failure(self.engine, error)
            raise
        self.records[self.current]["started"][index] = True

    def _observe(self):
        if self.current is None:
            return
        record = self.records[self.current]
        streams = [s for s in [*self.decoder.streams.values(), *self.decoder.filling]
                   if not s.done and s.st is not None]
        if len({id(s.st) for s in streams}) != len(streams):
            self._invalid("benchmark active owners alias GPU States")
        if any(self._identity(s)[0]["id"] != self.current for s in streams):
            self._invalid("benchmark active cohort was contaminated")
        record["active_owner_highwater"] = max(record["active_owner_highwater"], len(streams))
        self._publish()

    def _publish(self):
        rows = {}
        for key, record in self.records.items():
            rows[str(key)] = {name: record[name] for name in
                ("level", "active_owner_highwater", "prefill_batch_highwater", "decode_batch_highwater", "events_ready")}
            rows[str(key)].update(completed=len(record["done"]), prompt_hashes=sorted(record["hashes"]))
            for name in ("prefill_ms", "decode_ms"):
                rows[str(key)][name] = record.get(name)
        self.published = {"schema": "tensorfold.private-benchmark-v1", "run_identity": self.plan["run_identity"],
                          "server_instance_id": self.server_instance_id,
                          "clock": "rank0 actual CUDA current stream; overlapping inclusive envelopes/host gaps, no TP token multiplier", "cohorts": rows}
        if self.segment_journal is not None:
            self.published.update(P3_valid=False, diagnostic_only=True,
                scope='GDN parity fences invalidate all performance timings')
        elif self.preparation_witness is not None:
            self.published.update(P3_valid=False, diagnostic_only=True,
                scope='Q8 preparation parity fences invalidate all performance timings')
        if self.commit_trace is not None:
            self.published.update(P3_valid=False, diagnostic_only=True, commit_trace=self.commit_trace,
                scope='Commit-batch diagnostic instrumentation invalidates performance timings')

    def install(self):
        decoder, observations = self.decoder, self.observations
        admit, prepare, joined, take, round_, finish = (getattr(decoder, name) for name in
                                                       ("admit", "_prepare_pass", "_joined", "_take", "round", "finish"))
        target, record = observations.target, observations.record
        incomplete = getattr(observations, 'incomplete', None)
        if self.segment_journal is not None:
            if not callable(incomplete):
                raise ValueError('GDN segment journal lacks existing incomplete-frame authority')
            def incomplete_frame(primary):
                self.segment_journal.release_prefill()
                return incomplete(primary)
            observations.incomplete = incomplete_frame
        def admitted(stream):
            self._identity(stream)
            result = admit(stream)
            self._cohort(stream)
            self._observe()
            return result
        def preparing(pieces):
            for stream, start, count in pieces:
                identity = self._cohort(stream)
                if identity is None:
                    continue
                cohort, request, state = identity
                state["hashes"].add(request["prompt_sha256"])
                if count and not state["started"][0]:
                    self._record(0)
            result = prepare(pieces)
            if self.segment_journal is not None:
                self.segment_journal.segment('prepare-prefill', result[1], pieces)
            return result
        def joining(pieces, *args):
            if self.segment_journal is not None:
                self.segment_journal.prefill_joined(pieces)
            for stream, start, count in pieces:
                identity = self._cohort(stream)
                if identity is None:
                    continue
                cohort, request, state = identity
                if start + count == len(stream.prompt) and request["index"] not in state["prefill_done"]:
                    state["prefill_done"].add(request["index"])
                    if len(state["prefill_done"]) == cohort["level"]:
                        self._record(1)
            return joined(pieces, *args)
        def taking(stream, values):
            before = len(stream.out)
            result = take(stream, values)
            identity = self._cohort(stream)
            if identity is None:
                return result
            cohort, request, state = identity
            if len(stream.out) > before:
                if not state["started"][2]:
                    self._record(2)
                if stream.done and request["index"] not in state["done"]:
                    state["done"].add(request["index"])
                    if len(state["done"]) == cohort["level"]:
                        self._record(3)
            return result
        def targeted(segments, **kwargs):
            result = target(segments, **kwargs)
            if self.segment_journal is not None:
                self.segment_journal.segment('target-prefill' if kwargs.get('prefill', False) else 'target-decode', segments)
            if self.current is not None:
                name = "prefill_batch_highwater" if kwargs.get("prefill", False) else "decode_batch_highwater"
                self.records[self.current][name] = max(self.records[self.current][name],
                    len({id(state) for state, first, last in segments if last > first}))
            return result
        def rounded():
            witness = self.preparation_witness
            if witness is not None:
                witness.begin_round(self)
            try:
                result = round_()
            except BaseException as primary:
                if witness is not None:
                    witness.end_round(primary)
                raise
            if witness is not None:
                witness.end_round()
            self._observe()
            return result
        def finished(streams):
            if self.strict:
                for stream in streams:
                    identity = self._identity(stream)
                    if (identity is None or stream.error is not None or len(stream.out) != identity[0]['tokens']
                            or stream.cancel is not None and stream.cancel.is_set()):
                        self._invalid('closed benchmark request did not finish its exact planned work')
            result = finish(streams)
            for stream in streams:
                self.identities.pop(self.engine.tp_decoder.wires.get(id(stream)), None)
            self._observe()
            if self.strict:
                self.finished = (len(self.records) == len(self.plan['cohorts']) and all(
                    len(self.records[c['id']]['done']) == c['level'] and self.records[c['id']]['hashes'] == {
                        r['prompt_sha256'] for r in c['requests']} for c in self.plan['cohorts']))
            return result
        def recorded(*, credit=None):
            # Internal TP status readers never query/drain timers. HTTP reads
            # may query only a completely finished, idle model cohort.
            if (self.timed and self.current is not None and threading.current_thread() is not self.engine.scheduler.thread
                    and not self.decoder.streams and not self.decoder.filling):
                state = self.records[self.current]
                if (not state["events_ready"] and len(state["done"]) == state["level"]
                        and all(state["started"]) and all(event.query() for event in self.events)):
                    try:
                        state["prefill_ms"] = self.events[0].elapsed_time(self.events[1])
                        state["decode_ms"] = self.events[2].elapsed_time(self.events[3])
                    except BaseException as error:
                        if self.strict:
                            self.failed = error
                            _owned_model_failure(self.engine, error, self.failure_status)
                        _observer_failure(self.engine, error)
                        raise
                    if any(not math.isfinite(state[name]) or state[name] < 0 for name in ("prefill_ms", "decode_ms")):
                        self._invalid("invalid CUDA event envelope")
                    state["events_ready"] = True
                    self._publish()
            result = record(**({"credit": credit} if credit is not None else {}))
            published = self.published
            if credit is not None:
                credit(published)
            if self.segment_journal is not None:
                parity = self.segment_journal.published
                if credit is not None:
                    credit(parity)
                return {**result, "benchmark": published, 'gdn_batch_parity': parity}
            if self.preparation_witness is not None:
                witness = self.preparation_witness.published
                if credit is not None:
                    credit(witness)
                return {**result, 'benchmark': published, 'q8_preparation_parity': witness}
            return {**result, "benchmark": published}
        # The private before-listener capability is the only selector. Default
        # instances/class implementations retain their original methods.
        def guarded(callback):
            if not self.strict:
                return callback
            def run(*args, **kwargs):
                try:
                    return callback(*args, **kwargs)
                except BaseException as primary:
                    if self.segment_journal is not None:
                        self.segment_journal.release_prefill()
                    if self.failed is primary:
                        raise
                    self.failed = primary
                    _owned_model_failure(self.engine, primary, self.failure_status)
            return run
        decoder.admit, decoder._prepare_pass, decoder._joined, decoder._take = map(guarded, (admitted, preparing, joining, taking))
        decoder.round, decoder.finish = guarded(rounded), guarded(finished)
        observations.target, observations.record = guarded(targeted), guarded(recorded)


def prepare_benchmark(engine, plan, trace_output=None, *, trace_prefill_start=0, closed=None, segment_journal=None, preparation_witness=None, commit_trace=None):
    if trace_output is not None and _trace_c24_only(plan):
        decoder = engine.multi
        if (engine.w.mtp is not None or decoder.speculation.method != 'none'
                or decoder.speculation.maximum_proposals != 0 or len(decoder.states) != 24
                or engine.context_window != 262144):
            raise ValueError('C24 trace differs from actual NONE/full24/context262144 owner')
    if engine.rank != 0 and closed is None and segment_journal is None and preparation_witness is None:
        return
    path = Path(engine.model_dir) / "tokenizer.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != plan["tokenizer_sha256"]:
        raise ValueError("benchmark tokenizer differs from the actual prepared frontend")
    selection = None if trace_output is None else _trace_prefill_selection(plan, trace_prefill_start)
    observer = _BenchmarkObserver(engine, plan, timed=engine.rank == 0,
                                  strict=closed is not None or segment_journal is not None or preparation_witness is not None,
                                  segment_journal=segment_journal, preparation_witness=preparation_witness,
                                  commit_trace=commit_trace)
    if closed is not None:
        engine._owned_benchmark = observer  # Journal before callback installation.
        _prepare_closed_benchmark(engine, observer, closed)
    if segment_journal is not None:
        segment_journal.observer = observer
    if preparation_witness is not None:
        preparation_witness.observer = observer
    observer.install()
    if trace_output is not None and engine.rank == 0:
        _trace_inventory(engine, trace_output)
        tracer = _SchedulerTrace(engine, trace_output, prefill_start=trace_prefill_start, selection=selection)
        if closed is not None:
            tracer.headroom_check = lambda: _benchmark_headroom(engine, closed['trace'])
        tracer.install(observer)
    return observer


def _teacher_module():
    if __package__:
        from . import v100_rotorquant_quality
    else:
        import v100_rotorquant_quality
    return v100_rotorquant_quality


def _model_contract():
    if __package__:
        from . import v100_owned_model_contract
    else:
        import v100_owned_model_contract
    return v100_owned_model_contract


def _teacher_client_input(path, teacher, *, prepared=False):
    raw = _teacher_module()._owned_teacher_read(path, 8 << 20)
    contract = _model_contract().prepared_teacher_contract() if prepared else _model_contract()
    spec = contract.validate_teacher_client(_teacher_module()._owned_teacher_object(raw))
    if spec['teacher_plan_sha256'] != teacher['input_sha256'] or not 1 <= teacher['data']['max_frames'] <= 64:
        raise ValueError('one planned teacher client differs from its frozen plan/MAX64 region')
    return {'data': spec, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _prepared_teacher_input(operation, teacher, plan_path):
    """Validate the new fixed operation without changing historical reference DATA."""
    contract = _model_contract().prepared_teacher_contract()
    contract.validate_operation(operation)
    if (str(plan_path.absolute()) != operation['teacher_plan']['path']
            or teacher['input_sha256'] != operation['teacher_plan']['sha256']
            or teacher['resources']['input_bytes'] != operation['teacher_plan']['bytes']):
        raise ValueError('prepared teacher actual plan differs from closed operation')
    payloads, sizes = {}, []
    for label, maximum in (('case_authority', 8 << 20), ('execution_manifest', 16 << 20), ('teacher_client', 8 << 20)):
        expected = operation[label]
        raw = _teacher_module()._owned_teacher_read(expected['path'], maximum)
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('prepared teacher held DATA differs: ' + label)
        payloads[label] = _teacher_module()._owned_teacher_object(raw)
        sizes.append(len(raw))
    reference = None
    if teacher['data']['mode'] == 'replay':
        expected = operation['reference']['authority']
        raw = _teacher_module()._owned_teacher_read(expected['path'], 8 << 20)
        if len(raw) != expected['bytes'] or hashlib.sha256(raw).hexdigest() != expected['sha256']:
            raise ValueError('prepared teacher replay authority differs')
        reference = _teacher_module()._owned_teacher_object(raw)
    contract.validate_teacher_operation_plan(operation, teacher['data'], payloads['teacher_client'],
        prepared=teacher if reference is not None else None, reference=reference,
        case=payloads['case_authority'], profile=payloads['execution_manifest'])
    control = json.dumps(operation, indent=1, ensure_ascii=True, allow_nan=False).encode()
    if len(control) > 8 << 20:
        raise ValueError('prepared teacher control exceeds bounded metadata envelope')
    sizes.append(len(control))
    from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
    artifacts = payloads['execution_manifest'].get('artifacts')
    if (type(artifacts) is not dict or not artifacts or any(type(row) is not dict
            or type(row.get('size')) is not int or row['size'] < 0 for row in artifacts.values())):
        raise ValueError('prepared teacher execution manifest artifact census differs')
    return {'operation': operation, 'control': control, 'sizes': sizes,
            'manifest_canonical_sha256': digest(payloads['execution_manifest']),
            'manifest_artifact_files': len(artifacts),
            'manifest_artifact_bytes': sum(row['size'] for row in artifacts.values())}


def _prepared_teacher_credit(teacher, selected):
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        _json_parse_payload,
        _json_payload_policy,
        _json_tree_payload,
    )
    if 'prepared_teacher_input_bytes' in teacher['resources']:
        raise ValueError('prepared teacher control cannot be prepaid twice')
    policy = _json_payload_policy()
    # Parent and each of the eight workers own their parsed, bounded operation,
    # case/profile/client DATA through ordinary Engine/client retirement.
    one = sum(_json_parse_payload(size, policy)['total_bytes'] + _json_tree_payload(size, policy)
              for size in selected['sizes'])
    resources = dict(teacher['resources'])
    resources['prepared_teacher_input_bytes'] = list(selected['sizes'])
    resources['prepared_teacher_payload_bytes'] = 9 * one
    resources['global_host_obligation_bytes'] += 9 * one
    if resources['global_host_obligation_bytes'] > sys.maxsize:
        raise ValueError('prepared teacher host obligation exceeds selected size domain')
    teacher['resources'] = resources


def _prepared_teacher_startup(row, spec):
    return _model_contract().prepared_teacher_contract().validate_startup(row, spec)


_TEACHER_READINESS_ATTEMPTS = 3601


def _teacher_client_credit(teacher, client_bytes):
    """Parent-only client working set; original loading/serving bind pays it."""
    if type(client_bytes) is not int or not 0 < client_bytes <= 8 << 20:
        raise ValueError('bounded teacher client declaration required')
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        _json_payload_policy, _json_parse_payload, _json_tree_payload,
        _http_json_encoding_payload, _streamed_json_encoding_payload, _python_layout,
    )
    policy = _json_payload_policy()
    response = 1 << 20
    request = 2 * teacher['resources']['input_bytes'] + 4096
    # At most one full HTTP result plus initial/final profile graphs. Expected
    # readiness retries are connection-refused only (no response body); charge
    # even their possible retained exception frames, without relying on GC.
    attempts = _TEACHER_READINESS_ATTEMPTS
    refused = _json_tree_payload(4096, policy) + policy.frame_work
    parent = (_json_parse_payload(client_bytes, policy)['total_bytes'] + _json_tree_payload(client_bytes, policy)
              + 2 * (_json_parse_payload(teacher['resources']['receipt_bytes_limit'], policy)['total_bytes']
                     + _json_tree_payload(teacher['resources']['receipt_bytes_limit'], policy))
              + _http_json_encoding_payload(request, policy)['total_bytes']
              + 3 * (_json_parse_payload(response, policy)['total_bytes'] + _json_tree_payload(response, policy))
              + _streamed_json_encoding_payload(response, _python_layout())['total_bytes']
              + attempts * refused)
    resources = dict(teacher['resources'])
    resources['client_input_bytes'] = client_bytes
    resources['client_parent_payload_bytes'] = parent
    resources['global_host_obligation_bytes'] += parent
    teacher['resources'] = resources



def _owned_model_startup_fields(engine, profile, observer, role):
    from tensorfold.server.stats import INSTANCE_ID
    states = engine.multi.states
    fields = {role + '_profile': profile, role + '_worker_PID': os.getpid(), role + '_server_instance_id': INSTANCE_ID,
        role + '_windows': {'full_resident': engine.multi.full_resident, 'slots': len(states),
            'capacity_min': min(st.capacity for st in states), 'capacity_max': max(st.capacity for st in states),
            'limit_min': min(st.limit for st in states), 'limit_max': max(st.limit for st in states),
            'logical_min': min(st.logical_context for st in states), 'logical_max': max(st.logical_context for st in states)}}
    if observer is not None:
        fields[role + '_observer'] = observer['source']
    return fields


def _owned_client_service(stats, spec, instance, role, *, prepared_manifest=None):
    if __package__:
        from . import v100_adaptive_qualification as qualification
    else:
        import v100_adaptive_qualification as qualification
    effective = qualification.profile(stats)
    qualification.require_idle(stats)
    qualification.materialization(stats, effective)
    expected = spec['expected_service']
    expert = stats['capacity_plan']['expert_parallel']
    families = stats['runtime']['execution']['execution_families']
    prepared_teacher = spec.get('schema') == 'tensorfold.prepared-expert-teacher-client-v1'
    expected_trial = ({key: expected[key] for key in ('kernel_family', 'graph_family')}
                      if prepared_teacher else None)
    if (effective['instance_id'] != instance or effective['model']['id'] != spec['model']
            or effective['context_tokens'] != expected['context_tokens']
            or effective['resident_capacity'] != expected['resident_capacity']
            or effective['session_capacity']['configured_ceiling'] != expected['configured_ceiling']
            or effective['model']['kv_dtype'] != expected['kv_dtype']
            or any(expert.get(key) != expected[key] for key in ('tp', 'pp', 'etp', 'ep'))
            or families['effective'] != {'kernel_family': expected['kernel_family'], 'graph_family': expected['graph_family']}
            or families.get('trial') != expected_trial):
        raise ValueError(role + ' HTTP actual profile/instance differs from the frozen control')
    if prepared_teacher:
        binding = stats['runtime']['capacity_plan']['execution_profile_binding']
        if (prepared_manifest is None or binding.get('manifest_sha256') != prepared_manifest['manifest_canonical_sha256']
                or any(type(binding.get(key)) is not int for key in
                       ('verified_input_files', 'verified_input_bytes', 'content_passes', 'hashes_reused_this_binding'))
                or binding.get('verified_input_files') != prepared_manifest['manifest_artifact_files']
                or binding.get('verified_input_bytes') != prepared_manifest['manifest_artifact_bytes']
                # Initial NONE binding has no pairing/rebind path: the issuer
                # uses _capture and publishes literal reused=0 for this scope.
                or binding.get('content_passes') != 1 or binding.get('hashes_reused_this_binding') != 0
                or binding.get('shared_immutable_same_host') is not True
                or families.get('native_prepared') is not True or families.get('graph_rows') != 0):
            raise ValueError('prepared teacher HTTP sealed profile/native admission differs')
    return effective


def _owned_client_open(owner, role, *, deadline=None):
    import urllib.error
    if __package__:
        from . import openai_protocol as protocol
        from . import v100_rotorquant_live as live
    else:
        import openai_protocol as protocol
        import v100_rotorquant_live as live
    spec = owner.selected['data']
    raw = owner.output / (role + '-client-raw')
    raw.mkdir(mode=0o700)
    client = live.Client(spec['base'], raw, timeout=spec['socket_timeout_s'],
                         limits=protocol.Limits(response_bytes=spec['response_bytes']))
    deadline = time.monotonic() + spec['whole_timeout_s'] if deadline is None else deadline
    ready_deadline = min(deadline, time.monotonic() + spec['readiness_timeout_s'])
    control = live.ReadControl(deadline, owner.cancelled)
    while not all((owner.output / f'rank-{rank}.json').is_file() for rank in range(8)):
        if time.monotonic() >= ready_deadline:
            raise TimeoutError(role + ' startup receipts did not complete within readiness budget')
        control.check()
        owner.cancelled.wait(min(1.0, max(0., ready_deadline - time.monotonic())))
    startup = None
    for rank in range(8):
        row = _teacher_module()._owned_teacher_object(_teacher_module()._owned_teacher_read(owner.output / f'rank-{rank}.json', 8 << 20))
        windows, expected = row.get(role + '_windows', {}), spec['expected_service']
        if spec.get('schema') == 'tensorfold.prepared-expert-teacher-client-v1':
            _prepared_teacher_startup(row, spec)
        if (row.get('rank') != rank or row.get('mtp_banks') != 0 or windows.get('full_resident') is not True
                or windows.get('slots') != expected['resident_capacity']
                or windows.get('capacity_min') != windows.get('capacity_max')
                or windows.get('limit_min') != windows.get('limit_max')
                or windows.get('capacity_min', 0) < expected['context_tokens']
                or windows.get('limit_min', 0) < expected['context_tokens']
                or windows.get('logical_min') != expected['context_tokens']
                or windows.get('logical_max') != expected['context_tokens']):
            raise ValueError(role + ' startup lacks actual all-rank full-window materialization')
        if rank == 0:
            startup = row
        if role == 'benchmark' and row.get('tool_sha256') != owner.producer_sha:
            raise ValueError('benchmark startup source differs from actual owned client')
    instance = startup.get(role + '_server_instance_id')
    if type(instance) is not str or not instance or startup.get('state_count') != expected['resident_capacity']:
        raise ValueError('fresh ' + role + ' startup lacks its actual NONE/instance/State binding')
    initial, attempt = None, 0
    while initial is None:
        if attempt >= _TEACHER_READINESS_ATTEMPTS:
            raise TimeoutError(role + ' readiness exhausted its prepaid attempt count')
        control.check()
        if time.monotonic() >= ready_deadline:
            raise TimeoutError(role + ' HTTP readiness budget expired')
        try:
            initial = client.call(f'readiness-{attempt}', path='/stats', object_response=True,
                                  read_control=live.ReadControl(ready_deadline, owner.cancelled))
        except urllib.error.URLError as error:
            if type(error.reason) is not ConnectionRefusedError:
                raise
            attempt += 1
            owner.cancelled.wait(min(1., max(0., ready_deadline - time.monotonic())))
    prepared_manifest = owner.selected.get('prepared_manifest')
    observed = (_owned_client_service(initial, spec, instance, role) if prepared_manifest is None else
                _owned_client_service(initial, spec, instance, role, prepared_manifest=prepared_manifest))
    return client, control, observed, instance


class _TeacherClient:
    """One owned non-stream completion; no alternate serving/launch stack."""
    def __init__(self, teacher, selected, output):
        self.teacher, self.selected, self.output = teacher, selected, Path(output)
        self.cancelled = threading.Event()
        self.result = None
        self.post_attempts = 0

    def run(self):
        if __package__:
            from . import openai_protocol as protocol
        else:
            import openai_protocol as protocol
        spec, plan = self.selected['data'], self.teacher['data']
        client, control, before, instance = _owned_client_open(self, 'teacher')
        body = {'model': spec['model'], 'prompt': plan['input_ids'], 'max_tokens': plan['max_frames'],
                'temperature': 0, 'draft': False, 'stream': False, 'return_token_ids': True}
        self.post_attempts += 1
        reply = client.call('teacher-completion', body, path='/v1/completions', read_control=control)
        protocol.chunk_fields(reply)
        count = protocol.completion_usage(reply['usage'], plan['max_frames'], prompt=True)
        choices, runtime = reply.get('choices'), reply.get('tensorfold', {})
        ids = runtime.get('token_ids')
        if (type(choices) is not list or len(choices) != 1 or type(choices[0].get('text')) is not str
                or choices[0].get('index') != 0 or choices[0].get('finish_reason') not in ('stop', 'length')
                or type(ids) is not list or len(ids) != count or any(type(token) is not int or not 0 <= token < plan['vocab'] for token in ids)
                or runtime.get('token_sha') != hashlib.sha256(','.join(map(str, ids)).encode()).hexdigest()[:12]
                or reply['usage']['prompt_tokens'] != len(plan['input_ids']) or runtime.get('drafts') is not False
                or type(reply['usage'].get('total_tokens')) is not int
                or reply['usage']['total_tokens'] != len(plan['input_ids']) + count
                or any(type(runtime.get(key)) is not int or runtime[key] != 0 for key in ('drafted', 'accepted', 'cached'))
                or type(reply['usage'].get('prompt_tokens_details', {}).get('cached_tokens')) is not int
                or reply['usage']['prompt_tokens_details']['cached_tokens'] != 0):
            raise ValueError('teacher completion is not complete hashed cold NONE output')
        final = client.call('final-stats', path='/stats', object_response=True, read_control=control)
        prepared_manifest = self.selected.get('prepared_manifest')
        observed = (_owned_client_service(final, spec, instance, 'teacher') if prepared_manifest is None else
                    _owned_client_service(final, spec, instance, 'teacher', prepared_manifest=prepared_manifest))
        if observed != before:
            raise ValueError('teacher service identity/configuration changed during its one request')
        self.result = {'schema': 'tensorfold.owned-teacher-client-report-v1', 'complete': True,
                       'post_attempts': self.post_attempts, 'teacher_plan_sha256': self.teacher['input_sha256'],
                       'teacher_client_sha256': self.selected['sha256'], 'mode': plan['mode'],
                       'server_instance_id': instance, 'token_ids': ids, 'finish_reason': choices[0]['finish_reason'],
                       'completion_tokens': count, 'prompt_tokens': len(plan['input_ids']), 'cached_tokens': 0,
                       'drafts': False, 'drafted': 0, 'accepted': 0, 'client_protocol_passed': True,
                       'quality_acceptance': False, 'P3_valid': False}
        write_json(self.output / 'teacher-client-report.json', self.result)


def _owned_model_observer(output, role):
    """The closed operation supplies the current inert observer, never user code."""
    if role not in ('teacher', 'benchmark'):
        raise ValueError('closed owned model observer role required')
    contract = _model_contract()
    names = (contract.CHILD_ENV, contract.DRIVER_ENV, contract.DRIVER_SHA_ENV, contract.PREFIX_ENV)
    values = [os.environ.get(name) for name in names]
    if any(value is None for value in values):
        raise ValueError(role + ' client requires the complete closed child-observer operation')
    directory, source, expected, prefix_raw = values
    if (not contract.absolute(directory) or Path(directory).parent != Path(output).parent
            or not contract.absolute(source) or not contract.sha(expected)
            or _teacher_module()._owned_teacher_object('{"prefixes":' + prefix_raw + '}')['prefixes'] != contract.REQUIRED_PREFIXES):
        raise ValueError(role + ' child observer input differs from the closed model contract')
    info = Path(source).stat()
    if _teacher_module()._owned_teacher_hash(source, info.st_size, 8 << 20) != expected:
        raise ValueError('current staged child-observer source SHA differs')
    import importlib.util
    spec = importlib.util.spec_from_file_location('_owned_model_native_observer', source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)  # Inert __name__, no main/forced native load.
    if _teacher_module()._owned_teacher_hash(source, info.st_size, 8 << 20) != expected:
        raise ValueError('child observer changed while importing')
    return {'module': module, 'directory': Path(directory), 'source': {'path': source, 'bytes': info.st_size, 'sha256': expected},
            'resources': contract.child_observer_resources()}



def _teacher_model_observer(output):
    return _owned_model_observer(output, 'teacher')

def _teacher_operation_credit(teacher, client_bytes, observer):
    _teacher_client_credit(teacher, client_bytes)
    resources = dict(teacher['resources'])
    teacher['resources'] = _owned_observer_credit(resources, observer['resources'])


def _owned_observer_credit(resources, native):
    if 'child_native_observer' in resources:
        raise ValueError('owned native observer cannot be prepaid twice')
    return {**resources, 'child_native_observer': native,
            'global_host_obligation_bytes': resources['global_host_obligation_bytes'] + native['global_host_bytes']}


def _owned_model_failure(engine, primary, status):
    try:
        _observer_failure(engine, primary)
    except BaseException as notification:
        if notification is primary:
            raise
        status[0], status[1] = notification, primary
        try:
            raise_failures(primary, status)
        except BaseException as failure:
            if failure is primary:
                raise
            raise primary from failure
    raise primary


def _attach_owned_model_observer(engine, selected, *, role, complete, summary=None):
    """One shared native observer, after real fences and before positive close."""
    original = engine._close_resources
    attempted = False
    status = [None] * 2
    setattr(engine, '_' + role + '_native_observer', selected)
    def resources(*args, **kwargs):
        nonlocal attempted
        if attempted:
            raise RuntimeError(role + ' native-retirement observer already attempted; no retry')
        attempted = True
        result = original(*args, **kwargs)
        if complete():
            try:
                selected['module'].record_native_elfs(_model_contract().REQUIRED_PREFIXES,
                    rank=engine.rank, tag=role + '-closed', evidence_dir=selected['directory'],
                    limits=_model_contract().MODEL_OBSERVER_LIMITS)
            except BaseException as primary:
                _owned_model_failure(engine, primary, status)
        return result
    engine._close_resources = resources
    if summary is not None:
        close, close_status = engine.close, [None] * 2
        close_attempted = False
        def closing():
            nonlocal close_attempted
            if close_attempted:
                raise RuntimeError('owned benchmark close already attempted; uncertain owners retained')
            close_attempted = True
            primary, result = None, None
            try:
                result = close()
            except BaseException as error:
                primary = error
            retired = getattr(engine, '_lifecycle', None) is not None and getattr(engine, '_closed', False) is True
            if not retired and primary is None:
                primary = RuntimeError('owned benchmark lacks positive actual Engine retirement')
            try:
                summary(retired)
            except BaseException as error:
                close_status[0] = error
                if primary is None:
                    primary = error
            if primary is not None:
                if close_status[0] is None:
                    close_status[0] = primary
                close_status[1] = primary
                try:
                    raise_failures(primary, close_status)
                except BaseException as failure:
                    if failure is primary:
                        raise
                    raise primary from failure
            return result
        engine.close = closing


def _attach_teacher_native_observer(engine, selected):
    return _attach_owned_model_observer(engine, selected, role='teacher',
        complete=lambda: engine._owned_teacher_frames.finished and engine._owned_teacher_frames.failed is None)


def _teacher_file(path, root, *, rank=None):
    path, root = Path(path), Path(root)
    size = path.stat().st_size
    result = {'file': str(path.relative_to(root)), 'bytes': size,
              'sha256': _teacher_module()._owned_teacher_hash(path, size, size)}
    if rank is not None:
        result['rank'] = rank
    return result


def _teacher_model_result(output, teacher, selected, observer, authority):
    root = Path(output).parent
    native = []
    for rank in range(8):
        paths = list(observer['directory'].glob(f'teacher-closed-rank-{rank}-pid-*.json'))
        if len(paths) != 1:
            raise ValueError('teacher operation requires exactly one actual child native observation per rank')
        startup = _teacher_module()._owned_teacher_object(_teacher_module()._owned_teacher_read(Path(output) / f'rank-{rank}.json', 8 << 20))
        record = _teacher_module()._owned_teacher_object(_teacher_module()._owned_teacher_read(paths[0], 4 << 20))
        if (record.get('rank') != rank or record.get('PID') != startup.get('teacher_worker_PID')
                or record.get('operation_limits') != _model_contract().MODEL_OBSERVER_LIMITS
                or record.get('missing_required_prefixes') != []
                or record.get('observer', {}).get('sha256') != observer['source']['sha256']):
            raise ValueError('teacher child native record differs from its actual worker/source/contract')
        native.append(_teacher_file(paths[0], root, rank=rank))
    result = {'schema': _model_contract().RESULT_SCHEMA, 'teacher_plan_sha256': teacher['input_sha256'],
              'teacher_client_sha256': selected['sha256'], 'launch_exit': 0,
              'client_report': _teacher_file(Path(output) / 'teacher-client-report.json', root),
              'teacher_authority': _teacher_file(Path(output) / 'teacher-authority.json', root),
              'rank_receipts': [_teacher_file(Path(output) / row['file'], root, rank=row['rank']) for row in authority['rank_receipts']],
              'startup_receipts': [_teacher_file(Path(output) / row['file'], root, rank=row['rank']) for row in authority['startup_receipts']],
              'child_native_records': native}
    _model_contract().validate_model_result(result)
    write_json(Path(output) / 'teacher-model-result.json', result)
    return result


def _benchmark_module():
    if __package__:
        from . import bench_concurrent
    else:
        import bench_concurrent
    return bench_concurrent


def _source_sha(path, maximum=8 << 20):
    path = Path(path)
    return _teacher_module()._owned_teacher_hash(path, path.stat().st_size, maximum)


def _benchmark_client_input(path, plan_sha):
    raw = _teacher_module()._owned_teacher_read(path, 8 << 20)
    data = _model_contract().validate_benchmark_client(_teacher_module()._owned_teacher_object(raw))
    if data['benchmark_plan_sha256'] != plan_sha:
        raise ValueError('benchmark client differs from its actual immutable plan')
    return {'data': data, 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}


def _benchmark_resources(plan, plan_bytes, selected, trace, native):
    from tensorfold.families.qwen4_exp.cuda.retention_resources import (
        _json_payload_policy, _json_parse_payload, _json_tree_payload, _http_json_encoding_payload,
    )
    policy = _json_payload_policy()
    count = sum(c['level'] for c in plan['cohorts'])
    # Actual State24/bank48/cohort/source rows share the existing 1024B-row
    # qualification serializer bound. No materialized allocation credit.
    resources = _history_resources(24 + 48 + count + 8, plan_bytes)
    spec = selected['data']
    skeleton = {'model': spec['model'], 'prompt': [], 'max_tokens': 2, 'temperature': 0,
                'draft': True, 'ignore_eos': True, 'stream': True,
                'stream_options': {'include_usage': True}, 'return_token_ids': True}
    # load_matrix guarantees IDs<2^31: <=10 decimal digits plus one comma.
    request_bytes = len(json.dumps(skeleton).encode()) + 11 * max(c['input_tokens'] for c in plan['cohorts'])
    matrix_limit = plan_bytes + (6 * count + 4) * spec['response_bytes']
    worker = (_json_parse_payload(selected['bytes'], policy)['total_bytes']
              + _json_tree_payload(selected['bytes'], policy) + _json_tree_payload(request_bytes, policy))
    resources['per_rank_payload_bytes'] += worker
    resources['global_host_obligation_bytes'] += 8 * worker + resources['per_rank_payload_bytes']
    prepared = {'resources': {**resources, 'input_bytes': request_bytes}}
    _teacher_client_credit(prepared, selected['bytes'])  # Same bounded readiness/error ownership.
    resources = prepared['resources']
    extra = (_http_json_encoding_payload(request_bytes, policy)['total_bytes']
             + 2 * _json_tree_payload(matrix_limit, policy)
             + _json_parse_payload(matrix_limit, policy)['total_bytes'])
    resources['global_host_obligation_bytes'] += extra + trace['host_obligation_bytes']
    resources = _owned_observer_credit(resources, native)
    if resources['global_host_obligation_bytes'] > sys.maxsize:
        raise ValueError('closed benchmark host obligation exceeds size domain')
    resources.update(trace_grants=dict(trace), matrix_report_bytes=matrix_limit,
        parent_matrix_payload_bytes=extra, benchmark_client_bytes=selected['bytes'],
        scope='original immutable host admission; trace grant additional, no materialized credit; allocator/RSS separate')
    return resources


def _benchmark_sources(engine):
    values = _teacher_module()._owned_runtime_sources(engine)
    return {**{key: values[key] for key in ('engine', 'multi', 'decode', 'sampling')},
            'observer_tool': _source_sha(__file__)}


def _benchmark_headroom(engine, trace):
    import torch
    free, total = map(int, torch.cuda.mem_get_info(engine.w.device))
    if not 0 < total or free < trace['device_headroom_bytes']:
        raise MemoryError('closed diagnostic lacks declared fresh physical CUDA headroom')
    return {'free_bytes': free, 'total_bytes': total, 'required_free_bytes': trace['device_headroom_bytes'],
            'scope': 'fresh CUDA free after full States; admission observation, NOT a reservation or profiler peak guarantee'}


def _prepare_closed_benchmark(engine, observer, prepared):
    from tensorfold.families.qwen4_exp.kv_formats import get_pair
    decoder, expected = engine.multi, _model_contract().SERVICE
    states = decoder.states
    obligation = (prepared['resources']['global_host_obligation_bytes'], prepared['plan_sha256'])
    if (engine._owned_qualification_host != obligation
            or engine.retention_catalog.host_resources.baseline_host_bytes < obligation[0]):
        raise ValueError('benchmark resource grants differ from original immutable admission')
    if (getattr(engine, '_lifecycle', None) is None or getattr(engine, '_closed', None) is not False
            or engine.w.mtp is not None or decoder.speculation.method != 'none' or decoder.speculation.maximum_proposals != 0
            or engine.sm70_profile != 'reference' or engine.kv_pair.identity != get_pair('int8').identity or decoder.graphs is not None
            or engine.context_window != expected['context_tokens'] or not decoder.full_resident
            or len(states) != expected['resident_capacity'] or any(st.capacity < engine.context_window
                or st.limit < engine.context_window or st.logical_context != engine.context_window for st in states)
            or any(st.kv_identity != states[0].kv_identity for st in states)):
        raise ValueError('closed benchmark requires actual full C24 NONE/reference/noGraphs States')
    profile = bound_receipt(engine)
    def family(buf):
        return {'kernel_family': buf.owner_kernel_family, 'graph_family': buf.owner_graph_family}
    wanted = {'kernel_family': expected['kernel_family'], 'graph_family': expected['graph_family']}
    if family(decoder.buf) != wanted or family(decoder.pbuf) != wanted:
        raise ValueError('closed benchmark actual prepared family differs')
    coordinate = ('coordinate-profile-v1', profile['coordinate_profile_identity'])
    if (states[0].kv_identity[0] != engine.kv_pair.identity
            or ('logical-context-v1', engine.context_window) not in states[0].kv_identity
            or ('position-profile-v1', engine.position_profile.identity, engine.position_profile.epoch) not in states[0].kv_identity
            or coordinate not in states[0].kv_identity):
        raise ValueError('closed benchmark actual State numerical identity differs')
    windows = _owned_model_startup_fields(engine, profile, prepared['native'], 'benchmark')['benchmark_windows']
    semantics = {'method': decoder.speculation.method, 'state_kv_identity': states[0].kv_identity,
        'context': engine.context_window, 'resident_slots': len(states), **windows,
        'target_rows': decoder.buf.rows, 'prefill_rows': decoder.pbuf.rows,
        'target_family': family(decoder.buf), 'prefill_family': family(decoder.pbuf)}
    semantics.pop('slots')  # resident_slots is the single semantic count owner.
    observer.actual_profile = {**profile, 'benchmark_semantics': semantics}
    observer.sources = _benchmark_sources(engine)
    observer.headroom = _benchmark_headroom(engine, prepared['trace'])
    observer.prepared = prepared
    observer.directory = Path(prepared['output']) / f'benchmark-rank{engine.rank}'
    observer.directory.mkdir(mode=0o700)
    def summary(retired):
        current = _benchmark_sources(engine)
        if current != observer.sources:
            raise ValueError('benchmark source changed before positive retirement')
        cohorts = [{'id': c['id'], 'level': c['level'],
                    'completed': len(observer.records.get(c['id'], {}).get('done', ())),
                    'prompt_hashes': sorted(observer.records.get(c['id'], {}).get('hashes', ()))}
                   for c in observer.plan['cohorts']]
        write_json(observer.directory / 'summary.json', {
            'schema': 'tensorfold.owned-benchmark-rank-result-v1', 'rank': engine.rank, 'world': engine.tp,
            'closed': retired, 'finished': observer.finished, 'failed': observer.failed is not None or not retired or not observer.finished,
            'benchmark_plan_sha256': prepared['plan_sha256'], 'benchmark_client_sha256': prepared['client']['sha256'],
            'run_identity': observer.plan['run_identity'], 'server_instance_id': observer.server_instance_id,
            'actual_owned_profile': observer.actual_profile, 'runtime_sources': current, 'cohorts': cohorts,
            'headroom': observer.headroom, 'resources': prepared['resources'], 'qualification': False, 'P3_valid': False})
    _attach_owned_model_observer(engine, prepared['native'], role='benchmark',
        complete=lambda: observer.finished and observer.failed is None, summary=summary)


def _benchmark_trace_files(output, plan, trace):
    directory = Path(output) / 'trace'
    selection = _teacher_module()._owned_teacher_object(_teacher_module()._owned_teacher_read(directory / 'prefill-selection.json', 8 << 20))
    wanted = _trace_prefill_selection(plan, trace['prefill_start'])
    if (selection.get('status') != 'OBSERVED' or any(selection.get(key) != wanted[key] for key in
            ('run_identity', 'cohort_id', 'prompt_sha256', 'input_tokens', 'requested_start'))
            or selection.get('prepared_calls') != 1 or selection.get('prepared_returned') is not True
            or type(selection.get('actual_count')) is not int or selection['actual_count'] <= 0
            or type(selection.get('actual_start')) is not int
            or not trace['prefill_start'] <= selection['actual_start'] < selection['input_tokens']
            or selection['actual_start'] + selection['actual_count'] > selection['input_tokens']
            or any(type(selection.get(key)) is not int or selection[key] < 0 for key in
                   ('slot', 'slot_generation', 'wire', 'prefix_operation', 'actual_state_pos'))
            or not 0 <= selection['slot'] < 24):
        raise ValueError('closed benchmark lacks exactly observed planned prefill geometry')
    entry = selection.get('profiler_entry_headroom')
    if (type(entry) is not dict or any(type(entry.get(k)) is not int for k in
            ('free_bytes', 'total_bytes', 'required_free_bytes'))
            or entry['required_free_bytes'] != trace['device_headroom_bytes']
            or not entry['required_free_bytes'] <= entry['free_bytes'] <= entry['total_bytes']):
        raise ValueError('closed benchmark lacks its fresh physical profiler-entry admission observation')
    receipts = list(directory.glob('*-receipt.json'))
    chromes = [p for p in directory.glob('*.json') if p.name not in ('inventory.json', 'prefill-selection.json') and not p.name.endswith('-receipt.json')]
    label = ('c1-cold-prefill-containing-round' if trace['prefill_start'] == 0
             else f"c1-prefill-start-ge-{trace['prefill_start']}-round")
    if (len(receipts) != 1 or len(chromes) != 1 or receipts[0].name != label + '-receipt.json'
            or chromes[0].name != label + '.json' or chromes[0].stat().st_size <= 0):
        raise ValueError('closed benchmark requires exactly one prefill profiler window')
    receipt = _teacher_module()._owned_teacher_object(_teacher_module()._owned_teacher_read(receipts[0], 8 << 20))
    if ('primary' not in receipt or receipt.get('primary') is not None or receipt.get('post_errors') != []
            or receipt.get('prefill_selection') != selection or receipt.get('P3_valid') is not False
            or receipt.get('exported_trace_bytes') != chromes[0].stat().st_size):
        raise ValueError('closed benchmark profiler failed or source geometry changed')
    paths = {'inventory': directory / 'inventory.json', 'selection': directory / 'prefill-selection.json',
             'receipt': receipts[0], 'chrome': chromes[0]}
    if sum(path.stat().st_size for path in paths.values()) > trace['disk_bytes']:
        raise ValueError('closed benchmark trace exceeds declared disk extent')
    return paths


class _BenchmarkClient:
    def __init__(self, prepared, output):
        self.prepared, self.output = prepared, Path(output)
        self.selected = prepared['client']
        self.cancelled, self.result = threading.Event(), None
        self.post_attempts = 0
        self.producer_sha = _source_sha(__file__)

    def run(self):
        from types import SimpleNamespace
        spec, plan = self.selected['data'], self.prepared['plan']
        deadline = time.monotonic() + spec['whole_timeout_s']
        client, control, before, instance = _owned_client_open(self, 'benchmark', deadline=deadline)
        module = _benchmark_module()
        source = _source_sha(module.__file__)
        args = SimpleNamespace(base=spec['base'], model=spec['model'], output=self.output / 'matrix-results.json',
            timeout=spec['socket_timeout_s'], response_mib=spec['response_bytes'] // (1 << 20),
            sse_line_kib=spec['sse_line_bytes'] // 1024)
        code = module.run_matrix(args, plan, read_control=control)
        control.check()
        matrix = _teacher_module()._owned_teacher_object(_teacher_module()._owned_teacher_read(
            args.output, self.prepared['resources']['matrix_report_bytes']))
        if (type(code) is not int or code != 0 or matrix.get('passed') is not True
                or matrix.get('unrun_cohorts') != [] or matrix.get('instance_id') != instance or matrix.get('plan') != plan):
            raise ValueError('closed benchmark matrix did not complete its actual planned work')
        final = client.call('final-stats', path='/stats', object_response=True, read_control=control)
        if _owned_client_service(final, spec, instance, 'benchmark') != before:
            raise ValueError('closed benchmark final idle service identity changed')
        _benchmark_trace_files(self.output, plan, self.prepared['trace'])
        if _source_sha(module.__file__) != source or _source_sha(__file__) != self.producer_sha:
            raise ValueError('closed benchmark source changed during client work')
        self.result = {'schema': 'tensorfold.owned-benchmark-client-report-v1', 'complete': True,
            'client_protocol_passed': True, 'benchmark_plan_sha256': self.prepared['plan_sha256'],
            'benchmark_client_sha256': self.selected['sha256'], 'run_identity': plan['run_identity'],
            'server_instance_id': instance, 'matrix_passed': True, 'observed_trace': True,
            'request_count': sum(c['level'] for c in plan['cohorts']), 'matrix_tool_sha256': source,
            'quality_acceptance': False, 'P3_valid': False}
        write_json(self.output / 'benchmark-client-report.json', self.result)


def _benchmark_model_result(output, prepared):
    root, native, summaries, startup = Path(output).parent, [], [], []
    common = common_semantics = common_producer = None
    for rank in range(8):
        start_path = Path(output) / f'rank-{rank}.json'
        summary_path = Path(output) / f'benchmark-rank{rank}/summary.json'
        start = _teacher_module()._owned_teacher_object(_teacher_module()._owned_teacher_read(start_path, 8 << 20))
        record = _teacher_module()._owned_teacher_object(_teacher_module()._owned_teacher_read(summary_path, 8 << 20))
        paths = list(prepared['native']['directory'].glob(f'benchmark-closed-rank-{rank}-pid-*.json'))
        if len(paths) != 1:
            raise ValueError('benchmark requires one actual closed native record per rank')
        observed = _teacher_module()._owned_teacher_object(_teacher_module()._owned_teacher_read(paths[0], 4 << 20))
        if (record.get('schema') != 'tensorfold.owned-benchmark-rank-result-v1'
                or record.get('rank') != rank or record.get('world') != 8 or record.get('closed') is not True
                or record.get('finished') is not True or record.get('failed') is not False
                or record.get('benchmark_plan_sha256') != prepared['plan_sha256']
                or record.get('benchmark_client_sha256') != prepared['client']['sha256']
                or record.get('actual_owned_profile') != start.get('benchmark_profile')
                or record.get('resources') != prepared['resources'] or start.get('benchmark_resources') != prepared['resources']
                or record.get('headroom') != start.get('benchmark_headroom')
                or observed.get('PID') != start.get('benchmark_worker_PID') or observed.get('rank') != rank
                or observed.get('operation_limits') != _model_contract().MODEL_OBSERVER_LIMITS
                or observed.get('missing_required_prefixes') != []
                or observed.get('observer', {}).get('sha256') != prepared['native']['source']['sha256']):
            raise ValueError('closed benchmark rank completion/source/origin binding differs')
        wanted = [{'id': c['id'], 'level': c['level'], 'completed': c['level'],
                   'prompt_hashes': sorted(r['prompt_sha256'] for r in c['requests'])} for c in prepared['plan']['cohorts']]
        if record.get('cohorts') != wanted:
            raise ValueError('benchmark rank did not consume exactly its planned requests')
        semantics = record['actual_owned_profile'].get('benchmark_semantics')
        producer = {key: start.get(key) for key in ('tool_sha256', 'runtime_cli_sha256', 'dense_implementation',
            'half_partial_policy', 'private_router_policy', 'empty_prefix_policy', 'grouped_gate_up')}
        producer['reference_tiles_requested'] = start.get('reference_tile_preparation', {}).get('requested')
        if common_semantics is None:
            common_semantics, common_producer = semantics, producer
        elif semantics != common_semantics or producer != common_producer:
            raise ValueError('benchmark immutable State/producer semantics differ across ranks')
        sources = record.get('runtime_sources')
        if (type(sources) is not dict or set(sources) != {'engine', 'multi', 'decode', 'sampling', 'observer_tool'}
                or any(not _model_contract().sha(value) for value in sources.values())):
            raise ValueError('benchmark closed rank lacks complete source identity')
        if common is None:
            common = sources
        elif sources != common:
            raise ValueError('benchmark runtime sources disagree across ranks')
        native.append(_teacher_file(paths[0], root, rank=rank))
        summaries.append(_teacher_file(summary_path, root, rank=rank))
        startup.append(_teacher_file(start_path, root, rank=rank))
    paths = _benchmark_trace_files(output, prepared['plan'], prepared['trace'])
    result = {'schema': _model_contract().BENCHMARK_RESULT_SCHEMA, 'benchmark_plan_sha256': prepared['plan_sha256'],
        'benchmark_client_sha256': prepared['client']['sha256'], 'launch_exit': 0,
        'client_report': _teacher_file(Path(output) / 'benchmark-client-report.json', root),
        'matrix_report': _teacher_file(Path(output) / 'matrix-results.json', root),
        'startup_receipts': startup, 'rank_receipts': summaries, 'child_native_records': native,
        'trace': {key: _teacher_file(path, root) for key, path in paths.items()}, 'P3_valid': False}
    _model_contract().validate_benchmark_model_result(result)
    write_json(Path(output) / 'benchmark-model-result.json', result)
    return result


def _teacher_preflight(args, teacher, cli, *, prepared_operation=None, selected=None):
    if (args.draft_method != 'none' or args.context != teacher['data']['context']
            or str(args.parallel).lower() == 'auto' or cli._parallel(args.parallel) < 9):
        raise ValueError('teacher requires explicit none/context and at least nine fixed slots')
    if prepared_operation is not None:
        expected = prepared_operation['selector']
        actual = {'gu': selected.onecat_gu_method or 'incumbent', 'down': selected.onecat_down_method,
                  'iq2_source': selected.onecat_iq2_source, 'prefill_method': selected.onecat_prefill_method}
        grouped = selected.grouped_gate_up_rows
        if (actual != expected or grouped != (4 if expected['gu'] == 'incumbent' else None)
                or not args.no_graphs or cli._parallel(args.parallel) != 24
                or os.environ.get('TENSORFOLD_PREFILL_ROWS') != '256'
                or str(args.execution_profile_manifest) != prepared_operation['execution_manifest']['path']
                or selected.execution_rows is not None or selected.reference_tiles
                or selected.dense_implementation is not None
                or selected.onecat_device_budget_bytes != (None if expected['gu'] == 'incumbent' else 117976064)
                or selected.onecat_global_host_bytes != (None if expected['gu'] == 'incumbent' else 67108864)):
            raise ValueError('prepared teacher requires its exact expert selector and P256/E24/NONE CLI')
    if not args.no_graphs:
        # This admits intent only. The typed, funded HC catalogue is mandatory
        # after construction and before any teacher hook can be installed.
        from tensorfold.cuda.sm70.execution_policy import PROFILE_ENV
        if (getattr(args, 'indexer_history', 'gpu') != 'gpu'
                or getattr(args, 'vram_experts', None) is not None
                or os.environ.get(PROFILE_ENV, 'reference') != 'reference'):
            raise ValueError('teacher graph intent requires the private owned reference GPU-history region')


def prepare_owned_teacher(engine, prepared, output, *, hc_graphs=False):
    """Install only after original immutable loading/serving admission paid it."""
    expected = (prepared['resources']['global_host_obligation_bytes'], prepared['input_sha256'])
    if engine._owned_qualification_host != expected:
        raise ValueError('teacher staging obligation differs from original admitted baseline')
    observer = object.__new__(_teacher_module().OwnedTeacherFrames)
    engine._owned_teacher_frames = observer  # Journal before setup/host allocation, as in the v1 path.
    operation = prepared.get('prepared_expert_operation')
    if operation is not None:
        specification = getattr(getattr(engine, '_owner_preparation', None), 'specification', None)
        if (specification != operation['prepared_execution'] or hc_graphs
                or engine.w.mtp is not None or engine.multi.mbuf is not None
                or engine.multi.buf.rows != 24 or engine.multi.prefill_rows != 256
                or not engine.multi.full_resident or len(engine.multi.states) != 24):
            raise ValueError('prepared teacher actual full preparation/NONE geometry differs')
    profile = bound_receipt(engine)
    if operation is not None:
        from tensorfold.families.qwen4_exp.cuda.execution_identity import digest
        local = engine._execution_bind_local
        binding = engine.capacity_plan.get('execution_profile_binding')
        contents = {}
        for authority in engine._execution_input_authorities:
            authority._check_issued()
            for item in authority.files:
                if not item.logical_name.startswith(('source/', 'library/')):
                    continue
                value = {'size': item.size, 'sha256': item.sha256}
                if item.logical_name in contents and contents[item.logical_name] != value:
                    raise ValueError('prepared teacher issued input authorities disagree')
                contents[item.logical_name] = value
        if (type(binding) is not dict or binding.get('content_passes') != 1
                or type(binding.get('hashes_reused_this_binding')) is not int or binding['hashes_reused_this_binding'] != 0
                or binding.get('manifest_sha256') != local['manifest']
                or any(binding.get('identity', {}).get(key) != digest({label: {
                    name: value for name, value in contents.items() if name.startswith(prefix)}})
                    for key, label, prefix in (('source', 'files', 'source/'), ('kernel_build', 'libraries', 'library/')))):
            raise ValueError('prepared teacher registration differs from its issued source/native contents')
        profile['teacher_execution_registration'] = _teacher_module()._owned_teacher_object(json.dumps({
            'binding': binding, 'numerics': local['numerics'], 'contents': contents}, allow_nan=False))
    observer.__init__(engine, prepared, Path(output) / f'teacher-rank{engine.rank}',
                      fail=_observer_failure, profile=profile, hc_graphs=hc_graphs)
    observer.setup()
    receipt = observer.install()
    write_json(observer.directory / 'setup.json', receipt)
    return receipt


def bound_receipt(engine):
    import torch
    from tensorfold.families.qwen4_exp.cuda.owned_experts import Plan, POLICY
    from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFExperts
    from tensorfold.families.qwen4_exp.cuda.position_coordinates import _digest

    w, plan, decoder = engine.w, engine._owned_expert_plan, engine.multi
    if (type(plan) is not Plan or w.meta.get("owned_expert_plan") is not plan
            or decoder is None or decoder.w is not w or engine.tp != 8
            or plan.partition.rank != engine.rank or plan.partition != engine.expert_partition
            or plan.hidden != 2560 or plan.slots != 11 or plan.partition.experts != 512
            or plan.partition.full_width != 640 or w.cfg.layers != 48 or len(w.layers) != 48):
        raise ValueError("qualification requires the actual bound SC117 TP8/ETP4 plan")
    if w.device.type != "cuda" or torch.cuda.get_device_capability(w.device) != (7, 0):
        raise ValueError("qualification requires actual SM70 CUDA weights")
    banks = [layer.moe.experts for layer in w.layers]
    if w.mtp is not None:
        banks.append(w.mtp.layer.moe.experts)
    if (bool(w.mtp is not None) is not plan.mtp or any(
            type(bank) is not GGUFExperts or bank.implementation != POLICY
            or bank._owned_expert_plan is not plan or bank.partition != plan.partition
            or bank.dims != plan.hidden for bank in banks)):
        raise ValueError("actual target/MTP banks differ from the exclusive native policy")
    resources = plan.reconcile(decoder)  # Post-warm boundary verification, not a second admission/allocation.
    if (resources != decoder.owned_expert_resources
            or resources != engine.capacity_plan.get("owned_expert_native")
            or engine.capacity_plan["expert_parallel"]["implementation"] != POLICY):
        raise ValueError("actual native arenas differ from their admitted/materialized receipt")
    numerical = ("owned-expert-native", plan.identity)
    coordinate = w.meta.get("coordinate_profile_identity")
    if not _digest(coordinate):
        raise ValueError("qualification requires the actual bound coordinate/cohort identity")
    coordinate_marker = ("coordinate-profile-v1", coordinate)
    if not decoder.states or any(numerical not in state.kv_identity or
            coordinate_marker not in state.kv_identity or state.coordinate_profile_identity != coordinate
            for state in decoder.states):
        raise ValueError("actual State numerical identity differs from the native plan")
    return {"schema": "tensorfold.private-owned-qualification-v1", "rank": engine.rank, "world": engine.tp,
            "policy": POLICY, "plan_identity": list(plan.identity), "expert_partition": plan.partition.identity(),
            "target_banks": len(w.layers), "mtp_banks": int(plan.mtp), "resources": resources,
            "coordinate_profile_identity": coordinate, "state_count": len(decoder.states),
            "device": str(w.device), "capability": [7, 0],
            "qualification": "private candidate binding only; no full-native, quality or performance claim"}


def _configured_selector(module, requested, field, selection, label):
    """Read one process-fixed mode; only an absent field can mean legacy."""
    missing = object()
    configured = getattr(module, field, missing)
    supported = configured is not missing
    expected = "inherit" if requested is None else requested
    if (not supported and requested is not None or supported and (
            configured not in ("inherit", selection) or configured != expected)):
        raise ValueError(f"private {label} policy request is unsupported or differs from the installed configuration")
    return {"requested": requested, "configured": configured if supported else "legacy-inherited",
            "private_selector_supported": supported}


def _startup_selector(module, requested, field, native_field, selection, label, inherited):
    result = _configured_selector(module, requested, field, selection, label)
    native = getattr(module, native_field, None)
    if type(native) is not bool:
        raise ValueError(f"private {label} inherited {inherited} status must be an actual bool")
    result["inherited_native_" + inherited] = native
    return result


def _half_policy(requested):
    """Attest process-fixed selection; never turn an environment edit into a policy update."""
    from tensorfold.cuda.sm70 import family_ops

    return _startup_selector(family_ops, requested, "_PRIVATE_HALF_PARTIAL", "_NATIVE_FAMILY",
                             "native-half-v1", "Half", "family")


def _router_policy(requested):
    from tensorfold.cuda.sm70 import routing_native

    return _startup_selector(routing_native, requested, "_PRIVATE_SERIAL_LOGITS", "_NATIVE_LOGITS",
                             "native-serial-v1", "router", "logits")



def _empty_prefix_policy(requested):
    from tensorfold.families.qwen4_exp.cuda import history_owner

    return _configured_selector(history_owner, requested, "_PRIVATE_EMPTY_PREFIX", "cpu-empty-v1", "empty-prefix")



def _execution_rows_receipt(engine, requested):
    decoder = engine.multi
    values = {"engine": engine.execution_rows, "target": decoder.buf.rows,
              "maximum": decoder.controller.maximum["row_budget"],
              "configured": decoder.controller.configured["row_budget"]}
    if decoder.mbuf is not None:
        values["mtp"] = decoder.mbuf.rows
    if any(type(value) is not int or value != requested for value in values.values()):
        raise ValueError("private execution rows differ from prepared engine/buffers/controller")
    return {"requested": requested, **values}



def _reference_tile_receipt(engine, requested):
    actual = getattr(engine, "_reference_owner_tiles_requested", False)
    if type(actual) is not bool or actual is not requested:
        raise ValueError("reference tile request differs from actual Engine preparation")
    if not requested:
        return {"requested": False, "configured": False}
    from tensorfold.cuda.sm70.reference_owner_tiles import ReferenceTiles, preparation_identity
    preparation = engine._owner_preparation
    identity = preparation_identity()
    if (preparation is None or preparation.reference_preparation != identity
            or engine.w.meta.get("reference_tile_preparation") != identity):
        raise ValueError("reference tile pre-budget/source identity differs from actual weights")
    roles = {"decode": engine.multi.buf, "prefill": engine.multi.pbuf}
    if engine.w.mtp is not None:
        roles["mtp"] = engine.multi.mbuf
    if set(preparation.workspaces) != set(roles):
        raise ValueError("reference tile preparation roles differ from actual decoder")
    records = {}
    owners = set()
    for role, buffers in roles.items():
        owner = buffers.history_owner
        if (preparation.workspaces[role] is not owner or id(owner) in owners
                or type(owner.reference_tiles) is not ReferenceTiles or owner.rows != min(buffers.rows, 256)):
            raise ValueError("reference tile role did not adopt its exact journalled receive owner")
        owners.add(id(owner))
        records[role] = owner.reference_tiles.verify(owner.partials, identity)
    return {"requested": True, "configured": True, "prebudget_preparation": identity, "roles": records}

def _onecat_qualification_module():
    if __package__:
        from . import v100_onecat_qualification as module
    else:
        import v100_onecat_qualification as module
    return module


def _onecat_capability_from_args(args):
    """Cold private-trial selector; no public serving flag or automatic credit."""
    if getattr(args, "graph_use_receipts", False) and any(getattr(args, name, False) for name in (
            "prepared_feature_receipts", "normalization_parity", "gdn_batch_parity", "q8_preparation_parity")):
        raise ValueError("graph-use receipts exclude parity/prepared-feature close observers")
    method = args.onecat_gu_method
    down_method = getattr(args, "onecat_down_method", "tensorfold")
    iq2_source = getattr(args, "onecat_iq2_source", "active-frame")
    prefill_method = getattr(args, "onecat_prefill_method", "tensorfold")
    device, host = args.onecat_device_budget_bytes, args.onecat_global_host_bytes
    if method is None:
        if (device is not None or host is not None or down_method != "tensorfold"
                or iq2_source != "active-frame" or prefill_method != "tensorfold"):
            raise ValueError("1Cat credits require an explicit private method")
        return None
    if args.grouped_gate_up_rows is not None:
        raise ValueError("1Cat carries its fixed Grouped4 reference; selectors are exclusive")
    if args.prepared_feature_receipts:
        raise ValueError("1Cat has a distinct cold receipt; prepared-feature intersection is unqualified")
    return _onecat_qualification_module().create_capability(
        method, device_budget_bytes=device, global_host_bytes=host,
        **({"down_method": down_method} if down_method != "tensorfold" else {}),
        **({"iq2_source": iq2_source} if iq2_source != "active-frame" else {}),
        **({"prefill_method": prefill_method} if prefill_method != "tensorfold" else {}))


def _grouped_gate_up_capability(cli, rows):
    if rows is None:
        return {}
    if "grouped_gate_up" not in cli._OwnedExpertQualification.__dataclass_fields__:
        raise ValueError("installed application lacks private grouped gate/up preparation")
    import inspect
    from tensorfold.cuda.sm70.gguf_ops import _GroupedGateUp
    from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine

    if "_owned_expert_gate_up" not in inspect.signature(FlashNextEngine.__init__).parameters:
        raise ValueError("installed application lacks private grouped gate/up preparation")
    return {"grouped_gate_up": _GroupedGateUp(rows)}


def _grouped_gate_up_receipt(engine, requested):
    selection = getattr(engine, "_owned_expert_gate_up_requested", None)
    plan = getattr(engine, "_owned_expert_plan", None)
    actual = getattr(plan, "_gate_up", None)
    if requested is None:
        if selection is not None or actual is not None:
            raise ValueError("unrequested grouped gate/up is active in actual Engine")
    else:
        from tensorfold.cuda.sm70.gguf_ops import _GroupedGateUp
        from tensorfold.families.qwen4_exp.cuda.owned_experts import Plan

        if (type(requested) is not int or requested not in (2, 4)
                or type(selection) is not _GroupedGateUp or selection.rows != requested
                or type(plan) is not Plan or actual is not selection):
            raise ValueError("requested grouped gate/up differs from actual Engine/Plan")
    return {"requested_rows": requested, "actual_rows": None if actual is None else actual.rows}


def _owned_q8_preparation_receipt(engine):
    """Cold retained-capability evidence; never an invocation counter or loader."""
    decoder = getattr(engine, 'multi', None)
    buffers = getattr(decoder, 'buf', None)
    owner = getattr(buffers, 'history_owner', None)
    module = getattr(owner, 'owned_q8_preparation', None)
    if module is None:
        return None  # Baseline profiles retain their existing receipt shape.
    from tensorfold.families.qwen4_exp.cuda.history_owner import OwnerBuffers
    preparation = getattr(engine, '_owner_preparation', None)
    states = decoder.states
    if (type(owner) is not OwnerBuffers or owner.closed or owner.prefill or buffers.prefill
            or owner.native_pair is None
            or (owner.native_pair.key.bits, owner.native_pair.key.codec,
                owner.native_pair.value.bits, owner.native_pair.value.codec) != (8, 0, 8, 0)
            or preparation is None
            or not preparation.prepared or preparation.closed
            or preparation.workspaces.get('decode') is not owner
            or buffers.owner_kernel_family != 'sm70-paged-owner-rn-ex2-p2051-v1'
            or buffers.owner_graph_family != 'uncaptured'
            or owner.native_states is None or not states
            or len(owner.native_states) != len(states)
            or any(owner.native_states.get(id(state), (None,))[0] is not state for state in states)):
        raise ValueError('owned Q8 capability differs from its actual prepared decode owner')
    record = owner.native_record()
    policy = record.get('owned_q8_preparation')
    name, path = getattr(module, '__name__', None), getattr(module, '__file__', None)
    if (not record.get('prepared') or record.get('bound_states') != len(states)
            or record.get('kernel_family') != buffers.owner_kernel_family
            or policy != {'policy': 'owned-q8-decode-preparation-v1', 'module': name,
                          'rows_per_call': 1, 'qualified': False}
            or type(name) is not str or not name.startswith('tf_sm70_family_')
            or type(path) is not str or not Path(path).is_absolute()
            or not callable(getattr(module, 'owned_q8_prepare', None))):
        raise ValueError('owned Q8 native policy/module metadata differs from its retained capability')
    return {'schema': 'tensorfold.owned-q8-preparation-selection-v1',
            'policy': policy, 'state_count': len(states), 'row_capacity': owner.rows,
            'kernel_family': buffers.owner_kernel_family, 'graph_family': buffers.owner_graph_family,
            'native_module': {'name': name, 'path': path},
            'scope': 'cold prepared/selection evidence only; target decode one issued Q8 row under no-grad; not kernel invocation evidence',
            'invocation_evidence': False, 'P3_valid': False}


def _qualification_failure_record(engine, primary, requested_rows):
    """Cold validation-only scalars, never prompts, paths, environment or inputs.

    A malformed field is unavailable diagnostic data, not an inferred zero.
    Avoid arbitrary exception __str__ implementations and unbounded containers.
    """
    def integer(value):
        return value if type(value) is int and -sys.maxsize - 1 <= value <= sys.maxsize else None

    def scalars(value, keys):
        value = value if type(value) is dict else {}
        return {key: integer(value.get(key)) for key in keys}

    plan = getattr(engine, "capacity_plan", {})
    plan = plan if type(plan) is dict else {}
    session = plan.get("session_capacity", {})
    session = session if type(session) is dict else {}
    selected = scalars(session, ("context_tokens", "configured_ceiling", "resident_capacity", "full_context_commitments"))
    resource = session.get("limiting_resource")
    selected["limiting_resource"] = resource if resource in ("administrator-ceiling", "host-memory", "rank-device-memory") else None
    binding = session.get("binding_ranks")
    selected["binding_ranks"] = ([integer(n) for n in binding[:8]] if type(binding) is list else None)
    rank_fields = ("rank", "budget_bytes", "resident_weight_bytes", "loading_overlap_bytes",
                   "session_execution_bytes", "next_session_execution_bytes")
    ledger_fields = ("session_state", "retained_recovery", "execution_arenas", "slot_metadata", "history",
                     "replacement_overlap", "serving_workspace", "graphs", "attention_scratch", "sm70_native_workspace",
                     "kv_cohort_workspace", "expert_routing_extra", "migration_overlap_limit", "device_library_workspace",
                     "retained_idle_recovery", "retained_restore_overlap", "media_mtp_boundary", "draft_history_execution",
                     "draft_replacement_overlap", "dflash_window_state", "dflash_recovery_snapshots",
                     "dflash_execution_workspace", "dflash_target_probability", "retained_device_dependencies")
    baseline_fields = ("physical_free_bytes", "device_total_bytes", "effective_device_floor_bytes", "preallocated_project_bytes",
                       "usable_serving_budget_bytes", "allocator_allocated_bytes", "allocator_reserved_bytes")
    ranks = session.get("ranks")
    selected["ranks"] = []
    if type(ranks) is list:
        for row in ranks[:8]:
            if type(row) is dict:
                selected["ranks"].append({**scalars(row, rank_fields),
                    "resource_ledger_bytes": scalars(row.get("resource_ledger_bytes"), ledger_fields),
                    "allocation_baseline": scalars(row.get("allocation_baseline"), baseline_fields)})
    envelope = plan.get("execution_envelope", {})
    envelope = envelope if type(envelope) is dict else {}
    execution = scalars(envelope, ("target_rows", "max_proposals", "prefill_tokens"))
    method = envelope.get("method")
    execution["method"] = method if method in ("none", "mtp", "dspark", "dflash2") else None
    windows = plan.get("kv_full_windows", {})
    windows = windows if type(windows) is dict else {}
    kv = scalars(windows, ("requested", "reserved", "stream_slots", "context_window", "cache_slots_each",
                          "workspace_bytes", "allocated_cache_bytes"))
    kv["materialized"] = windows.get("materialized") if type(windows.get("materialized")) is bool else None
    kv["ranks"] = []
    materialized = windows.get("ranks")
    if type(materialized) is list:
        for row in materialized[:8]:
            if type(row) is dict:
                sizes = row.get("startup_capacities")
                kv["ranks"].append({**scalars(row, ("rank", "allocated_cache_bytes", "torch_allocated_bytes", "torch_reserved_bytes")),
                    "materialized": row.get("materialized") if type(row.get("materialized")) is bool else None,
                    "startup_capacities": [integer(n) for n in sizes[:64]] if type(sizes) is list else None,
                    "capacities_truncated": type(sizes) is list and len(sizes) > 64})
    decoder = getattr(engine, "multi", None)
    controller = getattr(decoder, "controller", None)
    effective = {"engine": integer(getattr(engine, "execution_rows", None)),
                 "target": integer(getattr(getattr(decoder, "buf", None), "rows", None)),
                 "mtp": integer(getattr(getattr(decoder, "mbuf", None), "rows", None))}
    for key in ("maximum", "configured"):
        value = getattr(controller, key, None)
        effective["controller_" + key] = integer(value.get("row_budget")) if type(value) is dict else None
    args = BaseException.args.__get__(primary)
    message = args[0][:2048] if type(args) is tuple and len(args) == 1 and type(args[0]) is str else "compound/non-string exception arguments; retained in propagated failure"
    record = {"schema": "tensorfold.qualification-startup-failure-v1", "rank": integer(getattr(engine, "rank", None)),
              "world": integer(getattr(engine, "tp", None)), "failed": True, "qualification": False,
              "error": {"type": type(primary).__name__[:128], "message": message},
              "requested_execution_rows": integer(requested_rows), "effective_execution_rows": effective,
              "capacity_plan": {"session_capacity": selected, "execution_envelope": execution, "kv_full_windows": kv},
              "scope": "private cold startup validation failure only; no profile/model/quality/performance pass"}
    if len(json.dumps(record, allow_nan=False).encode()) + 1 > 65536:
        raise ValueError("qualification failure metadata exceeds65536bytes")
    return record


def _publish_qualification_failure(engine, output, primary, requested_rows):
    """Publish before fatal supervisor notification can retire the printer."""
    errors = [None, None, None]
    record = None
    try:
        record = _qualification_failure_record(engine, primary, requested_rows)
        target = Path(output) / f"qualification-failure-rank-{engine.rank}.json"
        if target.exists() or target.is_symlink():
            raise ValueError("qualification failure rank receipt already exists")
        write_json(target, record, indent=None)
    except BaseException as error:
        errors[0] = error
    try:
        # This validation surface has no HTTP requests or prompt/env payloads.
        # The record is an explicit whitelist, never engine/argument repr().
        error = dict(record["error"]) if record is not None else {"type": type(primary).__name__[:128],
            "message": "qualification diagnostic unavailable; retained primary failure"}
        error["diagnostic_publication_error_type"] = None if errors[0] is None else type(errors[0]).__name__[:128]
        print("[tensorfold] private qualification failure " + json.dumps(error, allow_nan=False), file=sys.stderr, flush=True)
    except BaseException as error:
        errors[1] = error
    try:
        if not notify_failure():
            raise RuntimeError("qualification lost its TP supervisor capability")
    except BaseException as error:
        errors[2] = error
    raise_failures(primary, [error for error in errors if error is not None])


def publish_receipt(engine, output, *, dense_implementation=None, half_request=None, router_request=None, empty_request=None,
                    execution_rows=None, reference_tiles=False, normalization_parity=None, gdn_batch_parity=None, q8_preparation_parity=None, grouped_gate_up=None, onecat_gu=None, prepared_features=None,
                    teacher_profile=None):
    try:
        from tensorfold import cli
        receipt = bound_receipt(engine)
        q8_preparation = _owned_q8_preparation_receipt(engine)
        if q8_preparation is not None:
            receipt['owned_q8_preparation'] = q8_preparation
        if teacher_profile is not None:
            receipt.update(_owned_model_startup_fields(engine, teacher_profile,
                getattr(engine, '_teacher_native_observer', None), 'teacher'))
        benchmark = getattr(engine, '_owned_benchmark', None)
        if benchmark is not None:
            receipt.update(_owned_model_startup_fields(engine, benchmark.actual_profile,
                getattr(engine, '_benchmark_native_observer', None), 'benchmark'))
            receipt['benchmark_resources'] = benchmark.prepared['resources']
            receipt['benchmark_headroom'] = benchmark.headroom
        if onecat_gu is None:
            receipt["grouped_gate_up"] = _grouped_gate_up_receipt(engine, grouped_gate_up)
        else:
            receipt["onecat_experts"] = _onecat_qualification_module().bound_receipt(engine, onecat_gu)
        if prepared_features is not None:
            receipt["prepared_features"] = prepared_features
        receipt["reference_tile_preparation"] = _reference_tile_receipt(engine, reference_tiles)
        if normalization_parity is not None:
            receipt["normalization_parity"] = normalization_parity
        if gdn_batch_parity is not None:
            receipt['gdn_batch_parity'] = gdn_batch_parity
        if q8_preparation_parity is not None:
            receipt['q8_preparation_parity'] = q8_preparation_parity
        receipt["half_partial_policy"] = _half_policy(half_request)
        receipt["private_router_policy"] = _router_policy(router_request)
        receipt["empty_prefix_policy"] = _empty_prefix_policy(empty_request)
        if execution_rows is not None:
            receipt["execution_rows"] = _execution_rows_receipt(engine, execution_rows)
        from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat
        actual = GGUFDense.implementation
        if dense_implementation is not None and (
                actual != dense_implementation or GGUFConcat.implementation != "reference"
                or engine.w.meta.get("sm70_profile") != "reference"):
            raise ValueError("private dense trial differs from its selected reference-profile binding")
        receipt["dense_implementation"] = {"requested": dense_implementation, "actual": actual,
                                           "concat": GGUFConcat.implementation}
        obligation = getattr(engine, "_owned_qualification_host", ())
        if obligation:
            receipt["history_host_obligation"] = {"global_bytes": obligation[0], "input_sha256": obligation[1]}
            if hasattr(engine, "_history_run_identity"):
                receipt["run_identity"] = engine._history_run_identity
                receipt["server_instance_id"] = engine._history_server_instance
        receipt["tool_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        receipt["runtime_cli_sha256"] = hashlib.sha256(Path(cli.__file__).read_bytes()).hexdigest()
        target = Path(output) / f"rank-{engine.rank}.json"
        if target.exists() or target.is_symlink():
            raise ValueError("qualification rank receipt already exists")
        write_json(target, receipt)  # Existing atomic file+directory durability owner.
        print(f"[tensorfold] private owned qualification rank={engine.rank}/8 bound; receipt={target.name}", flush=True)
        engine.comm.ready("owned_expert_qualification")  # All eight receipts precede follower/listener publication.
    except BaseException as primary:
        engine._abort_comm = True
        _publish_qualification_failure(engine, output, primary, execution_rows)


def _graph_use_sources(engine, graphs):
    roles = {"engine": engine}
    if graphs is not None:
        roles.update(graph_controller=graphs, graph_policy=graphs.bound)
    sources = {name: _source_sha(Path(sys.modules[type(owner).__module__].__file__))
               for name, owner in roles.items()}
    sources["observer_tool"] = _source_sha(Path(__file__))
    return sources


def _attach_graph_use_observer(engine, output, capability, grouped_gate_up):
    """Snapshot at graph retirement, after Engine drains accepted model work.

    No serving callback is installed. The original graph close still owns its
    fence and retirement; only positive Engine close makes this a closed report.
    """
    graph_module = _onecat_qualification_module()
    graph_module._graph_receipt(engine)  # Refuse an unprepared graph request before listening.
    graphs = getattr(engine.multi, "graphs", None)
    sources = _graph_use_sources(engine, graphs)
    startup = Path(output) / f"rank-{engine.rank}.json"
    startup_sha = _source_sha(startup)
    target = Path(output) / f"graph-use-rank-{engine.rank}.json"
    if target.exists() or target.is_symlink():
        raise ValueError("graph-use rank receipt already exists")
    selector = ({"gu": "incumbent", "grouped_gate_up_rows": grouped_gate_up,
                 "down": "tensorfold", "iq2_source": "not-applicable", "prefill_method": "tensorfold"} if capability is None else
                {"gu": capability.method, "down": capability.down_method,
                 "iq2_source": capability.iq2_source, "prefill_method": capability.prefill_method,
                 "identity": capability.identity})
    original_close, original_graph_close = engine.close, None if graphs is None else graphs.close
    had_close, old_close = "close" in vars(engine), vars(engine).get("close")
    graph_had_close = graphs is not None and "close" in vars(graphs)
    old_graph_close = None if graphs is None else vars(graphs).get("close")
    attempted, snapshot, snapshot_error = False, None, None

    def observe():
        record = graph_module._graph_receipt(engine)
        if graphs is None:
            return {**record, "serving_hits": None, "serving_misses": None}
        raw = graphs.record()
        for name in ("serving_hits", "serving_misses"):
            counts = raw.get(name)
            if (type(counts) is not dict or set(counts) != {"verify", "mtp"}
                    or any(type(n) is not int or not 0 <= n <= sys.maxsize for n in counts.values())):
                raise ValueError("graph-use serving counters lack their exact per-role schema")
            record[name] = dict(counts)
        return record

    def graph_closing(*args, **kwargs):
        nonlocal snapshot, snapshot_error
        try:
            snapshot = observe()
        except BaseException as error:
            snapshot_error = error
        primary, result = snapshot_error, None
        try:
            result = original_graph_close(*args, **kwargs)
        except BaseException as error:
            if primary is None:
                primary = error
            else:
                raise_failures(primary, [error])
        if primary is not None:
            raise primary
        return result

    def closing(*args, **kwargs):
        nonlocal attempted, snapshot
        if attempted:
            raise RuntimeError("graph-use close observer already attempted; no uncertain retirement retry")
        attempted = True
        primary, result, errors = None, None, []
        try:
            if graphs is None:
                snapshot = observe()
            result = original_close(*args, **kwargs)
            if engine._closed is not True or engine._abort_comm is not False:
                raise RuntimeError("graph-use observer lacks positive non-aborted Engine retirement")
            if snapshot is None:
                raise RuntimeError("graph-use retirement boundary was not observed")
            if _graph_use_sources(engine, graphs) != sources or _source_sha(startup) != startup_sha:
                raise ValueError("graph-use source/startup identity changed during serving")
        except BaseException as error:
            primary = error
        try:
            report = {"schema": "tensorfold.graph-use-rank-v1", "rank": engine.rank, "world": engine.tp,
                "closed": primary is None, "failed": primary is not None, "mtp_depth": engine.depth,
                "mtp_active": engine.w.mtp is not None, "selector": selector,
                "startup_receipt_sha256": startup_sha, "runtime_sources": sources, "graph_use": snapshot,
                "primary_error_type": None if primary is None else type(primary).__name__[:512],
                "snapshot_error_type": None if snapshot_error is None else type(snapshot_error).__name__[:512],
                "scope": "per-role serving counters exclude startup; closed only after positive Engine retirement; no quality/P3 acceptance"}
            if len(json.dumps(report, allow_nan=False).encode()) > 65536:
                raise ValueError("graph-use receipt exceeds its bounded metadata envelope")
            write_json(target, report)
        except BaseException as error:
            errors.append(error)
        if primary is None and not errors:
            if had_close:
                engine.close = old_close
            else:
                del engine.close
            if graphs is not None:
                if graph_had_close:
                    graphs.close = old_graph_close
                else:
                    del graphs.close
        raise_failures(primary, errors)
        return result

    if graphs is not None:
        graphs.close = graph_closing
    engine.close = closing


def worker(argv):
    # Validate the borrowed supervisor capability before importing CLI/Torch.
    import fcntl

    value = os.environ.get(FAILURE_FD_ENV, "")
    if not value.isdecimal():
        raise ValueError("private worker requires the supervised failure pipe")
    descriptor = int(value)
    flags = fcntl.fcntl(descriptor, fcntl.F_GETFL)
    if (not stat.S_ISFIFO(os.fstat(descriptor).st_mode)
            or flags & os.O_ACCMODE != os.O_WRONLY or not flags & os.O_NONBLOCK):
        raise ValueError("private worker requires the borrowed nonblocking supervisor writer")
    parser = argparse.ArgumentParser(description="Internal supervised worker")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--router-row-input", type=Path)
    parser.add_argument("--history-input", type=Path)
    parser.add_argument("--history-run-id")
    parser.add_argument("--benchmark-plan", type=Path)
    parser.add_argument("--benchmark-client", type=Path)
    parser.add_argument("--benchmark-trace-data")
    parser.add_argument("--teacher-plan", type=Path)
    parser.add_argument('--teacher-client-bytes', type=int)
    parser.add_argument('--prepared-teacher-operation', type=Path)
    parser.add_argument('--prepared-teacher-operation-sha')
    parser.add_argument("--scheduler-trace", type=Path)
    parser.add_argument("--scheduler-trace-prefill-start", type=int, help="private committed C1 prefill cursor threshold; default0 with trace")
    parser.add_argument("--dense-implementation", choices=("reference", "native32"))
    parser.add_argument("--reference-tiles", action="store_true", help="private pre-budget reference attention preparation")
    parser.add_argument("--grouped-gate-up-rows", type=int, choices=(2, 4),
                        help="private prepared grouped gate/up trial; requires grouped down")
    parser.add_argument("--onecat-gu-method", help="private typed 1Cat GU trial; explicit credits required")
    parser.add_argument("--onecat-down-method", choices=("tensorfold", "onecat-q8-slots"), default="tensorfold",
                        help="target down projection; Q8_1 candidate applies only to IQ4_NL, MTP remains TensorFold FP32")
    parser.add_argument("--onecat-iq2-source", choices=("active-frame", "raw"), default="active-frame",
                        help="IQ2_S target GU source; raw bypasses active-frame packing for that format")
    parser.add_argument("--onecat-prefill-method", choices=("tensorfold", "candidate"), default="tensorfold",
                        help="target prefill expert implementation; candidate uses the selected GU/down path in bounded panels")
    parser.add_argument("--graph-use-receipts", action="store_true",
                        help="bounded per-rank serving graph counters at positive Engine retirement")
    parser.add_argument('--commit-trace-events', type=int, help='finite diagnostic commit-batch event credit; never P3')
    parser.add_argument('--commit-trace-nvtx', action='store_true', help='numeric NVTX batch markers; diagnostic only')
    parser.add_argument("--onecat-device-budget-bytes", type=int, help="explicit per-rank additional plane credit")
    parser.add_argument("--onecat-global-host-bytes", type=int, help="explicit global cold packing host credit")
    parser.add_argument("--gdn-batch-parity", action="store_true", help="diagnostic live batch-vs-scalar GDN and ordered segments; P3 invalid")
    parser.add_argument("--q8-preparation-parity", action="store_true", help="diagnostic one warmed C24 round same-input Q8 writer witness; P3 invalid")
    parser.add_argument("--normalization-parity", action="store_true", help="diagnostic same-input raw-bit normalization witness; invalidates performance timings")
    parser.add_argument("--prepared-feature-receipts", action="store_true",
                        help="private bounded post-close Graph/Plan receipts; requires benchmark")
    parser.add_argument("--execution-rows", type=int)
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    selected = parser.parse_args(argv)
    if ((selected.prepared_teacher_operation is None) != (selected.prepared_teacher_operation_sha is None)
            or selected.prepared_teacher_operation is not None
            and (selected.teacher_plan is None or selected.teacher_client_bytes is None)):
        raise ValueError('prepared teacher worker requires its complete closed parent binding')
    onecat_gu = _onecat_capability_from_args(selected)
    commit_spec = _commit_trace_spec(selected)
    if selected.benchmark_client is not None and (selected.benchmark_plan is None or selected.scheduler_trace is None
            or selected.benchmark_trace_data is None or selected.teacher_plan is not None or selected.gdn_batch_parity):
        raise ValueError('closed benchmark client requires benchmark/trace and trusted trace DATA only')
    if selected.benchmark_trace_data is not None and selected.benchmark_client is None:
        raise ValueError('internal trace DATA requires its closed benchmark client')
    trace_data = None
    if selected.benchmark_trace_data is not None:
        if len(selected.benchmark_trace_data) > 4096:
            raise ValueError('internal benchmark trace DATA exceeds bounded DTO')
        trace_data = _model_contract().validate_benchmark_trace(_teacher_module()._owned_teacher_object(selected.benchmark_trace_data))
        if trace_data['prefill_start'] != (selected.scheduler_trace_prefill_start or 0):
            raise ValueError('benchmark trace DATA differs from selected threshold')
    if selected.scheduler_trace_prefill_start is not None and (selected.scheduler_trace is None or selected.benchmark_plan is None
            or not 0 <= selected.scheduler_trace_prefill_start <= sys.maxsize):
        raise ValueError('late scheduler trace requires trace+benchmark and a nonnegative prefill start')
    if selected.teacher_client_bytes is not None and selected.teacher_plan is None:
        raise ValueError('private client credit requires its teacher plan')
    if selected.teacher_plan is not None and any((selected.benchmark_plan, selected.scheduler_trace,
            selected.router_row_input, selected.history_input, selected.normalization_parity, selected.prepared_feature_receipts, selected.gdn_batch_parity, selected.q8_preparation_parity)):
        raise ValueError('teacher capture/replay is a separate controlled diagnostic')
    if selected.gdn_batch_parity and (selected.benchmark_plan is None or any((selected.normalization_parity,
            selected.prepared_feature_receipts, selected.scheduler_trace, selected.router_row_input,
            selected.history_input, selected.teacher_plan, selected.benchmark_client))):
        raise ValueError('GDN parity requires only its finite benchmark plan and excludes other captures')
    if selected.q8_preparation_parity and (selected.benchmark_plan is None or any((
            selected.gdn_batch_parity, selected.normalization_parity, selected.prepared_feature_receipts,
            selected.scheduler_trace, selected.router_row_input, selected.history_input,
            selected.teacher_plan, selected.benchmark_client))):
        raise ValueError('Q8 preparation parity requires only its finite benchmark plan and excludes other captures')
    if selected.normalization_parity and (selected.benchmark_plan is None or not selected.reference_tiles
            or selected.scheduler_trace is not None or selected.router_row_input is not None
            or selected.history_input is not None):
        raise ValueError("normalization parity requires benchmark/reference-tiles and excludes other captures")
    if selected.prepared_feature_receipts and (selected.benchmark_plan is None or selected.normalization_parity
            or selected.scheduler_trace is not None or selected.router_row_input is not None or selected.history_input is not None):
        raise ValueError("prepared feature receipts require benchmark and exclude other captures")
    tail = selected.serve_args[1:] if selected.serve_args[:1] == ["--"] else selected.serve_args
    receipt_failed = False
    prepared = router_prepared = None
    benchmark = benchmark_resources = parity_resources = feature_resources = gdn_resources = q8_resources = None
    model_setup_errors = [None] * 2  # Teacher/benchmark are exclusive; prepaid before either setup hook.
    teacher = model_observer = closed_benchmark = prepared_teacher = None
    commit_trace = commit_engine = None
    commit_finished = False
    half_request = os.environ.get("TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL")
    router_request = os.environ.get("TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS")
    empty_request = os.environ.get("TENSORFOLD_PRIVATE_SM70_EMPTY_PREFIX")

    def before_listener(engine):
        nonlocal receipt_failed, commit_trace, commit_engine
        try:
            teacher_profile = None
            gdn_receipt = q8_receipt = None
            if teacher is not None:
                try:
                    teacher_profile = prepare_owned_teacher(engine, teacher, selected.out,
                        hc_graphs=not args.no_graphs)['actual_profile']
                    if model_observer is not None:
                        _attach_teacher_native_observer(engine, model_observer)
                except BaseException as primary:
                    _owned_model_failure(engine, primary, model_setup_errors)
            if router_prepared is not None:
                try:
                    prepare_router_rows(engine, router_prepared, selected.out)
                except BaseException as primary:
                    _observer_failure(engine, primary)
                    raise
            if prepared is not None:
                try:
                    from tensorfold.server.stats import INSTANCE_ID
                    engine._history_run_identity = selected.history_run_id
                    engine._history_server_instance = INSTANCE_ID
                    prepare_history(engine, prepared, selected.out)
                except BaseException as primary:
                    _observer_failure(engine, primary)
                    raise
            if benchmark is not None:
                try:
                    if selected.scheduler_trace is None:
                        gdn_witness = prepare_gdn_batch_parity(engine, gdn_resources, selected.out) if gdn_resources is not None else None
                        q8_witness = prepare_q8_preparation_parity(engine, q8_resources, selected.out) if q8_resources is not None else None
                        prepare_benchmark(engine, benchmark,
                            **({'commit_trace': commit_spec} if commit_spec is not None else {}),
                            **({'segment_journal': gdn_witness} if gdn_witness is not None else {}),
                            **({'preparation_witness': q8_witness} if q8_witness is not None else {}))
                        q8_receipt = None if q8_witness is None else q8_witness.published
                        gdn_receipt = None if gdn_witness is None else gdn_witness.published
                    else:
                        prepare_benchmark(engine, benchmark, selected.scheduler_trace,
                            trace_prefill_start=selected.scheduler_trace_prefill_start or 0,
                            **({'commit_trace': commit_spec} if commit_spec is not None else {}),
                            **({'closed': closed_benchmark} if closed_benchmark is not None else {}))
                except BaseException as primary:
                    _owned_model_failure(engine, primary, model_setup_errors)
            parity_receipt = None
            if parity_resources is not None:
                try:
                    parity_receipt = prepare_normalization_parity(engine, parity_resources, selected.out)
                except BaseException as primary:
                    _observer_failure(engine, primary)
                    raise
            feature_receipt = None
            if feature_resources is not None:
                try:
                    feature_receipt = prepare_prepared_features(engine, feature_resources, selected.out,
                                                               selected.grouped_gate_up_rows)
                except BaseException as primary:
                    _observer_failure(engine, primary)
                    raise
            if commit_spec is not None:
                try:
                    if __package__:
                        from .v100_commit_trace import CommitTrace
                    else:
                        from v100_commit_trace import CommitTrace
                    marker = None
                    if commit_spec['nvtx']:
                        import torch
                        marker = lambda sid, delta, index: torch.cuda.nvtx.mark(f'commit-batch:{sid}:{delta}:{index}')
                    commit_trace = CommitTrace(commit_spec['maximum_events'], marker=marker).install(engine.multi)
                    commit_engine = engine
                    engine.capacity_plan['commit_trace'] = {**commit_spec, 'P3_valid': False, 'diagnostic_only': True}
                except BaseException as primary:
                    _observer_failure(engine, primary)
                    raise
            publish_receipt(engine, selected.out, dense_implementation=selected.dense_implementation,
                            half_request=half_request, router_request=router_request, empty_request=empty_request,
                            execution_rows=selected.execution_rows, reference_tiles=selected.reference_tiles,
                            **({"grouped_gate_up": selected.grouped_gate_up_rows} if selected.grouped_gate_up_rows is not None else {}),
                            **({"onecat_gu": onecat_gu} if onecat_gu is not None else {}),
                            **({"normalization_parity": parity_receipt} if parity_receipt is not None else {}),
                            **({'gdn_batch_parity': gdn_receipt} if gdn_receipt is not None else {}),
                            **({'q8_preparation_parity': q8_receipt} if q8_receipt is not None else {}),
                            **({'teacher_profile': teacher_profile} if teacher_profile is not None else {}),
                            **({"prepared_features": feature_receipt} if feature_receipt is not None else {}))
            if selected.graph_use_receipts:
                try:
                    _attach_graph_use_observer(engine, selected.out, onecat_gu, selected.grouped_gate_up_rows)
                except BaseException as primary:
                    _observer_failure(engine, primary)
                    raise
        except BaseException:
            receipt_failed = True  # publish_receipt already poisoned/notified before CLI retirement.
            raise
    try:
        _half_policy(half_request)  # Before the ordinary constructor can prepare any model/workspace.
        _router_policy(router_request)
        _empty_prefix_policy(empty_request)
        if selected.router_row_input is not None:
            if selected.history_input is not None or selected.benchmark_plan is not None:
                raise ValueError("router row capture is a separate controlled procedure")
            data, input_sha = _router_input(selected.router_row_input)
            router_prepared = (data, input_sha, _router_resources(selected.out))
        if selected.benchmark_plan is not None:
            if selected.history_input is not None:
                raise ValueError("history witness and phase matrix are separate controlled procedures")
            if __package__:
                from .bench_concurrent import load_matrix
            else:
                from bench_concurrent import load_matrix
            benchmark = load_matrix(selected.benchmark_plan)
            if selected.scheduler_trace is not None:
                _trace_prefill_selection(benchmark, selected.scheduler_trace_prefill_start or 0)  # Before Engine/admission.
            size = selected.benchmark_plan.stat().st_size
            benchmark_resources = _history_resources(sum(c["level"] for c in benchmark["cohorts"]), size)
            if selected.benchmark_client is not None:
                from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
                plan_sha = _source_sha(selected.benchmark_plan, MAX_MESSAGE_BYTES)
                client_input = _benchmark_client_input(selected.benchmark_client, plan_sha)
                model_observer = _owned_model_observer(selected.out, 'benchmark')
                benchmark_resources = _benchmark_resources(benchmark, size, client_input, trace_data, model_observer['resources'])
                closed_benchmark = {'plan': benchmark, 'plan_sha256': plan_sha, 'client': client_input,
                    'trace': trace_data, 'native': model_observer, 'resources': benchmark_resources, 'output': selected.out}
            if selected.gdn_batch_parity:
                gdn_resources = _gdn_batch_resources(benchmark, selected.out)
                benchmark_resources = {**benchmark_resources, "global_host_obligation_bytes":
                    benchmark_resources["global_host_obligation_bytes"] + gdn_resources["global_host_obligation_bytes"],
                    "per_rank_payload_bytes": benchmark_resources["per_rank_payload_bytes"] + gdn_resources["per_rank_payload_bytes"]}
                gdn_resources = {**gdn_resources, "total_global_host_obligation_bytes": benchmark_resources["global_host_obligation_bytes"],
                    "input_sha256": hashlib.sha256(selected.benchmark_plan.read_bytes()).hexdigest()}
            if selected.q8_preparation_parity:
                q8_resources = _q8_parity_resources(benchmark, selected.out)
                benchmark_resources = {**benchmark_resources, 'global_host_obligation_bytes':
                    benchmark_resources['global_host_obligation_bytes'] + q8_resources['global_host_obligation_bytes'],
                    'per_rank_payload_bytes': benchmark_resources['per_rank_payload_bytes'] + q8_resources['per_rank_payload_bytes']}
                q8_resources = {**q8_resources, 'total_global_host_obligation_bytes': benchmark_resources['global_host_obligation_bytes'],
                    'input_sha256': _source_sha(selected.benchmark_plan)}
            if selected.normalization_parity:
                parity_resources = _normalization_resources(selected.out)
                benchmark_resources = {**benchmark_resources, "global_host_obligation_bytes":
                    benchmark_resources["global_host_obligation_bytes"] + parity_resources["global_host_obligation_bytes"],
                    "per_rank_payload_bytes": benchmark_resources["per_rank_payload_bytes"] + parity_resources["per_rank_payload_bytes"]}
                parity_resources = {**parity_resources,
                    "total_global_host_obligation_bytes": benchmark_resources["global_host_obligation_bytes"],
                    "input_sha256": hashlib.sha256(selected.benchmark_plan.read_bytes()).hexdigest()}
            if selected.prepared_feature_receipts:
                feature_resources = _prepared_feature_resources()
                benchmark_resources = {**benchmark_resources, "global_host_obligation_bytes":
                    benchmark_resources["global_host_obligation_bytes"] + feature_resources["global_host_obligation_bytes"],
                    "per_rank_payload_bytes": benchmark_resources["per_rank_payload_bytes"] + feature_resources["per_rank_payload_bytes"]}
                feature_resources = {**feature_resources,
                    "total_global_host_obligation_bytes": benchmark_resources["global_host_obligation_bytes"],
                    "input_sha256": hashlib.sha256(selected.benchmark_plan.read_bytes()).hexdigest()}
        if selected.history_input is not None:
            if (type(selected.history_run_id) is not str or len(selected.history_run_id) != 32
                    or any(char not in "0123456789abcdef" for char in selected.history_run_id)):
                raise ValueError("history worker requires its fresh parent execution identity")
            data, input_sha, input_limit = _history_input(selected.history_input)
            prepared = (data, input_sha, _history_resources(data["C"], input_limit))
        if selected.teacher_plan is not None:
            teacher = _teacher_module().owned_teacher_plan(selected.teacher_plan)
            if selected.prepared_teacher_operation is not None:
                raw = _teacher_module()._owned_teacher_read(selected.prepared_teacher_operation, 8 << 20)
                if (not _model_contract().sha(selected.prepared_teacher_operation_sha)
                        or hashlib.sha256(raw).hexdigest() != selected.prepared_teacher_operation_sha):
                    raise ValueError('prepared teacher worker control SHA differs')
                operation = _teacher_module()._owned_teacher_object(raw)
                if selected.teacher_client_bytes != operation.get('teacher_client', {}).get('bytes'):
                    raise ValueError('prepared teacher worker client credit differs from its actual DATA')
                prepared_teacher = _prepared_teacher_input(operation, teacher, selected.teacher_plan)
                _prepared_teacher_credit(teacher, prepared_teacher)
                teacher['prepared_expert_operation'] = operation
            if selected.teacher_client_bytes is not None:
                model_observer = _teacher_model_observer(selected.out)
                _teacher_operation_credit(teacher, selected.teacher_client_bytes, model_observer)
        from tensorfold import cli
        args = cli.build_parser().parse_args(["serve", *tail])
        if selected.scheduler_trace is not None and _trace_c24_only(benchmark):
            _trace_c24_preflight(args, cli)
        if selected.q8_preparation_parity:
            _trace_c24_preflight(args, cli)
        if selected.gdn_batch_parity:
            _gdn_batch_preflight(args, cli)
        if args.tp != 8 or args.expert_tensor_parallel_size != 4 or not 0 <= args.rank < 8:
            raise ValueError("private qualification requires explicit TP8/ETP4 rank ownership")
        if teacher is not None:
            if prepared_teacher is None:
                _teacher_preflight(args, teacher, cli)
            else:
                _teacher_preflight(args, teacher, cli,
                    prepared_operation=prepared_teacher['operation'], selected=selected)
        if selected.execution_rows is not None:
            # Keep this experiment explicit: resolve no model-dependent draft
            # defaults and pay its complete envelope before engine construction.
            if (args.draft_method != "mtp" or args.no_drafts or type(args.mtp_drafts) is not int
                    or args.mtp_drafts < 0 or str(args.parallel).lower() == "auto"):
                raise ValueError("private execution rows require explicit MTP depth and parallel count")
            slots = cli._parallel(args.parallel)
            each = args.mtp_drafts + 1
            if not each <= selected.execution_rows <= min(65535, slots * each):
                raise ValueError("private execution rows must fit the explicit complete-window envelope")
        if selected.dense_implementation is not None:
            from tensorfold.cuda.sm70.execution_policy import PROFILE_ENV
            from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
            if os.environ.get(PROFILE_ENV, "reference") != "reference":
                raise ValueError("private dense trial requires the reference SM70 profile")
            # Each supervised rank selects once, before construction, weight preparation and warmup.
            GGUFDense.implementation = selected.dense_implementation
        obligation = () if prepared is None else (prepared[2]["global_host_obligation_bytes"], prepared[1])
        if router_prepared is not None:
            obligation = (router_prepared[2]["global_host_obligation_bytes"], router_prepared[1])
        if benchmark_resources is not None:
            obligation = (benchmark_resources["global_host_obligation_bytes"],
                          hashlib.sha256(selected.benchmark_plan.read_bytes()).hexdigest())
        if teacher is not None:
            obligation = (teacher['resources']['global_host_obligation_bytes'], teacher['input_sha256'])
        if commit_spec is not None:
            trace_sha = hashlib.sha256(json.dumps(
                {'commit_trace': commit_spec, 'prior': obligation}, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            obligation = ((obligation[0] if obligation else 0) + commit_spec['global_host_obligation_bytes'], trace_sha)
        capability = ({"grouped_gate_up": onecat_gu} if onecat_gu is not None else
                      _grouped_gate_up_capability(cli, selected.grouped_gate_up_rows))
        if selected.execution_rows is not None:
            if "execution_rows" not in cli._OwnedExpertQualification.__dataclass_fields__:
                raise ValueError("installed CLI lacks private execution-row preparation")
            capability["execution_rows"] = selected.execution_rows
        if selected.reference_tiles:
            if "reference_tiles" not in cli._OwnedExpertQualification.__dataclass_fields__:
                raise ValueError("installed CLI lacks pre-budget reference tile preparation")
            capability["reference_tiles"] = True
        args._owned_expert_qualification = cli._OwnedExpertQualification(before_listener, obligation, **capability)
        result = cli.cmd_serve(args)
        if commit_trace is not None:
            commit_finished = True
            _finish_commit_trace(commit_trace, commit_engine, selected.out, commit_spec)
        return result
    except BaseException as primary:
        diagnostic = None
        if commit_trace is not None and not commit_finished:
            commit_finished = True
            try:
                _finish_commit_trace(commit_trace, commit_engine, selected.out, commit_spec)
            except BaseException as error:
                diagnostic = error
        try:
            if not receipt_failed and not notify_failure():
                raise RuntimeError("qualification lost its TP supervisor capability")
        except BaseException as cleanup:
            raise_failures(primary, [diagnostic, cleanup])
        if diagnostic is not None:
            raise_failures(primary, [diagnostic])
        raise



def accept_history(output, client_report):
    """Offline intersection; successful completed replies imply TP finish ACKs."""
    if __package__:
        from . import v100_adaptive_exercise as exercise
    else:
        import v100_adaptive_exercise as exercise
    from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
    def read(path, limit):
        with Path(path).open("rb") as source:
            raw = source.read(limit + 1)
        if len(raw) > limit:
            raise ValueError("qualification artifact exceeds its bounded input envelope")
        return json.loads(raw)
    client = read(client_report, MAX_MESSAGE_BYTES)
    result = client["supplied_long"]
    capacity = result["resident_capacity"]
    if type(capacity) is not int or capacity <= 0:
        raise ValueError("client has no positive resident capacity")
    initial, final = client["initial_stats"], client["final_stats"]
    instance = initial["server"]["instance_id"]
    if (not client.get("complete") or client.get("error") or instance != final["server"]["instance_id"]
            or not result.get("semantic_parity_passed") or not result.get("sse_nonstream_parity_passed")
            or result.get("drafted_total", 0) <= 0):
        raise ValueError("history acceptance requires completed same-service semantic/parity/MTP evidence")
    exercise.qualification.require_idle(final)
    concurrent = result["concurrent"]["results"]
    if len(result["cases"]) != capacity or len(concurrent) != capacity:
        raise ValueError("history acceptance requires exactly C client histories")
    names = set(result["cases"])
    if {row["request"] for row in concurrent} != names:
        raise ValueError("concurrent replies do not identify each supplied history")
    hashes, drafted_total = set(), 0
    for name, pair in result["cases"].items():
        hashes.add(pair["prompt_token_ids_sha256"])
        expected = result["oracles"][name]["expected"]
        replies = [pair["serial"], pair["mtp"], next(row["reply"] for row in concurrent if row["request"] == name)]
        for index, reply in enumerate(replies):
            stats = reply["tensorfold"]
            if (stats.get("drafts") is not (index != 0) or any(type(stats.get(key, 0)) is not int
                    or stats.get(key, 0) < 0 for key in ("drafted", "accepted"))):
                raise ValueError("client actual draft dispatch/counters differ from requested modes")
            if index == 0 and (stats.get("drafted", 0) or stats.get("accepted", 0)):
                raise ValueError("client serial oracle performed draft work")
            if index:
                drafted_total += stats.get("drafted", 0)
            record = exercise.correctness_reply(reply, result["declared_reply_tokens"], required_finish=None)
            exercise.correctness_answer({"id": name, **expected}, record)
            if reply["usage"]["prompt_tokens"] != result["prompt_tokens"]:
                raise ValueError("client prompt geometry differs from its oracle")
            if not exercise.live.compare(replies[0], reply, result["declared_reply_tokens"])["passed"]:
                raise ValueError("client parity evidence differs")
        if (pair["serial"]["tensorfold"].get("cached") != 0
                or pair["serial"]["usage"].get("prompt_tokens_details", {}).get("cached_tokens") != 0):
            raise ValueError("client serial oracle was not cold")
    if drafted_total <= 0 or drafted_total != result["drafted_total"]:
        raise ValueError("client MTP summary does not match positive actual proposal work")
    common, run = None, None
    for rank in range(8):
        receipt = read(Path(output) / f"history-rank-{rank}.json", 4096 + 1024 * capacity)
        startup = read(Path(output) / f"rank-{rank}.json", MAX_MESSAGE_BYTES)
        if (startup.get("schema") != "tensorfold.private-owned-qualification-v1"
                or type(startup.get("rank")) is not int or startup["rank"] != rank or startup.get("world") != 8
                or receipt["server_instance_id"] != startup["server_instance_id"]
                or receipt["schema"] != "tensorfold.private-owned-history-v1" or type(receipt["rank"]) is not int or receipt["rank"] != rank
                or receipt["world"] != 8 or receipt["C"] != capacity
                or receipt["N"] != result["prompt_tokens"] or receipt["R"] != result["declared_reply_tokens"]
                or receipt["L"] != result["context_tokens"] or receipt["N"] + receipt["R"] != receipt["L"]
                or receipt["run_identity"] != startup["run_identity"]
                or receipt["input_sha256"] != startup["history_host_obligation"]["input_sha256"]
                or receipt["coordinate_profile_identity"] != startup["coordinate_profile_identity"]
                or receipt["plan_identity"] != startup["plan_identity"]
                or rank == 0 and receipt["server_instance_id"] != instance):
            raise ValueError("worker receipt is not bound to this prepared execution/client")
        if run is None:
            run = receipt["run_identity"]
        if receipt["run_identity"] != run:
            raise ValueError("worker receipts belong to different executions")
        owners = receipt["owners"]
        if (len(owners) != capacity or len({row["wire"] for row in owners}) != capacity
                or len({row["slot"] for row in owners}) != capacity
                or {row["prompt_token_ids_sha256"] for row in owners} != hashes
                or len(hashes) != capacity or any(
                    any(type(row[key]) is not int or row[key] < 0 for key in
                        ("wire", "slot", "slot_generation", "ownership_generation", "cached_length", "prompt_tokens"))
                    or row["prompt_tokens"] != receipt["N"] or row["cached_length"] < receipt["N"] for row in owners)):
            raise ValueError("worker owners do not match the C actual client histories")
        facts = {key: receipt[key] for key in ("prefix_sequence", "input_sha256", "C", "N", "R", "L",
                 "coordinate_profile_identity", "plan_identity", "owners")}
        if common is not None and facts != common:
            raise ValueError("worker receipts do not share one logical prefix boundary")
        common = facts
    return {"schema": "tensorfold.private-owned-history-acceptance-v1", "passed": True,
            "run_identity": run, "server_instance_id": instance, "facts": common,
            "client_report_sha256": hashlib.sha256(Path(client_report).read_bytes()).hexdigest(),
            "scope": "coherent logical owner/device-stream ordering plus completed client parity; not GPU wall-clock simultaneity"}


def _run_owned_client(client, command, prefix, output, *, role, plan_sha, client_sha, mode):
    if __package__:
        from . import launch_cuda_tp
    else:
        import launch_cuda_tp
    if __package__:
        from .v100_rotorquant_live import ReadCancelled
        from .worker_lifetime import Task, drain
    else:
        from v100_rotorquant_live import ReadCancelled
        from worker_lifetime import Task, drain
    task = Task(client.run, name='owned-' + role + '-one-client')
    tasks, cleanup_status = [task], [None] * 4
    primary = None
    result = None
    worker_failure = False
    try:
        task.start()
        def failed():
            if task.done.is_set() and task.error is not None:
                raise task.error
            return False
        def stopping(requested):
            if requested:
                client.cancelled.set()
            if task.done.is_set() and task.error is not None:
                raise task.error
            return task.done.is_set() and task.error is None and client.result is not None and client.result['complete'] is True
        result = launch_cuda_tp.main(command, _command_prefix=prefix, _shutdown_probe=stopping, _failure_probe=failed)
        if result != 0 and task.error is None:
            primary = RuntimeError(role + ' cohort failed with launcher status ' + str(result))
            worker_failure = True
    except BaseException as error:
        primary = error
    if task.error is not None and task.error is not primary and not worker_failure:
        cleanup_status[0] = primary
        primary = task.error
    try:
        client.cancelled.set()
    except BaseException as error:
        cleanup_status[1] = error
        if primary is None:
            primary = error
    try:
        drain(tasks, primary=primary)  # Task.done, not Thread.is_alive, owns the captured state.
    except BaseException as error:
        cleanup_status[2] = error
        if primary is None:
            primary = error
    if task.error is not None and task.error is not primary and not isinstance(task.error, ReadCancelled):
        if worker_failure:
            cleanup_status[0] = task.error
        else:
            cleanup_status[0] = primary
            primary = task.error
    if primary is not None:
        try:
            failed_report = {'schema': 'tensorfold.owned-' + role + '-client-report-v1', 'complete': False,
                'post_attempts': client.post_attempts if role == 'teacher' else None, 'launch_exit': result, role + '_plan_sha256': plan_sha,
                role + '_client_sha256': client_sha, 'mode': mode,
                'client_protocol_passed': False, 'quality_acceptance': False, 'P3_valid': False,
                'error_type': type.__dict__['__name__'].__get__(type(primary))}
            write_json(output / (role + '-client-failure.json'), failed_report)
        except BaseException as report_error:
            cleanup_status[3] = report_error
        # Fixed status slots remain owned through any neutral-transport failure.
        index = 0
        while index < 4:
            if cleanup_status[index] is None:
                cleanup_status[index] = primary
            index += 1
        try:
            raise_failures(primary, cleanup_status)
        except BaseException as failure:
            if failure is primary:
                raise
            raise primary from failure
    if result != 0 or client.result is None or client.result.get('complete') is not True:
        raise RuntimeError(role + ' model lacks positive cohort and client completion')
    return result


def main(argv=None, *, _benchmark_operation=None, _prepared_teacher_operation=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--accept-history"]:
        parser = argparse.ArgumentParser(description="Offline private history evidence intersection")
        parser.add_argument("--out", type=Path, required=True)
        parser.add_argument("--client-report", type=Path, required=True)
        selected = parser.parse_args(argv[1:])
        target = selected.out / "history-acceptance.json"
        if target.exists() or target.is_symlink():
            raise ValueError("history acceptance output already exists")
        write_json(target, accept_history(selected.out, selected.client_report))
        return 0
    if argv[:1] == ["--worker"]:
        return worker(argv[1:])
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", required=True)
    parser.add_argument("--numa-plan", type=Path, help="sealed per-rank GPU-local CPU and memory placement plan")
    parser.add_argument("--numa-plan-sha256", help="exact NUMA placement plan SHA256")
    parser.add_argument("--master-port", type=int, default=29551)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark-plan", type=Path)
    parser.add_argument("--benchmark-client", type=Path)
    parser.add_argument("--teacher-plan", type=Path, help="private none/greedy raw full-vocabulary capture or frozen-prefix replay DATA; never P3")
    parser.add_argument('--teacher-client', type=Path, help='closed one-request client DATA; normal shutdown only after validated completion')
    parser.add_argument("--scheduler-trace", type=Path)
    parser.add_argument("--scheduler-trace-prefill-start", type=int, help="private committed C1 prefill cursor threshold; default0 with trace")
    parser.add_argument("--dense-implementation", choices=("reference", "native32"))
    parser.add_argument("--reference-tiles", action="store_true", help="private pre-budget reference attention preparation")
    parser.add_argument("--grouped-gate-up-rows", type=int, choices=(2, 4),
                        help="private prepared grouped gate/up trial; requires grouped down")
    parser.add_argument("--onecat-gu-method", help="private typed 1Cat GU trial; explicit credits required")
    parser.add_argument("--onecat-down-method", choices=("tensorfold", "onecat-q8-slots"), default="tensorfold",
                        help="target down projection; Q8_1 candidate applies only to IQ4_NL, MTP remains TensorFold FP32")
    parser.add_argument("--onecat-iq2-source", choices=("active-frame", "raw"), default="active-frame",
                        help="IQ2_S target GU source; raw bypasses active-frame packing for that format")
    parser.add_argument("--onecat-prefill-method", choices=("tensorfold", "candidate"), default="tensorfold",
                        help="target prefill expert implementation; candidate uses the selected GU/down path in bounded panels")
    parser.add_argument("--graph-use-receipts", action="store_true",
                        help="bounded per-rank serving graph counters at positive Engine retirement")
    parser.add_argument('--commit-trace-events', type=int, help='finite diagnostic commit-batch event credit; never P3')
    parser.add_argument('--commit-trace-nvtx', action='store_true', help='numeric NVTX batch markers; diagnostic only')
    parser.add_argument("--onecat-device-budget-bytes", type=int, help="explicit per-rank additional plane credit")
    parser.add_argument("--onecat-global-host-bytes", type=int, help="explicit global cold packing host credit")
    parser.add_argument("--gdn-batch-parity", action="store_true", help="diagnostic live batch-vs-scalar GDN and ordered segments; P3 invalid")
    parser.add_argument("--q8-preparation-parity", action="store_true", help="diagnostic one warmed C24 round same-input Q8 writer witness; P3 invalid")
    parser.add_argument("--normalization-parity", action="store_true", help="diagnostic same-input raw-bit normalization witness; invalidates performance timings")
    parser.add_argument("--prepared-feature-receipts", action="store_true",
                        help="private bounded post-close Graph/Plan receipts; requires benchmark")
    parser.add_argument("--execution-rows", type=int, help="private prepared MTP row envelope; requires explicit depth/parallel")
    parser.add_argument("--router-row-input", type=Path, help="private four-row layer0 capture DATA; diagnostic overhead only")
    parser.add_argument("--history-input", type=Path, help="private C/N/R/L and expected prompt-ID SHA DATA")
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if (args.numa_plan is None) != (args.numa_plan_sha256 is None):
        parser.error('NUMA plan and SHA256 must be supplied together')
    _onecat_capability_from_args(args)  # Validate before spawning any rank.
    _commit_trace_spec(args)
    teacher = selected_client = model_observer = prepared_teacher = None
    if _prepared_teacher_operation is not None and (args.teacher_plan is None or args.teacher_client is None
                                                   or _benchmark_operation is not None):
        raise ValueError('trusted prepared teacher operation requires only its closed teacher inputs')
    closed_benchmark = None
    if args.benchmark_client is not None:
        if (args.benchmark_plan is None or args.scheduler_trace is None or args.teacher_plan is not None
                or _benchmark_operation is None or args.normalization_parity or args.prepared_feature_receipts or args.gdn_batch_parity
                or args.router_row_input is not None or args.history_input is not None):
            parser.error('benchmark client requires the trusted closed benchmark/trace operation')
        from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
        operation = _model_contract().validate_benchmark_operation(_benchmark_operation)
        for name, path in (('benchmark_plan', args.benchmark_plan), ('benchmark_client', args.benchmark_client)):
            item = operation[name]
            if str(path.absolute()) != item['path'] or path.stat().st_size != item['bytes'] or _source_sha(path, MAX_MESSAGE_BYTES if name == 'benchmark_plan' else 8 << 20) != item['sha256']:
                raise ValueError('closed benchmark actual DATA differs from trusted operation')
        if (args.out.absolute() / 'trace' != args.scheduler_trace.absolute()
                or operation['trace']['prefill_start'] != (args.scheduler_trace_prefill_start or 0)):
            raise ValueError('closed benchmark output/trace selector differs from operation')
        plan = _benchmark_module().load_matrix(args.benchmark_plan)
        selected_client = _benchmark_client_input(args.benchmark_client, operation['benchmark_plan']['sha256'])
        _model_contract().validate_benchmark_operation_plan(operation, plan, selected_client['data'])
        model_observer = _owned_model_observer(args.out, 'benchmark')
        resources = _benchmark_resources(plan, operation['benchmark_plan']['bytes'], selected_client,
            operation['trace'], model_observer['resources'])
        closed_benchmark = {'plan': plan, 'plan_sha256': operation['benchmark_plan']['sha256'],
            'client': selected_client, 'native': model_observer, 'trace': operation['trace'], 'resources': resources,
            'output': args.out}
    elif _benchmark_operation is not None:
        raise ValueError('trusted benchmark operation requires its private client DATA')
    if args.scheduler_trace_prefill_start is not None and (args.scheduler_trace is None or args.benchmark_plan is None
            or not 0 <= args.scheduler_trace_prefill_start <= sys.maxsize):
        parser.error('late scheduler trace requires trace+benchmark and a nonnegative prefill start')
    if args.teacher_client is not None and args.teacher_plan is None:
        parser.error('teacher client requires its frozen teacher plan')
    if args.teacher_plan is not None:
        if any((args.benchmark_plan, args.scheduler_trace, args.router_row_input, args.history_input,
                args.normalization_parity, args.prepared_feature_receipts, args.gdn_batch_parity, args.q8_preparation_parity)):
            parser.error('teacher capture/replay excludes other diagnostic procedures')
        teacher = _teacher_module().owned_teacher_plan(args.teacher_plan)  # Pure bounded validation before parent effects.
        if _prepared_teacher_operation is not None:
            prepared_teacher = _prepared_teacher_input(_prepared_teacher_operation, teacher, args.teacher_plan)
            _prepared_teacher_credit(teacher, prepared_teacher)
            teacher['prepared_expert_operation'] = _prepared_teacher_operation
        if args.teacher_client is not None:
            selected_client = (_teacher_client_input(args.teacher_client, teacher) if prepared_teacher is None else
                               _teacher_client_input(args.teacher_client, teacher, prepared=True))
            if prepared_teacher is not None:
                expected_client = _prepared_teacher_operation['teacher_client']
                if (str(args.teacher_client.absolute()) != expected_client['path']
                        or selected_client['bytes'] != expected_client['bytes']
                        or selected_client['sha256'] != expected_client['sha256']):
                    raise ValueError('prepared teacher actual client differs from closed operation')
                selected_client['prepared_manifest'] = {key: prepared_teacher[key] for key in
                    ('manifest_canonical_sha256', 'manifest_artifact_files', 'manifest_artifact_bytes')}
            model_observer = _teacher_model_observer(args.out)
            _teacher_operation_credit(teacher, selected_client['bytes'], model_observer)
    if args.gdn_batch_parity and (args.benchmark_plan is None or any((args.normalization_parity,
            args.prepared_feature_receipts, args.scheduler_trace, args.router_row_input,
            args.history_input, args.teacher_plan, args.benchmark_client))):
        parser.error('GDN parity requires only its finite benchmark plan and excludes other captures')
    if args.q8_preparation_parity and (args.benchmark_plan is None or any((
            args.gdn_batch_parity, args.normalization_parity, args.prepared_feature_receipts,
            args.scheduler_trace, args.router_row_input, args.history_input,
            args.teacher_plan, args.benchmark_client))):
        raise ValueError('Q8 preparation parity requires only its finite benchmark plan and excludes other captures')
    if args.normalization_parity and (args.benchmark_plan is None or not args.reference_tiles
            or args.scheduler_trace is not None or args.router_row_input is not None or args.history_input is not None):
        parser.error("normalization parity requires benchmark/reference-tiles and excludes other captures")
    if args.prepared_feature_receipts and (args.benchmark_plan is None or args.normalization_parity
            or args.scheduler_trace is not None or args.router_row_input is not None or args.history_input is not None):
        parser.error("prepared feature receipts require benchmark and exclude other captures")
    if args.execution_rows is not None and not 1 <= args.execution_rows <= 65535:
        parser.error("private execution rows must be in 1..65535")
    if args.router_row_input is not None and (args.history_input is not None or args.benchmark_plan is not None):
        parser.error("router row capture cannot be combined with history/benchmark")
    if args.scheduler_trace is not None and args.benchmark_plan is None:
        parser.error("scheduler trace requires its private benchmark plan")
    devices = [value.strip() for value in args.gpus.split(",")]
    if len(devices) != 8 or len(set(devices)) != 8 or not all(devices):
        parser.error("private qualification requires eight distinct physical devices")
    if __package__:
        from . import launch_cuda_tp
    else:
        import launch_cuda_tp
    output = args.out.absolute()
    output.mkdir(mode=0o700)  # Fresh, caller-owned output; never reuse another qualification's receipts.
    prefix = [sys.executable, str(Path(__file__).resolve()), "--worker", "--out", str(output)]
    if args.teacher_plan is not None:
        prefix += ['--teacher-plan', str(args.teacher_plan.absolute())]
    if selected_client is not None and teacher is not None:
        prefix += ['--teacher-client-bytes', str(selected_client['bytes'])]
    if prepared_teacher is not None:
        control = output / 'prepared-teacher-operation.json'
        _teacher_module()._owned_teacher_json(control, prepared_teacher['operation'], 8 << 20)
        control.chmod(0o444)
        control_sha = hashlib.sha256(prepared_teacher['control']).hexdigest()
        if control.read_bytes() != prepared_teacher['control']:
            raise ValueError('prepared worker control publication differs from its prepaid bytes')
        prefix += ['--prepared-teacher-operation', str(control), '--prepared-teacher-operation-sha', control_sha]
    if args.router_row_input is not None:
        prefix += ["--router-row-input", str(args.router_row_input.absolute())]
    if args.history_input is not None:
        prefix += ["--history-input", str(args.history_input.absolute()), "--history-run-id", uuid.uuid4().hex]
    if args.dense_implementation is not None:
        prefix += ["--dense-implementation", args.dense_implementation]
    if args.reference_tiles:
        prefix += ["--reference-tiles"]
    if args.grouped_gate_up_rows is not None:
        prefix += ["--grouped-gate-up-rows", str(args.grouped_gate_up_rows)]
    if args.onecat_gu_method is not None:
        prefix += ["--onecat-gu-method", args.onecat_gu_method,
                   "--onecat-down-method", args.onecat_down_method,
                   "--onecat-iq2-source", args.onecat_iq2_source,
                   "--onecat-prefill-method", args.onecat_prefill_method,
                   "--onecat-device-budget-bytes", str(args.onecat_device_budget_bytes),
                   "--onecat-global-host-bytes", str(args.onecat_global_host_bytes)]
    if args.graph_use_receipts:
        prefix += ["--graph-use-receipts"]
    if args.commit_trace_events is not None:
        prefix += ['--commit-trace-events', str(args.commit_trace_events)]
        if args.commit_trace_nvtx:
            prefix += ['--commit-trace-nvtx']
    if args.gdn_batch_parity:
        prefix += ["--gdn-batch-parity"]
    if args.q8_preparation_parity:
        prefix += ["--q8-preparation-parity"]
    if args.normalization_parity:
        prefix += ["--normalization-parity"]
    if args.prepared_feature_receipts:
        prefix += ["--prepared-feature-receipts"]
    if args.execution_rows is not None:
        prefix += ["--execution-rows", str(args.execution_rows)]
    if args.benchmark_plan is not None:
        prefix += ["--benchmark-plan", str(args.benchmark_plan.absolute())]
    if closed_benchmark is not None:
        prefix += ['--benchmark-client', str(args.benchmark_client.absolute()), '--benchmark-trace-data',
                   json.dumps(closed_benchmark['trace'], sort_keys=True, separators=(',', ':'))]
    if args.scheduler_trace is not None:
        trace_output = args.scheduler_trace.absolute()
        trace_output.mkdir(mode=0o700)
        prefix += ["--scheduler-trace", str(trace_output)]
        if args.scheduler_trace_prefill_start is not None:
            prefix += ["--scheduler-trace-prefill-start", str(args.scheduler_trace_prefill_start)]
    prefix += ["--"]
    numa_options = ([] if args.numa_plan is None else
                    ["--numa-plan", str(args.numa_plan.absolute()), "--numa-plan-sha256", args.numa_plan_sha256])
    command = ["--gpus", args.gpus, "--master-port", str(args.master_port), *numa_options, *args.serve_args]
    if selected_client is None:
        result = launch_cuda_tp.main(command, _command_prefix=prefix)
        if teacher is not None and result == 0:
            _teacher_module().seal_owned_teacher(output, teacher, result)
        return result
    if closed_benchmark is not None:
        client = _BenchmarkClient(closed_benchmark, output)
        result = _run_owned_client(client, command, prefix, output, role='benchmark',
            plan_sha=closed_benchmark['plan_sha256'], client_sha=selected_client['sha256'], mode='diagnostic')
        _benchmark_model_result(output, closed_benchmark)
        return result
    client = _TeacherClient(teacher, selected_client, output)
    result = _run_owned_client(client, command, prefix, output, role='teacher',
        plan_sha=teacher['input_sha256'], client_sha=selected_client['sha256'], mode=teacher['data']['mode'])
    rank0 = _teacher_module()._owned_teacher_object(_teacher_module()._owned_teacher_read(output / 'teacher-rank0/summary.json', 8 << 20))
    if rank0.get('teacher_ids') != client.result['token_ids']:
        raise ValueError('teacher HTTP output differs from captured frozen-prefix frames')
    expected_family = (selected_client['data']['prepared_execution'] if prepared_teacher is not None else
                       {key: selected_client['data']['expected_service'][key] for key in ('kernel_family', 'graph_family')})
    if any(frame.get('execution_family') != expected_family for frame in rank0.get('frames', [])):
        raise ValueError('teacher first control did not execute its actual reference family')
    authority = _teacher_module().seal_owned_teacher(output, teacher, result)
    _teacher_model_result(output, teacher, selected_client, model_observer, authority)
    return result


if __name__ == "__main__":
    raise SystemExit(main())

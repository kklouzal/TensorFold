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



class _SchedulerTrace:
    """Five finite rank-zero scheduler calls, never a live cross-call profiler."""
    def __init__(self, engine, output, *, factory=None):
        self.engine, self.output, self.factory = engine, Path(output), factory
        self.counts, self.active = {}, False

    def label(self, observer):
        if observer.current is None:
            return None
        cohort = next(c for c in observer.plan["cohorts"] if c["id"] == observer.current)
        if cohort["warmup"] or cohort["level"] not in (1, 8):
            return None
        live = [s for s in observer.decoder.streams.values() if not s.done and s.st is not None]
        if cohort["level"] == 1 and observer.decoder.filling and not self.counts.get("prefill"):
            self.counts["prefill"] = 1
            return "c1-cold-prefill-containing-round"
        key = str(cohort["level"])
        if (not observer.decoder.filling and len(live) == cohort["level"]
                and all(len(s.out) > 1 for s in live) and self.counts.get(key, 0) < 2):
            self.counts[key] = self.counts.get(key, 0) + 1
            return f"c{key}-warmed-decode-round-{self.counts[key]}"
        return None

    @staticmethod
    def process():
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"io": Path("/proc/self/io").read_text(), "minor_faults": usage.ru_minflt,
                "major_faults": usage.ru_majflt, "maxrss_kib_cumulative": usage.ru_maxrss}

    def capture(self, label, callback):
        if threading.current_thread() is not self.engine.scheduler.thread:
            raise RuntimeError("profiler must be owned by the actual scheduler thread")
        if self.factory is None:
            import torch
            profiler = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA], record_shapes=False, profile_memory=False,
                with_stack=False)
        else:
            profiler = self.factory()
        before, started = self.process(), time.monotonic()
        primary, errors, entered, result = None, [], False, None
        try:
            profiler.__enter__()
            entered = True
            self.active = True
            entered_at = time.monotonic()
            result = callback()
        except BaseException as error:
            primary = error
        finally:
            self.active = False
            work_end = time.monotonic()
            if entered:
                try:
                    profiler.__exit__(type(primary) if primary else None, primary,
                                      primary.__traceback__ if primary else None)
                except BaseException as error:
                    errors.append(error)
                stopped = time.monotonic()
                trace_bytes = None
                try:
                    trace = self.output / (label + ".json")
                    profiler.export_chrome_trace(str(trace))
                    trace_bytes = trace.stat().st_size
                except BaseException as error:
                    errors.append(error)
                try:
                    write_json(self.output / (label + "-receipt.json"), {
                        "rank": self.engine.rank, "thread": threading.current_thread().name,
                        "thread_ident": threading.get_ident(), "label": label,
                        "profiler_enter_seconds": entered_at - started,
                        "work_call_seconds": work_end - entered_at,
                        "profiler_stop_seconds": stopped - work_end,
                        "export_seconds": time.monotonic() - stopped,
                        "process_before": before, "process_after": self.process(),
                        "primary": None if primary is None else repr(primary),
                        "post_errors": [repr(error) for error in errors],
                        "exported_trace_bytes": trace_bytes,
                        "resource_scope": "source bounds capture count to five calls; operator owns profiler memory and filesystem containment",
                        "scope": "rank0 TP adapter round; inclusive prefill/decode/control, diagnostic only; peers untraced"})
                except BaseException as error:
                    errors.append(error)
        raise_failures(primary, errors)
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
                (decoder, "_prepare_pass", "PLE.prepare_frame_stage"),
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
        original = adapter.round
        def rounded():
            label = self.label(observer)
            if label is None:
                return original()
            try:
                return self.capture(label, original)
            except BaseException as primary:
                _observer_failure(self.engine, primary)
                raise
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
    def __init__(self, engine, plan, *, events=None):
        import torch
        from tensorfold.server.stats import INSTANCE_ID
        self.server_instance_id = INSTANCE_ID
        self.engine, self.plan = engine, plan
        self.decoder, self.observations = engine.multi, engine.multi.controller.work_observations
        self.events = events if events is not None else [torch.cuda.Event(enable_timing=True) for _ in range(4)]
        self.current = None
        self.records = {}
        self.identities = {}
        self.cases = {tuple(request["prefix"]): (cohort, request)
                      for cohort in plan["cohorts"] for request in cohort["requests"]}
        self.prefix_lengths = sorted({len(prefix) for prefix in self.cases})
        self.published = {"schema": "tensorfold.private-benchmark-v1", "run_identity": plan["run_identity"],
                          "server_instance_id": self.server_instance_id, "cohorts": {}}
        self.stream = None

    def _invalid(self, message):
        error = ValueError(message)
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
                if self.current is None:
                    return None  # Explicit pre-matrix unobserved smoke phase.
                self._invalid("request does not belong to the private benchmark plan")
            cohort, request = match
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
            self.current = cohort["id"]
            self.records[self.current] = {"level": cohort["level"], "prefill_done": set(), "done": set(),
                "hashes": set(), "active_owner_highwater": 0, "prefill_batch_highwater": 0,
                "decode_batch_highwater": 0, "started": [False] * 4, "events_ready": False}
            self.stream = torch.cuda.current_stream(self.engine.w.device)
        return cohort, request, self.records[self.current]

    def _record(self, index):
        import torch
        if torch.cuda.current_stream(self.engine.w.device) != self.stream:
            self._invalid("benchmark changed the actual critical CUDA stream")
        if self.records[self.current]["started"][index]:
            self._invalid("benchmark phase event was already recorded")
        try:
            self.events[index].record(self.stream)
        except BaseException as error:
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

    def install(self):
        decoder, observations = self.decoder, self.observations
        admit, prepare, joined, take, round_, finish = (getattr(decoder, name) for name in
                                                       ("admit", "_prepare_pass", "_joined", "_take", "round", "finish"))
        target, record = observations.target, observations.record
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
            return prepare(pieces)
        def joining(pieces, *args):
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
            if self.current is not None:
                name = "prefill_batch_highwater" if kwargs.get("prefill", False) else "decode_batch_highwater"
                self.records[self.current][name] = max(self.records[self.current][name],
                    len({id(state) for state, first, last in segments if last > first}))
            return result
        def rounded():
            result = round_()
            self._observe()
            return result
        def finished(streams):
            result = finish(streams)
            for stream in streams:
                self.identities.pop(self.engine.tp_decoder.wires.get(id(stream)), None)
            self._observe()
            return result
        def recorded(*, credit=None):
            # Internal TP status readers never query/drain timers. HTTP reads
            # may query only a completely finished, idle model cohort.
            if (self.current is not None and threading.current_thread() is not self.engine.scheduler.thread
                    and not self.decoder.streams and not self.decoder.filling):
                state = self.records[self.current]
                if (not state["events_ready"] and len(state["done"]) == state["level"]
                        and all(state["started"]) and all(event.query() for event in self.events)):
                    try:
                        state["prefill_ms"] = self.events[0].elapsed_time(self.events[1])
                        state["decode_ms"] = self.events[2].elapsed_time(self.events[3])
                    except BaseException as error:
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
            return {**result, "benchmark": published}
        # The private before-listener capability is the only selector. Default
        # instances/class implementations retain their original methods.
        decoder.admit, decoder._prepare_pass, decoder._joined, decoder._take = admitted, preparing, joining, taking
        decoder.round, decoder.finish = rounded, finished
        observations.target, observations.record = targeted, recorded


def prepare_benchmark(engine, plan, trace_output=None):
    if engine.rank != 0:
        return
    path = Path(engine.model_dir) / "tokenizer.json"
    if hashlib.sha256(path.read_bytes()).hexdigest() != plan["tokenizer_sha256"]:
        raise ValueError("benchmark tokenizer differs from the actual prepared frontend")
    observer = _BenchmarkObserver(engine, plan)
    observer.install()
    if trace_output is not None:
        _trace_inventory(engine, trace_output)
        _SchedulerTrace(engine, trace_output).install(observer)


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


def publish_receipt(engine, output, *, dense_implementation=None, half_request=None, router_request=None, empty_request=None,
                    execution_rows=None, reference_tiles=False, normalization_parity=None, grouped_gate_up=None):
    try:
        from tensorfold import cli
        receipt = bound_receipt(engine)
        receipt["grouped_gate_up"] = _grouped_gate_up_receipt(engine, grouped_gate_up)
        receipt["reference_tile_preparation"] = _reference_tile_receipt(engine, reference_tiles)
        if normalization_parity is not None:
            receipt["normalization_parity"] = normalization_parity
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
        try:
            if not notify_failure():
                raise RuntimeError("qualification lost its TP supervisor capability")
        except BaseException as cleanup:
            raise_failures(primary, [cleanup])
        raise


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
    parser.add_argument("--scheduler-trace", type=Path)
    parser.add_argument("--dense-implementation", choices=("reference", "native32"))
    parser.add_argument("--reference-tiles", action="store_true", help="private pre-budget reference attention preparation")
    parser.add_argument("--grouped-gate-up-rows", type=int, choices=(2, 4),
                        help="private prepared grouped gate/up trial; requires grouped down")
    parser.add_argument("--normalization-parity", action="store_true", help="diagnostic same-input raw-bit normalization witness; invalidates performance timings")
    parser.add_argument("--execution-rows", type=int)
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    selected = parser.parse_args(argv)
    if selected.normalization_parity and (selected.benchmark_plan is None or not selected.reference_tiles
            or selected.scheduler_trace is not None or selected.router_row_input is not None
            or selected.history_input is not None):
        raise ValueError("normalization parity requires benchmark/reference-tiles and excludes other captures")
    tail = selected.serve_args[1:] if selected.serve_args[:1] == ["--"] else selected.serve_args
    receipt_failed = False
    prepared = router_prepared = None
    benchmark = benchmark_resources = parity_resources = None
    half_request = os.environ.get("TENSORFOLD_PRIVATE_SM70_MOE_PARTIAL")
    router_request = os.environ.get("TENSORFOLD_PRIVATE_SM70_ROUTER_LOGITS")
    empty_request = os.environ.get("TENSORFOLD_PRIVATE_SM70_EMPTY_PREFIX")

    def before_listener(engine):
        nonlocal receipt_failed
        try:
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
                        prepare_benchmark(engine, benchmark)
                    else:
                        prepare_benchmark(engine, benchmark, selected.scheduler_trace)
                except BaseException as primary:
                    _observer_failure(engine, primary)
                    raise
            parity_receipt = None
            if parity_resources is not None:
                try:
                    parity_receipt = prepare_normalization_parity(engine, parity_resources, selected.out)
                except BaseException as primary:
                    _observer_failure(engine, primary)
                    raise
            publish_receipt(engine, selected.out, dense_implementation=selected.dense_implementation,
                            half_request=half_request, router_request=router_request, empty_request=empty_request,
                            execution_rows=selected.execution_rows, reference_tiles=selected.reference_tiles,
                            **({"grouped_gate_up": selected.grouped_gate_up_rows} if selected.grouped_gate_up_rows is not None else {}),
                            **({"normalization_parity": parity_receipt} if parity_receipt is not None else {}))
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
            size = selected.benchmark_plan.stat().st_size
            benchmark_resources = _history_resources(sum(c["level"] for c in benchmark["cohorts"]), size)
            if selected.normalization_parity:
                parity_resources = _normalization_resources(selected.out)
                benchmark_resources = {**benchmark_resources, "global_host_obligation_bytes":
                    benchmark_resources["global_host_obligation_bytes"] + parity_resources["global_host_obligation_bytes"],
                    "per_rank_payload_bytes": benchmark_resources["per_rank_payload_bytes"] + parity_resources["per_rank_payload_bytes"]}
                parity_resources = {**parity_resources,
                    "total_global_host_obligation_bytes": benchmark_resources["global_host_obligation_bytes"],
                    "input_sha256": hashlib.sha256(selected.benchmark_plan.read_bytes()).hexdigest()}
        if selected.history_input is not None:
            if (type(selected.history_run_id) is not str or len(selected.history_run_id) != 32
                    or any(char not in "0123456789abcdef" for char in selected.history_run_id)):
                raise ValueError("history worker requires its fresh parent execution identity")
            data, input_sha, input_limit = _history_input(selected.history_input)
            prepared = (data, input_sha, _history_resources(data["C"], input_limit))
        from tensorfold import cli
        args = cli.build_parser().parse_args(["serve", *tail])
        if args.tp != 8 or args.expert_tensor_parallel_size != 4 or not 0 <= args.rank < 8:
            raise ValueError("private qualification requires explicit TP8/ETP4 rank ownership")
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
        capability = _grouped_gate_up_capability(cli, selected.grouped_gate_up_rows)
        if selected.execution_rows is not None:
            if "execution_rows" not in cli._OwnedExpertQualification.__dataclass_fields__:
                raise ValueError("installed CLI lacks private execution-row preparation")
            capability["execution_rows"] = selected.execution_rows
        if selected.reference_tiles:
            if "reference_tiles" not in cli._OwnedExpertQualification.__dataclass_fields__:
                raise ValueError("installed CLI lacks pre-budget reference tile preparation")
            capability["reference_tiles"] = True
        args._owned_expert_qualification = cli._OwnedExpertQualification(before_listener, obligation, **capability)
        return cli.cmd_serve(args)
    except BaseException as primary:
        try:
            if not receipt_failed and not notify_failure():
                raise RuntimeError("qualification lost its TP supervisor capability")
        except BaseException as cleanup:
            raise_failures(primary, [cleanup])
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


def main(argv=None):
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
    parser.add_argument("--master-port", type=int, default=29551)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--benchmark-plan", type=Path)
    parser.add_argument("--scheduler-trace", type=Path)
    parser.add_argument("--dense-implementation", choices=("reference", "native32"))
    parser.add_argument("--reference-tiles", action="store_true", help="private pre-budget reference attention preparation")
    parser.add_argument("--grouped-gate-up-rows", type=int, choices=(2, 4),
                        help="private prepared grouped gate/up trial; requires grouped down")
    parser.add_argument("--normalization-parity", action="store_true", help="diagnostic same-input raw-bit normalization witness; invalidates performance timings")
    parser.add_argument("--execution-rows", type=int, help="private prepared MTP row envelope; requires explicit depth/parallel")
    parser.add_argument("--router-row-input", type=Path, help="private four-row layer0 capture DATA; diagnostic overhead only")
    parser.add_argument("--history-input", type=Path, help="private C/N/R/L and expected prompt-ID SHA DATA")
    parser.add_argument("serve_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.normalization_parity and (args.benchmark_plan is None or not args.reference_tiles
            or args.scheduler_trace is not None or args.router_row_input is not None or args.history_input is not None):
        parser.error("normalization parity requires benchmark/reference-tiles and excludes other captures")
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
    if args.normalization_parity:
        prefix += ["--normalization-parity"]
    if args.execution_rows is not None:
        prefix += ["--execution-rows", str(args.execution_rows)]
    if args.benchmark_plan is not None:
        prefix += ["--benchmark-plan", str(args.benchmark_plan.absolute())]
    if args.scheduler_trace is not None:
        trace_output = args.scheduler_trace.absolute()
        trace_output.mkdir(mode=0o700)
        prefix += ["--scheduler-trace", str(trace_output)]
    prefix += ["--"]
    return launch_cuda_tp.main(["--gpus", args.gpus, "--master-port", str(args.master_port), *args.serve_args],
                               _command_prefix=prefix)


if __name__ == "__main__":
    raise SystemExit(main())

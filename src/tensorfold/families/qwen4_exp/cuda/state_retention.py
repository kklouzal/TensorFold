"""Complete actual State inventory, independent of RAM/catalog placement.

Capture is legal only at a resolved model boundary while the caller excludes
mutation/reuse. The common transfer lease retains these actual views through
DMA completion. Fixed model weights, PLE tables and execution scratch are never
session payload. Existing State.snapshot remains an interior-cut contract.
"""

from dataclasses import dataclass
import hashlib

import torch

from .method_retention import commit_method_restore, method_inventory, method_restore_targets
from .retained_parts import TensorPart, part_specs

SCHEMA = "qwen4exp-complete-state-v1"


def canonical_tokens(stream, position, *, new_turn=False):
    """Owned bounded actual IDs; no placeholder-derived media authority."""
    import itertools
    import numpy as np

    values = stream.prompt if new_turn or len(stream.context) < position else stream.context
    if len(values) < position:
        raise ValueError("retained target frontier lacks its actual canonical token IDs")

    def checked():
        for token in itertools.islice(values, position):
            if type(token) is not int or not 0 <= token < (1 << 31):
                raise ValueError("retained canonical token ID is outside its source representation")
            yield token

    result = np.fromiter(checked(), dtype=np.int32, count=position)
    return torch.from_numpy(result), hashlib.sha256(memoryview(result).cast("B")).hexdigest()


def request_record(stream):
    _ids, fingerprint = canonical_tokens(stream, len(stream.prompt), new_turn=True)
    if (type(stream.count) is not int or stream.count < 0
            or any(type(getattr(stream, field)) is not bool for field in ("draft", "stop_eos", "background"))):
        raise ValueError("retained request limits/modes require their actual native types")
    return {"prompt_tokens": len(stream.prompt), "prompt_sha256": fingerprint, "reply_limit": stream.count,
            "draft": stream.draft, "stop_eos": stream.stop_eos, "background": stream.background}


def new_turn_metadata(state, metadata, stream):
    """Fresh request inherits model prefix, never parent recovery/RNG state."""
    result = dict(metadata, recoveries=[])
    result["mtp_participating"] = bool(state.mtp_hidden_width and stream.draft)
    method = dict(metadata["method"], recoveries=[], draws=None)
    if not stream.draft:
        result.update(mtp_len=0, mtp_drafted=0, mtp_participating=False)
        method.update(mtp_len=0, mtp_drafted=0, boundary=None)
        if method["dflash"] is not None:
            method["dflash"] = dict(method["dflash"], end=0, available=0, enabled=False)
        if state.compact_indexer is not None:
            indexers = dict(result["indexers"])
            ordinal = str(len(state.kc))
            if ordinal in indexers:
                indexers[ordinal] = {"origin": 0, "length": 0, "base": None, "proposal_count": 0}
            result["indexers"] = indexers
    result["method"] = method
    return result


def new_turn_uses_part(state, spec, stream):
    if spec.name.startswith(("recovery.", "method.recovery.")):
        return False
    if stream.draft:
        return True
    if spec.name.startswith("method."):
        return False
    role, *values = spec.semantic
    return not (role in ("kv", "pool", "legacy-index", "indexer") and values[0] == len(state.kc))


def new_turn_gpu_snapshot(source, destination, stream):
    """Existing State restore wire with fresh participation, before any writes.

    Source views stay leased through the coordinator's copy completion event;
    no independent full recurrent or inactive draft-window clone is needed.
    """
    source.kv_check()
    mtp = source.mtp_len - source.mtp_drafted if stream.draft else 0
    current = source.cur[0] if source.cur else 0
    snapshot = {"pos": source.pos, "mtp_len": mtp, "rec": source.rec[current], "conv": source.conv,
        "ple_tail": source.ple_tail, "ple_history": source.ple_history,
        **source.kv_snapshot_metadata(), **source.indexer_snapshot_metadata(source.pos, mtp)}
    if source.media_profile_identity is not None:
        if stream.draft:
            from .media_mtp import snapshot_state

            snapshot.update(snapshot_state(source, mtp_len=mtp))
        else:
            snapshot["media_state"] = {"schema": "qwen4exp-media-state-v1", "profile": source.media_profile_identity,
                "media": source.media_identity, "generation": source.slot_generation, "rope_delta": source.rope_delta,
                "mtp_enabled": False, "mtp_boundary": None}
    if getattr(source, "dflash2", None) is not None:
        from .dflash2_runtime import inactive_snapshot_payload

        snapshot["dflash2"] = (source.dflash2_runtime.snapshot(source) if stream.draft else
                               inactive_snapshot_payload(source.dflash2, source.pos))
    destination.kv_validate_snapshot(snapshot)  # Normalize/validate before copy_prefix writes any rows.
    return snapshot


@dataclass(frozen=True)
class StateInventory:
    metadata: dict
    parts: tuple[TensorPart, ...]
    host: dict
    future_bytes: int

    @property
    def specs(self):
        return part_specs(self.parts)


def _caches(state):
    result = [(i, cache, state.pos, state.pooled[i]) for i, cache in enumerate(state.kc)]
    if hasattr(state, "mtp_kc"):
        result.append((len(state.kc), state.mtp_kc, state.mtp_len, state.mtp_pooled))
    return result


def _history_parts(state):
    placement = state.history_placement
    page_rows = 128 if placement is None else placement.page_tokens
    parts = []
    for ordinal, cache, length, pooled in _caches(state):
        committed = (state.mtp_len - state.mtp_drafted if ordinal == len(state.kc) else state.pos)
        head = 0 if placement is None else state.history_rank // placement.group_size
        for page, first in enumerate(range(0, length, page_rows)):
            rows = min(page_rows, length - first)
            if placement is None:
                offset = first
            elif cache.owners[page] == state.history_rank:
                offset = cache.offsets[page]
                if offset < 0:
                    raise ValueError("retained history page lacks its actual physical owner offset")
            else:
                continue
            for field, valid in (("k", True), ("v", True), ("ks", cache.key_format.quantized),
                                 ("vs", cache.value_format.quantized)):
                if valid:
                    immutable = rows == page_rows and first + rows <= committed
                    parts.append(TensorPart(f"history.kv.{ordinal}.{head}.{page}.{field}",
                        getattr(cache, field)[offset:offset + rows], "history" if immutable else "state",
                        ("kv", ordinal, head, first, rows, field, immutable)))
        groups = length // state.ratio
        if pooled.shape[0] < groups and pooled.shape[0] != 0:
            raise ValueError("retained pooled history is shorter than its committed cache frontier")
        if pooled.shape[0]:
            for first in range(0, groups, max(1, page_rows // state.ratio)):
                rows = min(page_rows // state.ratio, groups - first)
                immutable = rows == page_rows // state.ratio and first + rows <= committed // state.ratio
                parts.append(TensorPart(f"history.pool.{ordinal}.{first}", pooled[first:first + rows], "history" if immutable else "state",
                                        ("pool", ordinal, first, rows)))
    return parts


def _indexer_parts(state):
    indexer = state.compact_indexer
    parts, metadata = [], {}
    if indexer is None:
        if getattr(state, "indexer_history_owner", None) is not None:
            raise ValueError("compact retained profile cannot serialize a growing raw RAM indexer archive")
        for ordinal, _cache, length, _pooled in _caches(state):
            raw = state.mtp_ikc if ordinal == len(state.kc) else state.ikc[ordinal]
            parts.append(TensorPart(f"history.legacy-index.{ordinal}", raw[:length], "history",
                                    ("legacy-index", ordinal, length)))
        return parts, metadata
    for ordinal, entry in enumerate(indexer.entries):
        if entry is None:
            continue
        if entry.span is not None and entry.span.resolved is None:
            raise ValueError("retained compact indexer still has an unresolved append")
        record = {"origin": entry.origin, "length": entry.n, "base": entry.base,
                  "proposal_count": entry.proposal_count}
        metadata[str(ordinal)] = record
        for name in ("raw", "coordinates", "protected", "protected_coordinates"):
            parts.append(TensorPart(f"state.indexer.{ordinal}.{name}", getattr(entry, name), "state",
                                    ("indexer", ordinal, name)))
        if entry.base is not None:
            for name in ("base_raw", "base_coordinates"):
                parts.append(TensorPart(f"state.indexer.{ordinal}.{name}", getattr(entry, name), "recovery",
                                        ("indexer", ordinal, name)))
            for name in ("proposal_raw", "proposal_coordinates"):
                parts.append(TensorPart(f"state.indexer.{ordinal}.{name}", getattr(entry, name)[:entry.proposal_count],
                                        "recovery", ("indexer", ordinal, name)))
    return parts, metadata


def _recovery_parts(state, recoveries):
    parts, records = [], []
    for number, item in enumerate(recoveries):
        snapshot, tail = item if type(item) is tuple and len(item) == 2 else (item, None)
        if type(snapshot) is not dict:
            raise ValueError("retained recovery must be an actual snapshot dictionary")
        state.kv_validate_snapshot(snapshot)
        if snapshot["pos"] > state.pos or snapshot["mtp_len"] > state.mtp_len:
            raise ValueError("retained recovery lies beyond the actual retained source frontiers")
        record = {"pos": snapshot["pos"], "mtp_len": snapshot["mtp_len"], "ple_history": None,
                  "kv_identity": snapshot["kv_identity"], "compact_indexer": None}
        history = snapshot.get("ple_history")
        expected_history = state.ple_history
        if ((history is None) != (expected_history is None) or history is not None and
                (type(history) is not type(expected_history) or history.shape != expected_history.shape
                 or history.dtype != expected_history.dtype)):
            raise ValueError("retained PLE CPU recovery differs from actual State history storage")
        if history is not None:
            record["ple_history"] = history.tolist()
        for name in ("rec", "conv", "ple_tail"):
            value = snapshot[name]
            expected = state.rec[0] if name == "rec" else getattr(state, name)
            if (not isinstance(value, torch.Tensor) or value.shape != expected.shape
                    or value.dtype != expected.dtype or value.device != expected.device):
                raise ValueError("retained recurrent/conv/PLE recovery differs from actual State storage")
            parts.append(TensorPart(f"recovery.{number}.{name}", value, "recovery",
                                    (number, name, snapshot["pos"], snapshot["mtp_len"])))
        if tail is not None and state.mtp_hidden_width == 0:
            raise ValueError("target-only recovery supplied an unclassified HC dependency")
        compact = snapshot.get("compact_indexer")
        if state.compact_indexer is not None:
            state.compact_indexer.validate_snapshot(compact, snapshot["pos"], snapshot["mtp_len"])
        if compact is not None:
            record["compact_indexer"] = {"schema": compact["schema"], "lengths": compact["lengths"], "tails": []}
            for ordinal, value in enumerate(compact["tails"]):
                record["compact_indexer"]["tails"].append(None if value is None else {
                    name: value[name] for name in ("origin", "length", "pooled_valid")})
                if value is not None:
                    for field in ("raw", "coordinates"):
                        parts.append(TensorPart(f"recovery.{number}.indexer.{ordinal}.{field}", value[field],
                                                "recovery", (number, ordinal, field)))
        records.append(record)
    return parts, records


def state_inventory(state, *, stream=None, pending_hc=None, recoveries=(), recovery_versions):
    """Actual byte-preserving source views; no whole-history clone is created."""
    if type(recovery_versions) is not int or recovery_versions < 0 or len(recoveries) > recovery_versions:
        raise ValueError("retained recovery inventory exceeds the admitted version bound")
    state.retention_method_recovery_versions = recovery_versions
    state.kv_check()
    if state.target_tentative_length != state.pos or getattr(state, "media_input_span", None) is not None:
        raise ValueError("retained State capture requires its resolved target/input boundary")
    # A partially completed physical migration is a valid mixed page map.
    # Capture reads its actual committed owners/offsets; the actor excludes
    # further migration while the transfer holds these source views.
    for reader in tuple(state.history_native_readers):
        reader.wait_native_reuse()
    current = state.cur[0] if state.cur else 0
    if any(value != current for value in state.cur):
        raise ValueError("retained recurrent layers do not share a complete committed bank")
    has_method = state.mtp_hidden_width > 0 or getattr(state, "dflash2", None) is not None
    method = method_inventory(state, stream=stream, pending_hc=pending_hc,
                              recoveries=tuple(recoveries) if has_method else ())
    parts = [TensorPart("state.rec", state.rec[current], "state", ("rec",)),
             TensorPart("state.conv", state.conv, "state", ("conv",)),
             TensorPart("state.ple_tail", state.ple_tail, "state", ("ple_tail",))]
    canonical_sha = None
    if stream is not None:
        canonical, canonical_sha = canonical_tokens(stream, state.pos)
        if state.history_rank == 0:
            parts.append(TensorPart("state.canonical_ids", canonical, "state", ("canonical-ids",)))
    parts.extend(_history_parts(state))
    index_parts, index_metadata = _indexer_parts(state)
    parts.extend(index_parts)
    recovery_parts, recovery_metadata = _recovery_parts(state, recoveries)
    parts.extend(recovery_parts)
    parts.extend(method.parts)
    specs = part_specs(parts)
    metadata = {"schema": SCHEMA, "kv_identity": state.kv_identity, "limit": state.limit,
        "pos": state.pos, "mtp_len": state.mtp_len, "mtp_drafted": state.mtp_drafted,
        "rope_delta": state.rope_delta, "mtp_participating": state.media_mtp_enabled,
        "source_slot": state.history_slot, "source_rank": state.history_rank,
        "source_slot_generation": state.slot_generation, "source_history_generation": state.history_generation,
        "indexers": index_metadata, "recoveries": recovery_metadata, "method": method.metadata,
        "media_profile": state.media_profile_identity, "media_identity": state.media_identity,
        "canonical_sha256": canonical_sha,
        "request": None if stream is None else request_record(stream),
        "next_prompt_input": None if stream is None or state.pos >= len(stream.prompt) else stream.prompt[state.pos],
        "ple_history": None if state.ple_history is None else state.ple_history.tolist()}
    # Actual full-window allocator geometry is a payload upper bound, not a
    # current-short-history claim or a qualified physical host-memory result.
    caches = state.cache_bytes(state.limit)
    if getattr(state, "dflash2", None) is not None:
        caches -= state.dflash2.nbytes  # Method inventory charges its window/recovery exactly once.
    fixed = sum(part.tensor.numel() * part.tensor.element_size() for part in parts[:3])
    tails = sum(part.tensor.numel() * part.tensor.element_size() for part in index_parts
                if part.category == "state")
    future = caches + fixed * (1 + recovery_versions) + tails * recovery_versions + method.future_bytes + state.limit * 4
    if sum(spec.nbytes for spec in specs) > future:
        raise ValueError("actual retained components exceed their complete future payload promise")
    return StateInventory(metadata, tuple(parts), {"method": method.host}, future)


def rebind_state_inventory(state, previous, *, pending_hc=None, recoveries=(), recovery_versions):
    """Refresh actual physical views of an UNCHANGED committed generation.

    Migration can replace/cache-remap storage without changing canonical input,
    request or keyed-draw semantics. The catalog already owns those exact CPU
    int32 bytes and scalar metadata; no Stream/callback or duplicate Python token
    archive is needed merely to enumerate the new physical history views.
    """
    if type(previous) is not StateInventory:
        raise ValueError("physical rebinding needs the actual prior complete inventory")
    old = previous.metadata
    expected = {"schema": SCHEMA, "kv_identity": state.kv_identity, "limit": state.limit,
        "pos": state.pos, "mtp_len": state.mtp_len, "mtp_drafted": state.mtp_drafted,
        "source_slot": state.history_slot, "source_rank": state.history_rank,
        "source_slot_generation": state.slot_generation, "rope_delta": state.rope_delta,
        "mtp_participating": state.media_mtp_enabled, "media_profile": state.media_profile_identity,
        "media_identity": state.media_identity}
    if any(not _same(old.get(name), value) for name, value in expected.items()):
        raise ValueError("retained semantic generation changed before physical rebinding")
    fresh = state_inventory(state, pending_hc=pending_hc, recoveries=recoveries,
                            recovery_versions=recovery_versions)
    canonical = tuple(part for part in previous.parts if part.name == "state.canonical_ids")
    if bool(canonical) != (state.history_rank == 0 and old["request"] is not None):
        raise ValueError("retained canonical authority is missing or on the wrong rank")
    if canonical and (len(canonical) != 1 or canonical[0].tensor.shape != (state.pos,)
                      or canonical[0].tensor.dtype != torch.int32 or canonical[0].tensor.device.type != "cpu"):
        raise ValueError("owned canonical input no longer fits the retained frontier")
    metadata = dict(fresh.metadata, **{name: old[name] for name in
                    ("canonical_sha256", "request", "next_prompt_input")})
    metadata["method"] = dict(metadata["method"], draws=old["method"]["draws"])
    parts = fresh.parts[:3] + canonical + fresh.parts[3:]
    specs = part_specs(parts)
    if sum(spec.nbytes for spec in specs) > fresh.future_bytes:
        raise ValueError("physically rebound inventory exceeds its complete future claim")
    return StateInventory(metadata, parts, fresh.host, fresh.future_bytes)


def _same(left, right):
    # Store exchange uses JSON lists; tuple/list representation is not identity.
    import json

    return json.dumps(left, sort_keys=True, separators=(",", ":")) == json.dumps(right, sort_keys=True, separators=(",", ":"))


def _owner(state, ordinal):
    placement = state.history_placement
    return state.history_rank if placement is None else placement.indexer_owner(
        state.history_slot, ordinal, include_mtp=hasattr(state, "mtp_kc"))


def _cache(state, ordinal):
    return state.mtp_kc if ordinal == len(state.kc) else state.kc[ordinal]


def _destination(state, spec, source_rank):
    """Map a logical component to the CURRENT slot's physical owner/view."""
    role, *values = spec.semantic
    local = source_rank == state.history_rank
    if role == "kv":
        ordinal, head, first, rows, field, _immutable = values
        cache, placement = _cache(state, ordinal), state.history_placement
        if placement is None:
            return getattr(cache, field)[first:first + rows] if local else None
        page = first // placement.page_tokens
        if state.history_rank // placement.group_size != head or cache.owners[page] != state.history_rank:
            return None
        offset = cache.offsets[page]
        return getattr(cache, field)[offset:offset + rows]
    if role == "pool":
        ordinal, first, rows = values
        if _owner(state, ordinal) != state.history_rank or state.history_placement is None and not local:
            return None
        pooled = state.mtp_pooled if ordinal == len(state.kc) else state.pooled[ordinal]
        return pooled[first:first + rows]
    if role == "legacy-index":
        ordinal, rows = values
        value = state.mtp_ikc if ordinal == len(state.kc) else state.ikc[ordinal]
        return value[:rows] if local else None
    if role == "canonical-ids":
        return torch.empty(spec.shape, dtype=torch.int32) if state.history_rank == 0 else None
    if role == "indexer":
        ordinal, field = values
        entry = state.compact_indexer.entries[ordinal]
        if entry is None:
            return None
        return getattr(entry, field)[:spec.shape[0]]
    if spec.name.startswith("recovery."):
        if ".indexer." in spec.name:
            ordinal = int(spec.name.split(".")[3])
            if _owner(state, ordinal) != state.history_rank:
                return None
        elif not local:
            return None
        return torch.empty(spec.shape, dtype=getattr(torch, spec.dtype), device=state.pos_dev.device)
    if not local:
        return None
    if role == "rec":
        return state.rec[0]
    if role in ("conv", "ple_tail"):
        return getattr(state, role)
    raise ValueError("unknown retained core component semantic")


def state_restore_resources(state, records, *, stream=None, continuation="new-turn"):
    """Fresh payload allocations before restore targets are materialized.

    Existing full-window KV/pools/recurrent destinations are views and cost no
    new payload. Recovery versions and deferred HC are independent allocations;
    media coordinates also have CPU raw rows and the issuer's real device copy.
    The byte categories distinguish baseline recovery reserve from phase-only
    overlap, so the caller must not charge existing full windows a second time.
    """
    from .method_retention import method_restore_resources

    if continuation not in ("same-request", "new-turn") or not records:
        raise ValueError("restore resource projection needs actual records and continuation")
    if continuation == "new-turn":
        if stream is None:
            raise ValueError("new-turn resources require the actual fresh request")
        records = tuple((new_turn_metadata(state, metadata, stream), specs) for metadata, specs in records)
    selected = records[state.history_rank][0]
    extra = method_restore_resources(state, selected["method"])
    recovery = hidden = host = 0
    for rank, (_metadata, specs) in enumerate(records):
        for spec in specs:
            if continuation == "new-turn" and not new_turn_uses_part(state, spec, stream):
                continue
            if spec.name.startswith("method."):
                if rank != state.history_rank:
                    continue
                if spec.name.endswith(".coordinates"):
                    host += spec.nbytes
                    # Raw int64 rows are reissued, then copied to owned int64
                    # device coordinates; the int32 transient is separate.
                    if spec.name.startswith("method.recovery."):
                        recovery += spec.nbytes
                    else:
                        hidden += spec.nbytes
                elif spec.name.startswith("method.recovery."):
                    recovery += spec.nbytes
                elif spec.name == "method.mtp.hc":
                    hidden += spec.nbytes
            elif spec.semantic[0] == "canonical-ids":
                if state.history_rank == 0:
                    host += spec.nbytes
            elif spec.name.startswith("recovery."):
                if ".indexer." in spec.name:
                    ordinal = int(spec.name.split(".")[3])
                    local = _owner(state, ordinal) == state.history_rank
                else:
                    local = rank == state.history_rank
                if local:
                    recovery += spec.nbytes
    return {"recovery_device_bytes": recovery, "pending_hc_device_bytes": hidden,
            "phase_device_bytes": extra["device_bytes"], "host_destination_bytes": host,
            "schema": "tensorfold.restore-allocation-payload-v1"}


@dataclass
class RestorePlan:
    """Unpublished destinations; source RAM remains authoritative until commit."""
    state: object
    records: tuple
    targets: dict
    method_targets: tuple
    stream: object
    continuation: str
    byte_targets: dict

    def poison(self, error):
        from .media_mtp import _failed_state

        if getattr(self.state, "dflash2", None) is not None:
            self.state.dflash2.poisoned = True
        _failed_state(self.state, error)


def state_restore_plan(state, records, *, stream=None, continuation="new-turn", recovery_versions):
    """Validate a complete rank manifest before the first persistent write.

    records contains (actual source metadata, PartSpec tuple) in rank order.
    A caller-issued catalog manifest, not external JSON, owns these records.
    No source slot, pointer, owner offset or old generation is restored.
    """
    if continuation not in ("same-request", "new-turn") or not records:
        raise ValueError("retained restore requires an explicit continuation and complete rank records")
    state.kv_check()
    if state.media_input_span is not None or state.target_tentative_length != state.pos:
        raise ValueError("retained restore destination is not exclusively resolved")
    state.retention_method_recovery_versions = recovery_versions
    if continuation == "new-turn":
        if stream is None:
            raise ValueError("new-turn restoration requires its actual fresh request")
        records = tuple((new_turn_metadata(state, metadata, stream), specs) for metadata, specs in records)
    selected = records[state.history_rank][0]
    invariant = ("kv_identity", "limit", "pos", "mtp_len", "mtp_drafted", "rope_delta", "mtp_participating",
                 "media_profile", "media_identity", "canonical_sha256", "request", "next_prompt_input")
    for rank, (metadata, specs) in enumerate(records):
        if (metadata.get("schema") != SCHEMA or metadata.get("source_rank") != rank
                or not _same(metadata["kv_identity"], state.kv_identity)
                or metadata["limit"] != state.limit or len(metadata["recoveries"]) > recovery_versions
                or any(not _same(metadata[key], selected[key]) for key in invariant)
                or type(metadata["mtp_participating"]) is not bool
                or any(type(metadata[key]) is not int or not 0 <= metadata[key] <= state.limit
                       for key in ("pos", "mtp_len", "mtp_drafted"))
                or metadata["mtp_drafted"] > metadata["mtp_len"]
                or metadata["media_profile"] != state.media_profile_identity
                or metadata["media_identity"] != state.media_identity):
            raise ValueError("retained rank state/profile/frontier differs from the current admitted destination")
        if len({spec.name for spec in specs}) != len(specs):
            raise ValueError("retained rank manifest repeats a component")
    if state.capacity < max(selected["pos"], selected["mtp_len"]):
        raise ValueError("retained restore requires its complete GPU window before allocation/write")
    for reader in tuple(state.history_native_readers):
        reader.wait_native_reuse()
    method_specs = tuple(spec for spec in records[state.history_rank][1] if spec.name.startswith("method.")
                         and (continuation != "new-turn" or new_turn_uses_part(state, spec, stream)))
    method = method_restore_targets(state, selected["method"], method_specs, stream=stream, continuation=continuation)
    method_by_name = {part.name: part for part in method}
    targets = {}
    for rank, (_metadata, specs) in enumerate(records):
        for number, spec in enumerate(specs):
            if continuation == "new-turn" and not new_turn_uses_part(state, spec, stream):
                continue
            if spec.name.startswith("method."):
                part = method_by_name.get(spec.name) if rank == state.history_rank else None
            else:
                value = _destination(state, spec, rank)
                part = None if value is None else TensorPart(spec.name, value, spec.category, spec.semantic)
            if part is not None:
                if part.spec() != spec:
                    raise ValueError("retained component does not fit its current destination")
                targets[rank, number] = part
    byte_targets = {key: part.tensor.detach().view(torch.uint8).reshape(-1) for key, part in targets.items()}
    # Parts remain indexed by the original stored table; effective metadata
    # describes only the fresh request's selected committed model state.
    return RestorePlan(state, tuple(records), targets, method, stream, continuation, byte_targets)


def commit_state_restore(plan):
    """After all checked copies/fences; caller still owes all-rank publication."""
    import numpy as np

    state = plan.state
    metadata = plan.records[state.history_rank][0]
    try:
        handoff = commit_method_restore(state, metadata["method"], plan.method_targets,
                                        stream=plan.stream, continuation=plan.continuation)
        if state.compact_indexer is not None:
            for ordinal, entry in enumerate(state.compact_indexer.entries):
                if entry is None:
                    continue
                source = next(record["indexers"][str(ordinal)] for record, _ in plan.records
                              if str(ordinal) in record["indexers"])
                if (plan.continuation == "new-turn" and not plan.stream.draft and ordinal == len(state.kc)):
                    for name in ("raw", "coordinates", "protected", "protected_coordinates"):
                        getattr(entry, name).zero_()
                entry.n, entry.origin, entry.base = source["length"], source["origin"], source["base"]
                entry.proposal_count, entry.span = source["proposal_count"], None
                entry.base_expected = (state.slot_generation, state.history_generation) if entry.base is not None else None
        state.pos, state.target_tentative_length = metadata["pos"], metadata["pos"]
        state.mtp_len, state.mtp_drafted = handoff["mtp_len"], handoff["mtp_drafted"]
        state.pos_dev.fill_(state.pos)
        state.mtp_pos.fill_(state.mtp_len)
        state.cur = [0] * len(state.cur)
        state.ple_history = (None if metadata["ple_history"] is None else
                             np.asarray(metadata["ple_history"], dtype=np.int64))
        state.ple_last = None
        if plan.continuation == "new-turn" and state.mtp_drafted:
            # Rewind actual tentative MTP pools/raw base, not just Python
            # counters. Parent proposals belong to the parent's request/RNG.
            state.set_mtp_len(state.mtp_len - state.mtp_drafted)
            state.mtp_drafted = 0
            handoff["mtp_len"], handoff["mtp_drafted"] = state.mtp_len, 0
        state.media_mtp_enabled = metadata["mtp_participating"]
        restored = []
        by_name = {part.name: part.tensor for part in plan.targets.values()}
        for number, record in enumerate(metadata["recoveries"]):
            prefix = f"recovery.{number}."
            snapshot = {key: record[key] for key in ("pos", "mtp_len")}
            snapshot.update(state.kv_snapshot_metadata())
            for name in ("rec", "conv", "ple_tail"):
                snapshot[name] = by_name[prefix + name]
            snapshot["ple_history"] = (None if record["ple_history"] is None else
                                        np.asarray(record["ple_history"], dtype=np.int64))
            compact = record["compact_indexer"]
            if compact is not None:
                tails = []
                for ordinal, entry in enumerate(state.compact_indexer.entries):
                    if entry is None:
                        tails.append(None)
                        continue
                    source = next(row["recoveries"][number]["compact_indexer"]["tails"][ordinal]
                                  for row, _ in plan.records
                                  if row["recoveries"][number]["compact_indexer"]["tails"][ordinal] is not None)
                    tails.append({**source, **{field: by_name[prefix + f"indexer.{ordinal}.{field}"]
                                              for field in ("raw", "coordinates")}})
                snapshot["compact_indexer"] = {**compact, "tails": tails}
            methods = handoff["recoveries"]
            if methods:
                snapshot.update({key: value for key, value in methods[number].items()
                                 if key not in ("pos", "mtp_len", "pending_hc", "mtp_drafted")})
            tail = by_name.get(prefix + "pending_hc")
            if tail is None and methods:
                tail = methods[number]["pending_hc"]
            restored.append((snapshot, tail))
        handoff["recoveries"] = tuple(restored)
        handoff["canonical_ids"] = by_name.get("state.canonical_ids")
        return handoff
    except BaseException as error:
        plan.poison(error)
        raise

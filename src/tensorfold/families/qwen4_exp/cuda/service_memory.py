"""Remaining service storage after fixed CUDA state has been prepared, before requests start.

The caller selects the growth contract: fixed expert pools keep all selected
windows. Resident graph capture keeps one complete window by default; an
explicit count protects that many complete windows. Snapshot and transient
allowances share the same accounting in both policies; this module allocates no
device storage and imports no accelerator runtime.
"""

RAM_GRAPH_ADMISSION_VERSION = "observed-graph-admission-v1"
GPU_UNIQUE_GRAPH_ADMISSION_VERSION = "gpu-unique-full-resident-observed-v1"


def context_reservation(states, full_windows=None, *, preserve_overlap=False):
    """Remaining cache growth, including auxiliary caches through State.cache_bytes.

    The legacy resident policy promises one window. An explicit count reserves
    the largest N complete windows and every resize keeps its old storage live.
    DSpark replaces its whole auxiliary cache, while target/MTP resize by layer.
    """
    if full_windows is not None and (type(full_windows) is not int or full_windows < 1):
        raise ValueError("full KV windows must be a positive integer")
    each = [max(0, st.cache_bytes(st.limit) - st.cache_bytes()) for st in states]
    windows = min(len(each), 1 if full_windows is None else full_windows)
    growth = sum(sorted(each, reverse=True)[:windows])
    overlap = max((cache_resize_overlap(st, st.limit, auxiliary=full_windows is not None)
                   for st in states), default=0) if growth or preserve_overlap else 0
    return {"growth_bytes": growth, "copy_overlap_bytes": overlap,
            "full_context_growth_bytes": sum(each), "full_windows_reserved": windows,
            "stream_slots": len(states)}


def cache_resize_overlap(state, rows, *, auxiliary=False):
    target = (state.layer_resize_bytes(rows) if hasattr(state, "layer_resize_bytes") else state.layer_bytes(rows))
    draft = getattr(state, "dspark_runtime", None)
    return max(target, rows * draft.plan.cache_row_bytes) if auxiliary and draft is not None else target


def future_service(w, states, *, streams, retained, growth_bytes,
                   vision_workspace_bytes=0, minimum_workspace_bytes=0, copy_overlap_bytes=None):
    if copy_overlap_bytes is not None and (type(copy_overlap_bytes) is not int or copy_overlap_bytes < 0):
        raise ValueError("cache resize overlap must be nonnegative integer bytes")
    state = states[0]
    if copy_overlap_bytes is None:
        overlap = max((cache_resize_overlap(st, st.limit) for st in states), default=0) if growth_bytes else 0
    else:
        overlap = copy_overlap_bytes
    wide = w.cfg.streams * w.cfg.hidden
    snapshot = state.rec[0].numel() * state.rec.element_size() + state.conv.numel() * state.conv.element_size()
    snapshot += state.ple_tail.numel() * state.ple_tail.element_size()
    snapshot += wide * 2 if w.mtp is not None else 0
    draft_window = getattr(state, "dflash2", None)
    snapshot += draft_window.nbytes if draft_window is not None else 0
    snapshots = retained * snapshot
    transient = streams * (w.head.n * 4 + wide * 2) + vision_workspace_bytes
    workspace = max(transient, minimum_workspace_bytes)
    return {"growth_bytes": growth_bytes, "copy_overlap_bytes": overlap,
            "snapshot_bytes_each": snapshot, "snapshot_bytes": snapshots,
            "transient_bytes": transient, "workspace_bytes": workspace,
            "future_bytes": growth_bytes + overlap + snapshots + workspace}


def automatic_graph_bytes(live_bytes, future_bytes, header_room=None):
    """No arbitrary cap: the floor is already charged by the caller's live-memory provider."""

    room = max(0, live_bytes - future_bytes)
    return room if header_room is None else min(room, max(0, header_room))


def materialized_graph_room(states, capacity, live_bytes, promise, future):
    """Observed spare after every RAM slot already owns its full GPU window.

    ``live_bytes`` is the existing physical-only reader after its device floor
    and absolute cap. No allocator/private-pool reclaim credit is introduced.
    """
    if (type(live_bytes) is not int or live_bytes < 0 or not states
            or promise["growth_bytes"] != 0 or promise["full_windows_reserved"] != len(states)
            or any(st.capacity != capacity or st.limit != capacity
                   or getattr(st, "indexer_history_owner", None) is None
                   or st.indexer_history_owner.limit != capacity for st in states)):
        raise ValueError("observed graph credit requires every full RAM window materialized with zero future growth")
    return _materialized_service_room(live_bytes, future)


def materialized_gpu_graph_room(states, capacity, live_bytes, promise, future):
    """Fixed unique-GPU arenas: observe spare only after all promised slots exist.

    State construction allocates full Torch-backed KV arenas, compact indexers
    and pooled histories before publication. Neither fixed arena growth nor an
    in-flight ownership migration can be credited as already materialized.
    """
    if (type(live_bytes) is not int or live_bytes < 0 or not states
            or promise["growth_bytes"] != 0 or promise["copy_overlap_bytes"] != 0
            or promise["full_windows_reserved"] != len(states)):
        raise ValueError("observed GPU graph credit requires every full fixed window with zero future growth")
    for state in states:
        placement = getattr(state, "history_placement", None)
        indexer = getattr(state, "compact_indexer", None)
        caches = (*state.kc, state.mtp_kc)
        if (state.capacity != capacity or state.limit != capacity or placement is None
                or state.indexer_history_mode != "gpu" or state.indexer_history_owner is not None
                or indexer is None or indexer.capacity != capacity or indexer.placement is not placement
                or getattr(state, "_history_migration", None) is not None
                or any(cache.logical_capacity != capacity or cache.placement is not placement
                       or cache.slot != state.history_slot or cache.rank != state.history_rank for cache in caches)):
            raise ValueError("observed GPU graph credit requires materialized unique KV/indexer ownership")
    if future.get("copy_overlap_bytes") != 0:
        raise ValueError("fixed GPU histories cannot credit a pending replacement overlap")
    return _materialized_service_room(live_bytes, future)


def _materialized_service_room(live_bytes, future):
    """One unchanged service reservation for both fixed-history policies."""
    fields = ("growth_bytes", "copy_overlap_bytes", "snapshot_bytes", "workspace_bytes", "future_bytes")
    if (any(type(future.get(key)) is not int or future[key] < 0 for key in fields)
            or future["workspace_bytes"] < 2 * 2**30 or future["growth_bytes"] != 0
            or future["future_bytes"] != sum(future[key] for key in fields[:-1])):
        raise ValueError("observed graph credit requires the unchanged complete service promises")
    if live_bytes < future["future_bytes"]:
        raise MemoryError("materialized windows no longer leave room for required service promises")
    return live_bytes - future["future_bytes"]


def refine_materialized_graph_plan(receipt, proof):
    """Retain header estimates and publish a separately proven serving envelope."""
    from copy import deepcopy

    room, budget = proof["agreed_graph_bytes"], receipt["budget_bytes"]
    if (receipt.get("observed_graph_admission") is not None or type(room) is not int
            or not 0 <= room <= budget or proof.get("policy") not in
            ("ram-full-resident-observed-v1", GPU_UNIQUE_GRAPH_ADMISSION_VERSION)):
        raise RuntimeError("invalid or repeated observed full-window graph admission")
    original = {key: deepcopy(receipt[key]) for key in (
        "native_window", "requested_context", "context_window", "cache_slots", "budget_bytes",
        "weight_bytes_estimate", "loading_bytes_estimate", "cache_workspace_bytes_estimate",
        "startup_peak_bytes_estimate", "serving_peak_bytes_estimate", "total_bytes_estimate",
        "full_mapped_working_set_bytes_estimate") if key in receipt}
    serving = budget - room
    if serving < receipt["weight_bytes_estimate"]:
        raise RuntimeError("observed serving envelope cannot cover admitted resident weights")
    receipt["observed_graph_admission"] = {"original_header": original, **deepcopy(proof),
        "observed_serving_base_bytes": serving,
        "basis": "physical spare after unchanged device floor and full service promises"}
    receipt.update(serving_admission_basis=proof["policy"],
        cache_workspace_bytes_estimate=serving - receipt["weight_bytes_estimate"],
        serving_peak_bytes_estimate=serving,
        total_bytes_estimate=max(receipt["startup_peak_bytes_estimate"], serving),
        full_mapped_working_set_bytes_estimate=max(receipt["startup_peak_bytes_estimate"], serving)
                                                + receipt["mapped_table_bytes"])


def reserve_graph_plan(receipt, graph_bytes):
    """Retain graph-free context admission, then publish its resolved serving ceiling before requests."""
    from copy import deepcopy

    serving = receipt["serving_peak_bytes_estimate"] + graph_bytes
    total = max(receipt["startup_peak_bytes_estimate"], serving)
    if graph_bytes < 0 or total > receipt["budget_bytes"]:
        raise RuntimeError("resolved captured-round budget exceeds its admitted service plan")
    initial = deepcopy(receipt)
    receipt.update(initial_graph_admission=initial,
                   cache_workspace_bytes_estimate=receipt["cache_workspace_bytes_estimate"] + graph_bytes,
                   serving_peak_bytes_estimate=serving, total_bytes_estimate=total,
                   full_mapped_working_set_bytes_estimate=total + receipt["mapped_table_bytes"])


def release_uncaptured_graph_plan(receipt, graph_bytes):
    """Release only a pre-load reservation declined before a graph owner was created."""
    if type(graph_bytes) is not int or not 0 <= graph_bytes <= receipt["cache_workspace_bytes_estimate"]:
        raise RuntimeError("invalid uncaptured graph reservation")
    serving = receipt["serving_peak_bytes_estimate"] - graph_bytes
    total = max(receipt["startup_peak_bytes_estimate"], serving)
    receipt.update(uncaptured_graph_reservation_bytes=graph_bytes,
                   cache_workspace_bytes_estimate=receipt["cache_workspace_bytes_estimate"] - graph_bytes,
                   serving_peak_bytes_estimate=serving, total_bytes_estimate=total,
                   full_mapped_working_set_bytes_estimate=total + receipt["mapped_table_bytes"])


def private_pool_spare(segments, pool, device):
    """Quiescent public memory_snapshot schema; no addresses escape preparation.

    Replays allocate no Python tensors into this pool. Its captured pieces and
    persistent outputs retain the same allocation lifetimes until the whole
    decoder retires. Awaiting-free blocks would invalidate a constant receipt.
    """
    spare = 0
    for segment in segments:
        if (not isinstance(segment, dict) or segment.get("device") != device
                or not isinstance(segment.get("segment_pool_id"), (tuple, list))
                or tuple(segment["segment_pool_id"]) != tuple(pool)):
            raise ValueError("graph pool snapshot contains a different device or pool")
        sizes = [segment.get(name) for name in ("total_size", "allocated_size", "active_size")]
        if (any(type(value) is not int for value in sizes)
                or not 0 <= sizes[1] == sizes[2] <= sizes[0]):
            raise ValueError("graph pool snapshot must expose settled allocated/active/total byte counts")
        spare += sizes[0] - sizes[1]
    return spare


def failure_frame_values(primary, traceback, cause, context):
    """Own failure-local storage before consumers may clear the exception's frames.

    The journal is bounded by this single capture's distinct error graph and
    frames; cycles are visited once. Values are strongly retained, not copied
    or interpreted as tensors. Capture callbacks have completed before entry;
    the error payload/journal is operation-owned and read-only to consumers.
    Allocation failure propagates to the capture's poisoned recovery path.
    """
    pending = [primary, cause, context]
    seen_errors, seen_frames, values = set(), set(), []
    while pending:
        error = pending.pop()
        if error is None or id(error) in seen_errors:
            continue
        seen_errors.add(id(error))
        pending.append(BaseException.__cause__.__get__(error))
        pending.append(BaseException.__context__.__get__(error))
        if isinstance(error, BaseExceptionGroup):
            pending.extend(BaseExceptionGroup.exceptions.__get__(error))
        attributes = BaseException.__dict__["__dict__"].__get__(error)
        values.append(tuple(attributes.values()))
        trace = traceback if error is primary else BaseException.__traceback__.__get__(error)
        while trace is not None:
            frame = trace.tb_frame
            if id(frame) not in seen_frames:
                seen_frames.add(id(frame))
                values.append(tuple(frame.f_locals.values()))
            trace = trace.tb_next
    return tuple(values)

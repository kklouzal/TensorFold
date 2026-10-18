"""Remaining service storage after fixed CUDA state has been prepared, before requests start.

The caller selects the growth contract: fixed expert pools keep all selected
windows, resident graph capture keeps one complete window and lets additional
concurrent contexts grow under the existing memory gate. Snapshot and transient
allowances share the same accounting in both policies; this module allocates no
device storage and imports no accelerator runtime.
"""


def future_service(w, states, *, streams, retained, growth_bytes,
                   vision_workspace_bytes=0, minimum_workspace_bytes=0):
    state = states[0]
    overlap = max((st.layer_bytes(st.limit) for st in states), default=0) if growth_bytes else 0
    wide = w.cfg.streams * w.cfg.hidden
    snapshot = state.rec[0].numel() * state.rec.element_size() + state.conv.numel() * state.conv.element_size()
    snapshot += state.ple_tail.numel() * state.ple_tail.element_size()
    snapshot += wide * 2 if w.mtp is not None else 0
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

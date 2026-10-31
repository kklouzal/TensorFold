"""Frame lifecycle bridge for lossless RAM indexer histories.

Only enqueue belongs inside a captured forward. Registration and external
completion records belong outside real eager/graph execution; the common
State numeric latch publishes validated archive bytes after its existing wait.
GPU-history States are no-ops and retain their original forward protocol.
"""


def _owned(segs):
    result, seen = [], set()
    for state, start, end in segs:
        owner = getattr(state, "indexer_history_owner", None)
        if owner is None:
            continue
        owner.check()
        if id(owner) in seen:
            raise ValueError("one indexer history owner per forward segment required")
        seen.add(id(owner))
        result.append((state, owner, start, end))
    return result


def _capturing(records):
    if not records or not records[0][1].ring.is_cuda:
        return False
    import torch

    return torch.cuda.is_current_stream_capturing()


def _layers(state, owner, mtp):
    target = len(state.kc)
    if mtp:
        if not hasattr(state, "mtp_ikc") or target >= owner.plan.layers:
            raise ValueError("MTP history requires its separately owned layer")
        return (target,)
    return tuple(range(target))


def begin(segs, *, prefill=False, mtp=False, positions=None, counts=None):
    """Register logical rows before execution, never padded graph row counts.

    Bound wrappers may supply explicit ``positions``/``counts``. Otherwise a
    bound view must expose the host ``history_rows`` refreshed alongside its
    device graph_rows scalar; no device readback or padded-count guess occurs.
    The normal compute wrapper may call this during capture: it is a no-op,
    because the bound wrapper has already registered the actual replay frame.
    """
    records = _owned(segs)
    if not records:
        return ()
    if _capturing(records):
        return ()
    if positions is not None and len(positions) != len(segs):
        raise ValueError("one host history position per segment required")
    if counts is not None and len(counts) != len(segs):
        raise ValueError("one logical history row count per segment required")
    source_index = {id(state): index for index, (state, _, _) in enumerate(segs)}
    prepared = []
    kind = "prefill" if prefill else "decode"
    for state, owner, start, end in records:
        index = source_index[id(state)]
        position = positions[index] if positions is not None else (state.mtp_len if mtp else state.pos)
        if counts is not None:
            rows = counts[index]
        elif getattr(state, "graph_rows", None) is not None:
            if not hasattr(state, "history_rows"):
                raise ValueError("bound history views need their host logical row count")
            rows = state.history_rows
        else:
            rows = end - start
        if (type(position) is not int or position < 0 or type(rows) is not int or rows < 0
                or rows > end - start or position + rows > owner.limit):
            raise ValueError("invalid logical history frame extent")
        prepared.append((owner, position, rows, _layers(state, owner, mtp)))
    return tuple(owner.begin(kind, position, rows, layers=layers)
                 for owner, position, rows, layers in prepared)


def enqueue(segs, *, prefill=False, mtp=False):
    """Capture-safe copies with explicit immutable model-layer selections."""
    kind = "prefill" if prefill else "decode"
    for state, owner, _, _ in _owned(segs):
        owner.frames[kind].enqueue(_layers(state, owner, mtp))


def submitted(segs, *, prefill=False, mtp=False, stream=None):
    """Record actual completion outside capture, before the existing status wait."""
    records = _owned(segs)
    if _capturing(records):
        return
    kind = "prefill" if prefill else "decode"
    for _, owner, _, _ in records:
        owner.frames[kind].mark_submitted(stream)


def abort(segs, *, prefill=None, fenced=False):
    """Discard provisional journals during explicit warm/failure retirement."""
    kind = None if prefill is None else "prefill" if prefill else "decode"
    for _, owner, _, _ in _owned(segs):
        owner.abort(kind, fenced=fenced)

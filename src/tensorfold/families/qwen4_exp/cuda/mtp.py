"""The MTP head reads final residual streams and next-token embeddings, appending its own attention entries that the next absorb trims after chained drafts."""

from __future__ import annotations

from typing import Sequence

import torch

from tensorfold.cleanup import rollback

from . import glue, history_runtime
from .forward import _embed, _mm, _kv_begin, _kv_check, candidates, finish, layer_forward
from .state import Buffers, State
from .weights import Weights


def mtp_stage(b: Buffers, windows: Sequence[tuple[State, Sequence[int], torch.Tensor | tuple[torch.Tensor, ...]]], *,
              indexer_coordinates: tuple | None = None) -> list:
    """Own all MTP input/window preparation before any dependent collective."""
    context = getattr(b, "input_preparation_context", None)
    if context is not None:
        return context.prepare("mtp-stage", lambda: _mtp_stage(b, windows, indexer_coordinates=indexer_coordinates))
    if getattr(b, "mtp_indexer_position", None) is not None:
        raise RuntimeError("bound MTP staging requires its installed cohort preparation owner")
    return _mtp_stage(b, windows, indexer_coordinates=indexer_coordinates)


def _mtp_stage(b: Buffers, windows: Sequence[tuple[State, Sequence[int], torch.Tensor | tuple[torch.Tensor, ...]]], *,
              indexer_coordinates: tuple | None = None) -> list:
    """Stage original next IDs and complete HC inputs under one buffer event.

    A deferred window may provide two HC spans (the previous boundary and the
    current target chunk), avoiding a concatenated embedding/feature archive.
    Explicit coordinates describe those hidden/cache rows, never next IDs.
    Strided HC reads are supported. Sources aliasing any staged destination
    are snapshotted before writes unless their semantic layout exactly matches
    their own HC destination; snapshots total at most this frame's HC bytes.
    """

    if not windows:
        raise ValueError("MTP staging requires at least one nonempty window")
    segs, pieces = [], []
    for st, next_tokens, streams in windows:
        a0 = segs[-1][2] if segs else 0
        if not next_tokens or a0 + len(next_tokens) > b.rows:
            raise ValueError("MTP staged rows exceed the admitted nonempty buffer extent")
        if st.mtp_len + len(next_tokens) > st.capacity:
            raise ValueError("MTP context past the cache capacity")
        segs.append((st, a0, a0 + len(next_tokens)))
        parts = (streams,) if isinstance(streams, torch.Tensor) else streams
        if (type(parts) is not tuple or not 1 <= len(parts) <= 2
                or any(not isinstance(part, torch.Tensor) or part.ndim != 2
                       or part.shape[1] != b.mtp_in.shape[1] or part.dtype != b.mtp_in.dtype
                       or part.device != b.mtp_in.device for part in parts)
                or sum(part.shape[0] for part in parts) != len(next_tokens)):
            raise ValueError("MTP staging requires one complete HC window in at most two owned spans")
        pieces.append(parts)
    n = segs[-1][2]
    if indexer_coordinates is None and any(getattr(st, "media_input_span", None) is not None
                                           for st, _begin, _end in segs):
        raise ValueError("ready media MTP hidden rows require their actual issued coordinate windows")
    b.staged.synchronize()
    def span(tensor):
        if not tensor.numel():
            return None
        offsets = [(size - 1) * stride for size, stride in zip(tensor.shape, tensor.stride())]
        first = tensor.data_ptr() + sum(min(0, offset) for offset in offsets) * tensor.element_size()
        last = tensor.data_ptr() + (sum(max(0, offset) for offset in offsets) + 1) * tensor.element_size()
        return tensor.device, first, last

    writes = [span(tensor) for tensor in (b.mtp_in[:n], b.ids_host[:n], b.ids[:n],
                                          b.last_host[:len(segs)], b.last[:len(segs)])]
    copies = []
    for parts, (_, a0, _a1) in zip(pieces, segs):
        for part in parts:
            rows = part.shape[0]
            destination = b.mtp_in[a0:a0 + rows]
            exact = (part.data_ptr() == destination.data_ptr() and part.shape == destination.shape
                     and part.stride() == destination.stride() and part.is_neg() == destination.is_neg()
                     and part.is_conj() == destination.is_conj())
            if rows and not exact:
                device, first, last = span(part)
                overlap = any(region is not None and region[0] == device and first < region[2] and region[1] < last
                              for region in writes)
                copies.append([destination, part, overlap])
            a0 += rows
    owner = (b, copies)
    try:
        for copy in copies:
            if copy[2]:
                copy[1] = copy[1].clone()
        b.ids_host[:n].numpy()[:] = [int(t) for _, next_tokens, _ in windows for t in next_tokens]
        b.ids[:n].copy_(b.ids_host[:n], non_blocking=True)
        b.last_host[:len(segs)].numpy()[:] = [a1 - 1 for _, _, a1 in segs]
        b.last[:len(segs)].copy_(b.last_host[:len(segs)], non_blocking=True)
        for destination, source, _overlap in copies:
            destination.copy_(source, non_blocking=True)
        if indexer_coordinates is None and getattr(b, "mtp_indexer_position", None) is not None:
            from .compact_indexer import CoordinateWindow, issue_text_indexer_coordinates

            coordinate_windows = []
            for st, begin, end in segs:
                at = begin
                boundary = getattr(st, "mtp_boundary", None)
                if boundary is not None:
                    if boundary.row != st.mtp_len or boundary.generation != st.slot_generation:
                        raise ValueError("deferred MTP boundary lost its cache-row/generation binding")
                    coordinate_windows.append(CoordinateWindow(at, at + 1, boundary.coordinate_record, boundary.position))
                    at += 1
                if at < end:
                    coordinates = issue_text_indexer_coordinates(st, b.mtp_indexer_position,
                        st.mtp_len + at - begin, end - at, max_rows=b.rows)
                    coordinate_windows.append(CoordinateWindow(at, end, coordinates, b.mtp_indexer_position))
            indexer_coordinates = tuple(coordinate_windows)
        if indexer_coordinates is not None or hasattr(b, "mtp_indexer_coordinate_storage"):
            from .compact_indexer import stage_mtp_indexer_frame

            stage_mtp_indexer_frame(b, segs, indexer_coordinates)
    except BaseException as error:
        rollback(owner, error, b.staged.record)  # Fence issued snapshots/copies even after a later failure.
    b.staged.record()
    return segs


def mtp_compute(w: Weights, segs: Sequence, b: Buffers, *, last_only: bool = True, logits: bool = True,
                context: int | None = None, check: bool = True,
                next_inputs: torch.Tensor | None = None) -> torch.Tensor | None:
    """The MTP head's GPU work (capturable); ``last_only``: the draft head's logits of each stream's last row.
    ``logits=False`` absorbs into the MTP cache and skips the unused draft-vocabulary projection. ``check=False``
    (a draft step's logits): the codec status stays pending for the caller's check before its rows are used.

    ``next_inputs`` supplies the complete window's actual merged next-position
    input embeddings, including projected visual rows. Its dtype/device match
    the already-admitted MTP input buffer; no visual placeholder lookup occurs.
    The caller validates row/ID/profile/generation/coordinate binding and owns
    immutable producer storage through its staging-completion lease. Embedding
    input positions t+1 remain distinct from the HC/cache row coordinates t.
    """

    n = segs[-1][2]
    c = w.cfg
    if next_inputs is not None and (
            not isinstance(next_inputs, torch.Tensor) or next_inputs.shape != (n, c.hidden)
            or next_inputs.dtype != b.mtp_e.dtype or next_inputs.device != b.mtp_e.device
            or next_inputs.requires_grad):
        from .input_preparation import reject_prepared

        reject_prepared(b, ValueError("MTP next inputs must cover the complete window on its admitted dtype/device without gradients"))
    if next_inputs is None and any(getattr(st, "media_input_span", None) is not None for st, _a, _z in segs):
        from .input_preparation import reject_prepared

        reject_prepared(b, ValueError("ready media MTP input cannot fall back to placeholder-token lookup"))
    if getattr(b, "compact_indexer_stage", None) is not None:
        from .compact_indexer import validate_mtp_indexer_frame

        validate_mtp_indexer_frame(b, segs)

    archive = getattr(w, "meta", {}).get("indexer_history") == "ram"
    if archive:
        history_runtime.begin(segs, prefill=b.prefill, mtp=True)
    _kv_begin(segs)
    m = w.mtp
    if next_inputs is None:
        _embed(w, b.ids[:n], 1, b.mtp_e[:n])
    else:
        b.mtp_e[:n].copy_(next_inputs, non_blocking=True)
        if next_inputs.is_cuda:
            # The producer's explicit lease also governs reuse/mutation. This
            # records allocator lifetime when the source allocation was made
            # on another stream; consumers use this operation's stream.
            next_inputs.record_stream(torch.cuda.current_stream(next_inputs.device))
    en, xe = glue.rmsnorm(b.mtp_e[:n], m.norm_e, c.eps, out=b.mixed[:n], xs=b.xs_mixed[:n])
    _mm(en, m.fc_e, xe, b.mtp_eo[:n], b)
    hn, xh = glue.rmsnorm(b.mtp_in[:n], m.norm_h, c.eps, out=b.mtp_hn[:n], xs=b.mtp_xh[:n])
    _mm(hn.view(n * c.streams, c.hidden), m.fc_h, xh.view(n * c.streams, c.hidden // 32), b.mtp_hs[:n * c.streams], b)
    glue.add_streams(b.mtp_eo[:n], b.mtp_hs[:n * c.streams], b.h[:n], c.streams)
    pending = layer_forward(m.layer, w, segs, b, n, None, mtp=True, context=context)
    finish(w, m.mixer, b, n, pending, logits=False)
    # The archive copy precedes every possible validation/return branch. The
    # external completion event is recorded only for actual eager submissions;
    # graph owners record it after replay on the real producing stream.
    if archive:
        history_runtime.enqueue(segs, prefill=b.prefill, mtp=True)
        history_runtime.submitted(segs, prefill=b.prefill, mtp=True)
    if not logits:
        _kv_check(segs, comm=w.comm)
        return None
    if not last_only:
        result = _mm(b.mixed[:n], w.head, b.xs_mixed[:n], b.logits[:n], b)
        _kv_check(segs, comm=w.comm)
        return result
    head = w.head if w.draft_head is None else w.draft_head
    k = len(segs)
    if b.prefill:                                # prefill buffers mix the last row into row 0: its logits only
        rows, xs, k = b.mixed[:1], b.xs_mixed[:1], 1
    elif k == 1 and not getattr(b, "dynamic_last", False):
        rows, xs = b.mixed[n - 1:n], b.xs_mixed[n - 1:n]
    else:
        # A bound graph's physical rows include padding even for one segment.
        # Its real last row arrives through the same device index as batching.
        rows = b.mixed.index_select(0, b.last[:k])
        xs = b.xs_mixed.index_select(0, b.last[:k])
    out = _mm(rows, head, xs, b.logits.view(-1)[:k * head.n].view(k, head.n), b)     # contiguous at any k
    if w.draft_filter is not None:
        # Shared-target MTP always executes the original full target head.
        # Filtering changes only this explicitly selected proposal support;
        # target logits and all-row history initialization keep their full law.
        out.masked_fill_(w.draft_filter, -float("inf"))
    if w.comm is not None:
        candidates(w, b, out, k, id_map=w.draft_ids, offset=int(w.meta["vocab_offset"]))
    if check:
        _kv_check(segs, comm=w.comm)
    return out


@torch.no_grad()
def mtp_forward(w: Weights, st: State, b: Buffers, next_tokens: Sequence[int], streams: torch.Tensor,
                *, last_only: bool = True, logits: bool = True,
                next_inputs: torch.Tensor | None = None,
                indexer_coordinates: tuple | None = None) -> torch.Tensor | None:
    """Process bf16 main-model streams and next tokens into last-row or all-row logits plus b.streams[:n], appending n cache entries while the caller advances ``st.mtp_len``. ``logits=False`` skips the unused draft-vocabulary projection."""

    def staging():
        return mtp_stage(b, [(st, next_tokens, streams)],
                         **({"indexer_coordinates": indexer_coordinates} if indexer_coordinates is not None else {}))

    context = getattr(b, "input_preparation_context", None)
    segs = staging() if context is None else context.prepare("mtp-forward-window", staging)
    return mtp_compute(w, segs, b, last_only=last_only, logits=logits,
                       **({"next_inputs": next_inputs} if next_inputs is not None else {}))

"""Bounded merged-input consumers and one deferred, completed-HC MTP row.

The input producer owns a current chunk plus optional one-row lookahead. State
checkpoints own only logical identity, continuation state and their one HC row;
they never copy a live producer lease or retain a prompt embedding archive.
"""

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass

import torch

from tensorfold.cleanup import raise_failures, rollback

from .media_inputs import ReadyInputSpan, _digest
from .position_coordinates import IssuedCoordinates, copy_coordinates, restore_coordinates


def boundary_resources(*, hidden_width, slots, keep, points):
    """Actual deferred-HC payload plus each protected coordinate version.

    Existing MTP geometry already charges one HC tail per protected snapshot.
    Consumers add the live boundary and protected coordinates once, rather
    than adding a second protected HC copy for the new metadata container.
    """
    if (type(hidden_width) is not int or hidden_width <= 0 or type(slots) is not int or slots <= 0
            or any(type(value) is not int or value < 0 for value in (keep, points))):
        raise ValueError("explicit positive boundary geometry and bounded protected-version counts required")
    hc, coordinates, versions = hidden_width * 2, 3 * 8, keep + slots * points
    return {"live_hc_bytes": slots * hc, "live_coordinate_bytes": slots * coordinates,
            "protected_versions": versions, "protected_hc_bytes_already_charged": versions * hc,
            "protected_coordinate_bytes": versions * coordinates,
            "additional_device_bytes": slots * (hc + coordinates) + versions * coordinates}


@dataclass(frozen=True)
class MTPBoundary:
    row: int
    hidden: torch.Tensor
    coordinate_record: IssuedCoordinates
    position: object
    generation: int

    @property
    def coordinates(self):
        return self.coordinate_record.tensor

    @property
    def nbytes(self):
        return self.hidden.numel() * self.hidden.element_size() + self.coordinates.numel() * self.coordinates.element_size()

    def clone(self, generation, *, profile_identity, media_identity, position):
        copied = restore_coordinates(self.coordinate_record, source_position=self.position, position=position,
            profile_identity=profile_identity, media_identity=media_identity, generation=generation,
            row_start=self.row, row_count=1, dtype=torch.int64)
        return MTPBoundary(self.row, self.hidden.clone(), copied, position, generation)


@dataclass(frozen=True)
class CommittedMTPChunk:
    state: object
    prompt: list[int]
    start: int
    end: int
    hidden: torch.Tensor
    checkpoints: tuple[int, ...] = ()


def condition_chunks(weights, buffers, chunks):
    """Stage committed HC/actual next inputs uniformly, then condition MTP.

    Row q consumes h_q and the actual committed input at q+1. The final HC
    row stays pending for the next prompt input or permitted sampled anchor.
    Producer leases remain owned through staging, the cohort vote and compute.
    """
    from .mtp import mtp_compute

    with ExitStack() as consumers:
        context = getattr(buffers, "input_preparation_context", None)

        def operation():
            return _condition_inputs(weights, buffers, chunks, consumers)

        if context is None:
            if getattr(weights, "positions", None) is not None and getattr(buffers, "compact_indexer_stage", None) is not None:
                raise RuntimeError("bound MTP inputs require their installed cohort preparation owner")
            staged = operation()
        else:
            staged = context.prepare("mtp-committed-inputs", operation)
        segments, plans, results = staged
        if segments:
            mtp_compute(weights, segments, buffers, logits=False, next_inputs=buffers.mtp_e[:segments[-1][2]])
            for st, _ids, _parts, _values, _carry, _coords, new_mtp_len in plans:
                st.set_mtp_len(new_mtp_len)
        for chunk, (ending, _cuts) in zip(chunks, results):
            chunk.state.mtp_boundary = ending
    return results


def _condition_inputs(weights, buffers, chunks, consumers):
    from .mtp import mtp_stage
    from .prompt_plan import POINTS

    if (type(chunks) not in (tuple, list) or not chunks or len(chunks) > buffers.rows
            or len({id(chunk.state) for chunk in chunks}) != len(chunks)
            or sum(chunk.end - chunk.start for chunk in chunks) > buffers.rows):
        raise ValueError("MTP committed chunks must fit one bounded unique-State frame")
    plans, results, total, segments = [], [], 0, None
    for chunk in chunks:
        st, start, end, hidden = chunk.state, chunk.start, chunk.end, chunk.hidden
        span = st.media_input_span
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= len(chunk.prompt)
                or st.pos != end or not st.media_mtp_enabled or st.mtp_drafted
                or hidden.shape != (end - start, st.mtp_hidden_width)
                or len(chunk.checkpoints) > POINTS):
            raise ValueError("MTP conditioning needs committed target rows and their bounded ready span")
        if span is None:
            if st.media_profile_identity is not None or st.image_features is not None:
                raise ValueError("media conditioning cannot fall back to placeholder embeddings")
            ids = chunk.prompt[start + 1:end + 1]
            if ids:
                plans.append((st, ids, (hidden[:len(ids)],), None, None, None, st.mtp_len + len(ids)))
                total += len(ids)
            results.append((None, {}))
            continue
        if span.start != start:
            raise ValueError("MTP ready span is not aligned with this target chunk")
        lease = consumers.enter_context(borrow_span(st, "mtp"))
        lease.rows(start, chunk.prompt[start:end])  # validate hidden-row source/identity before reading coordinates
        z = st.mtp_len
        if z not in (max(0, start - 1), start):
            raise ValueError("MTP valid frontier is not aligned with its deferred target chunk")
        current = hidden[:end - start - 1]
        parts = (current,)
        carry = None
        if z < start:
            carry = st.mtp_boundary
            if (carry is None or carry.row != z or carry.generation != st.slot_generation):
                raise ValueError("MTP deferred HC boundary is absent or belongs to another sequence")
            parts = (carry.hidden, current)
        ending = capture_boundary(st, end, hidden[-1:], span.coordinates, source_position=span.position)
        cuts = {}
        for cut in chunk.checkpoints:
            if type(cut) is not int or not start < cut <= end:
                raise ValueError("MTP checkpoint cut is outside its committed target chunk")
            cuts[cut] = capture_boundary(st, cut, hidden[cut - start - 1:cut - start],
                                         span.coordinates, source_position=span.position)
        results.append((ending, cuts))
        ids = chunk.prompt[z + 1:end]
        if ids:
            values = lease.rows(z + 1, ids)
            plans.append((st, ids, parts, values, carry, span, end - 1))
            total += len(ids)
    if total:
        from .compact_indexer import CoordinateWindow, issue_text_indexer_coordinates

        coordinate_windows = []
        at = 0
        for _st, ids, _parts, _values, carry, span, _end in plans:
            row = at
            if span is None:
                coordinates = issue_text_indexer_coordinates(_st, buffers.mtp_indexer_position,
                    _st.mtp_len, len(ids), max_rows=buffers.rows)
                coordinate_windows.append(CoordinateWindow(at, at + len(ids), coordinates,
                                                             buffers.mtp_indexer_position))
            else:
                if carry is not None:
                    coordinate_windows.append(CoordinateWindow(row, row + 1, carry.coordinate_record,
                                                                 carry.position))
                    row += 1
                if row < at + len(ids):
                    coordinate_windows.append(CoordinateWindow(row, at + len(ids), span.coordinates, span.position))
            at += len(ids)
        segments = mtp_stage(buffers, [(st, ids, parts) for st, ids, parts, *_ in plans],
                             indexer_coordinates=tuple(coordinate_windows))
        for (_st, begin, end), (_owner, _ids, _parts, values, *_rest) in zip(segments, plans):
            if values is None:
                from .forward import _embed

                _embed(weights, buffers.ids[begin:end], 1, buffers.mtp_e[begin:end])
            else:
                buffers.mtp_e[begin:end].copy_(values, non_blocking=True)
    return segments, plans, results


def _require_media(state):
    if getattr(state, "media_inputs_enabled", False) is not True:
        raise ValueError("prepared media inputs require the Engine's admitted media capability/resource envelope")


def bind_context(state, profile_identity, media_identity):
    _require_media(state)
    if not _digest(profile_identity) or media_identity is not None and not _digest(media_identity):
        raise ValueError("media input context needs explicit numerical-profile and content identities")
    if profile_identity != state.coordinate_profile_identity:
        raise ValueError("media context differs from the Engine's persistent numerical/profile binding")
    previous = state.media_profile_identity
    if previous is not None and previous != profile_identity:
        raise ValueError("media input context cannot replace a live model's numerical profile")
    if (state.media_identity != media_identity and (
            state.pos or state.mtp_len or state.mtp_boundary is not None or state.media_input_span is not None)):
        raise ValueError("media input identity cannot change after target consumption")
    state.media_profile_identity, state.media_identity = profile_identity, media_identity


def attach_span(state, span, *, profile_identity, media_identity):
    _require_media(state)
    if not isinstance(span, ReadyInputSpan) or span.closed or span.poisoned:
        raise ValueError("a live producer-owned merged input span is required")
    if state.media_input_span is not None:
        raise ValueError("the prior merged input span must retire before buffer reuse")
    if (span.start != state.pos or span.generation != state.slot_generation
            or span.profile_identity != profile_identity or span.media_identity != media_identity
            or span.values.shape[1] != state.media_hidden_size or span.values.device != state.pos_dev.device
            or state.execution_dtype != torch.float16 or not -(1 << 31) <= span.rope_delta < (1 << 31)):
        raise ValueError("merged input span differs from the admitted State/profile/generation/position contract")
    if state.media_target_position is None:
        raise ValueError("media input coordinates require prepared current target positional operands")
    span.position_proof.validate(span.positions, state.media_target_position, profile_identity=profile_identity,
        generation=state.slot_generation, media_identity=media_identity, row_start=span.start,
        row_count=len(span.token_ids))
    bind_context(state, profile_identity, media_identity)
    state.media_input_span = span  # ownership transfers only after the metadata admission above
    try:
        state.set_rope_delta(span.rope_delta)
    except BaseException as error:
        _failed_state(state, error)
        raise


def _failed_state(state, error):
    state._history_error = state._kv_error = error
    context = getattr(state, "input_preparation_context", None)
    if context is not None:
        try:
            context.fail(error)
        except BaseException as notification:
            raise_failures(error, (notification,))


def retire_span(state, *, cancelled):
    span = state.media_input_span
    if span is None:
        return
    try:
        span.retire(cancelled=cancelled)
    except BaseException as error:
        _failed_state(state, error)
        raise
    state.media_input_span = None


def _complete(state, lease):
    try:
        lease.complete()
    except BaseException as error:
        _failed_state(state, error)
        raise


@contextmanager
def borrow_span(state, role):
    _require_media(state)
    span = state.media_input_span
    if span is None:
        raise ValueError("merged input staging requires its admitted ready span")
    lease = span.borrow(role, profile_identity=state.media_profile_identity, generation=state.slot_generation,
                        media_identity=state.media_identity)
    try:
        yield lease
    except BaseException as error:
        rollback(span, error, lambda: _complete(state, lease))
    else:
        _complete(state, lease)


@contextmanager
def target_inputs(weights, buffers, windows, segments):
    """Stage actual merged target rows under the shared cohort preparation owner."""
    with ExitStack() as consumers:
        context = getattr(buffers, "input_preparation_context", None)

        def operation():
            return _target_inputs(weights, buffers, windows, segments, consumers)

        if context is None:
            if getattr(weights, "positions", None) is not None and getattr(buffers, "compact_indexer_stage", None) is not None:
                raise RuntimeError("bound target inputs require their installed cohort preparation owner")
            inputs = operation()
        else:
            inputs = context.prepare("target-merged-inputs", operation)
        yield inputs


def _target_inputs(weights, buffers, windows, segments, consumers):
    if not any(getattr(st, "media_input_span", None) is not None for st, _ids in windows):
        if any(getattr(st, "media_profile_identity", None) is not None for st, _ids in windows):
            raise ValueError("media prefill requires its actual ready merged inputs")
        return None
    if (not windows or len(windows) != len(segments)
            or segments[-1][2] > buffers.rows
            or sum(len(ids) for _st, ids in windows) != segments[-1][2]):
        raise ValueError("merged target staging must cover the complete admitted frame")
    from .compact_indexer import CoordinateWindow, issue_text_indexer_coordinates, stage_target_indexer_frame
    from .forward import _embed

    total = segments[-1][2]
    coordinate_windows = []
    for (st, ids), (owner, begin, end) in zip(windows, segments):
        if owner is not st or end - begin != len(ids):
            raise ValueError("merged target staging lost its State/token window binding")
        if st.media_input_span is None:
            if st.media_profile_identity is not None:
                raise ValueError("media target rows cannot fall back to placeholder embeddings")
            _embed(weights, buffers.ids[begin:end], 1, buffers.mixed[begin:end])
            coordinates = issue_text_indexer_coordinates(st, buffers.target_indexer_position,
                st.pos, len(ids), max_rows=buffers.rows)
            coordinate_windows.append(CoordinateWindow(begin, end, coordinates, buffers.target_indexer_position))
        else:
            lease = consumers.enter_context(borrow_span(st, "target"))
            buffers.mixed[begin:end].copy_(lease.rows(st.pos, ids), non_blocking=True)
            span = st.media_input_span
            coordinate_windows.append(CoordinateWindow(begin, end, span.coordinates, span.position))
    stage_target_indexer_frame(buffers, segments, tuple(coordinate_windows))
    return buffers.mixed[:total]

def capture_boundary(state, position, hidden, coordinates, *, source_position=None):
    """Capture h_(position-1), never an output-mixed or HC-mean substitute."""
    if type(position) is not int or not 0 <= position <= state.limit:
        raise ValueError("MTP boundary exceeds the admitted logical row range")
    if position == 0:
        return None
    _require_media(state)
    if (state.mtp_hidden_width == 0 or not state.media_mtp_enabled or state.media_profile_identity is None
            or hidden.shape != (1, state.mtp_hidden_width) or hidden.dtype != state.execution_dtype
            or hidden.device != state.pos_dev.device or type(coordinates) is not IssuedCoordinates
            or source_position is None or state.media_mtp_position is None):
        raise ValueError("MTP boundary needs its complete HC row and separate hidden/cache-row coordinates")
    coordinates.proof.validate(coordinates.tensor, source_position, profile_identity=state.media_profile_identity,
        generation=state.slot_generation, media_identity=state.media_identity,
        row_start=coordinates.proof.row_start, row_count=coordinates.proof.row_count)
    copied = copy_coordinates(coordinates, source_position=source_position, position=state.media_mtp_position,
                              row_start=position - 1, row_count=1, dtype=torch.int64)
    return MTPBoundary(position - 1, hidden.clone(), copied, state.media_mtp_position, state.slot_generation)


def snapshot_state(state, *, position=None, mtp_len=None, boundary=..., take_boundary=False):
    if state.media_profile_identity is None:
        if state.mtp_boundary is not None or state.media_identity is not None:
            raise ValueError("media state lost its numerical profile identity")
        return {}
    _require_media(state)
    position = state.pos if position is None else position
    mtp_len = state.mtp_len - state.mtp_drafted if mtp_len is None else mtp_len
    boundary = state.mtp_boundary if boundary is ... else boundary
    if boundary is not None and (not isinstance(boundary, MTPBoundary) or boundary.generation != state.slot_generation):
        raise ValueError("MTP boundary belongs to another slot generation")
    if take_boundary and boundary is not None and boundary is state.mtp_boundary:
        raise ValueError("a live State boundary cannot transfer into its independent snapshot")
    # A cut producer already made this one owned row. Its immutable record may
    # share the snapshot's lifetime; the current State's live boundary never
    # aliases a protected snapshot. This avoids a second protected HC copy.
    coordinate_copy = None
    if boundary is not None:
        boundary.coordinate_record.proof.validate(boundary.coordinates, state.media_mtp_position,
            profile_identity=state.media_profile_identity, generation=state.slot_generation,
            media_identity=state.media_identity, row_start=boundary.row, row_count=1)
        coordinate_copy = (boundary.coordinate_record if take_boundary else copy_coordinates(
            boundary.coordinate_record, source_position=boundary.position, position=state.media_mtp_position,
            row_start=boundary.row, row_count=1, dtype=torch.int64))
    copied = (None if boundary is None else {"row": boundary.row,
        "hidden": boundary.hidden if take_boundary else boundary.hidden.clone(),
        "coordinates": coordinate_copy})
    payload = {"schema": "qwen4exp-media-state-v1", "profile": state.media_profile_identity,
               "media": state.media_identity, "generation": state.slot_generation,
               "rope_delta": state.rope_delta, "mtp_enabled": state.media_mtp_enabled, "mtp_boundary": copied}
    validate_snapshot(state, payload, position, mtp_len)
    return {"media_state": payload}


def validate_snapshot(state, payload, position, mtp_len):
    """Pure metadata/shape admission before any State writes or device copies."""
    if payload is None:
        if getattr(state, "media_profile_identity", None) is not None:
            raise ValueError("media State restoration requires its explicit identity and boundary metadata")
        return
    _require_media(state)
    if (type(payload) is not dict or set(payload) != {
            "schema", "profile", "media", "generation", "rope_delta", "mtp_enabled", "mtp_boundary"}
            or payload["schema"] != "qwen4exp-media-state-v1"
            or getattr(state, "media_profile_identity", None) is None or payload["profile"] != state.media_profile_identity
            or payload["profile"] != state.coordinate_profile_identity
            or payload["media"] != state.media_identity or type(payload["generation"]) is not int
            or not 0 <= payload["generation"] < (1 << 63) or type(payload["rope_delta"]) is not int
            or not -(1 << 31) <= payload["rope_delta"] < (1 << 31)
            or type(payload["mtp_enabled"]) is not bool or payload["mtp_enabled"] != state.media_mtp_enabled):
        raise ValueError("media snapshot differs from the current numerical/input identity or positional schema")
    boundary = payload["mtp_boundary"]
    if not state.mtp_hidden_width or not state.media_mtp_enabled:
        if boundary is not None or mtp_len:
            raise ValueError("target-only media State cannot restore MTP conditioning")
        return
    if position == 0 or mtp_len == position:
        if boundary is not None or position == 0 and mtp_len:
            raise ValueError("a resolved or zero-row MTP state has no pending HC boundary")
        return
    if (mtp_len != position - 1 or type(boundary) is not dict or set(boundary) != {"row", "hidden", "coordinates"}
            or type(boundary["row"]) is not int or boundary["row"] != position - 1):
        raise ValueError("media MTP snapshot lost its own valid frontier and deferred boundary")
    hidden, coordinates = boundary["hidden"], boundary["coordinates"]
    if (not isinstance(hidden, torch.Tensor) or hidden.shape != (1, state.mtp_hidden_width)
            or hidden.dtype != state.execution_dtype or hidden.device != state.pos_dev.device
            or type(coordinates) is not IssuedCoordinates or coordinates.tensor.shape != (1, 3)
            or coordinates.tensor.dtype != torch.int64 or coordinates.tensor.device != hidden.device
            or state.media_mtp_position is None):
        raise ValueError("media MTP boundary snapshot has incompatible HC/coordinate storage")
    coordinates.proof.validate(coordinates.tensor, state.media_mtp_position,
        profile_identity=state.media_profile_identity, generation=payload["generation"],
        media_identity=state.media_identity, row_start=boundary["row"], row_count=1)


def restore_state(state, payload):
    if payload is None:
        return
    boundary = payload["mtp_boundary"]
    if boundary is None:
        state.mtp_boundary = None
    else:
        copied = restore_coordinates(boundary["coordinates"], source_position=state.media_mtp_position,
            position=state.media_mtp_position, profile_identity=state.media_profile_identity,
            media_identity=state.media_identity, generation=state.slot_generation,
            row_start=boundary["row"], row_count=1, dtype=torch.int64)
        state.mtp_boundary = MTPBoundary(boundary["row"], boundary["hidden"].clone(), copied,
                                         state.media_mtp_position, state.slot_generation)
    state.set_rope_delta(payload["rope_delta"])

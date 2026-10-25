"""Issue coordinate bounds from actual host rows and their controlled transfers.

No public operation accepts claimed extrema. The producer owns immutable input
rows; this adapter validates them before narrowing, creates the device tensor,
and binds its proof to that exact tensor/storage. Derived copies retain scalar
bounds and context, never a parent span or a full host-coordinate archive.
All operations belong to the request coordinator and its ordered device stream.
"""

from dataclasses import dataclass
import os
import weakref

import numpy as np
import torch

from ..position_profile import MAX_POSITION, PositionOperands, PositionRole

_ISSUER = object()  # Immutable process-local provenance, not serialized authority.
_MAX_MUTATION_VERSION = (1 << 32) - 1  # Pinned Torch c10::VariableVersion uses uint32_t.


def _digest(value):
    return type(value) is str and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _position(position):
    if (type(position) is not PositionOperands or type(position.domain) is not PositionRole
            or not _digest(position.identity) or type(position.epoch) is not int or position.epoch < 0):
        raise ValueError("coordinates require resolved positional operands and their domain identity")
    return position.domain


def _context(profile_identity, generation, media_identity):
    if (not _digest(profile_identity) or type(generation) is not int or not 0 <= generation < 1 << 63
            or media_identity is not None and not _digest(media_identity)):
        raise ValueError("coordinate request/profile/media generation is invalid")


def _signature(tensor, *, track_mutation=True):
    storage = tensor.untyped_storage()
    base = (tensor.data_ptr(), storage.data_ptr(), storage.nbytes(), tuple(tensor.shape), tuple(tensor.stride()),
            tensor.storage_offset(), tensor.dtype, tensor.device)
    if not track_mutation:
        return base
    try:
        version = tensor._version
    except RuntimeError as error:
        raise ValueError("issued coordinate storage must track Tensor mutations") from error
    return (*base, version)


def _proof(proof):
    if (type(proof) is not CoordinateProof or getattr(proof, "_issuer", None) is not _ISSUER
            or proof._pid != os.getpid()):
        raise ValueError("coordinates require a current-process issued proof")


class CoordinateProof:
    """Opaque issued bounds; a receipt dictionary cannot construct a proof."""
    __slots__ = ("_issuer", "_pid", "_tensor", "_signature", "profile_identity", "generation", "media_identity",
                 "position_identity", "position_epoch", "role", "row_start", "row_count", "minimum", "maximum",
                 "_owner")

    def __init__(self, *args, **kwargs):
        raise TypeError("coordinate proofs are issued from actual rows, not caller-supplied bounds")

    def __setattr__(self, name, value):
        raise TypeError("issued coordinate proofs are immutable")

    def validate(self, tensor, position, *, profile_identity, generation, media_identity, row_start, row_count,
                 owner=None):
        # PID/provenance checks precede tensor/device access, including in a fork.
        _proof(self)
        if self._owner is None:
            if owner is not None:
                raise ValueError("coordinate proof is not bound to this reusable frame")
        elif self._owner() is None or self._owner() is not owner:
            raise ValueError("coordinate proof belongs to a retired or different reusable frame")
        _context(profile_identity, generation, media_identity)
        domain = _position(position)
        if (self.profile_identity != profile_identity or self.generation != generation
                or self.media_identity != media_identity or self.position_identity != position.identity
                or self.position_epoch != position.epoch or self.role != domain.name
                or self.row_start != row_start or self.row_count != row_count):
            raise ValueError("coordinate proof belongs to another row/role/profile/media generation")
        if self._tensor() is not tensor or _signature(tensor) != self._signature:
            raise ValueError("issued coordinate tensor/storage changed or retired")
        domain.validate_span(row_start, row_count, coordinate_min=self.minimum, coordinate_max=self.maximum)


@dataclass(frozen=True)
class IssuedCoordinates:
    tensor: torch.Tensor
    proof: CoordinateProof


def _issued(tensor, position, *, profile_identity, generation, media_identity, row_start, minimum, maximum,
            owner=None):
    proof = object.__new__(CoordinateProof)
    fields = {"_issuer": _ISSUER, "_pid": os.getpid(), "_tensor": weakref.ref(tensor),
              "_signature": _signature(tensor), "profile_identity": profile_identity, "generation": generation,
              "media_identity": media_identity, "position_identity": position.identity, "position_epoch": position.epoch,
              "role": position.domain.name, "row_start": row_start, "row_count": tensor.shape[0],
              "minimum": minimum, "maximum": maximum, "_owner": None if owner is None else weakref.ref(owner)}
    for name, value in fields.items():
        object.__setattr__(proof, name, value)
    return IssuedCoordinates(tensor, proof)


def issue_coordinates(host_positions, *, position, profile_identity, generation, media_identity,
                      row_start, max_rows, device):
    """Validate actual CPU THW rows before int32 conversion and device transfer."""
    domain = _position(position)
    _context(profile_identity, generation, media_identity)
    if type(max_rows) is not int or max_rows <= 0:
        raise ValueError("coordinate preparation requires an explicit admitted row bound")
    if isinstance(host_positions, torch.Tensor):
        if host_positions.device.type != "cpu" or host_positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("coordinate issuance requires actual CPU integer rows")
        raw = host_positions.numpy()
    elif type(host_positions) is np.ndarray:
        raw = host_positions
    else:
        raise ValueError("coordinate issuance requires a CPU integer matrix")
    if raw.ndim != 2 or raw.shape[1] != 3 or raw.shape[0] > max_rows or raw.dtype.kind not in ("i", "u"):
        raise ValueError("coordinate rows exceed the admitted integer THW layout")
    # Copy the producer-owned bounded rows before validation/transfer; callers
    # must not mutate their source concurrently with this operation.
    owned = raw.copy(order="C")
    minimum = tuple(int(v) for v in owned.min(axis=0)) if len(owned) else None
    maximum = tuple(int(v) for v in owned.max(axis=0)) if len(owned) else None
    domain.validate_span(row_start, len(owned), coordinate_min=minimum, coordinate_max=maximum)
    if len(owned) and (min(minimum) < 0 or max(maximum) >= MAX_POSITION):
        raise ValueError("coordinates exceed the signed-int31 device position representation")
    device = torch.device(device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("coordinate transfer requires a CPU reference or CUDA target")
    # Normal tensors retain mutation versions even when the surrounding model
    # runs under inference_mode. Coordinate proofs require that counter.
    with torch.inference_mode(False):
        tensor = torch.tensor(owned.astype(np.int32), dtype=torch.int32, device=device)
    return _issued(tensor, position, profile_identity=profile_identity, generation=generation,
                   media_identity=media_identity, row_start=row_start, minimum=minimum, maximum=maximum)


@dataclass(frozen=True)
class CoordinateTransfer:
    source: IssuedCoordinates
    source_position: PositionOperands
    position: PositionOperands
    row_start: int
    row_count: int
    destination: torch.Tensor
    profile_identity: str
    generation: int
    media_identity: str | None
    dtype: object = torch.int64
    source_owner: object = None


@dataclass(frozen=True)
class _Copy:
    source: IssuedCoordinates
    selected: torch.Tensor
    position: PositionOperands
    row_start: int
    row_count: int
    dtype: object
    destination: torch.Tensor | None
    owner: object
    generation: int


def _prepare_copy(source, *, source_position, position, row_start, row_count, dtype,
                  destination, source_owner, owner, generation=None, expected_context=None):
    """Copy actual issued rows and mint proof for the final copied tensor.

    Supplied destination views require the caller's new frame owner, retained
    through its readers; prior frame authorization must retire before copying.
    On failure the caller contains uncertain writes and retains source/destination
    owners until its coordinated fence. This operation precedes persistent writes on their
    stream; reuse requires retirement of prior readers. Bounds may conservatively
    cover the source span; no GPU scalar readback or parent tensor retention.
    """
    if type(source) is not IssuedCoordinates or type(source.proof) is not CoordinateProof:
        raise ValueError("coordinate copies require actual issued source rows")
    proof = source.proof
    _proof(proof)
    profile, source_generation, media = ((proof.profile_identity, proof.generation, proof.media_identity)
                                         if expected_context is None else expected_context)
    proof.validate(source.tensor, source_position, profile_identity=profile,
                   generation=source_generation, media_identity=media,
                   row_start=proof.row_start, row_count=proof.row_count, owner=source_owner)
    if (type(row_start) is not int or type(row_count) is not int or row_count < 0
            or row_start < proof.row_start or row_count > proof.row_start + proof.row_count - row_start):
        raise ValueError("coordinate copy rows are outside the issued logical span")
    domain = _position(position)
    if position.identity != proof.position_identity or position.epoch != proof.position_epoch:
        raise ValueError("coordinate copies cannot cross positional profile identities/epochs")
    domain.validate_span(row_start, row_count, coordinate_min=proof.minimum, coordinate_max=proof.maximum)
    if dtype not in (torch.int32, torch.int64):
        raise ValueError("coordinate copies retain integer device positions")
    if destination is not None and owner is None:
        raise ValueError("reusable coordinate destinations require their current frame owner")
    if owner is not None:
        weakref.ref(owner)  # Validate lifetime support before any copy/allocation.
    lo = row_start - proof.row_start
    selected = source.tensor[lo:lo + row_count]
    if destination is not None:
        if (destination.shape != (row_count, 3) or destination.dtype != dtype
                or destination.device != source.tensor.device or not destination.is_contiguous()):
            raise ValueError("coordinate destination must match the final row/device/integer layout")
        if _signature(destination)[-1] >= _MAX_MUTATION_VERSION:
            raise ValueError("coordinate mutation version exhausted; retire and replace its storage")
        if row_count:
            source_storage, destination_storage = source.tensor.untyped_storage(), destination.untyped_storage()
            left, right = source_storage.data_ptr(), destination_storage.data_ptr()
            if (left < right + destination_storage.nbytes()
                    and right < left + source_storage.nbytes()):
                raise ValueError("coordinate copy destination must not overlap immutable source rows")
    return _Copy(source, selected, position, row_start, row_count, dtype, destination, owner,
                 proof.generation if generation is None else generation)


def _perform_copy(plan):
    with torch.inference_mode(False):
        if plan.destination is None:
            return plan.selected.to(dtype=plan.dtype, copy=True)
        plan.destination.copy_(plan.selected)
        return plan.destination


def _publish_copy(plan, result):
    proof = plan.source.proof
    return _issued(result, plan.position, profile_identity=proof.profile_identity, generation=plan.generation,
                   media_identity=proof.media_identity, row_start=plan.row_start,
                   minimum=proof.minimum if plan.row_count else None, maximum=proof.maximum if plan.row_count else None,
                   owner=plan.owner)


def _copy_coordinates(source, **options):
    plan = _prepare_copy(source, **options)
    return _publish_copy(plan, _perform_copy(plan))


def copy_coordinate_batch(transfers, *, owner, max_rows):
    """Validate all segments, copy all, then bind final shared-storage versions.

    The caller revokes old frame authorization before entering and publishes its
    candidate frame only after this returns. Partial failure returns no proofs;
    the caller retains all issued readers/writers through its coordinated fence.
    """
    if (type(transfers) is not tuple or type(max_rows) is not int or max_rows <= 0
            or len(transfers) > max_rows or owner is None):
        raise ValueError("packed coordinate copies require a bounded tuple and current frame owner")
    weakref.ref(owner)
    plans = []
    for transfer in transfers:
        if type(transfer) is not CoordinateTransfer or transfer.destination is None:
            raise ValueError("packed coordinates require complete typed segment transfers")
        plans.append(_prepare_copy(transfer.source, source_position=transfer.source_position,
            position=transfer.position, row_start=transfer.row_start, row_count=transfer.row_count,
            dtype=transfer.dtype, destination=transfer.destination, source_owner=transfer.source_owner, owner=owner,
            expected_context=(transfer.profile_identity, transfer.generation, transfer.media_identity)))
    if sum(plan.row_count for plan in plans) > max_rows:
        raise ValueError("packed coordinate rows exceed their admitted frame")
    # Distinct destinations may share one version counter. Allow enough counter
    # space for every write before starting; never permit uint32 version ABA.
    if any(_signature(plan.destination)[-1] > _MAX_MUTATION_VERSION - len(plans) for plan in plans):
        raise ValueError("coordinate mutation version exhausted; retire and replace its storage")

    def interval(tensor):
        start = tensor.data_ptr()
        return (str(tensor.device), start, start + tensor.numel() * tensor.element_size())

    destinations = sorted(interval(plan.destination) for plan in plans if plan.row_count)
    for left, right in zip(destinations, destinations[1:]):
        if left[0] == right[0] and left[2] > right[1]:
            raise ValueError("packed coordinate destination segments overlap")
    # Reject writes into ANY immutable source, including another segment's.
    # Even disjoint source/destination views cannot share mutation counters:
    # coordinate sources remain valid immutable bindings after this operation.
    sources = sorted({(str(plan.source.tensor.device), plan.source.tensor.untyped_storage().data_ptr(),
                       plan.source.tensor.untyped_storage().data_ptr() + plan.source.tensor.untyped_storage().nbytes())
                      for plan in plans if plan.source.tensor.numel()})
    at = 0
    for device, start, end in destinations:
        while at < len(sources) and (sources[at][0] < device or sources[at][0] == device and sources[at][2] <= start):
            at += 1
        if at < len(sources) and sources[at][0] == device and sources[at][1] < end:
            raise ValueError("packed coordinate destinations overlap immutable source rows")
    results = [_perform_copy(plan) for plan in plans]
    return tuple(_publish_copy(plan, result) for plan, result in zip(plans, results))


def copy_coordinates(source, *, source_position, position, row_start, row_count, dtype=torch.int64,
                     destination=None, source_owner=None, owner=None):
    """Copy current issued data without changing its request generation."""
    return _copy_coordinates(source, source_position=source_position, position=position, row_start=row_start,
                             row_count=row_count, dtype=dtype, destination=destination,
                             source_owner=source_owner, owner=owner)


def restore_coordinates(source, *, source_position, position, profile_identity, media_identity, generation,
                        row_start, row_count, dtype=torch.int64, destination=None, source_owner=None, owner=None):
    """Copy a validated snapshot into its caller's current admitted generation.

    State/prefix/media admission validation and old-reader retirement must finish
    first. Current expected identities come from admission, never the snapshot.
    This operation cannot convert profiles/transcripts or relabel an old tensor;
    it checks unchanged context/position identity and performs the actual copy.
    """
    if type(source) is not IssuedCoordinates:
        raise ValueError("coordinate restore requires actual issued snapshot rows")
    _proof(source.proof)
    _context(profile_identity, generation, media_identity)
    if source.proof.profile_identity != profile_identity or source.proof.media_identity != media_identity:
        raise ValueError("coordinate restore differs from current admitted profile/media identity")
    return _copy_coordinates(source, source_position=source_position, position=position, row_start=row_start,
                             row_count=row_count, dtype=dtype, destination=destination,
                             source_owner=source_owner, owner=owner, generation=generation)

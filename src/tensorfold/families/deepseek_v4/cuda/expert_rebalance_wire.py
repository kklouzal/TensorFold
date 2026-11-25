"""Bounded DeepSeek expert-rebalance peer acknowledgments for the TP8 control pipe."""
from dataclasses import dataclass
from enum import IntEnum
from hashlib import sha256
import struct

from .expert_transfer import Placement, _validate

WORLD, LAYERS, EXPERTS = 8, 40, 384
_MAX_EPOCH = (1 << 63) - 1  # Runtime control headers use signed int64 fields.
_DIGEST_BYTES = 32
_MAGIC = b"ER01"
_HEADER = struct.Struct("<4sBBBBqqq32s32s")
_LAYER = struct.Struct("<9H")
_IDS = struct.Struct("<" + "H" * EXPERTS)
_CARRIER = {16: 66, 17: 74, 29: 56}
_LAYOUT_BYTES = LAYERS * _LAYER.size

CarrierLayout = tuple[tuple[int, int, int, tuple[tuple[int, int], ...]], ...]


def _check_etp(etp):
    if type(etp) is not int or etp not in (1, 2, 4):
        raise ValueError("expert-rebalance receipts require TP8 ETP1, ETP2, or ETP4")


class Phase(IntEnum):
    READY = 1
    APPLIED = 2
    ABORTED = 3


class Refusal(IntEnum):
    NONE = 0
    PREFLIGHT = 1
    STALE = 2
    LAYOUT = 3
    CAPACITY = 4


class PeerRefusal(RuntimeError):
    """A matching, stamped READY refusal before any rank applies the plan."""

    def __init__(self, refusals):
        self.refusals = tuple(refusals)
        detail = ", ".join(f"rank {rank} {reason.name}" for rank, reason in self.refusals)
        super().__init__(f"expert rebalance refused before mutation: {detail}")


@dataclass(frozen=True, slots=True)
class Transaction:
    """One attempt with a fresh process-local nonce, including after an abort."""

    etp: int
    current_epoch: int
    target_epoch: int
    nonce: int
    current_digest: bytes
    target_digest: bytes

    def __post_init__(self):
        _check_etp(self.etp)
        if (type(self.current_epoch) is not int or type(self.target_epoch) is not int
                or not 0 <= self.current_epoch < self.target_epoch <= _MAX_EPOCH
                or self.target_epoch != self.current_epoch + 1):
            raise ValueError("invalid expert-rebalance epochs")
        if type(self.nonce) is not int or not 1 <= self.nonce <= _MAX_EPOCH:
            raise ValueError("invalid expert-rebalance attempt nonce")
        if any(type(value) is not bytes or len(value) != _DIGEST_BYTES
               for value in (self.current_digest, self.target_digest)):
            raise ValueError("expert-rebalance placement digests must be 32 bytes")


@dataclass(frozen=True, slots=True)
class Receipt:
    phase: Phase
    rank: int
    transaction: Transaction
    refusal: Refusal = Refusal.NONE
    carrier_layout: CarrierLayout | None = None


def placement_digest(placement: Placement, *, etp: int) -> bytes:
    """Bind the exact layer and physical slot order to a degree-specific digest."""
    _check_etp(etp)
    _validate(placement, label="digest", layers=LAYERS, experts=EXPERTS)
    digest = sha256(b"tensorfold.deepseek-v41.expert-placement.v1\0" + bytes((etp,)))
    for layer in placement:
        digest.update(_IDS.pack(*layer))
    return digest.digest()


def placement_from_ids(ids) -> Placement:
    """Recover the fixed-size physical plan from the engine's int32 control payload."""
    if type(ids) not in (tuple, list) or len(ids) != LAYERS * EXPERTS:
        raise ValueError("expert-rebalance payload must have exactly 40 * 384 slot IDs")
    placement = tuple(tuple(ids[start:start + EXPERTS]) for start in range(0, len(ids), EXPERTS))
    _validate(placement, label="received", layers=LAYERS, experts=EXPERTS)
    return placement


def _validate_receipt(receipt):
    phase, rank, transaction, refusal, layout = (
        receipt.phase, receipt.rank, receipt.transaction, receipt.refusal, receipt.carrier_layout)
    if type(phase) is not Phase or type(refusal) is not Refusal or not isinstance(transaction, Transaction):
        raise ValueError("invalid expert-rebalance receipt phase, refusal or transaction type")
    if type(rank) is not int or not 0 <= rank < WORLD:
        raise ValueError("invalid expert-rebalance peer rank")
    if phase != Phase.READY or refusal != Refusal.NONE:
        if layout is not None or (phase != Phase.READY and refusal != Refusal.NONE):
            raise ValueError("expert-rebalance status does not admit a carrier layout or refusal")
        return
    if type(layout) is not tuple or len(layout) != LAYERS:
        raise ValueError("accepted expert-rebalance readiness requires all 40 carrier layouts")
    width = 2304 // transaction.etp
    left = (rank % transaction.etp) * width % 256
    for layer in layout:
        if type(layer) is not tuple or len(layer) != 4:
            raise ValueError("invalid expert-rebalance carrier layout")
        gate_kind, down_kind, offset, shapes = layer
        if (any(type(value) is not int for value in (gate_kind, down_kind, offset))
                or gate_kind not in _CARRIER or down_kind not in _CARRIER or offset != left):
            raise ValueError("expert-rebalance carrier codec or TP cut differs")
        if (type(shapes) is not tuple or len(shapes) != 3 or any(type(shape) is not tuple or len(shape) != 2
                or any(type(value) is not int or not 0 < value <= 65535 for value in shape) for shape in shapes)):
            raise ValueError("invalid expert-rebalance carrier dimensions")
        gate, up, down = shapes
        if gate != up or gate != (width, 20 * _CARRIER[gate_kind]) or down[0] != 5120:
            raise ValueError("expert-rebalance carrier shapes differ from fixed model geometry")
        # Row stride may differ by raw/compact layout. Ready peers must
        # compare the exact loaded dimension before borrowing either buffer.


def encode_receipt(receipt: Receipt) -> bytes:
    _validate_receipt(receipt)
    transaction = receipt.transaction
    header = _HEADER.pack(_MAGIC, receipt.phase, receipt.rank, transaction.etp, receipt.refusal,
                          transaction.current_epoch, transaction.target_epoch, transaction.nonce,
                          transaction.current_digest, transaction.target_digest)
    if receipt.carrier_layout is None:
        return header
    return header + b"".join(
        _LAYER.pack(gate_kind, down_kind, offset, *gate, *up, *down)
        for gate_kind, down_kind, offset, (gate, up, down) in receipt.carrier_layout
    )


def decode_receipt(payload: bytes) -> Receipt:
    if type(payload) is not bytes or len(payload) not in (_HEADER.size, _HEADER.size + _LAYOUT_BYTES):
        raise ValueError("invalid expert-rebalance receipt size")
    magic, phase, rank, etp, refusal, current, target, nonce, source, destination = _HEADER.unpack_from(payload)
    if magic != _MAGIC:
        raise ValueError("unsupported expert-rebalance receipt version")
    layout = None
    if len(payload) > _HEADER.size:
        layout = tuple((gate_kind, down_kind, offset, ((gm, gb), (um, ub), (dm, db)))
                       for gate_kind, down_kind, offset, gm, gb, um, ub, dm, db
                       in _LAYER.iter_unpack(payload[_HEADER.size:]))
    receipt = Receipt(Phase(phase), rank, Transaction(etp, current, target, nonce, source, destination),
                      Refusal(refusal), layout)
    _validate_receipt(receipt)
    return receipt


def validate_receipts(payloads, *, phase: Phase, transaction: Transaction) -> tuple[Receipt, ...]:
    """Check one rank-zero local receipt and seven peer frames before allowing progress."""
    if type(phase) is not Phase or not isinstance(transaction, Transaction):
        raise ValueError("invalid expected expert-rebalance phase or transaction")
    if type(payloads) not in (tuple, list) or len(payloads) != WORLD:
        raise ValueError("expert rebalance requires one receipt from every TP8 rank")
    receipts = tuple(decode_receipt(payload) for payload in payloads)
    for rank, receipt in enumerate(receipts):
        if receipt.rank != rank:
            raise ValueError(f"expert-rebalance receipt rank {receipt.rank} does not match channel {rank}")
        if receipt.phase != phase or receipt.transaction != transaction:
            raise ValueError(f"stale or mismatched expert-rebalance receipt from rank {receipt.rank}")
    if phase == Phase.READY:
        layouts, refused = {}, []
        for receipt in receipts:
            if receipt.refusal != Refusal.NONE:
                refused.append((receipt.rank, receipt.refusal))
                continue
            lane = receipt.rank % transaction.etp
            if layouts.setdefault(lane, receipt.carrier_layout) != receipt.carrier_layout:
                raise ValueError(f"expert-rebalance carrier layout mismatch for rank {receipt.rank} lane {lane}")
        if refused:
            raise PeerRefusal(refused)
    return receipts

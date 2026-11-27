"""Fixed binary receipt checks without CUDA or a live rank transport."""
from dataclasses import replace

import pytest

from tensorfold.families.deepseek_v4.cuda.expert_rebalance_wire import (
    PeerRefusal, Phase, Receipt, Refusal, Transaction, decode_receipt, encode_receipt,
    placement_digest, placement_from_ids, validate_receipts,
)


def _txn(etp=4):
    return Transaction(etp, 3, 4, 9, b"a" * 32, b"b" * 32)


def _ready(rank, transaction):
    width = 2304 // transaction.etp
    left = rank % transaction.etp * width % 256
    row_bytes = -(-(width + left) // 256) * 56
    layer = 16, 29, left, ((width, 1320), (width, 1320), (5120, row_bytes))
    return Receipt(Phase.READY, rank, transaction, carrier_layout=(layer,) * 40)


@pytest.mark.parametrize("etp", (1, 2, 4))
def test_ready_each_lane_and_terminal_records(etp):
    txn = _txn(etp)
    ready = tuple(_ready(rank, txn) for rank in range(8))
    frames = [encode_receipt(receipt) for receipt in ready]
    assert all(len(frame) == 816 for frame in frames)
    assert validate_receipts(frames, phase=Phase.READY, transaction=txn) == ready
    with pytest.raises(ValueError, match="does not match channel"):
        validate_receipts(frames[::-1], phase=Phase.READY, transaction=txn)
    for phase in (Phase.APPLIED, Phase.ABORTED):
        terminal = [encode_receipt(Receipt(phase, rank, txn)) for rank in range(8)]
        assert all(len(frame) == 96 for frame in terminal)
        assert [r.rank for r in validate_receipts(terminal, phase=phase, transaction=txn)] == list(range(8))


def test_preflight_refusal_rejects_and_conflicting_accepted_layout_is_invalid():
    txn = _txn()
    ready = [_ready(rank, txn) for rank in range(8)]
    ready[0] = Receipt(Phase.READY, 0, txn, Refusal.CAPACITY)
    frames = [encode_receipt(receipt) for receipt in ready]
    with pytest.raises(PeerRefusal) as caught:
        validate_receipts(frames, phase=Phase.READY, transaction=txn)
    assert caught.value.refusals == ((0, Refusal.CAPACITY),)
    layer = ready[5].carrier_layout[0]
    other = (*layer[:3], (*layer[3][:2], (5120, 132)))
    frames[5] = encode_receipt(replace(ready[5], carrier_layout=(other,) * 40))
    with pytest.raises(ValueError, match="carrier layout mismatch"):
        validate_receipts(frames, phase=Phase.READY, transaction=txn)


def test_stale_attempt_or_phase_never_counts_as_ready():
    txn = _txn()
    frames = [encode_receipt(_ready(rank, txn)) for rank in range(8)]
    with pytest.raises(ValueError, match="stale or mismatched"):
        validate_receipts(frames, phase=Phase.READY, transaction=replace(txn, nonce=10))
    with pytest.raises(ValueError, match="stale or mismatched"):
        validate_receipts(frames, phase=Phase.APPLIED, transaction=txn)
    with pytest.raises(ValueError, match="every TP8 rank"):
        validate_receipts(frames[:-1], phase=Phase.READY, transaction=txn)
    with pytest.raises(ValueError, match="does not match channel"):
        validate_receipts([*frames[:-1], frames[0]], phase=Phase.READY, transaction=txn)


def test_malformed_bytes_and_phase_payloads_fail():
    txn = _txn()
    base = encode_receipt(_ready(0, txn))
    assert decode_receipt(base) == _ready(0, txn)
    for blob in (base[:-1], base + b"\0", b"ER02" + base[4:], base[:4] + b"\xff" + base[5:]):
        with pytest.raises(ValueError):
            decode_receipt(blob)
    with pytest.raises(ValueError, match="requires all 40"):
        decode_receipt(base[:96])
    with pytest.raises(ValueError, match="does not admit"):
        encode_receipt(Receipt(Phase.APPLIED, 1, txn, Refusal.PREFLIGHT))
    with pytest.raises(ValueError, match="does not admit"):
        encode_receipt(replace(_ready(0, txn), phase=Phase.ABORTED))
    with pytest.raises(ValueError, match="epochs"):
        replace(txn, target_epoch=5)
    with pytest.raises(ValueError, match="nonce"):
        replace(txn, nonce=0)


def test_digest_preserves_degree_and_slot_order_and_reads_fixed_payload():
    line = tuple(range(384))
    original = (line,) * 40
    revised = ((line[1], line[0], *line[2:]), *original[1:])
    flat = [expert for row in original for expert in row]
    assert placement_from_ids(flat) == original
    assert placement_digest(original, etp=4) != placement_digest(revised, etp=4)
    assert placement_digest(original, etp=4) != placement_digest(original, etp=2)
    with pytest.raises(ValueError, match="exactly 40"):
        placement_from_ids(flat[:-1])
    flat[0] = flat[1]
    with pytest.raises(ValueError, match="each of"):
        placement_from_ids(flat)

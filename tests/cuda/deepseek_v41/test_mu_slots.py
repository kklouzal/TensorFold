"""Pure MU bookkeeping (ported from the friend overlay's tests/test_ds41mu.py: test_slot_table_lifecycle,
test_split_requests, the PackedLayout half of test_packed_layout_and_admission; admission policy is not
vendored)."""

import pytest
import torch

from tensorfold.families.deepseek_v4.cuda.kernels import mu_packed as pk
from tensorfold.families.deepseek_v4.cuda.kernels import mu_slots as m


def test_slot_table_lifecycle():
    t = m.SlotTable(3)
    assert t.assign("a") == (0, True)
    assert t.assign("b") == (1, True)
    assert t.assign("a") == (0, False)
    assert t.release("a") == 0
    assert t.assign("c") == (0, True)  # lowest free slot is reused
    assert t.assign("d") == (2, True)
    with pytest.raises(RuntimeError, match="slot table full"):
        t.assign("e")
    assert t.release("missing") is None
    assert t.scratch == 3 and len(t) == 3


def test_split_requests():
    qsl = torch.tensor([0, 1, 2, 40, 41, 41], dtype=torch.int32)
    assert m.split_requests(qsl, 5, decode_max=8) == [(2, 2, 40)]
    assert m.split_requests(qsl, 5, decode_max=0) == [(0, 0, 1), (1, 1, 2), (2, 2, 40), (3, 40, 41)]


def test_packed_layout():
    stage0 = pk.PackedLayout({8: 2, 2: 2, 14: 2}, 64)
    assert stage0.bytes_per_token == 534 and stage0.page_bytes == 64 * 534
    assert stage0.regions[2] == (2, 32, 0, 32 * 288)
    assert stage0.regions[8][2] == 32 * 356 and stage0.regions[14][2] == 2 * 32 * 356
    stage1 = pk.PackedLayout({20: 1}, 64)
    assert stage1.bytes_per_token == 356 and stage1.regions[20] == (1, 64, 0, 64 * 288)
    with pytest.raises(ValueError):
        pk.PackedLayout({2: 2}, 15)
    with pytest.raises(ValueError):
        pk.PackedLayout({}, 16)

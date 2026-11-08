"""CPU ownership/layout controls
no SM70 arithmetic claim."""
import pytest
import torch

from tensorfold.cuda.sm70.reference_owner_tiles import ReferenceTiles, WORKSPACE_BYTES


def aligned():
    storage = torch.empty(WORKSPACE_BYTES // 4 + 128, dtype=torch.float32)
    start = ((-storage.data_ptr()) % 512) // 4
    return storage[start:start + WORKSPACE_BYTES // 4]


def test_paid_slab_views_are_disjoint_and_no_additional_storage_is_allocated():
    slab = aligned()
    p = ReferenceTiles(slab)
    assert p.mask.data_ptr() == slab.data_ptr()
    assert p.indices.data_ptr() == slab.data_ptr() + 128
    assert p.half.data_ptr() == slab.data_ptr() + 1536
    assert p.record()["borrowed_device_bytes"] == WORKSPACE_BYTES
    assert p.record()["additional_device_bytes"] == 0
    assert WORKSPACE_BYTES <= 8 * 1 * 12 * 256 * 4
    p._lease.acquire()
    with pytest.raises(RuntimeError, match="borrowed"):
        p.begin_retirement()
    p._lease.release()
    p.begin_retirement()
    p.release_views()
    assert p.workspace is p.mask is p.indices is p.half is None


@pytest.mark.parametrize("slot,generation,layer", [(True, 0, 0), (-1, 0, 0), (0, 2**63, 0), (0, 0, 2**31)])
def test_context_refuses_invalid_epoch_controls_before_device_use(slot, generation, layer):
    p = ReferenceTiles(aligned())
    with pytest.raises(ValueError, match="bounded integers"):
        p.context(status=None, generations=None, slot_generation=slot, owner_generation=generation, layer=layer)


def test_insufficient_or_unaligned_slab_refuses_before_view_construction():
    with pytest.raises(ValueError, match="aligned paid"):
        ReferenceTiles(torch.empty(1))
    slab = aligned()
    with pytest.raises(ValueError, match="aligned paid"):
        ReferenceTiles(slab[1:])

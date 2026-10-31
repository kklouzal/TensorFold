"""Large address metadata with sparse CPU mappings; no GPU capacity claim."""

import math
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda.history_owner import OwnedKVCache, OwnerBuffers
from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement
from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache
from tensorfold.families.qwen4_exp.cuda.state import State
from tensorfold.families.qwen4_exp.kv_formats import get_pair
from tests.test_native_owner_adapter import _weights


@pytest.mark.parametrize("include_mtp,caches", [(False, 12), (True, 13)])
@pytest.mark.parametrize("sessions", [20, 21, 37])
def test_indexer_owner_uses_the_exact_fixed_inventory_phase(include_mtp, caches, sessions):
    placement = HistoryPlacement(8, 2, 12)
    # Independent stream oracle: enumerate each cache once in slot-major order.
    rank = 0
    expected = [[] for _ in range(8)]
    for slot in range(sessions):
        for ordinal in range(caches):
            assert placement.indexer_owner(slot, ordinal, include_mtp=include_mtp) == rank
            expected[rank].append((slot, ordinal))
            rank = (rank + 1) % 8
    actual = [[(slot, ordinal) for slot in range(sessions) for ordinal in range(caches)
               if placement.indexer_owner(slot, ordinal, include_mtp=include_mtp) == owner] for owner in range(8)]
    assert actual == expected
    # Reusing a physical slot cannot depend on request ID or enumeration order.
    assert [placement.indexer_owner(sessions - 1, ordinal, include_mtp=include_mtp) for ordinal in range(caches)] == [
        owner for ordinal in range(caches) for owner in range(8) if (sessions - 1, ordinal) in expected[owner]]
    if not include_mtp:
        with pytest.raises(ValueError, match="target-only"):
            placement.indexer_owner(0, 12, include_mtp=False)


@pytest.mark.parametrize("boundary", [1 << 31, 1 << 32])
def test_shared_page_offsets_and_native_descriptor_addresses_cross_byte_boundaries(tmp_path, monkeypatch, boundary):
    # Sparse zero-backed mappings reserve virtual address space, not a complete
    # GPU arena or resident CPU payload. Touch only two rows. All production
    # cache/page/descriptor construction and append methods run unchanged.
    pair, dim, page = get_pair("int8"), 256, 128
    first = (boundary // (dim * page) + 1) * page
    physical = first + page
    files = []

    def sparse(shape, dtype):
        path = tmp_path / (str(len(files)) + ".sparse")
        elements = math.prod(shape)
        size = elements * torch.empty((), dtype=dtype).element_size()
        with path.open("wb") as stream:
            stream.truncate(size)
        files.append(path)
        return torch.from_file(str(path), shared=False, size=elements, dtype=dtype).view(shape)

    def mapped_side(self, fmt, device):
        assert str(device) == "cpu"
        return (sparse((self.capacity, 1, dim * fmt.bits // 8), getattr(torch, fmt.payload_dtype)),
                sparse((self.capacity, 1, dim // fmt.group), getattr(torch, fmt.scale_dtype)))

    with monkeypatch.context() as allocation:
        allocation.setattr(KVCache, "_side", mapped_side)
        arena = KVCache(physical, 1, dim, "cpu", pair=pair)
    weights = _weights(0)
    state = State(weights, page, 4, "int8", slot=0)
    cache = OwnedKVCache(page, 1, dim, "cpu", pair=pair, placement=state.history_placement,
                         slot=0, layer=0, rank=0, arena=arena, arena_offset=first)
    state.kc[0] = cache
    codes = torch.arange(2 * dim).reshape(2, 1, dim).to(torch.int8)
    scales = torch.ones((2, 1, dim // 32), dtype=torch.float16)
    cache.append(0, codes, -codes, scales, scales)
    assert cache.page_offsets.dtype == torch.int64 and cache.page_offsets.tolist() == [first]
    assert torch.equal(cache.k[first:first + 2], codes)
    positions, owned = cache.local_indices(torch.tensor([0, 1], dtype=torch.int64))
    assert positions.dtype == torch.int64 and positions.tolist() == [first, first + 1] and bool(owned.all())
    assert first * dim > boundary
    assert cache.k[first].data_ptr() - cache.k.data_ptr() == first * dim
    workspace = OwnerBuffers(weights, 4, NS(idw=2052))
    workspace.prepare_native((state,))
    workspace.stage_native_descriptors([(state, 0, 1)], NS(index=0), 0, 1)
    words = workspace.descriptors[0].tolist()
    assert words[0] == cache.k.data_ptr() and words[4] == cache.page_offsets.data_ptr()
    assert words[12] == physical
    assert words[0] + first * dim == cache.k[first].data_ptr()
    workspace.close()

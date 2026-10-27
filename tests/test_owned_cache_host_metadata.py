"""Pay exact added cache slots, then credit only real materialized ownership."""
import sys

import pytest

from tensorfold.cuda import capacity
from tensorfold.families.qwen4_exp.cuda.history_placement import HistoryPlacement, cache_owner_host_bytes
from test_adaptive_session_capacity import admission as _admission

admission = _admission


@pytest.mark.parametrize("mtp", [False, True])
def test_added_slot_count_matches_actual_target_mtp_clone_and_resize_objects(mtp):
    pytest.importorskip("torch")
    from tensorfold.families.qwen4_exp.cuda.history_owner import OwnedKVCache, allocate_target_caches
    from tensorfold.families.qwen4_exp.cuda.kvcache import KVCache
    from tensorfold.families.qwen4_exp.kv_formats import get_pair

    class PreviousLayout(KVCache):
        __slots__ = tuple(name for name in OwnedKVCache.__slots__ if name != "_empty_physical")

    old_size = sys.getsizeof(object.__new__(PreviousLayout))
    placement, caches = HistoryPlacement(8, 2, 12), []
    for slot in range(2):
        caches.extend(allocate_target_caches(257, 256, "cpu", pair=get_pair("int8"),
                                            placement=placement, slot=slot, rank=0))
        if mtp:
            caches.append(OwnedKVCache(257, 1, 256, "cpu", pair=get_pair("int8"),
                                       placement=placement, slot=slot, layer=12, rank=0))
    actual = sum(sys.getsizeof(cache) - old_size for cache in caches)
    assert actual == cache_owner_host_bytes(len(caches), world=1)
    assert actual == 8 * 2 * (12 + int(mtp))
    # Standalone A/B callers also count every extra object, even though each
    # independent non-arena copy has a false empty proof.
    extra = [caches[0].clone(), caches[1].resized(128, 0)]
    assert all(not cache._empty_physical for cache in extra)
    assert sum(sys.getsizeof(cache) - old_size for cache in extra) == cache_owner_host_bytes(2, world=1)


def test_preallocated_pool_funding_and_postpool_credit_are_separate_actual_phases(admission, monkeypatch):
    def charge(slots):
        return cache_owner_host_bytes(slots * 13, world=8)
    plan = capacity.admit(**admission, host_resident=11, host_workspace_bytes=17,
                          session_host_workspace=charge, session_ceiling=3)
    paid = charge(plan["session_capacity"]["resident_capacity"])
    assert plan["host_steady_commitment_bytes"] == 11 + 17 + paid
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 100000, "MemAvailable": 100000})
    monkeypatch.setattr(capacity, "host_stream_reserve_bytes", lambda total: 0)
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda **kwargs: None)
    before = capacity.admit_host_commitment(plan["host_steady_commitment_bytes"])
    after = capacity.admit_host_commitment(plan["host_steady_commitment_bytes"], materialized_owned_bytes=paid)
    assert before["remaining_commitment_bytes"] == after["remaining_commitment_bytes"] + paid
    assert after["remaining_commitment_bytes"] == 28


@pytest.mark.parametrize("count,world", [(-1, 8), (True, 8), (1, 0), (1, True)])
def test_cache_owner_slot_count_rejects_invalid_domains(count, world):
    with pytest.raises(ValueError):
        cache_owner_host_bytes(count, world=world)

"""NUMA grant and collective failure controls without CUDA or actual placement."""

from types import SimpleNamespace as NS

import pytest

from tensorfold.cuda import capacity, numa_memory
from tensorfold.families.qwen4_exp.cuda import indexer_admission as admission
from tensorfold.families.qwen4_exp.cuda.indexer_history import HistoryPlan


GIB = 1 << 30


def snapshot(node=0):
    return {"node": node, "counters": {"MemTotal": 200 * GIB, "MemFree": 20 * GIB,
            "FilePages": 120 * GIB, "Active(file)": 70 * GIB, "Inactive(file)": 40 * GIB,
            "Dirty": 3 * GIB, "Writeback": 2 * GIB,
            "Shmem": 50 * GIB, "Mlocked": 7 * GIB, "Unevictable": 10 * GIB}, "snapshot_is_reservation": False}


def test_clean_node_credit_preserves_global_reserve_without_borrowing_dirty_or_nonfile_pages():
    values = {0: snapshot(), 1: snapshot(1)}
    # 20free +min(120file,70active+40inactive) -3dirty -2writeback =125GiB.
    # Anonymous/shmem/unevictable are never added to file-LRU credit.
    assert admission.node_allowances(values, 2 * GIB) == {0: 124 * GIB, 1: 124 * GIB}
    assert admission.node_allowances(values, 3) == {0: 125 * GIB - 2, 1: 125 * GIB - 1}
    assert admission.node_allowances(values, 0) == {0: 125 * GIB, 1: 125 * GIB}


def test_empty_clean_lru_credit_is_conservative_not_a_reclaim_promise():
    value = snapshot()
    value["counters"].update({"Active(file)": 0, "Inactive(file)": 0})
    assert admission.node_allowances({0: value}, 2 * GIB) == {0: 18 * GIB}


@pytest.mark.parametrize("field,value", [("MemFree", True), ("Dirty", -1), ("FilePages", 201 * GIB), ("MemTotal", 0)])
def test_invalid_node_byte_counters_refuse(field, value):
    data = snapshot()
    data["counters"][field] = value
    with pytest.raises(ValueError, match="NUMA"):
        admission.node_allowances({0: data}, 2 * GIB)


@pytest.fixture
def placement(monkeypatch):
    target = NS(node=0, sockets=(0,), page_size=4096,
                gpu_uuid="GPU-00000000-0000-0000-0000-000000000001", receipt={"node": 0})
    target.probe = lambda: {"verified": True}
    monkeypatch.setattr(numa_memory, "configure_gpu_numa", lambda *a, **kw: target)
    monkeypatch.setattr(numa_memory, "node_memory_snapshot", snapshot)
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 400 * GIB, "MemAvailable": 300 * GIB})
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setattr(admission, "_same_host_words", lambda: [11, 22, 33, 44])
    created = []

    def allocator(target, **kwargs):
        value = NS(target=target, **kwargs)
        created.append(value)
        return value

    monkeypatch.setattr(numa_memory, "HostAllocator", allocator)
    return target, created


def prepare(gather, **kwargs):
    plan = HistoryPlan(3, 128, 4, 16, 4)
    allocator, receipt = admission.prepare(plan, limit=128, maximum_owners=6, torch=object(),
                                         device="unused", rank=0, world=2, gather=gather, **kwargs)
    return plan, allocator, receipt


def test_rank_packets_agree_before_allocator_creation_and_keep_conservative_node_observations(placement):
    target, created = placement
    calls = []

    def gather(packet):
        assert created == []
        calls.append(list(packet))
        peer = list(packet)
        if len(calls) == 1:
            peer[1:3] = [1, 1]
            peer[9] += 1  # distinct exact physical GPU UUID
            peer[-1] = 1
        else:
            peer[2] -= GIB
        return [packet, peer]

    plan, allocator, receipt = prepare(gather)
    assert len(calls) == 2 and allocator.target is target
    assert allocator.byte_limit == 6 * plan.host_bytes(128) and allocator.pinned_limit == 6 * plan.pinned_bytes
    assert receipt["observed_node_allowances"] == {0: 123 * GIB, 1: 124 * GIB}
    assert receipt["node_grants"]["admitted"] and not receipt["snapshot_is_reservation"]
    assert [row["rank"] for row in receipt["ranks"]] == [0, 1]


@pytest.mark.parametrize("failure", ["local-probe", "peer-probe", "duplicate-gpu", "different-host", "local-counters", "peer-counters", "node-exhaustion"])
def test_discovery_counter_and_budget_failures_never_allocate_or_strand_a_peer(monkeypatch, placement, failure):
    target, created = placement
    calls = []
    if failure == "local-probe":
        def probe():
            raise PermissionError("controlled mbind EPERM")
        target.probe = probe
    if failure == "local-counters":
        monkeypatch.setattr(numa_memory, "node_memory_snapshot", lambda node: (_ for _ in ()).throw(OSError("counter unavailable")))
    if failure == "node-exhaustion":
        def no_room(node):
            value = snapshot(node)
            value["counters"].update({"MemFree": 0, "FilePages": 0, "Active(file)": 0, "Inactive(file)": 0})
            return value
        monkeypatch.setattr(numa_memory, "node_memory_snapshot", no_room)

    def gather(packet):
        calls.append(list(packet))
        peer = list(packet)
        if len(calls) == 1:
            peer[0], peer[-1] = (0 if failure == "peer-probe" else 1), 1
            if failure != "duplicate-gpu":
                peer[9] += 1
            if failure == "different-host":
                peer[10] += 1
        elif failure == "peer-counters":
            peer[0] = 0
        return [packet, peer]

    with pytest.raises((ValueError, RuntimeError, PermissionError, OSError, MemoryError)):
        prepare(gather)
    assert not created
    assert len(calls) == (2 if failure in ("local-counters", "peer-counters", "node-exhaustion") else 1)
    if failure == "local-probe":
        assert calls[0][0] == 0  # failure is transmitted before the local exception propagates
    if failure == "local-counters":
        assert calls[1][0] == 0


def test_existing_host_reserve_authority_is_unchanged_for_numa_users(monkeypatch):
    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    assert capacity.host_stream_reserve_bytes(400 * GIB) == 2 * GIB
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "6")
    assert capacity.host_stream_reserve_bytes(400 * GIB) == 6 * GIB

"""Already-owned prep is deducted once from absolute caps, never twice from free."""

from types import SimpleNamespace as NS

import pytest


@pytest.fixture
def planner(monkeypatch):
    from tensorfold.cuda import build, capacity

    gib = capacity.GIB
    monkeypatch.setattr(build, "refuse_old_gpu", lambda floor: None)
    monkeypatch.setattr(capacity, "floor", lambda model: (7, 0))
    monkeypatch.setattr(capacity, "config", lambda model: {"max_position_embeddings": 8})
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    monkeypatch.setenv("TENSORFOLD_CUDA_MEMORY_RESERVE_GIB", "4")

    def run(free, *, preallocated=0):
        runtime = NS(cuda=NS(mem_get_info=lambda: (free * gib, 40 * gib),
                             get_device_properties=lambda index: NS(is_integrated=False)))
        return capacity.admit("unused", 8, True, runtime, capacity.Geometry(lambda slots: slots * gib // 4, 0),
                              lambda name, info: (1, 0),
                              weight_estimator=lambda path: (capacity.Weights(gib, 0), 0),
                              preallocated_device_bytes=preallocated)

    return capacity, run


@pytest.mark.parametrize("limit,free,expected", [(30, 20, 16), (20, 30, 17)])
def test_preallocated_bytes_only_reduce_the_absolute_cap_when_it_binds(planner, monkeypatch, limit, free, expected):
    capacity, run = planner
    monkeypatch.setenv(capacity.LIMIT_ENV, str(limit))
    receipt = run(free, preallocated=3 * capacity.GIB)
    assert receipt["budget_bytes"] == expected * capacity.GIB
    assert receipt["device_preallocation"]["absolute_limit_bytes"] == limit * capacity.GIB
    assert receipt["device_preallocation"]["remaining_absolute_limit_bytes"] == (limit - 3) * capacity.GIB
    assert receipt["device_preallocation"]["included_in_physical_free_baseline"]


def test_no_absolute_cap_keeps_the_physical_baseline_and_zero_path_is_unchanged(planner, monkeypatch):
    capacity, run = planner
    monkeypatch.delenv(capacity.LIMIT_ENV, raising=False)
    plain, prepared = run(20), run(20, preallocated=3 * capacity.GIB)
    assert plain["budget_bytes"] == prepared["budget_bytes"] == 16 * capacity.GIB
    assert {key: value for key, value in prepared.items() if key != "device_preallocation"} == plain
    assert "device_preallocation" not in plain


@pytest.mark.parametrize("value", [-1, True, 21 * 2**30])
def test_invalid_preallocation_refuses_before_planning(planner, monkeypatch, value):
    capacity, run = planner
    monkeypatch.setenv(capacity.LIMIT_ENV, "20")
    monkeypatch.setattr(capacity, "make_plan", lambda *args, **kwargs: pytest.fail("invalid owners entered planning"))
    with pytest.raises(ValueError, match="preallocated device commitments"):
        run(30, preallocated=value)

"""Serving-only sampling buffers share device/host admission without overlapping loading phases."""

from pathlib import Path

import pytest

from tensorfold.cuda import build, capacity


@pytest.fixture
def admission(monkeypatch):
    monkeypatch.setattr(build, "refuse_old_gpu", lambda floor: None)
    monkeypatch.setattr(capacity, "floor", lambda path: 89)
    monkeypatch.setattr(capacity, "config", lambda path: {"max_position_embeddings": 100})
    monkeypatch.setattr(capacity, "available_bytes", lambda torch: 1000)
    monkeypatch.setattr(capacity, "page_room", lambda torch: None)
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda: None)

    def run(*, integrated=False, host_free=1000, **extras):
        monkeypatch.setattr(capacity, "unified", lambda torch: integrated)
        monkeypatch.setattr(capacity, "host_stream_bytes", lambda: host_free)
        return capacity.admit(
            Path("metadata-only"), 10, True, object(), capacity.Geometry(lambda slots: 10 * slots, 0),
            lambda name, info: (0, 0),
            weight_estimator=lambda path: (capacity.Weights(100, 150), 80), **extras)

    return run


def test_discrete_serving_host_scratch_does_not_coexist_with_loading_scratch(admission):
    # Retained host20 + control9 + max(loading80, serving120) =149; adding both phases would reject149.
    receipt = admission(host_free=149, host_resident=20, host_extra_staging=9,
                        host_workspace_bytes=120, device_workspace_bytes=200)
    assert receipt["total_bytes_estimate"] == 400
    assert receipt["startup_peak_bytes_estimate"] == 250
    assert receipt["serving_peak_bytes_estimate"] == 400
    assert receipt["host_workspace_bytes_estimate"] == 120
    assert receipt["device_workspace_bytes_estimate"] == 200
    with pytest.raises(ValueError, match="host weights/staging/workspace"):
        admission(host_free=148, host_resident=20, host_extra_staging=9,
                  host_workspace_bytes=120, device_workspace_bytes=200)


def test_loading_larger_than_runtime_keeps_original_host_peak(admission):
    receipt = admission(host_free=109, host_resident=20, host_extra_staging=9, host_workspace_bytes=70)
    assert receipt["context_window"] == 10
    with pytest.raises(ValueError, match="host weights/staging/workspace"):
        admission(host_free=108, host_resident=20, host_extra_staging=9, host_workspace_bytes=70)


def test_unified_runtime_host_and_device_buffers_coexist_with_cache(admission):
    receipt = admission(integrated=True, host_resident=20, host_extra_staging=9,
                        host_workspace_bytes=120, device_workspace_bytes=200)
    assert receipt["startup_peak_bytes_estimate"] == 279
    assert receipt["serving_peak_bytes_estimate"] == 540
    assert receipt["total_bytes_estimate"] == 540
    assert receipt["cache_workspace_bytes_estimate"] == 420


def test_host_cgroup_rejects_runtime_workspace_even_without_expert_pool(admission, monkeypatch):
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda: 119)
    with pytest.raises(ValueError, match="host weights/staging/workspace"):
        admission(host_workspace_bytes=120)


def test_serving_workspace_refuses_explicit_context_before_allocation(admission):
    with pytest.raises(ValueError, match="context|fit"):
        admission(device_workspace_bytes=1000)


def test_zero_workspace_preserves_existing_receipt(admission):
    assert admission() == admission(device_workspace_bytes=0, host_workspace_bytes=0)
    assert "host_workspace_bytes_estimate" not in admission()


@pytest.mark.parametrize("name", ["device_workspace_bytes", "host_workspace_bytes"])
@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_workspace_byte_contract_rejects_invalid_values(admission, name, value):
    with pytest.raises(ValueError, match="workspace.*nonnegative integers"):
        admission(**{name: value})

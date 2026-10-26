"""Physical remaining-host authority and declared global commitment accounting."""

import pytest

from tensorfold.cuda import capacity

GIB = 1 << 30


@pytest.fixture
def physical(monkeypatch):
    calls = {"memory": 0, "cgroup": 0}

    def memory():
        calls["memory"] += 1
        assert calls["memory"] == 1, "host commitment resampled its physical authority"
        return {"MemTotal": 64 * GIB, "MemAvailable": 10 * GIB}

    def cgroup(*, strict_membership):
        assert strict_membership is True
        calls["cgroup"] += 1
        return 12 * GIB

    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setattr(capacity, "_meminfo", memory)
    monkeypatch.setattr(capacity, "host_cgroup_bytes", cgroup)
    monkeypatch.setattr(capacity, "host_stream_bytes", lambda: pytest.fail("admission must not resample memory"))
    return calls


def test_one_global_sample_credits_only_materialized_ownership_not_physical_free(physical):
    result = capacity.admit_host_commitment(12 * GIB, materialized_owned_bytes=4 * GIB)
    assert physical == {"memory": 1, "cgroup": 1}
    assert result["physical_available_bytes"] == 10 * GIB
    assert result["effective_os_reserve_bytes"] == 2 * GIB
    assert result["usable_remaining_bytes"] == result["remaining_commitment_bytes"] == 8 * GIB
    assert result["total_commitment_bytes"] == 12 * GIB and result["materialized_owned_bytes"] == 4 * GIB
    assert result["observed_at_unix_ns"] > 0
    assert 0 <= result["interval_monotonic_ns"][0] <= result["interval_monotonic_ns"][1]
    assert result["snapshot_is_reservation"] is result["release_qualified"] is False


@pytest.mark.parametrize("cgroup", [0, 7 * GIB])
def test_the_smallest_actual_remaining_authority_refuses_an_unfitted_claim(physical, monkeypatch, cgroup):
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda **_kwargs: cgroup)
    with pytest.raises(ValueError, match="remaining host commitment"):
        capacity.admit_host_commitment(12 * GIB, materialized_owned_bytes=4 * GIB)
    assert physical["memory"] == 1


def test_physical_headroom_still_limits_an_unlimited_cgroup(physical, monkeypatch):
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda **_kwargs: None)
    with pytest.raises(ValueError, match="remaining host commitment"):
        capacity.admit_host_commitment(9 * GIB)


def test_no_exposed_cgroup_controller_uses_physical_authority(physical, monkeypatch):
    monkeypatch.setattr(capacity, "host_cgroup_bytes", lambda **_kwargs: None)
    result = capacity.admit_host_commitment(8 * GIB)
    assert result["cgroup_remaining_bytes"] is None and result["usable_remaining_bytes"] == 8 * GIB


def test_existing_explicit_host_reserve_is_consumed_without_another_knob(physical, monkeypatch):
    monkeypatch.setenv("TENSORFOLD_MEMORY_RESERVE_GIB", "6")
    result = capacity.admit_host_commitment(4 * GIB)
    assert result["effective_os_reserve_bytes"] == 6 * GIB and result["usable_remaining_bytes"] == 4 * GIB


@pytest.mark.parametrize("total,credit", [(-1, 0), (True, 0), (1.5, 0), ("8", 0),
                                         (8, -1), (8, True), (8, 1.0), (8, 9)])
def test_invalid_commitment_domains_refuse_before_physical_io(monkeypatch, total, credit):
    monkeypatch.setattr(capacity, "_meminfo", lambda: pytest.fail("invalid claim sampled memory"))
    with pytest.raises(ValueError, match="host commitment"):
        capacity.admit_host_commitment(total, materialized_owned_bytes=credit)


def test_missing_physical_authority_is_not_fabricated_from_a_cgroup(physical, monkeypatch):
    monkeypatch.setattr(capacity, "_meminfo", lambda: None)
    with pytest.raises(ValueError, match="actual physical memory availability"):
        capacity.admit_host_commitment(1)
    assert physical["cgroup"] == 0


@pytest.mark.parametrize("memory", [{"MemTotal": 4, "MemAvailable": 5},
                                   {"MemTotal": 0, "MemAvailable": 0}])
def test_invalid_physical_counter_contract_is_not_clamped(physical, monkeypatch, memory):
    monkeypatch.setattr(capacity, "_meminfo", lambda: memory)
    with pytest.raises(ValueError, match="observation is inconsistent"):
        capacity.admit_host_commitment(1)


def test_materialized_credit_cannot_exceed_actual_physical_memory(physical):
    with pytest.raises(ValueError, match="observation is inconsistent"):
        capacity.admit_host_commitment(66 * GIB, materialized_owned_bytes=65 * GIB)


def test_unreadable_cgroup_authority_preserves_the_original_failure(physical, monkeypatch):
    primary = ValueError("exposed cgroup memory limit is unreadable")

    def unavailable(**_kwargs):
        raise primary

    monkeypatch.setattr(capacity, "host_cgroup_bytes", unavailable)
    with pytest.raises(ValueError) as caught:
        capacity.admit_host_commitment(1)
    assert caught.value is primary


def test_resource_exhaustion_during_observation_is_not_an_unknown_budget(physical, monkeypatch):
    primary = MemoryError("physical counter allocation failed")

    def exhausted():
        raise primary

    monkeypatch.setattr(capacity, "_meminfo", exhausted)
    with pytest.raises(MemoryError) as caught:
        capacity.admit_host_commitment(1)
    assert caught.value is primary and physical["cgroup"] == 0


def test_unreadable_actual_membership_is_not_misreported_as_no_cgroup(monkeypatch):
    from pathlib import Path

    original = Path.read_text
    primary = PermissionError("membership denied")

    def read(path, *args, **kwargs):
        if path == Path("/proc/self/cgroup"):
            raise primary
        return original(path, *args, **kwargs)

    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": 10 * GIB})
    monkeypatch.setattr(Path, "read_text", read)
    with pytest.raises(ValueError, match="cannot read cgroup-v2 process membership") as caught:
        capacity.admit_host_commitment(1)
    assert caught.value.__cause__ is primary
    assert capacity.host_cgroup_bytes() is None  # Unrelated legacy API behavior is preserved.


def test_missing_actual_membership_interface_keeps_physical_authority(monkeypatch):
    from pathlib import Path

    original = Path.read_text

    def read(path, *args, **kwargs):
        if path == Path("/proc/self/cgroup"):
            raise FileNotFoundError(path)
        return original(path, *args, **kwargs)

    monkeypatch.delenv("TENSORFOLD_MEMORY_RESERVE_GIB", raising=False)
    monkeypatch.setattr(capacity, "_meminfo", lambda: {"MemTotal": 64 * GIB, "MemAvailable": 10 * GIB})
    monkeypatch.setattr(Path, "read_text", read)
    result = capacity.admit_host_commitment(1)
    assert result["cgroup_remaining_bytes"] is None and result["usable_remaining_bytes"] == 8 * GIB

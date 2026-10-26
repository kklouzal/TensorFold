"""Explicit promise bounds; no GPU/host capacity or performance qualification."""

from dataclasses import replace
import threading

import pytest

from tensorfold.families.qwen4_exp.cuda.retention_policy import RetentionPolicy, parse_retention_policy


def ram():
    return RetentionPolicy("ram", 3, 60., 1 << 20, 1024, 16384, 0)


def test_unset_is_request_scoped_and_gpu_has_no_unused_transfer_configuration():
    assert parse_retention_policy() is None
    policy = parse_retention_policy(retention_tier="gpu", retained_sessions=2, retention_lifetime_s=30.)
    assert policy == RetentionPolicy("gpu", 2, 30.)
    with pytest.raises(ValueError, match="unused"):
        replace(policy, ram_bytes=0)


@pytest.mark.parametrize("changes", [{"retained_sessions": 1}, {"retention_tier": "ram"},
                                    {"retention_tier": "gpu", "retained_sessions": 1},
                                    {"retention_tier": "storage", "retained_sessions": 1, "retention_lifetime_s": 2}])
def test_partial_or_unsupported_policy_never_invents_a_budget(changes):
    with pytest.raises(ValueError):
        parse_retention_policy(**changes)


@pytest.mark.parametrize("changes", [{"max_sessions": True}, {"max_sessions": 0}, {"lifetime_s": True},
                                    {"lifetime_s": 0.}, {"lifetime_s": float("inf")},
                                    {"lifetime_s": threading.TIMEOUT_MAX * 2}, {"lifetime_s": 10**1000},
                                    {"chunk_bytes": 0}, {"ram_bytes": None}, {"progress_bytes": 1024},
                                    {"overlap_bytes": -1}])
def test_invalid_ranges_are_boundary_errors(changes):
    with pytest.raises(ValueError):
        replace(ram(), **changes)


def test_actual_world_controls_the_global_progress_promise():
    policy = ram()
    policy.validate_world(8)
    with pytest.raises(ValueError, match="one pinned"):
        replace(policy, progress_bytes=8192).validate_world(8)


def test_record_roundtrip_preserves_units_and_absolute_lifetime():
    policy = ram()
    assert RetentionPolicy.from_record(policy.record()) == policy
    for change in ({"lifetime": "inactivity"}, {"overlap_scope": "global"}, {"unexpected": 1}):
        with pytest.raises(ValueError, match="unsupported"):
            RetentionPolicy.from_record({**policy.record(), **change})

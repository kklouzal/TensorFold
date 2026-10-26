"""Existing admission inputs expose separate loading and selected steady costs."""

import pytest

from tensorfold.cuda import capacity
from test_adaptive_session_capacity import admission as _admission

admission = _admission


def test_selected_session_workspace_and_existing_header_are_not_multiplied_again(admission):
    result = capacity.admit(**admission, host_resident=11, host_extra_staging=13,
                            host_workspace_bytes=117, session_host_workspace=lambda count: 19 * count,
                            session_ceiling=3)
    assert result["session_capacity"]["resident_capacity"] == 3
    assert result["host_steady_workspace_bytes_estimate"] == result["host_workspace_bytes_estimate"] == 174
    assert result["host_resident_bytes_estimate"] == 11 and result["host_extra_staging_bytes_estimate"] == 13
    assert result["host_loading_staging_bytes_estimate"] == 50
    assert result["host_loading_commitment_bytes"] == 74 and result["host_steady_commitment_bytes"] == 198
    # A caller's known100-byte retention header is a subset of workspace.
    # Excluding it leaves exactly ordinary resident/extra/workspace/session cost.
    assert result["host_steady_commitment_bytes"] - 100 == 11 + 13 + 17 + 3 * 19


def test_loading_peak_does_not_become_a_steady_retention_promise(admission):
    admission["weight_estimator"] = lambda _path: (capacity.Weights(100, 5), 250)
    result = capacity.admit(**admission, host_resident=11, host_extra_staging=13,
                            host_workspace_bytes=17, session_host_workspace=lambda count: 19 * count,
                            session_ceiling=3)
    assert result["host_loading_commitment_bytes"] == 274
    assert result["host_steady_commitment_bytes"] == 98  # No sum of disjoint load250 and steady74.


@pytest.mark.parametrize("room,kept,mapped", [(None, None, 0), (300, False, 0), (10_000, True, 700)])
def test_mapped_host_charge_follows_existing_residency_authority_once(admission, monkeypatch, room, kept, mapped):
    admission.pop("session_geometry")
    admission["weight_estimator"] = lambda _path: (capacity.Weights(100, 50, 700), 50)
    monkeypatch.setattr(capacity, "page_room", lambda _torch: room)
    result = capacity.admit(**admission, host_resident=11, host_extra_staging=13, host_workspace_bytes=17,
                            world=2, gather=lambda row: [row, list(row)])
    assert result["mapped_tables_resident"] is kept
    assert result["host_shared_mapped_commitment_bytes"] == mapped
    assert result["host_steady_commitment_bytes"] == 41 + mapped  # Shared once, not multiplied by world2.


def test_shared_gguf_ple_already_in_host_resident_is_not_charged_as_mapped_again(admission):
    result = capacity.admit(**admission, host_resident=700, host_workspace_bytes=17, session_ceiling=1)
    assert result["mapped_table_bytes"] == result["host_shared_mapped_commitment_bytes"] == 0
    assert result["host_steady_commitment_bytes"] == 717

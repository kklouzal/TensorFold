"""Global native/Python commitments retain their source category and ownership."""

import json
import resource
import sys
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda.retention_resources import (
    HostBaseline, finalize_retention_resources, native_sync_fixed_payload, native_sync_prefix_payload,
)
from tests.test_retention_resources import resource_decoder


def _catalog(baseline):
    return SimpleNamespace(policy=None, world=2, states=[SimpleNamespace(limit=19, logical_context=16)],
        _manifest_bytes=0, _buffer=None, _cohort=SimpleNamespace(multi=resource_decoder(16)),
        check_health=lambda: None, has_retained=False, prefix_limit=2, minimum_future_bytes=1,
        evidence="CPU source-reference", host_resources=baseline)


def _finalize(catalog, monkeypatch):
    from tensorfold.cuda import capacity
    from test_retention_resources import tokenizer

    monkeypatch.setattr(capacity, "admit_host_commitment", lambda total, **_kw: {"total": total})
    return finalize_retention_resources(catalog, tokenizer=tokenizer(), max_pending_requests=2,
        max_waiting_input_bytes=1000, max_http_connections=3, output_queue_tokens=1)


def test_prefix_values_cover_each_bounded_group_and_native_ownership():
    for world in (1, 2, 8):
        for slots in (1, 2, 7):
            record = native_sync_prefix_payload(world=world, slots=slots)
            raw = json.dumps({"decisions": [[sys.maxsize, False]] * slots, "failed": False},
                             separators=(",", ":")).encode()
            assert record["frame_bytes"] == len(raw)
            peers = world - 1
            assert record["stored_value_bytes"] == (slots + 1) * peers * len(raw)
            assert record["active_native_value_bytes"] == ((3 * peers + 1) * len(raw) if peers else 0)
            assert record["global_native_prefix_payload_bytes"] == sum(record[key] for key in (
                "stored_value_bytes", "active_native_value_bytes", "shared_daemon_handler_delta_bytes"))


def test_all_mode_finalizer_names_and_pays_native_and_python_once(monkeypatch):
    monkeypatch.setattr(resource, "getrlimit", lambda _which: (32, 64))
    record = _finalize(_catalog(HostBaseline(100, 0, native_state_slots=3, native_fd_limit=64)), monkeypatch)
    assert record["native_fixed"]["maximum_state_slots"] == 3
    assert record["native_prefix"]["maximum_state_slots"] == 3
    assert record["copy_enabled"] is False
    assert record["global_unpaid_native_fixed_payload_bytes"] == record["native_fixed"]["global_native_fixed_payload_bytes"]
    assert record["global_unpaid_native_prefix_payload_bytes"] == record["native_prefix"]["global_native_prefix_payload_bytes"]
    rpc = record["tp_rpc_python"]
    continuation = record["continuation"]
    assert record["generation_owner_bound"]["owners"] == 2
    assert record["global_model_cpu_projection_bytes"] == 2 * (
        continuation["project_payload_bytes"] + rpc["per_follower_scope_bytes"])
    assert record["global_rpc_python_working_bytes"] == rpc["global_rpc_python_working_bytes"] > 0
    assert record["global_http_rpc_error_tail_bytes"] == 3 * rpc["per_http_rpc_error_tail_bytes"]
    assert record["global_host_commitment_bytes"] == sum((
        record["ordinary_host_baseline"]["baseline_host_bytes"],
        sum(record["header"][key] for key in ("global_host_backing_bytes", "global_host_progress_bytes",
            "global_host_native_data_bytes", "global_host_native_control_bytes", "global_host_native_rpc_bytes",
            "global_host_preparation_bytes")),
        record["global_waiting_input_bytes"], record["global_future_cpu_payload_bytes"],
        record["global_transport_projection_bytes"], record["global_manifest_projection_bytes"],
        record["global_logical_api_projection_bytes"], record["global_rpc_python_working_bytes"],
        record["global_unpaid_native_fixed_payload_bytes"], record["global_unpaid_native_prefix_payload_bytes"]))


def test_prepaid_header_protocol_obligations_are_not_charged_again(monkeypatch):
    monkeypatch.setattr(resource, "getrlimit", lambda _which: (32, 64))
    fixed = native_sync_fixed_payload(world=2, ram=False, slots=3, fd_limit=64)["global_native_fixed_payload_bytes"]
    prefix = native_sync_prefix_payload(world=2, slots=3)["global_native_prefix_payload_bytes"]
    prepaid = HostBaseline(100 + fixed + prefix, 0, fixed, 3, 64, prefix)
    paid = _finalize(_catalog(prepaid), monkeypatch)
    late = _finalize(_catalog(HostBaseline(100, 0, native_state_slots=3, native_fd_limit=64)), monkeypatch)
    assert paid["global_unpaid_native_fixed_payload_bytes"] == 0
    assert paid["global_unpaid_native_prefix_payload_bytes"] == 0
    assert paid["global_host_commitment_bytes"] == late["global_host_commitment_bytes"]


def test_profile_cannot_credit_a_smaller_protocol_state_or_descriptor_envelope(monkeypatch):
    monkeypatch.setattr(resource, "getrlimit", lambda _which: (32, 64))
    catalog = _catalog(HostBaseline(100, 0, native_state_slots=1, native_fd_limit=64))
    catalog.states.append(SimpleNamespace(limit=19, logical_context=16))
    with pytest.raises(ValueError, match="State highwater"):
        _finalize(catalog, monkeypatch)
    catalog.states.pop()
    catalog.host_resources = HostBaseline(100, 0, native_state_slots=1, native_fd_limit=32)
    with pytest.raises(ValueError, match="descriptor capability"):
        _finalize(catalog, monkeypatch)
    assert not hasattr(catalog, "host_resource_envelope")


def test_single_rank_has_no_native_store_or_rpc_envelope(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda.retention_resources import retention_header_resources

    def forbidden(_which):
        pytest.fail("single-rank Engine has no Store descriptor capability to sample")
    monkeypatch.setattr(resource, "getrlimit", forbidden)
    catalog = _catalog(HostBaseline(100, 0))
    catalog.world = 1
    record = _finalize(catalog, monkeypatch)
    assert retention_header_resources(None, 1).global_host_bytes == 0
    assert record["native_fixed"] is record["native_prefix"] is None
    assert record["global_rpc_python_working_bytes"] == record["global_http_rpc_error_tail_bytes"] == 0
    assert record["global_unpaid_native_fixed_payload_bytes"] == 0
    assert record["global_unpaid_native_prefix_payload_bytes"] == 0


@pytest.mark.parametrize("change", [{"native_state_slots": 0}, {"native_fd_limit": -1},
                                     {"prepaid_native_prefix_payload_bytes": 101}])
def test_host_baseline_rejects_unfunded_or_invalid_native_contract(change):
    with pytest.raises(ValueError):
        HostBaseline(100, 0, **change)

"""Ordinary HTTP owner and C-encoder payload boundaries, without CUDA execution."""
from dataclasses import replace
import json
import sys
from types import SimpleNamespace

import pytest

from tensorfold.families.qwen4_exp.cuda import retention_resources as resources
from tests.test_finite_backend_resources import owners
from tests.test_cuda_retention_resources import selected_tokenizer


def policy():
    # A mathematical object-layout fixture; not selected runtime qualification.
    return resources._JsonPayloadPolicy("payload-fixture", (), sys.getrecursionlimit(), 8192)


def scope(selected, scheduler, limit, **changes):
    args = dict(pending=limit.maximum, waiting=scheduler.max_waiting_input_bytes,
                output=scheduler.output_queue_tokens, logical_api_limits=None)
    args.update(changes)
    return resources._http_fifo_projection_scope(selected, selected_tokenizer(), scheduler, limit, **args)


def test_http_scope_binds_actual_fifo_model_and_unpublished_request_owner():
    selected, scheduler, limit, _ = owners()
    record = scope(selected, scheduler, limit)
    assert record["logical_requests"] == 1
    assert record["input_credit_bytes"] == scheduler.max_waiting_input_bytes
    assert record["fifo_child_schema"] == resources._scheduler_child_schema(scheduler)
    assert scheduler._request_limit is None and not scheduler._submitted and limit.used == 0


@pytest.mark.parametrize("change", ["wrong-model", "wrong-pending", "wrong-credit", "fake-scheduler"])
def test_http_scope_rejects_false_ownership(change):
    selected, scheduler, limit, _ = owners()
    changes = {}
    if change == "wrong-model":
        scheduler.decoder = SimpleNamespace(states=[], w=selected._cohort.multi.w)
    elif change == "wrong-pending":
        changes["pending"] = 2
    elif change == "wrong-credit":
        changes["waiting"] = scheduler.max_waiting_input_bytes + 1
    else:
        scheduler = SimpleNamespace(max_waiting_input_bytes=16384, output_queue_tokens=None)
    with pytest.raises(ValueError, match="actual|differ"):
        scope(selected, scheduler, limit, **changes)


@pytest.mark.parametrize("changed", ["submitted", "used", "retained", "other-limit"])
def test_http_scope_does_not_select_outside_initial_ordinary_region(changed):
    selected, scheduler, limit, _ = owners()
    if changed == "submitted":
        scheduler._submitted = True
    elif changed == "used":
        with limit.request():
            assert scope(selected, scheduler, limit) is None
        return
    elif changed == "retained":
        selected.policy = object()
    else:
        scheduler._request_limit = type(limit)(1)
    assert scope(selected, scheduler, limit) is None


@pytest.mark.parametrize("value", [{}, {"é": "\\\"" * 300}, [[1]] * 100, {1 << 200: -1.25}])
def test_c_encoder_projection_covers_actual_ascii_wire_and_framing(value):
    text = json.dumps(value)
    amount = len(text.encode("ascii"))
    bound = resources._http_json_encoding_payload(amount, policy())
    frame = "data: " + text + "\n\n"
    actual_wire = sys.getsizeof(text) + sys.getsizeof(frame) + sys.getsizeof(frame.encode())
    assert bound["total_bytes"] >= actual_wire
    assert bound["integer_decimal_scratch_bytes"] == 1964
    assert bound["numeric_key_repr_bytes"] == 4342
    assert bound["writer_old_new_bytes"] >= 2 * sys.getsizeof(text)


def test_c_encoder_projection_retains_marker_churn_and_fixed_depth():
    smaller = resources._http_json_encoding_payload(100, policy())
    larger = resources._http_json_encoding_payload(96 << 20, policy())
    assert smaller["c_container_depth_bound"] == 50
    assert larger["c_container_depth_bound"] == 10000
    assert larger["marker_payload_bytes"] > smaller["marker_payload_bytes"]
    assert larger["total_bytes"] == max(larger["construction_peak_bytes"], larger["completed_str_and_bytes_peak"])
    assert resources._http_json_encoding_payload(100, replace(policy(), frame_work=9192))["total_bytes"] == smaller["total_bytes"] + 1000


def test_c_encoder_projection_refuses_changed_dispatch(monkeypatch):
    monkeypatch.setattr(json._default_encoder, "ensure_ascii", False)
    with pytest.raises(ValueError, match="default C ASCII"):
        resources._http_json_encoding_payload(10, policy())


def finalized(selected, scheduler, limit, *, connections=2):
    return resources.finalize_retention_resources(selected, tokenizer=selected_tokenizer(),
        scheduler=scheduler, request_limit=limit, max_pending_requests=limit.maximum,
        max_waiting_input_bytes=scheduler.max_waiting_input_bytes, max_http_connections=connections,
        output_queue_tokens=scheduler.output_queue_tokens, max_http_body_bytes=96 << 20)


def test_actual_http_receipt_pays_completed_handler_after_logical_owner_ends(monkeypatch):
    from tensorfold.cuda import capacity
    selected, scheduler, limit, _ = owners()
    admitted = []
    monkeypatch.setattr(capacity, "admit_host_commitment", lambda total, **kw: admitted.append((total, kw)) or {})
    receipt = finalized(selected, scheduler, limit)
    assert receipt["http_owner_scope"] is not None
    assert receipt["max_http_body_bytes"] == 96 << 20
    assert receipt["generation_owner_bound"]["owners"] == 1
    assert receipt["global_completed_http_projection_bytes"] > 0
    assert receipt["global_http_encoder_payload_bytes"] == 2 * receipt["http_encoder"]["per_handler_peak_bytes"]
    assert receipt["logical_api"]["per_http_old_source_clone_memo_bytes"] > 0
    assert receipt["http_encoder"]["cached_reply"]["encoded_bytes"] == 256 << 20
    assert admitted == [(receipt["global_host_commitment_bytes"],
                         {"materialized_owned_bytes": receipt["materialized_owned_bytes"]})]
    assert scheduler._request_limit is None and not scheduler._submitted
    with limit.request():
        assert receipt["global_completed_http_projection_bytes"] > 0  # independent H, never borrowed P credit


def test_zero_generation_owner_is_not_a_ready_http_profile(monkeypatch):
    from tensorfold.cuda import capacity
    selected, scheduler, limit, _ = owners()
    scheduler.max_waiting_input_bytes = 1
    monkeypatch.setattr(capacity, "admit_host_commitment", lambda *_a, **_kw: pytest.fail("zero owner reached admission"))
    with pytest.raises(ValueError, match="cannot admit one full-context"):
        finalized(selected, scheduler, limit)
    assert not hasattr(selected, "host_resource_envelope")

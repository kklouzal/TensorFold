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


def test_monitor_shell_source_counts_cover_owned_static_fields_and_indexer_cells():
    """Independently census owned schema literals, not observed sample sizes."""
    import ast
    import inspect
    import textwrap
    from tensorfold.server import stats, runtime
    from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController
    from tensorfold.families.qwen4_exp.cuda.execution_observations import ExecutionObservations, _COUNTERS
    from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder
    from tensorfold.families.qwen4_exp.cuda.state import State

    def fields(fn):
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
        count = sum(len(node.keys) for node in ast.walk(tree) if isinstance(node, ast.Dict))
        count += sum(len(node.keywords) for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Attribute) and node.func.attr == "update")
        return count
    fixed = sum(fields(fn) for fn in (stats.snapshot, stats._describe, runtime._snapshot_current,
        AdaptiveController._record, ExecutionObservations.record, MultiDecoder.usage, MultiDecoder.history_record))
    assert 2 * fixed + 2 * len(_COUNTERS) <= 1024
    assert 2 * fields(State.history_record) + 8 <= 256  # Includes maximum8 holder ranks.
    from tensorfold.families.qwen4_exp.cuda import compact_indexer
    record = ast.parse(textwrap.dedent(inspect.getsource(compact_indexer.CompactIndexer.record)))
    per_layer = sum(1 if isinstance(node, ast.ListComp) else 2 for node in ast.walk(record)
                    if isinstance(node, (ast.ListComp, ast.DictComp)))
    assert per_layer <= 16


def finalized(selected, scheduler, limit, *, connections=2, segments=None):
    return resources.finalize_retention_resources(selected, tokenizer=selected_tokenizer(),
        scheduler=scheduler, request_limit=limit, max_pending_requests=limit.maximum,
        max_waiting_input_bytes=scheduler.max_waiting_input_bytes, max_http_connections=connections,
        output_queue_tokens=scheduler.output_queue_tokens, max_http_body_bytes=96 << 20,
        segment_encoder=segments)


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


def test_c27_recorded_geometry_pays_all_small_replies_and_monitor_together(monkeypatch):
    """Source arithmetic against recorded inputs, NOT a new physical grant.

    Geometry/baseline: actual-auto-capacity-v1/stats-before-request.json.
    Token bound: actual-cpu-v5/host-counterfactual-result.json (0acd1c81).
    The tiny native tokenizer is only the adapter eligibility fixture; its
    presentation estimate is replaced with the captured SC117 bound below.
    No weights, GPU State allocations or model requests are created.
    """
    import hashlib
    from tensorfold.cuda import capacity
    from tensorfold.cuda.scheduler import Scheduler
    from tensorfold.server.request_limits import RequestLimit
    from tensorfold.families.qwen4_exp.cuda.tp_scheduler import TensorParallelDecoder
    from tests.test_cuda_retention_resources import catalog
    from tests.test_retention_resources import resource_decoder
    from tensorfold.server.segments import SegmentEncoder, LIMIT
    from tensorfold.families.qwen4_exp.cuda.adaptive_controller import AdaptiveController
    from pathlib import Path
    from copy import deepcopy

    fixture = json.loads((Path(__file__).parent / "fixtures" / "c27_capacity_plan_shape.json").read_text())
    plan = fixture["capacity_plan"]

    context, pending, waiting = 262144, 27, 254804724
    selected = catalog(None)
    selected.world = 8
    selected.states = [SimpleNamespace(limit=context + 3, logical_context=context) for _ in range(pending)]
    decoder = selected._cohort.multi = resource_decoder(context)
    decoder.depth, decoder.capacity = 4, context + 5  # Actual default MTP4 geometry.
    decoder.states, decoder.logical_context = selected.states, context
    decoder.w.cfg.vocab = 248320
    decoder.w.layers = [None] * fixture["provenance"]["target_layers"]
    scheduler = Scheduler(decoder, max_streams=pending, admission_policy="fifo",
                          max_waiting_input_bytes=waiting, output_queue_tokens=32)
    adapter = TensorParallelDecoder.__new__(TensorParallelDecoder)
    adapter.decoder, adapter.admission_policy = decoder, "fifo"
    scheduler.decoder = adapter
    header = resources.retention_header_resources(None, 8, slots=pending)
    baseline = resources.HostBaseline(35921772001 - header.global_host_bytes + 45889656, 0,
        prepaid_native_fixed_payload_bytes=1315966903, native_state_slots=pending,
        native_fd_limit=1048576, prepaid_native_prefix_payload_bytes=171348, native_host_bytes=9)
    selected.host_resources = baseline
    captured = resources.PresentationBound(
        "c1c1e57c6e7ad5a37159fdf7eaadc6dcaa800fc4c02f86e1ce31fa468a75c527",
        248320, 256, 128, 0, 256, 0, "tokenizers-0.23.3/bytelevel-concat-v2",
        (256, 6), (128, 6), (128, 6))
    monkeypatch.setattr(resources, "tokenizer_presentation_bound", lambda *args, **kwargs: captured)
    sampled_allowance = 351202996224
    def recorded_sample(total, *, materialized_owned_bytes):
        assert total - materialized_owned_bytes < sampled_allowance
        return {"usable_remaining_bytes": sampled_allowance,
                "remaining_commitment_bytes": total - materialized_owned_bytes,
                "materialized_owned_bytes": materialized_owned_bytes}
    monkeypatch.setattr(capacity, "admit_host_commitment", recorded_sample)
    limit = RequestLimit(pending)
    segments = SegmentEncoder.__new__(SegmentEncoder)
    segments.limit = LIMIT
    segments.opener = "<|im_start|>"
    receipt = finalized(selected, scheduler, limit, connections=28, segments=segments)
    assert selected.host_resources is baseline
    assert receipt["ordinary_host_baseline"] == baseline.__dict__
    assert receipt["dynamic_host_payload"]["capacity_bytes"] > 0
    declared = {key: value for key, value in receipt.items()
                if key not in ("identity", "declaration_identity", "authority")}
    assert receipt["identity"] == hashlib.sha256(json.dumps(
        declared, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    pool = selected._http_payload_pool
    body = {"messages": [{"role": "user", "content": "x" * 1000000}],
            "enable_thinking": False, "max_tokens": 64, "stream": False,
            "temperature": 0, "model": "selected"}
    leases = []
    try:
        for _ in range(pending):
            lease = pool.lease()
            leases.append(lease)
            lease.begin_body().receive(1 << 20)
            lease.generation(body_bytes=lease.body_bytes, output_tokens=64)
            lease.render_input(body)
            lease.render(1 << 20, 64)
            lease.segment_cache(LIMIT, 256, 1 << 20)
            lease.ids(4 * context, entries=256, positions=4 * context)
        monitor = pool.lease()
        leases.append(monitor)
        monitor.control_kind = "monitor"
        monitor.control(0)
        monitor.snapshot(plan)
        first = deepcopy(plan)
        monitor.snapshot(plan)
        second = deepcopy(plan)
        controller = AdaptiveController(slots=27, row_budget=135, proposal_limit=4, prefill_tokens=256)
        execution = controller.record(credit=monitor.snapshot)
        controller.record(credit=monitor.snapshot)  # Placement also reads controller migration.
        monitor.encode({"capacity_plan": first, "runtime": {"capacity_plan": second, "execution": execution}})
        del first, second, execution
        assert pool.used <= pool.capacity and pool.active_handlers == 28
        assert finalized(selected, scheduler, limit, connections=28, segments=segments) is receipt
    finally:
        for lease in leases:
            lease.close()
    assert pool.used == 0 and pool.active_handlers == 0
    detokenize = receipt["dynamic_host_payload"]["detokenize_phase"]
    assert detokenize["budget"]["encoded_id_bytes"] == 32 << 20
    assert detokenize["input_ids"] > context  # This route deliberately has no L cap.
    owner, occupied = pool.lease(), pool.lease()
    try:
        owner.control_kind, owner.control_required = "detokenize", True
        owner.begin_body()  # Native conversion AND future encoder funded before body read.
        assert owner.claims["control"] >= max(detokenize["native_heap_requested_bytes"],
                                             detokenize["result_to_python_payload_bytes"])
        assert owner.funded_output_bound >= 12 * detokenize["decoded_utf8_bytes"]
        occupied.set("other", pool.capacity - pool.used)
        owner.encode({"prompt": "ASCII😀" * 32})
        assert owner.claims["encoder"] == 0
    finally:
        occupied.close()
        owner.close()
    assert pool.used == 0

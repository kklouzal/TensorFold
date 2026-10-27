"""Named Python RPC allocations, independent from native Store copies."""
import json
import sys
from types import SimpleNamespace
import pytest

pytest.importorskip("torch")

from tensorfold.cuda.rank_protocol import MAX_MESSAGE_BYTES
from tensorfold.cuda.streams import Stream
from tensorfold.families.qwen4_exp.cuda.retention_resources import (
    continuation_payload_bytes, tp_rpc_python_payload,
)
from tensorfold.families.qwen4_exp.cuda.tp_scheduler import MAX_PROTOCOL_SEQUENCE, TensorParallelDecoder


def terms(*, context=128, body=96 << 20, copy=False):
    return continuation_payload_bytes(context=context, vocab_size=248320, presentation=768,
                                       max_body_bytes=body, copy_enabled=copy)


def project(value, *, world=8, context=128, pending=2, body=96 << 20):
    return tp_rpc_python_payload(value, world=world, context=context,
        max_pending_requests=pending, max_http_body_bytes=body)


@pytest.mark.parametrize("enabled", [False, True])
def test_list_partition_preserves_original_total_and_common_stream_contract(enabled):
    value = terms(copy=enabled)
    part = value["list_partition"]
    sequence = part["per_list_pointer_bytes"] + part["prompt_integer_pool_bytes"]
    assert part["common_stream_bytes"] == 6 * sequence
    assert part["leader_frontend_bytes"] == 5 * sequence
    assert part["copy_bytes"] == (4 * sequence if enabled else 0)
    assert value["continuation_lists"] == (
        part["common_stream_bytes"] + part["leader_frontend_bytes"] + part["copy_bytes"])
    fields = Stream.__dataclass_fields__
    common = [name for name in value["token_list_roles"] if name.startswith("stream.")]
    assert {name.removeprefix("stream.") for name in common} == {"out", "context", "committed", "drafts", "stops", "owed"}
    assert all(name.removeprefix("stream.") in fields for name in common)


@pytest.mark.parametrize("enabled", [False, True])
def test_followers_pay_prompt_once_and_never_replicate_leader_frontend(enabled):
    value = terms(copy=enabled)
    result = project(value)
    part, scope = value["list_partition"], result["per_follower_scope"]
    assert scope["prompt_integer_pool"] == part["prompt_integer_pool_bytes"]
    assert scope["prompt_pointer_list"] == part["per_list_pointer_bytes"]
    assert set(scope) == {"common_stream_lists", "copy_lists", "copy_index", "prompt_pointer_list", "prompt_integer_pool"}
    assert result["per_follower_scope_bytes"] == sum(scope.values())
    old_blind_replica = value["continuation_lists"] + value["copy_index"]
    sequence = scope["prompt_pointer_list"] + scope["prompt_integer_pool"]
    assert old_blind_replica - result["per_follower_scope_bytes"] == 4 * sequence


def test_actual_unpack_clones_pointer_lists_while_sharing_parsed_integer_objects():
    raw = json.dumps({"prompt": [987654, 987655], "owed": [987656], "count": 4,
        "sampling": None, "grammar": [], "draft": False, "stop_eos": True,
        "background": False, "carry": {"tag": "owned"}, "profile_epoch": 0,
        "committed_output_tokens": 0})
    body = json.loads(raw)
    decoder = TensorParallelDecoder.__new__(TensorParallelDecoder)
    decoder.admission_policy = "legacy"
    stream = decoder._unpack(body)
    assert stream.prompt is not body["prompt"] and stream.owed is not body["owed"]
    assert all(a is b for a, b in zip(stream.prompt, body["prompt"], strict=True))
    assert stream.owed[0] is body["owed"][0] and stream.carry is body["carry"]
    budget = project(terms(context=8), context=8)["per_follower_scope"]
    assert budget["prompt_pointer_list"] >= sys.getsizeof(stream.prompt)
    assert budget["prompt_integer_pool"] >= sum(sys.getsizeof(token) for token in stream.prompt)


def test_rpc_frame_authority_is_independent_of_http_body_limit_and_overlapping_fields():
    small = project(terms(body=32 << 20), body=32 << 20)
    large = project(terms(body=192 << 20), body=192 << 20)
    assert small == large
    assert small["frame_bytes"] == MAX_MESSAGE_BYTES == 96 << 20
    assert small["global_rpc_python_working_bytes"] > 0
    assert not any("http" in name or "manifest" in name or "replay" in name
                   for name in small["named_rpc_terms"])


def test_global_working_projection_keeps_current_error_and_publication_owners():
    result = project(terms())
    phases = result["phase_working_bytes"]
    assert result["global_rpc_python_working_bytes"] == (
        sum(result["named_rpc_terms"].values()) + max(phases.values()))
    assert result["named_rpc_terms"]["follower_owned_rpc_children"] > 0
    assert result["named_rpc_terms"]["preserved_rpc_error_owners"] > 0
    assert result["per_http_rpc_error_tail_bytes"] >= result["json_graph_bytes"]
    assert phases["source_encoding"] >= result["ascii_text_bytes"]
    assert "previous_body" not in result["named_rpc_terms"]
    assert "post_send_text" not in result["named_rpc_terms"]
    assert project(terms(), pending=3)["global_rpc_python_working_bytes"] > result["global_rpc_python_working_bytes"]


@pytest.mark.parametrize("kind,completion", [("round", None), ("retention_retire", True),
    ("retention_suspend", False), ("retention_boundary", {"drained": False, "changed": True})])
@pytest.mark.parametrize("failed", [False, True])
def test_actual_ack_source_shape_is_separate_from_malformed_received_graphs(kind, completion, failed):
    decoder = object.__new__(TensorParallelDecoder)
    decoder.engine, decoder.requests = SimpleNamespace(serving_profiles=None), {}
    decoder._state_digest = lambda: "f" * 64
    failure = type("A" * 2048 + "𐐀", (ValueError,), {})("bounded test error")
    def execute(_body, _source=None):
        if failed:
            raise failure
        return completion
    decoder._dispatch = execute
    _result, error, status = decoder._execute({"kind": kind})
    fields = {"error", "state"} | ({"complete"} if kind.startswith("retention_") else set())
    assert set(status) == fields and len(status["state"]) == 64
    assert error is (failure if failed else None)
    if "complete" in status:
        assert status["complete"] == (None if failed else completion)
    encoded = json.dumps(status).encode("ascii")
    assert len(encoded) <= MAX_MESSAGE_BYTES
    pending, seen, payload = [status], set(), 0
    while pending:
        value = pending.pop()
        if id(value) in seen:
            continue
        seen.add(id(value))
        payload += sys.getsizeof(value)
        if type(value) is dict:
            pending.extend(value.keys())
            pending.extend(value.values())
    bound = project(terms())
    assert payload <= bound["source_ack_graph_bytes"] < bound["json_graph_bytes"]
    assert bound["per_http_rpc_error_tail_bytes"] >= bound["json_graph_bytes"]


def test_one_rank_has_no_peer_rpc_payload():
    result = project(terms(), world=1)
    assert result["global_rpc_python_working_bytes"] == result["per_http_rpc_error_tail_bytes"] == 0
    assert result["named_rpc_terms"] == {}


def test_actual_leader_maps_share_one_wire_integer_and_a_distinct_identity_key():
    decoder = TensorParallelDecoder.__new__(TensorParallelDecoder)
    decoder.requests, decoder.wires = {}, {}
    stream = Stream([987654], 1)
    wire = MAX_PROTOCOL_SEQUENCE - 1
    stream._media_wire = wire
    decoder._remember(wire, stream)
    request_key, identity_key = next(iter(decoder.requests)), next(iter(decoder.wires))
    assert request_key is decoder.wires[identity_key] is stream._media_wire is wire
    assert identity_key == id(stream) and identity_key is not wire
    result = project(terms(), pending=1)
    row = result["leader_request_map_metadata"]
    limits = result["leader_request_map_limits"]
    assert row["wire_integer_pool"] == sys.getsizeof(MAX_PROTOCOL_SEQUENCE)
    assert row["identity_integer_pool"] >= sys.getsizeof(identity_key)
    assert row["next_wire_counter"] == sys.getsizeof(MAX_PROTOCOL_SEQUENCE)
    assert limits["wire_integer_bytes"] == sys.getsizeof(wire)
    decoder._forget(wire)
    assert decoder.requests == decoder.wires == {}


@pytest.mark.parametrize("pending", [1, 2, 3, 8, 17, 128])
def test_process_fixed_map_highwater_bounds_actual_insert_delete_churn(pending):
    decoder = TensorParallelDecoder.__new__(TensorParallelDecoder)
    decoder.requests, decoder.wires = {}, {}
    active = []
    result = project(terms(), pending=pending)
    limits = result["leader_request_map_limits"]
    table_limit = result["leader_request_map_metadata"]["request_and_identity_map_tables"]
    object_limit = result["leader_request_map_metadata"]["request_and_identity_map_objects"]
    for wire in range(pending):
        stream = Stream([987654], 1)
        decoder._remember(wire, stream)
        active.append((wire, stream))
    largest = 0
    for wire in range(pending, pending + 8 * pending + 64):
        old, _stream = active.pop(0)
        decoder._forget(old)
        stream = Stream([987654], 1)
        decoder._remember(wire, stream)
        active.append((wire, stream))
        assert len(decoder.requests) == len(decoder.wires) == pending
        current = sys.getsizeof(decoder.requests) + sys.getsizeof(decoder.wires)
        largest = max(largest, current)
        assert current <= object_limit + table_limit
    for wire, _stream in active:
        decoder._forget(wire)
    assert len(decoder.requests) == len(decoder.wires) == 0
    assert sys.getsizeof(decoder.requests) + sys.getsizeof(decoder.wires) <= object_limit + table_limit
    assert result["leader_request_map_limits"]["request_highwater"] == pending
    if pending == 2:
        # Actual 3.12.3 calculate_log2_keysize small-size OR permits sixteen
        # slots under churn; a naive ceilpow2(max(8,3P)) would miss this.
        assert limits["map_capacity"] >= 16
        assert largest > 2 * sys.getsizeof({0: None, 1: None})


def test_leader_metadata_is_not_multiplied_by_followers_or_live_occupancy():
    two = project(terms(), world=2, pending=8)
    eight = project(terms(), world=8, pending=8)
    assert two["leader_request_map_metadata"] == eight["leader_request_map_metadata"]
    row = two["leader_request_map_metadata"]
    assert two["named_rpc_terms"]["leader_request_map_metadata"] == sum(row.values())
    assert row["one_dictionary_resize_overlap"] == max(
        two["leader_request_map_limits"]["map_table_bytes"],
        two["leader_request_map_limits"]["stream_dictionary_table_bytes"])


def test_optional_wire_reference_pays_metadata_backing_without_another_integer():
    stream = Stream([987654], 1)
    fields = ("_fifo_started", "_media_outcome_retired", "admission_eligibility", "admission_status",
        "admission_ticket", "blocked_input", "delivery_owner", "output_capacity", "principal",
        "production_budget", "retirement_host", "_media_rank_owned", "_retention_staged", "_tp_token_limit",
        "retention_intent")
    for name in fields:
        setattr(stream, name, None)
    wire = MAX_PROTOCOL_SEQUENCE - 1
    before = sys.getsizeof(stream.__dict__)
    stream._media_wire = wire
    result = project(terms(), pending=1)
    backing = result["leader_request_map_metadata"]["optional_stream_metadata_backing"]
    assert backing >= sys.getsizeof(stream) + max(before, sys.getsizeof(stream.__dict__))
    assert stream._media_wire is wire
    assert result["leader_request_map_metadata"]["wire_integer_pool"] == sys.getsizeof(wire)


@pytest.mark.parametrize("option,value", [("world", True), ("world", 0), ("context", 0),
                                           ("max_pending_requests", 0), ("max_http_body_bytes", 0)])
def test_invalid_dimension_refuses_without_allocating_frame_payload(option, value):
    options = dict(world=8, context=128, max_pending_requests=2, max_http_body_bytes=96 << 20)
    options[option] = value
    with pytest.raises(ValueError):
        tp_rpc_python_payload(terms(), **options)


def test_stale_context_or_body_layout_refuses():
    with pytest.raises(ValueError, match="context differs"):
        project(terms(context=64))
    with pytest.raises(ValueError, match="HTTP/body layout"):
        project(terms(body=32 << 20))

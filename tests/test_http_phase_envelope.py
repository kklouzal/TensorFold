"""Ownership-state oracle for the explicitly supplied HTTP phase extents."""

from copy import deepcopy
from itertools import product
import sys

import pytest

from tensorfold.server.http_phase_envelope import http_phase_envelope


@pytest.fixture
def ownership():
    shared = {"ordinary_baseline": 101, "header": 103, "waiting_input": 107,
              "model_continuation": 109, "native_unpaid": 113, "prefix_unpaid": 127}
    fixed = {"follower_owned_rpc_children": 131, "preserved_rpc_error_owners": 137,
             "leader_prepared_prompt_pointer_lists": 139, "leader_request_map_metadata": 149}
    persistent = {"responses_current_cache": 151, "responses_cache_wrappers": 157,
                  "segment_cache_payload_and_metadata": 163, "bpe_thread_caches": 167,
                  "framing_and_runtime": 173}
    routes = ("read_and_parse", "responses_history_and_bridge", "anthropic_translate_and_bridge",
              "metadata_encode", "metadata_decode", "responses_get", "responses_put", "completed_output",
              "http_error", "live_gate_encode", "decisions", "runtime_and_control")
    rpc_names = ("source_encoding", "publication_and_peer_work", "dispatch_and_peer_work",
                 "ack_batch_and_peer_return")
    # Names identify allocations, so aliases count once even across roles.
    allocations = {**shared, **fixed, **persistent, "outer_body": 179, "old_entry": 181,
                   "clone_memo": 191, "native_batch": 193, "all_encodings": 197,
                   "fresh_arrays": 199, "output_wire": 211, "error_frames": 223,
                   "initial_body": 227, "initial_native": 229, "maintenance_control": 233,
                   "maintenance_wire": 239}
    handler_states = {}
    for index, name in enumerate(routes):
        objects = {"outer_body", "responses_current_cache", "framing_and_runtime"}
        objects.add(("old_entry", "clone_memo", "native_batch", "all_encodings", "fresh_arrays",
                     "output_wire", "error_frames")[index % 7])
        handler_states[name] = objects
    handler_states["responses_get"] |= {"old_entry", "clone_memo", "output_wire"}
    handler_states["metadata_encode"] |= {"native_batch", "all_encodings", "fresh_arrays"}
    handler_states["http_error"] |= {"old_entry", "clone_memo", "error_frames"}
    rpc_states = {}
    for index, name in enumerate(rpc_names):
        atom = f"rpc_work_{index}"
        allocations[atom] = 241 + index * 2
        rpc_states[name] = {atom}
    common_objects = set(shared) | set(fixed) | set(persistent)
    def weight(objects):
        return sum(allocations[name] for name in objects)
    rpc_rows = {name: weight(objects) for name, objects in rpc_states.items()}
    initial = {"initial_body", "initial_native", "responses_current_cache", "segment_cache_payload_and_metadata"}
    maintenance_states = ({"maintenance_control"}, {"maintenance_wire"}, set())
    arguments = {"max_pending_requests": 1, "max_http_connections": 1,
                 "shared_rows": shared, "rpc_python": {"named_rpc_terms": fixed,
                     "phase_working_bytes": rpc_rows,
                     "global_rpc_python_working_bytes": sum(fixed.values()) + max(rpc_rows.values())},
                 "persistent_frontend_rows": persistent,
                 "handler_phases": {name: weight(objects - common_objects)
                                    for name, objects in handler_states.items()},
                 "initial_preparation_bytes": weight(initial - common_objects),
                 "maintenance_during_initial_preparation_bytes": max(map(weight, maintenance_states))}
    return arguments, allocations, common_objects, handler_states, rpc_states, initial, maintenance_states


def test_all_permitted_ownership_states_fit_without_charging_aliases_twice(ownership):
    arguments, allocations, common, handlers, rpc, initial, maintenance = ownership
    record = http_phase_envelope(**arguments)
    # Enumerate legal live allocation sets independently of the projection's
    # addition/max expression. Initial prepare excludes generation RPC work;
    # independent maintenance is deliberately allowed in that state.
    states = [common | route | current for route, current in product(handlers.values(), rpc.values())]
    states.extend(common | initial | control for control in maintenance)
    measured = [sum(allocations[name] for name in state) for state in states]
    assert record["total_bytes"] == max(measured)
    assert all(value <= record["total_bytes"] for value in measured)
    assert record["rpc_fixed_rows"] == arguments["rpc_python"]["named_rpc_terms"]
    assert record["rpc_fixed_rows"]["preserved_rpc_error_owners"] == 137


def test_live_gate_and_metadata_still_overlap_the_current_worker(ownership):
    arguments = ownership[0]
    arguments["handler_phases"]["live_gate_encode"] = 10_000
    record = http_phase_envelope(**arguments)
    assert record["maximum_phases"] == ("live_gate_encode",)
    assert record["phase_bytes"]["live_gate_encode"] > 10_000
    arguments["handler_phases"]["metadata_encode"] = 20_000
    assert http_phase_envelope(**arguments)["maximum_phases"] == ("metadata_encode",)


def test_initial_preparation_pays_independent_maintenance_without_dropping_fixed_owners(ownership):
    arguments = ownership[0]
    arguments["initial_preparation_bytes"] = 30_000
    arguments["maintenance_during_initial_preparation_bytes"] = 40_000
    record = http_phase_envelope(**arguments)
    assert record["maximum_phases"] == ("initial_preparation",)
    assert record["phase_bytes"]["initial_preparation"] == 70_000
    assert record["total_bytes"] > 70_000 + arguments["rpc_python"]["named_rpc_terms"]["preserved_rpc_error_owners"]


@pytest.mark.parametrize("group", ["shared_rows", "persistent_frontend_rows", "handler_phases"])
def test_missing_or_unexpected_role_cannot_silently_disappear(ownership, group):
    arguments = ownership[0]
    for name in list(arguments[group]):
        incomplete = deepcopy(arguments)
        del incomplete[group][name]
        with pytest.raises(ValueError, match="requires exactly these named rows"):
            http_phase_envelope(**incomplete)
    arguments[group]["undeclared_role"] = 0
    with pytest.raises(ValueError, match="requires exactly these named rows"):
        http_phase_envelope(**arguments)


@pytest.mark.parametrize("group", ["named_rpc_terms", "phase_working_bytes"])
def test_incomplete_rpc_record_is_refused(ownership, group):
    arguments = ownership[0]
    del arguments["rpc_python"][group][next(iter(arguments["rpc_python"][group]))]
    with pytest.raises(ValueError, match="requires exactly these named rows"):
        http_phase_envelope(**arguments)


def test_rpc_total_must_keep_all_fixed_and_current_obligations(ownership):
    arguments = ownership[0]
    arguments["rpc_python"]["global_rpc_python_working_bytes"] -= 1
    with pytest.raises(ValueError, match="complete fixed and current obligations"):
        http_phase_envelope(**arguments)


@pytest.mark.parametrize("name", ["max_pending_requests", "max_http_connections"])
@pytest.mark.parametrize("value", [0, 2, 4, 16, True, 1.0, None])
def test_other_concurrency_regions_need_their_own_composition(ownership, name, value):
    arguments = ownership[0]
    arguments[name] = value
    with pytest.raises(ValueError, match="proved.*limit1"):
        http_phase_envelope(**arguments)


@pytest.mark.parametrize("value", [-1, True, 1.5, None, sys.maxsize + 1])
def test_invalid_extent_never_becomes_a_projection(ownership, value):
    arguments = ownership[0]
    arguments["handler_phases"]["decisions"] = value
    with pytest.raises(ValueError, match="nonnegative host-sized integer"):
        http_phase_envelope(**arguments)


def test_overflow_in_a_sum_is_rejected_even_when_every_leaf_fits(ownership):
    arguments = ownership[0]
    arguments["initial_preparation_bytes"] = sys.maxsize
    with pytest.raises(ValueError, match="nonnegative host-sized integer"):
        http_phase_envelope(**arguments)


def test_result_owns_its_rows_and_never_claims_physical_admission(ownership):
    arguments = ownership[0]
    record = http_phase_envelope(**arguments)
    saved = deepcopy(record)
    arguments["handler_phases"]["decisions"] = 0
    arguments["rpc_python"]["named_rpc_terms"]["preserved_rpc_error_owners"] = 0
    assert record == saved
    assert "no physical admission" in record["scope"]
    assert any("post-exhaustion" in value for value in record["separate_obligations"])

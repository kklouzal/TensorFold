"""Lifetime algebra for one static CUDA HTTP handler and logical owner.

This composes REQUIRED caller-proved payload extents; it does not derive route
bounds, authorize their proofs, or admit physical host memory. In particular,
one logical request does not serialize metadata, live CallGate encoding or
Responses translation against a retiring model worker. Only initial generation
preparation has that exclusion, and independent maintenance remains explicit.
"""

import sys


_SHARED = frozenset(("ordinary_baseline", "header", "waiting_input", "model_continuation",
                     "native_unpaid", "prefix_unpaid"))
_RPC_FIXED = frozenset(("follower_owned_rpc_children", "preserved_rpc_error_owners",
                        "leader_prepared_prompt_pointer_lists", "leader_request_map_metadata"))
_RPC_PHASES = frozenset(("source_encoding", "publication_and_peer_work", "dispatch_and_peer_work",
                         "ack_batch_and_peer_return"))
_PERSISTENT = frozenset(("responses_current_cache", "responses_cache_wrappers",
                         "segment_cache_payload_and_metadata", "bpe_thread_caches", "framing_and_runtime"))
_HANDLER_PHASES = frozenset(("read_and_parse", "responses_history_and_bridge", "anthropic_translate_and_bridge",
                            "metadata_encode", "metadata_decode", "responses_get", "responses_put",
                            "completed_output", "http_error", "live_gate_encode", "decisions",
                            "runtime_and_control"))


def _bytes(value, name):
    if type(value) is not int or not 0 <= value <= sys.maxsize:
        raise ValueError(f"HTTP phase {name} must be a nonnegative host-sized integer")
    return value


def _rows(value, names, label):
    if type(value) is not dict or value.keys() != names:
        raise ValueError(f"HTTP phase {label} requires exactly these named rows: {', '.join(sorted(names))}")
    return {name: _bytes(value[name], name) for name in sorted(names)}


def _sum(values, name):
    return _bytes(sum(values), name)


def http_phase_envelope(*, max_pending_requests, max_http_connections, shared_rows, rpc_python,
                        persistent_frontend_rows, handler_phases, initial_preparation_bytes,
                        maintenance_during_initial_preparation_bytes):
    """Compose the proved initial H=1/P=1 region without inventing leaf bounds.

    ``shared_rows`` retains every model/native/input/header obligation. The
    actual ``tp_rpc_python_payload`` record supplies ALL fixed/persistent/error
    rows unchanged, including its (P+1) error journals; only its current phase
    maximum is eligible for the initial-preparation lifetime exclusion.

    ``persistent_frontend_rows`` pays current cache payloads, historical table
    highwaters, native thread/BPE generations and framing/runtime owners once.
    Each ``handler_phases`` value bounds the COMPLETE simultaneously reachable
    ownership union for that phase, excluding only those globally paid owners.
    It includes outer/inner bodies, translated/history graphs, old generations,
    clone memos, returned Encodings/arrays, strings/wires, native working space,
    and error frames wherever reachable. Sequential subphases may already be
    combined by a caller-proved maximum. No route may be omitted: an explicit
    zero needs the caller's proof that the role is absent or already paid.

    Initial preparation includes all its retained caller/source/error owners.
    Its logical lease excludes prior generation work only after real worker
    retirement. The separately required maintenance extent pays any runtime,
    migration or external control work that can still overlap it; zero needs
    an enforced idle contract. Static App/tokenizer/template/cache-generation
    and exhaustion/retirement contracts belong to the caller. An H/P change or
    profiles/media/retained mode requires its own lifetime composition.
    """
    if type(max_pending_requests) is not int or max_pending_requests != 1:
        raise ValueError("HTTP phase composition requires the proved logical owner limit1")
    if type(max_http_connections) is not int or max_http_connections != 1:
        raise ValueError("HTTP phase composition requires the proved retained handler limit1")
    shared = _rows(shared_rows, _SHARED, "shared obligations")
    persistent = _rows(persistent_frontend_rows, _PERSISTENT, "persistent frontend obligations")
    handlers = _rows(handler_phases, _HANDLER_PHASES, "handler phases")
    if type(rpc_python) is not dict:
        raise ValueError("HTTP phase composition requires its actual RPC payload record")
    fixed = _rows(rpc_python.get("named_rpc_terms"), _RPC_FIXED, "fixed RPC obligations")
    rpc_phases = _rows(rpc_python.get("phase_working_bytes"), _RPC_PHASES, "current RPC phases")
    active = max(rpc_phases.values())
    rpc_total = _sum((*fixed.values(), active), "complete RPC payload")
    if _bytes(rpc_python.get("global_rpc_python_working_bytes"), "RPC record total") != rpc_total:
        raise ValueError("HTTP phase RPC record does not preserve its complete fixed and current obligations")
    initial = _bytes(initial_preparation_bytes, "initial preparation")
    maintenance = _bytes(maintenance_during_initial_preparation_bytes, "independent preparation maintenance")
    common = _sum((*shared.values(), *fixed.values(), *persistent.values()), "common retained payload")
    phases = {name: _sum((active, value), name) for name, value in handlers.items()}
    phases["initial_preparation"] = _sum((initial, maintenance), "initial preparation and maintenance")
    peak = max(phases.values())
    total = _sum((common, peak), "complete requested payload")
    return {"schema": "tensorfold.http-phase-requested-payload-v1",
            "limits": {"max_pending_requests": 1, "max_http_connections": 1},
            "shared_rows": shared, "rpc_fixed_rows": fixed, "rpc_phase_rows": rpc_phases,
            "persistent_frontend_rows": persistent, "handler_phase_extents": handlers,
            "initial_preparation_bytes": initial,
            "maintenance_during_initial_preparation_bytes": maintenance,
            "current_rpc_phase_bytes": active, "common_retained_bytes": common,
            "phase_bytes": phases, "peak_phase_bytes": peak,
            "maximum_phases": tuple(name for name, value in phases.items() if value == peak),
            "total_bytes": total,
            "scope": "caller-proved requested payload lifetime composition; no physical admission or RSS grant",
            "separate_obligations": ("actual route/renderer/Segment/native/cache/error/framing leaf proofs",
                "enforced static caller, logical/handler ownership and independent maintenance lifetimes",
                "post-exhaustion retirement, allocator/C-stack/native RSS and actual target qualification")}

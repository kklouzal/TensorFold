"""Published adaptive execution controls; request threads never touch CUDA state.

Only controls bounded by the allocated execution envelope are writable. The
single scheduling coordinator applies them between batches. Structural changes
have no write interface until their state-preserving transition is implemented.
Readings are advisory snapshots, never new admission or capacity proofs.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from tensorfold.server.errors import RequestError

ROUTE = "/v1/runtime"
SETTINGS = frozenset({"prefill_tokens", "proposal_limit", "row_budget", "session_ceiling"})
MAX_BODY_BYTES = 1 << 20


def _owner(app: Any):
    engine = getattr(app, "engine", None)
    scheduler = getattr(engine, "scheduler", None)
    decoder = getattr(scheduler, "decoder", None)
    controller = getattr(decoder, "controller", None)
    return engine, scheduler, decoder, controller


def snapshot(app: Any) -> dict[str, Any] | None:
    """Read host-owned state without allocating, inspecting or synchronizing GPUs."""
    engine, scheduler, decoder, controller = _owner(app)
    if controller is None:
        return None
    execution = controller.record()
    streams = list(getattr(decoder, "streams", {}).values())
    filling = list(getattr(decoder, "filling", ()))
    kept = list(getattr(decoder, "kept", ()))
    # Prefix bytes are an evictable derived cache, not a persistent user session.
    # This API creates request-scoped sequences and has no retained-session owner.
    cached_slots = {id(point[1]) for point in kept}
    placement = getattr(decoder, "history_record", None)
    manifest = getattr(scheduler, "execution_manifest_record", None)
    return {
        "object": "tensorfold.runtime", "schema_version": 1,
        "context_tokens": getattr(engine, "context_window", None),
        "capacity_plan": deepcopy(getattr(engine, "capacity_plan", None)),
        "execution": execution,
        "sessions": {"lifetime": "request_scoped", "retained": 0,
                     "active": len(streams) + len(filling),
                     "derived_prefix_slots": len(cached_slots), "derived_prefix_entries": len(kept),
                     "decode": len(streams), "prefill": len(filling),
                     "allocated_slots": controller.slots,
                     "admission_ceiling": controller.admission_limit(),
                     "queued": scheduler.waiting.qsize() + int(scheduler.held is not None)},
        "placement": placement() if placement is not None else None,
        "profile_manifest": manifest() if manifest is not None else None,
        "reading_contract": "advisory_host_snapshot_no_gpu_synchronization",
    }


def request(app: Any, body: Any) -> dict[str, Any]:
    """Validate the HTTP boundary, then publish a coalesced controller request."""
    if not isinstance(body, dict) or not body:
        raise RequestError("runtime settings must be a nonempty JSON object")
    standalone = set(body) & {"profile_table", "profile_manifest", "attention_degree"}
    if standalone and len(body) != 1:
        raise RequestError("profile registration or attention placement must be requested separately")
    unknown = set(body) - SETTINGS - standalone
    if unknown:
        raise RequestError("unsupported runtime setting: " + ", ".join(sorted(unknown)))
    for name, value in body.items():
        if name == "profile_manifest":
            if not isinstance(value, str):
                raise RequestError("profile_manifest must be a relative file name")
            continue
        if name == "profile_table":
            if not isinstance(value, dict):
                raise RequestError("profile_table must be a JSON object")
            continue
        if type(value) is not int:
            raise RequestError(f"{name} must be an integer")
    _engine, scheduler, _decoder, controller = _owner(app)
    if controller is None:
        raise RequestError("this engine has no adaptive runtime controls")
    if "profile_manifest" in body:
        owner = getattr(scheduler, "request_execution_manifest", None)
        if not callable(owner):
            raise RequestError("this engine has no execution manifest publication owner")
        try:
            manifest = owner(body["profile_manifest"])
        except (ValueError, RuntimeError) as error:
            raise RequestError(str(error)) from error
        return {"object": "tensorfold.runtime.update", "schema_version": 1, "profile_manifest": manifest}
    def publish():
        if "profile_table" in body:
            return controller.install_profiles(body["profile_table"])
        if "attention_degree" in body:
            return controller.request_migration(body["attention_degree"])
        return controller.request(**body)

    owner = getattr(scheduler, "request_runtime", None)
    if not callable(owner):
        raise RequestError("this engine has no runtime control publication owner")
    try:
        # The Scheduler's publication lock excludes closing and coalesces an
        # idle wake; request threads never acquire decoder execution ownership.
        execution = owner(publish)
    except (ValueError, RuntimeError) as error:
        raise RequestError(str(error)) from error
    # The request may be applied while an advisory response is being assembled.
    # Return the controller's locked publication receipt, not an inferred status.
    return {"object": "tensorfold.runtime.update", "schema_version": 1, "execution": execution}

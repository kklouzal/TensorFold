"""Published adaptive execution controls; request threads never touch CUDA state.

The scheduling coordinator applies bounded execution controls between batches.
Prepared structural selections use the separate opt-in serving-profile owner.
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
    """Keep model projection readers scoped; draining reads only host profile status."""
    engine = getattr(app, "engine", None)
    profiles = getattr(engine, "serving_profiles", None)
    if profiles is None:
        return _snapshot_current(app)
    from tensorfold.cuda.profile_binding import ProfileUnavailable

    reading = False
    try:
        with profiles.gate.scope():
            reading = True
            try:
                return _snapshot_current(app)
            except BaseException as error:
                # An unexpected projection traceback may retain decoder/state
                # objects. It cannot be presented as a retired reader while a
                # different full model is loaded; retain it under containment.
                profiles._poison(error)
    except ProfileUnavailable:
        if reading:
            raise
        return {"object": "tensorfold.runtime", "schema_version": 1,
                "serving_profile": profiles.record(), "context_tokens": None, "capacity_plan": None,
                "execution": None, "sessions": None, "placement": None,
                "reading_contract": "advisory_host_snapshot_no_gpu_synchronization"}


def _snapshot_current(app: Any) -> dict[str, Any] | None:
    """Project only host metadata; caller owns the profile read until return."""
    engine, scheduler, decoder, controller = _owner(app)
    profiles = getattr(engine, "serving_profiles", None)
    serving = None if profiles is None else profiles.record()
    if controller is None or serving is not None and not serving["available"]:
        return (None if serving is None else {"object": "tensorfold.runtime", "schema_version": 1,
                "serving_profile": serving, "context_tokens": None, "capacity_plan": None,
                "execution": None, "sessions": None, "placement": None})
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
        "serving_profile": serving,
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
    standalone = set(body) & {"profile_table", "profile_manifest", "attention_degree", "serving_profile"}
    if standalone and len(body) != 1:
        raise RequestError("profile registration or attention placement must be requested separately")
    unknown = set(body) - SETTINGS - standalone
    if unknown:
        raise RequestError("unsupported runtime setting: " + ", ".join(sorted(unknown)))
    for name, value in body.items():
        if name in ("profile_manifest", "serving_profile"):
            if not isinstance(value, str):
                raise RequestError(f"{name} must be a relative file name")
            continue
        if name == "profile_table":
            if not isinstance(value, dict):
                raise RequestError("profile_table must be a JSON object")
            continue
        if type(value) is not int:
            raise RequestError(f"{name} must be an integer")
    if "serving_profile" in body:
        # Structural controls must never capture the old decoder in a caller
        # or preserved notification-error traceback across model retirement.
        engine = getattr(app, "engine", None)
        owner = getattr(engine, "serving_profiles", None)
        if owner is None:
            raise RequestError("serving profile selection was not enabled at startup")
        try:
            result = owner.request(body["serving_profile"])
        except RequestError:
            raise
        except (ValueError, OSError) as error:
            if getattr(owner, "failure", None) is not None:
                raise  # Fatal ownership is not a malformed selector, even if wrapper allocation failed.
            raise RequestError(str(error)) from error
        return {"object": "tensorfold.runtime.update", "schema_version": 1, "serving_profile": result}
    _engine, scheduler, decoder, controller = _owner(app)
    del decoder  # The adaptive controller is host-only; keep no GPU model in HTTP error frames.
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

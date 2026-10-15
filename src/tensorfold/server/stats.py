"""GET /stats and /v1/stats: one JSON snapshot of the server, model, limits, load, KV memory, prefix cache and totals.

Schema ``tensorfold.stats`` version 1 (docs/api.md lists every field). Both servers answer it; a field the server
cannot read is null, never a fabricated zero. A scrape reads Python scalars and ``len()``s of the tables the engine's
worker already keeps (a few dozen loads, O(lanes) and O(kept prefix points)); it makes no GPU call, takes no engine
lock and adds no work to a round. Each reading is taken on its own, as /metrics' are: the snapshot is not atomic
across sections, and its load and memory readings are advisory, never an admission promise.

The process's identity is fixed when this module first loads (before the model): ``INSTANCE_ID`` and the start time
name the serving process, whose prefix cache and counters a restart empties.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from tensorfold import __version__
from tensorfold.server import metrics

SCHEMA_VERSION = 1
INSTANCE_ID = uuid.uuid4().hex
STARTED_UNIX_S = time.time()            # wall clock, for display and Prometheus's process_start_time_seconds
STARTED_MONOTONIC = time.monotonic()    # uptime runs on the monotonic clock
SOURCE_REVISION_ENV = "TENSORFOLD_SOURCE_REVISION"      # the container build's source commit (a label: not a proof)
ROUTES = ("/stats", "/v1/stats")


def snapshot(app: Any, *, server: Any = None, reading: metrics.Metrics | None = None) -> dict[str, Any]:
    """The /stats body. ``reading`` is a scrape's copy of the counters (``metrics.counters_now``), else one is taken."""

    if reading is None:
        reading = metrics.counters_now(app)[0]
    fixed = _fixed(app)
    engine = getattr(app, "engine", None)
    sched = getattr(engine, "scheduler", None) if engine is not None else None     # a CUDA concurrent engine's
    decoder = getattr(sched, "decoder", None) if sched is not None else None
    usage = decoder.usage() if callable(getattr(decoder, "usage", None)) else None
    mac = getattr(app, "scheduler", None)
    mac = mac if mac is not None and hasattr(mac, "active") and hasattr(mac, "waiting") else None
    running, waiting = metrics.requests_now(app)
    if decoder is not None:
        streams = list(getattr(decoder, "streams", {}).values())
        held = sum(1 for s in streams if getattr(s, "waiting", False))
        lanes = {"decode": len(streams) - held, "prefill": len(getattr(decoder, "filling", ()) or ()),
                 "memory_wait": held}
    elif mac is not None:
        lanes = {"decode": int(mac.active), "prefill": len(getattr(mac, "filling", None) or ()), "memory_wait": None}
    else:
        lanes = None
    transport = server.connection_snapshot() if server is not None and hasattr(server, "connection_snapshot") else None
    cuda = engine is not None and mac is None
    limits = dict(fixed["limits"])
    limits["http_connections"] = transport["limit"] if transport is not None else None
    memory = usage or {}
    return {
        "object": "tensorfold.stats", "schema_version": SCHEMA_VERSION,
        "time": {"unix_s": round(time.time(), 3), "uptime_s": round(time.monotonic() - STARTED_MONOTONIC, 3)},
        "server": fixed["server"], "model": fixed["model"], "limits": limits,
        "load": {"requests_in_flight": reading.live_requests if cuda else running + waiting,
                 "requests_running": running, "requests_waiting": waiting, "lanes": lanes,
                 "held_for_memory": (getattr(sched, "held", None) is not None) if decoder is not None else None,
                 "http_connections_in_use": transport["in_use"] if transport is not None else None,
                 "stream_tokens": metrics.stream_tokens(app)},
        "kv": memory.get("kv"),
        "prefix_cache": memory.get("prefix_cache"),
        "totals": {"requests": reading.requests,
                   "finished": {reason: reading.outcomes[reason] for reason in ("stop", "length", "tool_calls")},
                   "cancelled": reading.outcomes["abort"], "failed": reading.outcomes["error"],
                   "prompt_tokens": reading.prompt, "cached_prompt_tokens": reading.cached,
                   "generated_tokens": reading.generation,
                   "generated_tokens_live": reading.generation + reading.live_tokens if cuda else None,
                   "queue_seconds": round(reading.queue_s, 6), "prefill_seconds": round(reading.prefill_s, 6),
                   "decode_seconds": round(reading.decode_s, 6), "rounds": reading.rounds,
                   "drafted": reading.drafted, "accepted": reading.accepted,
                   "preemptions": metrics.preemptions(app),
                   "memory_waits": memory.get("memory_waits"), "memory_stops": memory.get("memory_stops")},
    }


def send(handler: Any, app: Any, extra: dict[str, Any] | None = None) -> None:
    """Write ``snapshot`` as JSON (``extra``: a backend's own sections, the Mac's ``mlx``)."""

    body = snapshot(app, server=getattr(handler, "server", None))
    if extra:
        body.update(extra)
    data = json.dumps(body).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "application/json")
    handler.send_header("Content-Length", str(len(data)))
    handler.end_headers()
    try:
        handler.wfile.write(data)
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = True


def _fixed(app: Any) -> dict[str, Any]:
    """The sections that never change while the server runs, read once per app (on its first scrape)."""

    counters = metrics.of(app)
    fixed = counters.fixed
    if fixed is None:
        fixed = counters.fixed = _describe(app)       # two first scrapes at once make the same value; either is kept
    return fixed


def _describe(app: Any) -> dict[str, Any]:
    engine = getattr(app, "engine", None)
    mac = getattr(app, "scheduler", None)
    mac = mac if mac is not None and hasattr(mac, "active") and hasattr(mac, "waiting") else None
    cuda = engine is not None and mac is None
    sched = getattr(engine, "scheduler", None) if cuda else None
    decoder = getattr(sched, "decoder", None) if sched is not None else None
    name = str(getattr(app, "served_name", "") or getattr(app, "served", "") or "")
    ids = list(getattr(app, "model_ids", None) or ([name] if name else []))
    if cuda:
        context = getattr(app, "max_model_len", None)
        lanes = getattr(sched, "max_streams", None) if sched is not None else 1     # one request at a time
        default_max, thinking = getattr(app, "max_tokens", None), getattr(app, "default_thinking", None)
    else:
        context = int(getattr(app, "context_window", 0) or 0) or None
        lanes = getattr(app, "max_batch_size", None)
        default_max, thinking = getattr(app, "default_max_tokens", None), getattr(app, "enable_thinking", None)
    pair = getattr(engine, "kv_pair", None)
    kv_dtype = None
    if pair is not None and getattr(pair, "key_dtype", None):
        kv_dtype = pair.key_dtype if pair.key_dtype == pair.value_dtype else f"{pair.key_dtype}/{pair.value_dtype}"
    depth = getattr(engine, "depth", None)
    limit = getattr(mac, "max_engine_calls", None)
    return {
        "server": {"version": __version__, "backend": "cuda" if cuda else ("mlx" if mac is not None else None),
                   "instance_id": INSTANCE_ID, "started_unix_s": round(STARTED_UNIX_S, 3),
                   "source_revision": os.environ.get(SOURCE_REVISION_ENV) or None},
        "model": {"id": name or None, "aliases": ids[1:], "model_type": _model_type(getattr(app, "model_dir", None)),
                  "context_length": context, "default_max_tokens": default_max,
                  "kv_dtype": kv_dtype, "mtp_drafts": depth if isinstance(depth, int) else None,
                  "vision": getattr(app, "vision", None) is not None,
                  "thinking_default": thinking if isinstance(thinking, bool) else None},
        "limits": {"lanes": lanes, "pending_requests": getattr(app, "max_pending_requests", None),
                   "engine_calls": limit, "http_connections": None,
                   "prefix_cache_entries": getattr(decoder, "keep", None) if decoder is not None else None},
    }


def _model_type(model_dir: Any) -> str | None:
    """The checkpoint's ``model_type`` (config.json), or None where it has none or cannot be read."""

    if model_dir is None:
        return None
    try:
        config = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError, TypeError):
        return None
    kind = config.get("model_type") if isinstance(config, dict) else None
    return kind if isinstance(kind, str) else None


__all__ = ["INSTANCE_ID", "ROUTES", "SCHEMA_VERSION", "SOURCE_REVISION_ENV", "STARTED_MONOTONIC", "STARTED_UNIX_S",
           "send", "snapshot"]

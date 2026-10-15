"""The finished-request counters both servers keep, and the Prometheus text of GET /metrics.

One store per app (``Metrics``) holds every counter and histogram of finished requests, and on CUDA the requests in
flight, under one lock: /health's totals, /metrics and /stats read the same numbers. A scrape copies them under the
lock and reads its gauges from ``stats.snapshot``, the same snapshot GET /stats returns; each gauge is taken on its own,
not as one atomic snapshot. vLLM-named families carry the same reading under vLLM's name and meaning, so a dashboard
copied from vLLM fills its panels by swapping the ``tensorfold:`` prefix for the name alone. A family the server cannot
count honestly is left out, never emitted at a fabricated zero.
"""

from __future__ import annotations

from contextlib import nullcontext
import copy
import threading
import time
from typing import Any

from tensorfold.server.memory_budget import process_footprint

PREFIX = "tensorfold:"
# Every histogram's upper edges, seconds: a request near a million tokens prefills for minutes. +Inf is added rendered.
BUCKETS = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1200.0, 2400.0, 4800.0)
# How a finished request ended: its finish_reason, "abort" (its client left) or "error" (it failed).
OUTCOMES = ("stop", "length", "tool_calls", "abort", "error")
_MADE = threading.Lock()
_local = threading.local()


class Histogram:
    """Counts in one bucket each. Render adds them up into Prometheus's cumulative buckets."""

    def __init__(self) -> None:
        self.counts = [0] * (len(BUCKETS) + 1)
        self.total = 0.0
        self.n = 0

    def observe(self, value: float) -> None:
        value = max(0.0, float(value))
        self.n += 1
        self.total += value
        for i, edge in enumerate(BUCKETS):
            if value <= edge:
                self.counts[i] += 1
                return
        self.counts[-1] += 1

    def copy(self) -> "Histogram":
        other = Histogram()
        other.counts = list(self.counts)
        other.total, other.n = self.total, self.n
        return other


HISTOGRAMS = ("latency", "ttft", "decode", "queue", "prefill", "tpot")


class Metrics:
    """Counters and histograms of finished requests, and the CUDA requests in flight (``live``, /health's
    ``Request``s); ``lock`` guards all of it. Gauges are not stored here."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.live: set = set()
        self.requests = 0
        self.prompt = 0
        self.cached = 0
        self.generation = 0
        self.drafted = 0
        self.accepted = 0
        self.rounds = 0
        self.queue_s = 0.0
        self.prefill_s = 0.0
        self.decode_s = 0.0
        self.outcomes = dict.fromkeys(OUTCOMES, 0)
        self.latency, self.ttft, self.decode, self.queue, self.prefill, self.tpot = (Histogram() for _ in HISTOGRAMS)
        self.http_requests: dict[tuple[str, int], int] = {}
        self.http_replies: dict[int, int] = {}
        self._http_labels: frozenset[str] | None = None
        self.fixed: dict[str, Any] | None = None      # /stats' static sections, made once (stats._fixed)

    def reconcile_http_labels(self, labels: frozenset[str]) -> None:
        """With both owner locks held, retire departed label series without changing lifetime totals."""
        if labels is self._http_labels:
            return
        self.http_requests = {pair: count for pair, count in self.http_requests.items()
                              if pair[0] == "unauthenticated" or pair[0] in labels}
        self._http_labels = labels

    def add(self, **values: Any) -> None:
        with self.lock:
            self.add_locked(**values)

    def add_locked(self, *, prompt: int, generation: int, drafted: int = 0, accepted: int = 0, cached: int = 0,
                   rounds: int = 0, latency: float | None = None, ttft: float | None = None,
                   decode: float | None = None, queue: float | None = None, prefill: float | None = None,
                   outcome: str | None = None) -> None:
        """Fold one finished request (the caller holds ``lock``). A None time is one the request did not have."""

        self.requests += 1
        self.prompt += int(prompt)
        self.generation += int(generation)
        self.drafted += int(drafted)
        self.accepted += int(accepted)
        self.cached += int(cached)
        self.rounds += int(rounds)
        if latency is not None:
            self.latency.observe(latency)
        if ttft is not None:
            self.ttft.observe(ttft)
        if decode is not None:
            self.decode_s += max(0.0, float(decode))
            self.decode.observe(decode)
            if generation > 1:                       # vLLM's time per output token: decode over the n-1 after the first
                self.tpot.observe(max(0.0, float(decode)) / (int(generation) - 1))
        if queue is not None:
            self.queue_s += max(0.0, float(queue))
            self.queue.observe(queue)
        if prefill is not None:
            self.prefill_s += max(0.0, float(prefill))
            self.prefill.observe(prefill)
        if outcome in self.outcomes:
            self.outcomes[outcome] += 1

    def reading(self) -> "Metrics":
        """A copy of every counter, with ``live_requests``/``live_tokens`` (the requests in flight and the tokens
        their replies hold so far), for one scrape; the caller holds ``lock``."""

        other = copy.copy(self)
        other.lock = None
        other.outcomes = dict(self.outcomes)
        for name in HISTOGRAMS:
            setattr(other, name, getattr(self, name).copy())
        other.http_requests, other.http_replies = dict(self.http_requests), dict(self.http_replies)
        other.live = frozenset()
        other.live_requests = len(self.live)
        other.live_tokens = sum(len(request.out) for request in self.live)
        return other


def of(app: Any) -> Metrics:
    """The app's counters, made on first use."""

    with _MADE:
        found = app.__dict__.get("metrics")
        if found is None:
            found = app.__dict__["metrics"] = Metrics()
        return found


def note(app: Any, **values: Any) -> None:
    """Fold one finished request (``Metrics.add_locked``'s fields). A missing app is a no-op."""

    if app is None:
        return
    of(app).add(**values)


def http_request(app: Any, key: str, status: int) -> None:
    """Count each gated HTTP reply globally, and by label while that label remains configured."""

    counters = of(app)
    with app.auth.metric_labels() as labels, counters.lock:
        counters.reconcile_http_labels(labels)
        status = int(status)
        counters.http_replies[status] = counters.http_replies.get(status, 0) + 1
        # A completion authenticated before rotation cannot resurrect a retired label.
        if key == "unauthenticated" or key in labels:
            pair = (key, status)
            counters.http_requests[pair] = counters.http_requests.get(pair, 0) + 1


def begin(app: Any, prompt: int, started: float) -> None:
    """The Mac request on this thread, from arrival, counted once when ``finish_request`` runs."""

    _local.armed = True
    _local.app = app
    _local.prompt = int(prompt)
    _local.generation = 0
    _local.started = float(started)
    _local.first = 0.0
    _local.job = None
    _local.ended = {}


def bind(job: Any) -> None:
    """The job whose stream holds this request's draft counts (a rerun replaces a preempted one)."""

    if getattr(_local, "armed", False):
        _local.job = job


def tokens(count: int, first: float) -> None:
    """Generated tokens so far, and the clock time of the first one."""

    if not getattr(_local, "armed", False):
        return
    _local.generation = int(count)
    if first and not _local.first:
        _local.first = float(first)


def ended(outcome: str, **values: Any) -> None:
    """How the Mac request on this thread ended (``OUTCOMES``), with what its reply counted (``cached``, ``rounds``,
    ``queue``, ``prefill``); kept for ``finish_request``."""

    if getattr(_local, "armed", False):
        _local.ended = {**_local.ended, "outcome": outcome, **values}


def finish_request() -> None:
    """Count the Mac request begun on this thread. Safe when none was begun."""

    if not getattr(_local, "armed", False):
        return
    _local.armed = False
    job, app, values = _local.job, _local.app, _local.ended
    # Persistent HTTP worker threads retain counters, not completed request/model owners.
    _local.job = _local.app = None
    _local.ended = {}
    stream = getattr(job, "stream", None) if job is not None else None
    finished = time.perf_counter()
    ttft = (_local.first - _local.started) if _local.first else None
    # Decode spans first generated token to completion; no generated token means no decode observation.
    decode = max(0.0, finished - _local.first) if _local.first else None
    note(app, prompt=_local.prompt, generation=_local.generation,
         drafted=int(getattr(stream, "drafted", 0) or 0),
         accepted=int(getattr(stream, "accepted", 0) or 0),
         latency=max(0.0, finished - _local.started), ttft=ttft, decode=decode, **values)


def capacity_snapshot(app: Any, *, server: Any = None) -> dict[str, dict[str, int]]:
    """Configured policy counts; omit a policy whose ownership is unavailable."""
    result = {}
    request_owner = getattr(app, "_request_limit", None)
    request_limit = getattr(app, "max_pending_requests", None)
    if request_owner is not None and request_limit is not None:
        result["requests"] = {"in_use": request_owner.used, "limit": request_limit}
    scheduler = getattr(app, "scheduler", None)
    engine_limit = getattr(scheduler, "max_engine_calls", None)
    if engine_limit is not None:
        result["engine_calls"] = {"in_use": scheduler.engine_calls, "limit": engine_limit}
    if getattr(server, "max_connections", None) is not None:
        result["http_connections"] = server.connection_snapshot()
    return result


def counters_now(app: Any) -> tuple[Metrics, dict[tuple[str, int], int], dict[int, int]]:
    """One scrape's copy of the app's counters, its key-label series reconciled with the configured keys."""

    counters = of(app)
    auth = getattr(app, "auth", None)
    labels_scope = auth.metric_labels() if auth is not None and auth.enabled else nullcontext()
    with labels_scope as labels, counters.lock:
        if labels is not None:
            counters.reconcile_http_labels(labels)
        reading = counters.reading()
    return reading, reading.http_requests, reading.http_replies


def render(app: Any, *, server: Any = None) -> str:
    """The scrape body, ending in a newline: Prometheus text of ``stats.snapshot`` and the histograms."""

    from tensorfold.server import stats

    reading, requests, replies = counters_now(app)
    snap = stats.snapshot(app, server=server, reading=reading)
    load, kv, prefix, totals = snap["load"], snap["kv"], snap["prefix_cache"], snap["totals"]
    lines: list[str] = []
    if load["http_connections_in_use"] is not None:
        _family(lines, "http_connections_in_use", "gauge", "Accepted HTTP handler owners, including retirement.",
                [f"{PREFIX}http_connections_in_use {load['http_connections_in_use']}"])
        if snap["limits"]["http_connections"] is not None:
            _family(lines, "http_connections_limit", "gauge", "Configured maximum accepted HTTP handler owners.",
                    [f"{PREFIX}http_connections_limit {snap['limits']['http_connections']}"])
    capacities = capacity_snapshot(app)
    for key, label, help_text in (("requests", "request_admission", "Unfinished logical requests"),
                                  ("engine_calls", "engine_calls", "Queued or running engine calls")):
        if key in capacities:
            values = capacities[key]
            for field in ("in_use", "limit"):
                name = label + "_" + field
                _family(lines, name, "gauge", help_text + (" in use." if field == "in_use" else " configured limit."),
                        [f"{PREFIX}{name} {values[field]}"])
    if requests:
        _family(lines, "requests_total", "counter", "HTTP replies by currently configured API key label and status.",
                [f'{PREFIX}requests_total{{key="{key}",status="{status}"}} {count}'
                 for (key, status), count in sorted(requests.items())])
    if replies:
        _family(lines, "http_replies_total", "counter", "Lifetime HTTP replies with API-key authentication configured.",
                [f'{PREFIX}http_replies_total{{status="{status}"}} {count}'
                 for status, count in sorted(replies.items())])
    running, waiting = load["requests_running"], load["requests_waiting"]
    _family(lines, "requests_running", "gauge", "Requests in prefill or decode.",
            [f"{PREFIX}requests_running {running}"])
    _family(lines, "requests_waiting", "gauge", "Requests queued or held until a lane is free.",
            [f"{PREFIX}requests_waiting {waiting}"])
    _family(lines, "prompt_tokens_total", "counter", "Prompt tokens of finished requests.",
            [f"{PREFIX}prompt_tokens_total {totals['prompt_tokens']}"])
    _family(lines, "generation_tokens_total", "counter", "Generated tokens of finished requests.",
            [f"{PREFIX}generation_tokens_total {totals['generated_tokens']}"])
    _family(lines, "kv_cache_usage_ratio", "gauge",
            "Tokens in a stream cache divided by that stream's context window.",
            [f'{PREFIX}kv_cache_usage_ratio{{pool="{pool}"}} {_num(ratio)}'
             for pool, ratio in _pools(load["stream_tokens"], _window(app))])
    _family(lines, "mtp_drafted_total", "counter", "Draft tokens verified on finished requests.",
            [f"{PREFIX}mtp_drafted_total {totals['drafted']}"])
    _family(lines, "mtp_accepted_total", "counter", "Draft tokens kept on finished requests.",
            [f"{PREFIX}mtp_accepted_total {totals['accepted']}"])
    _histogram(lines, "request_latency_seconds", "Seconds from arrival to the reply leaving.", reading.latency)
    _histogram(lines, "time_to_first_token_seconds", "Seconds from arrival to the first generated token.",
               reading.ttft)
    _histogram(lines, "request_decode_seconds",
               "Seconds a finished request spent decoding. Its sum over generation_tokens_total is the decode rate.",
               reading.decode)
    # the process's own footprint, where the platform counts it (macOS): weights, caches and streams as one number
    footprint = process_footprint()
    if footprint is not None:
        _family(lines, "process_footprint_bytes", "gauge",
                "This process's physical footprint as the OS counts it, Metal buffers included; "
                "only where the platform reports one (macOS).",
                [f"{PREFIX}process_footprint_bytes {footprint}"])
    # vLLM names and meanings: a vLLM dashboard needs only the "tensorfold:" prefix swapped.
    _family(lines, "num_requests_running", "gauge",
            "Requests in prefill or decode. A mirror of tensorfold:requests_running.",
            [f"{PREFIX}num_requests_running {running}"])
    _family(lines, "num_requests_waiting", "gauge",
            "Requests queued or held until a lane is free. A mirror of tensorfold:requests_waiting.",
            [f"{PREFIX}num_requests_waiting {waiting}"])
    if kv is not None and kv["usage_ratio"] is not None:
        _family(lines, "kv_cache_usage_perc", "gauge",
                "vLLM's KV-cache usage: cache memory the running requests hold over the stream caches' budget "
                "(kept prefix-cache entries count as free, as vLLM's cached blocks do; 1 is full, and one stream "
                "alone may pass it).", [f"{PREFIX}kv_cache_usage_perc {_num(kv['usage_ratio'])}"])
    _family(lines, "spec_decode_num_draft_tokens_total", "counter",
            "Draft tokens verified on finished requests, this server's single draft counter.",
            [f"{PREFIX}spec_decode_num_draft_tokens_total {totals['drafted']}"])
    _family(lines, "spec_decode_num_accepted_tokens_total", "counter", "Draft tokens kept on finished requests.",
            [f"{PREFIX}spec_decode_num_accepted_tokens_total {totals['accepted']}"])
    _histogram(lines, "e2e_request_latency_seconds",
               "Seconds from arrival to the reply leaving, under vLLM's name.", reading.latency)
    _histogram(lines, "request_decode_time_seconds",
               "Seconds a finished request spent decoding, under vLLM's name.", reading.decode)
    _histogram(lines, "request_queue_time_seconds",
               "Seconds a finished request waited for a lane (and, with --max-pending-requests, for admission).",
               reading.queue)
    _histogram(lines, "request_prefill_time_seconds",
               "Seconds a finished request's prompt took to prefill, as its engine counts them.", reading.prefill)
    _histogram(lines, "request_time_per_output_token_seconds",
               "Decode seconds over the tokens after the first, per finished request (vLLM's TPOT); drafted rounds "
               "commit several tokens at once.", reading.tpot)
    _family(lines, "request_success_total", "counter",
            "Finished requests by how they ended: their finish_reason, abort (the client left) or error.",
            [f'{PREFIX}request_success_total{{finished_reason="{reason}"}} {count}'
             for reason, count in reading.outcomes.items()])
    _family(lines, "prefix_cache_queries_total", "counter",
            "Prompt tokens of finished requests, vLLM's prefix-cache queries (counted at completion, not admission).",
            [f"{PREFIX}prefix_cache_queries_total {totals['prompt_tokens']}"])
    _family(lines, "prefix_cache_hits_total", "counter",
            "Prompt tokens finished requests found in the prefix cache, vLLM's prefix-cache hits (counted at "
            "completion).", [f"{PREFIX}prefix_cache_hits_total {totals['cached_prompt_tokens']}"])
    if prefix is not None:
        for name, key, help_text in (("prefix_cache_entries", "entries", "Kept prompt points the prefix cache holds."),
                                     ("prefix_cache_capacity", "capacity", "Kept prompt points the prefix cache may hold."),
                                     ("prefix_cache_tokens", "tokens", "Distinct prompt tokens the kept points' states hold.")):
            _family(lines, name, "gauge", help_text, [f"{PREFIX}{name} {prefix[key]}"])
    if kv is not None:
        for name, key, help_text in (
                ("kv_memory_budget_bytes", "budget_bytes", "Bytes the stream caches may grow into (less the reserve)."),
                ("kv_memory_held_bytes", "held_bytes", "Bytes the stream caches have grown into, kept entries included."),
                ("kv_memory_active_bytes", "active_bytes", "Grown cache bytes of the slots running requests use.")):
            _family(lines, name, "gauge", help_text, [f"{PREFIX}{name} {kv[key]}"])
    if load["lanes"] is not None:
        _family(lines, "lanes", "gauge", "Request lanes by phase: decoding, prefilling, or held out of rounds for memory.",
                [f'{PREFIX}lanes{{phase="{phase}"}} {count}' for phase, count in load["lanes"].items()
                 if count is not None])
    if snap["limits"]["lanes"] is not None:
        _family(lines, "lanes_max", "gauge", "Requests decoded together at most.",
                [f"{PREFIX}lanes_max {snap['limits']['lanes']}"])
    if load["held_for_memory"] is not None:
        _family(lines, "requests_held_for_memory", "gauge", "1 while a request waits, first in line, for memory.",
                [f"{PREFIX}requests_held_for_memory {int(load['held_for_memory'])}"])
    # per-request event counts; a family is left out where this server doesn't count the event, never a fake zero
    disconnects = _disconnects(app, reading)
    if disconnects is not None:
        _family(lines, "client_disconnections_total", "counter",
                "Requests a client left before the reply left the server.",
                [f"{PREFIX}client_disconnections_total {disconnects}"])
    if totals["preemptions"] is not None:
        _family(lines, "preemptions_total", "counter", "Requests that had to give a lane up to a later one.",
                [f"{PREFIX}preemptions_total {totals['preemptions']}"])
        _family(lines, "num_preemptions_total", "counter",
                "vLLM's preemptions: lanes given up to a later request, replayed later. A mirror of "
                "tensorfold:preemptions_total.", [f"{PREFIX}num_preemptions_total {totals['preemptions']}"])
    for name, key, help_text in (("memory_waits_total", "memory_waits", "Rounds a stream sat out waiting for memory."),
                                 ("memory_stops_total", "memory_stops", "Streams stopped because memory ran out.")):
        if totals[key] is not None:
            _family(lines, name, "counter", help_text, [f"{PREFIX}{name} {totals[key]}"])
    server_info = snap["server"]
    _family(lines, "process_start_time_seconds", "gauge", "This server process's start, Unix seconds.",
            [f"{PREFIX}process_start_time_seconds {_num(server_info['started_unix_s'])}"])
    model = snap["model"]
    info = {"version": server_info["version"], "backend": server_info["backend"], "instance": server_info["instance_id"],
            "source_revision": server_info["source_revision"], "model": model["id"], "kv_dtype": model["kv_dtype"],
            "mtp_drafts": model["mtp_drafts"], "lanes": snap["limits"]["lanes"]}
    _family(lines, "build_info", "gauge", "1, labelled with this server's build, model and configuration.",
            [f"{PREFIX}build_info{{" + ",".join(f'{k}="{_label(v)}"' for k, v in info.items() if v is not None)
             + "} 1"])
    return "\n".join(lines) + "\n"


def send(handler: Any, app: Any) -> None:
    """Write ``render`` as Prometheus text, version 0.0.4."""

    body = render(app, server=getattr(handler, "server", None)).encode()
    handler.send_response(200)
    handler.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
    handler.send_header("Content-Length", str(len(body)))
    handler.end_headers()
    try:
        handler.wfile.write(body)
    except (BrokenPipeError, ConnectionResetError):
        handler.close_connection = True


def _family(lines: list[str], name: str, kind: str, help_text: str, samples: list[str]) -> None:
    full = PREFIX + name
    lines.append(f"# HELP {full} {help_text}")
    lines.append(f"# TYPE {full} {kind}")
    lines.extend(samples)


def _histogram(lines: list[str], name: str, help_text: str, hist: Histogram) -> None:
    full = PREFIX + name
    lines.append(f"# HELP {full} {help_text}")
    lines.append(f"# TYPE {full} histogram")
    cumulative = 0
    for edge, count in zip(BUCKETS, hist.counts):
        cumulative += count
        lines.append(f'{full}_bucket{{le="{_edge(edge)}"}} {cumulative}')
    lines.append(f'{full}_bucket{{le="+Inf"}} {cumulative + hist.counts[-1]}')
    lines.append(f"{full}_sum {_num(hist.total)}")
    lines.append(f"{full}_count {hist.n}")


def requests_now(app: Any) -> tuple[int, int]:
    """(running, waiting). Prefilling Mac prompts are running and not yet in the active set."""

    scheduler = getattr(app, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "active") and hasattr(scheduler, "waiting"):
        filling = len(getattr(scheduler, "filling", None) or ())
        return int(scheduler.active) + filling, int(scheduler.waiting)
    engine = getattr(app, "engine", None)
    sched = getattr(engine, "scheduler", None) if engine is not None else None
    decoder = getattr(sched, "decoder", None) if sched is not None else None
    if decoder is not None and hasattr(decoder, "live"):
        live = decoder.live
        running = int(live() if callable(live) else live)
        waiting = 0
        queue = getattr(sched, "waiting", None)
        if queue is not None and hasattr(queue, "qsize"):
            waiting += int(queue.qsize())
        if getattr(sched, "held", None) is not None:
            waiting += 1
        return running, waiting
    health = getattr(app, "health", None)
    running = len(getattr(health, "live", ()) or ())
    turns = getattr(app, "turns", None)
    if turns is None:
        return running, 0
    parked = getattr(turns, "parked", None)
    return running, int(parked if parked is not None else getattr(turns, "waiting", 0) or 0)


def _disconnects(app: Any, reading: Metrics) -> int | None:
    """Requests whose client left: the Mac scheduler's own count, else (CUDA) the requests that ended in ``abort``."""

    scheduler = getattr(app, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "active") and hasattr(scheduler, "waiting"):
        return count(scheduler, "cancelled")
    return reading.outcomes["abort"] if getattr(app, "engine", None) is not None else None


def preemptions(app: Any) -> int | None:
    """Background requests that gave their lane up, where the server counts them (Mac jobs, CUDA lane yields)."""

    scheduler = getattr(app, "scheduler", None)
    if scheduler is not None and hasattr(scheduler, "active") and hasattr(scheduler, "waiting"):
        return count(scheduler, "preemptions")
    engine = getattr(app, "engine", None)
    sched = getattr(engine, "scheduler", None) if engine is not None else None
    return count(sched, "yields") if sched is not None else None


def count(owner: Any, name: str) -> int | None:
    """A scheduler's own count, or None where this server keeps no such count."""

    value = getattr(owner, name, None)
    return None if value is None else max(0, int(value))


def _pools(lengths: list[int], window: int) -> list[tuple[str, float]]:
    """One ratio per live stream. An idle server still publishes pool 0 at 0."""

    if not lengths:
        return [("0", 0.0)]
    if window <= 0:
        return [(str(i), 0.0) for i in range(len(lengths))]
    return [(str(i), min(1.0, n / window)) for i, n in enumerate(lengths)]


def _window(app: Any) -> int:
    engine = getattr(app, "engine", None)
    decoder = getattr(getattr(engine, "scheduler", None), "decoder", None) if engine is not None else None
    for owner, name in ((engine, "context_window"), (app, "context_window"), (decoder, "context")):
        n = _positive(getattr(owner, name, None) if owner is not None else None)
        if n:
            return n
    return 0


def stream_tokens(app: Any) -> list[int]:
    """Tokens each live stream's cache holds (decoding and prefilling), oldest first."""

    engine = getattr(app, "engine", None)
    decoder = getattr(getattr(engine, "scheduler", None), "decoder", None) if engine is not None else None
    if decoder is not None:
        streams = list(getattr(decoder, "streams", {}).values())
        streams += list(getattr(decoder, "filling", ()) or ())
        return [_occupied(stream) for stream in streams]
    live = getattr(engine, "_live", None) if engine is not None else None
    if not live:
        return []
    return [_occupied(stream) for stream, _cache in live if not getattr(stream, "finished", False)]


def _occupied(stream: Any) -> int:
    """Mac streams publish ``cache_len``. A CUDA stream's ``context`` gains the prompt when decode starts."""

    cache = getattr(stream, "cache_len", None)
    if isinstance(cache, int):
        return cache
    context = getattr(stream, "context", None) or ()
    if len(context):
        return len(context)
    prompt = getattr(stream, "prompt", None) or ()
    return len(prompt)


def _positive(value: Any) -> int:
    try:
        n = int(value)
    except (TypeError, ValueError):
        return 0
    return n if n > 0 else 0


def _num(value: float) -> str:
    if value == int(value):
        return str(int(value))
    return f"{value:.6f}".rstrip("0").rstrip(".")


def _edge(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def _label(value: Any) -> str:
    """A Prometheus label value: backslash, quote and newline escaped."""

    return str(value).replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


__all__ = ["BUCKETS", "OUTCOMES", "PREFIX", "Histogram", "Metrics", "begin", "bind", "capacity_snapshot", "count",
           "counters_now", "ended", "finish_request", "note", "of", "preemptions", "render", "requests_now", "send",
           "stream_tokens", "tokens"]

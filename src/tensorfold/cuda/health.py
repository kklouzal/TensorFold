"""The CUDA server's requests in flight and their folding into the app's one counter store (``server.metrics``), and
the /health body read from that store: finished requests' own engine stats plus live replies' tokens."""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from typing import Any

from tensorfold.server import metrics
from tensorfold.server.cancellation import RequestCancelled

_MADE = threading.Lock()


class Request:
    """One running request: its prompt length and the server's own list of its reply tokens (only ever read here).
    ``started`` is its arrival (the handler's clock), ``first`` its first generated token, ``outcome`` how it ended
    (``metrics.OUTCOMES``: the run sets its finish_reason; an exception makes it ``abort`` or ``error``)."""

    def __init__(self, prompt: int, out: list[int], arrived: float | None = None) -> None:
        self.prompt, self.out, self.stats = prompt, out, None
        self.started = time.perf_counter() if arrived is None else float(arrived)
        self.first: float | None = None
        self.outcome: str | None = None

    def saw(self) -> None:
        """The first generated token has landed in ``out``."""

        if self.first is None and self.out:
            self.first = time.perf_counter()


class Health:
    """The app's requests in flight; they and the totals live in its ``metrics.Metrics`` under its lock. The engine's
    rounds never call in here."""

    def __init__(self, counters: metrics.Metrics) -> None:
        self.counters = counters
        self.lock = counters.lock
        self.live = counters.live

    @contextmanager
    def running(self, prompt: int, out: list[int], arrived: float | None = None):
        """Count a request as running while its ``generate`` runs, then fold its reply and ``stats`` into the totals."""

        request = Request(prompt, out, arrived)
        with self.lock:
            self.live.add(request)
        try:
            yield request
        except RequestCancelled:
            request.outcome = "abort"
            raise
        except BaseException:
            request.outcome = "error"
            raise
        finally:
            fold = _fold(request)
            with self.lock:
                self.live.discard(request)
                self.counters.add_locked(**fold)

    def snapshot(self, app) -> dict[str, Any]:
        """The counters now: finished totals, live replies' tokens so far, and a concurrent engine's streams."""

        c = self.counters
        with self.lock:
            running = len(self.live)
            body: dict[str, Any] = {
                "requests_total": c.requests, "prompt_tokens_total": c.prompt,
                "completion_tokens_total": c.generation + sum(len(r.out) for r in self.live),
                "prefill_seconds_total": round(c.prefill_s, 6), "decode_seconds_total": round(c.decode_s, 6),
                "cached_tokens_total": c.cached, "rounds_total": c.rounds, "drafted_total": c.drafted,
                "accepted_total": c.accepted}
        body = {"ok": True, "backend": "tensorfold", "busy": running > 0, "requests_running": running, **body}
        scheduler = getattr(getattr(app, "engine", None), "scheduler", None)     # /health answers whatever the app
        decoder = getattr(scheduler, "decoder", None)
        if decoder is not None:                         # read, never locked: sizes of the decoder's own tables
            body["streams"] = {"decoding": len(getattr(decoder, "streams", ())),
                               "prefilling": len(getattr(decoder, "filling", ())), "max": scheduler.max_streams}
        window = getattr(app, "effective_context_window", None)
        if window:
            body["context_length"] = int(window)
        return body


def _fold(request: Request) -> dict[str, Any]:
    """A finished request's fields for ``Metrics.add_locked``: the engine's own decode time when it reports one (the
    figure /health sums), else first token to end."""

    stats = request.stats or {}
    ended = time.perf_counter()
    decode = _seconds(stats, "decode_s")
    if decode is None and request.first is not None:
        decode = ended - request.first
    return {"prompt": request.prompt, "generation": len(request.out), "drafted": _stat(stats, "drafted"),
            "accepted": _stat(stats, "accepted"), "cached": _stat(stats, "cached"), "rounds": _stat(stats, "rounds"),
            "latency": max(0.0, ended - request.started),
            "ttft": (request.first - request.started) if request.first is not None else None,
            "decode": None if decode is None else max(0.0, decode), "queue": _seconds(stats, "queue_s"),
            "prefill": _seconds(stats, "prefill_s"), "outcome": request.outcome}


def of(app) -> Health:
    """The app's requests in flight, made on first use over its counters."""

    with _MADE:
        found = app.__dict__.get("health")
        if found is None:
            found = app.__dict__["health"] = Health(metrics.of(app))
        return found


def _stat(stats: dict[str, Any], key: str) -> int:
    value = stats.get(key)
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else 0


def _seconds(stats: dict[str, Any], key: str) -> float | None:
    value = stats.get(key)
    return float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


__all__ = ["Health", "Request", "of"]

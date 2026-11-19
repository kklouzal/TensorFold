"""Bounded CPU reply ownership without blocking the GPU coordinator.

The already validated per-sequence reply allowance bounds token payloads and
nonempty chunks. Progress is one coalesced cell; termination has a distinct
reserved cell, so a full token backlog cannot strand completion or cleanup.
This is callback transport, not proof that network bytes reached the client.
"""

from __future__ import annotations

from collections import deque
import queue
import threading
import time
import weakref


class OutputMailbox:
    def __init__(self, maximum_tokens: int, *, pending_tokens: int | None = None):
        if type(maximum_tokens) is not int or maximum_tokens < 1:
            raise ValueError("output ownership requires a positive validated reply token allowance")
        self.maximum_tokens = maximum_tokens
        if pending_tokens is not None and (type(pending_tokens) is not int or pending_tokens < 1):
            raise ValueError("pending output token capacity must be an explicit positive integer")
        self.production_capacity = maximum_tokens if pending_tokens is None else min(maximum_tokens, pending_tokens)
        self._bounded_production = pending_tokens is not None
        self._changed = threading.Condition()
        self._chunks = deque()
        self._queued_tokens = self._published_tokens = self._dequeued_tokens = 0
        self._progress = None
        self._progress_after_tokens = 0
        self._has_progress = False
        self._terminal = None
        self._terminal_seen = False
        self._inflight = None
        self._reserved = 0
        self._reservation = None
        self._wake = None
        self._funded = None          # paid output ceiling (OutputFunding), None: the reply allowance alone

    def set_wake(self, callback):
        with self._changed:
            if not callable(callback) or self._wake is not None and self._wake is not callback:
                raise ValueError("output production must use one stable scheduler wake callback")
            self._wake = callback

    def reserve(self, count):
        """One legal model emission window, before that window issues GPU work."""
        if type(count) is not int or count < 1:
            raise ValueError("output production reservation needs a positive legal emission width")
        with self._changed:
            if self._reservation is not None:
                raise RuntimeError("previous output production window has not retired")
            if not self._fits(count):
                return None
            lease = _Production(self)
            self._reservation, self._reserved = lease, count
            return lease

    def _fits(self, count):
        inflight = 0 if self._inflight is None else self._inflight[1]
        funded = self.maximum_tokens if self._funded is None else min(self.maximum_tokens, self._funded)
        return count <= min(funded - self._published_tokens - self._reserved,
                            self.production_capacity - self._queued_tokens - inflight - self._reserved)

    def fund(self, tokens):
        """Raise the paid output ceiling (monotonic); a lane parked on it becomes producible."""
        if type(tokens) is not int or tokens < 0:
            raise ValueError("funded output must be a nonnegative token count")
        with self._changed:
            if self._funded is not None and tokens < self._funded:
                raise ValueError("funded output cannot shrink")
            self._funded = tokens
            wake = self._wake
        if wake is not None:
            wake()

    def available(self, count):
        """Nonblocking readmission eligibility; this does not acquire GPU/production credit."""
        with self._changed:
            return self._fits(count)

    def complete_delivery(self, value):
        """The actual calling-thread callback completed; dequeue alone is not output credit."""
        if not self._bounded_production:
            return
        with self._changed:
            if self._inflight is None or self._inflight[0] is not value:
                raise RuntimeError("output callback retirement lost its actual owned token chunk")
            self._inflight = None
            wake = self._wake
        if wake is not None:
            wake()

    def put(self, item, block=True, timeout=None):
        """Publish without waiting for a consumer, preserving token chunk order.

        The producer transfers ownership of each fresh token list and must not
        mutate it afterward. Duplicate terminal attempts retain the first owned
        outcome; the scheduler independently reports global retirement failures.
        """
        kind, value = item
        with self._changed:
            if kind == "tokens":
                if self._terminal is not None:
                    raise RuntimeError("tokens published after reply retirement")
                count = len(value)
                if not count:
                    return
                if self._published_tokens + count > self.maximum_tokens:
                    raise RuntimeError("committed output exceeded the validated reply token allowance")
                inflight = 0 if self._inflight is None else self._inflight[1]
                if count > self.production_capacity - self._queued_tokens - inflight:
                    raise RuntimeError("model output exceeded its reserved CPU production capacity")
                if self._reservation is not None:
                    if count > self._reserved:
                        raise RuntimeError("model output exceeded its legal reserved emission window")
                    self._reserved -= count
                node = (value, count)
                queued, published = self._queued_tokens, self._published_tokens
                try:
                    self._chunks.append(node)
                    self._queued_tokens = queued + count
                    self._published_tokens = published + count
                except BaseException:
                    # Publication can complete before interruption. The lock
                    # still excludes the consumer, so the exact immutable node
                    # proves whether its payload became visible; never invent
                    # free output credit or negative dequeue counters.
                    if self._chunks and self._chunks[-1] is node:
                        self._queued_tokens = queued + count
                        self._published_tokens = published + count
                        self._changed.notify()
                    raise
            elif kind == "filled":
                if self._terminal is not None:
                    raise RuntimeError("progress published after reply retirement")
                self._progress, self._has_progress = value, True
                self._progress_after_tokens = self._published_tokens
            elif kind in ("done", "error"):
                if self._terminal is None:
                    self._terminal = item
            else:
                raise ValueError("unsupported output mailbox message")
            self._changed.notify()

    def get(self, block=True, timeout=None):
        if timeout is not None and timeout < 0:
            raise ValueError("timeout must be nonnegative")
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._changed:
            while True:
                if self._has_progress and self._dequeued_tokens >= self._progress_after_tokens:
                    value, self._progress, self._has_progress = self._progress, None, False
                    return "filled", value
                if self._chunks:
                    if self._bounded_production and self._inflight is not None:
                        raise RuntimeError("the previous token callback has not completed")
                    value, count = self._chunks.popleft()
                    self._queued_tokens -= count
                    self._dequeued_tokens += count
                    if self._bounded_production:
                        self._inflight = (value, count)
                    return "tokens", value
                if self._terminal is not None and not self._terminal_seen:
                    self._terminal_seen = True
                    return self._terminal
                if not block:
                    raise queue.Empty
                remaining = None if deadline is None else deadline - time.monotonic()
                if remaining is not None and remaining <= 0:
                    raise queue.Empty
                self._changed.wait(remaining)

    def get_nowait(self):
        return self.get(block=False)

    def record(self):
        with self._changed:
            return {"maximum_token_ids": self.maximum_tokens, "published_token_ids": self._published_tokens,
                    "consumer_dequeued_token_ids": self._dequeued_tokens, "queued_token_ids": self._queued_tokens,
                    "queued_chunks": len(self._chunks), "progress_cells": int(self._has_progress),
                    "production_capacity_tokens": self.production_capacity,
                    "callback_inflight_tokens": 0 if self._inflight is None else self._inflight[1],
                    "reserved_production_tokens": self._reserved,
                    "terminal_cells": int(self._terminal is not None and not self._terminal_seen),
                    "scope": "engine_sequence",
                    "ownership": "CPU callback transport; network delivery is independent"}


class OutputFunding:
    """A request's paid output ceiling across its engine calls (budget/call continuations are new mailboxes).

    The HTTP owner pays host payload for ``tokens`` reply tokens before raising it; ``bind`` gives each new mailbox
    the remainder after the reply tokens that request already holds, so the scheduler never commits unpaid output.
    The mailbox is held weakly: it carries the request's host-payload borrow, which must retire with the mailbox
    (the funding lives on the payload lease, so a strong reference would keep that credit forever).
    """

    def __init__(self, tokens):
        if type(tokens) is not int or tokens < 1:
            raise ValueError("initial funded output must be a positive token count")
        self.tokens, self.base, self._box = tokens, 0, None
        self._lock = threading.Lock()

    def start(self, produced):
        """Before an engine call: the reply tokens the request already holds (its continuation's base)."""
        if type(produced) is not int or not 0 <= produced <= self.tokens:
            raise ValueError("a continuation cannot start past its funded output")
        with self._lock:
            self.base = produced

    def bind(self, box):
        with self._lock:
            self._box = weakref.ref(box)
            box.fund(self.tokens - self.base)

    def extend(self, tokens):
        with self._lock:
            if tokens < self.tokens:
                raise ValueError("funded output cannot shrink")
            self.tokens = tokens
            box = None if self._box is None else self._box()
            if box is not None:
                box.fund(tokens - self.base)


class _Production:
    def __init__(self, mailbox):
        self.mailbox = mailbox

    def close(self):
        mailbox = self.mailbox
        if mailbox is None:
            return
        with mailbox._changed:
            if mailbox._reservation is not self:
                raise RuntimeError("output production reservation changed owner before retirement")
            mailbox._reservation = None
            mailbox._reserved = 0
            self.mailbox = None

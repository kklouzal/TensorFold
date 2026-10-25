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


class OutputMailbox:
    def __init__(self, maximum_tokens: int):
        if type(maximum_tokens) is not int or maximum_tokens < 1:
            raise ValueError("output ownership requires a positive validated reply token allowance")
        self.maximum_tokens = maximum_tokens
        self._changed = threading.Condition()
        self._chunks = deque()
        self._queued_tokens = self._published_tokens = self._dequeued_tokens = 0
        self._progress = None
        self._progress_after_tokens = 0
        self._has_progress = False
        self._terminal = None
        self._terminal_seen = False

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
                    value, count = self._chunks.popleft()
                    self._queued_tokens -= count
                    self._dequeued_tokens += count
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
                    "terminal_cells": int(self._terminal is not None and not self._terminal_seen),
                    "scope": "engine_sequence",
                    "ownership": "CPU callback transport; network delivery is independent"}

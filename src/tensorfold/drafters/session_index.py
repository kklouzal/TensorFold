"""Session suffix drafts: exact earlier occurrences of a request's latest tokens, within its own prompt and reply.

A request's tokens (its prompt, then the reply as it is committed) are indexed by the k-gram (``ANCHOR`` tokens) that
ends at each position. A query takes the newest ``RECENT`` earlier occurrences of the sequence's last k-gram, measures
how far each matches backwards (up to ``LONGEST`` tokens) and picks the longest, then the newest; the proposal is
what followed it, repeating with the match's period where it runs into the sequence's end. Every candidate is
compared token by token, so a hash collision is never proposed. Drafts never change a reply (only how many of its
tokens a round verifies), so a proposal needs no particular quality to be exact: it only has to be cheap.

The prompt's grams are one sorted ``uint64`` array of packed keys (a gram hash's high bits, then its end position),
so the occurrences of a gram are one ascending run found by binary search. A long prompt's array is built off the
engine thread (``Builder``, at admission): until it is ready the request has no session drafts (its MTP drafts go
on). Reply grams go to a dict as they are committed.

Memory (host; ``tools/bench_draft_host.py``'s memory pass, Python 3.12): 12 bytes a prompt token (its id, 4, and its
key, 8: 12 MB at 1M tokens), a build's transient peak of about 28 bytes a prompt token (one build at a time on the
builder's thread), and up to about 250 bytes a reply token (a dict entry with its key tuple and end list, when every
reply gram is new). A stream's prompt and reply stay within its context (admission's clamp), so a decoder's indexes are
bounded by its slots x context. The decoder's ``MemoryGate`` does not book them, as it books no other per-request host
state: on a unified GPU (GB10) its live check reads MemAvailable, which they reduce, before any cache grows; 4 streams
of 1M-token prompts hold about 48 MB plus one ~28 MB build, against the startup's host reserve (``reserve_bytes``: at
least 4 GiB, a tenth of a GB10's memory) and the gate's own 2 GiB. A reply of a million new grams would add ~250 MB to
its stream. (The A/B baseline's ``CopyIndex``, built on the engine thread, is not booked either: 16 bytes a prompt token
and more a reply token.)

Design 4.2 S1 (``design-3-drafting``); the best match is the longest among the ``RECENT`` newest anchor occurrences,
not necessarily the longest anywhere (a suffix automaton's), by design.
"""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
import time
from typing import Sequence

import numpy as np

ANCHOR = 4          # tokens a match must share at least (the gram indexed at each position)
RECENT = 32         # the newest anchor occurrences compared at a query
LONGEST = 64        # the longest backward match measured
BULK = 4096         # prompts this long or longer build their sorted keys on the builder's thread
CHUNK = 16384       # prompt ids converted a step on the builder's thread (each holds the GIL ~0.5 ms, then yields it)
GROW = 4096         # token room added past the prompt, then a quarter more whenever the reply outgrows it
_MULT = 0x9E3779B97F4A7C15
_M64 = (1 << 64) - 1


def gram_hashes(tokens: np.ndarray, k: int) -> np.ndarray:
    """A wrapping 64-bit polynomial hash of each k-gram of ``tokens`` (index i: the gram ending at i + k - 1),
    multiplied once more at the end so its last token reaches the high bits a packed key keeps."""

    count = len(tokens) - k + 1
    h = np.zeros(max(0, count), dtype=np.uint64)
    if count <= 0:
        return h
    with np.errstate(over="ignore"):
        for j in range(k):
            h = h * np.uint64(_MULT) + tokens[j:j + count].astype(np.uint64) + np.uint64(1)
        h *= np.uint64(_MULT)
    return h


def _hash(gram: Sequence[int]) -> int:
    """``gram_hashes`` of one gram, in Python integers (the same value)."""

    h = 0
    for t in gram:
        h = (h * _MULT + int(t) + 1) & _M64
    return (h * _MULT) & _M64


def _ids(prompt: Sequence[int]) -> np.ndarray:
    """The prompt as int32 ids, converted in ``CHUNK`` steps. Converting Python ints holds the GIL; between steps it
    is handed over (``sleep(0)``), so an engine thread back from a GPU wait takes it within a step, not after the
    interpreter's switch interval. (Hashing and sorting below release the GIL themselves.)"""

    ids = np.empty(len(prompt), dtype=np.int32)
    for at in range(0, len(prompt), CHUNK):
        ids[at:at + CHUNK] = prompt[at:at + CHUNK]
        time.sleep(0)
    return ids


def build(prompt: Sequence[int], k: int = ANCHOR) -> tuple[np.ndarray, np.ndarray, int]:
    """The prompt's ids, its packed sorted gram keys, and the key's position bits (pure: any thread)."""

    ids = _ids(prompt)
    if np.any(ids < 0):
        raise ValueError("session drafts index non-negative token ids")
    bits = max(1, len(ids).bit_length())
    h = gram_hashes(ids, k)
    if len(h):
        keys = (h >> np.uint64(bits) << np.uint64(bits)) | np.arange(k - 1, len(ids), dtype=np.uint64)
        keys.sort()                                      # unique keys: ascending positions within a gram's run
    else:
        keys = h
    return ids, keys, bits


class Builder:
    """One background thread building long prompts' keys (a ``ThreadPoolExecutor``, started on first use): one build
    runs at a time and each admitted request queues at most one, which leaves the queue (``SessionIndex.cancel``)
    when the request does, so a decoder holds at most its slots' builds plus one running for a request that left. A
    job's result is handed over through its future (``SessionIndex.ready``: its lock publishes the arrays; the engine
    thread never waits on it). Its owner closes it (``MultiDecoder.close_drafting``), cancelling queued builds and
    joining the thread; a build submitted after that fails its request's admission."""

    def __init__(self) -> None:
        self.pool: ThreadPoolExecutor | None = None
        self.closed = False

    def submit(self, prompt: Sequence[int], k: int) -> Future:
        if self.closed:
            raise RuntimeError("the session index builder is closed")
        if self.pool is None:
            self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tensorfold-session-index")
        return self.pool.submit(build, prompt, k)

    def close(self) -> None:
        """Cancel queued builds and wait for a running one (it holds no lock the engine needs)."""

        self.closed = True
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            self.pool = None


class SessionIndex:
    """One request's session drafts. ``prompt`` is read on the builder's thread for a long prompt: the caller hands
    over a list it never mutates (a stream's prompt). ``sync`` appends committed reply tokens; ``propose`` drafts."""

    def __init__(self, prompt: Sequence[int], builder: Builder | None = None, *, k: int = ANCHOR,
                 recent: int = RECENT, longest: int = LONGEST) -> None:
        if k < 1 or recent < 1 or longest < k:
            raise ValueError("a session index needs anchor >= 1, recent >= 1 and longest >= anchor")
        self.k, self.recent, self.longest = k, recent, longest
        self.reply = 0                                   # reply tokens appended (once the prompt is in)
        self.backlog: list[int] = []                     # reply tokens committed before the prompt's keys are ready
        self.tokens = np.empty(0, dtype=np.int32)
        self.n = 0
        self.keys = np.empty(0, dtype=np.uint64)
        self.bits = 1
        self.grams: dict[tuple[int, ...], list[int]] = {}     # reply grams: their end positions, ascending
        self.last: list[int] = []                        # the sequence's last k tokens, as Python ints
        self.future: Future | None = None
        if builder is not None and len(prompt) >= BULK:
            self.future = builder.submit(prompt, k)
        else:
            self._adopt(build(prompt, k))

    # -- lifecycle ---------------------------------------------------------------------------------------------
    def ready(self) -> bool:
        """Whether the prompt's keys are in (adopting a finished build without waiting; its failure, or its
        cancellation by a closed builder, is raised here, in a round: ``MultiDecoder`` ends that request alone)."""

        if self.future is None:
            return True
        if not self.future.done():
            return False
        future, self.future = self.future, None
        self._adopt(future.result())
        return True

    def cancel(self) -> None:
        """The request left: a queued build is dropped (a running one finishes unread: its result, or its failure,
        goes with its future, as the request no longer drafts)."""

        if self.future is not None:
            self.future.cancel()
            self.future = None

    def _adopt(self, built: tuple[np.ndarray, np.ndarray, int]) -> None:
        ids, self.keys, self.bits = built
        self.tokens = np.empty(len(ids) + GROW, dtype=np.int32)
        self.tokens[:len(ids)] = ids
        self.n = len(ids)
        self.last = ids[-self.k:].tolist()
        backlog, self.backlog = self.backlog, []
        self._append(backlog)

    # -- tokens ------------------------------------------------------------------------------------------------
    def sync(self, out: Sequence[int], pending: Sequence[int] = ()) -> None:
        """Bring the index to the committed reply ``out`` plus ``pending`` (tokens committed but not yet in ``out``:
        they will be, or the request ends)."""

        if len(out) > self.reply:
            self._push(out[self.reply:])
        if pending:
            self._push(pending)
        if len(out) + len(pending) != self.reply:
            raise ValueError(f"a session index of {self.reply} reply tokens synced to {len(out) + len(pending)}")

    def _push(self, tokens: Sequence[int]) -> None:
        self.reply += len(tokens)
        if self.future is not None:
            self.backlog.extend(int(t) for t in tokens)
        else:
            self._append(tokens)

    def _append(self, tokens: Sequence[int]) -> None:
        tokens = [int(t) for t in tokens]
        if not tokens:
            return
        if min(tokens) < 0:
            raise ValueError("session drafts index non-negative token ids")
        need = self.n + len(tokens)
        if need > len(self.tokens):
            grown = np.empty(max(need, len(self.tokens) + len(self.tokens) // 4) + GROW, dtype=np.int32)
            grown[:self.n] = self.tokens[:self.n]
            self.tokens = grown
        self.tokens[self.n:need] = tokens
        k, last, grams = self.k, self.last, self.grams
        for end, t in enumerate(tokens, self.n):
            last.append(t)
            if len(last) > k:
                del last[0]
            if end >= k - 1:
                grams.setdefault(tuple(last), []).append(end)
        self.n = need

    # -- queries -----------------------------------------------------------------------------------------------
    def match(self) -> tuple[int, int]:
        """The best earlier occurrence of the sequence's end: (backward match length, its end position), or (0, -1).
        Only occurrences ending before the last token count; length >= the anchor."""

        n, k = self.n, self.k
        if self.future is not None or n < k + 1:
            return 0, -1
        t, needle = self.tokens, self.last
        newest = self.grams.get(tuple(needle), [])
        ends = newest[-self.recent - 1:]
        if ends and ends[-1] == n - 1:                   # the end's own gram
            ends = ends[:-1]
        ends = ends[-self.recent:]
        want = self.recent - len(ends)
        if want > 0 and len(self.keys):
            lo = _hash(needle) >> self.bits << self.bits
            # the gram's run: no key holds the all-ones position (the bits exceed every prompt position)
            a, b = self.keys.searchsorted(np.asarray([lo, lo | ((1 << self.bits) - 1)], dtype=np.uint64)).tolist()
            mask = np.uint64((1 << self.bits) - 1)
            older = (self.keys[max(a, b - want):b] & mask).astype(np.int64)
            older = older[older < n - 1]
            ends = older.tolist() + ends
        if not ends:
            return 0, -1
        e = np.asarray(ends, dtype=np.int64)
        depth = min(self.longest, n - 1)
        offsets = np.arange(depth, dtype=np.int64)
        idx = e[:, None] - offsets[None, :]
        same = (t[np.maximum(idx, 0)] == t[n - 1 - offsets][None, :]) & (idx >= 0)
        lengths = np.where(same.all(axis=1), depth, same.argmin(axis=1))
        best = len(lengths) - 1 - int(np.argmax(lengths[::-1]))      # the longest, then the newest
        length = int(lengths[best])
        return (length, int(e[best])) if length >= k else (0, -1)

    def propose(self, count: int) -> tuple[int, list[int]]:
        """Up to ``count`` drafts after the sequence's end: (match length, tokens), or (0, []) without a match. The
        continuation repeats with the match's period where it reaches the end (``count`` tokens whenever it
        matches)."""

        if count <= 0 or not self.ready():
            return 0, []
        length, e = self.match()
        if not length:
            return 0, []
        n, period = self.n, self.n - 1 - e
        at = e + 1 + np.arange(count, dtype=np.int64) % period
        return length, self.tokens[at].tolist()


__all__ = ["ANCHOR", "BULK", "Builder", "LONGEST", "RECENT", "SessionIndex", "build", "gram_hashes"]

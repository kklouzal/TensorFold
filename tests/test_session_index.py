"""Session suffix drafts (``tensorfold.drafters.session_index``, design 4.2 S1) against brute force and against the
offline replay's independent pure-Python S1 (``tools/draft_replay.py``), with its background build's lifecycle."""

from concurrent.futures import CancelledError
import importlib.util
from pathlib import Path
import random
import sys
import threading
import time

import numpy as np
import pytest

from tensorfold.drafters import session_index as si
from tensorfold.drafters.session_index import Builder, SessionIndex

TOOL = Path(__file__).resolve().parents[1] / "tools/draft_replay.py"
_spec = importlib.util.spec_from_file_location("draft_replay_oracle", TOOL)
replay = importlib.util.module_from_spec(_spec)
sys.modules["draft_replay_oracle"] = replay
_spec.loader.exec_module(replay)

ALPHABETS = (2, 3, 7, 50, 248_320)


def brute(seq, k, recent, longest):
    """The newest ``recent`` earlier occurrences of the last k-gram (ending before the last token), the longest
    backward match (at most ``longest``, never past the sequence's start), then the newest; (0, -1) under k."""

    n = len(seq)
    ends = [e for e in range(k - 1, n - 1) if seq[e - k + 1:e + 1] == seq[n - k:n]][-recent:]
    best = (0, -1)
    for e in ends:
        length = 0
        while length < longest and e - length >= 0 and seq[e - length] == seq[n - 1 - length]:
            length += 1
        if length >= best[0]:
            best = (length, e)
    return best if best[0] >= k else (0, -1)


def global_longest(seq, longest):
    """The longest backward match at any earlier end (the suffix automaton's answer), then the newest."""

    n, best = len(seq), (0, -1)
    for e in range(n - 1):
        length = 0
        while length < longest and e - length >= 0 and seq[e - length] == seq[n - 1 - length]:
            length += 1
        if length >= best[0] and length:
            best = (length, e)
    return best


def periodic(seq, e, count):
    n = len(seq)
    return [seq[e + 1 + j % (n - 1 - e)] for j in range(count)]


def _grow(rng, seq, alphabet):
    """A round's new tokens: half the time a copy of earlier text (so matches happen), else fresh ones."""

    if rng.random() < 0.5 and len(seq) > 8:
        at = rng.randrange(len(seq) - 4)
        return seq[at:at + rng.randint(1, 6)]
    return [rng.randrange(alphabet) for _ in range(rng.randint(1, 3))]


@pytest.mark.parametrize("alphabet", ALPHABETS)
def test_matches_and_proposals_equal_brute_force_after_every_round(alphabet):
    rng = random.Random(alphabet)
    for trial in range(3):
        prompt = [rng.randrange(alphabet) for _ in range(rng.randint(0, 120))]
        index = SessionIndex(list(prompt), k=4, recent=6, longest=12)
        seq, out = list(prompt), []
        for _ in range(150):
            new = _grow(rng, seq, alphabet)
            index.sync(out, new)                         # the round's tokens, before the reply takes them
            seq += new
            out += new
            got = index.match()
            assert got == brute(seq, 4, 6, 12), (alphabet, trial, len(seq))
            length, tokens = index.propose(7)
            assert length == got[0]
            assert tokens == (periodic(seq, got[1], 7) if length else [])
            assert all(0 <= t < max(alphabet, 1) for t in tokens)
        assert index.n == len(seq) and index.reply == len(out)


def test_s1_finds_the_globally_longest_match_when_it_lies_among_its_recent_anchors():
    rng = random.Random(5)
    checked = 0
    for alphabet in (3, 7, 50):
        seq = [rng.randrange(alphabet) for _ in range(40)]
        index = SessionIndex(list(seq), k=3, recent=8, longest=24)
        out = []
        for _ in range(200):
            new = _grow(rng, seq, alphabet)
            index.sync(out, new)
            seq += new
            out += new
            length, e = global_longest(seq, 24)
            k = 3
            recent = [x for x in range(k - 1, len(seq) - 1) if seq[x - k + 1:x + 1] == seq[-k:]][-8:]
            if length >= k and e in recent:
                assert index.match()[0] == length
                checked += 1
            assert index.match()[0] <= max(length, 0)
    assert checked > 50


@pytest.mark.parametrize("alphabet", ALPHABETS)
def test_the_runtime_index_equals_the_replays_independent_s1(alphabet):
    rng = random.Random(100 + alphabet)
    seq = [rng.randrange(alphabet) for _ in range(300)]
    while len(seq) < 900:
        seq += _grow(rng, seq, alphabet)
    oracle = replay.Index(4)
    oracle.extend(seq)
    index = SessionIndex(seq[:300], k=4, recent=32, longest=64)
    out = []
    for n in range(300, len(seq)):
        index.sync(out, seq[n:n + 1])
        out.append(seq[n])
        want = replay.match(oracle, seq, n + 1, n, 32, 64)
        assert index.match() == want, n
        if want[0]:
            assert index.propose(5)[1] == replay.continuation(seq, want[1], n + 1, 5)


def test_a_periodic_tail_continues_with_its_period_and_never_reads_past_the_end():
    index = SessionIndex([9, 8, 1, 2, 3, 1, 2, 3, 1, 2], k=3)
    length, tokens = index.propose(8)                    # "1 2 3" repeats up to the end: period 3
    assert length == 5 and tokens == [3, 1, 2, 3, 1, 2, 3, 1]     # "8 1 2 3 1 2" vs the end "3 1 2 3 1 2": 5
    assert SessionIndex([5, 6, 7, 8], k=2).propose(4) == (0, [])          # nothing repeats
    assert SessionIndex([1, 2, 1, 2], k=2).propose(0) == (0, [])          # no room
    assert SessionIndex([], k=2).propose(3) == (0, [])


def test_growth_by_rounds_equals_an_index_built_over_everything():
    rng = random.Random(9)
    seq = [rng.randrange(7) for _ in range(200)]
    index = SessionIndex(seq[:50], k=4, recent=16, longest=32)
    out = []
    at = 50
    while at < len(seq):
        new = seq[at:at + rng.randint(1, 5)]
        index.sync(out, new)
        out += new
        at += len(new)
    whole = SessionIndex(list(seq), k=4, recent=16, longest=32)
    assert index.match() == whole.match() and index.propose(6) == whole.propose(6)
    assert np.array_equal(index.tokens[:index.n], whole.tokens[:whole.n])


def test_a_long_prompt_builds_on_the_builders_thread_and_the_reply_catches_up(monkeypatch):
    monkeypatch.setattr(si, "BULK", 64)
    release, started = threading.Event(), threading.Event()
    real = si.build

    def slow(prompt, k):
        started.set()
        assert release.wait(10)
        return real(prompt, k)

    monkeypatch.setattr(si, "build", slow)
    rng = random.Random(3)
    prompt = [rng.randrange(20) for _ in range(500)]
    builder = Builder()
    try:
        index = SessionIndex(prompt, builder, k=4)
        assert started.wait(10)
        reply = prompt[100:130]                          # the reply quotes the prompt
        index.sync([], reply[:10])
        index.sync(reply[:10], reply[10:20])
        assert not index.ready() and index.propose(4) == (0, []) and index.backlog == reply[:20]
        release.set()
        index.future.result(10)
        assert index.ready() and not index.backlog
        assert index.propose(4) == SessionIndex(prompt + reply[:20], k=4).propose(4)
        index.sync(reply[:20], reply[20:30])
        assert index.match() == SessionIndex(prompt + reply, k=4).match()
    finally:
        release.set()
        builder.close()


def test_a_cancelled_queued_build_never_runs_and_a_closed_builder_takes_no_more(monkeypatch):
    monkeypatch.setattr(si, "BULK", 16)
    release, ran = threading.Event(), []
    real = si.build

    def slow(prompt, k):
        ran.append(len(prompt))
        assert release.wait(10)
        return real(prompt, k)

    monkeypatch.setattr(si, "build", slow)
    builder = Builder()
    first = SessionIndex(list(range(40)), builder)
    second = SessionIndex(list(range(50)), builder)       # queued behind the first
    second.cancel()
    assert second.future is None
    release.set()
    builder.close()                                      # waits for the running build
    assert ran == [40] and first.ready() and first.n == 40
    with pytest.raises(RuntimeError, match="closed"):
        SessionIndex(list(range(40)), builder)
    builder.close()                                      # again: nothing left


def test_a_failed_build_is_raised_where_the_index_is_adopted(monkeypatch):
    monkeypatch.setattr(si, "BULK", 8)

    def broken(prompt, k):
        raise MemoryError("no room for the keys")

    monkeypatch.setattr(si, "build", broken)
    builder = Builder()
    try:
        index = SessionIndex(list(range(20)), builder)
        index.future.exception(10)
        with pytest.raises(MemoryError, match="no room"):
            index.propose(3)
    finally:
        builder.close()


def test_closing_the_builder_cancels_queued_builds_and_their_indexes_raise_when_adopted(monkeypatch):
    """Shutdown with a build running and one queued: the running one finishes, the queued one never runs, and its
    index raises the cancellation where a round would adopt it (the decoder ends that request alone); one thread at
    most ever runs."""

    monkeypatch.setattr(si, "BULK", 16)
    release, started, running, peak = threading.Event(), threading.Event(), [0], [0]
    real = si.build

    def slow(prompt, k):
        running[0] += 1
        peak[0] = max(peak[0], running[0])
        started.set()
        assert release.wait(10)
        running[0] -= 1
        return real(prompt, k)

    monkeypatch.setattr(si, "build", slow)
    builder = Builder()
    first = SessionIndex(list(range(40)), builder)
    assert started.wait(10)
    queued = SessionIndex(list(range(50)), builder)
    closer = threading.Thread(target=builder.close)
    closer.start()
    deadline = time.monotonic() + 10
    while not queued.future.cancelled() and time.monotonic() < deadline:     # close cancels before it joins
        time.sleep(0.001)
    assert queued.future.cancelled()
    release.set()
    closer.join(10)
    assert not closer.is_alive() and first.ready() and first.n == 40 and peak[0] == 1
    with pytest.raises(CancelledError):
        queued.ready()


def test_ids_are_validated_and_a_resync_must_match():
    with pytest.raises(ValueError, match="non-negative"):
        SessionIndex([1, -2, 3])
    index = SessionIndex([1, 2, 3])
    with pytest.raises(ValueError, match="non-negative"):
        index.sync([], [-1])
    index = SessionIndex([1, 2, 3])
    index.sync([], [4, 5])
    index.sync([4, 5], [])                               # taken: nothing new
    with pytest.raises(ValueError, match="synced"):
        index.sync([4], [])                              # a reply shorter than what the index holds
    with pytest.raises(ValueError, match="anchor"):
        SessionIndex([1, 2], k=4, longest=2)


def test_memory_is_twelve_bytes_a_prompt_token_plus_the_reply_room():
    rng = random.Random(1)
    prompt = [rng.randrange(248_320) for _ in range(20_000)]
    index = SessionIndex(prompt, k=4)
    assert index.keys.dtype == np.uint64 and index.keys.nbytes == 8 * (len(prompt) - 3)
    assert index.tokens.dtype == np.int32 and index.tokens.nbytes == 4 * (len(prompt) + si.GROW)
    index.sync([], prompt[:si.GROW + 10])                # past the room: a quarter more
    assert len(index.tokens) >= index.n and len(index.tokens) <= index.n + len(prompt) // 4 + 2 * si.GROW

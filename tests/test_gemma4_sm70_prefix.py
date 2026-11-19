"""Gemma 4 SM70 prefix reuse (families/gemma4/sm70/prefix.py) against a recording fake engine: placement, kept
points, snapshot lifetime and the KV ledger. No GPU."""

from types import SimpleNamespace

import pytest

from tensorfold.cuda.memory_gate import NoRoom
from tensorfold.families.gemma4.sm70 import ops
from tensorfold.families.gemma4.sm70.prefix import PrefixOwner

TOKEN = 10            # ledger bytes per full-cache row
SNAP = 1000           # ledger bytes per snapshot


class FakeEngine:
    def __init__(self, budget, streams=1, keep_slots=2):
        self.streams, self.keep_slots, self.block, self.budget = streams, keep_slots, 3, budget
        self.target = SimpleNamespace(alloc=1 << 20, pool=SimpleNamespace(snapshot_bytes=lambda: SNAP))
        self.head = None
        self.token_bytes = TOKEN
        self.ops = []
        self.snaps = set()

    def slot_bytes(self, capacity):
        return capacity * TOKEN

    def announce(self, op, a=0, b=0, c=0, d=0, ids=()):
        self.ops.append((op, a, b, c))
        if op == ops.OP_SNAP:
            self.snaps.add(c)
        elif op == ops.OP_DROP:
            self.snaps.remove(a)
        elif op == ops.OP_RESTORE:
            assert b in self.snaps
        elif op == ops.OP_HSPILL:
            assert all(sid in self.snaps for sid in ids[1:1 + ids[0]])
        elif op == ops.OP_HRESTORE:
            self.snaps.add(ids[0])


class Owner(PrefixOwner):
    def __init__(self, engine, points=None, host=0):
        self.e, self.streams, self.filling, self.broken = engine, {}, [], None
        self._init_prefix(engine, points)
        if host:
            from tensorfold.families.gemma4.sm70.prefix_host import HostTier
            self.host_tier = HostTier(self, host)


def stream(prompt, count=8, draft=True):
    return SimpleNamespace(prompt=list(prompt), count=count, draft=draft, st=None, slot=None, cached=0, pos=0,
                           head=None, stops=[])


class Logits:
    """Stand-in for a device logits row: clone/host/device copies are itself."""

    def clone(self):
        return self

    def cpu(self):
        return self

    def to(self, *args, **kwargs):
        return self


LOGITS = Logits()


def fill(owner, s, logits=LOGITS):
    """Prefill the prompt in aligned 1,024-row chunks, keeping its points, as the Decoder's _fill does."""
    owner.filling.append(s)
    start = s.pos
    while start < len(s.prompt):
        stop = min(len(s.prompt), start + 1024)
        owner.keep_before(s, start)
        if stop == len(s.prompt) and s.keep_end:
            owner.keep_end(s, logits)
        start = stop
    owner.filling.remove(s)


def finish(owner, s):
    owner.retire_slot(s)


def ledger(owner):
    return sum(owner._bytes(st) for st in owner.slots) + SNAP * len(owner.snap_ids)


def test_extension_resumes_at_aligned_boundary_and_exact_repeat_uses_prompt_end():
    e = FakeEngine(budget=10**9)
    o = Owner(e)
    a = stream(range(2600))
    o.place(a, 4096, True)
    assert a.cached == 0 and a.stops == [2048]
    fill(o, a)
    assert o.held == ledger(o)
    finish(o, a)
    b = stream(list(range(2600)) + [7, 8, 9])       # extends a's exact end: resumes at the boundary below it
    o.place(b, 4096, True)
    assert b.cached == 2048 and b.head is None and o.prefix_counters["resume_tokens"] == 2048
    assert any(op[0] == ops.OP_RESTORE for op in e.ops)
    fill(o, b)
    finish(o, b)
    c = stream(list(range(2600)) + [7, 8, 9])       # an exact repeat: no prompt row, the kept logits
    o.place(c, 4096, True)
    assert c.cached == 2603 and c.head is LOGITS
    assert o.held == ledger(o)


def test_short_extension_without_boundary_starts_cold_in_place():
    e = FakeEngine(budget=10**9)
    o = Owner(e)
    a = stream(range(600))
    o.place(a, 1024, True)
    fill(o, a)
    finish(o, a)
    b = stream(list(range(600)) + [1])
    o.place(b, 1024, True)
    assert b.cached == 0 and e.ops[-1][0] == ops.OP_ATTACH      # rings emptied, rows rewritten identically


def test_message_points_and_no_reuse():
    e = FakeEngine(budget=10**9)
    o = Owner(e, points=lambda ids: [1300, 2500])
    a = stream(range(3000))
    o.place(a, 4096, True)
    assert a.stops == [1024, 2048]
    fill(o, a)
    finish(o, a)
    b = stream(list(range(2500)) + [5] * 400)       # diverges after the second message start
    o.place(b, 4096, True)
    assert b.cached == 2048
    finish(o, b)
    c = stream(range(3000))
    o.place(c, 4096, False)                          # reuse declined: cold, nothing kept
    assert c.cached == 0 and c.stops == [] and not c.keep_end


def test_eviction_frees_memory_and_snapshots():
    # Room for one 2048-row slot plus a few snapshots: a second conversation evicts the first idle chain.
    e = FakeEngine(budget=2048 * TOKEN + 3 * SNAP)
    o = Owner(e)
    a = stream(range(1600))
    o.place(a, 2048, True)
    fill(o, a)
    finish(o, a)
    b = stream([9] * 1600)
    o.place(b, 2048, True)
    assert b.cached == 0 and o.held <= e.budget and o.held == ledger(o)
    fill(o, b)
    finish(o, b)
    assert o.held == ledger(o) and len(e.snaps) == len(o.snap_ids)


def test_live_streams_holding_memory_wait():
    e = FakeEngine(budget=1024 * TOKEN + SNAP, streams=2)
    o = Owner(e)
    a = stream(range(600))
    o.place(a, 1024, True)
    o.filling.append(a)                              # live: its memory cannot be taken
    with pytest.raises(NoRoom):
        o.place(stream([3] * 600), 1024, True)
    assert o.held == ledger(o)


def test_host_tier_spills_evicted_chain_and_restores_it():
    e = FakeEngine(budget=2048 * TOKEN + 3 * SNAP)
    o = Owner(e, host=64 << 20)
    a = stream(range(1600))
    o.place(a, 2048, True)
    fill(o, a)
    finish(o, a)
    b = stream([9] * 1600)                           # evicts a's chain: spilled to the host tier
    o.place(b, 2048, True)
    fill(o, b)
    finish(o, b)
    assert o.host_tier.counters["spills"] == 1 and len(o.host_tier.entries) == 1
    c = stream(list(range(1600)) + [1, 2])           # extends a: its aligned point restored from host
    o.place(c, 2048, True)
    assert c.cached == 1024 and o.prefix_counters["host_hits"] == 1
    assert any(op[0] == ops.OP_HRESTORE for op in e.ops)
    assert o.held == ledger(o) and o.held <= e.budget
    fill(o, c)
    finish(o, c)
    d = stream(range(1600))                          # exact repeat of a: its exact end, from host
    o.place(d, 2048, True)
    assert d.cached == 1600 and d.head is LOGITS


def test_host_record_feeds_the_metrics_families():
    from tensorfold.server import metrics
    o = Owner(FakeEngine(budget=10**9), host=64 << 20)
    record = o.prefix_usage()["host"]
    for key in ("capacity_bytes_per_rank", "used_bytes_per_rank", "entries", "spills", "restores", "evictions"):
        assert key in record
    assert metrics is not None

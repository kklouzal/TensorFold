"""DeepSeek-V4.1 prompt-prefix reuse (families/deepseek_v4/cuda/prefix.py) on CPU: the rank-local slot-state copies
(snapshots in pool pages, forks, host row extents) and the owner's placement rules on a recording engine."""

from types import SimpleNamespace

import pytest

from tensorfold.families.gemma4.sm70.ops import OP_RESTORE

torch = pytest.importorskip("torch")


def _target(slots=4, pages=64, layers=3):
    from tensorfold.families.deepseek_v4.cuda.target import Pool
    ratios = [0] * 21
    for layer in (2, 8, 14):
        ratios[layer] = 2
    ratios[20] = 1
    geometry = SimpleNamespace(compress_ratios=ratios, kv_sources=(2, 8, 14, 20), layers=layers)
    pool = Pool(geometry, pages, slots, torch.device("cpu"))
    c2 = {layer: SimpleNamespace(partial=torch.zeros(slots, 2, 512), committed=torch.zeros(slots, dtype=torch.int64))
          for layer in (2, 8, 14)}
    history = torch.full((slots, 4), -1, dtype=torch.int64)
    return SimpleNamespace(pool=pool, c2=c2, hash_state=SimpleNamespace(history=history))


def _fill(target, slot, seed):
    g = torch.Generator().manual_seed(seed)
    for ring in target.pool.rings:
        ring[slot] = torch.randn(128, 512, generator=g).half()
    for state in target.c2.values():
        state.partial[slot] = torch.randn(2, 512, generator=g)
        state.committed[slot] = seed
    target.hash_state.history[slot] = torch.tensor([seed, seed + 1, seed + 2, seed + 3])


def _state(target, slot):
    return ([r[slot].clone() for r in target.pool.rings],
            [(s.partial[slot].clone(), s.committed[slot].clone()) for s in target.c2.values()],
            target.hash_state.history[slot].clone())


def _same(a, b):
    return (all(torch.equal(x, y) for x, y in zip(a[0], b[0]))
            and all(torch.equal(p, q) and torch.equal(c, d) for (p, c), (q, d) in zip(a[1], b[1]))
            and torch.equal(a[2], b[2]))


def test_snapshot_pages_roundtrip_and_fork():
    from tensorfold.families.deepseek_v4.cuda import prefix
    t = _target()
    state = prefix.SlotState(t)
    page = t.pool.layout.page_bytes
    assert t.pool.layout.bytes_per_token == 890
    assert state.raw == prefix.snapshot_bytes(3, 3) and state.pages == -(-state.raw // page)
    free = len(t.pool.free)
    _fill(t, 1, 7)
    want = _state(t, 1)
    state.run(prefix.OP_SNAP, 1, 640, 5, 0, [])
    assert len(t.pool.free) == free - state.pages          # snapshots are ledgered pool pages
    _fill(t, 1, 9)
    state.run(prefix.OP_RESTORE, 2, 5, 0, 0, [])           # into another slot: slot-independent values
    assert _same(_state(t, 2), want)
    # fork: KV pages [0, rows) of slot 1 into slot 2
    t.pool.attach(1, 200)
    t.pool.attach(2, 200)
    for p in t.pool.owned[1]:
        t.pool.cache[p] = torch.randint(0, 255, (page,), dtype=torch.uint8)
    state.run(prefix.OP_COPY, 1, 2, 130, 0, [])
    for a, b in zip(t.pool.owned[1][:3], t.pool.owned[2][:3]):
        assert torch.equal(t.pool.cache[a], t.pool.cache[b])
    state.run(prefix.OP_DROP, 5, 0, 0, 0, [])
    assert ("snap", 5) not in t.pool.owned
    with pytest.raises(RuntimeError):
        state.copy(1, 3, 64)                               # the destination holds no pages


def _rows(t, slot, first, end):
    """Every source's latent and index rows of positions [first, end) of ``slot`` (reference: plain loops)."""
    out = []
    layout = t.pool.layout
    for layer in sorted(layout.regions):
        ratio, per, c_off, i_off = layout.regions[layer]
        for off, size in ((c_off, 288), (i_off, 68)):
            for k in range(first // ratio, end // ratio):
                page = t.pool.owned[slot][k // per]
                at = off + (k % per) * size
                out.append(t.pool.cache[page, at:at + size].clone())
    return out


def test_host_extents_exact_bytes_and_chain_restore():
    from tensorfold.families.deepseek_v4.cuda import prefix
    from tensorfold.families.qwen4_exp.cuda.prefix_host import BLOCK_BYTES
    t = _target(pages=96)
    slab = torch.zeros(8 * BLOCK_BYTES, dtype=torch.uint8)
    state = prefix.SlotState(t, slab)
    t.pool.attach(0, 1000)
    for p in t.pool.owned[0]:
        t.pool.cache[p] = torch.randint(0, 255, (t.pool.layout.page_bytes,), dtype=torch.uint8)
    _fill(t, 0, 3)
    state.snap(0, 333, 1)
    # a parent extent [0, 333) and a child [333, 701): odd boundaries split ratio-2 groups
    state.run(prefix.OP_HSPILL, 10, 0, 333, 1, [0, 11, 0, 0])
    state.run(prefix.OP_HSPILL, 12, 0, 701, 3, [1, 1, 13, 333, 2])
    for first, end in ((0, 333), (333, 701), (5, 6), (6, 7)):
        assert prefix.extent_bytes(t.pool.layout, first, end) == sum(r.numel() for r in _rows(t, 0, first, end))
    assert prefix.extent_bytes(t.pool.layout, 0, 640) == 640 * 890
    want_rows, want_state = _rows(t, 0, 0, 701), _state(t, 0)
    t.pool.attach(3, 1000)
    t.pool.cache[t.pool.owned[3]] = 0
    # restore: the child's rows, then the parent's below the child's first row, and the entry's snapshot as sid 4
    state.run(prefix.OP_HRESTORE, 12, 3, 2, 1, [0, 4, 13, 333, 701, 11, 0, 333])
    got = _rows(t, 3, 0, 701)
    assert all(torch.equal(a, b) for a, b in zip(got, want_rows))
    assert state.restore(2, 4) == 333 and _same(_state(t, 2), want_state)


class _Engine:
    """The surface PrefixOwner reads, recording announcements and checking the page ledger against them."""

    def __init__(self, rows=64, pages=400):
        self.context, self.prompt_rows, self.streams, self.keep_slots = 4096, rows, 2, 2
        self.host_per_rank, self.host_slab, self.page_bytes = 0, None, 100
        self.snap_bytes = 3 * self.page_bytes
        self.budget = pages * self.page_bytes
        self.ops, self.snaps = [], {}

    def slot_bytes(self, capacity):
        return -(-capacity // 64) * self.page_bytes

    def announce(self, op, a=0, b=0, c=0, d=0, ids=()):
        from tensorfold.families.deepseek_v4.cuda import prefix
        self.ops.append((op, a, b, c))
        if op == prefix.OP_SNAP:
            self.snaps[c] = b
        elif op == prefix.OP_DROP:
            del self.snaps[a]


def _owner(rows=64):
    from tensorfold.families.deepseek_v4.cuda.prefix import PrefixOwner

    class Owner(PrefixOwner):
        broken = None

    o = Owner()
    o.e, o.streams, o.filling = _Engine(rows), {}, []
    o._init_prefix(o.e, None)
    return o


def _serve(o, prompt, reuse=True):
    """A request through admission, its chunks (keep_before / keep_end as the Decoder calls them) and retirement;
    returns (cached tokens, request) with ``s.chunks`` the chunk starts it computed."""
    from tensorfold.families.deepseek_v4.cuda.prefix import chunk_end
    s = SimpleNamespace(prompt=list(prompt), count=4, draft=True)
    s.bounds = o.bounds(s.prompt)
    o.place(s, o._cap(len(prompt) + 4), reuse)
    cached = s.cached
    s.chunks = []
    if s.head is None:
        o.filling.append(s)
        start = s.pos
        assert start == 0 or start in s.bounds           # every computed chunk is a cold chunk
        while start < len(prompt):
            s.chunks.append(start)
            o.keep_before(s, start)
            start = chunk_end(s.bounds, start, len(prompt))
        o.filling.remove(s)
        if s.keep_end:
            o.keep_end(s, torch.zeros(1, 8))
    o.retire_slot(s)
    return cached, s


def _restored(o):
    return o.e.snaps[[x for x in o.e.ops if x[0] == OP_RESTORE][-1][2]]


def test_owner_exact_repeat_extension_and_boundary_end():
    from tensorfold.families.deepseek_v4.cuda import prefix
    o = _owner()
    p = list(range(1, 201))                              # 200 tokens: chunks [0,64) [64,128) [128,192) [192,200)
    assert _serve(o, p)[0] == 0
    cached, s = _serve(o, p)
    assert cached == 200                                 # exact repeat: the kept end's state and logits
    assert _restored(o) == 200                           # (a fork: the chain keeps its end at 200 too)
    cached, _ = _serve(o, p + [7, 8, 9])                 # an extension resumes the boundary point below the end
    assert cached == 192 and _restored(o) == 192
    # n - 1 on a boundary: the exact end joins that boundary point (no resume of the state at n at n - 1)
    q = list(range(500, 629))                            # 129 tokens: n - 1 = 128
    _serve(o, q)
    cached, _ = _serve(o, q + [1])
    assert cached == 128
    assert _restored(o) == 128
    cached, _ = _serve(o, q + [1])                       # (the in-place extension superseded q's own end)
    assert cached == 130
    assert o.held == sum(o._bytes(st) for st in o.slots) + len(o.snap_ids) * o.snap_bytes
    assert set(o.e.snaps) == o.snap_ids


def test_owner_no_reuse_keeps_nothing():
    o = _owner()
    p = list(range(1, 150))
    _serve(o, p, reuse=False)
    assert not o.kept and not o.snap_ids
    assert _serve(o, p)[0] == 0


def test_chunk_bounds_rules_and_prefix_stability():
    import random
    from tensorfold.families.deepseek_v4.cuda.prefix import MIN_CHUNK, chunk_bounds, chunk_end
    assert chunk_bounds([], 100, 64) == [64]
    assert chunk_bounds([], 128, 64) == [64]                   # a boundary never ends the prompt
    assert chunk_bounds([1, 300, 400, 900], 1000, 8192) == [300, 900]   # closer than MIN_CHUNK: not a boundary
    assert chunk_bounds([300], 20000, 8192) == [300, 300 + 8192, 300 + 2 * 8192]
    assert chunk_end([300, 900], 0, 1000) == 300 and chunk_end([300, 900], 900, 1000) == 1000
    rng = random.Random(5)
    for _ in range(300):
        n2 = rng.randrange(2, 3000)
        pts = sorted(rng.sample(range(1, n2), min(n2 - 1, rng.randrange(0, 12))))
        n1 = rng.randrange(1, n2 + 1)
        a = chunk_bounds([p for p in pts if p < n1], n1, 512)
        b = chunk_bounds(pts, n2, 512)
        # a shorter prompt (its points: the longer one's below its end) has the longer one's boundaries below it
        assert a == [x for x in b if x < n1]
        assert all(y - x >= MIN_CHUNK or y - x == 512 for x, y in zip([0] + b, b))


U, A, T, E = 9000, 9001, 9002, 9003           # toy controls: message start, assistant, think token, EOS


def _toy_points(ids):
    return [p for p in range(1, len(ids)) if ids[p] == U or (p >= 2 and ids[p - 2] == A and ids[p - 1] == T)]


def _msg(seed, n):
    return [100 + (seed * 131 + i * 7) % 5000 for i in range(n)]


def _cold_chunks(prompt):
    o = _owner(rows=8192)
    o.points = _toy_points
    return _serve(o, prompt, reuse=False)[1].chunks


def test_owner_resumes_at_message_boundaries_bitwise_by_construction():
    o = _owner(rows=8192)
    o.e.context = 1 << 16
    o.points = _toy_points
    p1 = [1] + _msg(1, 300) + [U] + _msg(2, 300) + [A, T]                    # system, user, generation header
    n1 = len(p1)
    assert _serve(o, p1)[1].chunks == [0, 301] == _cold_chunks(p1)
    # an agent turn: the reply (reasoning, call) and a tool result extend the prompt; it resumes at its end
    p2 = p1 + _msg(3, 280) + [E, U] + _msg(4, 400) + [A, T]
    cached, s = _serve(o, p2)
    assert cached == n1 and _restored(o) == n1
    assert s.chunks == [c for c in _cold_chunks(p2) if c >= n1] == [n1, n1 + 281]
    # an exact repeat computes nothing; a regenerated last turn resumes at that message's start
    assert _serve(o, p2)[0] == len(p2)
    p3 = p2[:n1 + 282] + _msg(5, 500) + [A, T]
    cached, s = _serve(o, p3)
    assert cached == n1 + 281 and s.chunks == [c for c in _cold_chunks(p3) if c >= cached]
    # thinking without tools re-renders the previous header (</think>): the turn resumes at that user message
    p4 = p1[:n1 - 1] + [T + 10] + _msg(6, 300) + [E, U] + _msg(7, 300) + [A, T]
    cached, s = _serve(o, p4)
    assert cached == 301 and s.chunks == [c for c in _cold_chunks(p4) if c >= cached]
    # another conversation with the same system prompt resumes after it
    cached, s = _serve(o, [1] + _msg(1, 300) + [U] + _msg(9, 600) + [A, T])
    assert cached == 301
    assert o.held == sum(o._bytes(st) for st in o.slots) + len(o.snap_ids) * o.snap_bytes
    assert set(o.e.snaps) == o.snap_ids


def test_owner_lands_only_on_the_new_prompts_boundaries():
    """A kept point that is not a boundary of the new prompt (its next id is no message start) is not resumed:
    the request lands on the longest kept boundary below it, and the chain's points past that go."""
    o = _owner(rows=8192)
    o.e.context = 1 << 16
    o.points = _toy_points
    p1 = [1] + _msg(1, 300) + [U] + _msg(2, 300) + [U] + _msg(3, 300)        # raw prompt, ends mid-message
    _serve(o, p1)
    p2 = p1 + _msg(4, 50)                                                     # extends, but p1's end is no boundary
    cached, s = _serve(o, p2)
    assert cached == 602 and s.chunks == [c for c in _cold_chunks(p2) if c >= cached]
    assert all(len(k[0]) <= 602 or len(k[0]) >= len(p2) - 1 for k in o.kept)

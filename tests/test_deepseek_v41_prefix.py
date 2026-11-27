"""DeepSeek-V4.1 prompt-prefix reuse (families/deepseek_v4/cuda/prefix.py) on CPU: the rank-local slot-state copies
(snapshots in pool pages, forks, host row extents) and the owner's placement rules on a recording engine."""

from types import SimpleNamespace

import pytest

from tensorfold.families.gemma4.sm70.ops import OP_RESTORE

torch = pytest.importorskip("torch")


def _target(slots=4, pages=64, layers=3, *, rotorquant=False):
    from tensorfold.families.deepseek_v4.cuda.kernels.mu_packed import PackedLayout
    from tensorfold.families.deepseek_v4.cuda.target import BLOCK, Pool
    ratios = [0] * 21
    for layer in (2, 8, 14):
        ratios[layer] = 2
    ratios[20] = 1
    geometry = SimpleNamespace(compress_ratios=ratios, kv_sources=(2, 8, 14, 20), layers=layers)
    layout = PackedLayout({layer: ratios[layer] for layer in geometry.kv_sources}, BLOCK, rotorquant=rotorquant)
    pool = Pool(geometry, pages, slots, torch.device("cpu"), layout=layout)
    c2 = {layer: SimpleNamespace(partial=torch.zeros(slots, 2, 512), committed=torch.zeros(slots, dtype=torch.int64))
          for layer in (2, 8, 14)}
    history = torch.full((slots, 4), -1, dtype=torch.int64)
    return SimpleNamespace(pool=pool, c2=c2, hash_state=SimpleNamespace(history=history))


def test_native_full_windows_use_real_snapshot_geometry_and_exact_graph_buckets():
    from tensorfold.families.deepseek_v4.cuda.graphs import shape_keys
    from tensorfold.families.deepseek_v4.cuda.prefix import SlotState, full_window_plan

    target = _target(slots=2, pages=1, layers=40, rotorquant=True)
    snapshots = SlotState(target)
    ring_bytes = sum(ring[0].nbytes for ring in target.pool.rings)
    assert snapshots.raw == sum(part.nbytes for part in snapshots._parts(0)) == 1_404_984
    assert snapshots.pages == 26
    layout = target.pool.layout
    assert layout.page_bytes == 54_400 and layout.block_size == 64

    def plan(free, ceiling, keep=None):
        return full_window_plan(free, 1_048_576, ceiling, keep, layout, ring_bytes, len(target.c2),
                                prompt_reserve=3072 << 20, graph_fixed=128 << 20, graph_each=40 << 20)

    def fits(lanes, keep, graphs):
        return (((3072 + 128 + graphs * 40) << 20) + (lanes + keep + 1) * 1_392_640
                + (lanes * 16_384 + 26 * 6 * max(1, keep)) * 54_400)

    exact = fits(17, 17, 45)
    funded = plan(exact, 32)
    assert funded["funded_lanes"] == funded["keep_slots"] == 17
    assert funded["graph_keys"] == shape_keys(17, 1_048_576)
    assert (17, 1_048_576) in funded["graph_keys"]
    assert funded["pool_pages"] == 17 * 16_384 + 102 * 26
    assert plan(exact - 1, 32)["funded_lanes"] == 16

    explicit = plan(fits(1, 0, 1), 1, 0)
    assert (explicit["keep_slots"], explicit["stable_snapshot_points"], explicit["funded_lanes"]) == (0, 6, 1)
    assert plan(fits(1, 12, 1), 1, 12)["keep_slots"] == 12
    with pytest.raises(ValueError, match="cannot fund one"):
        plan(fits(1, 8, 1) - 1, 32)


def _fill(target, slot, seed):
    g = torch.Generator().manual_seed(seed)
    for ring in target.pool.rings:
        if ring.dtype == torch.uint8:
            ring[slot] = torch.randint(0, 256, ring.shape[1:], dtype=torch.uint8, generator=g)
        else:
            ring[slot] = torch.randn(ring.shape[1:], generator=g).half()
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


@pytest.mark.parametrize("rotorquant,row_bytes,token_bytes", [(False, 1024, 890), (True, 272, 850)])
def test_snapshot_pages_roundtrip_and_fork(rotorquant, row_bytes, token_bytes):
    from tensorfold.families.deepseek_v4.cuda import prefix
    from tensorfold.families.deepseek_v4.cuda.target import window_ring_bytes

    t = _target(rotorquant=rotorquant)
    state = prefix.SlotState(t)
    page = t.pool.layout.page_bytes
    assert t.pool.layout.bytes_per_token == token_bytes and page == token_bytes * 64
    assert window_ring_bytes(rotorquant) == 128 * row_bytes
    assert state.raw == 3 * 128 * row_bytes + 3 * (2 * 512 * 4 + 8) + 32
    assert state.pages == -(-state.raw // page)
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
    latent_bytes = 272 if layout.rotorquant else 288
    for layer in sorted(layout.regions):
        ratio, per, c_off, i_off = layout.regions[layer]
        for off, size in ((c_off, latent_bytes), (i_off, 68)):
            for k in range(first // ratio, end // ratio):
                page = t.pool.owned[slot][k // per]
                at = off + (k % per) * size
                out.append(t.pool.cache[page, at:at + size].clone())
    return out


@pytest.mark.parametrize("rotorquant,token_bytes", [(False, 890), (True, 850)])
def test_host_extents_exact_bytes_and_chain_restore(rotorquant, token_bytes):
    from tensorfold.families.deepseek_v4.cuda import prefix
    from tensorfold.families.qwen4_exp.cuda.prefix_host import BLOCK_BYTES
    t = _target(pages=96, rotorquant=rotorquant)
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
    assert prefix.extent_bytes(t.pool.layout, 0, 640) == 640 * token_bytes
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
        self.ids = ids
        if op == prefix.OP_SNAP:
            self.snaps[c] = b
        elif op == prefix.OP_HRESTORE:                    # (entry snapshot index, new snapshot id) pairs
            for sid in list(ids)[1:2 * d:2]:
                self.snaps[sid] = "host"
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


@pytest.mark.parametrize("failure,message", [
    ("stream", "idle prefix owner"), ("filling", "idle prefix owner"),
    ("duplicate", "inconsistent prefix slot ownership"), ("bytes", "inconsistent retained bytes"),
    ("snapshot", "unowned retained prefix"),
])
def test_placement_retirement_refuses_invalid_prefix_ownership(failure, message):
    owner = _owner()
    if failure == "stream":
        owner.streams[1] = SimpleNamespace(st=owner.free[-1])
    elif failure == "filling":
        owner.filling.append(SimpleNamespace(st=owner.free[-1]))
    elif failure == "duplicate":
        owner.free.append(owner.free[-1])
    elif failure == "bytes":
        owner.held = 1
    else:
        slot = owner.free.pop()
        slot.capacity = 64
        owner.kept.append(([1, 2], slot, {"sid": 37}, None))
    before = (list(owner.free), list(owner.kept), owner.held)
    with pytest.raises(RuntimeError, match=message):
        owner.retire_placement_prefixes()
    assert (owner.free, owner.kept, owner.held) == before and owner.e.ops == []


def test_placement_retirement_releases_gpu_and_host_prefix_owners():
    from tensorfold.families.deepseek_v4.cuda import prefix
    from tensorfold.families.gemma4.sm70.ops import OP_ATTACH, OP_DETACH
    from tensorfold.families.qwen4_exp.cuda.prefix_host import BLOCK_BYTES

    target = _target(pages=96, rotorquant=True)
    slab = torch.zeros(4 * BLOCK_BYTES, dtype=torch.uint8)
    rank = prefix.SlotState(target, slab)

    class Engine(_Engine):
        def __init__(self):
            super().__init__()
            self.layout = target.pool.layout
            self.host_per_rank, self.host_slab = slab.numel(), slab
            self.snap_bytes, self.budget = rank.snap_bytes, target.pool.pages * self.layout.page_bytes
            self.page_bytes = self.layout.page_bytes

        def announce(self, op, a=0, b=0, c=0, d=0, ids=()):
            super().announce(op, a, b, c, d, ids)
            if op == OP_ATTACH:
                target.pool.attach(a, b)
            elif op == OP_DETACH:
                target.pool.detach(a)
            else:
                return rank.run(op, a, b, c, d, ids)

    owner = _owner()
    owner.e = Engine()
    owner._init_prefix(owner.e, None)
    kept = owner.free.pop()
    attached_free = owner.free[-1]
    assert owner._grow(kept, 256) and owner._grow(attached_free, 64)
    for page in target.pool.owned[kept.index]:
        target.pool.cache[page].fill_(17)
    _fill(target, kept.index, 3)
    first = owner._snapshot(kept, 128)
    owner.kept.append((list(range(128)), kept, {"sid": first, "pos": 128}, None))
    tier = owner.host_tier
    tier.spill(owner.kept, None)
    later = owner._snapshot(kept, 256)
    owner.kept.append((list(range(256)), kept, {"sid": later, "pos": 256}, None))
    tier.spill(owner.kept, None)
    assert len(tier.entries) == 2 and tier.extents == 2 and any(tier.taken)
    assert rank.host_layout and rank.host_extents and kept.host_rows is not None
    assert owner.held > 0 and rank.snaps and target.pool.owned
    counters = dict(tier.counters)
    ids = (owner.next_snap, tier.next_id)
    before_slab = slab.clone()
    before_pool = target.pool.cache.clone()
    with pytest.raises(RuntimeError, match="rank-local KV borrowers"):
        rank.retire_placement_prefixes()
    assert rank.host_layout and rank.host_extents

    offset = len(owner.e.ops)
    owner.retire_placement_prefixes()
    assert owner.e.ops[offset:] == [
        (prefix.OP_DROP, first, 0, 0), (prefix.OP_DROP, later, 0, 0),
        (OP_DETACH, kept.index, 0, 0), (OP_DETACH, attached_free.index, 0, 0),
    ]
    assert owner.free == list(reversed(owner.slots)) and not owner.kept and not owner.turn_of and owner.held == 0
    assert all(slot.capacity == 0 and slot.host_rows is None for slot in owner.slots)
    assert not tier.entries and not tier._gone and tier.extents == 0 and not any(tier.taken)
    assert owner.host_tier is tier and tier.counters == counters and (owner.next_snap, tier.next_id) == ids
    assert not rank.snaps and not target.pool.owned and len(target.pool.free) == target.pool.pages
    assert rank.host_layout and rank.host_extents
    rank.retire_placement_prefixes()
    assert not rank.host_layout and not rank.host_extents
    assert rank.slab is slab and torch.equal(slab, before_slab)
    assert torch.equal(target.pool.cache, before_pool)


class _Head:
    """A last-row logits stand-in whose device/host copies are itself."""

    def clone(self):
        return self

    def to(self, *args, **kwargs):
        return self

    def numel(self):
        return 1 << 18

    def element_size(self):
        return 4


def _serve(o, prompt, reuse=True):
    """A request through admission, its chunks (keep_before / keep_end as the Decoder calls them) and retirement;
    returns (cached tokens, request) with ``s.chunks`` the chunk starts it computed."""
    from tensorfold.families.deepseek_v4.cuda.prefix import chunk_end
    s = SimpleNamespace(prompt=list(prompt), count=4, draft=True)
    o.place(s, o._cap(len(prompt) + 4), reuse)
    cached = s.cached
    s.chunks = []
    if s.head is None:
        o.filling.append(s)
        start = s.pos
        assert all(s.cached < b < len(prompt) for b in s.bounds)
        while start < len(prompt):
            s.chunks.append(start)
            o.keep_before(s, start)
            start = chunk_end(s.bounds, start, len(prompt))
        o.filling.remove(s)
        if s.keep_end:
            o.keep_end(s, _Head())
    o.retire_slot(s)
    return cached, s


def _restored(o):
    return o.e.snaps[[x for x in o.e.ops if x[0] == OP_RESTORE][-1][2]]


def test_owner_exact_repeat_extension_and_boundary_end():
    o = _owner()
    o.keep = 64                                          # placement only: no entry evicted
    p = list(range(1, 201))                              # 200 tokens: chunks [0,64) [64,128) [128,192) [192,200)
    assert _serve(o, p)[0] == 0
    cached, s = _serve(o, p)
    assert cached == 200                                 # exact repeat: the kept end's state and logits
    assert _restored(o) == 200                           # (a fork: the chain keeps its end at 200 too)
    cached, _ = _serve(o, p + [7, 8, 9])                 # an extension resumes the exact end's state
    assert cached == 200 and _restored(o) == 200
    assert not any(len(k[0]) % 64 == 0 for k in o.kept)  # the 64-row cap's splits keep no point
    assert o.held == sum(o._bytes(st) for st in o.slots) + len(o.snap_ids) * o.snap_bytes
    # n - 1 on a kept (message) split: that point serves the exact repeat (no resume of the state at n at n - 1)
    o = _owner(rows=8192)
    o.keep, o.points = 64, _toy_points
    q = list(range(500, 800)) + [U]                      # 301 tokens: a message point at n - 1 = 300
    _serve(o, q)
    cached, _ = _serve(o, q[:300] + [3, 4])              # the split at 300 resumes
    assert cached == 300 and _restored(o) == 300
    _serve(o, q)
    cached, _ = _serve(o, q + [1])
    assert cached == 301
    cached, _ = _serve(o, q + [1])                       # its own end (q's was resumed without its logits)
    assert cached == 302
    assert o.held == sum(o._bytes(st) for st in o.slots) + len(o.snap_ids) * o.snap_bytes
    assert set(o.e.snaps) == o.snap_ids


def test_owner_no_reuse_keeps_nothing():
    o = _owner()
    p = list(range(1, 150))
    _serve(o, p, reuse=False)
    assert not o.kept and not o.snap_ids
    assert _serve(o, p)[0] == 0


@pytest.mark.parametrize("outcome", ("text-finished", "vision-finished", "vision-cancelled"))
def test_decoder_finish_releases_uncached_slot(outcome):
    from tensorfold.families.deepseek_v4.cuda.engine import Decoder, DS41Engine, OP_ATTACH, OP_DETACH

    engine = _Engine(pages=16)
    engine.target = _target(pages=16)
    engine.target.reset_slot = lambda slot: None
    pool = engine.target.pool
    engine.page_bytes = pool.layout.page_bytes
    engine.budget = pool.pages * engine.page_bytes
    engine.token_bytes = pool.token_bytes()
    slot_pages, tower_frees = {}, []
    engine.slot_pages = SimpleNamespace(set=lambda slot, pages: slot_pages.__setitem__(slot, tuple(pages)))
    engine.tower_free = lambda handles: tower_frees.append(tuple(handles))
    record = engine.announce

    def announce(op, slot, capacity=0):
        record(op, slot, capacity)
        if op == OP_ATTACH:
            DS41Engine._attach(engine, slot, capacity)
        else:
            assert op == OP_DETACH
            DS41Engine._detach(engine, slot)

    engine.announce = announce
    decoder = object.__new__(Decoder)
    decoder.e, decoder.streams, decoder.filling, decoder._broken = engine, {}, [], None
    decoder.split = decoder.finishing = None
    decoder._init_prefix(engine, None)
    vision = outcome.startswith("vision")
    stream = SimpleNamespace(sid=7, prompt=list(range(1, 150)), count=4, draft=False,
                             vision=object() if vision else None, encoded={0: 11} if vision else {},
                             encoding=(1, 13) if vision else None)
    capacity = decoder._cap(len(stream.prompt) + stream.count)
    decoder.place(stream, capacity, reuse=False)
    state, slot = stream.st, stream.slot
    assert stream.cached == 0 and state.capacity == capacity
    assert decoder.held == engine.slot_bytes(capacity) > 0
    assert slot_pages[slot] == tuple(pool.owned[slot]) and pool.owned[slot]
    if vision:
        decoder.filling.append(stream)
    else:
        decoder.streams[stream.sid] = stream
    if outcome == "vision-cancelled":
        assert decoder.cancel([stream]) == [stream]
        assert stream.done and stream.st is state and slot in pool.owned
        assert tower_frees == [(11, 13)]

    decoder.finish([stream])

    assert decoder.live() == 0 and decoder.broken is None
    assert stream.st is None and stream.slot is None
    assert state.capacity == decoder.held == 0
    assert tower_frees == ([(11, 13)] if vision else [])
    assert not stream.encoded and stream.encoding is None
    assert list(engine.ops) == [(OP_ATTACH, slot, capacity, 0), (OP_DETACH, slot, 0, 0)]
    assert not pool.owned and len(pool.free) == pool.pages and slot_pages[slot] == ()
    assert sum(free is state for free in decoder.free) == 1
    assert decoder.usage()["kv"]["free_tokens_estimate"] == pool.pages * 64


def test_chunk_splits_rules():
    from tensorfold.families.deepseek_v4.cuda.prefix import chunk_end, chunk_splits
    assert chunk_splits([], 0, 100, 64) == [64]
    assert chunk_splits([], 0, 128, 64) == [64]                 # a split never ends the prompt
    assert chunk_splits([300, 400, 900], 0, 1000, 8192) == [300, 900]          # the first and the last point
    assert chunk_splits([1, 300, 900], 0, 1000, 8192) == [300, 900]   # (a point at the prompt's start: no split)
    assert chunk_splits([1, 900], 0, 1000, 8192) == [900]
    assert chunk_splits([300, 400, 900], 400, 1000, 8192) == [900]             # resumed: the last point only
    assert chunk_splits([300], 0, 20000, 8192) == [300, 300 + 8192, 300 + 2 * 8192]
    assert chunk_splits([300, 900], 950, 20000, 8192) == [950 + 8192, 950 + 2 * 8192]
    assert chunk_end([300, 900], 0, 1000) == 300 and chunk_end([300, 900], 900, 1000) == 1000


U, A, T, E = 9000, 9001, 9002, 9003           # toy controls: message start, assistant, think token, EOS


def _toy_points(ids):
    return [p for p in range(1, len(ids)) if ids[p] == U or (p >= 2 and ids[p - 2] == A and ids[p - 1] == T)]


def _msg(seed, n):
    return [100 + (seed * 131 + i * 7) % 5000 for i in range(n)]


def _cold_chunks(prompt):
    o = _owner(rows=8192)
    o.points = _toy_points
    return _serve(o, prompt, reuse=False)[1].chunks


def test_owner_resumes_at_message_boundaries():
    o = _owner(rows=8192)
    o.e.context = 1 << 16
    o.points = _toy_points
    p1 = [1] + _msg(1, 300) + [U] + _msg(2, 300) + [A, T]                    # system, user, generation header
    n1 = len(p1)
    assert _serve(o, p1)[1].chunks == [0, 301] == _cold_chunks(p1)          # split at the system prompt's end
    # an agent turn: the reply (reasoning, call) and a tool result extend the prompt; it resumes at its end
    p2 = p1 + _msg(3, 280) + [E, U] + _msg(4, 400) + [A, T]
    cached, s = _serve(o, p2)
    assert cached == n1 and _restored(o) == n1
    assert s.chunks == [n1, n1 + 281]                                        # and at the tool result's start
    assert _cold_chunks(p2) == [0, 301, n1 + 281]
    # an exact repeat computes nothing; a regenerated last turn resumes at that message's start
    assert _serve(o, p2)[0] == len(p2)
    p3 = p2[:n1 + 282] + _msg(5, 500) + [A, T]
    cached, s = _serve(o, p3)
    assert cached == n1 + 281 and s.chunks == [cached]
    # thinking without tools re-renders the previous header (</think>): the turn resumes at that user message
    p4 = p1[:n1 - 1] + [T + 10] + _msg(6, 300) + [E, U] + _msg(7, 300) + [A, T]
    cached, s = _serve(o, p4)
    assert cached == 301 and s.chunks == [301, 905]
    # another conversation with the same system prompt resumes after it
    cached, s = _serve(o, [1] + _msg(1, 300) + [U] + _msg(9, 600) + [A, T])
    assert cached == 301
    assert o.held == sum(o._bytes(st) for st in o.slots) + len(o.snap_ids) * o.snap_bytes
    assert set(o.e.snaps) == o.snap_ids


def test_owner_extension_of_an_exact_end_and_of_its_ids_less_one():
    """A prompt extending a kept end resumes its state; one sharing all but its last id (a re-rendered header)
    lands on the longest kept point below, and the chain's points past it go."""
    o = _owner(rows=8192)
    o.e.context = 1 << 16
    o.points = _toy_points
    p1 = [1] + _msg(1, 300) + [U] + _msg(2, 300) + [U] + _msg(3, 300)        # raw prompt, ends mid-message
    _serve(o, p1)
    assert _serve(o, p1 + _msg(4, 50))[0] == len(p1)
    p3 = p1[:-1] + _msg(5, 80)
    cached, s = _serve(o, p3)
    assert cached == 602 and s.chunks == [602]
    assert all(len(k[0]) <= 602 or k[0][:len(p3) - 80] == p3[:len(p3) - 80] for k in o.kept)


def _host_owner(mib=64):
    """An owner with the DeepSeek host tier over a real page layout (slab copies are the ranks' work: stubbed)."""
    from tensorfold.families.deepseek_v4.cuda.prefix import PrefixOwner

    class Owner(PrefixOwner):
        broken = None

    o = Owner()
    o.e, o.streams, o.filling = _Engine(8192, pages=4000), {}, []
    o.e.context = 1 << 16
    o.e.layout = _target().pool.layout
    o.e.host_per_rank = mib << 20
    o.e.host_slab = torch.zeros(mib << 20, dtype=torch.uint8)
    o._init_prefix(o.e, _toy_points)
    return o


def test_owner_extension_spill_restore_and_dead_exact_points(monkeypatch):
    """A conversation's chain is extended (in place), superseded by others until it spills to the host tier, then
    continued from host RAM on a boundary; exact points whose end a resume removed never stay kept or spill."""
    from tensorfold.families.gemma4.sm70 import prefix_host
    monkeypatch.setattr(prefix_host, "_to_slab", lambda slab, offset, tensor: tensor)   # the ranks' copies
    o = _host_owner()
    p1 = [1] + _msg(1, 300) + [U] + _msg(2, 300) + [A, T]
    n1 = len(p1)
    _serve(o, p1)
    # thinking without tools: the next turn re-renders p1's header, resuming p1's exact point (popped in place)
    p2 = p1[:n1 - 1] + [T + 10] + _msg(3, 300) + [E, U] + _msg(4, 300) + [A, T]
    assert _serve(o, p2)[0] == 301
    p3 = p2 + _msg(5, 280) + [E, U] + _msg(6, 400) + [A, T]                  # extension in place
    assert _serve(o, p3)[0] == len(p2)
    for k in o.kept:
        assert not (k[2].get("exact") and k[2].get("prompt_end") is None)
    for i in range(12):                                                      # longer conversations take every slot
        _serve(o, [1] + _msg(100 + i, 2500) + [U] + _msg(200 + i, 300) + [A, T])
    assert all(len(k[0]) < 1000 or k[0][:20] != p3[:20] for k in o.kept)    # p3's chain left VRAM
    spilled = [m for e in o.host_tier.entries for _ids, m in e["points"]]
    assert spilled and not any(m["exact"] and m["prompt_end"] is None for m in spilled)
    hits = o.prefix_counters["host_hits"]
    p4 = p3 + _msg(7, 300) + [E, U] + _msg(8, 300) + [A, T]
    cached, s = _serve(o, p4)
    assert o.prefix_counters["host_hits"] == hits + 1
    assert cached == len(p3) and s.chunks == [len(p3), len(p3) + 301]
    assert _serve(o, p4)[0] == len(p4)                                        # and its exact repeat
    assert o.held == sum(o._bytes(st) for st in o.slots) + len(o.snap_ids) * o.snap_bytes
    assert set(o.e.snaps) == o.snap_ids


def _honest(o):
    """No kept or spilled point claims a state other than its snapshot's (``prefix.at``)."""
    from tensorfold.families.deepseek_v4.cuda.prefix import state_at as at
    for ids, _st, snap, _tail in o.kept:
        if snap.get("exact"):
            assert snap.get("prompt_end") is not None and at(snap) == len(ids) + 1
        else:                                            # a split, or an end resumed without its end (``next``)
            assert at(snap) == len(ids) or (at(snap) == len(ids) + 1 and "next" in snap)
    for entry in o.host_tier.entries:
        for _ids, meta in entry["points"]:
            assert not meta["exact"] or meta["prompt_end"] is not None


def test_owner_extension_demotion_spill_and_repeats(monkeypatch):
    """host_probe3's path on this lane: A keeps its exact end, B extends A in place (``prefixes`` demotes A's
    end: its state stays one row past the point), C diverges at A's last token (lands below; never resumed one
    row early), other prompts push the chains to the host tier, then B's repeat and extension restore from host."""
    from tensorfold.families.gemma4.sm70 import prefix_host
    monkeypatch.setattr(prefix_host, "_to_slab", lambda slab, offset, tensor: tensor)
    o = _host_owner()
    a = [1] + _msg(1, 2099)
    b = a + _msg(2, 100)
    assert _serve(o, a)[0] == 0
    assert _serve(o, b)[0] == len(a)                     # in place, from A's end (demoted: no logits now)
    assert _serve(o, b)[0] == len(b)
    c = a[:-1] + [7] + _msg(3, 50)
    cached, s = _serve(o, c)
    assert cached < len(a) - 1 and (cached == 0 or _restored(o) == cached)
    _honest(o)
    for i in range(12):
        _serve(o, [1] + _msg(100 + i, 2500) + [U] + _msg(200 + i, 300) + [A, T])
        _honest(o)
    hits = o.prefix_counters["host_hits"]
    assert _serve(o, b)[0] == len(b)                     # B's repeat: its exact end from host RAM
    assert _serve(o, b + _msg(4, 30))[0] == len(b)       # and an extension of it
    assert o.prefix_counters["host_hits"] > hits
    _honest(o)
    assert o.held == sum(o._bytes(st) for st in o.slots) + len(o.snap_ids) * o.snap_bytes
    assert set(o.e.snaps) == o.snap_ids


def test_owner_fork_of_an_exact_end_copies_its_last_row():
    """A prompt continuing a kept end while that chain holds a longer point forks it: the fork copies the
    point's rows and the end's last one, and resumes the end's state."""
    from tensorfold.families.deepseek_v4.cuda.prefix import OP_COPY
    o = _owner(rows=8192)
    o.e.context = 1 << 16
    a = [1] + _msg(1, 699)
    _serve(o, a)
    _serve(o, a + _msg(2, 300))                          # in place: the chain now holds a longer end
    cached, s = _serve(o, a + _msg(3, 40))               # continues a's end, not the longer one: a fork
    assert cached == len(a) and _restored(o) == len(a)
    copies = [x for x in o.e.ops if x[0] == OP_COPY]
    assert copies[-2][3] == len(a) - 1 and copies[-1][1:] == (copies[-2][1], copies[-2][2], len(a))


def test_owner_cap_splits_keep_no_point_and_resumes_compute_the_cold_chunks():
    """Points only at natural breaks: a 20K-token message splits its chunks at the 8192-row cap but keeps no point
    there; a resume from the system prompt's end computes exactly the chunks its cold prefill computes from there."""
    o = _owner(rows=8192)
    o.e.context, o.e.budget, o.keep, o.points = 1 << 16, 10 ** 8, 64, _toy_points
    system = [1] + _msg(1, 300)
    p1 = system + [U] + _msg(2, 20000) + [A, T]
    _, s = _serve(o, p1)
    assert s.chunks == [0, 301, 301 + 8192, 301 + 2 * 8192] == _cold_chunks(p1)
    assert sorted(len(k[0]) for k in o.kept) == [301, len(p1) - 1]
    p2 = system + [U] + _msg(3, 20000) + [A, T]
    cached, s = _serve(o, p2)
    assert cached == 301 and s.chunks == _cold_chunks(p2)[1:]
    # an edit inside the long message (a capped split at 8493 would have served it: a mid-message point) resumes
    # at the system prompt's end
    p3 = p1[:12000] + [5] + p1[12001:]
    assert _serve(o, p3)[0] == 301


def test_owner_keeps_the_newest_turns_last_message_point():
    """Under the entry cap, each conversation's newest points (its last message's start and its exact end) outlive
    its older history points: a turn whose header re-renders (thinking without tools) resumes at the last message's
    start of the latest turn, not further back."""
    o = _owner(rows=8192)
    o.e.context, o.e.budget, o.points = 1 << 16, 10 ** 8, _toy_points
    system = [1] + _msg(1, 300)
    convs = [system + [U] + _msg(10 + i, 300) + [A, T] for i in range(4)]     # four chains in four slots
    for i in range(4):
        _serve(o, convs[i])
    for turn in range(6):                    # (the plain GreedyDual ladder lost 20 of these 24 newest points)
        for i in range(4):
            p = convs[i] + _msg(20 + 7 * turn + i, 300) + [E, U] + _msg(60 + 7 * turn + i, 300) + [A, T]
            assert _serve(o, p)[0] == len(convs[i])
            convs[i] = p
            assert len(o.kept) <= o.keep
            for c in convs[:i + 1] if turn == 0 else convs:
                last = max(_toy_points(c))
                assert any(k[0] == c[:last] for k in o.kept), (turn, i)
    p = convs[0][:-1] + [T + 10] + _msg(99, 100) + [E, U] + _msg(98, 100) + [A, T]
    assert _serve(o, p)[0] == max(_toy_points(convs[0]))


def test_host_point_beats_an_exact_end_its_prompt_cannot_continue(monkeypatch):
    """The host tier's MIN_GAIN compares against where a VRAM point resumes: an exact end whose next token the
    prompt does not continue reaches only the point below it, so a longer host point is restored."""
    from tensorfold.families.gemma4.sm70 import prefix_host
    monkeypatch.setattr(prefix_host, "_to_slab", lambda slab, offset, tensor: tensor)
    o = _host_owner()
    p1 = [1] + _msg(1, 300) + [U] + _msg(2, 300) + [U] + _msg(3, 300) + [A, T]
    _serve(o, p1)
    assert sorted(len(k[0]) for k in o.kept) == [301, 602, len(p1) - 1]
    o.host_tier.spill(list(o.kept), None)                # the chain in host RAM too, then its 602 point gone
    o.kept = [k for k in o.kept if len(k[0]) != 602]
    o._sweep()
    q = p1[:-1] + [T + 10] + _msg(4, 100) + [E, U] + _msg(5, 100) + [A, T]   # a re-rendered header
    hits = o.prefix_counters["host_hits"]
    cached, s = _serve(o, q)
    assert cached == 602 and o.prefix_counters["host_hits"] == hits + 1
    assert o.held == sum(o._bytes(st) for st in o.slots) + len(o.snap_ids) * o.snap_bytes
    assert set(o.e.snaps) == o.snap_ids

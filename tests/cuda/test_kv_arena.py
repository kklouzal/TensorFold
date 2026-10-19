"""Attention-cache arenas: growth maps memory in place (no row moves), views own the arena, release returns the memory
to the driver, and qwen3_5's ``grow`` moves only caches a state does not own as an arena's current view."""

import gc
import threading

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda import kv_arena  # noqa: E402
from tensorfold.cuda.memory_gate import torch_live  # noqa: E402
from tensorfold.families.qwen3_5.cuda import forward, multi  # noqa: E402
from tensorfold.families.qwen3_5.cuda.decode import clone_state  # noqa: E402

DEV = torch.device("cuda", torch.cuda.current_device())
ROW = (4, 256)                                  # the 27B's attention row: 4 KV heads x 256, bf16 (2 KiB)


def _settle():
    gc.collect()
    torch.cuda.synchronize()
    kv_arena.reap()


def _pattern(lo, hi, salt=0):
    rows = torch.arange(lo, hi, device=DEV, dtype=torch.float32)[:, None, None]
    cols = torch.arange(ROW[0] * ROW[1], device=DEV, dtype=torch.float32).view(1, *ROW)
    return ((rows * 7 + cols + salt) % 251).to(torch.bfloat16)


def _state(limit=0, rows=0):
    """A one-layer qwen3_5 state whose attention cache starts empty (as ``State`` makes it) or holds ``rows``."""

    st = object.__new__(forward.State)
    st.pos, st.limit, st.room, st.rope_delta = rows, limit, None, 0
    st.conv, st.rec, st.kv_owner = [None], [None], object()
    st.kv = [(torch.empty((0, *ROW), device=DEV, dtype=torch.bfloat16),) * 2]
    if rows:
        forward.grow(st, 0, rows)
        st.kv[0][0][:rows], st.kv[0][1][:rows] = _pattern(0, rows), _pattern(0, rows, 1)
    return st


def test_an_arena_maps_in_place_and_keeps_every_row():
    _settle()
    a = kv_arena.Arena(300_000, ROW, torch.bfloat16, DEV)
    g, done = a.drv.granularity, 0
    for rows in (1, 1000, 5000, 70_000, 300_000):
        extra = a.extra(rows)
        assert extra == -(-rows * a.row_bytes // g) * g - a.mapped and extra % g == 0
        a.ensure(rows)
        v = a.view()
        assert v.data_ptr() == a.base and v.shape == (a.rows, *ROW) and a.rows >= rows
        v[done:rows] = _pattern(done, rows)
        done = rows
        assert torch.equal(v[:done], _pattern(0, done))                 # earlier rows survive every map
    assert a.extra(300_000) == 0 and a.rows == 300_000
    with pytest.raises(ValueError, match="reservation"):
        a.ensure(300_001)


def test_views_hold_the_arena_and_its_release_returns_the_memory():
    _settle()
    three = torch.full((50, *ROW), 3, device=DEV, dtype=torch.bfloat16)
    torch.cuda.empty_cache()
    base_mapped = kv_arena.mapped_bytes()
    free0 = torch.cuda.mem_get_info()[0]
    a = kv_arena.Arena(100_000, ROW, torch.bfloat16, DEV)
    a.ensure(100_000)
    nbytes = a.mapped
    v = a.view()
    v.fill_(3)                                                        # no temporaries: the free memory stays exact
    assert kv_arena.mapped_bytes() - base_mapped == nbytes
    assert torch.cuda.mem_get_info()[0] <= free0 - nbytes + (8 << 20)
    del a
    gc.collect()                                                      # the view holds the arena
    assert kv_arena.pending_bytes() == 0 and torch.equal(v[:50], three)
    del v
    gc.collect()                                                      # its last view went: queued, still mapped
    assert kv_arena.pending_bytes() == nbytes and kv_arena.mapped_bytes() - base_mapped == nbytes
    kv_arena.reap()
    assert kv_arena.pending_bytes() == 0 and kv_arena.mapped_bytes() == base_mapped
    assert torch.cuda.mem_get_info()[0] >= free0 - (8 << 20)          # back to the driver (not the allocator)


def test_a_last_view_dropped_inside_an_arena_operation_queues_its_release():
    """A garbage collection can finalize an arena on a thread already inside an arena operation (it runs at any
    bytecode boundary): the release queues instead of deadlocking on the driver's lock."""

    _settle()
    a = kv_arena.Arena(1000, ROW, torch.bfloat16, DEV)
    a.ensure(1000)
    views, nbytes, lock = [a.view()], a.mapped, a.drv.lock
    del a
    done = threading.Event()

    def drop():
        with lock:
            views.clear()                                             # the last view: finalized right here
        done.set()

    worker = threading.Thread(target=drop, daemon=True)
    worker.start()
    worker.join(60)
    if worker.is_alive():                                             # deadlocked: free the lock for later tests
        lock.release()
    assert done.is_set() and kv_arena.pending_bytes() == nbytes
    kv_arena.reap()
    assert kv_arena.pending_bytes() == 0


def test_only_the_owner_grows_its_current_view():
    _settle()
    owner = object()
    a = kv_arena.Arena(50_000, ROW, torch.bfloat16, DEV, owner)
    a.ensure(1000)
    v1 = a.view()
    assert kv_arena.owned(v1, owner) is a
    assert kv_arena.owned(v1[:10], owner) is None                     # a slice: someone else's prefix
    assert kv_arena.owned(v1, object()) is None                       # the same view under another owner
    assert kv_arena.owned(torch.empty((4, *ROW), device=DEV, dtype=torch.bfloat16), owner) is None
    a.ensure(40_000)
    v2 = a.view()
    assert kv_arena.owned(v1, owner) is None and kv_arena.owned(v2, owner) is a and v1.data_ptr() == v2.data_ptr()


def test_grow_maps_in_place_for_a_state_and_its_clones():
    _settle()
    st = _state(limit=200_000, rows=1000)
    ptr = st.kv[0][0].data_ptr()
    clone = clone_state(st)
    forward.grow(clone, 0, 150_000)                                   # the shared list: the state sees it too
    k, v = st.kv[0]
    assert k.data_ptr() == ptr and k.shape[0] >= 150_000 and st.kv is clone.kv
    assert torch.equal(k[:1000], _pattern(0, 1000)) and torch.equal(v[:1000], _pattern(0, 1000, 1))
    assert kv_arena.owned(k, st.kv_owner).capacity == 200_000         # reserved once for the admitted window
    with pytest.raises(ValueError):
        kv_arena.owned(k, st.kv_owner).ensure(200_001)


def test_a_kept_slice_and_a_copied_list_move_into_arenas_of_their_own():
    _settle()
    st = _state(limit=20_000, rows=3000)
    st.pos = 2000
    kept = multi.kept(st)                                             # rows below pos viewed in place
    forward.grow(kept, 0, 6000)
    assert kept.kv[0][0].data_ptr() != st.kv[0][0].data_ptr()
    kept.kv[0][0][2000:6000] = _pattern(2000, 6000, 5)                # its own rows past pos ...
    assert torch.equal(st.kv[0][0][:3000], _pattern(0, 3000))         # ... leave the original's untouched
    assert torch.equal(kept.kv[0][0][:2000], _pattern(0, 2000))
    mine = multi.private(st, 9000)                                    # a copied list with the same view objects
    assert mine.kv[0][0].data_ptr() not in (st.kv[0][0].data_ptr(), kept.kv[0][0].data_ptr())
    assert torch.equal(mine.kv[0][1][:2000], _pattern(0, 2000, 1))
    mine.kv[0][0][2000:9000] = 0
    assert torch.equal(st.kv[0][0][:3000], _pattern(0, 3000)) and mine.limit == 9000


def test_an_unbounded_state_moves_once_past_its_reservation(monkeypatch):
    _settle()
    monkeypatch.setattr(forward, "UNBOUNDED", 4096)
    st = _state(rows=3000)
    ptr = st.kv[0][0].data_ptr()
    forward.grow(st, 0, 4000)
    assert st.kv[0][0].data_ptr() == ptr
    forward.grow(st, 0, 5000)                                         # past 4,096: one move, doubled reservation
    k = st.kv[0][0]
    assert k.data_ptr() != ptr and kv_arena.owned(k, st.kv_owner).capacity == 8192
    assert torch.equal(k[:3000], _pattern(0, 3000))


def test_the_room_is_charged_with_the_bytes_a_grow_maps():
    _settle()
    st = _state(limit=100_000, rows=10)
    charged = []
    st.room = lambda s, extra: charged.append(extra)
    before = sum(kv_arena.owned(t, st.kv_owner).mapped for t in st.kv[0])
    forward.grow(st, 0, 60_000)
    after = sum(kv_arena.owned(t, st.kv_owner).mapped for t in st.kv[0])
    assert charged == [after - before] and after > before
    forward.grow(st, 0, 60_000)
    assert charged == [after - before]                                # nothing to map: no charge


def test_the_live_room_keeps_retirements_allocated_until_the_driver_releases_them(monkeypatch):
    _settle()
    limit = torch.cuda.memory_allocated() + kv_arena.mapped_bytes() + (512 << 20)
    monkeypatch.setattr("tensorfold.cuda.memory_gate.cuda_limit_bytes", lambda: limit)
    live = torch_live(torch, lambda runtime: 1 << 50)                 # plenty of free memory: the limit decides
    start = live()
    a = kv_arena.Arena(200_000, ROW, torch.bfloat16, DEV)
    a.ensure(100_000)
    v = a.view()
    assert start - live() == a.mapped
    mapped = a.mapped
    del a, v
    gc.collect()                                                      # queued mappings are still physically allocated
    assert live() == start - mapped
    kv_arena.reap()
    assert live() == start

"""Host-RAM prefix tier: spilled points restore bit-identically into a slot with the same page ownership."""
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import prefixes
from tensorfold.families.qwen4_exp.cuda.prefix_host import HostPrefixTier

PAGE = 4


class Placement:
    page_tokens, group_size, degree, target_layers, world = PAGE, 4, 1, 2, 8

    def indexer_owner(self, slot, layer, *, include_mtp=True):
        return (slot * (self.target_layers + int(include_mtp)) + layer) % self.world

    def local_rows(self, slot, layer, rows, rank):      # every page of a (slot, layer) on one rank of the group
        return -(-rows // PAGE) * PAGE if rank % 4 == (slot + layer) % 4 else 0


def cache(rows, rank, slot, layer, seed):
    owners = [4 * (rank // 4) + ((slot + layer) % 4) for _ in range(-(-rows // PAGE))]
    offsets, owned = [], 0
    for owner in owners:
        offsets.append(owned if owner == rank else -1)
        owned += PAGE if owner == rank else 0
    generator = torch.Generator().manual_seed(seed)
    return NS(k=torch.randint(0, 255, (owned, 1, 8), dtype=torch.uint8, generator=generator),
              v=torch.randint(0, 255, (owned, 1, 8), dtype=torch.uint8, generator=generator),
              ks=torch.randn((owned, 1, 2), generator=generator).half(), vs=torch.randn((owned, 1, 2), generator=generator).half(),
              key_format=NS(quantized=True), value_format=NS(quantized=True), offsets=tuple(offsets), owners=owners)


def state(slot, rank=None, rows=32, seed=0):
    rank = (slot % 4) if rank is None else rank          # the rank that owns this slot's layer-0 pages
    st = NS(kc=[cache(rows, rank, slot, layer, seed + layer) for layer in range(2)],
            pooled=[torch.randn((rows // 4, 3), generator=torch.Generator().manual_seed(seed + 9 + layer))
                    for layer in range(2)],
            history_placement=Placement(), history_rank=rank, history_slot=slot, ratio=4, kv_identity=("q4", 262144),
            limit=rows, history_native_readers=(), name=f"slot{slot}")
    st.slot_generation = 0

    def begin():
        st.slot_generation += 1
    st.begin_sequence_generation = begin
    return st


def snapshot(pos, principal=None, seed=1):
    generator = torch.Generator().manual_seed(seed)
    snap = {"pos": pos, "mtp_len": 0, "rec": torch.randn((3, 5), generator=generator),
            "kv_identity": ("q4", 262144), "ple_history": [1, 2, 3],
            "compact_indexer": {"schema": "c", "lengths": (pos, pos), "tails": (None, {"raw": torch.ones(4, 2)})}}
    if principal is not None:
        snap["prefix_principal"] = principal
    return snap


def owned_rows(st, layer, rows):
    out = []
    for page in range(-(-rows // PAGE)):
        offset = st.kc[layer].offsets[page]
        if offset >= 0:
            count = min(PAGE, rows - page * PAGE)
            out.append(tuple(t[offset:offset + count].clone() for t in (st.kc[layer].k, st.kc[layer].v,
                                                                         st.kc[layer].ks, st.kc[layer].vs)))
    return out


def tier(capacity=1 << 16, block=256):
    return HostPrefixTier(capacity, device="cpu", block_bytes=block, pin=False)


def test_spilled_point_restores_bit_identical_rows_pool_and_snapshot_into_a_same_ownership_slot():
    host = tier()
    source, target, other = state(0, rank=0), state(8, rank=0, seed=50), state(1, rank=0, seed=70)
    ids = list(range(17))
    snap, tail = snapshot(17), torch.arange(6.0)
    host.spill([(ids, source, snap, tail)], None)
    entry, index = host.best(ids + [99, 100], None)
    assert host.restorable(entry, target) and not host.restorable(entry, other)
    got_ids, got_snap, got_tail = host.restore(entry, index, target)
    assert got_ids == ids and torch.equal(got_tail, tail)
    assert torch.equal(got_snap["rec"], snap["rec"]) and got_snap["ple_history"] == [1, 2, 3]
    assert got_snap["compact_indexer"]["tails"][1]["raw"].equal(snap["compact_indexer"]["tails"][1]["raw"])
    for layer in range(2):
        for a, b in zip(owned_rows(source, layer, 17), owned_rows(target, layer, 17), strict=True):
            assert all(torch.equal(x, y) for x, y in zip(a, b))
        owner = Placement().indexer_owner(0, layer, include_mtp=False)
        if owner == 0:
            assert torch.equal(target.pooled[layer][:17 // 4], source.pooled[layer][:17 // 4])
    assert host.record()["restores"] == 1 and host.record()["entries"] == 1


def test_lookup_is_longest_extending_point_scoped_by_principal_and_never_the_whole_prompt():
    host = tier()
    source = state(0, rank=0)
    host.spill([(list(range(8)), source, snapshot(8, "a"), None),
                (list(range(16)), source, snapshot(16, "a"), None)], "a")
    assert host.best(list(range(16)), "a")[0]["points"][host.best(list(range(16)), "a")[1]][0] == list(range(8))
    entry, index = host.best(list(range(20)), "a")
    assert len(entry["points"][index][0]) == 16
    assert host.best(list(range(20)), "b") is None and host.best([5] * 20, "a") is None


def test_capacity_is_bounded_and_eviction_drops_the_lowest_value_entry():
    probe = tier()
    a, b = state(0, rank=0), state(4, rank=0, seed=20)
    probe.spill([(list(range(8)), a, snapshot(8), None)], None)
    each = len(probe.entries[0]["blocks"]) + len(probe.entries[0]["extent"]["blocks"])
    host = tier(capacity=3 * each * 256, block=256)
    host.spill([(list(range(8)), a, snapshot(8), None)], None)
    host.spill([([100] * 8, b, snapshot(8, seed=3), None)], None)
    b.begin_sequence_generation()                                   # other content in the slot: a new admission
    host.spill([([101] * 8, b, snapshot(8, seed=4), None)], None)
    first, second, _third = host.entries
    host.restore(first, 0, state(8, rank=0, seed=30))              # used again: worth more than the others
    b.begin_sequence_generation()
    host.spill([([102] * 8, b, snapshot(8, seed=5), None)], None)  # full: the least valuable goes
    assert host.record()["evictions"] == 1 and first in host.entries and second not in host.entries
    assert len(host.free) == 0 and host.blocks == 3 * each
    too_big = state(12, rank=0, rows=4096)
    host.spill([(list(range(4000)), too_big, snapshot(4000), None)], None)
    assert host.record()["spill_skipped"] == 1 and len(host.entries) == 3


def test_identical_points_are_not_copied_twice():
    host = tier()
    source = state(0, rank=0)
    host.spill([(list(range(8)), source, snapshot(8), None)], None)
    host.spill([(list(range(8)), source, snapshot(8), None)], None)
    assert len(host.entries) == 1 and host.entries[0]["uses"] == 2


def owner(states, host):
    decoder = NS(kept=[], free=list(states), used={}, clock=0, keep=8, depth=0, host_tier=host,
                 prefix_counters=dict.fromkeys(("lookups", "resume_hits", "resume_tokens", "fork_hits", "fork_tokens",
                                                "host_hits", "host_tokens"), 0))
    decoder._busy = lambda: set()
    decoder._grow = lambda st, rows, **_kw: True
    decoder._drop_kept = lambda st: setattr(decoder, "kept", [k for k in decoder.kept if k[1] is not st])
    return decoder


def test_eviction_spills_the_chain_and_a_later_prompt_restores_it_from_host():
    host = tier()
    source = state(0, rank=0)
    decoder = owner([], host)
    ids = list(range(20))
    prefixes.remember(decoder, ids, source, snapshot(20), None)
    # A new unrelated prompt needs a slot: the idle kept chain is evicted, so it spills instead of being lost.
    slot, resume, cached = prefixes.slot_for(decoder, [7] * 30, True)
    assert slot is source and resume is None and cached == 0 and decoder.kept == []
    assert host.record()["spills"] == 1
    target = state(8, rank=0, seed=40)
    decoder.free.append(target)
    slot, resume, cached = prefixes.slot_for(decoder, ids + [1, 2, 3], True)
    assert slot is target and cached == 20 and resume["state"]["pos"] == 20
    assert decoder.prefix_counters["host_hits"] == 1 and decoder.prefix_counters["host_tokens"] == 20
    assert [k[0] for k in decoder.kept] == [ids] and decoder.kept[0][1] is target
    for a, b in zip(owned_rows(source, 0, 20), owned_rows(target, 0, 20), strict=True):
        assert all(torch.equal(x, y) for x, y in zip(a, b))


def test_an_extension_restored_from_host_drops_the_prompt_end_it_rewrites():
    # Its rows from the point on are rewritten, the end's last row first: a later exact repeat must not join the end.
    host = tier()
    decoder = owner([], host)
    ids = list(range(20))
    snap = snapshot(20)
    snap["prompt_end"] = {"token": 5, "state": snapshot(21, seed=2), "tail": None, "last": torch.ones(3),
                          "head": torch.ones(4)}
    prefixes.remember(decoder, ids, state(0, rank=0), snap, None)
    prefixes.slot_for(decoder, [7] * 30, True)                     # spilled with its end
    decoder.free.append(state(8, rank=0, seed=40))
    _slot, resume, cached = prefixes.slot_for(decoder, ids + [1, 2, 3], True)
    assert cached == 20 and "head" not in resume and decoder.prefix_counters["host_hits"] == 1
    assert [k[0] for k in decoder.kept] == [ids] and "prompt_end" not in decoder.kept[0][2]


def test_a_slot_with_other_page_ownership_is_never_a_restore_target():
    host = tier()
    decoder = owner([], host)
    ids = list(range(20))
    prefixes.remember(decoder, ids, state(0, rank=0), snapshot(20), None)
    prefixes.slot_for(decoder, [7] * 30, True)                     # spills slot 0's chain
    decoder.free.append(state(1, rank=0, seed=40))                  # residue 1: other ranks own its pages
    _slot, resume, cached = prefixes.slot_for(decoder, ids + [1], True)
    assert resume is None and cached == 0 and decoder.prefix_counters["host_hits"] == 0


@pytest.mark.parametrize("gain,restored", [(64, True), (63, False)])
def test_host_point_must_save_min_gain_tokens_beyond_the_best_vram_point(gain, restored):
    host = tier(capacity=1 << 20)
    decoder = owner([], host)
    long_ids = list(range(200))
    prefixes.remember(decoder, long_ids, state(0, rank=0, rows=256), snapshot(200), None)
    prefixes.slot_for(decoder, [7] * 30, True)                     # spilled
    short = long_ids[:200 - gain]
    prefixes.remember(decoder, short, state(4, rank=0, rows=256, seed=60), snapshot(len(short)), None)
    decoder.free.append(state(8, rank=0, rows=256, seed=80))
    _slot, _resume, cached = prefixes.slot_for(decoder, long_ids + [1], True)
    assert (cached == 200) is restored


def test_spills_agree_point_sizes_once_per_shape_not_per_spill(monkeypatch):
    host = tier(capacity=1 << 20)
    calls = []
    monkeypatch.setattr(host, "_agree", lambda local: calls.append(local) or local)
    source = state(0, rank=0)
    for n in range(5):
        source.begin_sequence_generation()                          # each other content: a new admission
        host.spill([([50 + n] * 8, source, snapshot(8, seed=n), None)], None)
    assert len(calls) == 1 and len(host.entries) == 5
    source.begin_sequence_generation()
    host.spill([([90] * 8, source, snapshot(8), torch.ones(3))], None)   # a point with an MTP tail: a new shape
    assert len(calls) == 2


def test_restoring_a_shorter_point_grows_the_slot_to_the_entrys_longest_point():
    # A slot chain spills every ladder point with one copy of the rows below its longest point; restoring a
    # shorter point copies those rows too, so the target's page table must cover them before restore().
    host = tier(capacity=1 << 20)
    decoder = owner([], host)
    grown = []
    decoder._grow = lambda st, rows, **_kw: grown.append(rows) or True
    source = state(0, rank=0, rows=256)
    long_ids = list(range(200))
    prefixes.remember(decoder, long_ids[:80], source, snapshot(80), None)
    prefixes.remember(decoder, long_ids, source, snapshot(200), None)
    prefixes.slot_for(decoder, [7] * 30, True)                     # spills both points of slot 0's chain
    decoder.free.append(state(8, rank=0, rows=256, seed=80))
    grown.clear()
    _slot, _resume, cached = prefixes.slot_for(decoder, long_ids[:80] + [5000] * 10, True)
    assert cached == 80 and grown[0] >= 200


def test_an_entry_evicted_while_its_restore_target_grows_is_not_restored():
    # Growing the target may spill another chain into the tier, which can evict this entry (its blocks then hold
    # the new spill): the request must prefill cold instead of restoring overwritten bytes.
    host = tier()
    decoder = owner([], host)
    ids = list(range(20))
    prefixes.remember(decoder, ids, state(0, rank=0), snapshot(20), None)
    prefixes.slot_for(decoder, [7] * 30, True)                     # spilled
    target = state(8, rank=0, seed=40)
    decoder.free.append(target)

    def grow(st, rows, **_kw):
        host.entries.clear()                                        # the spill this growth made evicted it
        return True
    decoder._grow = grow
    slot, resume, cached = prefixes.slot_for(decoder, ids + [1, 2, 3], True)
    assert resume is None and cached == 0 and host.record()["restores"] == 0
    assert decoder.prefix_counters["host_hits"] == 0


def test_slab_sizes_are_what_pinning_allocates():
    # The pinned allocator rounds a request up to a power of two: a slab is one (so pinned == charged bytes).
    from tensorfold.families.qwen4_exp.cuda.prefix_host import BLOCK_BYTES, pinned_slab_bytes

    gib = 1 << 30
    assert pinned_slab_bytes(8 * gib) == 8 * gib                      # 64 GiB over eight ranks: unchanged
    assert pinned_slab_bytes(6 * gib + 5) == 4 * gib                  # never above the request
    assert pinned_slab_bytes(BLOCK_BYTES) == BLOCK_BYTES and pinned_slab_bytes(BLOCK_BYTES - 1) == 0


def striped(rows, rank, seed, arena_offset=0, gap=None):
    """A degree-4 layer: the group's ranks own every fourth page in turn, packed back to back in the rank's arena
    from ``arena_offset`` (``gap``: a page index after which the packing skips a page, i.e. a target arena laid out
    differently from the source)."""
    owners = [4 * (rank // 4) + page % 4 for page in range(-(-rows // PAGE))]
    offsets, owned = [], arena_offset
    for page, owner in enumerate(owners):
        offsets.append(owned if owner == rank else -1)
        owned += PAGE if owner == rank else 0
        owned += PAGE if owner == rank and page == gap else 0
    generator = torch.Generator().manual_seed(seed)
    size = owned + PAGE
    return NS(k=torch.randint(0, 255, (size, 1, 8), dtype=torch.uint8, generator=generator),
              v=torch.randint(0, 255, (size, 1, 8), dtype=torch.uint8, generator=generator),
              ks=torch.randn((size, 1, 2), generator=generator).half(), vs=torch.randn((size, 1, 2), generator=generator).half(),
              key_format=NS(quantized=True), value_format=NS(quantized=True), offsets=tuple(offsets), owners=owners)


@pytest.mark.parametrize("gap", [None, 5])
def test_striped_pages_spill_as_one_run_and_restore_page_by_page_into_any_arena_layout(gap):
    rows = 46                                                   # 12 pages, the last partial: rank 1 owns 1, 5, 9
    host = tier(capacity=1 << 20)
    source = state(0, rank=1, rows=rows)
    source.kc = [striped(rows, 1, 3 + layer) for layer in range(2)]
    target = state(8, rank=1, rows=rows, seed=50)
    target.kc = [striped(rows, 1, 60 + layer, arena_offset=PAGE, gap=gap) for layer in range(2)]
    host.spill([(list(range(rows - 1)), source, snapshot(rows - 1), None)], None)
    layers = host.entries[0]["extent"]["layers"]
    assert all(len(runs) == 1 and [page for page, _n in runs[0][0]] == [1, 5, 9] for _o, _f, _c, _w, runs, _p in layers)
    host.restore(host.entries[0], 0, target)
    for layer in range(2):
        for page in (1, 5, 9):
            count = min(PAGE, rows - page * PAGE)
            a, b = source.kc[layer], target.kc[layer]
            for name in ("k", "v", "ks", "vs"):
                assert torch.equal(getattr(a, name)[a.offsets[page]:a.offsets[page] + count],
                                   getattr(b, name)[b.offsets[page]:b.offsets[page] + count])


def test_slab_copies_span_consecutive_blocks():
    host = tier(capacity=1 << 12, block=256)
    pieces = list(host._views([3, 4, 5, 9], 100, 700))
    assert [(view.numel(), at) for view, at in pieces] == [(668, 0), (32, 668)]
    assert pieces[0][0].data_ptr() == host.slab[3 * 256 + 100:].data_ptr()
    assert pieces[1][0].data_ptr() == host.slab[9 * 256:].data_ptr()


def same_rows(source, target, rows):
    for layer in range(2):
        for a, b in zip(owned_rows(source, layer, rows), owned_rows(target, layer, rows), strict=True):
            assert all(torch.equal(x, y) for x, y in zip(a, b))
        if Placement().indexer_owner(source.history_slot, layer, include_mtp=False) == source.history_rank:
            assert torch.equal(target.pooled[layer][:rows // 4], source.pooled[layer][:rows // 4])


def test_a_chain_spilled_point_by_point_stores_its_shared_rows_once_and_restores_every_point():
    host = tier(capacity=1 << 20)
    source = state(0, rank=0, rows=48)
    full, previous = [], 0
    for n in (20, 30, 33):          # unaligned ends (partial pages copied again); each smaller than its base: no merge
        before = host.counters["spill_bytes"]
        host.spill([(list(range(n)), source, snapshot(n, seed=n), None)], None)
        full.append(host._history_bound(source, n, 0))
        rows = host.counters["spill_bytes"] - before - host.entries[-1]["point_bytes"]
        assert rows == host._history_bound(source, n, 0, (previous // PAGE * PAGE, 0))
        assert host.entries[-1]["extent"]["start"] == (previous // PAGE * PAGE, 0)
        previous = n
    assert host.counters["shared_bytes"] > 0 and host.record()["extents"] == 3
    assert host.counters["spill_bytes"] - sum(e["point_bytes"] for e in host.entries) < sum(full)
    for n in (33, 20, 30):
        target = state(8, rank=0, rows=48, seed=100 + n)
        entry, index = host.best(list(range(n)) + [999], None)
        host.restore(entry, index, target)
        same_rows(source, target, n)


def test_evicting_a_point_keeps_rows_its_sharers_still_restore_and_frees_them_with_the_last():
    host = tier(capacity=1 << 20)
    source = state(0, rank=0, rows=48)
    for n in (24, 36, 40):
        host.spill([(list(range(n)), source, snapshot(n, seed=n), None)], None)
    root = host.entries[0]["extent"]
    assert host.entries[2]["extent"]["parent"]["parent"] is root and root["users"] == 3
    victim = host.entries[0]
    free = len(host.free)
    victim["used"] = -1e9                                           # the least valuable: evicted next
    host._evict_one()
    assert root["alive"] and root["users"] == 2 and len(host.free) == free + len(victim["blocks"])
    target = state(8, rank=0, rows=48, seed=7)
    entry, index = host.best(list(range(40)) + [1], None)
    host.restore(entry, index, target)
    same_rows(source, target, 40)
    while host.entries:
        host._evict_one()
    assert not root["alive"] and host.record()["extents"] == 0 and len(host.free) == host.blocks


def test_a_shared_entry_is_worth_more_than_the_same_entry_holding_its_rows_alone():
    host = tier(capacity=1 << 20)
    source = state(0, rank=0, rows=48)
    host.spill([(list(range(40)), source, snapshot(40), None)], None)
    alone = host._value(host.entries[0])
    host.spill([(list(range(44)), source, snapshot(44), None)], None)
    assert host._value(host.entries[0]) > alone


def test_a_resume_in_place_carries_the_slots_host_rows_and_a_new_admission_drops_them():
    host = tier(capacity=1 << 20)
    source = state(0, rank=0, rows=64)
    decoder = owner([], host)
    turn = list(range(20))
    prefixes.remember(decoder, turn, source, snapshot(20), None)
    host.spill([(turn, source, snapshot(20), None)], None)          # (as remember would past keep)
    slot, _resume, cached = prefixes.slot_for(decoder, turn + [7] * 10, True)
    assert slot is source and cached == 20 and host.held(source) is not None
    longer = turn + [7] * 10 + [8] * 10
    before = host.counters["shared_bytes"]
    host.spill([(longer, source, snapshot(40), None)], None)
    assert host.counters["shared_bytes"] > before and host.entries[-1]["extent"]["parent"] is host.entries[0]["extent"]
    target = state(8, rank=0, rows=64, seed=9)
    host.restore(host.entries[-1], 0, target)
    same_rows(source, target, 40)
    source.begin_sequence_generation()                              # a new admission that does not carry them
    assert host.held(source) is None


def test_history_that_changes_without_a_new_admission_is_refused():
    host = tier(capacity=1 << 20)
    source = state(0, rank=0)
    host.spill([(list(range(8)), source, snapshot(8), None)], None)
    with pytest.raises(RuntimeError, match="stale"):
        host.spill([([5] * 12, source, snapshot(12), None)], None)


def test_a_restored_conversation_spills_only_its_new_rows():
    host = tier(capacity=1 << 20)
    source = state(0, rank=0, rows=64)
    decoder = owner([], host)
    turn = list(range(30))
    host.spill([(turn, source, snapshot(30), None)], None)
    target = state(8, rank=0, rows=64, seed=11)
    decoder.free.append(target)
    slot, _resume, cached = prefixes.slot_for(decoder, turn + [3] * 5, True)
    assert slot is target and cached == 30 and host.held(target)[1] == 30
    # the next turn's rows past 30 are the target's own; its spill copies only them (and the partial page below)
    nxt = turn + [3] * 20
    host.spill([(nxt, target, snapshot(50), None)], None)
    extent = host.entries[-1]["extent"]
    assert extent["parent"] is host.entries[0]["extent"] and extent["start"] == (28, 0)
    third = state(16, rank=0, rows=64, seed=13)
    host.restore(host.entries[-1], 0, third)
    same_rows(target, third, 50)
    same_rows(source, third, 30)


def test_a_fork_shares_the_source_rows_below_its_copy_only_with_the_same_page_ownership():
    host = tier(capacity=1 << 20)
    source = state(0, rank=0, rows=64)
    host.spill([(list(range(40)), source, snapshot(40), None)], None)
    same_class, other = state(8, rank=0, rows=64), state(1, rank=0, rows=64)
    for spare in (same_class, other):
        spare.begin_sequence_generation()
        host.hold(spare, host.held(source), 36, 0)
    assert host.held(same_class)[1:] == (36, 0) and host.held(other) is None


def test_rows_mostly_past_what_a_spill_can_use_are_copied_again_not_pinned():
    host = tier(capacity=1 << 20)
    source = state(0, rank=0, rows=64)
    host.spill([(list(range(60)), source, snapshot(60), None)], None)
    long_extent = host.entries[0]["extent"]
    source.begin_sequence_generation()
    host.hold(source, host.held(source), 8, 0)                       # a resume at 8 of a 60-row extent
    host.spill([(list(range(20)), source, snapshot(20), None)], None)
    assert host.entries[-1]["extent"]["parent"] is None and long_extent["users"] == 1


def test_mtp_rows_share_and_restore_as_their_own_dimension():
    host = tier(capacity=1 << 20)
    source = state(0, rank=0, rows=48)
    source.mtp_kc, source.mtp_pooled = cache(48, 0, 0, 4, 77), torch.randn((12, 3))
    for n in (24, 36, 40):
        snap = snapshot(n, seed=n)
        snap["mtp_len"] = n - 2                                     # the MTP rows lag the target's
        host.spill([(list(range(n)), source, snap, None)], None)
    assert host.entries[-1]["extent"]["start"] == (36, 32) and host.entries[-1]["mtp_rows"] == 38
    target = state(8, rank=0, rows=48, seed=5)
    target.mtp_kc, target.mtp_pooled = cache(48, 0, 8, 4, 78), torch.randn((12, 3))
    entry, index = host.best(list(range(40)) + [1], None)
    host.restore(entry, index, target)
    same_rows(source, target, 40)
    a, b = source.mtp_kc, target.mtp_kc                             # rank 0 owns these MTP pages
    assert all(offset >= 0 for offset in a.offsets)
    for page in range(-(-38 // PAGE)):
        if a.offsets[page] >= 0:
            count = min(PAGE, 38 - page * PAGE)
            assert torch.equal(a.k[a.offsets[page]:a.offsets[page] + count], b.k[b.offsets[page]:b.offsets[page] + count])
    if Placement().indexer_owner(0, 2, include_mtp=True) == 0:
        assert torch.equal(target.mtp_pooled[:38 // 4], source.mtp_pooled[:38 // 4])


def test_chains_merge_so_a_restore_reads_few_extents_and_rows_are_copied_a_logarithmic_number_of_times():
    from tensorfold.families.qwen4_exp.cuda.prefix_host import MAX_CHAIN

    host = tier(capacity=1 << 22)
    rows = 4 * 40 + 4
    source = state(0, rank=0, rows=rows)
    ends = [4 * n for n in range(1, 41)]                            # page ends: no partial page copied again
    for n in ends:
        host.spill([(list(range(n)), source, snapshot(n, seed=n), None)], None)
    depths = [host.entries[i]["extent"]["depth"] for i in range(len(ends))]
    assert max(depths) <= MAX_CHAIN and depths[2] == 2              # equal spills merge as a binary counter
    copied = host.counters["spill_bytes"] - sum(e["point_bytes"] for e in host.entries)
    assert copied < 4 * host._history_bound(source, ends[-1], 0), copied / host._history_bound(source, ends[-1], 0)
    for n in (ends[-1], ends[MAX_CHAIN], ends[3]):
        target = state(8, rank=0, rows=rows, seed=200 + n)
        entry, index = host.best(list(range(n)) + [999], None)
        host.restore(entry, index, target)
        same_rows(source, target, n)


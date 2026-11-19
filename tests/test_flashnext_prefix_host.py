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
    st.begin_sequence_generation = lambda: 0
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
    each = len(probe.entries[0]["blocks"])
    host = tier(capacity=3 * each * 256, block=256)
    host.spill([(list(range(8)), a, snapshot(8), None)], None)
    host.spill([([100] * 8, b, snapshot(8, seed=3), None)], None)
    host.spill([([101] * 8, b, snapshot(8, seed=4), None)], None)
    first, second, _third = host.entries
    host.restore(first, 0, state(8, rank=0, seed=30))              # used again: worth more than the others
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
        host.spill([([50 + n] * 8, source, snapshot(8, seed=n), None)], None)
    assert len(calls) == 1 and len(host.entries) == 5
    host.spill([([90] * 8, source, snapshot(8), torch.ones(3))], None)   # a point with an MTP tail: a new shape
    assert len(calls) == 2

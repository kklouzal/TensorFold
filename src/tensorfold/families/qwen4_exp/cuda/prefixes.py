"""Select and retain prompt prefixes without consuming a longer chain when a spare slot can hold a copy.

Kept entries are ordered by value, lowest first (GreedyDual-Size over prompt tokens): an entry is worth the clock at its
slot's last use plus the tokens it saves. What must go (an entry past ``keep``, a slot for a new prompt, room for a
growing stream) is the lowest value, and the clock rises to it, so a long chain outlives many short requests while an
unused one still ages out. A slot is used when a prompt keeps a point in it or resumes from one of its entries.

Past ``keep``, a chain's spare points go before any value: a chain keeps one point a doubling of distance from its end
(its ``LADDER``), so a prompt that diverges anywhere in it resumes within about that distance again, not cold.
"""

from tensorfold.cleanup import raise_failures

LADDER = 4096          # tokens: the nearest band of a chain's ladder, each farther band twice as wide
_UNSCOPED = object()   # Legacy policy only; None is the explicit local/no-auth domain in selected FIFO profiles.


def _best(kept, prompt, busy=(), *, principal=_UNSCOPED):
    if principal is not _UNSCOPED:
        kept = (entry for entry in kept if "prefix_principal" in entry[2] and entry[2]["prefix_principal"] == principal)
    return max((k for k in kept if id(k[1]) not in busy and len(k[0]) < len(prompt)
                and prompt[:len(k[0])] == k[0]), key=lambda k: len(k[0]), default=None)


def _longer(kept, entry):
    return any(k[1] is entry[1] and len(k[0]) > len(entry[0]) for k in kept)


def _lost(kept, entry) -> int:
    """The kept tokens a prompt resuming ``entry`` in place discards from its chain."""

    return max(len(k[0]) for k in kept if k[1] is entry[1]) - len(entry[0])


def _value(owner, entry) -> int:
    return owner.used.get(id(entry[1]), 0) + len(entry[0])


def _use(owner, st) -> None:
    owner.used[id(st)] = owner.clock
    # stable: equal values keep their order. A new list, published by one store: a /stats scrape on another thread
    # reads ``kept`` whole (sorting in place empties a list while it sorts)
    owner.kept = sorted(owner.kept, key=lambda k: _value(owner, k))


def _idle(owner, busy, *spared):
    """The idle kept slot (none of ``spared``) whose chain is worth least, as its most valuable entry, or None."""

    chains: dict[int, tuple] = {}
    for k in owner.kept:                     # ascending: a slot's last entry is its most valuable
        if id(k[1]) not in busy and all(k[1] is not s for s in spared):
            chains[id(k[1])] = k
    return min(chains.values(), key=lambda k: _value(owner, k), default=None)


def _spare(kept) -> list:
    """Points a chain's ladder does not need: in each band of distance from its end, [2^b, 2^(b+1)) * LADDER, all but
    the farthest (a prompt diverging inside the band resumes there rather than a band farther out). Never its end, nor
    its root (its shortest point, as a rule the system block other conversations share). ``kept`` ascends by value, as
    ``remember`` sorts it, so a chain's points come shortest first."""

    ends: dict[int, tuple[int, int]] = {}
    for k in kept:
        low, high = ends.get(id(k[1]), (len(k[0]), len(k[0])))
        ends[id(k[1])] = (min(low, len(k[0])), max(high, len(k[0])))
    bands, spare = set(), []
    for k in kept:
        root, end = ends[id(k[1])]
        gap = end - len(k[0])
        if gap == 0 or len(k[0]) == root:
            continue
        band = (id(k[1]), (gap // LADDER).bit_length())
        if band in bands:                            # its band's farthest point came first
            spare.append(k)
        bands.add(band)
    return spare


def slot_for(owner, prompt: list[int], reuse: bool, *, principal=_UNSCOPED):
    """Copy a fork into spare capacity; else resume a chain in place when that keeps more than it discards."""

    busy = owner._busy()
    best = _best(owner.kept, prompt, principal=principal) if reuse else None
    if best is not None and owner.free and (id(best[1]) in busy or _longer(owner.kept, best)):
        spare = owner.free.pop()
    else:
        spare = None
    if spare is not None:
        try:
            if owner._grow(spare, len(prompt) + owner.depth + 2, protect=best[1]):
                spare.begin_sequence_generation()
                restored = spare.copy_prefix(best[1], len(best[0]), best[2]["mtp_len"],
                    **({"indexer_snapshot": best[2]["compact_indexer"], "poison_control": owner.poison_control}
                       if spare.compact_indexer is not None else {}))
                _use(owner, best[1])
                state = best[2] if restored is None else {**best[2], "compact_indexer": restored}
                return spare, {"state": state, "tail": best[3]}, len(best[0])
        except BaseException as primary:  # noqa: BLE001 - retain a popped fork slot even on cancellation
            try:
                owner._quarantine(spare)
                owner._retire_failed_slot(spare)
            except BaseException as cleanup:  # noqa: BLE001 - preserve fork and exclusive-retirement failures
                raise_failures(primary, [cleanup])
            raise
        owner.free.append(spare)        # no room for the copy: a cold prefill would need as much, so resume in place
    best = _best(owner.kept, prompt, busy, principal=principal) if reuse else None
    if best is not None:
        # resume in place unless that discards more kept tokens than going cold costs: the reused tokens prefilled
        # again, and with no free slot the whole chain a cold prefill evicts (best's own, if it is worth least)
        cold = len(best[0])
        if not owner.free:
            victim = _idle(owner, busy)[1]
            cold += max(len(k[0]) for k in owner.kept if k[1] is victim)
        if _lost(owner.kept, best) > cold:
            best = None
    if best is not None:
        best[1].begin_sequence_generation()
        n = len(best[0])
        owner.kept = [k for k in owner.kept if k[1] is not best[1]
                      or len(k[0]) <= n and best[0][:len(k[0])] == k[0]]
        _use(owner, best[1])
        return best[1], {"state": best[2], "tail": best[3]}, n
    if owner.free:
        idle = owner.free.pop()
        try:
            idle.begin_sequence_generation()
        except BaseException:
            owner.free.append(idle)
            raise
        return idle, None, 0
    idle = evict(owner, None, new_sequence=True)
    if idle is None:
        raise RuntimeError("no free stream slot")
    return idle, None, 0


def evict(owner, keep, protect=None, *, new_sequence=False):
    """The lowest-valued idle kept slot (never ``keep`` or ``protect``), its entries dropped and the clock risen to its
    value, or None."""

    idle = _idle(owner, owner._busy(), keep, protect)
    if idle is None:
        return None
    if new_sequence:
        idle[1].begin_sequence_generation()
    owner.clock = max(owner.clock, _value(owner, idle))
    owner._drop_kept(idle[1])
    return idle[1]


def _worth(owner, entry) -> int:
    """What dropping one entry loses: its slot's clock plus the tokens it saves past the next shorter point of its chain
    (all of them for a chain's end, which a prompt that extends it resumes from, or a lone point)."""

    lengths = [len(k[0]) for k in owner.kept if k[1] is entry[1]]
    shorter = [n for n in lengths if n < len(entry[0])]
    saved = len(entry[0]) - max(shorter) if shorter and len(entry[0]) < max(lengths) else len(entry[0])
    return owner.used.get(id(entry[1]), 0) + saved


def remember(owner, ids, st, snap, tail, *, principal=_UNSCOPED) -> None:
    """Keep a point of ``st``'s chain; past ``keep`` entries, spare ladder points go first, then the one worth least
    (near-end ladder points before deep history), displaced idle slots back to the free list."""

    if principal is _UNSCOPED:
        gone = [k[1] for k in owner.kept if k[0] == ids]
        owner.kept = [k for k in owner.kept if k[0] != ids] + [(ids, st, snap, tail)]
    else:
        if "prefix_principal" not in snap or snap["prefix_principal"] != principal:
            raise ValueError("prefix publication differs from its actual request principal")
        def same(entry):
            return entry[0] == ids and "prefix_principal" in entry[2] and entry[2]["prefix_principal"] == principal
        gone = [entry[1] for entry in owner.kept if same(entry)]
        owner.kept = [entry for entry in owner.kept if not same(entry)] + [(ids, st, snap, tail)]
    _use(owner, st)
    while len(owner.kept) > owner.keep:
        spare = _spare(owner.kept)
        if spare:                                                   # a ladder's spare point: the clock stays
            entry = min(spare, key=lambda k: _value(owner, k))
        else:
            entry = min(owner.kept, key=lambda k: _worth(owner, k))
            owner.clock = max(owner.clock, _worth(owner, entry))
        owner.kept = [k for k in owner.kept if k is not entry]      # by identity: entries hold tensors
        gone.append(entry[1])
    busy = owner._busy()
    for old in gone:
        if old is not st and id(old) not in busy and all(k[1] is not old for k in owner.kept) and \
                all(f is not old for f in owner.free):
            owner.free.append(old)

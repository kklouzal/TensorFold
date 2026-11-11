"""Byte-funded retention contracts; payload sentinels never allocate GPU storage."""
import pytest
from collections import deque
from tensorfold.engine.prefix_blocks import PrefixBlocks, PrefixCapacity, retention_config_charge, validate_prefix_budget

WINDOWS = {'full': None, 'window': 2}


def store(budget=2_000_000, windows=WINDOWS):
    result = PrefixBlocks(budget, block_tokens=1)
    result.configure_retention(windows)
    return result


def publish(cache, tokens, start, stop, sizes=None, tags=()):
    sizes = sizes or {'full': 1, 'window': 100, 'hidden': 1}
    writer = cache.begin(tokens, start, stop, sizes, tags=tags, retain_frontier=True)
    writer.commit({name: (name, start, stop) for name in sizes})
    writer.finish()


def exact(cache, tokens, windows=WINDOWS, tags=()):
    hit = cache.match(tokens, windows, tags=tags)
    assert hit is not None and hit.length == len(tokens)
    hit.close()


def test_surplus_hidden_beats_required_window_under_tight_pressure():
    cache = store()
    tokens = (1, 2, 3)
    for begin in range(3):
        publish(cache, tokens, begin, begin + 1)
    # Remove only incidental lookup/token credits first. The adversarial
    # obligation exceeds window1+hidden1 NET refunds by one byte; hidden2 suffices.
    # Each first-history entry owns its component-name bytes, so the net
    # refunds are GROUP512+window100=612 and GROUP512+hidden1=513.
    cache._drop_path()
    cache._charged -= cache._validated_charge
    cache._validated_tokens, cache._validated_charge, cache._cursor = None, 0, None
    cache.budget_bytes = cache.stats['charged_bytes']
    # Temporary payload614 + its lease512 = total1126 charged bytes.
    temporary = cache.reserve_temporary(1126 - 512)
    temporary.close()
    assert cache._select(tokens, WINDOWS, (), 'hidden')[0] == 3
    assert cache.stats['retained_frontiers'] == 1
    # A subsequent read has its own additional lease/span obligation.
    cache.budget_bytes = 2_000_000
    exact(cache, tokens)
    cache.close()


def test_branches_share_full_support_and_keep_both_frontiers():
    cache = store()
    a, b = (1, 2, 3, 4), (1, 2, 30, 40)
    for start in range(4):
        publish(cache, a, start, start + 1)
    for start in range(2, 4):
        publish(cache, b, start, start + 1)
    exact(cache, a)
    exact(cache, b)
    assert cache.stats['retained_frontiers'] == 2
    roots = [r.groups['full'] for r in cache._records if r.start == 0]
    assert len(roots) == 1 and roots[0].retained == 2
    cache.close()


def test_abort_and_capacity_failure_leave_no_prospective_memberships():
    cache = store()
    tokens = (1, 2, 3)
    publish(cache, tokens, 0, 1)
    writer = cache.begin(tokens, 1, 2, {'full': 1, 'window': 100, 'hidden': 1}, retain_frontier=True)
    writer.abort()
    assert not cache._frontiers
    assert all(g.retained == 0 for r in cache._records for g in r.groups.values())
    # Independent resident support can establish a new honest descriptor.
    assert cache.retain_frontier(tokens, 1)
    assert not cache.retain_frontier(tokens, 3)
    with pytest.raises(PrefixCapacity):
        cache.reserve_temporary(cache.budget_bytes)
    assert cache.stats['write_leases'] == 0
    assert cache.stats['charged_bytes'] <= cache.budget_bytes
    cache.close()


def test_read_pins_survive_whole_frontier_relinquishment():
    cache = store()
    tokens = (1, 2, 3)
    for start in range(3):
        publish(cache, tokens, start, start + 1)
    hit = cache.match(tokens, WINDOWS)
    with pytest.raises(PrefixCapacity):
        cache.reserve_temporary(cache.budget_bytes)
    assert hit.spans('full', 0, 3)
    hit.close()
    cache.clear()
    assert not cache._frontiers and cache.stats['pins'] == 0
    assert cache.stats['charged_bytes'] == 18432 + 1024 + retention_config_charge(WINDOWS)
    cache.close()


def test_tag_identity_and_configuration_are_funded():
    charge = retention_config_charge(WINDOWS)
    with pytest.raises(ValueError):
        validate_prefix_budget(18432 + 1024 + charge - 1, retention_bytes=charge)
    assert validate_prefix_budget(18432 + 1024 + charge, retention_bytes=charge)
    cache = store()
    tokens = (1, 2, 3)
    tags = ((1, 2, ('image', 'one')),)
    for start in range(3):
        publish(cache, tokens, start, start + 1, tags=tags)
    exact(cache, tokens, tags=tags)
    assert not cache.retain_frontier(tokens, 3, tags=((1, 2, ('image', 'other')),))
    cache.close()


def test_native_cold_self_eviction_preserves_latest_owner_closure():
    windows = {'target_full': None, 'target_window': 2048, 'draft_full': None, 'draft_window': 2112}
    per_row = {'target_full': 10280, 'target_window': 204800, 'draft_full': 4096, 'draft_window': 16384, 'hidden': 5632}
    cache = PrefixBlocks(12_667_846_656)
    cache.configure_retention(windows)
    tokens = tuple(range(262130))
    for start in range(0, len(tokens), 1024):
        stop = min(start + 1024, len(tokens))
        publish(cache, tokens, start, stop, {n: b*(stop-start) for n,b in per_row.items()})
    exact(cache, tokens, windows)
    assert cache.stats['evictions'] > 0
    assert cache.stats['retained_frontiers'] == 1
    # Independent lifecycle count: five additions per batch, expiry at most
    # once per sliding/endpoint membership; full memberships never rescan.
    assert cache.stats['frontier_membership_work'] <= 8*257
    assert cache.stats['charged_bytes'] <= cache.budget_bytes
    cache.close()


@pytest.mark.parametrize('after', [False, True])
def test_descriptor_registry_allocation_failure_is_terminal(after):
    cache = store()
    class BrokenRegistry(list):
        def append(self, value):
            if after:
                super().append(value)
            raise MemoryError('descriptor registry allocation')
    cache._frontiers = BrokenRegistry()
    with pytest.raises(MemoryError, match='descriptor registry'):
        cache.begin((1,), 0, 1, {'full': 1, 'window': 100, 'hidden': 1}, retain_frontier=True)
    assert cache.stats['poisoned']
    with pytest.raises(RuntimeError, match='poisoned'):
        cache.reserve_temporary(0)
    cache.close()


def test_append_transition_failure_is_terminal(monkeypatch):
    cache = store()
    tokens = (1, 2)
    publish(cache, tokens, 0, 1)
    frontier = cache._frontiers[0]
    class BrokenRows(deque):
        def __bool__(self):
            raise MemoryError('membership expiry')
    frontier.groups['window'] = BrokenRows(frontier.groups['window'])
    with pytest.raises(MemoryError, match='membership expiry'):
        cache.begin(tokens, 1, 2, {'full': 1, 'window': 100, 'hidden': 1}, retain_frontier=True)
    assert cache.stats['poisoned']
    with pytest.raises(RuntimeError, match='poisoned'):
        cache.reserve_temporary(0)
    cache.close()


def test_endpoint_role_cannot_narrow_full_or_larger_window():
    for windows in ({'hidden': None}, {'hidden': 2}):
        with pytest.raises(ValueError):
            retention_config_charge(windows)
    assert retention_config_charge({'hidden': 1}) > 0


def test_failed_writer_retirement_keeps_payload_and_stops_regrant():
    cache = store()
    writer = cache.begin((1,), 0, 1, {'full': 1, 'window': 100, 'hidden': 1}, retain_frontier=True)
    writer.commit({'full': ('f',), 'window': ('w',), 'hidden': ('h',)})
    class BrokenRetirement(list):
        def remove(self, value):
            raise RuntimeError('writer retirement failed')
    cache._writers = BrokenRetirement(cache._writers)
    before = cache.stats['charged_bytes']
    with pytest.raises(RuntimeError, match='writer retirement failed'):
        writer.finish()
    assert cache.stats['poisoned'] and cache.stats['pins'] == 3
    assert cache.stats['charged_bytes'] == before
    assert all(g.data is not None for r in cache._records for g in r.groups.values())
    with pytest.raises(RuntimeError, match='poisoned'):
        cache.reserve_temporary(0)
    with pytest.raises(RuntimeError, match='no retry'):
        writer.finish()


def test_resident_interior_cut_gets_funded_radix_anchor():
    cache = PrefixBlocks(2_000_000)
    cache.configure_retention(WINDOWS)
    tokens = tuple(range(32))
    publish(cache, tokens, 0, 32, {'full': 32, 'window': 3200, 'hidden': 32})
    assert cache.retain_frontier(tokens, 17)
    interior = [f for f in cache._frontiers if f.cut == 17]
    assert len(interior) == 1 and interior[0].node.depth == 17
    exact(cache, tokens[:17])
    exact(cache, tokens)
    cache.close()


def test_partial_frontier_release_failure_is_terminal():
    cache = store()
    publish(cache, (1,), 0, 1)
    frontier = cache._frontiers[0]
    class BrokenIteration(deque):
        def __iter__(self):
            iterator = super().__iter__()
            yield next(iterator)
            raise MemoryError('partial membership retirement')
    frontier.groups['full'] = BrokenIteration(frontier.groups['full'])
    with pytest.raises(MemoryError, match='partial membership retirement'):
        cache.clear()
    assert cache.stats['poisoned']
    with pytest.raises(RuntimeError, match='poisoned'):
        cache.reserve_temporary(0)


def test_atomic_inside_cut_cannot_be_retained_as_exact_frontier():
    cache = PrefixBlocks(2_000_000)
    cache.configure_retention(WINDOWS)
    tokens = tuple(range(32))
    tags = ((4, 24, ('image', 'atomic')),)
    publish(cache, tokens, 0, 32, {'full': 32, 'window': 3200, 'hidden': 32}, tags=tags)
    before = cache.stats['retained_frontiers']
    assert not cache.retain_frontier(tokens, 17, tags=tags)
    assert cache.stats['retained_frontiers'] == before + 1
    assert cache.stats['provisional_frontiers'] == 1
    assert cache._select(tokens, WINDOWS, tags, 'hidden')[0] != 17
    assert cache.retain_frontier(tokens, 24, tags=tags)
    assert cache.stats['provisional_frontiers'] == 0
    exact(cache, tokens[:24], tags=tags)
    cache.close()


def test_capacity_cleanup_failure_preserves_primary_and_poisons(monkeypatch):
    cache = store()
    tokens = (1, 2)
    publish(cache, tokens, 0, 1)
    primary = PrefixCapacity('admission exhausted')
    cleanup = MemoryError('frontier retirement failed')
    def reject(_extra):
        raise primary
    def fail_release(_frontier):
        raise cleanup
    monkeypatch.setattr(cache, '_ensure', reject)
    monkeypatch.setattr(cache, '_release_frontier', fail_release)
    with pytest.raises(RuntimeError, match='frontier cleanup failed') as caught:
        cache.begin(tokens, 1, 2, {'full': 1, 'window': 100, 'hidden': 1}, retain_frontier=True)
    assert caught.value.__cause__ is primary
    assert primary.__cause__ is cleanup
    assert cache.stats['poisoned']


def test_branch_union_survives_pressure_funded_by_surplus_only():
    cache = store()
    a, b = (1, 2, 3, 4), (1, 2, 30, 40)
    for start in range(4):
        publish(cache, a, start, start + 1)
    for start in range(2, 4):
        publish(cache, b, start, start + 1)
    cache._drop_path()
    cache._charged -= cache._validated_charge
    cache._validated_tokens, cache._validated_charge, cache._cursor = None, 0, None
    # This fresh store's history slots initially own no names; first eviction
    # funds that name, leaving GROUP+payload as each net reclaim obligation.
    surplus = sum(512 + group.nbytes
                  for record in cache._records for name,group in record.groups.items()
                  if group.retained == 0)
    cache.budget_bytes = cache.stats['charged_bytes']
    temporary = cache.reserve_temporary(surplus - 1 - 512)
    assert cache.stats['retained_frontiers'] == 2
    assert cache._select(a, WINDOWS, (), 'hidden')[0] == len(a)
    assert cache._select(b, WINDOWS, (), 'hidden')[0] == len(b)
    temporary.close()
    cache.close()


def test_mutable_tape_change_between_calls_does_not_reuse_prefix_hint():
    cache = store()
    tape = [1, 2, 3]
    publish(cache, tape, 0, 1)
    tape[0] = 99
    publish(cache, tape, 1, 2)
    assert cache.match(tape[:2], WINDOWS) is None
    assert cache._frontiers[0].cut == 1
    cache.close()


@pytest.mark.parametrize('roles', [10, 100])
def test_per_role_owned_container_extent_is_funded(roles):
    import sys
    windows = {f'group{index}': None for index in range(roles)}
    cache = store(5_000_000, windows)
    sizes = {name: 1 for name in windows}
    sizes['hidden'] = 1
    publish(cache, (1,), 0, 1, sizes)
    frontier = cache._frontiers[0]
    # Distinct logical owned containers, excluding shared Group owners, keys,
    # scalar payloads, allocator overhead and physical RSS (not claimed here).
    containers = sys.getsizeof(frontier) + sys.getsizeof(frontier.groups) + sys.getsizeof(frontier.tags)
    containers += sum(sys.getsizeof(rows) + sum(sys.getsizeof(row) for row in rows)
                      for rows in frontier.groups.values())
    assert containers <= frontier.charge
    cache.close()


def test_atomic3000_after_prefix_preserves_provisional_full_dependencies():
    windows = {'target_full': None, 'target_window': 2048, 'draft_full': None, 'draft_window': 2112}
    per_row = {'target_full': 10280, 'target_window': 204800, 'draft_full': 4096, 'draft_window': 16384, 'hidden': 5632}
    tokens = tuple(range(7168))
    tags = ((4096, 7096, ('image', 'span3000')),)
    outcomes = []
    for decline_only in (True, False):
        cache = PrefixBlocks(600_000_000)
        cache.configure_retention(windows)
        if decline_only:
            # Historical failed policy: no dependency protection for atomic
            # interior publication. All other accounting/selection is shared.
            prepare = cache._prepare_frontier
            def decline(tokens, start, stop, incoming, tags):
                if any(begin < stop < end for begin,end,_identity in tags):
                    return None, dict(windows, hidden=1)
                return prepare(tokens, start, stop, incoming, tags)
            cache._prepare_frontier = decline
        for start in range(0, len(tokens), 1024):
            stop = start + 1024
            publish(cache, tokens, start, stop, {name:1024*size for name,size in per_row.items()}, tags=tags)
            if not decline_only and 4096 < stop < 7096:
                assert cache.stats['provisional_frontiers'] == 1
                partial = cache._select(tokens, windows, tags, 'hidden')
                assert partial is None or partial[0] != stop
        selected = cache._select(tokens, windows, tags, 'hidden')
        outcomes.append(0 if selected is None else selected[0])
        if not decline_only:
            assert cache.stats['provisional_frontiers'] == 0
            # Actual indivisible owners: seven full, two target-window, three
            # draft-window and one hidden owner; payload only, not metadata.
            closure = 7168*(10280+4096) + 2048*204800 + 3072*16384 + 1024*5632
            assert closure < 600_000_000 < 7168*sum(per_row.values())
            exact(cache, tokens, windows, tags=tags)
        cache.close()
    assert outcomes[0] < len(tokens) and outcomes[1] == len(tokens)


@pytest.mark.parametrize('capacity_primary', [False, True])
def test_scratch_retirement_preserves_authentic_body_failure(monkeypatch, capacity_primary):
    cache = store()
    primary = PrefixCapacity('body capacity') if capacity_primary else RuntimeError('body failure')
    cleanup = MemoryError('scratch retirement failed')
    reserve = cache.reserve_temporary
    accepted = []
    class FailedScratch:
        def __init__(self, lease):
            self.lease = lease
        def close(self):
            raise cleanup
    def acquire(amount):
        lease = reserve(amount)
        accepted.append(lease)
        return FailedScratch(lease)
    def fail_coverage(*args):
        raise primary
    monkeypatch.setattr(cache, 'reserve_temporary', acquire)
    monkeypatch.setattr(cache, '_coverage', fail_coverage)
    with pytest.raises(RuntimeError, match='frontier cleanup failed') as caught:
        cache.begin((1,), 0, 1, {'full': 1, 'window': 100, 'hidden': 1}, retain_frontier=True)
    assert caught.value.__cause__ is primary
    assert primary.__cause__ is cleanup
    assert cache.stats['poisoned'] and cache.stats['temporary_leases'] == 1
    assert accepted and not accepted[0]._closed
    with pytest.raises(RuntimeError, match='poisoned'):
        cache.reserve_temporary(0)


def test_committed_abort_invalidates_dependent_sibling_frontiers_only():
    cache = store()
    a, b, unrelated = (1, 2), (1, 3), (9, 9)
    writer = cache.begin(a, 0, 2, {'full': 2, 'window': 200, 'hidden': 2}, retain_frontier=True)
    writer.commit({'full': ('a-full',), 'window': ('a-window',), 'hidden': ('a-hidden',)})
    discarded = tuple(writer._record.groups.values())
    publish(cache, b, 1, 2)
    publish(cache, unrelated, 0, 2)
    assert cache.stats['retained_frontiers'] == 3
    writer.abort()
    assert writer._frontier is None
    assert all(group.data is None and group.retained == 0 for group in discarded)
    assert cache.stats['retained_frontiers'] == 1
    assert cache.match(b, WINDOWS) is None
    exact(cache, unrelated)
    cache.close()


def test_older_committed_abort_preserves_advanced_independent_window_closure():
    windows = {'window': 1}
    cache = store(windows=windows)
    tokens = (1, 2)
    writer = cache.begin(tokens, 0, 1, {'window': 100, 'hidden': 1}, retain_frontier=True)
    writer.commit({'window': ('first-window',), 'hidden': ('first-hidden',)})
    advanced = cache.begin(tokens, 1, 2, {'window': 100, 'hidden': 1}, retain_frontier=True)
    advanced.commit({'window': ('second-window',), 'hidden': ('second-hidden',)})
    advanced.finish()
    assert advanced._frontier is None
    writer.abort()
    assert writer._frontier is None and cache.stats['retained_frontiers'] == 1
    exact(cache, tokens, windows)
    cache.close()


def test_interrupted_atomic_provisional_yields_to_useful_complete_branch():
    windows = {'target_full': None, 'target_window': 2048, 'draft_full': None, 'draft_window': 2112}
    per_row = {'target_full': 10280, 'target_window': 204800, 'draft_full': 4096, 'draft_window': 16384, 'hidden': 5632}
    useful = tuple(range(100000, 100064))
    interrupted = tuple(range(7168))
    tags = ((4096, 7096, ('image', 'interrupted3000')),)
    newcomer = tuple(range(200000, 201024))
    outcomes = []
    for old_priority in (True, False):
        cache = PrefixBlocks(600_000_000)
        cache.configure_retention(windows)
        publish(cache, useful, 0, 64, {name:64*size for name,size in per_row.items()})
        for start in range(0, 5120, 1024):
            publish(cache, interrupted, start, start+1024,
                    {name:1024*size for name,size in per_row.items()}, tags=tags)
        assert cache.stats['provisional_frontiers'] == 1
        assert cache.stats['pins'] == 0  # all actual producers positively retired
        if old_priority:
            # Historical policy ranked this unservable cut as complete saved
            # work; lookup remains atomic-aware regardless of this policy bit.
            next(f for f in cache._frontiers if f.provisional).provisional = False
        publish(cache, newcomer, 0, 1024, {name:1024*size for name,size in per_row.items()})
        selected = cache._select(useful, windows, (), 'hidden')
        outcomes.append(0 if selected is None else selected[0])
        if not old_priority:
            assert cache.stats['provisional_frontiers'] == 0
            assert cache.stats['retained_frontiers'] == 2
            assert cache.stats['charged_bytes'] <= cache.budget_bytes
            exact(cache, useful, windows)
            exact(cache, newcomer, windows)
        cache.close()
    assert outcomes[0] < len(useful) and outcomes[1] == len(useful)


def test_partial_final_radix_label_preserves_both_sibling_endpoints_under_pressure():
    windows = {'full': None, 'window': 2048}
    cache = PrefixBlocks(12_000_000)  # 64-token radix labels, not one-token nodes.
    cache.configure_retention(windows)
    original = tuple(range(945))
    canonical = original[:941] + tuple(range(1000000, 1000005))
    publish(cache, original, 0, 4, {'full': 4, 'window': 4, 'hidden': 4})
    publish(cache, original, 4, 945,
            {'full': 941*1024, 'window': 941*1024, 'hidden': 941*5632})
    original_frontier = cache._frontiers[0]
    original_hidden = original_frontier.groups['hidden'][-1][0]
    path, matched = cache._path(canonical, len(canonical))
    assert matched == 941
    assert path[-1][0] is original_frontier.node
    assert (path[-1][0].depth, path[-1][1]) == (945, 941)
    publish(cache, canonical, 941, 946, {'full': 5*1024, 'window': 5*1024, 'hidden': 5*5632})
    assert sorted(frontier.cut for frontier in cache._frontiers) == [945, 946]
    assert original_frontier in cache._frontiers and original_hidden.retained == 1
    # Independent historical endpoint payload has no selected continuation.
    # Its newer score makes the broken policy evict original_hidden first;
    # the correct policy must reclaim this surplus while retaining both cuts.
    surplus_tape = tuple(range(2000000, 2001000))
    writer = cache.begin(surplus_tape, 0, 1000, {'hidden': 3_000_000})
    writer.commit({'hidden': bytes(3_000_000)})
    writer.finish()
    selected_payload = sum(group.nbytes for record in cache._records
                           for group in record.groups.values() if group.retained)
    assert selected_payload + 3_000_000 + 512 < cache.budget_bytes
    assert cache.stats['charged_bytes'] + 3_000_000 + 512 > cache.budget_bytes
    temporary = cache.reserve_temporary(3_000_000)
    assert cache.stats['evictions'] > 0
    assert sorted(frontier.cut for frontier in cache._frontiers) == [945, 946]
    exact(cache, original, windows)
    exact(cache, canonical, windows)
    old = cache.match(original, windows)
    new = cache.match(canonical, windows)
    assert sum(count for _owner,_offset,count in old.spans('hidden',944,945)) == 1
    assert sum(count for _owner,_offset,count in new.spans('hidden',945,946)) == 1
    old.close()
    new.close()
    assert cache.stats['charged_bytes'] <= cache.budget_bytes
    temporary.close()
    cache.close()

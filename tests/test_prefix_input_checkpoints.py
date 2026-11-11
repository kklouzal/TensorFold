"""Observed EOF checkpoints versus replaceable prefill progress; no model proof."""
import pytest

from tensorfold.engine.prefix_blocks import PrefixBlocks, PrefixCapacity

WINDOWS = {'full': None, 'window': 2}


def emit(store, tape, start, stop, sizes, *, checkpoint=False, retained=True):
    writer = store.begin(tape, start, stop, sizes, retain_frontier=retained, checkpoint=checkpoint)
    writer.commit({name: (name, start, stop) for name in sizes})
    writer.finish()


def full(store, tape):
    lease = store.match(tape, WINDOWS)
    assert lease is not None and lease.length == len(tape)
    assert sum(count for _owner,_offset,count in lease.spans('hidden', len(tape)-1, len(tape))) == 1
    lease.close()


def test_observed_input_and_completed_eof_survive_funded_pressure():
    store = PrefixBlocks(2_000_000)
    store.configure_retention(WINDOWS)
    original, completed = (1, 2, 3, 4), (1, 2, 3, 4, 5, 6)
    emit(store, original, 0, 4, {'full': 128, 'window': 100_000, 'hidden': 100_000}, checkpoint=True)
    emit(store, completed, 4, 6, {'full': 64, 'window': 100_000, 'hidden': 100_000}, checkpoint=True)
    assert sorted(f.cut for f in store._frontiers if f.checkpoint) == [4, 6]
    surplus = tuple(range(10000, 11000))
    emit(store, surplus, 0, 1000, {'hidden': 1_000_000}, retained=False)
    stats = store.stats
    metadata = stats['charged_bytes'] - stats['payload_bytes']
    assert stats['selected_payload_bytes'] + metadata + 750_000 + 512 < store.budget_bytes
    assert stats['charged_bytes'] + 750_000 + 512 > store.budget_bytes
    temporary = store.reserve_temporary(750_000)
    assert store.stats['evictions'] > 0 and store.stats['checkpoint_frontiers'] == 2
    full(store, original)
    full(store, completed)
    assert store.stats['pins'] == 0 and store.stats['charged_bytes'] <= store.budget_bytes
    temporary.close()
    store.close()


def test_intermediate_progress_advances_and_checkpoints_remain_soft():
    store = PrefixBlocks(2_000_000)
    store.configure_retention(WINDOWS)
    tape = tuple(range(32))
    for start in range(32):
        emit(store, tape, start, start+1, {'full': 1, 'window': 10, 'hidden': 10}, checkpoint=start==31)
    assert store.stats['retained_frontiers'] == store.stats['checkpoint_frontiers'] == 1
    completed = tape + (32, 33)
    emit(store, completed, 32, 34, {'full': 2, 'window': 20, 'hidden': 20}, checkpoint=True)
    assert store.stats['checkpoint_frontiers'] == 2
    full(store, tape)
    full(store, completed)
    with pytest.raises(PrefixCapacity):
        store.reserve_temporary(store.budget_bytes)
    assert store.stats['checkpoint_frontiers'] == 0 and not store.stats['poisoned']
    assert store.stats['pins'] == 0
    store.close()


def test_fresh_equivalent_lists_deduplicate_checkpoint_with_cursor_elsewhere():
    store = PrefixBlocks(2_000_000)
    store.configure_retention(WINDOWS)
    original, other = tuple(range(16)), tuple(range(100, 116))
    emit(store, original, 0, 16, {'full': 16, 'window': 160, 'hidden': 16}, checkpoint=True)
    original_node = next(f.node for f in store._frontiers if f.cut == 16)
    emit(store, other, 0, 16, {'full': 16, 'window': 160, 'hidden': 16}, checkpoint=True)
    assert store._cursor is not original_node
    before = store.stats
    members = [(g.generation, g.retained, g.pins, id(g.data)) for r in store._records for g in r.groups.values()]
    for index in range(64):
        assert store.retain_frontier(list(original), 16, checkpoint=bool(index % 2))
    after = store.stats
    assert after['checkpoint_frontiers'] == after['retained_frontiers'] == 2
    assert after['charged_bytes'] == before['charged_bytes']
    assert after['frontier_membership_work'] == before['frontier_membership_work']
    assert [(g.generation, g.retained, g.pins, id(g.data)) for r in store._records for g in r.groups.values()] == members
    assert store._cursor is not original_node
    store.close()


def test_archive_checkpoint_is_metadata_only_and_deduplicated(monkeypatch):
    from tests.test_gemma_prefix_archive import archive, fill
    engine, owned = archive()
    tape = tuple(range(32))
    hidden = fill(engine, tape)
    assert owned.capture(tape, 0, 32, hidden, checkpoint_input=True)
    owned.reap()
    before = owned.store.stats
    owners = [(g.generation, id(g.data), g.retained, g.pins) for r in owned.store._records for g in r.groups.values()]
    def forbidden(*args, **kwargs):
        raise AssertionError('metadata checkpoint attempted model/import/projection/wait')
    monkeypatch.setattr(engine.target, 'project_logits', forbidden)
    monkeypatch.setattr(engine._state, 'cache_import_rows', forbidden)
    monkeypatch.setattr(owned, '_wait_oldest', forbidden)
    for _ in range(32):
        assert owned.checkpoint(list(tape))
    assert owned.store.stats['charged_bytes'] == before['charged_bytes']
    assert owned.store.stats['checkpoint_frontiers'] == 1
    assert [(g.generation, id(g.data), g.retained, g.pins) for r in owned.store._records for g in r.groups.values()] == owners
    owned.close()


def test_archive_full_hit_checkpoints_while_real_read_pins_are_owned(monkeypatch):
    from tests.test_gemma_prefix_archive import archive, fill
    engine, owned = archive()
    tape = tuple(range(32))
    hidden = fill(engine, tape)
    assert owned.capture(tape, 0, 32, hidden)
    owned.reap()
    assert owned.store.stats['checkpoint_frontiers'] == 0
    checkpoint = owned.checkpoint
    seen = []
    def observe(tokens, prepared_media=None):
        seen.append(owned.store.stats['pins'])
        return checkpoint(tokens, prepared_media)
    monkeypatch.setattr(owned, 'checkpoint', observe)
    cut, _logits = owned.restore(list(tape), checkpoint_input=True)
    assert cut == 32 and seen and min(seen) > 0
    owned.reap()
    assert owned.store.stats['checkpoint_frontiers'] == 1 and owned.store.stats['pins'] == 0
    owned.close()


def test_archive_healthy_small_budget_identity_declines_without_poison():
    from tests.test_gemma_prefix_archive import archive
    _engine, owned = archive()
    owned.store.close()
    owned.store = PrefixBlocks(60_000)
    owned.store.configure_retention(owned.windows)
    assert owned.identity([i % 100 for i in range(1500)]) is None
    assert not owned.checkpoint([i % 100 for i in range(1500)])
    assert not owned.store.stats['poisoned']
    owned.close()


def test_archive_checkpoint_capacity_decline_never_masks_poison(monkeypatch):
    from tests.test_gemma_prefix_archive import archive
    _engine, owned = archive()
    def declined(*args, **kwargs):
        raise PrefixCapacity('no checkpoint metadata credit')
    monkeypatch.setattr(owned.store, 'retain_frontier', declined)
    assert owned.checkpoint((1,)) is False
    owned.store._poison(RuntimeError('terminal metadata fault'))
    with pytest.raises(RuntimeError, match='failed; retain its owners'):
        owned.checkpoint((1,))
    owned.close()

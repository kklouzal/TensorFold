"""Complete canonical publication skips State imports, never its retirement fence."""
import time

import pytest

from tests.test_gemma_cuda_engine import _engine


def cached(monkeypatch):
    engine = _engine(monkeypatch)
    tokens = [1, 2, 3, 4]
    engine._prefill(tokens, None, None, time.perf_counter())
    assert engine.prefix_cache_identity(tokens).position == len(tokens)
    return engine, tokens


def test_complete_entry_checks_metadata_fence_reap_recheck_without_state_import(monkeypatch):
    engine, tokens = cached(monkeypatch)
    archive = engine._prefix_archive
    engine._state.reset()
    engine._draft.reset()
    order, identity, reap, checkpoint = [], archive.identity, archive.reap, archive.checkpoint
    def query(*args):
        order.append('identity')
        return identity(*args)
    def retire(*, positive_retirement=False):
        if positive_retirement:
            order.append('reap')
        return reap(positive_retirement=positive_retirement)
    def fence():
        assert archive.store.stats['read_leases'] == 0  # no borrowed payload across this fence
        order.append('fence')
    def retain(*args):
        order.append('checkpoint')
        return checkpoint(*args)
    monkeypatch.setattr(archive, 'identity', query)
    monkeypatch.setattr(archive, 'checkpoint', retain)
    monkeypatch.setattr(archive, 'reap', retire)
    engine._stream.synchronize = fence
    monkeypatch.setattr(engine, '_prefill', lambda *a, **k: pytest.fail('unnecessary State restore/import'))
    intent = engine.cache_publication_intent('frontend')
    result = engine.publish_canonical(tokens, intent=intent)
    assert order == ['identity', 'fence', 'reap', 'checkpoint', 'identity']
    assert result['status'] == 'published' and result['complete']
    assert result['matched_tokens'] == result['materialized_tokens'] == len(tokens)
    assert result['computed_tokens'] == 0 and engine._state.position == engine._draft.length == 0
    assert intent.closed
    with pytest.raises(RuntimeError, match='consumed'):
        engine.publish_canonical(tokens, intent=intent)


def test_eviction_between_fence_and_recheck_falls_through_to_real_publication(monkeypatch):
    engine, tokens = cached(monkeypatch)
    archive = engine._prefix_archive
    original, prefill = archive.reap, engine._prefill
    calls, fences = [], []
    def retire(*, positive_retirement=False):
        original(positive_retirement=positive_retirement)
        if positive_retirement and not calls:
            calls.append('evicted')
            archive.store.clear()
    def recompute(*args, **kwargs):
        calls.append('prefill')
        return prefill(*args, **kwargs)
    monkeypatch.setattr(archive, 'reap', retire)
    monkeypatch.setattr(engine, '_prefill', recompute)
    engine._stream.synchronize = lambda: fences.append(True)
    result = engine.publish_canonical(tokens, intent=engine.cache_publication_intent('frontend'))
    assert calls == ['evicted', 'prefill'] and len(fences) == 2
    assert result['complete'] and result['matched_tokens'] == 0 and result['computed_tokens'] == len(tokens)
    assert engine._state.position == engine._draft.length == len(tokens)


def test_partial_entry_still_restores_and_computes_suffix(monkeypatch):
    engine, tokens = cached(monkeypatch)
    tokens.append(5)
    result = engine.publish_canonical(tokens, intent=engine.cache_publication_intent('frontend'))
    assert result['complete'] and result['matched_tokens'] == 4 and result['computed_tokens'] == 1
    assert engine._state.position == engine._draft.length == 5


@pytest.mark.parametrize('failure_stage', ['fence', 'reap', 'checkpoint', 'recheck'])
def test_existing_failure_containment_preserves_first_primary(monkeypatch, failure_stage):
    engine, tokens = cached(monkeypatch)
    archive = engine._prefix_archive
    primary, queries = RuntimeError('publication ' + failure_stage), []
    native_identity, native_reap = archive.identity, archive.reap
    def query(*args):
        queries.append(True)
        if failure_stage == 'recheck' and len(queries) == 2:
            raise primary
        return native_identity(*args)
    def fence():
        if failure_stage == 'fence':
            raise primary
    def reap(*, positive_retirement=False):
        if failure_stage == 'reap' and positive_retirement:
            raise primary
        return native_reap(positive_retirement=positive_retirement)
    engine._stream.synchronize = fence
    if failure_stage == 'checkpoint':
        monkeypatch.setattr(archive, 'checkpoint', lambda *a: (_ for _ in ()).throw(primary))
    monkeypatch.setattr(archive, 'identity', query)
    monkeypatch.setattr(archive, 'reap', reap)
    intent = engine.cache_publication_intent('frontend')
    with pytest.raises(RuntimeError) as caught:
        engine.publish_canonical(tokens, intent=intent)
    assert caught.value is engine._prefix_error is archive.failure is primary
    assert intent.closed
    with pytest.raises(RuntimeError):
        engine._check_open()


@pytest.mark.parametrize('state', ['cancelled', 'metadata_failed'])
def test_existing_declines_do_not_fence_or_restore(monkeypatch, state):
    engine, tokens = cached(monkeypatch)
    if state == 'metadata_failed':
        engine._prefix_archive.metadata_failure = RuntimeError('CPU metadata unavailable')
    engine._stream.synchronize = lambda: pytest.fail('declined publication fenced')
    monkeypatch.setattr(engine, '_prefill', lambda *a, **k: pytest.fail('declined publication restored'))
    result = engine.publish_canonical(tokens, intent=engine.cache_publication_intent('frontend'),
                                     cancelled=(lambda: True) if state == 'cancelled' else None)
    assert result['status'] == state and result['complete'] is False


def test_metadata_failure_during_fence_is_not_reported_complete(monkeypatch):
    engine, tokens = cached(monkeypatch)
    archive = engine._prefix_archive
    def fence():
        archive.metadata_failure = RuntimeError('concurrent optional metadata failure')
    engine._stream.synchronize = fence
    monkeypatch.setattr(engine, '_prefill', lambda *a, **k: pytest.fail('metadata failure restored'))
    result = engine.publish_canonical(tokens, intent=engine.cache_publication_intent('frontend'))
    assert result['status'] == 'metadata_failed' and result['complete'] is False



def test_input_and_canonical_eof_mark_only_actual_boundaries(monkeypatch):
    engine = _engine(monkeypatch)
    archive = engine._prefix_archive
    restore, capture, calls = archive.restore, archive.capture, []
    def import_rows(*args, **kwargs):
        calls.append(('restore', kwargs['checkpoint_input']))
        return restore(*args, **kwargs)
    def export_rows(*args, **kwargs):
        calls.append(('capture', kwargs['checkpoint_input']))
        return capture(*args, **kwargs)
    monkeypatch.setattr(archive, 'restore', import_rows)
    monkeypatch.setattr(archive, 'capture', export_rows)
    engine.generate([1, 2], 1, None, lambda ids: False, stop_eos=False)
    assert calls == [('restore', True), ('capture', True)]
    calls.clear()
    result = engine.publish_canonical([1, 2, 3], intent=engine.cache_publication_intent('frontend'))
    assert result['complete'] and calls == [('restore', True), ('capture', True)]
    calls.clear()
    engine.generate([7, 8], 1, None, lambda ids: False, stop_eos=False, cache_prompt=False)
    assert not calls  # gate continuation cannot replace the logical request's checkpoint



def test_checkpoint_decline_cannot_take_complete_fastpath(monkeypatch):
    engine, tokens = cached(monkeypatch)
    archive, original = engine._prefix_archive, engine._prefill
    calls = []
    monkeypatch.setattr(archive, 'checkpoint', lambda *args: False)
    def restore(*args, **kwargs):
        calls.append(True)
        return original(*args, **kwargs)
    monkeypatch.setattr(engine, '_prefill', restore)
    result = engine.publish_canonical(tokens, intent=engine.cache_publication_intent('frontend'))
    assert calls == [True] and result['complete']  # ordinarypath stillauthoritativelychecksitsresult


def test_tiny_funded_cache_does_not_make_long_publication_a_fatal_error(monkeypatch):
    from tests.test_gemma_cuda_engine import _native_metadata_cache
    from tensorfold.engine.prefix_blocks import PrefixBlocks, retention_config_charge
    from tensorfold.families.gemma4.cuda.prefix_archive import GemmaPrefixArchive
    engine = _native_metadata_cache(monkeypatch)
    windows = dict(engine._prefix_archive.windows)
    engine._prefix_archive.close()
    empty = PrefixBlocks(100000)
    budget = empty.stats['charged_bytes'] + retention_config_charge(windows)
    empty.close()
    engine._prefix_archive = GemmaPrefixArchive(engine, budget)
    tokens = [3] * 500  # valid native context; lookup metadata cannot fit this cache grant
    result = engine.publish_canonical(tokens, intent=engine.cache_publication_intent('frontend'))
    assert result['status'] == 'partial' and result['complete'] is False
    assert result['matched_tokens'] == 0 and result['computed_tokens'] == result['materialized_tokens'] == 500
    assert engine._prefix_error is engine._prefix_archive.failure is None
    assert not engine._prefix_archive.store.stats['poisoned']

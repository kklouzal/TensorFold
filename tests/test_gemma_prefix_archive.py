"""Architecture-A archive CPU transport/ownership; no model/GPU qualification."""
from types import SimpleNamespace

import pytest
import torch

from gemma_cuda_target_metadata_test import tiny_config
from tensorfold.families.gemma4.cuda.state import State, capacity_bytes
from tensorfold.families.gemma4.cuda.weights import Config
from tensorfold.families.gemma4.cuda.dflash import DraftState
from tensorfold.families.gemma4.cuda.prefix_archive import GemmaPrefixArchive
from tensorfold.engine.prefix_blocks import PrefixCapacity


def archive():
    raw = tiny_config()
    raw['text_config']['max_position_embeddings'] = 2048
    cfg = Config.from_dict(raw)
    target = State(cfg, 'cpu', 2048, rows=1024)
    draft_cfg = SimpleNamespace(layer_types=('sliding_attention', 'full_attention'),
                                sliding_window=8, kv_heads=1, head_dim=4)
    draft = DraftState(draft_cfg, 'cpu', 2048, rows=1024, block=5)
    engine = SimpleNamespace(_state=target, _draft=draft, _prefix_draft_kind='dflash',
        context_window=2048, target=SimpleNamespace(config=cfg, project_logits=lambda row: row[:, :3].float().clone()))
    budget = 3 * (capacity_bytes(cfg, 2048, 1024) + sum(t.numel() * t.element_size()
                                                       for t in (*draft.keys, *draft.values)))
    return engine, GemmaPrefixArchive(engine, budget)


def fill(engine, tokens):
    state, draft = engine._state, engine._draft
    for row, token in enumerate(tokens):
        for layer in range(state.raw.shape[0]):
            state.raw[layer, row].view(torch.int16).fill_(0x3f00 + (row + token + layer * 17) % 2048)
            state.inv[layer, row].view(torch.int32).fill_(0x3f000000 + row + token + layer * 17)
        for layer in range(state.keys.shape[0]):
            word = 0x3f00 + (row + token + layer * 19) % 2048
            state.keys[layer, row % state.ring].view(torch.int16).fill_(word)
            state.values[layer, row % state.ring].view(torch.int16).fill_(word + 3)
        for layer, ring in enumerate(draft.rings):
            word = 0x3f00 + (row + token + layer * 23) % 2048
            draft.keys[layer][row % ring].view(torch.int16).fill_(word)
            draft.values[layer][row % ring].view(torch.int16).fill_(word + 5)
    state.position = state.high_water = draft.length = len(tokens)
    hidden = torch.empty((len(tokens), engine.target.config.hidden), dtype=torch.bfloat16)
    for row in range(len(tokens)):
        hidden[row].view(torch.int16).fill_(0x3f00 + row % 2048)
    return hidden


@pytest.mark.parametrize('cut', [1, 17, 63, 64, 65, 517, 1024])
def test_arbitrary_interior_cut_imports_actual_target_draft_and_hidden(cut):
    engine, owned = archive()
    tokens = tuple(i % 100 for i in range(1024))
    hidden = fill(engine, tokens)
    assert owned.capture(tokens, 0, len(tokens), hidden)
    first = owned.identity(tokens[:cut])
    assert first is not None and first.position == cut
    before = tuple(t.data_ptr() for t in (engine._state.raw, engine._state.inv, engine._state.keys, engine._state.values))
    for value in (engine._state.raw, engine._state.inv, engine._state.keys, engine._state.values,
                  *engine._draft.keys, *engine._draft.values):
        value.zero_()
    actual, logits = owned.restore(tokens[:cut])
    assert actual == cut and engine._state.position == engine._state.high_water == engine._draft.length == cut
    assert not engine._state.poisoned
    assert torch.equal(logits, hidden[cut - 1:cut, :3].float())
    assert before == tuple(t.data_ptr() for t in (engine._state.raw, engine._state.inv, engine._state.keys, engine._state.values))
    for row in (0, cut - 1):
        assert torch.all(engine._state.raw[0, row].view(torch.int16) == 0x3f00 + (row + tokens[row]) % 2048)
        assert torch.all(engine._draft.keys[0][row % engine._draft.rings[0]].view(torch.int16)
                         == 0x3f00 + (row + tokens[row]) % 2048)
    assert owned.identity(tokens[:cut]) == first and not owned.pending
    owned.close()


def test_old_interior_fork_survives_source_ring_overwrite_without_whole_prompt_duplicate():
    engine, owned = archive()
    tokens = tuple(i % 100 for i in range(1130))
    hidden = fill(engine, tokens[:1024])
    assert owned.capture(tokens, 0, 1024, hidden)
    original = owned.identity(tokens[:517])
    hidden = fill(engine, tokens)
    assert owned.capture(tokens, 1024, 1130, hidden[1024:])
    query = tokens[:517] + (110,) * 33
    cut, logits = owned.restore(query)
    assert cut == 517 and logits is None
    assert engine._draft.length == engine._state.position == 517
    assert owned.identity(tokens[:517]) == original
    assert torch.all(engine._state.keys[0, 0].view(torch.int16) == 0x3f00)
    owned.close()


def test_true_credit_exhaustion_is_nonpublication_not_native_poison(monkeypatch):
    engine, owned = archive()
    tokens = tuple(range(32))
    hidden = fill(engine, tokens)
    def refused(*args, **kwargs):
        raise PrefixCapacity('no evictable cache credit')
    monkeypatch.setattr(owned.store, 'begin', refused)
    assert owned.capture(tokens, 0, len(tokens), hidden) is False
    assert owned.failure is None and not owned.pending
    assert owned.store.stats['write_leases'] == 0 and owned.store.stats['read_leases'] == 0
    owned.close()


def test_restore_journal_credit_refusal_happens_before_read_borrow(monkeypatch):
    engine, owned = archive()
    called = []
    def refused(*args, **kwargs):
        raise PrefixCapacity('scratch credit')
    monkeypatch.setattr(owned, '_journal', refused)
    monkeypatch.setattr(owned.store, 'match', lambda *args, **kwargs: called.append(True))
    assert owned.restore((1, 2, 3)) == (0, None)
    assert not called and owned.failure is None
    assert owned.store.stats['read_leases'] == 0
    owned.close()


def test_host_credit_tracks_all_pending_claims_and_only_expected_physical_decline(monkeypatch):
    from tensorfold.cuda import capacity
    _engine, owned = archive()
    totals = []
    def admit(total, *, materialized_owned_bytes):
        assert materialized_owned_bytes == 0
        totals.append(total)
        return {'snapshot_is_reservation': False}
    monkeypatch.setattr(capacity, 'admit_host_commitment', admit)
    a = owned.reserve_metadata(100)
    b = owned.reserve_metadata(200)
    assert totals == [a.nbytes, a.nbytes + b.nbytes]
    a.close()
    b.close()
    primary = ValueError('inconsistent physical authority')
    def invalid(*args, **kwargs):
        raise primary
    monkeypatch.setattr(capacity, 'admit_host_commitment', invalid)
    with pytest.raises(ValueError) as caught:
        owned.reserve_metadata(100)
    assert caught.value is primary
    def insufficient(*args, **kwargs):
        raise capacity.HostCommitmentCapacityError('physical headroom')
    monkeypatch.setattr(capacity, 'admit_host_commitment', insufficient)
    with pytest.raises(PrefixCapacity):
        owned.reserve_metadata(100)
    assert owned.failure is None and owned.metadata_failure is None and not owned.host_metadata
    owned.close()


def test_prepare_credit_survives_payload_cold_clear_and_owner_failure_is_cache_only(monkeypatch):
    from tensorfold.cuda import capacity
    monkeypatch.setattr(capacity, 'admit_host_commitment', lambda *args, **kwargs: {})
    engine, owned = archive()
    class Owner:
        def __init__(self):
            self.credit = owned.reserve_metadata(100)
            self.closed = 0
        def close(self):
            self.closed += 1
            self.credit.close()
    first, second = Owner(), Owner()
    a, b = owned.register_metadata_owner(first), owned.register_metadata_owner(second)
    tokens = tuple(range(16))
    assert owned.capture(tokens, 0, 16, fill(engine, tokens))
    credit = owned.store.stats['temporary_bytes']
    owned.clear()  # no metadata refund, even though a prepared request exists
    assert owned.store.stats['payload_bytes'] == 0 and owned.store.stats['temporary_bytes'] == credit
    one, two = MemoryError('first metadata owner'), RuntimeError('second pending owner')
    owned.metadata_failed(one, first)
    owned.metadata_failed(two, second)
    owned.metadata_failed(one, first)
    a.close()
    b.close()  # only declare CPU completion; failed owners remain paid
    assert not first.closed and not second.closed and owned.metadata_failure is one
    with pytest.raises(PrefixCapacity):
        owned.reserve_metadata(1)
    assert owned.failure is None and not engine._state.poisoned
    owned.close()
    assert first.closed == second.closed == 1 and not owned.host_metadata and not owned.metadata_owners


def test_miss_replaces_old_hit_lookup_with_actual_zero_frontier():
    engine, owned = archive()
    tokens = tuple(range(32))
    assert owned.capture(tokens, 0, 32, fill(engine, tokens))
    assert owned.restore(tokens)[0] == 32
    assert owned.restore((100, 101, 102)) == (0, None)
    assert owned.stats()['last_lookup'] == {'requested_tokens': 3, 'matched_tokens': 0, 'materialized_tokens': 0}
    owned.close()


def test_native_import_failure_keeps_target_poisoned_and_pins_paid(monkeypatch):
    engine, owned = archive()
    tokens = tuple(range(32))
    assert owned.capture(tokens, 0, 32, fill(engine, tokens))
    primary = RuntimeError('second native range copy')
    before = engine._state.cache_import_rows
    calls = []
    def failed(group, *args):
        calls.append(group)
        if len(calls) == 2:
            raise primary
        return before(group, *args)
    monkeypatch.setattr(engine._state, 'cache_import_rows', failed)
    with pytest.raises(RuntimeError) as caught:
        owned.restore(tokens[:17])
    assert caught.value is primary and owned.failure is primary and engine._state.poisoned
    stats = owned.store.stats
    assert stats['read_leases'] == 1 and stats['pins'] > 0 and stats['temporary_bytes'] > 0
    assert owned.stats()['last_lookup']['matched_tokens'] == 0
    with pytest.raises(RuntimeError) as rejected:
        owned.restore(tokens)
    assert rejected.value.__cause__ is primary
    # CPU copies are synchronously retired; the owning engine's real native
    # close may make this declaration only after its actual stream succeeds.
    owned.close()


def test_multiple_failed_owner_close_does_not_retry_and_other_owners_retire(monkeypatch):
    from tensorfold.cuda import capacity
    monkeypatch.setattr(capacity, 'admit_host_commitment', lambda *args, **kwargs: {})
    _engine, owned = archive()
    class Owner:
        def __init__(self, fail=False):
            self.credit = owned.reserve_metadata(100)
            self.calls, self.fail = 0, fail
        def close(self):
            self.calls += 1
            if self.fail:
                raise first
            self.credit.close()
    a, b = Owner(True), Owner()
    ra, rb = owned.register_metadata_owner(a), owned.register_metadata_owner(b)
    first = RuntimeError('owner retirement already attempted')
    with pytest.raises(RuntimeError):
        a.close()
    owned.metadata_failed(first, a, close_failed=True)
    owned.metadata_failed(MemoryError('other existing owner'), b)
    ra.close()
    rb.close()
    with pytest.raises(RuntimeError) as closing:
        owned.close()
    assert closing.value is first and a.calls == 1 and b.calls == 1
    assert ra.owner is a and not ra.closed and rb.closed
    assert owned.host_metadata  # unresolved first owner remains fully charged


def test_direct_sdk_raw_response_materializes_terminal_once(monkeypatch):
    from test_gemma_cuda_engine import _engine
    e = _engine(monkeypatch)
    prompt, out = [3, 1, 4], []
    reference = _engine(monkeypatch)
    expected = []
    reference.generate(prompt, 7, None, lambda ids: expected.extend(ids), stop_eos=False)
    stats = e._delivery_generate(prompt, 7, None, lambda ids: out.extend(ids), stop_eos=False)
    assert out == expected
    publication = stats['cache_publication']
    assert publication['complete'] and publication['mode'] == 'raw_ids'
    assert publication['computed_tokens'] == len(out)
    assert publication['materialized_tokens'] == len(prompt) + len(out)
    assert e.prefix_cache_identity(prompt + out).position == len(prompt) + len(out)
    assert e._state.tokens == e._draft.taps == prompt + out


def test_frontend_intent_suppresses_raw_then_publishes_only_projected_divergence(monkeypatch):
    from test_gemma_cuda_engine import _engine
    e = _engine(monkeypatch)
    prompt, out = [3, 1, 4], []
    intent = e.cache_publication_intent('frontend')
    result = e._delivery_generate(prompt, 5, None, lambda ids: out.extend(ids), stop_eos=False,
                                  _cache_publication=intent)
    assert 'cache_publication' not in result and e.prefix_cache_identity(prompt + out) is None
    canonical = [3, 1, 9, 9]
    record = e.publish_canonical(canonical, intent=intent)
    assert record['complete'] and record['matched_tokens'] == 2 and record['computed_tokens'] == 2
    assert e._state.tokens == e._draft.taps == canonical
    assert [pair for pair in e.head.append_roles if pair[1]] == [(3, True), (2, True)]
    e.cancel_cache_publication(intent)  # cleanup idempotent after consumption
    with pytest.raises(RuntimeError, match='consumed'):
        e.publish_canonical(canonical, intent=intent)


def test_prepared_metadata_clear_generate_publish_same_intent(monkeypatch):
    from tensorfold.cuda import capacity
    from test_gemma_cuda_engine import _engine
    monkeypatch.setattr(capacity, 'admit_host_commitment', lambda *args, **kwargs: {})
    e = _engine(monkeypatch)
    credit = e.reserve_cache_metadata(100)
    owner = SimpleNamespace(close=credit.close)
    registered = e.register_cache_metadata_owner(owner)
    intent = e.cache_publication_intent('frontend')
    e.clear_prefix_cache()
    assert e._prefix_archive.store.stats['temporary_bytes'] > 0
    out = []
    e._delivery_generate([1, 2], 3, None, lambda ids: out.extend(ids), stop_eos=False,
                          _cache_publication=intent)
    assert e.publish_canonical([1, 2, *out], intent=intent)['complete']
    owner.close()
    registered.close()
    e.close()
    assert e._closed


def test_raw_sdk_metadata_decline_does_not_reject_generation_or_fallback_publish(monkeypatch):
    from test_gemma_cuda_engine import _engine
    e = _engine(monkeypatch)
    def refused(_amount):
        raise PrefixCapacity('optional raw tape has no credit')
    monkeypatch.setattr(e._prefix_archive, 'reserve_metadata', refused)
    out = []
    result = e._delivery_generate([3, 1, 4], 5, None, lambda ids: out.extend(ids), stop_eos=False)
    assert len(out) == 5 and result['cache_publication']['status'] == 'metadata_unfunded'
    assert result['cache_publication']['complete'] is False
    assert e._prefix_archive.failure is None and e._prefix_archive.metadata_failure is None
    assert e.prefix_cache_identity([3, 1, 4, *out]) is None


def test_raw_metadata_preallocation_retires_when_history_validation_fails_before_lane(monkeypatch):
    from tensorfold.engine import penalties
    from test_gemma_cuda_engine import _engine
    from tensorfold.cuda import capacity
    monkeypatch.setattr(capacity, 'admit_host_commitment', lambda *args, **kwargs: {})
    e = _engine(monkeypatch)
    primary = ValueError('invalid history configuration before native effects')
    def invalid(*args):
        raise primary
    monkeypatch.setattr(penalties, 'History', invalid)
    with pytest.raises(ValueError) as caught:
        e._delivery_generate([1, 2], 3, None, lambda ids: pytest.fail('no generation'),
                             penalties=SimpleNamespace(active=True))
    assert caught.value is primary and not e._prefix_archive.host_metadata
    assert not e._prefix_archive.metadata_owners and e._prefix_archive.failure is None
    assert e._state.position == 0


def test_raw_credit_close_failure_is_cache_only_and_cannot_be_retried(monkeypatch):
    from tensorfold.cuda import capacity
    from tensorfold.families.gemma4.cuda import prefix_archive as module
    from test_gemma_cuda_engine import _engine
    monkeypatch.setattr(capacity, 'admit_host_commitment', lambda *args, **kwargs: {})
    e = _engine(monkeypatch)
    before = module._MetadataCredit.close
    primary = RuntimeError('raw tape credit retirement is uncertain')
    failed = []
    def close(lease):
        if lease.nbytes > 1000:
            failed.append(lease)
            raise primary
        return before(lease)
    monkeypatch.setattr(module._MetadataCredit, 'close', close)
    out = []
    with pytest.raises(RuntimeError) as caught:
        e._delivery_generate([3, 1, 4], 7, None, lambda ids: out.extend(ids), stop_eos=False)
    assert caught.value is primary and len(out) == 7
    assert e._prefix_archive.failure is None and e._prefix_archive.metadata_failure is primary
    assert len(failed) == 1
    next_out = []
    e.generate([3, 1, 4], 2, None, lambda ids: next_out.extend(ids), stop_eos=False)
    assert len(next_out) == 2  # healthy model, no new metadata/cache work
    with pytest.raises(RuntimeError) as closing:
        e.close()
    assert closing.value is primary and len(failed) == 1


@pytest.mark.parametrize('retirement_fails', [False, True])
def test_owned_vision_budget_callbacks_retire_only_after_positive_engine_close(monkeypatch, retirement_fails):
    import gc
    import weakref
    from test_gemma_cuda_engine import _engine
    e = _engine(monkeypatch, with_head=False)
    vision = SimpleNamespace(metadata_credit=e.reserve_cache_metadata,
                             metadata_register=e.register_cache_metadata_owner,
                             metadata_failed=e.cache_metadata_failed)
    e.vision = vision
    reference = weakref.ref(e)
    if retirement_fails:
        primary = RuntimeError('working stream has unknown consumers')
        def fail():
            raise primary
        e._stream.synchronize = fail
        with pytest.raises(RuntimeError) as caught:
            e.close()
        assert caught.value is primary and vision.metadata_credit is not None
    else:
        e.close()
        assert vision.metadata_credit is vision.metadata_register is vision.metadata_failed is None
        del e
        gc.collect()
        assert reference() is None


def test_native_retirement_failure_is_sticky_and_keeps_owners(monkeypatch):
    from test_gemma_cuda_engine import _engine
    e = _engine(monkeypatch)
    primary, calls = RuntimeError('native fence failed'), []
    def failed():
        calls.append(True)
        raise primary
    e._stream.synchronize = failed
    for retire in (e.drain, e.close, e.close):
        with pytest.raises(RuntimeError) as caught:
            retire()
        assert caught.value is primary
    assert len(calls) == 1 and e._prefix_error is primary
    assert e._prefix_archive.failure is primary and not e._prefix_archive.closed


def test_disabled_archive_stats_has_no_removed_records_alias(monkeypatch):
    from test_gemma_cuda_engine import _engine
    e = _engine(monkeypatch)
    e._prefix_archive = None
    stats = e.prefix_cache_stats()
    assert 'records' not in stats and stats['chunks'] == 0


def test_lookup_capacity_is_cold_generation_and_retires_journal(monkeypatch):
    engine, owned = archive()
    def refused(*args, **kwargs):
        raise PrefixCapacity('lookup pin credit unavailable')
    monkeypatch.setattr(owned.store, 'match', refused)
    assert owned.restore((1, 2, 3)) == (0, None)
    assert owned.failure is None and not owned.pending
    assert owned.store.stats['temporary_bytes'] == 0
    assert engine._state.position == engine._draft.length == 0
    owned.close()


def test_journal_registration_failure_accepts_no_temporary_credit():
    _engine, owned = archive()
    primary = MemoryError('pre-effect journal registration')
    class NoAppend(list):
        def append(self, value):
            raise primary
    owned.pending = NoAppend()
    with pytest.raises(MemoryError) as caught:
        owned._journal(2, 256)
    assert caught.value is primary and owned.store.stats['temporary_bytes'] == 0
    assert not owned.pending
    owned.close()


@pytest.mark.parametrize('stage', ['record', 'query', 'synchronize'])
def test_native_event_failure_keeps_accepted_payload_and_credit(monkeypatch, stage):
    engine, owned = archive()
    primary = RuntimeError('native event ' + stage)
    calls = []
    class Event:
        def record(self, stream):
            calls.append('record')
            if stage == 'record':
                raise primary
        def query(self):
            calls.append('query')
            if stage == 'query':
                raise primary
            return False
        def synchronize(self):
            calls.append('synchronize')
            raise primary
    original = owned._journal
    def journal(*args):
        pending = original(*args)
        pending.event = Event()
        return pending
    monkeypatch.setattr(owned, '_journal', journal)
    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: object())
    tokens = tuple(range(32))
    hidden = fill(engine, tokens)
    if stage == 'synchronize':
        assert owned.capture(tokens, 0, len(tokens), hidden)
        with pytest.raises(RuntimeError) as caught:
            owned._credit(lambda: (_ for _ in ()).throw(PrefixCapacity('pressure')))
    else:
        with pytest.raises(RuntimeError) as caught:
            owned.capture(tokens, 0, len(tokens), hidden)
    assert caught.value is primary and owned.failure is primary
    before = tuple(calls)
    with pytest.raises(RuntimeError, match='retain its owners') as refused:
        owned.capture(tokens, 0, len(tokens), hidden)
    assert refused.value.__cause__ is primary and tuple(calls) == before
    assert len(owned.pending) == 1 and owned.pending[0].effects
    assert owned.store.stats['payload_bytes'] > 0 and owned.store.stats['pins'] > 0
    # A real adapter may declare this only after the working stream has been
    # positively retired. The CPU event control supplies that declaration.
    owned.close()
    final = owned.store.stats
    assert not owned.pending and final['closed']
    assert final['payload_bytes'] == final['reserved_payload_bytes'] == final['temporary_bytes'] == 0
    assert final['read_leases'] == final['write_leases'] == final['temporary_leases'] == 0



def test_metadata_table_registration_failure_accepts_no_core_credit():
    _engine, owned = archive()
    primary = MemoryError('metadata table registration')
    class NoInsert(dict):
        def __setitem__(self, key, value):
            raise primary
    owned.host_metadata = NoInsert()
    with pytest.raises(MemoryError) as caught:
        owned.reserve_metadata(100)
    assert caught.value is primary
    assert owned.store.stats['temporary_bytes'] == 0 and not owned.host_metadata
    owned.close()


def test_owner_registration_failure_accepts_no_header_credit():
    _engine, owned = archive()
    primary = MemoryError('owner registry insertion')
    class NoInsert(dict):
        def __setitem__(self, key, value):
            raise primary
    owned.metadata_owners = NoInsert()
    with pytest.raises(MemoryError) as caught:
        owned.register_metadata_owner(SimpleNamespace(close=lambda: None))
    assert caught.value is primary
    assert owned.store.stats['temporary_bytes'] == 0 and not owned.host_metadata
    owned.close()


def test_expected_raw_registration_refusal_cannot_hide_credit_retirement_failure(monkeypatch):
    from tensorfold.cuda import capacity
    from tensorfold.engine.prefix_blocks import TemporaryLease
    monkeypatch.setattr(capacity, 'admit_host_commitment', lambda *args, **kwargs: {})
    _engine, owned = archive()
    primary, secondary = PrefixCapacity('registry credit refused'), RuntimeError('credit retirement failed')
    monkeypatch.setattr(owned, 'register_metadata_owner', lambda owner: (_ for _ in ()).throw(primary))
    monkeypatch.setattr(TemporaryLease, 'close', lambda self: (_ for _ in ()).throw(secondary))
    with pytest.raises(PrefixCapacity) as caught:
        owned.prepare_raw_response([1, 2], 3)
    assert caught.value is primary and caught.value.__cause__ is secondary
    assert owned.metadata_failure is secondary and owned.host_metadata
    assert owned.store.stats['temporary_bytes'] > 0
    # No caller may retry/refund uncertain credit retirement.


def test_generated_unprepared_visual_marker_is_nonpublication_not_generation_failure(monkeypatch):
    from test_gemma_cuda_engine import _engine
    reference, expected = _engine(monkeypatch), []
    reference.generate([1, 2], 3, None, lambda ids: expected.extend(ids), stop_eos=False)
    e = _engine(monkeypatch)
    prepared = SimpleNamespace(cache_metadata_complete=True, token_ids=(1, 2), media_token_ids=(expected[0], 12345))
    def forbidden(*args):
        raise AssertionError('unprepared generated media must not be rebound')
    e.vision = SimpleNamespace(encode_suffix=lambda *args: None, rebind_prepared=forbidden)
    monkeypatch.setattr(e._prefix_archive, 'tags', lambda media: ())
    out = []
    result = e._delivery_generate([1, 2], 3, None, lambda ids: out.extend(ids), vision=prepared, stop_eos=False)
    assert out == expected
    assert result['cache_publication']['status'] == 'generated_media_unprepared'
    assert result['cache_publication']['complete'] is False
    assert e.prefix_cache_identity([1, 2]).position == 2  # initial prompt remains valid
    assert not e._prefix_archive.metadata_owners and not e._prefix_archive.host_metadata

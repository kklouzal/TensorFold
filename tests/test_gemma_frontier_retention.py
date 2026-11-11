"""Archive retention integration; real CPU words and native-size meta geometry."""
import time

import pytest

from tensorfold.engine.prefix_blocks import PrefixBlocks, PrefixCapacity
from tests.test_gemma_prefix_archive import archive, fill
from tests.test_gemma_cuda_engine import _native_metadata_cache


def test_archive_configures_actual_target_draft_requirements_once(monkeypatch):
    calls, original = [], PrefixBlocks.configure_retention
    def configure(self, windows, *, require_end_group):
        calls.append((dict(windows), require_end_group))
        return original(self, windows, require_end_group=require_end_group)
    monkeypatch.setattr(PrefixBlocks, 'configure_retention', configure)
    engine, owned = archive()
    assert calls == [({'target_full': None, 'target_window': engine._state.ring,
                      'draft_full': None, 'draft_window': engine._draft.rings[0]}, 'hidden')]
    owned.close()


def test_complete_projected_frontier_is_admitted_before_export(monkeypatch):
    engine, owned = archive()
    tokens = tuple(range(32))
    hidden = fill(engine, tokens)
    order, begin, export = [], owned.store.begin, engine._state.cache_export_rows
    def planned(*args, **kwargs):
        assert kwargs['retain_frontier'] is True and not order
        result = begin(*args, **kwargs)
        order.append('planned')
        return result
    def exported(*args, **kwargs):
        assert order == ['planned']
        order.append('export')
        return export(*args, **kwargs)
    monkeypatch.setattr(owned.store, 'begin', planned)
    monkeypatch.setattr(engine._state, 'cache_export_rows', exported)
    assert owned.capture(tokens, 0, 32, hidden)
    assert order == ['planned', 'export'] and owned.identity(tokens).position == 32
    assert owned.store.stats['charged_bytes'] <= owned.store.budget_bytes
    owned.close()


def test_fully_existing_payload_renews_frontier_under_read_pins_without_copy(monkeypatch):
    engine, owned = archive()
    tokens = tuple(range(32))
    hidden = fill(engine, tokens)
    assert owned.capture(tokens, 0, 32, hidden)
    before = owned.store.stats
    calls, original = [], owned.store.retain_frontier
    def retain(*args, **kwargs):
        assert owned.store.stats['pins'] > 0
        calls.append(args[1])
        return original(*args, **kwargs)
    monkeypatch.setattr(owned.store, 'retain_frontier', retain)
    monkeypatch.setattr(engine._state, 'cache_export_rows', lambda *a, **k: pytest.fail('redundant payload export'))
    assert owned.capture(tokens, 0, 32, hidden)
    after = owned.store.stats
    assert calls == [32] and before['payload_bytes'] == after['payload_bytes']
    assert after['read_leases'] == after['write_leases'] == after['pins'] == 0
    owned.close()


def test_frontier_credit_refusal_precedes_export_and_remains_optional(monkeypatch):
    engine, owned = archive()
    tokens = tuple(range(32))
    hidden = fill(engine, tokens)
    def refused(*args, **kwargs):
        assert kwargs['retain_frontier'] is True
        raise PrefixCapacity('complete frontier cannot be funded')
    monkeypatch.setattr(owned.store, 'begin', refused)
    monkeypatch.setattr(engine._state, 'cache_export_rows', lambda *a, **k: pytest.fail('export after refusal'))
    assert owned.capture(tokens, 0, 32, hidden) is False
    assert owned.failure is None and not owned.pending
    assert owned.store.stats['temporary_bytes'] == owned.store.stats['pins'] == 0
    owned.close()


def test_native262128_and_262130_latest_closures_survive_same_old_byte_budget(monkeypatch):
    engine = _native_metadata_cache(monkeypatch)
    assert engine._prefix_budget_bytes == 12667846656
    original = tuple([3] * 262120 + [11] + [4] * 7)
    engine._prefill(original, None, None, time.perf_counter())
    assert engine.prefix_cache_identity(original).position == 262128
    published = original + (156274, 106)
    _, _, cached = engine._prefill(published, None, None, time.perf_counter(), _need_logits=False)
    assert cached == 262128  # actualv8 counterexamplewas0/full262130recompute
    identity = engine.prefix_cache_identity(published)
    assert identity is not None and identity.position == 262130
    stats = engine.prefix_cache_stats()
    assert stats['charged_bytes'] <= stats['byte_budget'] == 12667846656
    assert stats['pins'] == stats['pending_batches'] == stats['temporary_bytes'] == 0
    assert stats['reserved_payload_bytes'] == stats['uncertain_payload_bytes'] == 0
    # Native tensor geometry/retained logical credit only: meta has no GPU data,
    # so this is not allocator/physical capacity/model/performance evidence.


def test_actual_profile_price_equals_constructed_archive_requirements(tmp_path):
    import json
    from gemma_cuda_target_metadata_test import tiny_config
    from test_gemma_cuda_dflash_metadata import config as draft_config
    from tensorfold.families.gemma4.cuda.capacity import archive_retention_charge
    from tensorfold.engine.prefix_blocks import retention_config_charge
    model, draft = tmp_path / 'model', tmp_path / 'draft'
    model.mkdir()
    draft.mkdir()
    (model / 'config.json').write_text(json.dumps(tiny_config()))
    (draft / 'config.json').write_text(json.dumps(draft_config()))
    _engine, owned = archive()
    assert archive_retention_charge(model, draft) == retention_config_charge(owned.windows, require_end_group='hidden')
    owned.close()


@pytest.mark.parametrize('caller', ['capacity', 'engine'])
def test_configured_budget_refuses_before_torch_and_weights(caller, monkeypatch, tmp_path):
    import ast
    import inspect
    import textwrap
    from tensorfold.families.gemma4.cuda import capacity
    from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
    from tensorfold.engine.prefix_blocks import retention_config_charge
    empty = PrefixBlocks(1 << 20)
    base = empty.stats['charged_bytes']
    empty.close()
    fee = retention_config_charge({'target_full': None, 'target_window': 2048,
                                   'draft_full': None, 'draft_window': 2112})
    monkeypatch.setattr(capacity, 'archive_retention_charge', lambda *args: fee)
    function = capacity.admit if caller == 'capacity' else Gemma4Engine.__init__
    with pytest.raises(ValueError, match='empty index metadata'):
        if caller == 'capacity':
            capacity.admit(tmp_path, 262144, True, prefix_cache_bytes=base + fee - 1)
        else:
            Gemma4Engine(tmp_path, prefix_cache_bytes=base + fee - 1)
    source = ast.parse(textwrap.dedent(inspect.getsource(function)))
    body = source.body[0].body
    before_torch = body[:next(i for i, n in enumerate(body) if isinstance(n, ast.Import)
                              and any(alias.name == 'torch' for alias in n.names))]
    assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
               and n.func.id == 'archive_retention_charge' for statement in before_torch for n in ast.walk(statement))


@pytest.mark.parametrize('path', ['begin', 'retain'])
def test_poisoned_capacity_cleanup_is_not_swallowed_as_optional_miss(monkeypatch, path):
    engine, owned = archive()
    tokens = tuple(range(32))
    hidden = fill(engine, tokens)
    if path == 'retain':
        assert owned.capture(tokens, 0, 32, hidden)
    primary, secondary = PrefixCapacity('frontier credit'), RuntimeError('lease cleanup failed')
    primary.__cause__ = secondary
    def broken(*args, **kwargs):
        owned.store._poison(primary)
        raise primary
    monkeypatch.setattr(owned.store, 'begin' if path == 'begin' else 'retain_frontier', broken)
    monkeypatch.setattr(engine._state, 'cache_export_rows', lambda *a, **k: pytest.fail('producer after failed cleanup'))
    with pytest.raises(PrefixCapacity) as caught:
        owned.capture(tokens, 0, 32, hidden)
    assert caught.value is owned.failure is primary and primary.__cause__ is secondary
    assert owned.pending  # no false resource refund on a poisoned outcome
    owned.close()  # CPU stand-in: caller positively retires, no native producer was accepted


def test_storage_chunks_inside_atomic_media_only_omit_intermediate_frontier():
    import numpy as np
    from types import SimpleNamespace
    from gemma_cuda_target_metadata_test import tiny_config
    from tensorfold.families.gemma4.cuda.weights import Config
    from tensorfold.families.gemma4.cuda.state import State
    from tensorfold.families.gemma4.cuda.dflash import DraftState
    from tensorfold.families.gemma4.cuda.prefix_archive import GemmaPrefixArchive
    from tensorfold.families.gemma4.cuda.vision import VisionFrame, PreparedGemmaVision, _CacheMetadata, _CacheData, prefix_media_tags
    raw = tiny_config()
    raw['text_config']['max_position_embeddings'] = 4096
    cfg = Config.from_dict(raw)
    target = State(cfg, 'cpu', 4096, rows=2048)  # whole1500-token bidirectional frame fits nativeforward
    draft = DraftState(SimpleNamespace(layer_types=('sliding_attention', 'full_attention'),
        sliding_window=8, kv_heads=1, head_dim=4), 'cpu', 4096, rows=2048, block=5)
    engine = SimpleNamespace(_state=target, _draft=draft, _prefix_draft_kind='dflash',
        target=SimpleNamespace(config=cfg, project_logits=lambda row: row[:, :3].float().clone()))
    owned = GemmaPrefixArchive(engine, 8 << 20)
    tokens = (3,) + (90,) * 1500 + (4,)
    frame = VisionFrame(np.zeros((1500, 3), dtype=np.float32), np.zeros((1500, 2), dtype=np.int64), 1500, 'image')
    metadata = _CacheMetadata(SimpleNamespace(close=lambda: None), 1, lambda *a: None)
    metadata.registration = SimpleNamespace(failed=False, retiring=False)
    identity = ('gemma-prepared-frame-v1', 'image', 'a' * 64, 0, b'', 1500,
                frame.pixels.shape, frame.positions.shape, 'b' * 64, 'binding')
    metadata.data = _CacheData((90, 91), (identity,), (), (), 'binding', object())
    media = PreparedGemmaVision(tokens, (frame,), ((1, 1501),), (), (), metadata)
    assert prefix_media_tags(media) == ((1, 1501, identity),)
    from tensorfold.families.gemma4.cuda.engine import _prefill_end
    start = 0
    native_chunks = []
    while start < len(tokens):
        end = _prefill_end(start, len(tokens), 1024, media.blocks)
        native_chunks.append((start, end))
        hidden = fill(engine, tokens[:end])
        assert owned.capture(tokens, start, end, hidden[start:end], media)
        start = end
    assert native_chunks == [(0, 1), (1, 1501), (1501, 1502)]
    assert owned.identity(tokens, media).position == 1502 and owned.failure is None
    # Only storage cuts split the frame. An actual full-query mismatch inside
    # its tag can never expose an interior restore cut.
    query = tokens[:1024] + (99,) + tokens[1025:]
    read = owned.store.match(query, owned.windows, tags=prefix_media_tags(media))
    assert read is None or read.length <= 1
    if read is not None:
        read.close()
    hidden = fill(engine, tokens)
    assert owned.capture(tokens, 0, len(tokens), hidden, media)  # all-existing atomic cuts stay optional
    owned.close()

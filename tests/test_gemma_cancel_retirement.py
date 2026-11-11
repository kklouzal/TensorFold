"""Producer-stop/cancel-retirement order; CPU stand-ins, no target effects."""
import pytest

from tensorfold.cuda import health
from tensorfold.server.cancellation import RequestCancelled
from tests.test_cuda_canonical_publication import app
from tests.test_gemma_cuda_engine import _engine
from tests.test_gemma_dflash_graph import engine_for, runtime as runtime  # pytest fixture re-export


def held_prompt(monkeypatch):
    engine = _engine(monkeypatch)
    archive = engine._prefix_archive
    monkeypatch.setattr(archive, '_record', lambda pending: setattr(pending, 'recorded', True))
    engine.generate([1, 2, 3], 2, None, lambda ids: False, stop_eos=False)
    assert archive.pending and archive.store.stats['pins'] > 0
    return engine, archive


def test_existing_native_fence_precedes_positive_archive_retirement(monkeypatch):
    engine, archive = held_prompt(monkeypatch)
    order, original = [], archive.reap
    engine._stream.synchronize = lambda: order.append('fence')
    def reap(*, positive_retirement=False):
        assert order == ['fence'] and positive_retirement
        order.append('reap')
        return original(positive_retirement=True)
    monkeypatch.setattr(archive, 'reap', reap)
    engine.drain()
    assert order == ['fence', 'reap']
    assert not archive.pending and archive.store.stats['pins'] == archive.store.stats['temporary_bytes'] == 0


def test_failed_fence_keeps_real_journal_pins_and_never_retries(monkeypatch):
    engine, archive = held_prompt(monkeypatch)
    primary, calls = RuntimeError('failed native fence'), []
    before = archive.store.stats
    def fail():
        calls.append('fence')
        raise primary
    engine._stream.synchronize = fail
    monkeypatch.setattr(archive, 'reap', lambda **kwargs: pytest.fail('reap before positive fence'))
    for operation in (engine.drain, engine.drain, engine.close):
        with pytest.raises(RuntimeError) as caught:
            operation()
        assert caught.value is primary
    assert calls == ['fence'] and archive.pending
    assert archive.store.stats['pins'] == before['pins'] and archive.store.stats['temporary_bytes'] == before['temporary_bytes']


def test_reap_failure_after_fence_latches_without_second_native_attempt(monkeypatch):
    engine = _engine(monkeypatch)
    calls, primary = [], RuntimeError('lease retirement failed')
    engine._stream.synchronize = lambda: calls.append('fence')
    def fail(**kwargs):
        assert kwargs == {'positive_retirement': True}
        raise primary
    monkeypatch.setattr(engine._prefix_archive, 'reap', fail)
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            engine.drain()
        assert caught.value is primary
    assert calls == ['fence'] and engine._prefix_error is primary


def test_generator_consumer_close_does_not_poison_graph(request):
    graph_runtime = request.getfixturevalue('runtime')
    engine = engine_for(graph_runtime)
    primary = GeneratorExit('consumer closed')
    with pytest.raises(GeneratorExit) as caught:
        with engine._lane(False):
            raise primary
    assert caught.value is primary and graph_runtime.owner.failure is None and engine._owner is None


@pytest.mark.parametrize('mode', ['decode', 'prefill', 'generator'])
def test_cancellation_retires_after_producer_before_health_turns_metadata(tmp_path, mode):
    frontend, engine = app(tmp_path)
    order, flag = [], [False]
    primary = GeneratorExit('consumer closed') if mode == 'generator' else RequestCancelled('prefill cancelled')
    if mode != 'decode':
        def generate(prompt, count, sampling, feed, draft=True, _cache_publication=None, progress=None):
            order.append('producer-start')
            try:
                if mode == 'prefill':
                    flag[0] = True
                    assert progress(len(prompt), 0, 1, 1.)
                raise primary
            finally:
                order.append('producer-stop')
        engine.generate = generate
    def drain():
        if mode == 'decode':
            assert engine.calls[-1]['done']
        else:
            assert order[-1] == 'producer-stop'
        assert frontend._turns().busy and health.of(frontend).live
        assert all(not lease.closed for lease in engine.metadata_leases)
        order.extend(('fence', 'reap'))
    engine.drain = drain
    body = {'messages': [{'role': 'user', 'content': 'public'}], 'max_tokens': 3, 'stream': True}
    with pytest.raises(GeneratorExit if mode == 'generator' else RequestCancelled) as caught:
        frontend.run(body, True, lambda delta: False if mode == 'decode' else True,
                     cancelled=lambda: flag[0])
    if mode != 'decode':
        assert caught.value is primary
    assert order[-2:] == ['fence', 'reap'] and not frontend._turns().busy
    assert not health.of(frontend).live and all(lease.closed for lease in engine.metadata_leases)
    assert engine.metadata_failure is None


@pytest.mark.parametrize('kind', ['normal', 'stop', 'unstarted', 'unsupported'])
def test_normal_stop_and_unstarted_or_unsupported_paths_do_not_fence(tmp_path, kind):
    frontend, engine = app(tmp_path)
    if kind == 'unsupported':
        engine.supports_canonical_prefix_archive = False
    engine.drain = lambda: pytest.fail('not a started owned native cancellation')
    body = {'messages': [{'role': 'user', 'content': 'public'}], 'max_tokens': 3}
    if kind == 'stop':
        body['stop'] = 'b'
    if kind == 'unstarted':
        with pytest.raises(RequestCancelled):
            frontend.run(body, True, lambda delta: True, cancelled=lambda: True)
        assert not engine.calls
    elif kind == 'unsupported':
        with pytest.raises(RequestCancelled):
            frontend.run(body, True, lambda delta: False)
    else:
        frontend.run(body, True, lambda delta: True)



def test_zero_cache_positive_native_fence_needs_no_archive(monkeypatch):
    engine = _engine(monkeypatch)
    engine._prefix_archive = None
    calls = []
    engine._stream.synchronize = lambda: calls.append('fence')
    engine.drain()
    assert calls == ['fence'] and engine._native_retirement_error is None


def test_failed_cancel_fence_keeps_primary_and_retires_independent_metadata(tmp_path):
    frontend, engine = app(tmp_path)
    primary, secondary = RequestCancelled('producer cancelled'), RuntimeError('native fence failed')
    engine.failure = primary
    def fail():
        assert frontend._turns().busy and health.of(frontend).live
        raise secondary
    engine.drain = fail
    body = {'messages': [{'role': 'user', 'content': 'public'}], 'max_tokens': 3}
    with pytest.raises(RequestCancelled) as caught:
        frontend.run(body, True, lambda delta: True)
    assert caught.value is primary
    assert not frontend._turns().busy and not health.of(frontend).live
    assert all(lease.closed for lease in engine.metadata_leases)
    assert primary.__cause__ is secondary or primary.__context__ is secondary

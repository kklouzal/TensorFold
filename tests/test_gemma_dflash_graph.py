"""DFlash graph dispatch, ownership, failure and header-admission contracts on CPU."""

from contextlib import contextmanager
from types import SimpleNamespace as NS
import gc
import threading

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')

from tensorfold.families.gemma4.cuda import capacity, dflash  # noqa: E402
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine  # noqa: E402


@pytest.fixture
def runtime(monkeypatch):
    """CUDA API adapter; the tiny body validates dispatch, not CUDA arithmetic."""
    events, active = [], {'capture': None, 'fail': None, 'graphs': 0}

    class Stream:
        def __init__(self, device=None, name='warm'):
            self.name = name
        def synchronize(self):
            events.append(('sync', self.name))
        def wait_stream(self, other):
            events.append(('wait', self.name, other.name))

    current = Stream(name='current')

    class Graph:
        def __init__(self):
            active['graphs'] += 1
            self.run = None
        def replay(self):
            events.append(('replay',))
            if active['fail'] == 'replay':
                raise RuntimeError('native replay failed')
            self.run()

    @contextmanager
    def stream(value):
        yield value

    @contextmanager
    def capture(graph, **kwargs):
        assert kwargs['capture_error_mode'] == 'thread_local'
        active['capture'] = graph
        try:
            yield
        finally:
            active['capture'] = None

    def pool():
        events.append(('pool',))
        return object()

    monkeypatch.setattr(torch.cuda, 'current_stream', lambda device: current)
    monkeypatch.setattr(torch.cuda, 'Stream', Stream)
    monkeypatch.setattr(torch.cuda, 'stream', stream)
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', Graph)
    monkeypatch.setattr(torch.cuda, 'graph', capture)
    monkeypatch.setattr(torch.cuda, 'graph_pool_handle', pool)
    head = NS(block=5, mask_id=4, device=torch.device('cpu'))
    head.raw_embed_tokens = lambda ids: ids.reshape(-1, 1)

    def staged(state, x, length, slots):
        def value():
            return (x[1:] + x[0] + state.pos + slots).float().expand(-1, 7)
        out = value().clone()
        if active['capture'] is not None:
            if active['fail'] == 'capture':
                raise RuntimeError('native capture failed')
            active['capture'].run = lambda: out.copy_(value())
        return out

    head._logits_staged = staged
    return NS(owner=dflash._LogitsGraph(head), events=events, active=active, current=current)


def state(length=17):
    return NS(length=length, pos=torch.zeros(1, dtype=torch.int32))


def engine_for(runtime, *, enabled=True):
    engine = Gemma4Engine.__new__(Gemma4Engine)
    engine._dflash_graph, engine._closed, engine._owner = enabled, False, None
    engine._closed_lock, engine._run_lock = threading.Lock(), threading.Lock()
    engine._stream, engine.head = runtime.current, NS(_logits_graph=runtime.owner)
    return engine


def test_replay_refreshes_anchor_and_position_and_preserves_owning_outputs(runtime):
    owner, lane = runtime.owner, state()
    first = owner.forward(lane, 2, 5, 1)
    saved = first.clone()
    lane.length = 18
    second = owner.forward(lane, 3, 5, 1)
    assert runtime.active['graphs'] == 1
    assert first.data_ptr() != owner.output.data_ptr() != second.data_ptr()
    assert torch.equal(first, saved)
    assert torch.equal(second, torch.full((4, 7), 26.0))
    assert lane.length == 18 and lane.pos.item() == 18


@pytest.mark.parametrize('change', ['length', 'slots', 'owner'])
def test_actual_state_or_geometry_change_retires_before_replacing_pool(runtime, change):
    owner, lane = runtime.owner, state()
    first = owner.forward(lane, 2, 5, 1)
    expected = first.clone()
    runtime.events.clear()
    other = state() if change == 'owner' else lane
    owner.forward(other, 3, 3 if change == 'length' else 5, 2 if change == 'slots' else 1)
    assert runtime.events[0] == ('sync', 'current')
    assert runtime.events.index(('sync', 'current')) < runtime.events.index(('pool',))
    assert runtime.active['graphs'] == 2 and owner.state is other
    assert torch.equal(first, expected)
    owner.close()  # caller has already retired its stream
    owner.close()
    assert owner.graph is owner.pool is owner.output is owner.ids is None
    with pytest.raises(RuntimeError, match='closed'):
        owner.forward(other, 2, 5, 1)


@pytest.mark.parametrize('failure', ['capture', 'replay'])
def test_native_failure_retains_generation_and_stops_before_more_cuda(runtime, failure):
    runtime.active['fail'] = failure
    enabled = gc.isenabled()
    with pytest.raises(RuntimeError, match=f'native {failure} failed') as caught:
        runtime.owner.forward(state(), 2, 5, 1)
    assert runtime.owner.failure is caught.value and runtime.owner.graph is not None
    assert gc.isenabled() is enabled
    events = list(runtime.events)
    with pytest.raises(RuntimeError, match='process containment'):
        runtime.owner.forward(state(), 3, 5, 1)
    with pytest.raises(RuntimeError, match='process containment'):
        runtime.owner.close()
    engine = engine_for(runtime)
    with pytest.raises(RuntimeError, match='process containment'):
        engine._check_open()
    with pytest.raises(RuntimeError, match='process containment'):
        engine.drain()
    assert runtime.events == events


@pytest.mark.parametrize('stage', ['proposal readback', 'target probabilities'])
def test_late_operation_fault_keeps_original_roots_and_blocks_retirement(runtime, stage):
    engine = engine_for(runtime)
    primary, native_cause, native_context = RuntimeError(stage), OSError('driver cause'), LookupError('native context')

    def late_readback():
        try:
            raise native_context
        except LookupError:
            raise primary from native_cause

    with pytest.raises(RuntimeError, match=stage) as caught:
        with engine._lane(False):
            runtime.owner.forward(state(), 2, 5, 1)
            assert runtime.owner.failure is None  # the graph itself returned successfully
            late_readback()
    assert caught.value is runtime.owner.failure is primary
    assert primary.__cause__ is native_cause and primary.__context__ is native_context
    assert engine._owner is None and runtime.owner.graph is not None
    events = list(runtime.events)
    with pytest.raises(RuntimeError, match='process containment'):
        engine.drain()
    with pytest.raises(RuntimeError, match='process containment'):
        engine._check_open()
    with pytest.raises(RuntimeError, match='process containment'):
        engine.close()
    assert runtime.events == events


def test_first_drain_fault_is_latched_before_any_second_cuda_attempt(runtime):
    engine = engine_for(runtime)
    primary, calls = RuntimeError('final append failure'), []

    def synchronize():
        calls.append('CUDA synchronization')
        raise primary

    engine._stream = NS(synchronize=synchronize)
    with pytest.raises(RuntimeError, match='final append failure') as caught:
        engine.drain()
    assert caught.value is runtime.owner.failure is primary
    with pytest.raises(RuntimeError, match='process containment'):
        engine.drain()
    with pytest.raises(RuntimeError, match='process containment'):
        engine._check_open()
    assert calls == ['CUDA synchronization']


def test_normal_request_cancel_remains_recoverable_after_graph_work(runtime):
    from tensorfold.server.cancellation import RequestCancelled

    engine = engine_for(runtime)
    cancelled = RequestCancelled('cooperative cancellation')
    with pytest.raises(RequestCancelled) as caught:
        with engine._lane(False):
            runtime.owner.forward(state(), 2, 5, 1)
            raise cancelled
    assert caught.value is cancelled and runtime.owner.failure is None and engine._owner is None
    engine._check_open()
    engine.drain()
    with engine._lane(False):
        runtime.owner.forward(state(), 3, 5, 1)
    assert runtime.owner.failure is None


def test_cancellation_type_import_failure_occurs_before_lane_work(runtime, monkeypatch):
    import builtins

    engine = engine_for(runtime)
    events = list(runtime.events)
    original = builtins.__import__
    primary = MemoryError('cancellation import failed')

    def unavailable(name, *args, **kwargs):
        if name == 'tensorfold.server.cancellation':
            raise primary
        return original(name, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(builtins, '__import__', unavailable)
        with pytest.raises(MemoryError, match='cancellation import failed') as caught:
            with engine._lane(False):
                pytest.fail('mutable GPU operation was entered')
    assert caught.value is primary
    assert runtime.owner.failure is None and engine._owner is None and runtime.events == events
    with engine._lane(False):
        pass  # admission/import failure did not poison an operation that never began


def test_routine_prompt_validation_does_not_poison_graph_owner(runtime):
    engine = engine_for(runtime)
    engine.target = NS(config=NS(vocab=7))
    events = list(runtime.events)
    with pytest.raises(ValueError, match='nonempty sequence'):
        engine.generate([], 1, None, lambda tokens: False)
    assert runtime.owner.failure is None and engine._owner is None and runtime.events == events
    engine._check_open()


def test_disabled_graph_keeps_existing_lane_and_drain_error_policy(runtime):
    engine = engine_for(runtime, enabled=False)
    primary, calls = RuntimeError('legacy operation'), []
    with pytest.raises(RuntimeError) as caught:
        with engine._lane(False):
            raise primary
    assert caught.value is primary and runtime.owner.failure is None and engine._owner is None
    engine._check_open()

    def synchronize():
        calls.append('legacy synchronization')
        raise primary

    engine._stream = NS(synchronize=synchronize)
    for _ in range(2):
        with pytest.raises(RuntimeError) as caught:
            engine.drain()
        assert caught.value is primary and runtime.owner.failure is None
    assert len(calls) == 2  # this candidate does not change the default-off policy


@pytest.mark.parametrize('status', ['failed', 'closed'])
def test_make_state_refuses_failed_or_closed_head_before_allocating(runtime, monkeypatch, status):
    if status == 'failed':
        runtime.owner.failure = RuntimeError('native failure')
    else:
        runtime.owner.close()
    head = dflash.DFlash.__new__(dflash.DFlash)
    head._logits_graph = runtime.owner
    monkeypatch.setattr(dflash, 'DraftState', lambda *args: pytest.fail('new device allocation'))
    with pytest.raises(RuntimeError, match='process containment' if status == 'failed' else 'closed'):
        head.make_state()


@pytest.mark.parametrize('rows', [2, 3, 4, 5, 16])
def test_small_lane_profiles_are_explicitly_admitted(rows):
    capacity.check_dflash_graph(True, 8, rows, 'dflash')


@pytest.mark.parametrize('flag,bits,rows,kind', [
    (1, 8, 5, 'dflash'), (None, 8, 5, 'dflash'), (True, 16, 5, 'dflash'),
    (True, 8, 1, 'dflash'), (True, 8, 17, 'dflash'), (True, 8, 5, None),
    (True, 8, 5, 'sparkulator'),
])
def test_invalid_candidate_policy_rejected_before_gpu_allocation(flag, bits, rows, kind):
    with pytest.raises(ValueError, match='dflash_graph'):
        capacity.check_dflash_graph(flag, bits, rows, kind)


def test_default_policy_does_not_narrow_existing_profiles():
    for bits in (8, 16):
        for rows in (1, 16, 128):
            capacity.check_dflash_graph(False, bits, rows, None)


def test_payload_estimate_charges_context_partials_and_fresh_head_output():
    cfg = NS(hidden=128, width=256, heads=4, kv_heads=2, head_dim=64,
             layer_types=('sliding_attention', 'full_attention'), causal_layers=(True, False),
             sliding_window=32, vocab=262144)
    before = capacity.dflash_graph_workspace(cfg, 5, 1024)
    after = capacity.dflash_graph_workspace(cfg, 5, 1025)
    assert after - before == 5 * 4 * (64 + 2) * 4
    # Independent mandatory payload: all four head/softcap allocations plus
    # a distinct owning caller return, before other activations are charged.
    assert before >= 5 * 4 * cfg.vocab * 4
    assert capacity.dflash_graph_workspace(cfg, 5, 1) == 5 * 4


def test_header_admission_adds_one_graph_generation_without_snapshot_multiplier(tmp_path, monkeypatch):
    from types import ModuleType
    import sys

    from gemma_cuda_target_metadata_test import inventory, tiny_config
    from test_gemma_cuda_dflash_metadata import config, write_checkpoint
    from tensorfold.cuda import capacity as common

    raw = tiny_config()
    text = raw['text_config']
    _, where = inventory(raw)
    headers = {name: {'dtype': value[3], 'shape': value[4]} for name, value in where.items()}
    directory = tmp_path / 'draft'
    directory.mkdir()
    cfg, _ = write_checkpoint(directory, {**config(), 'num_target_layers': text['num_hidden_layers'],
                                         'vocab_size': text['vocab_size'],
                                         'dflash_config': {'target_layer_ids': [0, 1], 'mask_token_id': 4}})
    metadata = ModuleType('tensorfold.checkpoint_metadata')
    metadata.read_config = lambda directory: raw
    monkeypatch.setitem(sys.modules, 'tensorfold.checkpoint_metadata', metadata)
    monkeypatch.setitem(sys.modules, 'torch', ModuleType('torch'))
    original = common.headers
    monkeypatch.setattr(common, 'headers', lambda path: original(path) if path == directory else headers)
    monkeypatch.setattr(common, 'admit', lambda *args, **kwargs: {
        'geometry': args[4], 'context_window': 128})
    for keep in (1, 3):
        plain = capacity.admit(tmp_path, 128, True, drafter_dir=directory, drafter_bits=8, max_rows=5, keep=keep)
        graph = capacity.admit(tmp_path, 128, True, drafter_dir=directory, drafter_bits=8, max_rows=5,
                               keep=keep, dflash_graph=True)
        expected = capacity.dflash_graph_workspace(cfg, 5, 128)
        assert graph['geometry'].needed(128) - plain['geometry'].needed(128) == expected
        assert graph['dflash_graph']['entries'] == 1
        assert graph['dflash_graph']['payload_workspace_bytes'] == expected


@pytest.mark.parametrize('enabled,bits,rows,kind,graph', [
    (1, 8, 5, 'dflash', False), (True, 16, 5, 'dflash', False),
    (True, 8, 4, 'dflash', False), (True, 8, 5, None, False),
    (True, 8, 5, 'sparkulator', False), (True, 8, 5, 'dflash', True),
])
def test_native_a8_head_region_refused_before_runtime(enabled, bits, rows, kind, graph):
    with pytest.raises(ValueError):capacity.check_dflash_head_a8(enabled, bits, rows, kind, graph)


def test_native_a8_head_default_off_and_eager_payload_contract():
    capacity.check_dflash_head_a8(False, 16, 128, None, True)
    capacity.check_dflash_head_a8(True, 8, 5, 'dflash', False)
    payload = capacity.dflash_head_a8_workspace(2816, 262144, 4)
    assert payload == {'activation_bytes': 4 * (2816 * 2 + 2816 + 4),
                       'logits_softcap_bytes': 3 * 4 * 262144 * 4}
    assert payload['activation_bytes'] == 33808 and payload['logits_softcap_bytes'] == 12582912
    for bad in (0, -1, True):
        with pytest.raises(ValueError):capacity.dflash_head_a8_workspace(2816, 262144, bad)


@pytest.mark.parametrize('options', [
    {'dflash_head_native_a8': 1},
    {'dflash_head_native_a8': True},
    {'dflash_head_native_a8': True, 'drafter': 'not-read', 'drafter_bits': 16, 'depth': 4},
    {'dflash_head_native_a8': True, 'drafter': 'not-read', 'drafter_bits': 8, 'depth': 3},
    {'dflash_head_native_a8': True, 'drafter': 'not-read', 'drafter_bits': 8, 'depth': 4, 'dflash_graph': True},
])
def test_native_a8_engine_unsupported_option_refuses_before_checkpoint_read(options):
    with pytest.raises(ValueError, match='native_a8|native A8'):
        Gemma4Engine('nonexistent-must-not-be-read', **options)

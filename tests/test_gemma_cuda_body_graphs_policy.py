"""Private graph-adapter flow/lifetime controls; CPU fakes do not emulate CUDA math."""
import weakref
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('triton')

from tensorfold.families.gemma4.cuda import body_graphs, dflash


@pytest.fixture
def lane(monkeypatch):
    events, calls, body_outputs, chain_inputs, warm_alive = [], [], [], [], []
    faults = {'replay': None, 'readback': None}

    class Pool:
        def __del__(self):
            events.append('pool-release')

    class Graph:
        def capture_begin(self, **options):
            assert set(options) == {'pool', 'capture_error_mode'}
            assert options['capture_error_mode'] == 'thread_local'
            events.append('begin')
            warm_alive.append(body_outputs[-1]() is not None)

        def capture_end(self):
            events.append('end')

        def replay(self):
            events.append('replay')
            if faults['replay'] is not None:
                raise faults['replay']

        def reset(self):
            events.append('reset')

    class Chains:
        def __init__(self, output, histories, rows, owners, **policy):
            chain_inputs.append(id(output))
            self.owners, self._source, self.host = owners, output, None

        def _readback(self):
            events.append('readback')
            if faults['readback'] is not None:
                raise faults['readback']
            self.owners = self._source = None
            self.host = ()

    class Head:
        bits, block, window, mask_id, device = 8, 5, 8192, 7, torch.device('cpu')
        config = NS(vocab=64, heads=2, head_dim=64)
        logits = dflash.DFlash.logits

        def raw_embed_tokens(self, ids):
            calls.append(tuple(ids.tolist()))
            return torch.zeros((len(ids), 128), dtype=torch.bfloat16)

        def _logits_body(self, state, x, *, full_slots, query=None):
            events.append(('body', len(x), full_slots, int(state.pos[0])))
            output = torch.zeros((len(x)-1, self.config.vocab), dtype=torch.float32)
            body_outputs.append(weakref.ref(output))
            return output

    monkeypatch.setattr(torch.cuda, 'MemPool', Pool)
    monkeypatch.setattr(torch.cuda, 'CUDAGraph', Graph)
    monkeypatch.setattr(torch.cuda, 'empty_cache', lambda: pytest.fail('no global allocator flush'))
    monkeypatch.setattr(body_graphs, 'Chains', Chains)
    state = NS(length=1018, pos=torch.zeros(1, dtype=torch.int32))
    stream = NS(synchronize=lambda: events.append('drain'))
    head = Head()
    owner = body_graphs.DraftBodyGraphs(head, state, stream)
    return NS(owner=owner, state=state, stream=stream, head=head, events=events, calls=calls,
              faults=faults, body_outputs=body_outputs, chain_inputs=chain_inputs, warm_alive=warm_alive)


def test_exact_grid_hit_forward_backward_replacement_and_completed_consumer(lane):
    owner, state = lane.owner, lane.state
    chain = owner.propose(state, 2, 4)
    assert chain.host is not None and chain.owners is chain._source is None
    assert lane.events.index('drain') < lane.events.index('begin') < lane.events.index('end')
    assert lane.events.index('end') < lane.events.index('readback') and 'replay' not in lane.events
    first = owner.entries[3]
    owner.propose(state, 3, 4)
    assert owner.entries[3] is first and owner.captures == 1
    state.length = 1020  # five rows end at1025: the companion's exact1024 grid grows
    owner.propose(state, 4, 4)
    assert owner.entries[3] is not first and owner.entries[3].full_slots == 2
    assert lane.events.index('readback') < lane.events.index('reset')
    state.length = 1018  # restored prefix moves the grid backward too
    owner.propose(state, 5, 4)
    assert owner.entries[3].full_slots == 1 and owner.captures == 3 and owner.retirements == 2
    assert all(ids[1:] == (7, 7, 7, 7) for ids in lane.calls)


def test_miss_uses_live_warm_result_never_uninitialized_capture_output(lane):
    lane.owner.propose(lane.state, 2, 4)
    entry = lane.owner.entries[3]
    assert lane.chain_inputs[0] != id(entry.output)
    assert lane.warm_alive == [True]  # only a weak reference observes this lifetime
    assert lane.body_outputs[0]() is None  # completed readback releases warm GPU ownership
    assert lane.owner.replays == 0 and entry.output_initialized is False
    lane.owner.propose(lane.state, 3, 4)
    assert lane.chain_inputs[1] == id(entry.output)
    assert lane.owner.replays == 1 and entry.output_initialized is True


def test_real_nonfinite_warm_result_is_not_replaced_by_finite_capture_storage(lane, monkeypatch):
    from tensorfold.cuda.speculative import Chains

    calls = [0]

    def body(state, x, **kwargs):
        fill = float('nan') if calls[0] == 0 else 0.0
        calls[0] += 1
        output = torch.full((len(x)-1, lane.head.config.vocab), fill, dtype=torch.float32)
        lane.body_outputs.append(weakref.ref(output))
        return output

    monkeypatch.setattr(lane.head, '_logits_body', body)
    monkeypatch.setattr(body_graphs, 'Chains', Chains)
    with pytest.raises(ValueError, match='confidence scores must be finite'):
        lane.owner.propose(lane.state, 2, 4)
    assert lane.owner.failed and lane.owner.captures == 1 and lane.owner.replays == 0
    assert lane.owner.entries[3].output_initialized is False


def test_one_entry_per_actual_width_and_lane_private_arena(lane):
    for count in (1, 2, 3, 4):
        lane.owner.propose(lane.state, 2, count)
    assert len(lane.owner.entries) == 4 and all(entry is not None for entry in lane.owner.entries)
    other_state = NS(length=1018, pos=torch.zeros(1, dtype=torch.int32))
    other = body_graphs.DraftBodyGraphs(lane.head, other_state, lane.stream)
    assert other.ids.data_ptr() != lane.owner.ids.data_ptr()
    assert other.query.data_ptr() != lane.owner.query.data_ptr()
    before = len(lane.events)
    with pytest.raises(ValueError, match='one model and DraftState'):
        lane.owner.propose(other_state, 2, 4)
    assert len(lane.events) == before


@pytest.mark.parametrize('count', [True, -1, 128, 1.0])
def test_public_count_boundary_before_any_device_work(lane, count):
    before = list(lane.events)
    for operation in (lane.owner.propose, lane.head.logits):
        with pytest.raises(ValueError, match='proposal count'):
            operation(lane.state, None, count)
    assert lane.events == before


def test_noop_keeps_anchor_validation_order_and_does_not_capture(lane):
    lane.state.length = 0
    assert lane.owner.propose(lane.state, None, 4) is None
    assert lane.head.logits(lane.state, None, 4) is None
    lane.state.length = 1
    assert lane.owner.propose(lane.state, None, 0) is None
    assert lane.head.logits(lane.state, None, 0) is None
    assert lane.events == []


@pytest.mark.parametrize('where,on_hit', [('replay', True), ('readback', False), ('readback', True)])
def test_failure_poison_retains_owners_until_drain_no_eager_recovery(lane, where, on_hit):
    if on_hit:
        lane.owner.propose(lane.state, 2, 4)
    primary = RuntimeError(where)
    lane.faults[where] = primary
    with pytest.raises(RuntimeError) as failed:
        lane.owner.propose(lane.state, 2, 4)
    assert failed.value is primary and lane.owner.failed
    entry = lane.owner.entries[3]
    assert entry.pool is not None and entry.graph is not None and entry.output is not None
    before = list(lane.events)
    with pytest.raises(RuntimeError, match='poisoned'):
        lane.owner.propose(lane.state, 2, 4)
    assert lane.events == before
    lane.stream.synchronize()
    lane.owner.close_after_drain()
    assert entry.graph is None and entry.pool is not None and entry.output is not None
    assert lane.owner.closed  # failure frames can still alias output: bounded model-lifetime quarantine
    after = list(lane.events)
    lane.owner.close_after_drain()
    assert lane.events == after


def test_normal_retirement_order_and_close_after_drain(lane):
    order = []

    class Output:
        def __del__(self):
            order.append('output')

    class Pool:
        def __del__(self):
            order.append('pool')

    entry = body_graphs._Entry(1, pool=Pool(), graph=NS(reset=lambda: order.append('graph')), output=Output())
    entry.retire()
    assert order == ['graph', 'output', 'pool']
    lane.owner.propose(lane.state, 2, 4)
    lane.stream.synchronize()
    lane.owner.close_after_drain()
    assert all(entry is None for entry in lane.owner.entries)
    assert lane.owner.ids is lane.owner.query is None and lane.owner.inputs == ()


def test_capture_primary_end_and_gc_restore_failures_are_all_retained(lane, monkeypatch):
    primary, ending, restoring = ValueError('body'), RuntimeError('end'), OSError('gc')
    graph = NS(capture_begin=lambda **kwargs: None,
               capture_end=lambda: (_ for _ in ()).throw(ending))
    entry = body_graphs._Entry(1, pool=object(), graph=graph)
    monkeypatch.setattr(body_graphs.gc, 'isenabled', lambda: True)
    monkeypatch.setattr(body_graphs.gc, 'disable', lambda: None)
    monkeypatch.setattr(body_graphs.gc, 'enable', lambda: (_ for _ in ()).throw(restoring))
    with pytest.raises(ValueError) as failed:
        lane.owner._capture(entry, lambda: (_ for _ in ()).throw(primary))
    assert failed.value is primary
    group = failed.value.__cause__
    assert isinstance(group, BaseExceptionGroup)
    assert ending in group.exceptions and restoring in group.exceptions


def test_engine_private_failure_closes_and_drain_precedes_teardown(monkeypatch):
    from test_gemma_cuda_engine import _engine

    engine = _engine(monkeypatch)
    events = []
    primary = RuntimeError('graph replay')

    def propose(*args, **kwargs):
        raise primary

    owner = NS(failed=True, propose=propose, close_after_drain=lambda: events.append('retire'))
    engine._body_graphs = engine._proposer = owner
    engine._stream.synchronize = lambda: events.append('drain')
    with pytest.raises(RuntimeError) as failed:
        engine.generate([1, 2], 5, None, lambda tokens: None, stop_eos=False)
    assert failed.value is primary and engine._closed
    assert events == []
    engine.close()
    assert events == ['drain', 'retire']


def test_failed_engine_drain_retains_graph_owner(monkeypatch):
    from test_gemma_cuda_engine import _engine

    engine = _engine(monkeypatch)
    owner = NS(close_after_drain=lambda: pytest.fail('unknown GPU completion must retain owners'))
    engine._body_graphs = owner
    primary = RuntimeError('drain')

    def drain():
        raise primary

    engine._stream.synchronize = drain
    with pytest.raises(RuntimeError) as failed:
        engine.close()
    assert failed.value is primary and engine._closed and engine._body_graphs is owner


def test_final_single_row_still_appends_accepted_taps(monkeypatch):
    from test_gemma_cuda_engine import _engine

    engine = _engine(monkeypatch)
    owner = NS(failed=False, propose=lambda *args, **kwargs: pytest.fail('last greedy slot needs no proposal'))
    engine._body_graphs = engine._proposer = owner
    tokens = []
    engine.generate([1, 2], 2, None, lambda values: tokens.extend(values), stop_eos=False)
    assert len(tokens) == 2 and engine.head.append_roles[-1] == (1, False)
    assert len(engine._draft.taps) == engine._state.position

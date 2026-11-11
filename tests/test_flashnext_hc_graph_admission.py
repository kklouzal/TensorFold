"""Owning joint HC graph admission/caller controls without CUDA or model loads."""
import ast
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace as NS
from unittest.mock import patch

import pytest

from tensorfold.cleanup import raise_failures
from tensorfold.cuda.memory_gate import MemoryGate
from tensorfold.families.qwen4_exp.cuda.hc_readout_graphs import HCGraphBudgetExceeded
from tensorfold.families.qwen4_exp.cuda.service_memory import automatic_graph_bytes, context_reservation, future_service
from tests.test_flashnext_graph_budget_host import Array, source_class

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = 'tensorfold.families.qwen4_exp.cuda.hc_plans'


def decoder(*, request=2000, live_bytes=20000, behavior='success', peer=None, charge=300, residual=0):
    live, events, owners = [live_bytes], [], []
    memory = {'free': 1000000, 'active': 1000, 'reserved': 1000}
    runtime = NS(cuda=NS(mem_get_info=lambda device: (memory['free'], 2000000),
                        memory_allocated=lambda device: memory['active'],
                        memory_reserved=lambda device: memory['reserved']))
    def state():
        return NS(capacity=10, limit=10, cache_bytes=lambda rows=None: 4096,
                  layer_bytes=lambda rows: 64, rec=Array(16, 4, Array(8, 4)),
                  conv=Array(3, 2), ple_tail=Array(5, 2))
    fake_module = ModuleType(PACKAGE)
    def votes(torch, gather, packet, world):
        events.append(('vote', tuple(packet)))
        rows = [list(packet) for _ in range(world)]
        if peer is not None:
            peer(rows, packet)
        return rows
    cls = source_class('multi.py', 'MultiDecoder', {'capture_hc_graphs', '_capture_hc_graphs', '_resolve_graph_admission'},
        {'__name__': 'tensorfold.families.qwen4_exp.cuda.multi', 'torch': runtime,
         'multi_graphs': NS(prepare_capture=lambda: events.append('quiescent-floor'), live_bytes=lambda: live[0]),
         'gather_ints': votes, 'POINTS': 4, 'GIB': 1 << 30,
         'context_reservation': context_reservation, 'future_service': future_service,
         'automatic_graph_bytes': automatic_graph_bytes, 'raise_failures': raise_failures})
    dec = cls()
    dec.w = NS(device='owned-device', cfg=NS(hidden=4, streams=2), head=NS(n=10), mtp=None,
               comm=NS(world=8, all_gather=object()), meta={'execution_provider': 'sm70', 'indexer_history': 'gpu'})
    dec.free = [state() for _ in range(24)]
    dec.full_windows, dec.full_resident, dec.capacity = 24, True, 10
    dec.streams, dec.filling, dec.kept, dec.keep = {}, [], [], 1
    dec.round_tables = NS(slots=24)
    dec.hc_graphs_requested, dec._hc_capture_started = True, False
    dec._hc_notification_errors = [None]
    dec.poison_control = lambda error: events.append(('poison', error))
    dec._graph_memory_live = lambda extra: lambda: max(0, min(live[0], 50000-extra))
    dec.hc_graphs = dec._failed_hc_graphs = None
    dec.hc_graph_record = {'captured': False, 'reason': 'not requested'}
    dec.graph_record = {'captured': False, 'reason': 'unique history has no qualified whole-round capture'}
    dec.graph_request, dec.graph_budget = request, request or 0
    dec.graph_header_room, dec.graph_floor, dec.graph_floor_extra_bytes = 2000, 256, 0
    dec.vision_workspace_bytes = 0
    dec.memory_gate = MemoryGate(24*4096+120, reserve=120, live=lambda: live[0])
    dec.memory_gate.held = 24*4096
    dec.buf = NS(hc_plans=None)
    class Plans:
        def __init__(self, budget):
            self.budget, self.closed, self.budget_declined = budget, False, False
            self.settled_bytes, self.retirement_failure = 0, None
        def prepare(self, *, quiescent):
            assert quiescent() and dec._failed_hc_graphs is self
            events.append(('capture', self.budget))
            if behavior == 'library':
                raise MemoryError('unrelated library allocation')
            if behavior in ('budget', 'budget-record'):
                live[0] -= residual
                memory['free'] -= residual
                self.closed, self.budget_declined = True, True
                self.error = HCGraphBudgetExceeded('observed grant')
                raise self.error
            self.settled_bytes = charge
            live[0] -= charge
            memory['free'] -= charge
            memory['reserved'] += charge-residual
            if behavior == 'unattributable':
                memory['active'] += charge+1
            if behavior == 'free-change':
                memory['free'] += charge+1
            return self.record()
        def record(self):
            if behavior == 'budget-record':
                raise MemoryError('decline receipt')
            return {'captured_keys': 97, 'settled_charge_bytes': self.settled_bytes}
        def close(self):
            events.append('retire')
            if behavior == 'close-error':
                raise OSError('native retirement')
            if not self.closed:
                live[0] += self.settled_bytes-residual
                memory['free'] += self.settled_bytes-residual
                memory['reserved'] -= self.settled_bytes-residual
            self.closed = True
        def validate_adoption(self):
            events.append('validate-adoption')
    def construct(w, b, *, budget_bytes):
        events.append('journal')
        owner = Plans(budget_bytes)
        owners.append(owner)
        return owner
    fake_module.readout_graphs = construct
    dec.run = lambda: None
    def run():
        with patch.dict(sys.modules, {PACKAGE: fake_module}):
            dec.capture_hc_graphs()
    return dec, run, live, events, owners


def test_full24_service_promise_and_one_physical_charge_precede_publication():
    dec, run, live, events, owners = decoder()
    held, room = dec.memory_gate.held, dec.memory_gate.room
    run()
    assert events.index('journal') > next(i for i, event in enumerate(events) if isinstance(event, tuple) and event[0] == 'vote')
    assert dec.buf.hc_plans is dec.hc_graphs is owners[0] and dec._failed_hc_graphs is None
    assert dec.hc_graph_record['captured'] and not dec.graph_record['captured']
    future = dec.hc_graph_record['future_service']
    assert future['growth_bytes'] == 0 and future['copy_overlap_bytes'] == 64
    assert future['snapshot_bytes'] == 97*48 and future['workspace_bytes'] == 24*56
    assert dec.hc_graph_record['materialized_full_windows']['slots'] == 24
    assert len(dec.free) == 24 and dec.memory_gate.held == held and dec.memory_gate.room == room
    assert live == [19700] and dec.memory_gate.live() == 19700 and dec.graph_budget == 2000
    before = list(events)
    with pytest.raises(RuntimeError, match='immutable'):
        run()
    assert events[:-1] == before and events[-1][0] == 'poison'


def test_one_peer_lacking_future_room_declines_before_any_capture():
    def peer(rows, packet):
        if len(packet) == 3:
            rows[7][0] = 0
    dec, run, live, events, owners = decoder(peer=peer)
    run()
    assert not owners and not dec.hc_graph_record['captured'] and dec.graph_budget == 2000
    assert live == [20000] and dec.buf.hc_plans is None


def test_automatic_grant_takes_allrank_minimum_without_new_cap():
    def peer(rows, packet):
        if len(packet) == 3:
            rows[7][1] = 128
    dec, run, _, events, _ = decoder(request=None, peer=peer, charge=100)
    run()
    assert ('capture', 128) in events and dec.graph_budget == 128
    assert dec.hc_graph_record['automatic']


@pytest.mark.parametrize('behavior', ['budget', 'success'])
def test_local_or_peer_budget_decline_requires_every_rank_retired_before_release(behavior):
    def peer(rows, packet):
        if len(packet) == 5:
            rows[7][0:2] = [0, 1]
    dec, run, live, events, owners = decoder(behavior=behavior, peer=peer)
    run()
    assert owners[0].closed and dec._failed_hc_graphs is None
    assert not dec.hc_graph_record['captured'] and dec.graph_budget == 2000
    assert dec.buf.hc_plans is None and live == [20000]
    retired_index = events.index(('vote', (1,)))
    assert events.index('retire') < retired_index < len(events)-1
    assert dec.hc_graph_record['retirement'] == [[1]]*8


def test_arbitrary_memory_error_is_fatal_retains_owner_and_reservation():
    dec, run, _, events, owners = decoder(behavior='library')
    with pytest.raises(MemoryError, match='library'):
        run()
    assert dec._failed_hc_graphs is owners[0] and dec.graph_budget == 2000
    assert not owners[0].closed and dec.buf.hc_plans is None
    assert sum(isinstance(e, tuple) and e[0] == 'vote' for e in events) == 1


def test_failed_retirement_after_peer_decline_never_releases_reserved_bytes():
    def peer(rows, packet):
        if len(packet) == 5:
            rows[7][0:2] = [0, 1]
    dec, run, _, _, owners = decoder(behavior='close-error', peer=peer)
    with pytest.raises(OSError, match='retirement'):
        run()
    assert dec.graph_budget == 2000 and dec._failed_hc_graphs is owners[0] and not owners[0].closed


def test_decline_metadata_exhaustion_keeps_original_budget_primary():
    dec, run, _, _, owners = decoder(behavior='budget-record')
    with pytest.raises(HCGraphBudgetExceeded) as raised:
        run()
    assert raised.value is owners[0].error and isinstance(raised.value.__cause__, MemoryError)
    assert dec._failed_hc_graphs is owners[0] and dec.graph_budget == 2000


@pytest.mark.parametrize('field,value', [('full_resident', False), ('full_windows', 23), ('streams', {1: object()})])
def test_unmaterialized_or_busy_caller_refuses_before_native_effects(field, value):
    dec, run, _, events, owners = decoder()
    setattr(dec, field, value)
    with pytest.raises(RuntimeError, match='materialized'):
        run()
    assert len(events) == 1 and events[0][0] == 'poison' and not owners


def test_readout_false_miss_continues_exact_original_but_none_is_legacy_handled():
    path = ROOT/'src/tensorfold/families/qwen4_exp/cuda/forward.py'
    fn = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.FunctionDef) and n.name == '_readout_plain')
    calls = []
    namespace = {'glue': NS(hc_normed=lambda *a: calls.append('norm'), hc_mix=lambda *a: calls.append('mix')),
                 '_down_act': lambda *a: calls.append('down'), '_mm': lambda *a: calls.append('up')}
    unit = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(unit), str(path), 'exec'), namespace)
    for handled, expected in [(False, ['plan', 'norm', 'down', 'up', 'mix']), (None, ['plan']), (True, ['plan'])]:
        calls.clear()
        def plan(*a, **kw):
            calls.append('plan')
            return handled
        plan.enabled = True
        b = NS(hc_plans=plan, prefill=False, **{key: [object()] for key in
               ('pss', 'normed', 'xs_normed', 'act', 'xs_act', 'up', 'mixed', 'xs_mixed')})
        namespace['_readout_plain'](NS(scale=object(), up=object()), b, [1], 1, .01, 4, 320, None)
        assert calls == expected


def engine_selection(*, requested=True, owned=True, depth=0, admin=False, dense='reference', concat='reference', **changes):
    path = ROOT/'src/tensorfold/families/qwen4_exp/cuda/engine.py'
    cls = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == 'FlashNextEngine')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_initialize')
    selected = {name: next(n for n in init.body if isinstance(n, ast.Assign)
                           and any(isinstance(t, ast.Name) and t.id == name for t in n.targets))
                for name in ('requested_graphs', 'hc_graphs', 'graph_request')}
    values = dict(self=NS(_owned_expert_requested=owned, _owned_expert_admin_requested=admin,
                         sm70_profile='reference'), graphs=requested, concurrent=True, managed=False,
                  source_is_gguf=True, sm70=True, tp=8, depth=depth, history_placement=object(),
                  indexer_history='gpu', vram_experts=None, vision=False, dspark_model_dir=None,
                  dflash2_model_dir=None, owner_specification={'graph_family': 'uncaptured'}, owner_preparation=object(), text={'hidden_size': 2560, 'num_hidden_layers': 48},
                  multi_graph_request=lambda: 2048 << 20)
    values.update(changes)
    exec(compile(ast.Module(body=[selected['requested_graphs']], type_ignores=[]), str(path), 'exec'), values)
    values['graphs'] = False  # Actual unique-history whole-round guard.
    selection_index = init.body.index(selected['hc_graphs'])
    guard = init.body[selection_index+1]
    fake_weights = ModuleType('tensorfold.families.qwen4_exp.cuda.gguf_weights')
    fake_weights.GGUFDense, fake_weights.GGUFConcat = NS(implementation=dense), NS(implementation=concat)
    values['__package__'] = 'tensorfold.families.qwen4_exp.cuda'
    with patch.dict(sys.modules, {'tensorfold.families.qwen4_exp.cuda.gguf_weights': fake_weights}):
        exec(compile(ast.Module(body=[selected['hc_graphs'], guard, selected['graph_request']], type_ignores=[]), str(path), 'exec'), values)
    return values


def test_original_request_survives_whole_graph_decline_but_no_graphs_stays_off():
    selected = engine_selection()
    assert selected['graphs'] is False and selected['hc_graphs'] is True
    assert selected['graph_request'] == 2048 << 20
    selected = engine_selection(requested=False)
    assert not selected['hc_graphs'] and selected['graph_request'] == 0


@pytest.mark.parametrize('changes', [dict(owned=False), dict(depth=1), dict(tp=4), dict(vision=True),
                                    dict(history_placement=None), dict(indexer_history='ram'),
                                    dict(owner_specification={'graph_family': 'captured-owner'}),
                                    dict(admin=True), dict(managed=True), dict(owner_preparation=None),
                                    dict(dense='native32'), dict(concat='fused')])
def test_generic_and_unqualified_callers_preserve_original_route(changes):
    selected = engine_selection(**changes)
    assert not selected['hc_graphs'] and selected['graph_request'] == 0


@pytest.mark.parametrize('fail', [False, True])
def test_engine_closes_local_graph_once_before_workspace_and_communicator(fail):
    from tests.test_flashnext_engine_lifetime import engine, Resource
    e = engine()
    def retire():
        e.events.append('local-graph-close')
        if fail:
            raise OSError('uncertain graph retirement')
    local = NS(close=retire)
    e.multi.hc_graphs = e.multi.buf.hc_plans = local
    e._owner_preparation = Resource(e.events, 'workspace-close')
    e.comm = NS(close=lambda *, abort: e.events.append('comm-close'))
    fake_torch = ModuleType('torch')
    fake_torch.cuda = NS(synchronize=lambda device: e.events.append('device-fence'))
    with patch.dict(sys.modules, {'torch': fake_torch}):
        if fail:
            with pytest.raises(OSError, match='uncertain'):
                e._close_resources()
            assert e.events == ['local-graph-close'] and e.multi.hc_graphs is local and e._owner_preparation
        else:
            e._close_resources()
            assert e.events.count('local-graph-close') == 1
            assert e.events.index('local-graph-close') < e.events.index('workspace-close') < e.events.index('comm-close')


def test_disabled_request_never_samples_or_captures():
    dec, run, _, events, owners = decoder()
    dec.hc_graphs_requested = False
    run()
    assert not events and not owners


@pytest.mark.parametrize('invalid', ['status', 'grant', 'retirement', 'future'])
def test_foreign_or_incomplete_peer_proof_cannot_publish_local_plan(invalid):
    def peer(rows, packet):
        if invalid == 'status' and len(packet) == 3:
            rows[7][0] = 2
        if invalid == 'grant' and len(packet) == 3:
            rows[7][1] -= 1
        if invalid == 'retirement' and len(packet) == 5:
            rows[7][0:2] = [0, 0]
        if invalid == 'future' and len(packet) == 5:
            rows[7][3] = rows[7][4]-1
    dec, run, _, _, _ = decoder(peer=peer)
    with pytest.raises(RuntimeError):
        run()
    assert dec.buf.hc_plans is None and not dec.hc_graph_record['captured'] and dec.graph_budget == 2000


def test_constructor_uses_one_graph_request_authority_and_rejects_two_live_catalogues():
    path = ROOT/'src/tensorfold/families/qwen4_exp/cuda/multi.py'
    cls = next(n for n in ast.parse(path.read_bytes()).body if isinstance(n, ast.ClassDef) and n.name == 'MultiDecoder')
    init = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '__init__')
    end = next(i for i, n in enumerate(init.body) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Name) and t.id == 'budget' for t in n.targets))
    probe = ast.parse('def call(w, graphs, hc_graphs, graph_budget=..., native_gdn_batch=False):\n    pass').body[0]
    probe.body = init.body[:end+1]+[ast.Return(value=ast.Name(id='budget', ctx=ast.Load()))]
    namespace = {'multi_graphs': NS(budget_bytes=lambda: 42)}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[probe], type_ignores=[])), str(path), 'exec'), namespace)
    w = NS(comm=object(), meta={'execution_provider': 'sm70'})
    assert namespace['call'](w, False, True) == 42
    assert namespace['call'](w, True, False) == 42
    assert namespace['call'](w, False, False) == 0
    assert namespace['call'](w, False, True, None) is None
    with pytest.raises(ValueError, match='declined'):
        namespace['call'](w, True, True)
    with pytest.raises(ValueError, match='boolean'):
        namespace['call'](w, False, 'yes')


def test_fresh_room_loss_after_vote_refuses_before_any_graph_allocation():
    live = None
    def peer(rows, packet):
        if len(packet) == 3:
            live[0] = 1
    dec, run, live, events, owners = decoder(peer=peer)
    with pytest.raises(MemoryError, match='changed after'):
        run()
    assert not owners and 'journal' not in events and dec.graph_budget == 2000


@pytest.mark.parametrize('behavior', ['budget', 'success'])
def test_clean_decline_retains_measured_driver_residual_and_admitted_ceiling(behavior):
    def peer(rows, packet):
        if len(packet) == 5:
            rows[7][0:2] = [0, 1]
    dec, run, live, events, owners = decoder(behavior=behavior, peer=peer, residual=64)
    run()
    assert owners[0].closed and dec._failed_hc_graphs is None and dec.hc_graphs is None
    assert dec.graph_budget == 2000 and dec.hc_graph_record['reservation_retained_after_attempt'] == 2000
    assert dec.hc_graph_record['additional_outside_active_bytes'] == 64
    assert live == [19936] and dec.memory_gate.live() == 19936
    assert not any(isinstance(e, tuple) and e[0] == 'poison' for e in events)


@pytest.mark.parametrize('behavior', ['unattributable', 'free-change'])
def test_unknown_or_negative_residency_never_turns_into_zero_credit(behavior):
    dec, run, _, events, owners = decoder(behavior=behavior)
    with pytest.raises(RuntimeError, match='unattributable'):
        run()
    assert dec._failed_hc_graphs is owners[0] and dec.graph_budget == 2000
    assert events[-1][0] == 'poison' and dec.buf.hc_plans is None


@pytest.mark.parametrize('notification_fails', [False, True])
def test_actual_poison_callback_precedes_engine_cleanup_and_preserves_notification_failure(notification_fails):
    from tests.test_flashnext_engine_lifetime import engine
    dec, run, _, events, owners = decoder(behavior='library')
    e = engine()
    e.multi, e._owner_preparation = dec, NS(close=lambda: events.append('workspace-close'))
    e.w.meta = {}
    e.comm = NS(close=lambda *, abort: events.append(('comm-close', abort)))
    e._abort_comm = False
    dec.recover_failed_slots = lambda: None
    cls = source_class('tp_scheduler.py', 'TensorParallelDecoder', {'_poison_control'}, {'raise_failures': raise_failures})
    control = cls()
    control.failure, control.engine = None, e
    dec.poison_control = control._poison_control
    notice = MemoryError('notification exhausted')
    module = ModuleType('tensorfold.cuda.cohort')
    def notify():
        events.append('cohort-notify')
        if notification_fails:
            raise notice
    module.notify_failure = notify
    with patch.dict(sys.modules, {'tensorfold.cuda.cohort': module}):
        with pytest.raises(MemoryError, match='library') as raised:
            run()
    assert control.failure is raised.value and dec._memory_error is raised.value and e._abort_comm
    if notification_fails:
        assert raised.value.__cause__ is notice
    assert not owners[0].closed and 'retire' not in events
    runtime = ModuleType('torch')
    runtime.cuda = NS(synchronize=lambda device: events.append('device-fence'))
    with patch.dict(sys.modules, {'torch': runtime}):
        e._close_resources()
    assert events.index('cohort-notify') < events.index('retire') < events.index('workspace-close')
    assert ('comm-close', True) in events and owners[0].closed

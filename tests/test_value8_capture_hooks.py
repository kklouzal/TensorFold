"""Private captured-only value8 caller/ownership checks; CPU metadata doubles."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70.gdn_value8 import Prepared, Workspace
from tensorfold.families.qwen4_exp.cuda import exact_graphs, forward, gdn_multi, multi, owned_experts


def weights():
    plan = object()
    bank = NS(_owned_expert_plan=plan, implementation=owned_experts.POLICY)
    return NS(device='cuda:0', comm=None, mtp=None,
        cfg=NS(nk=2, nv=6, execution_dtype='float16', eos=[]),
        layers=[NS(linear=True, ple=None, moe=NS(experts=bank))],
        meta={'execution_provider': 'sm70', 'history_layout': 'unique', 'owned_expert_plan': plan,
              'input_preparation_context': object(), 'execution_dtype': 'float16'})


@pytest.mark.parametrize('fault', ['resident', 'native', 'dtype', 'heads', 'slots', 'depth', 'rows', 'device'])
def test_optional_value8_rejects_unsupported_envelope_before_buffer_allocation(monkeypatch, fault):
    w = weights()
    prepared = Prepared(object(), torch.device('cuda:0'), True)
    args = dict(slots=24, capacity=4096, depth=4, graphs=True, graph_budget=1024,
                native_gdn_batch=True, native_gdn_capture=True, gdn_value8_prepared=prepared)
    if fault == 'resident':
        args['gdn_value8_prepared'] = Prepared(object(), prepared.device)
    elif fault == 'native':
        args['native_gdn_capture'], args['depth'] = False, 0
    elif fault == 'dtype':
        w.meta['execution_dtype'] = 'bfloat16'
    elif fault == 'heads':
        w.cfg.nv = 12
    elif fault == 'slots':
        args['slots'] = 33
    elif fault == 'depth':
        args['depth'] = 5
    elif fault == 'rows':
        args['execution_rows'] = 161
        args['slots'] = 32
    else:
        args['gdn_value8_prepared'] = Prepared(object(), torch.device('cuda:1'), True)
    monkeypatch.setattr(multi, 'Buffers', lambda *a, **kw: pytest.fail('unsupported value8 reached Buffer allocation'))
    with pytest.raises(ValueError):
        multi.MultiDecoder(w, **args)


def test_global_Q_reuse_wraps_already_paid_storage_and_old_readers_are_outside_half_path(monkeypatch):
    c = NS(nk=2, nv=6, execution_dtype='float16')
    scratch = gdn_multi.Scratch(NS(cfg=c, device='cpu', layers=[NS(linear=True)]), 10)
    tree = ast.parse(Path(multi.__file__).read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'MultiDecoder')
    init = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    node = next(node for node in init.body if isinstance(node, ast.If) and any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == 'Workspace'
        for call in ast.walk(node)))
    owner = NS(gdn=scratch)
    monkeypatch.setattr(torch, 'empty', lambda *a, **kw: pytest.fail('Q reuse allocated another tensor'))
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<actual paid Q reuse>', 'exec'),
         {'self': owner, 'gdn_value8_prepared': object(), 'rows': 10})
    assert owner.gdn_value8_workspace.q is scratch.q and owner.gdn_value8_workspace.capacity == 10
    gdn_tree = ast.parse(Path(gdn_multi.__file__).read_text())
    readers = [function.name for function in gdn_tree.body if isinstance(function, ast.FunctionDef)
               and any(isinstance(value, ast.Attribute) and value.attr == 'q' and isinstance(value.ctx, ast.Load)
                       for value in ast.walk(function))]
    assert readers == ['block']
    flush = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '_flush')
    assert not any(isinstance(value, ast.Attribute) and value.attr == 'q' for value in ast.walk(flush))
    chain = next(node for node in ast.parse(Path(forward.__file__).read_text()).body
                 if isinstance(node, ast.FunctionDef) and node.name == 'gdn_chain')
    branch = next(node for node in chain.body if isinstance(node, ast.If) and any(
        isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute) and call.func.attr == 'block'
        for call in ast.walk(node)))
    assert any(isinstance(compare, ast.Compare) and isinstance(compare.ops[0], ast.NotEq)
        and isinstance(compare.comparators[0], ast.Attribute) and compare.comparators[0].attr == 'float16'
        for compare in ast.walk(branch.test))


def graph_fixture(monkeypatch, *, slots=2):
    from tests.test_sm70_gdn_value8_batch import setup

    static, b, states, _, prepared, calls = setup(monkeypatch, layers=2, slots=slots)
    b.rotor_lookup = False
    for state in states:
        state.image_positions = None
    c = NS(nk=2, nv=6, execution_dtype='float16')
    w = NS(cfg=c, comm=None, device='cpu', layers=[NS(linear=True), NS(linear=True)])
    scratch = gdn_multi.Scratch(w, b.rows)
    dec = NS(w=w, free=states, buf=b, mbuf=None, round_tables=static, depth=4,
        streams={}, filling=[], kept=[], native_gdn_capture=True, gdn=scratch,
        gdn_value8_prepared=prepared, gdn_value8_workspace=Workspace(scratch.q, b.rows))
    outer = NS(dec=dec, verify={}, mtp={}, failure=None)
    return exact_graphs.ExactStateGraphs(outer), b, states, calls


def test_exact_cold_binds_every_layer_to_existing_Q_without_old_operand_or_extra_allocation(monkeypatch):
    graph, b, states, _ = graph_fixture(monkeypatch)
    graph.preparing = True
    monkeypatch.setattr(torch, 'empty', lambda *a, **kw: pytest.fail('capture binding allocated tensor'))
    graph._native_frame([(states[0], 0, 5)], b, capture=True)
    assert b.gdn_capture_ready_operands is None and len(b.gdn_value8_ready) == 2
    assert all(plan.workspace.q is graph.dec.gdn.q for plan in b.gdn_value8_ready)
    record = graph.record()
    assert record['gdn_value8_capture'] is True
    assert record['gdn_value8_workspace']['reused_bytes'] == b.rows*1024
    assert record['gdn_value8_workspace']['additional_device_bytes'] == 0


@pytest.mark.parametrize('split,exact,prefill,chosen', [(True, True, False, 'value8'),
    (False, True, False, 'old'), (True, False, False, 'old'), (True, True, True, 'prefill')])
def test_only_exact_target_capture_selects_value8_other_gdn_paths_stay_old(monkeypatch, split, exact, prefill, chosen):
    graph, b, states, calls = graph_fixture(monkeypatch)
    graph.preparing = True
    segs = [(states[0], 0, 5)]
    graph._native_frame(segs, b, capture=True)
    b.gdn_value8_capture, b.gdn_capture_operands = b.gdn_value8_ready, None
    b.split, b.exact_graph_handoffs, b.prefill = object() if split else None, exact, prefill
    b.mixed, b.xs_mixed = b.h, b.gxs
    states[0].lin_index = {7: 0}
    events = []
    monkeypatch.setattr(forward, '_mm', lambda *a: None)
    monkeypatch.setattr(forward, '_out_proj', lambda *a: 'out')
    monkeypatch.setattr(forward, 'gdn_chain', lambda *a: events.append('old'))
    monkeypatch.setattr(forward, '_prefill_chain', lambda *a: events.append('prefill'))
    g = NS(proj=object(), conv=torch.empty(1280, 4, dtype=torch.float16), a_log=None,
           a_neg=torch.zeros(6), dt_bias=torch.zeros(6), norm=torch.ones(128, dtype=torch.float16), out=object())
    graph.dec.w.cfg.eps = 1e-6
    assert forward.gdn_block(NS(index=7, gdn=g), graph.dec.w, segs, b, 5) == 'out'
    assert len(calls) == int(chosen == 'value8')
    assert events == ([] if chosen == 'value8' else [chosen])


def test_value8_replay_uses_one_existing_frame_gate_not_one_per_layer(monkeypatch):
    from tests.test_flashnext_exact_graphs import owner
    from tensorfold.families.qwen4_exp.cuda import compact_indexer

    graph, (state, _) = owner()
    graph.native_gdn_capture, graph.value8_prepared = True, object()
    events = []
    source = graph.dec.buf
    source.gdn_tables = NS(validate_current=lambda *a: events.append('current-frame'))
    segs = [(state, 0, 5)]
    key = graph._key('verify', segs)
    def validate(frame, *, segs):
        events.append('value8-storage')
        frame.validate_current(segs, source)
    graph.value8_bindings[key] = (NS(validate_frame=validate), NS(validate_frame=lambda *a: pytest.fail('duplicate layer validation')))
    graph.owner.verify[key] = NS(graphs=['graph'], inserts=[], out=object())
    monkeypatch.setattr(compact_indexer, 'validate_target_indexer_frame', lambda *a: None)
    monkeypatch.setattr(exact_graphs, '_kv_begin', lambda *a: events.append('entry'))
    monkeypatch.setattr(exact_graphs, 'replay', lambda *a: events.append('replay'))
    graph.forward('verify', segs)
    assert events == ['value8-storage', 'current-frame', 'entry', 'replay']


def test_value8_actual_segments_must_match_staged_native_frame_before_entry_or_replay(monkeypatch):
    from tensorfold.families.qwen4_exp.cuda import compact_indexer

    graph, b, states, _ = graph_fixture(monkeypatch)
    graph.preparing = True
    graph._native_frame([(states[0], 0, 1), (states[1], 1, 6)], b, capture=True)
    graph.preparing = False
    key = (6, 2)
    graph.value8_bindings[key] = b.gdn_value8_ready
    graph.owner.verify[key] = NS(graphs=['captured'], inserts=[], out=object())
    monkeypatch.setattr(compact_indexer, 'validate_target_indexer_frame', lambda *a: None)
    monkeypatch.setattr(exact_graphs, '_kv_begin', lambda *a: pytest.fail('mismatched GDN frame entered State'))
    monkeypatch.setattr(exact_graphs, 'replay', lambda *a: pytest.fail('mismatched GDN frame replayed'))
    with pytest.raises(ValueError, match='current prepared State/parity'):
        graph.forward('verify', [(states[0], 0, 3), (states[1], 3, 6)])


def test_capture_failure_clears_value8_and_exact_flags_while_retaining_plan_owners(monkeypatch):
    from tests.test_flashnext_exact_graphs import owner
    from tensorfold.families.qwen4_exp.cuda import multi_graphs

    graph, (state, _) = owner()
    graph.preparing, graph.value8_prepared = True, object()
    graph.selected_keys = frozenset({('verify', 1, 1)})
    graph.dec.w.device = 'cpu'
    graph.owner.room_bytes, graph.owner.budget = 0, 1024
    graph.owner._has_room = lambda: True
    binding = (object(),)
    def restage(kind, segs, source):
        source.gdn_value8_ready = binding
        return segs
    failure = RuntimeError('controlled piece failure')
    def pieces(run, source, *, retain_pending):
        assert retain_pending is True
        assert source.gdn_value8_capture is binding and source.exact_graph_handoffs is True
        raise failure
    graph.owner._pieces = pieces
    monkeypatch.setattr(graph, '_agree', lambda value: value)
    monkeypatch.setattr(graph, '_restage_warm', restage)
    monkeypatch.setattr(exact_graphs, 'compute', lambda *a, **kw: None)
    monkeypatch.setattr(multi_graphs, 'free_bytes', lambda: 1024)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda *a: 0)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda *a: 0)
    with pytest.raises(RuntimeError) as caught:
        graph._capture('verify', (1, 1), [(state, 0, 1)], graph.dec.buf)
    assert caught.value is failure
    assert graph.dec.buf.gdn_value8_capture is None and graph.dec.buf.gdn_value8_ready is None
    assert graph.dec.buf.exact_graph_handoffs is False
    assert graph.value8_bindings[(1, 1)] is binding


@pytest.mark.parametrize('failure_stage', [None, 'pending', 'published'])
def test_pending_and_published_graphs_retire_before_all_value8_bindings_release(monkeypatch, failure_stage):
    from tests.test_flashnext_exact_graphs import owner
    from tensorfold.families.qwen4_exp.cuda.multi_graphs import RoundGraphs

    graph, _ = owner()
    outer = graph.owner
    outer.dec.w.device = 'cpu'
    outer.bound = graph
    events, failure = [], RuntimeError('controlled graph retirement failure')
    graph.value8_bindings = {(1, 1): (object(),), (5, 1): (object(),)}
    def reset(stage):
        assert len(graph.value8_bindings) == 2
        events.append(stage)
        if failure_stage == stage:
            raise failure
    outer.pending = [NS(graphs=[NS(reset=lambda: reset('pending'))])]
    published = NS(graphs=[NS(reset=lambda: reset('published'))])
    graph.bindings[('verify', (1, 1))] = published
    outer.verify[(1, 1)] = published
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda *a: events.append('fence'))
    if failure_stage is None:
        RoundGraphs.close(outer)
        assert graph.closed and not graph.value8_bindings and not outer.pending
        assert events.index('pending') < events.index('published')
    else:
        with pytest.raises(RuntimeError) as caught:
            RoundGraphs.close(outer)
        assert caught.value is failure and len(graph.value8_bindings) == 2 and not graph.closed
        if failure_stage == 'pending':
            assert 'published' not in events and outer.pending


@pytest.mark.parametrize('fault', ['post-return-fence', 'binding-publication'])
def test_returned_pieces_remain_in_explicit_close_journal_across_transfer_failures(monkeypatch, fault):
    from tests.test_flashnext_exact_graphs import owner
    from tensorfold.families.qwen4_exp.cuda import multi_graphs

    graph, (state, _) = owner()
    outer, source = graph.owner, graph.dec.buf
    outer.bound = graph
    graph.preparing, graph.value8_prepared = True, object()
    graph.selected_keys = frozenset({('verify', 1, 1)})
    outer.dec.w.device = 'cpu'
    outer.room_bytes, outer.budget = 0, 1024
    outer._has_room = lambda: True
    binding, events = (object(),), []
    failure = MemoryError('controlled journal publication') if fault == 'binding-publication' else RuntimeError('controlled post-return fence')
    if fault == 'binding-publication':
        class Refusing(dict):
            def __setitem__(self, key, value):
                raise failure
        graph.bindings = Refusing()
    def reset():
        assert graph.value8_bindings[(1, 1)] is binding
        events.append('reset-borrower')
    pieces = NS(graphs=[NS(reset=reset)], inserts=[], out=object())
    def capture(*args, retain_pending):
        assert retain_pending is True
        outer.pending.append(pieces)
        return pieces
    def restage(kind, segs, buffers):
        buffers.gdn_value8_ready = binding
        return segs
    fences = []
    def synchronize(*args):
        fences.append(True)
        if fault == 'post-return-fence' and len(fences) == 1:
            raise failure
    outer._pieces = capture
    monkeypatch.setattr(graph, '_agree', lambda value: value)
    monkeypatch.setattr(graph, '_restage_warm', restage)
    monkeypatch.setattr(exact_graphs, 'compute', lambda *a, **kw: None)
    monkeypatch.setattr(multi_graphs, 'free_bytes', lambda: 1024)
    monkeypatch.setattr(torch.cuda, 'memory_allocated', lambda *a: 0)
    monkeypatch.setattr(torch.cuda, 'memory_reserved', lambda *a: 0)
    monkeypatch.setattr(torch.cuda, 'synchronize', synchronize)
    with pytest.raises(type(failure)) as caught:
        graph._capture('verify', (1, 1), [(state, 0, 1)], source)
    assert caught.value is failure
    assert (pieces in outer.pending) if fault == 'binding-publication' else (graph.bindings[('verify', (1, 1))] is pieces)
    assert graph.value8_bindings[(1, 1)] is binding and not source.exact_graph_handoffs
    assert source.gdn_value8_capture is None
    multi_graphs.RoundGraphs.close(outer)
    assert events == ['reset-borrower'] and not outer.pending and not graph.value8_bindings and graph.closed


def test_32_state_exact_catalogue_and_cold_views_reuse_paid_E160_Q(monkeypatch):
    graph,b,states,_=graph_fixture(monkeypatch,slots=32)
    graph._catalogue(states)
    assert ('verify',32,32) in graph.available_keys and ('verify',160,32) in graph.available_keys
    graph.preparing=True
    segs=[(state,index*5,(index+1)*5) for index,state in enumerate(states)]
    monkeypatch.setattr(torch,'empty',lambda *args,**kwargs:pytest.fail('cold view binding allocated extra Q'))
    graph._native_frame(segs,b,capture=True)
    assert len(b.gdn_value8_ready)==2
    assert all(bound.participants==32 and bound.rows==160 and bound.workspace.q is graph.dec.gdn.q
               for bound in b.gdn_value8_ready)
    assert graph.record()['gdn_value8_workspace']['reused_bytes']==163840
    assert graph.record()['gdn_value8_workspace']['additional_device_bytes']==0

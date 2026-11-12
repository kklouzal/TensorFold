"""Prepared HC ownership/dispatch spies; native numerical proof is separate."""
import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.cuda.sm70 import family_ops as ops
from tensorfold.families.qwen4_exp.cuda import forward


def buffers(rows=3):
    return NS(rows=rows, h=torch.zeros((rows, 10240), dtype=torch.float16),
        streams=torch.zeros((rows, 10240), dtype=torch.float16), pss=torch.empty((rows, 10, 4)),
        inj_a=torch.ones((rows, 4), dtype=torch.float16), inj_m=torch.ones((rows, 4), dtype=torch.float16),
        g_branch=torch.empty(8*rows*2560), g_moe=torch.empty(8*rows*2560),
        owner_graph_family='uncaptured', split=None, prefill=False)


def prepared(b, calls):
    def raw(h, hout, pss, streams, mode, *args):
        calls.append((h, hout, pss, streams, mode))
        # Raw shared-storage writes deliberately bypass Torch mutation metadata,
        # like the native ABI. This is an ownership/version spy, not arithmetic.
        pss.numpy().fill(31)
    return ops.PreparedHCWriteback(NS(writeback=raw), 'cpu').bind(b)


@pytest.mark.parametrize('inference', [False, True])
@pytest.mark.parametrize('mode', [0, 3])
def test_owned_writeback_publishes_each_output_version_once(inference, mode):
    with torch.inference_mode(inference):
        b, calls = buffers(), []
        cap = prepared(b, calls)
        h, pss = b.h[:2], b.pss[:2]
        before = None if inference else (h._version, pss._version)
        branch = b.g_branch[:8*2*2560].view(8, 2, 2560) if mode else None
        inject = b.inj_a[:2] if mode else None
        assert cap.run(h, h, pss, 4, mode, branch=branch, inject=inject)
        assert len(calls) == 1 and torch.all(pss == 31)
        if inference:
            assert h.is_inference() and pss.is_inference()
        else:
            assert (h._version, pss._version) == (before[0] + 1, before[1] + 1)
            assert b.h._version == h._version and b.pss._version == pss._version


@pytest.mark.parametrize('fault', ['mode', 'strided', 'subclass', 'autograd', 'source-grad', 'foreign', 'partial', 'pss', 'unused', 'empty'])
def test_unadmitted_domains_do_not_reach_raw_native(fault):
    b, calls = buffers(), []
    cap = prepared(b, calls)
    h, hout, pss, mode, branch, inject = b.h, b.h, b.pss, 3, b.g_branch.view(8, 3, 2560), b.inj_a
    kwargs = {}
    if fault == 'mode':
        mode = 4
    elif fault == 'strided':
        h = h[:, ::2]
    elif fault == 'subclass':
        class Other(torch.Tensor):
            pass
        h = h.as_subclass(Other)
    elif fault == 'autograd':
        h.requires_grad_(True)
    elif fault == 'source-grad':
        branch.requires_grad_(True)
    elif fault == 'foreign':
        h = h.clone()
    elif fault == 'partial':
        h, hout, pss = b.h[:2], b.h[1:], b.pss[:2]
        branch, inject = b.g_branch[:8*2*2560].view(8, 2, 2560), b.inj_a[:2]
    elif fault == 'pss':
        pss = pss.clone()
    elif fault == 'empty':
        h, hout, pss = b.h[:0], b.h[:0], b.pss[:0]
    else:
        kwargs['y'] = object()
    assert cap.run(h, hout, pss, 4, mode, branch=branch, inject=inject, **kwargs) is False
    assert calls == []


def test_constructor_rejects_overlapping_owner_allocations():
    b = buffers()
    b.pss = b.h.view(torch.float32).reshape(-1)[:3*40].view(3, 10, 4)
    with pytest.raises(ValueError, match='disjoint'):
        prepared(b, [])


@pytest.mark.parametrize('kind', ['graph', 'split', 'unbound'])
def test_existing_forward_hooks_keep_generic_route_without_capture_queries(monkeypatch, kind):
    b, calls = buffers(), []
    b.hc_writeback = prepared(b, calls)
    if kind == 'graph':
        b.owner_graph_family = 'owner-graph'
    elif kind == 'split':
        b.split = object()
    else:
        b.hc_writeback = None
    generic = []
    monkeypatch.setattr(forward.glue, 'hc_writeback', lambda *a, **k: generic.append((a, k)))
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: pytest.fail('hot capture query'))
    forward._hc_writeback(b, b.h, b.h, b.pss, 4, 0)
    assert len(generic) == 1 and calls == []


@pytest.mark.parametrize('mode', [0, 3])
@pytest.mark.parametrize('role', ['decode', 'mtp'])
@pytest.mark.parametrize('graph_only,split,exact,owner,prefill,native', [
    (False, False, False, 'uncaptured', False, True),
    (False, False, True, 'uncaptured', True, True),
    (False, True, False, 'uncaptured', False, False),
    (False, True, True, 'uncaptured', False, False),
    (True, False, False, 'uncaptured', False, False),
    (True, False, True, 'uncaptured', False, False),
    (True, True, False, 'uncaptured', False, False),
    (True, True, True, 'uncaptured', False, True),
    (True, True, True, 'owner-graph', False, False),
    (True, True, True, 'uncaptured', True, False),
])
def test_graph_only_hook_uses_only_private_decode_capture_and_preserves_old_policy(
        monkeypatch, mode, role, graph_only, split, exact, owner, prefill, native):
    b, calls = buffers(), []
    b.hc_writeback = prepared(b, calls)
    b.hc_writeback_graph_only, b.exact_graph_handoffs = graph_only, exact
    b.owner_role = role
    b.split, b.owner_graph_family, b.prefill = object() if split else None, owner, prefill
    generic = []
    monkeypatch.setattr(forward.glue, 'hc_writeback', lambda *a, **k: generic.append((a, k)))
    monkeypatch.setattr(torch.cuda, 'is_current_stream_capturing', lambda: pytest.fail('hot capture query'))
    branch, inject = (b.g_branch.view(8, 3, 2560), b.inj_a) if mode else (None, None)
    forward._hc_writeback(b, b.h, b.h, b.pss, 4, mode, branch=branch, inject=inject)
    assert len(calls) == int(native) and len(generic) == int(not native)
    if generic:
        assert all(actual is expected for actual, expected in zip(generic[0][0][:3], (b.h, b.h, b.pss)))
        assert generic[0][1]['branch'] is branch and generic[0][1]['inject'] is inject


@pytest.mark.parametrize('fault', ['mode', 'foreign', 'source-grad', 'unbound'])
def test_private_capture_hook_keeps_existing_capability_refusals_on_generic_route(monkeypatch, fault):
    b, calls, generic = buffers(), [], []
    b.hc_writeback = prepared(b, calls)
    b.hc_writeback_graph_only, b.exact_graph_handoffs, b.split = True, True, object()
    h, mode, branch = b.h, 3, b.g_branch.view(8, 3, 2560)
    if fault == 'mode':
        mode = 4
    elif fault == 'foreign':
        h = h.clone()
    elif fault == 'source-grad':
        branch.requires_grad_(True)
    else:
        b.hc_writeback = None
    monkeypatch.setattr(forward.glue, 'hc_writeback', lambda *a, **kw: generic.append((a, kw)))
    forward._hc_writeback(b, h, h, b.pss, 4, mode, branch=branch, inject=b.inj_a)
    assert not calls and len(generic) == 1
    assert generic[0][0][0] is h and generic[0][0][4] == mode
    assert generic[0][1]['branch'] is branch


@pytest.mark.parametrize('value,valid', [(None, False), ('true', False), (1, False), (True, True), (False, True)])
def test_buffer_capture_flag_is_retained_and_strict_before_allocations(value, valid):
    source = Path(forward.__file__).with_name('state.py')
    tree = ast.parse(source.read_text())
    buffers_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Buffers')
    initialize = next(node for node in buffers_class.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    index = next(index for index, node in enumerate(initialize.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == 'hc_writeback_graph_only' for target in node.targets))
    nodes = initialize.body[index:index+2]
    owner = NS()
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec')
    scope = {'self': owner, 'w': NS(meta={'hc_writeback_graph_only': value})}
    if valid:
        exec(code, scope)
        assert owner.hc_writeback_graph_only is value
    else:
        with pytest.raises(ValueError, match='explicit boolean'):
            exec(code, scope)
    empty = NS()
    exec(code, {'self': empty, 'w': NS(meta={})})
    assert empty.hc_writeback_graph_only is False
    assert nodes[-1].end_lineno < next(node.lineno for node in ast.walk(initialize)
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Attribute) and target.attr == 'h' for target in node.targets))


@pytest.mark.parametrize('rows,prepared_capability,match', [
    (257, object(), '1..256-row'), (0, object(), '1..256-row'),
    (5, None, 'resident prepared capability')])
def test_explicit_decode_capture_rejects_unsupported_envelope_before_any_allocation(
        monkeypatch, rows, prepared_capability, match):
    from tensorfold.families.qwen4_exp.cuda.state import Buffers

    def allocation(*args, **kwargs):
        pytest.fail('unsupported explicit HC capture reached allocation')
    monkeypatch.setattr(torch, 'empty', allocation)
    monkeypatch.setattr(torch, 'zeros', allocation)
    monkeypatch.setattr(torch.cuda, 'Event', allocation)
    weights = NS(cfg=NS(streams=4, hidden=2560), device='cuda:0', positions=None,
                 meta={'hc_writeback_graph_only': True, 'hc_writeback_prepared': prepared_capability})
    with pytest.raises(ValueError, match=match):
        Buffers(weights, rows, 4096)


@pytest.mark.parametrize('world,dtype,prefill,flag,refused', [
    (8, torch.float16, False, True, False),
    (7, torch.float16, False, True, True),
    (8, torch.float32, False, True, True),
    (7, torch.float32, True, True, False),
    (7, torch.float32, False, False, False)])
def test_optional_capture_must_bind_its_actual_decode_capability(world, dtype, prefill, flag, refused):
    tree = ast.parse(Path(forward.__file__).with_name('state.py').read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Buffers')
    initialize = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == '__init__')
    at = next(index for index, node in enumerate(initialize.body) if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Attribute) and target.attr == 'hc_writeback' for target in node.targets))
    code = compile(ast.Module(body=initialize.body[at:], type_ignores=[]), '<actual Buffer HC binding>', 'exec')
    cap = object()
    owner = NS(hc_writeback_graph_only=flag)
    prepared_capability = NS(bind=lambda actual: cap)
    scope = {'self': owner, 'w': NS(meta={'hc_writeback_prepared': prepared_capability}),
             'bf': dtype, 'c': NS(streams=4, hidden=2560), 'rows': 5, 'world': world,
             'prefill': prefill, 'torch': torch}
    if refused:
        with pytest.raises(ValueError, match='actual bound decode capability'):
            exec(code, scope)
    else:
        exec(code, scope)
        assert owner.hc_writeback is (cap if world == 8 and dtype == torch.float16 else None)


def test_preparation_resolves_function_and_residency_once_before_use(monkeypatch):
    from tensorfold.cuda.sm70 import family_native
    events = []
    module = NS(prepare_writeback=lambda: events.append('resident'), writeback=lambda *a: events.append('launch'))
    monkeypatch.setattr(torch.cuda, 'get_device_capability', lambda device: (7, 0))
    monkeypatch.setattr(family_native, 'extension', lambda: module)
    value = ops.prepare_owned_hc_writeback('cuda:0')
    assert value.module is module and events == ['resident']
    assert value.writeback is module.writeback
    with pytest.raises(ValueError, match='SM70'):
        ops.prepare_owned_hc_writeback('cpu')
    assert events == ['resident']


def test_both_actual_writeback_calls_use_buffer_capability_and_preparation_precedes_admission():
    root = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/qwen4_exp/cuda'
    tree = ast.parse((root / 'forward.py').read_text())
    functions = {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}
    for name in ('hc_block', '_writeback'):
        assert any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == '_hc_writeback'
                   for n in ast.walk(functions[name]))
    engine = (root / 'engine.py').read_text()
    assert engine.index('hc_prepared_rows = _phase') < engine.index('self.capacity_plan = admit(')
    assert engine.index("w.meta['hc_writeback_prepared']") < engine.index('self.multi = MultiDecoder(')


@pytest.mark.parametrize('mode', [0, 3])
@pytest.mark.parametrize('inference_outputs', ['h', 'pss', 'both'])
def test_outside_inference_mode_keeps_reference_output_refusal(mode, inference_outputs):
    b, calls = buffers(), []
    b.g_branch.zero_()
    with torch.inference_mode():
        if inference_outputs in ('h', 'both'):
            b.h = b.h.clone()
        if inference_outputs in ('pss', 'both'):
            b.pss = b.pss.clone()
    cap = prepared(b, calls)
    branch = b.g_branch.view(8, 3, 2560) if mode else None
    inject = b.inj_a if mode else None
    assert cap.run(b.h, b.h, b.pss, 4, mode, branch=branch, inject=inject) is False
    assert calls == []
    # The original adapter performs required copy_ updates; refusal is kept,
    # including its original partial-update semantics when only PSS is inference.
    with pytest.raises(RuntimeError, match='[Ii]nference'):
        ops._reference['hc_writeback'](b.h, b.h, b.pss, 4, mode, branch=branch, inject=inject)

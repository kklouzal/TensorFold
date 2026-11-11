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
        owner_graph_family='uncaptured', split=None)


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

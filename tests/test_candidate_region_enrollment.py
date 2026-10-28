"""Actual enrollment boundaries; local packet arithmetic has independent GPU oracles."""
import ast
from pathlib import Path
from types import SimpleNamespace as N

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import forward, multi_graphs
from tensorfold.families.qwen4_exp.cuda import candidate_region_catalogue, moe_region_catalogue


def test_actual_candidate_hook_preserves_collective_and_fresh_prefix():
    calls = []
    b = N(logits=torch.randn(32, 71), cand=torch.empty(32, 65),
          cand_all=torch.empty(2 * 32 * 65), world=2)
    w = N(meta={'execution_provider': 'sm70'},
          comm=N(all_gather=lambda source, out: calls.append((source.clone(), out.numel()))))
    prefix = b.logits[:3]
    forward.candidates_local(w, b, prefix, 3)
    expected = b.cand[:3].clone()

    class Owner:
        def local_packet(self, weights, buffers, logits, rows, mapping, offset):
            assert weights is w and buffers is b and logits is prefix
            assert rows == 3 and mapping is None and offset == 0
            forward.candidates_local(weights, buffers, logits, rows)
            return True

    b.candidate_region_graphs = Owner()
    forward.candidates(w, b, prefix, 3)
    assert len(calls) == 1 and calls[0][1] == 2 * 3 * 65
    assert torch.equal(calls[0][0].view(torch.uint8), expected.view(torch.uint8))
    b.candidate_region_graphs = N(local_packet=lambda *args: False)
    b.cand.fill_(123)
    forward.candidates(w, b, prefix, 3)
    assert len(calls) == 2
    assert torch.equal(calls[1][0].view(torch.uint8), expected.view(torch.uint8))


def test_actual_cold_hook_publishes_ownership_before_prepare_and_only_canonical_buffers(monkeypatch):
    events = []
    canonical, mtp = N(), N()
    parent = N(bound=N(warm=lambda states: events.append('warm')),
        dec=N(w=N(meta={'sm70_moe_region_graphs_trial': True, 'sm70_candidate_region_graphs_trial': True}),
              buf=canonical, mbuf=mtp), moe_regions=None, candidate_regions=None)

    class MoE:
        def __init__(self, p):
            assert p.moe_regions is self
        def prepare(self):
            events.append('moe')

    class Candidate:
        def __init__(self, p):
            assert p.candidate_regions is self
            assert not hasattr(canonical, 'candidate_region_graphs')
        def prepare(self):
            events.append('candidate')

    monkeypatch.setattr(moe_region_catalogue, 'Catalogue', MoE)
    monkeypatch.setattr(candidate_region_catalogue, 'Catalogue', Candidate)
    multi_graphs.RoundGraphs._keys(parent, [])
    assert events == ['warm', 'moe', 'candidate']
    assert canonical.moe_region_graphs is parent.moe_regions
    assert canonical.candidate_region_graphs is mtp.candidate_region_graphs is parent.candidate_regions


def test_actual_engine_both_future_host_terms_survive_authoritative_admission_and_floor():
    root = Path(__file__).resolve().parents[1] / 'src/tensorfold/families/qwen4_exp/cuda'
    tree = ast.parse((root / 'engine.py').read_text())
    initialize = next(n for cls in tree.body if isinstance(cls, ast.ClassDef) and cls.name == 'FlashNextEngine'
                      for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == '_initialize')
    blocks = []
    for field in ('_moe_region_graphs_requested', '_candidate_region_graphs_requested'):
        blocks.append(next(n for n in ast.walk(initialize) if isinstance(n, ast.If)
            and isinstance(n.test, ast.Call) and isinstance(n.test.func, ast.Name)
            and n.test.func.id == 'getattr' and n.test.args[1].value == field))
    promote = next(n for n in ast.walk(initialize) if isinstance(n, ast.For)
        and isinstance(n.iter, ast.Tuple) and any(isinstance(v, ast.Constant) and v.value == 'host_workspace_bytes' for v in n.iter.elts))
    admit = next(n for n in ast.walk(initialize) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Name) and n.func.id == 'admit')
    assert all(n.lineno < promote.lineno < admit.lineno for n in blocks)
    scope = {'self': N(_moe_region_graphs_requested=True, _candidate_region_graphs_requested=True),
        'options': {'host_workspace_bytes': 37, 'device_workspace_bytes': 19}, 'tp': 8, 'host_capacity': {},
        '__package__': 'tensorfold.families.qwen4_exp.cuda'}
    exec(compile(ast.fix_missing_locations(ast.Module(body=blocks + [promote], type_ignores=[])), 'funding', 'exec'), scope)
    assert scope['host_capacity']['host_workspace_bytes'] == 37 + 16 * (1 << 30)
    assert scope['options']['device_workspace_bytes'] == 19
    assert any(k.arg is None and isinstance(k.value, ast.Name) and k.value.id == 'host_capacity' for k in admit.keywords)
    floor = next(n for n in ast.walk(initialize) if isinstance(n, ast.If) and 'streams < 24' in ast.unparse(n.test))
    from tensorfold.cuda.sm70 import execution_policy
    names = {name: getattr(execution_policy, name) for name in dir(execution_policy) if name.endswith('_TRIAL')}
    names.update(self=N(sm70_profile=execution_policy.SHARED_FUSION_TRIAL), streams=23)
    code = compile(ast.fix_missing_locations(ast.Module(body=[floor], type_ignores=[])), 'floor', 'exec')
    with pytest.raises(ValueError, match='at least24'):
        exec(code, names)
    names['streams'] = 24
    exec(code, names)

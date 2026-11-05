"""Default-off fused-tail preparation/dispatch boundaries; no device work."""
import ast
import copy
import inspect
import textwrap
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.families.gemma4.cuda.engine import Gemma4Engine
from tensorfold.families.gemma4.cuda.forward import Gemma4Target
from tensorfold.families.gemma4.cuda.nvfp4_experts import Experts


@pytest.mark.parametrize('value', [None, 0, 1, 'yes'])
def test_strict_boolean_refused_before_model_or_device_work(tmp_path, value):
    with pytest.raises(ValueError, match='fused_moe_tail must be a bool'):
        Gemma4Engine(tmp_path, fused_moe_tail=value)
    with pytest.raises(ValueError, match='fused_moe_tail must be a bool'):
        Gemma4Target(None, fused_moe_tail=value)


def test_target_optin_rejects_unqualified_numerics_geometry_and_adapter():
    cfg = NS(hidden=2816, moe_width=704, experts=128, top_k=8)
    for weights in (NS(numerics='modelopt_nvfp4', config=cfg, layers=[]),
                    NS(numerics='ct_nvfp4', config=NS(hidden=1024, moe_width=704, experts=128, top_k=8), layers=[]),
                    NS(numerics='ct_nvfp4', config=cfg, layers=[NS(experts=object())])):
        with pytest.raises(ValueError, match='requires native CT'):
            Gemma4Target(weights, fused_moe_tail=True)


@pytest.mark.parametrize('count,width,dims,rows,slots,direct', [
    (3, 704, 2816, 1, 8, False), (128, 64, 2816, 1, 8, False),
    (128, 704, 128, 1, 8, False), (128, 704, 2816, 6, 8, False),
    (128, 704, 2816, 1, 3, False), (128, 704, 2816, 1, 8, True),
])
def test_borrow_refusals_precede_native_or_workspace_acquisition(count, width, dims, rows, slots, direct):
    adapter = object.__new__(Experts)
    adapter.count, adapter.width, adapter.dims, adapter.direct_input = count, width, dims, direct
    adapter._run_slots = lambda *_: pytest.fail('unsupported borrow acquired workspace/native work')
    with pytest.raises(ValueError, match='fused MoE tail slots require'):
        adapter.forward_slots(torch.zeros(rows, dims, dtype=torch.bfloat16),
            torch.zeros(rows, slots, dtype=torch.int32), torch.ones(rows, slots))


def test_engine_preparation_forwards_only_explicit_true():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Gemma4Engine.__init__)))
    call = next(n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == 'Gemma4Target')
    scope = {'Gemma4Target': lambda *a, **kw: kw}
    code = compile(ast.Expression(body=copy.deepcopy(call)), '<actual-target-preparation>', 'eval')
    options = dict(self=NS(w=object(), context_window=131072), prepare_global_prompt=False,
        global_trig=False, global_cache='none', global_cache_prompt=False, global_trig_prompt=True)
    for selected in (False, True):
        actual = eval(code, scope, options | {'fused_moe_tail': selected})
        assert ('fused_moe_tail' in actual) is selected
        if selected:
            assert actual['fused_moe_tail'] is True


def test_real_target_branch_keeps_prompt_and_generic_paths_and_consumes_borrow_before_reuse():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Gemma4Target._layers)))
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and any(
        isinstance(item, ast.Attribute) and item.attr == '_fused_moe_tail' for item in ast.walk(n.test)))
    code = compile(ast.Module(body=[copy.deepcopy(branch)], type_ignores=[]), '<actual-fused-tail-branch>', 'exec')
    events = []
    borrowed = NS(y=object(), picks=object(), routing=object())
    class Adapter:
        def forward_slots(self, *args):
            events.append('borrow');return borrowed
        def forward(self, *args):
            events.append('combine');return 'routed'
    def fused(*args):
        events.append('consume')
        assert args[2:5] == (borrowed.y, borrowed.picks, borrowed.routing)
        return 'h', 'normed'
    def old(*args):
        events.append('old-tail');return 'h', 'normed'
    layer = NS(experts=Adapter(), dense_output_norm=1, expert_output_norm=2, output_norm=3, scalar=4)
    common = dict(layer=layer, glue=NS(moe_tail_slots=fused, moe_tail=old), expert_in=0, picks=0,
        routing=0, h=0, dense=0, following=5, c=NS(eps=1e-6), mlx=False)
    for selected, prompt, rows, expected in ((True, False, 1, ['borrow', 'consume']),
        (True, False, 5, ['borrow', 'consume']), (True, True, 2, ['combine', 'old-tail']),
        (True, False, 6, ['combine', 'old-tail']), (False, False, 2, ['combine', 'old-tail'])):
        events.clear();scope = common | {'self': NS(_fused_moe_tail=selected), 'prompt': prompt, 'rows': rows}
        exec(code, scope)
        assert events == expected
        if selected and not prompt and rows <= 5:
            assert scope['borrowed'] is None


def test_consumer_failure_keeps_borrowed_owner_and_does_not_retry_or_fallback():
    tree = ast.parse(textwrap.dedent(inspect.getsource(Gemma4Target._layers)))
    branch = next(n for n in ast.walk(tree) if isinstance(n, ast.If) and any(
        isinstance(item, ast.Attribute) and item.attr == '_fused_moe_tail' for item in ast.walk(n.test)))
    code = compile(ast.Module(body=[copy.deepcopy(branch)], type_ignores=[]), '<actual-fused-tail-failure>', 'exec')
    owner = object()
    borrowed = NS(owner=owner, y=object(), picks=object(), routing=object())
    primary = MemoryError('consumer allocation exhausted')
    events = []
    def produce(*_):
        events.append('borrow');return borrowed
    def consume(*_):
        events.append('consume');raise primary
    def forbidden(*_):
        pytest.fail('failed fused consumer must not retry or fall back')
    layer = NS(experts=NS(forward_slots=produce, forward=forbidden), dense_output_norm=1,
        expert_output_norm=2, output_norm=3, scalar=4)
    scope = dict(self=NS(_fused_moe_tail=True), prompt=False, rows=2, layer=layer,
        glue=NS(moe_tail_slots=consume, moe_tail=forbidden), expert_in=0, picks=0, routing=0,
        h=0, dense=0, following=5, c=NS(eps=1e-6), mlx=False)
    try:
        exec(code, scope)
    except MemoryError as actual:
        assert actual is primary
        assert scope['borrowed'] is borrowed and scope['borrowed'].owner is owner
        assert events == ['borrow', 'consume']
    else:
        pytest.fail('consumer failure must propagate')

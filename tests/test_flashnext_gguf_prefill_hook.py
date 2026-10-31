"""Admission and load receive one frozen prefill plan; disabled kwargs stay absent."""

import ast
from pathlib import Path
from types import SimpleNamespace as NS

import pytest


@pytest.mark.parametrize('enabled',[False,True])
@pytest.mark.parametrize('stock_mtp',[False,True])
def test_same_single_plan_reaches_admission_loader_and_receipt(enabled, stock_mtp):
    path=Path(__file__).parents[1]/'src/tensorfold/families/qwen4_exp/cuda/engine.py'
    tree=ast.parse(path.read_text())
    method=next(n for n in ast.walk(tree) if isinstance(n,ast.FunctionDef) and n.name=='_initialize')
    assignments=[]
    for node in ast.walk(method):
        if not isinstance(node,ast.Assign):
            continue
        if isinstance(node.value,ast.Call) and isinstance(node.value.func,ast.Name) \
                and node.value.func.id in ('prefill_expert_plan','capacity_options'):
            assignments.append(node)
        elif any(isinstance(t,ast.Attribute) and t.attr=='_gguf_prefill_receipt' for t in node.targets):
            assignments.append(node)
    assignments.sort(key=lambda node:node.lineno)
    plan=NS(receipt=lambda:{'pairs':32}) if enabled else None
    events=[]
    target=NS(depth=3,mtp_model_dir=None,rope=None)

    def choose(text,**kwargs):
        events.append(('choose',text,kwargs))
        return plan

    def admission(*args,**kwargs):
        events.append(('admission',kwargs))
        return {}

    def load(*args,**kwargs):
        events.append(('load',kwargs))

    env=dict(self=target,prefill_expert_plan=choose,capacity_options=admission,load=load,text={'hidden_size':2560},
             rows0=2048,tp=4,rank=0,mtp=stock_mtp,stock_mtp=stock_mtp,ids=None,model_dir=Path('/checkpoint'),draft_vocab='default',
             ple_on_ssd=False,reads=None,prefetch=False,gguf=True,eos_token_ids=None,expert_session=None,
             vram_experts=None,vram_experts_pin='off')
    exec(compile(ast.Module(body=assignments,type_ignores=[]),str(path),'exec'),env)
    call=next(n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='load')
    eval(compile(ast.Expression(call),str(path),'eval'),env)
    assert [row[0] for row in events]==['choose','admission','load']
    assert events[0][2]=={'rows':2048,'world':4}
    assert events[1][1]['mtp'] is stock_mtp
    assert events[2][1]['mtp'] is stock_mtp
    assert target._gguf_prefill_receipt==({'pairs':32} if enabled else None)
    for row,key in ((events[1],'prefill_plan'),(events[2],'_prefill_plan')):
        assert (row[1].get(key) is plan) if enabled else key not in row[1]

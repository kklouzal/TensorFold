"""N128 role/shape/native-boundary controls; no CUDA execution."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT=Path(__file__).resolve().parents[1]/'src/tensorfold'


def definitions(path,names,scope):
    nodes=[n for n in ast.parse(path.read_text()).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    assert {n.name for n in nodes}==set(names)
    exec(compile(ast.fix_missing_locations(ast.Module(body=nodes,type_ignores=[])),str(path),'exec'),scope)
    return scope


def selector():
    return definitions(ROOT/'cuda/nvfp4/experts_a4.py',{'up_output_tile'}, {})['up_output_tile']


def eligible():return dict(decode=True,rows=5,common=True,direct=False,activation=1,width=704,dims=2816,plan_tile=16)


@pytest.mark.parametrize('rows',[1,2,3,4,5])
def test_actual_ct_geometry_selects_new_output_tile(rows):
    args=eligible();args['rows']=rows
    assert selector()(**args)==128


@pytest.mark.parametrize('field,value',[('decode',False),('rows',6),('common',False),('direct',True),
    ('activation',0),('width',128),('dims',128),('plan_tile',64)])
def test_other_supported_regions_keep_n64(field,value):
    args=eligible();args[field]=value
    assert selector()(**args)==64


@pytest.mark.parametrize('value',[1,None,'true'])
def test_invalid_role_rejected(value):
    args=eligible();args['decode']=value
    with pytest.raises(ValueError):selector()(**args)


def test_actual_run_plan_requires_prefill_arithmetic_before_n128():
    path=ROOT/'cuda/nvfp4/experts_a4.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ExpertsA4')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='run_plan')
    call=next(n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='up_output_tile')
    expression=next(k.value for k in call.keywords if k.arg=='decode')
    for prefill,decode,expected in [(True,True,True),(False,True,False),(True,False,False)]:
        assert eval(compile(ast.Expression(body=expression),'actual-role','eval'),{'prefill':prefill,'decode':decode}) is expected
    first_role_check=next(i for i,n in enumerate(method.body) if isinstance(n,ast.If) and 'decode' in ast.unparse(n.test))
    first_native=next(i for i,n in enumerate(method.body) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ext' for t in n.targets))
    assert first_role_check<first_native


def test_ct_prompt_adapter_is_explicit_and_legacy_default_conservative():
    path=ROOT/'families/gemma4/cuda/nvfp4_experts.py';tree=ast.parse(path.read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Experts')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    index=next(i for i,arg in enumerate(method.args.kwonlyargs) if arg.arg=='prompt')
    assert method.args.kw_defaults[index].value is True
    call=next(n for n in ast.walk(method) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='run_plan')
    expression=next(k.value for k in call.keywords if k.arg=='decode')
    assert eval(compile(ast.Expression(body=expression),'actual-ct-role','eval'),{'prompt':True}) is False
    assert eval(compile(ast.Expression(body=expression),'actual-ct-role','eval'),{'prompt':False}) is True


def test_forward_prompt_keyword_only_on_owned_ct_boundary():
    tree=ast.parse((ROOT/'families/gemma4/cuda/forward.py').read_text())
    assignment=next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='routed' for t in n.targets))
    expression=assignment.value
    assert isinstance(expression,ast.IfExp) and ast.unparse(expression.test)=='self._native'
    for native in (False,True):
        calls=[]
        layer=SimpleNamespace(experts=SimpleNamespace(forward=lambda *args,**kwargs:calls.append(kwargs)))
        eval(compile(ast.Expression(body=expression),'actual-forward-boundary','eval'),
             {'self':SimpleNamespace(_native=native),'layer':layer,'expert_in':None,'picks':None,'routing':None,'prompt':True})
        assert calls==([{'prompt':True}] if native else [{}])

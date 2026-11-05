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


@pytest.mark.parametrize('rows',[1,2,3,4,5,6,16,129])
@pytest.mark.parametrize('prompt',[False,True])
@pytest.mark.parametrize('enabled',[False,True])
def test_actual_target_dispatch_only_row2_and_other_calls_original_n64(rows,prompt,enabled):
    tree=ast.parse((ROOT/'families/gemma4/cuda/forward.py').read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If)
        and any(isinstance(t,ast.Attribute) and t.attr=='_expert_up_n128_row2' for t in ast.walk(n.test)))
    calls=[];inputs=(object(),object(),object())
    layer=SimpleNamespace(experts=SimpleNamespace(forward=lambda *args,**kw:calls.append((args,kw))))
    scope={'self':SimpleNamespace(_expert_up_n128_row2=enabled),'layer':layer,'expert_in':inputs[0],
        'picks':inputs[1],'routing':inputs[2],'rows':rows,'prompt':prompt}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),'actual-Target-boundary','exec'),scope)
    assert calls==[(inputs,{'prompt':False} if enabled and not prompt and rows==2 else {})]


def constructor(path,name):
    tree=ast.parse(path.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    return next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')


def test_private_engine_flag_strict_default_and_true_only_forwarding():
    ctor=constructor(ROOT/'families/gemma4/cuda/engine.py','Gemma4Engine')
    index=next(i for i,n in enumerate(ctor.args.kwonlyargs) if n.arg=='expert_up_n128_row2')
    assert ctor.args.kw_defaults[index].value is False
    guard=next(n for n in ctor.body if isinstance(n,ast.If) and 'type(expert_up_n128_row2)' in ast.unparse(n.test))
    for value in (0,1,None,'true'):
        with pytest.raises(ValueError):
            exec(compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'actual-bool','exec'),{'expert_up_n128_row2':value})
    call=next(n for n in ast.walk(ctor) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Gemma4Target')
    forwarding=next(k.value for k in call.keywords if k.arg is None and 'expert_up_n128_row2' in ast.unparse(k.value))
    for enabled in (False,True):
        assert eval(compile(ast.Expression(body=forwarding),'actual-forwarding','eval'),{'expert_up_n128_row2':enabled})==({'expert_up_n128_row2':True} if enabled else {})


def test_engine_optin_metadata_and_capability_gate_precedes_weight_allocation():
    ctor=constructor(ROOT/'families/gemma4/cuda/engine.py','Gemma4Engine')
    preflight=next(n for n in ctor.body if isinstance(n,ast.If) and ast.unparse(n.test)=='expert_up_n128_row2')
    guard=next(n for n in preflight.body if isinstance(n,ast.If))
    loads=[n for n in ast.walk(ctor) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute)
        and isinstance(n.func.value,ast.Name) and n.func.value.id=='Gemma4Weights']
    assert loads and guard.lineno<min(n.lineno for n in loads)
    for method,shape,capability,valid in [('compressed-tensors',(2816,704,128),(12,1),True),
            ('modelopt',(2816,704,128),(12,1),False),('compressed-tensors',(2816,512,128),(12,1),False),
            ('compressed-tensors',(2816,704,128),(12,0),False)]:
        calls=[];c=SimpleNamespace(hidden=shape[0],moe_width=shape[1],experts=shape[2])
        scope={'method':method,'c':c,'torch':SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda device:calls.append(device) or capability))}
        code=compile(ast.fix_missing_locations(ast.Module(body=[guard],type_ignores=[])),'actual-engine-preflight','exec')
        if valid:exec(code,scope)
        else:
            with pytest.raises(ValueError):exec(code,scope)
        if method!='compressed-tensors' or shape!=(2816,704,128):assert calls==[]


def test_direct_target_optin_checks_actual_geometry_before_internal_storage():
    ctor=constructor(ROOT/'families/gemma4/cuda/forward.py','Gemma4Target')
    end=next(i for i,n in enumerate(ctor.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=='_expert_up_n128_row2' for t in n.targets))
    preparation=ast.fix_missing_locations(ast.Module(body=ctor.body[:end+1],type_ignores=[]))
    for enabled,method,shape,capability,valid in [(False,'mlx',(1,2,3),(7,0),True),
            (True,'ct_nvfp4',(2816,704,128),(12,1),True),(True,'modelopt_nvfp4',(2816,704,128),(12,1),False),
            (True,'ct_nvfp4',(2816,512,128),(12,1),False),(True,'ct_nvfp4',(2816,704,128),(12,0),False),
            (1,'ct_nvfp4',(2816,704,128),(12,1),False)]:
        calls=[];owner=SimpleNamespace()
        weights=SimpleNamespace(numerics=method,config=SimpleNamespace(hidden=shape[0],moe_width=shape[1],experts=shape[2]),norm=SimpleNamespace(device='cuda:0'))
        scope={'self':owner,'weights':weights,'expert_up_n128_row2':enabled,
            'torch':SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda device:calls.append(device) or capability))}
        if valid:
            exec(compile(preparation,'actual-target-preparation','exec'),scope)
            assert owner._expert_up_n128_row2 is enabled
            assert calls==(['cuda:0'] if enabled else [])
        else:
            with pytest.raises(ValueError):exec(compile(preparation,'actual-target-preparation','exec'),scope)

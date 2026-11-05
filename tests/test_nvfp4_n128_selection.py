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


def eligible():return dict(decode=True,rows=5,common=True,direct=False,activation=1,width=704,dims=2816,plan_tile=16,n128_warps=8)


@pytest.mark.parametrize('rows',[1,2,3,4,5])
@pytest.mark.parametrize('warps',[4,8])
def test_actual_ct_geometry_selects_new_output_tile(rows,warps):
    args=eligible();args['rows']=rows;args['n128_warps']=warps
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


def test_forward_optin_only_on_owned_ct_boundary_default_call_is_unchanged():
    tree=ast.parse((ROOT/'families/gemma4/cuda/forward.py').read_text())
    branch=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and ast.unparse(n.test)=='self._expert_up_n128_8warp')
    for enabled in (False,True):
        calls=[]
        layer=SimpleNamespace(experts=SimpleNamespace(forward=lambda *args,**kwargs:calls.append(kwargs)))
        scope={'self':SimpleNamespace(_expert_up_n128_8warp=enabled),'layer':layer,'expert_in':None,'picks':None,'routing':None,'prompt':False}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[branch],type_ignores=[])),'actual-forward-boundary','exec'),scope)
        assert calls==([{'prompt':False,'n128_warps':8}] if enabled else [{}])


def test_default_mode_is_n64_and_invalid_modes_are_rejected():
    args=eligible();args.pop('n128_warps')
    assert selector()(**args)==64
    for value in (None,True,1,16):
        with pytest.raises(ValueError):selector()(**args,n128_warps=value)


def test_actual_CTA_copy_stride_covers_each_chunk_exactly_once_for_both_mappings():
    source=(ROOT/'cuda/nvfp4/experts_a4.cu').read_text()
    assert 'constexpr int TILE_WARPS = 4;' in source
    assert 'c += WARPS * 32' in source and 'stage_tile<WARPS>(stage[buffer][p][panel]' in source
    assert '__launch_bounds__(WARPS * 32) fused_up_kernel' in source
    def indices(launched,stride):
        return [c for thread in range(launched*32) for c in range(thread,144,stride*32)]
    for warps in (4,8):assert sorted(indices(warps,warps))==list(range(144))
    assert len(indices(8,4))!=len(set(indices(8,4)))  # retained overlapping-copy defect control


def test_private_engine_flag_is_strict_and_only_true_is_forwarded():
    tree=ast.parse((ROOT/'families/gemma4/cuda/engine.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Gemma4Engine')
    ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    index=next(i for i,arg in enumerate(ctor.args.kwonlyargs) if arg.arg=='expert_up_n128_8warp')
    assert ctor.args.kw_defaults[index].value is False
    check=next(n for n in ctor.body if isinstance(n,ast.If) and 'type(expert_up_n128_8warp)' in ast.unparse(n.test))
    for value in (0,None,'true'):
        with pytest.raises(ValueError):exec(compile(ast.fix_missing_locations(ast.Module(body=[check],type_ignores=[])),'actual-flag','exec'),{'expert_up_n128_8warp':value})
    call=next(n for n in ast.walk(ctor) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='Gemma4Target')
    forwarding=next(k.value for k in call.keywords if k.arg is None and 'expert_up_n128_8warp' in ast.unparse(k.value))
    for enabled in (False,True):
        assert eval(compile(ast.Expression(body=forwarding),'actual-forwarding','eval'),{'expert_up_n128_8warp':enabled})==({'expert_up_n128_8warp':True} if enabled else {})


def test_actual_target_optin_refuses_other_geometry_and_device_before_storage():
    tree=ast.parse((ROOT/'families/gemma4/cuda/forward.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Gemma4Target')
    ctor=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='__init__')
    end=next(i for i,n in enumerate(ctor.body) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=='_expert_up_n128_8warp' for t in n.targets))
    preparation=ast.fix_missing_locations(ast.Module(body=ctor.body[:end+1],type_ignores=[]))
    for enabled,numerics,geometry,capability,valid in (
            (False,'mlx',(1,2,3),(7,0),True),
            (True,'ct_nvfp4',(2816,704,128),(12,1),True),
            (True,'modelopt_nvfp4',(2816,704,128),(12,1),False),
            (True,'ct_nvfp4',(2816,512,128),(12,1),False),
            (True,'ct_nvfp4',(2816,704,128),(12,0),False)):
        calls=[];owner=SimpleNamespace()
        weights=SimpleNamespace(numerics=numerics,config=SimpleNamespace(hidden=geometry[0],moe_width=geometry[1],experts=geometry[2]),norm=SimpleNamespace(device='cuda:0'))
        scope={'self':owner,'weights':weights,'expert_up_n128_8warp':enabled,
            'torch':SimpleNamespace(cuda=SimpleNamespace(get_device_capability=lambda device:calls.append(device) or capability))}
        if valid:
            exec(compile(preparation,'actual-target-preparation','exec'),scope)
            assert owner._expert_up_n128_8warp is enabled
            assert calls==(['cuda:0'] if enabled else [])
        else:
            with pytest.raises(ValueError):exec(compile(preparation,'actual-target-preparation','exec'),scope)

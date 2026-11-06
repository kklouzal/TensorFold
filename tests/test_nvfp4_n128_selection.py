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


def test_ct_prompt_adapter_default_remains_conservative_without_keyword_dicts():
    tree=ast.parse((ROOT/'families/gemma4/cuda/nvfp4_experts.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Experts')
    method=next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    index=next(i for i,arg in enumerate(method.args.kwonlyargs) if arg.arg=='prompt')
    assert method.args.kw_defaults[index].value is True
    branch=next(n for n in method.body if isinstance(n,ast.If) and isinstance(n.test,ast.Name) and n.test.id=='prompt')
    assert not any(isinstance(n,ast.Dict) for n in ast.walk(branch))


class View:
    def __init__(self,shape,dtype='bf16'):
        self.shape=tuple(shape);self.dtype=dtype;self.device='cuda:0';self.ndim=len(self.shape)
    def __getitem__(self,key):
        keys=key if isinstance(key,tuple) else (key,)
        shape=list(self.shape)
        for index,item in enumerate(keys):
            if isinstance(item,slice):shape[index]=len(range(*item.indices(shape[index])))
        return View(shape,self.dtype)
    def view(self,*shape):
        from math import prod
        return View((prod(self.shape),) if shape==(-1,) else shape,self.dtype)
    def stride(self,index):
        from math import prod
        index=index%self.ndim
        return prod(self.shape[index+1:])


def run_plan_fixture(monkeypatch,*,rows=2,common=True,direct=False,activation=1,width=704,dims=2816,tile=16,prefill=True):
    import sys
    from types import ModuleType,MethodType
    tree=ast.parse((ROOT/'cuda/nvfp4/experts_a4.py').read_text())
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='ExpertsA4')
    methods=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in ('run_plan','_check')]
    native=[];selected=[];splits=[]
    def record(name):return lambda *args:native.append((name,args))
    ext=SimpleNamespace(quantize=record('quantize'),quantize_plan=record('quantize_plan'),
        fused_up=record('fused_up'),down_plan=record('down_plan'))
    def choose(**kwargs):selected.append(kwargs);return selector()(**kwargs)
    def forbidden(*args,**kwargs):raise AssertionError('preallocated valid path must not allocate/query capture')
    torch=SimpleNamespace(bfloat16='bf16',float32='f32',int32='i32',cuda=SimpleNamespace(is_current_stream_capturing=forbidden),empty=forbidden)
    scope={'torch':torch,'_ext':lambda:ext,'up_output_tile':choose,'_overlap':lambda a,b:False}
    exec(compile(ast.fix_missing_locations(ast.Module(body=methods,type_ignores=[])),'actual-run-plan','exec'),scope)
    module=ModuleType('tensorfold.cuda.kernels')
    module.qmm=SimpleNamespace(split_k=lambda n,k:splits.append((n,k)) or 2)
    monkeypatch.setitem(sys.modules,'tensorfold.cuda.kernels',module)
    owner=SimpleNamespace(dims=dims,width=width,count=128,device='cuda:0',shared_input=common,
        direct_input=direct,swiglu=SimpleNamespace(activation=activation),native_down=View((128,)),
        fields=tuple(View((128,3)) for _ in range(6)),device_tensors=lambda:())
    owner._check=MethodType(scope['_check'],owner)
    slots=8;capacity=8
    scratch=(View((2,capacity*slots,dims//2),'u8'),View((2,capacity*slots,dims//16),'u8'),
        View((1,capacity*slots,width//2),'u8'),View((1,capacity*slots,width//16),'u8'))
    buf=SimpleNamespace(rows=capacity,slots=slots,pick=View((capacity,slots),'i32'),
        y=View((capacity,slots,dims),'bf16' if prefill else 'f32'),a4_scratch={(dims,width):scratch},
        a4_row_plan=View((capacity,3),'i32'),plan=SimpleNamespace(tile=tile,items=View((128,4),'i32'),
            members=View((capacity*slots,),'i32'),counts=View((128,),'i32')))
    return scope['run_plan'],owner,View((rows,dims)),buf,native,selected,splits


@pytest.mark.parametrize('decode',[False,True])
@pytest.mark.parametrize('prefill',[False,True])
@pytest.mark.parametrize('changes',[{},dict(common=False),dict(direct=True),dict(activation=0),
    dict(rows=6),dict(width=128),dict(dims=128),dict(tile=64)])
def test_actual_run_plan_named_value_skip_and_every_native_scalar_argument(monkeypatch,decode,prefill,changes):
    run,owner,x,buf,native,selected,splits=run_plan_fixture(monkeypatch,prefill=prefill,**changes)
    run(owner,x,buf,prefill=prefill,decode=decode)
    expected_args=dict(decode=True,rows=x.shape[0],common=owner.shared_input,direct=owner.direct_input,
        activation=owner.swiglu.activation,width=owner.width,dims=owner.dims,plan_tile=buf.plan.tile)
    expected=selector()(**expected_args) if decode and prefill else 64
    assert selected==([expected_args] if decode and prefill else [])
    fused=next(args for name,args in native if name=='fused_up')
    assert fused[11:]==(owner.width,owner.dims,1 if prefill else 2,buf.slots,owner.shared_input,
        owner.direct_input,buf.plan.tile,owner.swiglu.activation,expected)
    assert [name for name,_ in native]==['quantize' if owner.shared_input else 'quantize_plan','fused_up','down_plan']
    assert splits==([] if prefill else [(owner.width,owner.dims),(owner.dims,owner.width)])


@pytest.mark.parametrize('decode',[False,True])
@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_actual_run_plan_invalid_prefill_is_rejected_before_selector_native_or_allocation(monkeypatch,decode,value):
    run,owner,x,buf,native,selected,_=run_plan_fixture(monkeypatch)
    with pytest.raises(ValueError):run(owner,x,buf,prefill=value,decode=decode)
    assert native==selected==[]


@pytest.mark.parametrize('value',[None,0,1,'true'])
def test_actual_run_plan_invalid_decode_is_rejected_before_buffer_use(monkeypatch,value):
    run,owner,_,_,native,selected,_=run_plan_fixture(monkeypatch)
    with pytest.raises(ValueError,match='decode role'):run(owner,None,None,prefill=True,decode=value)
    assert native==selected==[]


def test_actual_run_plan_omitted_role_uses_named_n64_without_selector(monkeypatch):
    run,owner,x,buf,native,selected,_=run_plan_fixture(monkeypatch)
    run(owner,x,buf,prefill=True)
    assert selected==[] and next(args for name,args in native if name=='fused_up')[-1]==64


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

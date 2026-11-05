"""Owned CPU weight-copy adapter controls; stand-ins are not native/performance proof."""
import json
from types import SimpleNamespace
import weakref

import pytest
import torch

from tensorfold.cuda import requant
from tensorfold.cuda.nvfp4 import linear
from tensorfold.families.gemma4.cuda import capacity, dflash
from tensorfold.families.gemma4.cuda.engine import Gemma4Engine


def bridge(monkeypatch, tmp_path, *, fault=None, indexed=False):
    events=[];refs=[];captured={};primary=RuntimeError('original failure');cleanup=RuntimeError('reader close')
    original=(torch.arange(128*64).reshape(128,64).remainder(31).float()/7).to(torch.bfloat16)
    # Independent same-library reference: one whole CPU recipe, not per-chunk globals.
    expected=requant.nvfp4_rows(original)
    path=tmp_path/('head.safetensors' if indexed else 'model.safetensors');path.touch()
    if indexed:(tmp_path/'model.safetensors.index.json').write_text(json.dumps({'weight_map':{'model.language_model.embed_tokens.weight':'head.safetensors'}}))
    class Reader:
        def __init__(self,files):
            assert files==[path];events.append('reader')
        def get(self,name):
            assert name=='model.language_model.embed_tokens.weight'
            if fault=='read':raise primary
            value=original.clone();refs.append(weakref.ref(value));return value
        def close(self):
            events.append('close')
            if fault=='read':raise cleanup
    monkeypatch.setattr(dflash,'SafeTensors',Reader)
    recipe=requant.nvfp4_rows
    def quantize(value):
        events.append('quantize');assert value.device.type=='cpu'
        captured['source_shape']=tuple(value.shape)
        return recipe(value)
    monkeypatch.setattr(requant,'nvfp4_rows',quantize)
    original_to=torch.Tensor.to
    def to(value,*args,**kwargs):
        if kwargs.get('device')=='cuda':
            assert refs and refs[0]() is None  # source retires before GPU upload; fields own their bytes
            assert events[-1] in ('close','upload')
            events.append('upload');return value.clone()
        return original_to(value,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',to)
    def pack(codes,scales,g,act=None):
        events.append('pack');assert act is None
        assert torch.equal(codes,expected[0]) and torch.equal(scales,expected[1]) and g==float(expected[2])
        captured.update(codes=codes,scales=scales,global_scale=g)
        if fault=='pack':raise primary
        return linear.Fp4Linear(torch.empty((2,1,8,32,2),dtype=torch.int32),torch.empty((2,1,64,4),dtype=torch.uint8),g,128,64,act=None)
    monkeypatch.setattr(linear.Fp4Linear,'from_checkpoint',staticmethod(pack))
    def sync(device):
        events.append('sync');assert device=='cuda'
        if fault=='initial-fence':raise primary
    monkeypatch.setattr(torch.cuda,'synchronize',sync)
    return events,refs,captured,primary,cleanup


@pytest.mark.parametrize('indexed',[False,True])
def test_original_cpu_whole_recipe_upload_then_existing_head(monkeypatch,tmp_path,indexed):
    events,refs,captured,_,_=bridge(monkeypatch,tmp_path,indexed=indexed)
    result=dflash._cpu_nvfp4_head(tmp_path,'cuda',(128,64),softcap=30.)
    assert events==['sync','reader','quantize','close','upload','upload','pack','sync']
    assert isinstance(result,requant.Head) and isinstance(result.linear,linear.Fp4Linear)
    assert result.linear.act is None and result.softcap==30. and result.n==128
    assert refs[0]() is None and captured['source_shape']==(128,64)
    # Exercise the real Head's owning F32/single-cap boundary; native product is separate GPU evidence.
    outputs=[]
    def product(mode,w,bs,scale,n,k,npad,x,out,cols=None):
        assert mode==linear.FP4 and out.dtype==torch.float32 and out.shape==(len(x),128)
        out.fill_(2.);outputs.append(out);return out
    monkeypatch.setattr(linear,'_matmul',product)
    x=torch.ones((3,64),dtype=torch.bfloat16)
    a=result(x);b=result(x)
    assert a.dtype==torch.float32 and a.data_ptr()!=b.data_ptr()
    assert all(a.data_ptr()!=out.data_ptr() for out in outputs)
    assert torch.equal(a,torch.tanh(torch.full((3,128),2.,dtype=torch.float32)/30.)*30.)


def test_cpu_read_failure_keeps_primary_when_close_also_fails(monkeypatch,tmp_path):
    events,_,_,primary,cleanup=bridge(monkeypatch,tmp_path,fault='read')
    caught=None
    try:dflash._cpu_nvfp4_head(tmp_path,'cuda',(128,64),softcap=30.)
    except RuntimeError as error:caught=error
    assert caught is primary and caught.__cause__ is cleanup
    assert events==['sync','reader','close']


def test_native_pack_failure_retains_operands_and_does_not_retry_fence(monkeypatch,tmp_path):
    events,_,captured,primary,_=bridge(monkeypatch,tmp_path,fault='pack')
    caught=None
    try:dflash._cpu_nvfp4_head(tmp_path,'cuda',(128,64),softcap=30.)
    except RuntimeError as error:caught=error
    assert caught is primary and events.count('sync')==1 and events.count('pack')==1
    frames=[];tb=caught.__traceback__
    while tb is not None:frames.append(tb.tb_frame);tb=tb.tb_next
    owner=next(frame for frame in frames if frame.f_code.co_name=='_cpu_nvfp4_head')
    assert owner.f_locals['gpu_codes'] is captured['codes'] and owner.f_locals['gpu_scales'] is captured['scales']


@pytest.mark.parametrize('value',[None,1,'true'])
def test_strict_flag_precedes_checkpoint_allocation(tmp_path,value):
    with pytest.raises(ValueError,match='bool'):Gemma4Engine(tmp_path,dflash_head_w4a16=value)
    with pytest.raises(ValueError,match='bool'):capacity.admit(tmp_path,128,True,dflash_head_w4a16=value)


@pytest.mark.parametrize('options',[{}, {'drafter':'missing','drafter_bits':16},
    {'drafter':'missing','drafter_bits':8,'dflash_graph':True},
    {'drafter':'missing','drafter_bits':8,'dflash_head_native_a8':True}])
def test_unsupported_optin_rejected_before_model_io(tmp_path,options):
    with pytest.raises(ValueError,match='W4A16'):Gemma4Engine(tmp_path,dflash_head_w4a16=True,**options)


def test_staged_owner_accounting_and_no_default_metadata_change():
    work=capacity.dflash_head_w4a16_memory(2816,262144,4)
    mib=1024**2
    assert work['resident_bytes']==396*mib
    assert work['cpu_quantization_peak_bytes']==2013*mib+4
    assert work['cpu_upload_bytes']==396*mib+4
    assert work['gpu_packing_peak_bytes']==1111*mib
    assert work['device_extra_bytes']==715*mib
    assert work['cpu_read_peak_bytes']==2816*mib+3*4096
    assert work['host_loading_peak_bytes']==work['cpu_read_peak_bytes']
    assert work['unified_extra_bytes']==2420*mib+3*4096
    # Three dummy n*k/32 terms fit within the existing two block-scale n*k/16 terms.
    assert 3*(262144*2816//32)<=2*(262144*2816//16)
    assert work['logits_softcap_bytes']==12*mib
    capacity.check_dflash_head_w4a16(False,16,None)
    capacity.check_dflash_head_w4a16(True,8,'dflash')
    with pytest.raises(ValueError):capacity.check_dflash_head_w4a16(True,8,'sparkulator')


def test_initial_fence_failure_precedes_cpu_reader_and_has_no_retry(monkeypatch,tmp_path):
    events,_,_,primary,_=bridge(monkeypatch,tmp_path,fault='initial-fence')
    caught=None
    try:dflash._cpu_nvfp4_head(tmp_path,'cuda',(128,64),softcap=30.)
    except RuntimeError as error:caught=error
    assert caught is primary and events==['sync']


@pytest.mark.parametrize('dtype,shape',[(torch.float32,(128,64)),(torch.bfloat16,(64,64))])
def test_original_header_mismatch_is_rejected_before_quantization(monkeypatch,tmp_path,dtype,shape):
    (tmp_path/'model.safetensors').touch();events=[]
    class Reader:
        def __init__(self,files):events.append('read')
        def get(self,name):return torch.zeros(shape,dtype=dtype)
        def close(self):events.append('close')
    monkeypatch.setattr(dflash,'SafeTensors',Reader)
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:events.append('fence'))
    monkeypatch.setattr(requant,'nvfp4_rows',lambda value:pytest.fail('invalid source quantized'))
    with pytest.raises(ValueError,match='original BF16 CPU'):dflash._cpu_nvfp4_head(tmp_path,'cuda',(128,64),softcap=30.)
    assert events==['fence','read','close']


@pytest.mark.parametrize('index',[{'weight_map':{}},{'weight_map':{'other':'a.safetensors'}},
    {'weight_map':{'model.language_model.embed_tokens.weight':7}},
    {'weight_map':{'model.language_model.embed_tokens.weight':'../outside.safetensors'}}])
def test_malformed_authoritative_index_never_opens_tensor_reader(monkeypatch,tmp_path,index):
    (tmp_path/'model.safetensors.index.json').write_text(json.dumps(index))
    monkeypatch.setattr(torch.cuda,'synchronize',lambda device:None)
    monkeypatch.setattr(dflash,'SafeTensors',lambda files:pytest.fail('invalid index opened'))
    with pytest.raises(ValueError):dflash._cpu_nvfp4_head(tmp_path,'cuda',(128,64),softcap=30.)


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_engine_forward_expressions_preserve_false_kwargs_and_source(enabled,tmp_path):
    import ast
    import inspect
    import textwrap
    tree=ast.parse(textwrap.dedent(inspect.getsource(Gemma4Engine.__init__)))
    calls={}
    for node in ast.walk(tree):
        if not isinstance(node,ast.Call) or not isinstance(node.func,ast.Name) or node.func.id not in ('admit','DFlash'):continue
        keyword=next(k for k in node.keywords if k.arg is None and isinstance(k.value,ast.IfExp)
            and isinstance(k.value.test,ast.Name) and k.value.test.id=='dflash_head_w4a16')
        calls[node.func.id]=eval(compile(ast.Expression(body=keyword.value),'actual-forward','eval'),
            {'dflash_head_w4a16':enabled,'model_dir':tmp_path})
    assert calls==({'admit':{'dflash_head_w4a16':True},'DFlash':{'dflash_head_w4a16':True,'head_source_dir':tmp_path}}
                  if enabled else {'admit':{},'DFlash':{}})


@pytest.mark.parametrize('enabled',[False,True])
def test_actual_dflash_constructor_selects_only_head_and_prepares_existing_providers(monkeypatch,tmp_path,enabled):
    from tests.test_gemma_cuda_dflash import raw
    from tensorfold.families.gemma4.dflash_metadata import DraftConfig
    cfg=DraftConfig.from_dict(raw());arrays={name:torch.ones(shape,dtype=torch.bfloat16) for name,shape in cfg.shapes().items()}
    target=SimpleNamespace(device=torch.device('cpu'),tap_layers=cfg.tap_layers,
        config=SimpleNamespace(hidden=cfg.hidden,layers=cfg.target_layers,vocab=cfg.vocab,softcap=30.),
        weights=SimpleNamespace(embed=SimpleNamespace(weight=torch.ones((cfg.vocab,cfg.hidden),dtype=torch.bfloat16))),
        raw_embed_tokens=object(),project_logits=object())
    events=[];old=object();new=object();copies=[]
    class Reader:
        def __init__(self,files):pass
        def get(self,name):return arrays[name]
        def close(self):events.append('draft-close')
    monkeypatch.setattr(dflash,'checked_headers',lambda directory:(cfg,{}))
    monkeypatch.setattr(dflash,'checkpoint_path',lambda directory,name:directory/name)
    monkeypatch.setattr(dflash,'SafeTensors',Reader)
    def copy_fp8(weight,kind):assert kind=='fp8';copies.append(weight);return object()
    monkeypatch.setattr(dflash,'requantize',copy_fp8);monkeypatch.setattr(dflash,'prefix',lambda value,rows:value)
    monkeypatch.setattr(dflash,'Head',lambda *args,**kwargs:old)
    def head(source,device,shape,*,softcap):
        assert source==tmp_path and device==target.device and shape==(cfg.vocab,cfg.hidden) and softcap==30.
        events.append('w4-head');return new
    monkeypatch.setattr(dflash,'_cpu_nvfp4_head',head)
    monkeypatch.setattr(linear,'_prompt_ext',lambda:events.append('prompt-ready'))
    monkeypatch.setattr(linear,'_ext',lambda:events.append('lane-ready'))
    kwargs={'dflash_head_w4a16':True,'head_source_dir':tmp_path} if enabled else {}
    result=dflash.DFlash(tmp_path,target,context_limit=256,bits=8,block=5,rows=128,**kwargs)
    assert result.project_logits is (new if enabled else old)
    assert result._logits_graph is None and result.bits==8 and len(copies)==len(cfg.fused_shapes())
    assert events==['draft-close',*(['w4-head'] if enabled else []),'prompt-ready','lane-ready']


@pytest.mark.skipif(not torch.cuda.is_available(),reason='native W4A16 product needs CUDA')
@pytest.mark.parametrize('rows',[1,2,3,4])
def test_actual_k2816_native_fp32_head_single_cap_and_row_invariance(rows):
    # Independent decoded-product coverage is retained in tests/cuda/test_requant_cuda.py;
    # this supplemental case targets the real K and owning F32/cap adapter boundary.
    from tests.test_requant import _nvfp4_values
    from tests.cuda.test_requant_cuda import _assert_product_envelope, _assert_softcap_envelope
    original=torch.randn((128,2816),generator=torch.Generator().manual_seed(910)).bfloat16()
    codes,scales,g=requant.nvfp4_rows(original)
    deq=_nvfp4_values(codes,scales,g).cuda()
    packed=linear.Fp4Linear.from_checkpoint(codes.cuda(),scales.cuda(),float(g),act=None)
    head=requant.Head(packed,'nvfp4',softcap=30.)
    x=torch.randn((rows,2816),generator=torch.Generator().manual_seed(911)).bfloat16().cuda()
    raw=torch.empty((rows,128),dtype=torch.float32,device='cuda');packed(x,raw)
    _assert_product_envelope(raw,x,deq)
    actual=head(x);again=head(x)
    _assert_softcap_envelope(actual,x,deq,30.)
    assert actual.dtype==torch.float32 and actual.data_ptr()!=again.data_ptr()
    assert torch.equal(actual.view(torch.int32),(torch.tanh(raw/30.)*30.).view(torch.int32))
    single=torch.cat([head(x[i:i+1]) for i in range(rows)])
    assert torch.equal(actual.view(torch.int32),single.view(torch.int32))

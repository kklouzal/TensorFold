"""Strict target-region physical MTP fc_h tiles; original bank ATen independent."""
from types import SimpleNamespace as NS
from pathlib import Path
import ast
import inspect

import pytest
import torch

from tensorfold.families.qwen4_exp.cuda import padded_linears, forward
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
from tests.cuda.test_sm70_large_dense_tiles import bank_standard_aten, target_math_policy

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7,0),
                               reason="actual SM70 MTP physical HC oracle")
PROFILE = "source-dense-owner-batch-prelude-trial"


@pytest.fixture(scope="module",autouse=True)
def actual_target_policy():
    before = target_math_policy()
    yield
    assert target_math_policy() == before


def fixture(logical, physical, dtype=torch.float16, strided=False):
    torch.manual_seed(8431+physical)
    matrix = GGUFDense(torch.randn((2560,2560),device="cuda",dtype=torch.float16)*.02)
    matrix.implementation = "native32"
    c = NS(hidden=2560,heads=3,kv_heads=1,head_dim=256,nk=2,nv=6,streams=4,low=320,
           index_heads=4,index_dim=128,ple_dim=2560,layer_types=["linear"]*36+["attention"]*12)
    w = NS(cfg=c,device=torch.device("cuda",torch.cuda.current_device()),head=NS(n=31040),
           meta={"world":8,"sm70_profile":PROFILE,"execution_provider":"sm70"},mtp=NS(fc_h=matrix))
    owner = padded_linears.create(w,logical,prefill=False,dtype=torch.float16,role="mtp")
    assert type(owner) is padded_linears.TiledLinears and owner.hc_only is (logical<=32)
    x = torch.randn((physical,2560*2 if strided else 2560),device="cuda",dtype=torch.float16)
    if strided:
        x = x[:,::2]
    backing = torch.full((logical*4,2568 if strided else 2560),-321,device="cuda",dtype=dtype)
    out = backing[:physical,4:2564] if strided else backing[:physical]
    return owner,matrix,x,out,backing,w


def same(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype
    assert torch.equal(a.contiguous().view(torch.uint8),b.contiguous().view(torch.uint8))


@pytest.mark.parametrize("logical,physical",[(32,36),(32,96),(32,128),(64,256),(72,288)])
@pytest.mark.parametrize("dtype",[torch.float16,torch.float32])
@pytest.mark.parametrize("strided",[False,True])
def test_original_reference_and_independent_bank_bits(logical,physical,dtype,strided):
    owner,matrix,x,out,backing,_w = fixture(logical,physical,dtype,strided)
    original = matrix.matmul_reference(x,f32=dtype==torch.float32)
    expected = bank_standard_aten(x,(matrix,),dtype)
    same(original,expected)
    assert owner.matmul(x,matrix,out) is out
    torch.cuda.synchronize()
    same(out,expected)
    if strided:
        assert torch.all(backing[:,:4]==-321) and torch.all(backing[:,-4:]==-321)
    assert torch.all(backing[physical:]==-321)
    assert owner.submitted_rows==physical and owner.submitted_calls==1


@pytest.mark.parametrize("logical,physical",[(32,96),(32,128),(64,256)])
def test_actual_mtp_fc_h_expression_and_existing_output_owner(logical,physical):
    owner,matrix,x,out,backing,w = fixture(logical,physical)
    n,c = physical//4,w.cfg
    b = NS(padded_linears=owner,mtp_hs=backing)
    expected = bank_standard_aten(x,(matrix,),torch.float16)
    from tensorfold.families.qwen4_exp.cuda import mtp
    source = Path(inspect.getfile(mtp))
    expression = next(v for v in ast.walk(ast.parse(source.read_text())) if isinstance(v,ast.Expr)
        and isinstance(v.value,ast.Call) and isinstance(v.value.func,ast.Name) and v.value.func.id=="_mm"
        and len(v.value.args)>1 and isinstance(v.value.args[1],ast.Attribute) and v.value.args[1].attr=="fc_h")
    exec(compile(ast.Module(body=[expression],type_ignores=[]),str(source),"exec"),
         {"_mm":forward._mm,"hn":x.view(n,c.streams*c.hidden),
          "xh":torch.zeros((n,c.streams*c.hidden//32),device="cuda"),"n":n,"c":c,"m":w.mtp,"b":b})
    torch.cuda.synchronize()
    same(out,expected)
    assert torch.all(backing[physical:]==-321)


@pytest.mark.parametrize("logical,physical",[(32,128),(64,256)])
def test_same_stream_capture_replay_retains_all_fixed_scratch_owners(logical,physical):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        owner,matrix,x,out,_backing,_w = fixture(logical,physical,strided=True)
        assert owner.matmul(x,matrix,out) is out
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            assert owner.matmul(x,matrix,out) is out
        x.mul_(.75)
        graph.replay()
    stream.synchronize()
    expected = bank_standard_aten(x,(matrix,),out.dtype)
    same(out,expected)
    del graph


@pytest.mark.parametrize("bad",["other-weight","input-output-alias","weight-output-alias","partial-group","beyond-span"])
def test_invalid_physical_binding_never_changes_output_or_weight(bad):
    owner,matrix,x,out,_backing,_w = fixture(32,96)
    selected = matrix
    if bad=="other-weight":
        selected = GGUFDense(matrix.weight)
        selected.implementation = "native32"
    elif bad=="input-output-alias":
        out = x
    elif bad=="weight-output-alias":
        out = matrix.weight[:96]
    elif bad in ("partial-group","beyond-span"):
        rows = 97 if bad=="partial-group" else 132
        x = torch.ones((rows,2560),device="cuda",dtype=torch.float16)
        out = torch.full_like(x,-19)
    before = out.clone()
    assert owner.matmul(x,selected,out) is None
    torch.cuda.synchronize()
    same(out,before)
    assert owner.submitted_calls==0


def test_foreign_stream_retains_original_route_without_scratch_mutation():
    owner,matrix,x,out,_backing,_w = fixture(32,96)
    before = out.clone()
    with torch.cuda.stream(torch.cuda.Stream()):
        assert owner.matmul(x,matrix,out) is None
    torch.cuda.synchronize()
    same(out,before)
    assert owner.submitted_calls==0


@pytest.mark.parametrize("logical,physical",[(32,96),(64,256)])
def test_abort_after_first_tile_preserves_prefix_and_live_scratch(monkeypatch,logical,physical):
    owner,matrix,x,out,_backing,_w = fixture(logical,physical,strided=True)
    expected = bank_standard_aten(x,(matrix,),out.dtype)
    original = matrix.matmul_prepared
    primary = RuntimeError("abort physical HC after first completed tile")
    calls = []
    def interrupted(source,target):
        calls.append(1)
        if len(calls)==2:
            raise primary
        return original(source,target)
    monkeypatch.setattr(matrix,"matmul_prepared",interrupted)
    with pytest.raises(RuntimeError) as caught:
        owner.matmul(x,matrix,out)
    assert caught.value is primary and len(calls)==2
    torch.cuda.synchronize()
    same(out[:32],expected[:32])
    assert torch.all(out[32:]==-321)
    assert len(owner.device_tensors())>=3

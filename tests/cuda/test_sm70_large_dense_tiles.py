"""Independent fixed-M32 ATen oracles for prepaid large-row Dense tiles."""
from types import SimpleNamespace as NS
import pytest
import torch
from tensorfold.families.qwen4_exp.cuda import padded_linears
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7,0),
                               reason="actual SM70 large-row Dense oracle")


def target_math_policy():
    """Observe the actual selected profile policy; never rewrite oracle flags."""
    policy = {"allow_fp16_reduced_precision_reduction": torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction,
              "allow_fp16_accumulation": torch.backends.cuda.matmul.allow_fp16_accumulation}
    if (policy["allow_fp16_reduced_precision_reduction"] is not True
            or policy["allow_fp16_accumulation"] is not False):
        raise RuntimeError("Dense oracle requires target FP16 reduction=True/accumulation=False")
    return policy


@pytest.fixture(scope="module",autouse=True)
def actual_target_math_region():
    before = target_math_policy()
    yield
    assert target_math_policy() == before


def workspace(rows):
    c=NS(hidden=2560,heads=3,kv_heads=1,head_dim=256,nk=2,nv=6,streams=4,low=320,
         index_heads=4,index_dim=128,ple_dim=2560,layer_types=["linear"]*36+["attention"]*12)
    w=NS(cfg=c,device=torch.device("cuda",torch.cuda.current_device()),head=NS(n=31040),meta={"world":8})
    return padded_linears.TiledLinears(w,rows)


def bank_standard_aten(x,parts,dtype):
    """No native helper, private workspace, or matrix wrapper participates."""
    banks=[]
    for part in parts:
        chunks=[]
        for start in range(0,x.shape[0],32):
            count=min(32,x.shape[0]-start)
            independent=torch.zeros((32,x.shape[1]),dtype=torch.float16,device=x.device)
            independent[:count].copy_(x[start:start+count])
            result=(torch.mm(independent,part.weight.T,out_dtype=torch.float32)if dtype==torch.float32
                    else torch.mm(independent,part.weight.T))
            chunks.append(result[:count].clone())
        banks.append(torch.cat(chunks,dim=0))
    return torch.cat(banks,dim=1)


def exercise(rows,dtype,widths,k,*,strided):
    torch.manual_seed(727+rows)
    parts=[GGUFDense(torch.randn((n,k),device="cuda",dtype=torch.float16)*.02)for n in widths]
    matrix=GGUFConcat(parts)if len(parts)>1 else parts[0]
    matrix.implementation="native32"
    x=torch.randn((rows,k*2 if strided else k),device="cuda",dtype=torch.float16)
    if strided:
        x=x[:,::2]
    columns=matrix.n+10 if strided else matrix.n
    backing=torch.full((rows,columns),-321,dtype=dtype,device="cuda")
    out=backing[:,5:5+matrix.n]if strided else backing
    expected=bank_standard_aten(x,parts,dtype)
    owner=workspace(rows)
    assert owner.matmul(x,matrix,out)is out
    torch.cuda.synchronize()
    torch.testing.assert_close(out,expected,rtol=0,atol=0)
    if strided:
        assert torch.all(backing[:,:5]==-321)and torch.all(backing[:,-5:]==-321)
    return owner,matrix,x,out


@pytest.mark.parametrize("rows",[48,64,72])
@pytest.mark.parametrize("dtype",[torch.float16,torch.float32])
@pytest.mark.parametrize("concat",[False,True])
@pytest.mark.parametrize("strided",[False,True])
def test_large_tiles_preserve_bank_standard_aten_bits_and_neighbors(rows,dtype,concat,strided):
    exercise(rows,dtype,(80,48)if concat else(80,),256,strided=strided)


@pytest.mark.parametrize("rows",[48,64,72])
@pytest.mark.parametrize("shape",["hc-f32","qkv-half"])
def test_actual_tp8_projection_shapes_keep_original_part_and_cast_boundaries(rows,shape):
    if shape=="hc-f32":
        exercise(rows,torch.float32,(320,4),10240,strided=True)
    else:
        exercise(rows,torch.float16,(1536,256,256,512,128),2560,strided=True)


def test_large_local_head_and_same_stream_graph_replay_keep_all_chunk_owners():
    exercise(72,torch.float32,(31040,),2560,strided=False)
    stream=torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        owner,matrix,x,out=exercise(48,torch.float32,(80,48),256,strided=True)
        graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph,stream=stream):
            assert owner.matmul(x,matrix,out)is out
        x.mul_(.75)
        graph.replay()
    stream.synchronize()
    expected=bank_standard_aten(x,matrix.parts,torch.float32)
    torch.testing.assert_close(out,expected,rtol=0,atol=0)


def test_abort_between_tiles_preserves_completed_output_and_retains_live_scratch(monkeypatch):
    owner,matrix,x,out=exercise(64,torch.float32,(80,),256,strided=True)
    original=matrix.matmul_prepared
    error=RuntimeError("abort after first submitted tile")
    calls=[]
    def interrupted(source,target):
        calls.append(1)
        if len(calls)==2:
            raise error
        return original(source,target)
    monkeypatch.setattr(matrix,"matmul_prepared",interrupted)
    out.fill_(-71)
    with pytest.raises(RuntimeError)as failure:
        owner.matmul(x,matrix,out)
    assert failure.value is error and len(calls)==2
    torch.cuda.synchronize()
    expected=bank_standard_aten(x[:32],(matrix,),torch.float32)
    torch.testing.assert_close(out[:32],expected,rtol=0,atol=0)
    assert torch.all(out[32:]==-71)

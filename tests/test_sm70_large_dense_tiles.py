"""Actual owned tile routing/storage controls; CUDA math has a separate oracle."""
from types import SimpleNamespace as NS
import pytest
import torch
from tensorfold.cuda.sm70 import gguf_ops
from tensorfold.families.qwen4_exp.cuda import padded_linears as prepared
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFConcat
from tests.test_sm70_padded_linears import metadata, local_config

PROFILE = "source-dense-owner-batch-prelude-trial"


def owner(rows=72):
    w=NS(cfg=local_config(8),device=torch.device("cpu"),head=NS(n=31040),meta={"world":8})
    return prepared.TiledLinears(w,rows)


@pytest.mark.parametrize("rows",[48,64,72])
def test_actual_workspace_storage_matches_admission_with_alignment_slack(rows):
    w=owner(rows)
    actual=sum(t.numel()*t.element_size() for t in w.device_tensors())
    expected=prepared.workspace_bytes_from_text(metadata(),rows,8,large_profile=PROFILE)
    assert actual==expected
    for value in (w.input_storage,w.half_storage,w.float_storage):
        assert value.data_ptr()%512==0 and value.is_contiguous()
    assert prepared.workspace_bytes_from_text(metadata(),32,8,large_profile=PROFILE)==0
    assert prepared.workspace_bytes_from_text(metadata(),rows,8,large_profile="reference")==0
    assert prepared.workspace_bytes_from_text(metadata(),rows,8,large_profile=PROFILE,prefill=True)==0


@pytest.mark.parametrize("rows",[48,64,72])
@pytest.mark.parametrize("dtype",[torch.float16,torch.float32])
@pytest.mark.parametrize("concat",[False,True])
def test_real_tiler_owns_each_output_and_reuses_only_completed_tile_storage(monkeypatch,rows,dtype,concat):
    workspace=owner()
    monkeypatch.setattr(torch.Tensor,"is_cuda",property(lambda self:True))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _device:None)
    gen=torch.Generator().manual_seed(710)
    parts=[GGUFDense(torch.randint(-3,4,(n,8),generator=gen).half()) for n in ((5,7)if concat else(12,))]
    matrix=GGUFConcat(parts)if concat else parts[0]
    matrix.implementation="native32"
    x=torch.randint(-3,4,(rows,16),generator=gen).half()[:,::2]
    backing=torch.full((rows,matrix.n+4),-19,dtype=dtype)
    out=backing[:,2:2+matrix.n]
    expected=torch.cat([(x.float()@p.weight.float().T).to(dtype)for p in parts],dim=1)
    calls=[]
    def native(source,weight,target):
        assert source.shape==(32,8) and source.is_contiguous() and source.data_ptr()%512==0
        assert target.shape==(32,weight.shape[0]) and target.is_contiguous() and target.data_ptr()%512==0
        calls.append((source.clone(),weight,target.data_ptr()))
        # Control only: this is CPU arithmetic, not a CUDA correctness claim.
        target.numpy()[:]=(source.float()@weight.float().T).to(target.dtype).numpy()
        return target
    monkeypatch.setattr(gguf_ops,"dense32",native)
    monkeypatch.setattr(torch,"empty",lambda *a,**k:pytest.fail("new hot empty storage"))
    monkeypatch.setattr(torch,"zeros",lambda *a,**k:pytest.fail("new hot zero storage"))
    version=out._version
    assert workspace.matmul(x,matrix,out) is out
    assert out._version==version+len(parts)*((rows+31)//32)
    assert workspace.submitted_calls==1 and workspace.submitted_rows==rows
    assert torch.equal(out,expected)
    assert torch.all(backing[:,:2]==-19) and torch.all(backing[:,-2:]==-19)
    # Recorded inputs are independent clones; earlier final outputs are never
    # views of the shared scratch overwritten by a subsequent tile.
    assert len(calls)==len(parts)*((rows+31)//32)
    if rows%32:
        assert torch.count_nonzero(calls[(rows+31)//32-1][0][rows%32:])==0


def test_small_reference_autograd_alias_and_foreign_stream_are_not_adopted(monkeypatch):
    workspace=owner()
    monkeypatch.setattr(torch.Tensor,"is_cuda",property(lambda self:True))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _device:None)
    matrix=GGUFDense(torch.ones((8,8),dtype=torch.float16))
    matrix.implementation="native32"
    x=torch.ones((64,8),dtype=torch.float16)
    out=torch.empty_like(x)
    monkeypatch.setattr(gguf_ops,"dense32",lambda *a:pytest.fail("ineligible native call"))
    assert workspace.matmul(x[:32],matrix,out[:32]) is None
    assert workspace.matmul(x,matrix,x) is None
    x.requires_grad_()
    assert workspace.matmul(x,matrix,out) is None
    x.requires_grad_(False)
    matrix.implementation="reference"
    assert workspace.matmul(x,matrix,out) is None
    matrix.implementation="native32"
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _device:object())
    assert workspace.matmul(x,matrix,out) is None


def test_native_failure_is_not_retried_or_concealed_after_partial_output(monkeypatch):
    workspace=owner()
    monkeypatch.setattr(torch.Tensor,"is_cuda",property(lambda self:True))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _device:None)
    matrix=GGUFDense(torch.ones((8,8),dtype=torch.float16))
    matrix.implementation="native32"
    error=RuntimeError("second tile failed")
    calls=[]
    def native(source,weight,target):
        calls.append(1)
        if len(calls)==2:
            raise error
        target.fill_(7)
    monkeypatch.setattr(gguf_ops,"dense32",native)
    out=torch.full((64,8),-1,dtype=torch.float16)
    with pytest.raises(RuntimeError)as failure:
        workspace.matmul(torch.ones((64,8),dtype=torch.float16),matrix,out)
    assert failure.value is error and len(calls)==2
    assert torch.all(out[:32]==7) and torch.all(out[32:]==-1)


def test_large_tile_host_headers_are_global_prepaid_before_host_capacity_solver():
    import ast
    from pathlib import Path
    assert prepared.host_workspace_bytes(64,PROFILE)==65536
    assert prepared.host_workspace_bytes(72,PROFILE)==65536
    assert prepared.host_workspace_bytes(32,PROFILE)==0
    assert prepared.host_workspace_bytes(72,"reference")==0
    source=Path(__file__).parents[1]/"src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree=ast.parse(source.read_text())
    call=next(n for n in ast.walk(tree)if isinstance(n,ast.Call)and isinstance(n.func,ast.Attribute)
              and n.func.attr=="host_workspace_bytes")
    # The real initializer adds the term to host_capacity before the original
    # startup plan/physical capacity commitment and Buffer materialization.
    text=source.read_text()
    assert call.lineno < next(n.lineno for n in ast.walk(tree)if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute)and t.attr=="_startup_content_plan"for t in n.targets)
        and isinstance(n.value,ast.Call))
    assert 'tp * (1 + int(mtp)) * padded_linears.host_workspace_bytes' in text
    assert 'host_capacity["host_workspace_bytes"] = host_capacity.get("host_workspace_bytes", 0) + dense_tile_host' in text


def test_concat_with_unsupported_part_policy_retains_original_route(monkeypatch):
    workspace=owner()
    monkeypatch.setattr(torch.Tensor,"is_cuda",property(lambda self:True))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _device:None)
    matrix=GGUFConcat([GGUFDense(torch.ones((8,8),dtype=torch.float16))for _ in range(2)])
    matrix.implementation="native32"
    matrix.parts[1].implementation="wmma"
    monkeypatch.setattr(gguf_ops,"dense32",lambda *a:pytest.fail("unsupported part adopted"))
    assert workspace.matmul(torch.ones((64,8),dtype=torch.float16),matrix,
                            torch.empty((64,16),dtype=torch.float16))is None


def test_large_projection_counter_saturates_without_growing_or_losing_output(monkeypatch):
    workspace=owner()
    monkeypatch.setattr(torch.Tensor,"is_cuda",property(lambda self:True))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _device:None)
    matrix=GGUFDense(torch.ones((8,8),dtype=torch.float16))
    matrix.implementation="native32"
    monkeypatch.setattr(gguf_ops,"dense32",lambda x,w,out:out.fill_(3))
    workspace.submitted_calls=(1<<63)-1
    out=torch.empty((64,8),dtype=torch.float16)
    assert workspace.matmul(torch.ones_like(out),matrix,out)is out
    assert workspace.counter_overflowed and workspace.submitted_calls==(1<<63)-1
    assert torch.all(out==3)


@pytest.mark.parametrize("adaptive,explicit,expected",[(False,None,1048576),(True,None,524288),(True,64,1048576),(False,32,524288)])
def test_actual_engine_host_term_covers_fixed72_and_adaptive_default32(adaptive,explicit,expected):
    import ast
    from pathlib import Path
    source=Path(__file__).parents[1]/"src/tensorfold/families/qwen4_exp/cuda/engine.py"
    tree=ast.parse(source.read_text())
    envelope=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=="row_envelope")
    block=next(n for n in ast.walk(tree)if isinstance(n,ast.If)and isinstance(n.test,ast.Name)
        and n.test.id=="sm70"and any(isinstance(c,ast.Call)and isinstance(c.func,ast.Attribute)
        and c.func.attr=="host_workspace_bytes"for c in ast.walk(n)))
    env={"__package__":"tensorfold.families.qwen4_exp.cuda","sm70":True,"adaptive_sessions":adaptive,
        "execution_rows":explicit,"each":3,"streams":24,"concurrent":True,"tp":8,"mtp":True,
        "session_ceiling":24 if adaptive else None,"host_capacity":{},"self":NS(sm70_profile=PROFILE),"text":metadata()}
    exec(compile(ast.Module(body=[envelope,block],type_ignores=[]),str(source),"exec"),env)
    assert env["host_capacity"]["host_workspace_bytes"]==expected


def test_actual_buffer_factory_selects_only_large_private_decode_fp16(monkeypatch):
    marker=object()
    calls=[]
    monkeypatch.setattr(prepared,"TiledLinears",lambda w,rows:calls.append(rows)or marker)
    w=NS(meta={"sm70_profile":PROFILE,"execution_provider":"sm70"},device=NS(type="cuda"))
    assert prepared.create(w,64,prefill=False,dtype=torch.float16)is marker and calls==[64]
    assert prepared.create(w,32,prefill=False,dtype=torch.float16)is None
    assert prepared.create(w,64,prefill=True,dtype=torch.float16)is None
    assert prepared.create(w,64,prefill=False,dtype=torch.bfloat16)is None
    w.meta["sm70_profile"]="reference"
    assert prepared.create(w,64,prefill=False,dtype=torch.float16)is None

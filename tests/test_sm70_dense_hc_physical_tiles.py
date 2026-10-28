"""Actual MTP fc_h expression/identity bounds; no hardware math claim."""
from types import SimpleNamespace as NS
from pathlib import Path
import ast
import pytest
import torch
from tensorfold.cuda.sm70 import gguf_ops
from tensorfold.families.qwen4_exp.cuda import padded_linears as prepared, forward
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
from tests.test_sm70_padded_linears import local_config, metadata


def context(matrix, logical=64):
    cfg=local_config(8)
    w=NS(cfg=cfg,device=torch.device("cpu"),head=NS(n=31040),meta={"world":8},mtp=NS(fc_h=matrix))
    return prepared.TiledLinears(w,logical),cfg


@pytest.mark.parametrize("physical",[96,128,256])
def test_actual_fc_h_callsite_uses_whole_hc_groups_and_existing_caller_output(monkeypatch,physical):
    matrix=GGUFDense(torch.ones((2560,2560),dtype=torch.float16))
    matrix.implementation="native32"
    owner,cfg=context(matrix)
    n=physical//cfg.streams
    hn=torch.ones((n,cfg.hidden*cfg.streams),dtype=torch.float16)
    xh=torch.ones((n,cfg.hidden*cfg.streams//32),dtype=torch.float32)
    b=NS(padded_linears=owner,mtp_hs=torch.full((64*cfg.streams,cfg.hidden),-7,dtype=torch.float16))
    monkeypatch.setattr(torch.Tensor,"is_cuda",property(lambda self:True))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _device:None)
    calls=[]
    def native(x,w,out):
        calls.append(x.shape[0])
        out.fill_(5)
        return out
    monkeypatch.setattr(gguf_ops,"dense32",native)
    source=Path(__file__).parents[1]/"src/tensorfold/families/qwen4_exp/cuda/mtp.py"
    tree=ast.parse(source.read_text())
    expression=next(v for v in ast.walk(tree)if isinstance(v,ast.Expr)and isinstance(v.value,ast.Call)
        and isinstance(v.value.func,ast.Name)and v.value.func.id=="_mm"
        and len(v.value.args)>1 and isinstance(v.value.args[1],ast.Attribute)and v.value.args[1].attr=="fc_h")
    exec(compile(ast.Module(body=[expression],type_ignores=[]),str(source),"exec"),
         {"_mm":forward._mm,"hn":hn,"xh":xh,"n":n,"c":cfg,"m":NS(fc_h=matrix),"b":b})
    assert calls==[32]*(physical//32)
    assert torch.all(b.mtp_hs[:physical]==5) and torch.all(b.mtp_hs[physical:]==-7)
    assert owner.submitted_rows==physical


def test_expansion_is_only_named_hidden_projection_and_exact_bound():
    matrix=GGUFDense(torch.ones((2560,2560),dtype=torch.float16))
    matrix.implementation="native32"
    owner,cfg=context(matrix)
    other=GGUFDense(matrix.weight)
    other.implementation="native32"
    assert owner._hc_rows_allowed(96,matrix) and owner._hc_rows_allowed(256,matrix)
    assert not owner._hc_rows_allowed(96,other)
    assert not owner._hc_rows_allowed(97,matrix)
    assert not owner._hc_rows_allowed(260,matrix)
    matrix.k=1
    assert not owner._hc_rows_allowed(96,matrix)
    assert prepared.workspace_bytes_from_text(metadata(),64,8,large_profile="source-dense-owner-batch-prelude-trial")>0


def test_hc_extension_changes_no_storage_or_partial_hc_capacity():
    matrix=GGUFDense(torch.ones((2560,2560),dtype=torch.float16))
    owner,cfg=context(matrix)
    ordinary,_=context(None)
    assert sum(t.numel()*t.element_size()for t in owner.device_tensors())==sum(
        t.numel()*t.element_size()for t in ordinary.device_tensors())
    assert all(t.shape[0]==64 for t in owner.partial.values())
    assert owner.hc_streams==4 and cfg.hidden==2560

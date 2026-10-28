"""Physical MTP fc_h ownership/factory/span/funding, without hardware claims."""
from types import SimpleNamespace as NS
from pathlib import Path
import ast

import pytest
import torch

from tensorfold.cuda.sm70 import gguf_ops
from tensorfold.families.qwen4_exp.cuda import padded_linears as prepared, forward
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
from tests.test_sm70_padded_linears import local_config, metadata
from tests.test_flashnext_padded_admission import geometry_wrapper

PROFILE = "source-dense-owner-batch-prelude-trial"


def weights(device="cpu"):
    matrix = GGUFDense(torch.ones((2560,2560),dtype=torch.float16))
    matrix.implementation = "native32"
    return NS(cfg=local_config(8),device=torch.device(device),head=NS(n=31040),
              meta={"world":8,"sm70_profile":PROFILE,"execution_provider":"sm70"},mtp=NS(fc_h=matrix))


def test_minimal_hc_owner_exact_device_bytes_and_no_logical_partial_expansion():
    w = weights()
    owner = prepared.TiledLinears(w,32,hc_only=True)
    assert sum(t.numel()*t.element_size() for t in owner.device_tensors()) == 656896
    assert owner.maximum_input == owner.maximum_output == 2560 and owner.partial == {}
    assert prepared.mtp_hc_workspace_bytes(metadata(),32,PROFILE) == 656896
    assert prepared.mtp_hc_host_bytes(metadata(),32,PROFILE) == 65536
    for rows in (1,8,64,72):
        assert prepared.mtp_hc_workspace_bytes(metadata(),rows,PROFILE) == 0
    assert prepared.mtp_hc_workspace_bytes(metadata(),32,"reference") == 0


def test_actual_factory_selects_only_private_mtp_role_and_complete_hc_groups(monkeypatch):
    w = weights()
    w.device = NS(type="cuda")
    calls = []
    marker = object()
    monkeypatch.setattr(prepared,"TiledLinears",lambda _w,rows,**kw:calls.append((rows,kw.get("hc_only",False))) or marker)
    assert prepared.create(w,32,prefill=False,dtype=torch.float16,role="mtp") is marker
    assert calls == [(32,True)]
    for role in (None,"decode","prefill"):
        assert prepared.create(w,32,prefill=False,dtype=torch.float16,role=role) is None
    assert prepared.create(w,8,prefill=False,dtype=torch.float16,role="mtp") is None
    assert prepared.create(w,32,prefill=True,dtype=torch.float16,role="mtp") is None
    assert prepared.create(w,32,prefill=False,dtype=torch.bfloat16,role="mtp") is None
    w.meta["sm70_profile"] = "reference"
    assert prepared.create(w,32,prefill=False,dtype=torch.float16,role="mtp") is None


def test_actual_buffers_factory_passes_mtp_role_without_changing_caller_output_allocation(monkeypatch):
    source = Path(__file__).parents[1]/"src/tensorfold/families/qwen4_exp/cuda/state.py"
    tree = ast.parse(source.read_text())
    assign = next(n for n in ast.walk(tree) if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=="padded_linears" for t in n.targets))
    seen = []
    marker = object()
    def create(*_args,**kwargs):
        seen.append(kwargs)
        return marker
    owner = NS()
    exec(compile(ast.Module(body=[assign],type_ignores=[]),str(source),"exec"),
         {"self":owner,"padded_linears":create,"w":weights(),"rows":32,"prefill":False,
          "bf":torch.float16,"owner_role":"mtp"})
    assert owner.padded_linears is marker and seen[0]["role"] == "mtp"


@pytest.mark.parametrize("logical,physical",[(32,96),(32,128),(64,256)])
def test_actual_mtp_call_and_owned_output_span_cover_all_physical_rows(monkeypatch,logical,physical):
    w = weights()
    owner = prepared.TiledLinears(w,logical,hc_only=logical<=32)
    cfg, matrix = w.cfg, w.mtp.fc_h
    n = physical//cfg.streams
    state_source = Path(__file__).parents[1]/"src/tensorfold/families/qwen4_exp/cuda/state.py"
    allocation = next(node for node in ast.walk(ast.parse(state_source.read_text())) if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Attribute) and t.attr=="mtp_hs" for t in node.targets))
    b = NS(padded_linears=owner)
    exec(compile(ast.Module(body=[allocation],type_ignores=[]),str(state_source),"exec"),
         {"self":b,"torch":torch,"rows":logical,"c":cfg,"bf":torch.float16,"dev":"cpu"})
    assert b.mtp_hs.shape == (logical*4,2560) and b.mtp_hs.numel()*2 == logical*4*2560*2
    b.mtp_hs.fill_(-7)
    hn, xh = torch.ones((n,cfg.hidden*4),dtype=torch.float16), torch.zeros((n,cfg.hidden*4//32))
    monkeypatch.setattr(torch.Tensor,"is_cuda",property(lambda _self:True))
    monkeypatch.setattr(torch.cuda,"current_stream",lambda _device:None)
    calls = []
    def native(x,_weight,out):
        calls.append(x.shape[0])
        return out.fill_(5)
    monkeypatch.setattr(gguf_ops,"dense32",native)
    source = Path(__file__).parents[1]/"src/tensorfold/families/qwen4_exp/cuda/mtp.py"
    expression = next(v for v in ast.walk(ast.parse(source.read_text())) if isinstance(v,ast.Expr)
        and isinstance(v.value,ast.Call) and isinstance(v.value.func,ast.Name) and v.value.func.id=="_mm"
        and len(v.value.args)>1 and isinstance(v.value.args[1],ast.Attribute) and v.value.args[1].attr=="fc_h")
    exec(compile(ast.Module(body=[expression],type_ignores=[]),str(source),"exec"),
         {"_mm":forward._mm,"hn":hn,"xh":xh,"n":n,"c":cfg,"m":w.mtp,"b":b})
    assert calls == [32]*(physical//32)
    assert torch.all(b.mtp_hs[:physical]==5) and torch.all(b.mtp_hs[physical:]==-7)
    other = GGUFDense(matrix.weight)
    other.implementation = "native32"
    assert not owner._hc_rows_allowed(physical,other)


def test_actual_geometry_adds_one_mtp_only_term_before_capacity_admission(monkeypatch):
    original = prepared.workspace_bytes_from_text
    wrapper, base, _calls, _observed = geometry_wrapper(monkeypatch,24,3,True,False,rows=32,profile=PROFILE)
    monkeypatch.setattr(prepared,"workspace_bytes_from_text",original)
    result = wrapper(metadata())
    assert result.bytes_at(262144) == base.bytes_at(262144)+656896
    assert result.reserve == base.reserve and result.minimum_slots == base.minimum_slots

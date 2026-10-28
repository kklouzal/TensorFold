"""Affected current-profile MTP HC factory/funding and unchanged native-loop proof."""
from types import SimpleNamespace as NS
from pathlib import Path
import ast
import hashlib
import inspect
import textwrap

import pytest
import torch
from tensorfold.cuda.capacity import Geometry
from tensorfold.cuda.sm70 import family_ops
from tensorfold.families.qwen4_exp.cuda import padded_linears as p
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense
from tools import v100_backend_probe as probe


def weights():
    matrix = GGUFDense(torch.ones((2560,2560),dtype=torch.float16))
    matrix.implementation = "native32"
    return NS(cfg=NS(hidden=2560,streams=4),mtp=NS(fc_h=matrix),device=torch.device("cpu"),
              meta={"sm70_profile":p.HC_PROFILE,"execution_provider":"sm70"})


def test_exact_scratch_and_actual_owner_receipt():
    w = weights()
    owner = p.MTPHCTiles(w,32)
    assert sum(t.numel()*t.element_size()for t in owner.device_tensors()) == 656896
    engine = NS(multi=NS(mbuf=NS(padded_linears=owner,mtp_hs=torch.empty((128,2560)))),w=w,
        execution_rows=32,capacity_plan={"host_workspace_bytes_estimate":524288})
    assert probe.check_mtp_hc_engine(engine)["device_bytes_per_rank"] == 656896
    assert p.mtp_hc_workspace_bytes({"hidden_size":2560,"hc_count":4},32,p.HC_PROFILE)==656896
    assert p.mtp_hc_host_bytes({"hidden_size":2560,"hc_count":4},32,p.HC_PROFILE)==65536
    assert p.mtp_hc_workspace_bytes({},32,"reference")==0
    assert p.mtp_hc_workspace_bytes({},32,"source-dense-owner-batch-prelude-trial")==0


def test_only_current_router_mtp_factory_eligibility(monkeypatch):
    w=weights()
    w.device=NS(type="cuda")
    calls=[]
    marker=object()
    monkeypatch.setattr(p,"MTPHCTiles",lambda _w,rows:calls.append(rows)or marker)
    assert p.create(w,32,prefill=False,dtype=torch.float16,role="mtp") is marker
    assert calls==[32]
    assert p.create(w,32,prefill=False,dtype=torch.float16) is None
    assert p.create(w,32,prefill=True,dtype=torch.float16,role="mtp") is None
    assert p.create(w,8,prefill=False,dtype=torch.float16,role="mtp") is None
    assert p.create(w,32,prefill=False,dtype=torch.bfloat16,role="mtp") is None


def test_qualified_validation_and_matmul_body_is_unchanged():
    tree=ast.parse(inspect.getsource(p.MTPHCTiles))
    f=next(n for n in tree.body[0].body if isinstance(n,ast.FunctionDef)and n.name=='matmul')
    start=next(i for i,n in enumerate(f.body)if isinstance(n,ast.Assign)and any(isinstance(t,ast.Name)and t.id=='parts'for t in n.targets))
    lines=inspect.getsource(p.MTPHCTiles).splitlines()
    raw=textwrap.dedent('\n'.join(lines[f.body[start].lineno-1:f.end_lineno])).encode()
    assert hashlib.sha256(raw).hexdigest()=='572b3df1bec8235c17a775dec47d700b0c0add3753ef41b07a4b036ba436346f'


def test_actual_geometry_adds_only_mtp_device_bytes(monkeypatch):
    source=Path(__file__).parents[1]/'src/tensorfold/families/qwen4_exp/cuda/engine.py'
    tree=ast.parse(source.read_text())
    node=next(n for n in ast.walk(tree)if isinstance(n,ast.FunctionDef)and n.name=='sm70_geometry')
    monkeypatch.setattr(family_ops,'_NATIVE_FAMILY',False)
    monkeypatch.setattr(family_ops,'_NATIVE_ATTENTION',False)
    base=Geometry(lambda cap:cap*10,123,7)
    env={'__package__':'tensorfold.families.qwen4_exp.cuda','target_geometry':lambda *_:base,
        'tp':8,'streams':24,'each':3,'mtp':True,'rows0':256,'head_dim':256,
        'self':NS(sm70_profile=p.HC_PROFILE),'concurrent':True,'row_envelope':lambda _count:32,
        'history_placement':None,'owner_preparation':None}
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),env)
    result=env['sm70_geometry']({'num_attention_heads':24,'num_key_value_heads':2,'hidden_size':2560,'hc_count':4})
    assert result.bytes_at(262144)==base.bytes_at(262144)+656896
    env['mtp']=False
    assert env['sm70_geometry']({'num_attention_heads':24,'num_key_value_heads':2}).bytes_at(262144)==base.bytes_at(262144)


@pytest.mark.parametrize('count,delta,passed',[(1,0,True),(1,1,False),(8,0,True),(16,1,True),(24,0,False)])
def test_hc_dispatch_evidence_uses_actual_large_cohort_submissions(count,delta,passed):
    before={'counters':{'mtp.hc_submitted_calls':10}}
    after={'counters':{'mtp.hc_submitted_calls':10+delta}}
    assert probe.mtp_hc_evidence(count,before,after)['passed'] is passed


def test_actual_host_funding_branch_adds_one_mtp_term():
    source=Path(__file__).parents[1]/'src/tensorfold/families/qwen4_exp/cuda/engine.py'
    tree=ast.parse(source.read_text())
    block=next(n for n in ast.walk(tree)if isinstance(n,ast.If)and isinstance(n.test,ast.BoolOp)
        and any(isinstance(c,ast.Call)and isinstance(c.func,ast.Attribute)and c.func.attr=='mtp_hc_host_bytes'for c in ast.walk(n)))
    env={'__package__':'tensorfold.families.qwen4_exp.cuda','sm70':True,'mtp':True,
         'adaptive_sessions':True,'execution_rows':None,'session_ceiling':24,'each':3,'streams':24,
         'concurrent':True,'row_envelope':lambda _count:32,'tp':8,
         'self':NS(sm70_profile=p.HC_PROFILE),'text':{'hidden_size':2560,'hc_count':4},
         'host_capacity':{'host_workspace_bytes':123}}
    exec(compile(ast.Module(body=[block],type_ignores=[]),str(source),'exec'),env)
    assert env['host_capacity']['host_workspace_bytes']==123+524288

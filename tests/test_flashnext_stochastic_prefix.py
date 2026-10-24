"""Stochastic cache refusal changes borrowing, not drafting or validation precedence."""

import ast
import inspect
import json
from pathlib import Path
import time
from types import SimpleNamespace as NS

import pytest
import torch

from tensorfold.engine.exact_sampling import Sampling
from tensorfold.families.qwen4_exp.cuda import prefix_policy
from tensorfold.families.qwen4_exp.cuda.engine import FlashNextEngine
from tensorfold.families.qwen4_exp.cuda.multi import MultiDecoder


def weights(provider="sm70", profile="optimized"):
    return NS(meta={"execution_provider":provider,"sm70_profile":profile}, dspark=object())


@pytest.mark.parametrize("sampling,allowed", [(None,True),(Sampling(19,0),True),
    (Sampling(19,-1),True),(Sampling(19,.7,40,.9),False),(Sampling(47,1,0,.95),False)])
def test_only_valid_positive_temperature_declines_cached_suffix(sampling,allowed):
    assert prefix_policy.reuse_allowed(weights(),sampling,draft=True,vision=None,position=79) is allowed


@pytest.mark.parametrize("provider,profile,draft,vision,allowed", [
    ("sm70","reference",True,None,True),("cuda","optimized",True,None,True),
    ("sm70","optimized",False,None,False),("sm70","optimized",True,object(),False)])
def test_other_profiles_and_original_nonreuse_conditions_stay_unchanged(provider,profile,draft,vision,allowed):
    assert prefix_policy.reuse_allowed(weights(provider,profile),Sampling(19,.7),
        draft=draft,vision=vision,position=79) is allowed


@pytest.mark.parametrize("sampling", [Sampling("bad",.7),Sampling(19,float("nan")),
    Sampling(19,float("inf")),Sampling(19,.7,40,1.1),Sampling(19,.7,40,.9,-.1)])
def test_malformed_metadata_keeps_original_error_path(sampling):
    assert prefix_policy.reuse_allowed(weights(),sampling,draft=True,vision=None,position=79)


def method(owner,name,namespace):
    path=Path(inspect.getsourcefile(owner))
    assert str(path).startswith("/opt/venv/"),path
    tree=ast.parse(path.read_text())
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name==owner.__name__)
    fn=next(node for node in cls.body if isinstance(node,ast.FunctionDef) and node.name==name)
    fn.decorator_list=[]
    namespace={"__package__":"tensorfold.families.qwen4_exp.cuda",**namespace}
    exec(compile(ast.Module(body=[fn],type_ignores=[]),str(path)+"[declared-prefix-policy]","exec"),namespace)
    return namespace[name]


@pytest.mark.parametrize("profile,sampling,reuse", [("optimized",Sampling(19,.7),False),
    ("reference",Sampling(19,.7),True),("optimized",Sampling(19,0),True),
    ("optimized",Sampling("bad",.7),True)])
def test_actual_admit_borrow_changes_only_reuse_and_keeps_drafter(monkeypatch,profile,sampling,reuse):
    admit=method(MultiDecoder,"admit",{"time":time,"Stream":NS})
    borrowed=[]

    class StopBorrow(Exception):
        pass

    def slot(prompt,allowed):
        borrowed.append((prompt,allowed))
        raise StopBorrow

    w=weights(profile=profile)
    provider=w.dspark
    decoder=NS(w=w,capacity=8192,depth=3,streams={},_slot_for=slot)
    stream=NS(prompt=[11]*79,count=128,draft=True,vision=None,sampling=sampling)
    with pytest.raises(StopBorrow):
        admit(decoder,stream)
    assert borrowed==[(stream.prompt,reuse)]
    assert stream.draft is True and w.dspark is provider and decoder.depth==3
    assert MultiDecoder._keep_at(NS(prompt=stream.prompt,draft=True,st=NS(image_positions=None)))==78


def test_worker_identity_changes_and_refuses_a_peer_using_old_policy(monkeypatch):
    precision=NS(fp8=lambda:False,same_on_ranks=lambda a,b:None)
    agree=method(FlashNextEngine,"_same_settings",{"json":json,"prompt_precision":precision})
    proxy=NS(int64=torch.int64,tensor=lambda data,**kw:torch.tensor(data,dtype=kw["dtype"]),
             empty=lambda shape,**kw:torch.empty(shape,dtype=kw["dtype"]),equal=torch.equal)
    packets=[]
    old_peer=None

    def gather(source,destination):
        packets.append(source.clone())
        destination.copy_(source.repeat(8))
        if old_peer is not None:
            destination.view(8,-1)[1,9]=old_peer

    engine=NS(execution_provider="sm70",sm70_profile="optimized",depth=3,confidence=0.,max_len=8192,
        streams=6,graphs_enabled=True,prefill_rows=2048,tp=8,rope=NS(metadata=lambda:{"rope_type":"default"}),
        kv_pair=NS(handshake=0),comm=NS(all_gather=gather))
    agree(engine,proxy,None)
    previous=int(packets[-1][9])
    monkeypatch.setattr(prefix_policy,"IDENTITY",prefix_policy.IDENTITY+"-changed")
    agree(engine,proxy,None)
    assert previous!=int(packets[-1][9])
    old_peer=previous
    with pytest.raises(RuntimeError,match="different settings"):
        agree(engine,proxy,None)

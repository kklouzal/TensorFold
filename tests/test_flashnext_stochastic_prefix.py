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
from tensorfold.families.qwen4_exp.kv_formats import get_pair


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
    decoder=NS(w=w,capacity=8192,depth=3,streams={},_slot_for=slot,kv_pair=get_pair("int8"))
    stream=NS(prompt=[11]*79,count=128,draft=True,vision=None,sampling=sampling)
    with pytest.raises(StopBorrow):
        admit(decoder,stream)
    assert borrowed==[(stream.prompt,reuse)]
    assert stream.draft is True and w.dspark is provider and decoder.depth==3
    assert MultiDecoder._keep_at(NS(prompt=stream.prompt,draft=True,st=NS(image_positions=None)))==78


@pytest.mark.parametrize("profile", ["optimized", "reference"])
def test_worker_identity_changes_and_refuses_a_peer_using_old_policy(monkeypatch, profile):
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

    engine=NS(execution_provider="sm70",sm70_profile=profile,depth=3,confidence=0.,max_len=8192,
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


@pytest.mark.parametrize("format", ["rotorquant-signed128-4", "rotorquant-signed64-norm4"])
@pytest.mark.parametrize("profile", ["optimized", "reference"])
@pytest.mark.parametrize("sampling", [None, Sampling(17, 0), Sampling(19, .7, 40, .9)])
@pytest.mark.parametrize("side", ["both", "key", "value"])
def test_rotor_prompt_borrowing_requires_separate_numerical_qualification(format, profile, sampling, side):
    pair = get_pair(format) if side == "both" else get_pair(
        "int8", format if side == "key" else "int8", format if side == "value" else "int8")
    assert not prefix_policy.reuse_allowed(weights(profile=profile), sampling,
        draft=True, vision=None, position=128, kv_pair=pair)


@pytest.mark.parametrize("format", ["fp16", "int8"])
@pytest.mark.parametrize("profile", ["optimized", "reference"])
def test_explicit_legacy_pair_keeps_its_existing_greedy_borrowing(format, profile):
    assert prefix_policy.reuse_allowed(weights(profile=profile), Sampling(17, 0),
        draft=True, vision=None, position=128, kv_pair=get_pair(format))


def test_rotor_borrowing_guard_keeps_invalid_sampling_precedence_and_other_providers():
    pair = get_pair("rotorquant-signed128-4")
    assert prefix_policy.reuse_allowed(weights(), Sampling(19, float("nan")),
        draft=True, vision=None, position=128, kv_pair=pair)
    assert prefix_policy.reuse_allowed(weights(provider="cuda"), Sampling(17, 0),
        draft=True, vision=None, position=128, kv_pair=pair)


def test_rotor_admit_declines_only_borrowing_and_retains_drafting():
    admit = method(MultiDecoder, "admit", {"time": time, "Stream": NS})
    borrowed = []

    class StopBorrow(Exception):
        pass

    def slot(prompt, allowed):
        borrowed.append(allowed)
        raise StopBorrow

    decoder = NS(w=weights(), capacity=8192, depth=3, streams={}, _slot_for=slot,
                 kv_pair=get_pair("rotorquant-signed128-4"))
    stream = NS(prompt=[11] * 128, count=128, draft=True, vision=None, sampling=Sampling(17, 0))
    with pytest.raises(StopBorrow):
        admit(decoder, stream)
    assert borrowed == [False]
    assert stream.draft is True and decoder.depth == 3


def test_reference_profile_agrees_the_new_cache_policy_before_payload():
    assert prefix_policy.identity("reference") == prefix_policy.IDENTITY
    assert prefix_policy.identity(None) is None


@pytest.mark.parametrize("format", ["rotorquant-signed128-4", "rotorquant-signed64-norm4"])
@pytest.mark.parametrize("profile", ["reference", "optimized"])
@pytest.mark.parametrize("sampling", [None, Sampling(17, 0), Sampling(19, .7, 40, .9)])
def test_serial_twin_rotor_borrowing_is_also_unqualified(format, profile, sampling):
    assert not prefix_policy.serial_reuse_allowed(weights(profile=profile), sampling,
        draft=True, position=128, kv_pair=get_pair(format))


@pytest.mark.parametrize("format", ["fp16", "int8"])
@pytest.mark.parametrize("profile", ["reference", "optimized"])
def test_serial_legacy_stochastic_policy_is_preserved(format, profile):
    assert prefix_policy.serial_reuse_allowed(weights(profile=profile), Sampling(19, .7, 40, .9),
        draft=True, position=128, kv_pair=get_pair(format))


@pytest.mark.parametrize("format,borrow", [("int8", True), ("rotorquant-signed128-4", False),
                                           ("rotorquant-signed64-norm4", False)])
def test_actual_serial_generate_keeps_drafts_and_shares_the_qualified_resume_count(format, borrow):
    engine = object.__new__(FlashNextEngine)
    engine.w, engine.kv_pair = weights(), get_pair(format)
    engine.tp, engine.served = 8, 0
    engine._begin_request = lambda _: None
    engine._limit = lambda _, count: count
    retired_calls, resume_calls, shared, decoded = [], [], [], []
    engine._retire_request = lambda request, primary, *, retired=False: retired_calls.append((primary, retired))
    hit = ([11] * 127, object())

    def resume(prompt):
        resume_calls.append(prompt)
        return hit

    def share(prompt, count, sampling, draft, cached, constraint, stop_eos, token_limit):
        shared.append((draft, cached, token_limit))
        return prompt, count, sampling, draft, cached, constraint, stop_eos, None, token_limit

    def decode(*args, **kwargs):
        decoded.append((args, kwargs))
        return {"cached": len(args[4][0]) if args[4] else 0}

    engine._resume, engine._share, engine._decode = resume, share, decode
    result = engine.generate([11] * 128, 128, Sampling(17, 0), lambda _: None, draft=True)
    assert bool(resume_calls) is borrow
    assert shared == [(True, 127 if borrow else 0, False)]
    assert decoded[0][0][4] is (hit if borrow else None)
    assert result == {"cached": 127 if borrow else 0}
    assert retired_calls == [(None, False)] and engine.served == 1

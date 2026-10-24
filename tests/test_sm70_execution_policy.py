"""The launch profile selects admitted existing operators before model ownership."""
from types import SimpleNamespace as NS
import pytest
import torch


@pytest.fixture
def policy(monkeypatch):
    from tensorfold.cuda.sm70 import execution_policy as policy, family_ops, routing_native
    from tensorfold.cuda.sm70 import attention_qualification as qualification
    from tensorfold.families.qwen4_exp.cuda import dspark_context_graphs, dspark_proposal_graphs, gguf_prefill, gguf_weights, padded_linears
    for owner,name in [(policy,"_SELECTED"),(family_ops,"_NATIVE_FAMILY"),(family_ops,"_NATIVE_ATTENTION"),
                       (routing_native,"_NATIVE_LOGITS"),(padded_linears,"_ENABLED"),
                       (dspark_context_graphs,"_ENABLED"),(dspark_proposal_graphs,"_ENABLED"),
                       (gguf_prefill,"WORKSPACE_BUDGET_BYTES"),(gguf_weights.GGUFExperts,"implementation"),
                       (gguf_weights.GGUFConcat,"implementation")]:
        monkeypatch.setattr(owner,name,getattr(owner,name))
    monkeypatch.setattr(qualification,"runtime_identity",lambda device:dict(qualification.REQUIRED_IDENTITY))
    monkeypatch.setattr(torch.cuda,"current_device",lambda:0)
    monkeypatch.setattr(torch.cuda,"get_device_capability",lambda *a:(7,0))
    return NS(module=policy,family=family_ops,routing=routing_native,padded=padded_linears,
              proposal=dspark_proposal_graphs,context=dspark_context_graphs,prefill=gguf_prefill,weights=gguf_weights,qualification=qualification)


def text():
    return dict(hidden_size=2560,num_hidden_layers=48,num_attention_heads=24,num_key_value_heads=2)


@pytest.mark.parametrize("profile",["reference","optimized"])
def test_explicit_profile_selects_same_engine_operators_and_memory_policy(policy,monkeypatch,profile):
    monkeypatch.setenv(policy.module.PROFILE_ENV,profile)
    assert policy.module.configure(text(),tp=8,torch=torch)==profile
    enabled=profile=="optimized"
    assert policy.family._NATIVE_FAMILY==policy.family._NATIVE_ATTENTION==enabled
    assert policy.routing._NATIVE_LOGITS==policy.padded._ENABLED==enabled
    assert policy.proposal._ENABLED==policy.context._ENABLED==enabled
    assert policy.weights.GGUFExperts.implementation==("packed_exact" if enabled else "reference")
    assert policy.weights.GGUFConcat.implementation==("native32" if enabled else "reference")
    assert policy.prefill.WORKSPACE_BUDGET_BYTES==((1<<30) if enabled else None)


@pytest.mark.parametrize("change",["geometry","tp","runtime"])
def test_unqualified_optimized_profile_refuses_before_changing_operators(policy,monkeypatch,change):
    monkeypatch.setenv(policy.module.PROFILE_ENV,"optimized")
    before=policy.weights.GGUFExperts.implementation
    metadata=text();tp=8
    if change=="geometry":metadata["hidden_size"]=4096
    if change=="tp":tp=4
    if change=="runtime":monkeypatch.setattr(policy.qualification,"runtime_identity",lambda d:{})
    with pytest.raises(ValueError,match="qualified"):
        policy.module.configure(metadata,tp=tp,torch=torch)
    assert policy.weights.GGUFExperts.implementation==before


def test_unknown_profile_refuses_before_cuda_or_model_ownership(policy,monkeypatch):
    monkeypatch.setenv(policy.module.PROFILE_ENV,"typo")
    with pytest.raises(ValueError,match="must be"):
        policy.module.configure({},tp=8,torch=NS())


def test_loaded_weights_receive_only_the_selected_profile_metadata(policy,monkeypatch):
    calls=[]
    monkeypatch.setattr(policy.module,"_SELECTED","optimized")
    monkeypatch.setattr(policy.qualification,"metadata_prepare",lambda w:calls.append(w))
    w=NS(meta={})
    policy.module.prepare_weights(w)
    assert calls==[w]
    assert w.meta==dict(sm70_profile="optimized",capture_ready=True,device_greedy_mtp_ready=True,device_greedy_native_ready=True)

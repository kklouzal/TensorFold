"""PACK4 is an explicit E4B-only candidate; default/reference policies stay native."""
import json
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("triton")
from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_target import E2BTarget  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_engine import E2BEngine  # noqa: E402 - optional runtime admitted above
from test_gemma4_e2b_metadata import e2b_config  # noqa: E402 - optional runtime admitted above
from test_gemma4_e4b import e4b_config  # noqa: E402 - optional runtime admitted above


def test_default_is_pack2_and_reference_leaves_configured_pack4_inactive():
    assert E2BRowPolicy().attention_query_rows == 2
    policy=E2BRowPolicy(attention_query_rows=4)
    policy.prompt=False
    with policy.native_reference():
        assert policy.prompt and policy.reference_depth==1 and policy.attention_query_rows==4
    assert not policy.prompt and policy.attention_query_rows==4


@pytest.mark.parametrize("value",[None,True,1,3,8,"4"])
def test_bad_packing_type_or_width_is_refused_before_gpu(tmp_path,monkeypatch,value):
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:pytest.fail("invalid packing reached GPU"))
    with pytest.raises(ValueError,match="packing"):
        E2BRowPolicy(attention_query_rows=value)
    with pytest.raises(ValueError,match="packing"):
        E2BEngine(tmp_path,attention_query_rows=value)


def test_pack4_e2b_refused_by_engine_and_target_before_runtime(tmp_path,monkeypatch):
    (tmp_path/"config.json").write_text(json.dumps(e2b_config()))
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:pytest.fail("E2B PACK4 reached GPU"))
    with pytest.raises(ValueError,match="E4B"):
        E2BEngine(tmp_path,attention_query_rows=4)
    with pytest.raises(ValueError,match="E4B"):
        E2BTarget(tmp_path,attention_query_rows=4)


def test_e4b_pack4_constructor_policy_is_forwarded_before_late_failure(tmp_path,monkeypatch):
    import tensorfold.cuda.chat_template as template
    import tensorfold.token_metadata as tokens
    from tensorfold.families.gemma4.cuda import e2b_engine
    (tmp_path/"config.json").write_text(json.dumps(e4b_config()))
    observed=[]
    failure=RuntimeError("stop before target allocation")

    def target(path,**kwargs):
        observed.append(kwargs)
        raise failure

    monkeypatch.setattr(template,"prepare_chat_template",lambda *_,**__:object())
    monkeypatch.setattr(tokens,"load_tokenizer",lambda _:SimpleNamespace(token_to_id=lambda _:2))
    monkeypatch.setattr(tokens,"resolve_eos",lambda *_,**__:[1])
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:None)
    monkeypatch.setattr(e2b_engine,"admit",lambda *_,**__:{"context_window":64})
    monkeypatch.setattr(e2b_engine,"E2BTarget",target)
    with pytest.raises(RuntimeError) as caught:
        E2BEngine(tmp_path,attention_query_rows=4)
    assert caught.value is failure and observed[0]["attention_query_rows"]==4


def test_primitive_rejects_pack4_kv1_before_any_workspace_allocation(monkeypatch):
    from tensorfold.families.gemma4.cuda.e2b_attention import chronological_attention
    q=torch.zeros(1,8,2,256,dtype=torch.bfloat16)
    k=torch.zeros(1,1,4,256,dtype=torch.bfloat16)
    monkeypatch.setattr(torch,"empty",lambda *_,**__:pytest.fail("invalid PACK4 allocated workspace"))
    with pytest.raises(ValueError,match="geometry"):
        chronological_attention(q,k,k,None,start=2,window=512,query_rows=4)


@pytest.mark.parametrize("packing",[2,4])
def test_pack_choice_reaches_only_canonical_decode_not_prompt_or_reference(monkeypatch,packing):
    from transformers import DynamicCache,Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention,eager_attention_forward
    from tensorfold.families.gemma4.cuda import e2b_attention,e2b_prompt_sdpa
    cfg=Gemma4TextConfig(hidden_size=16,num_hidden_layers=2,num_attention_heads=8,num_key_value_heads=2,
                        head_dim=4,global_head_dim=8,num_kv_shared_layers=0,sliding_window=8,
                        layer_types=["sliding_attention","full_attention"])
    cfg._attn_implementation="eager"
    native=Gemma4TextAttention(cfg,0).to(torch.bfloat16).eval()
    policy=E2BRowPolicy(attention_query_rows=packing)
    owned=e2b_attention.E2BTextAttention(native,policy)
    calls=[]

    def canonical(query,key,value,mask,*,start,window,query_rows):
        calls.append(query_rows)
        return eager_attention_forward(native,query,key,value,mask,scaling=1.0)[0]

    monkeypatch.setattr(e2b_attention,"chronological_attention",canonical)
    monkeypatch.setattr(e2b_prompt_sdpa,"prompt_sdpa_gqa",lambda *_,**__:pytest.fail("reference used prompt candidate"))
    hidden=torch.randn(1,1,16,dtype=torch.bfloat16)
    rope=(torch.ones(1,1,4,dtype=torch.bfloat16),torch.zeros(1,1,4,dtype=torch.bfloat16))
    with torch.inference_mode():
        owned(hidden,rope,None,{},past_key_values=DynamicCache(config=cfg))
        assert not calls
        policy.prompt=False
        owned(hidden,rope,None,{},past_key_values=DynamicCache(config=cfg))
        assert calls==[packing]
        policy.prompt_sdpa_gqa=True
        with policy.native_reference():
            owned(hidden,rope,None,{},past_key_values=DynamicCache(config=cfg))
        assert calls==[packing]

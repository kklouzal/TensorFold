"""Explicit State cache-policy ownership and in-place compact static restore."""
from dataclasses import replace
import gc
import json
import weakref
import pytest

torch=pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import DynamicCache,Gemma4TextConfig  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_target import E2BState,E2BTarget  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_full_cache import FullStaticCache  # noqa: E402 - optional runtime admitted above
from test_gemma4_e2b_metadata import e2b_config  # noqa: E402 - optional runtime admitted above


def state(*,static=True,assistant=False,owner=None):
    cfg=Gemma4TextConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=4,num_attention_heads=8,
                        num_key_value_heads=2,head_dim=4,global_head_dim=8,num_kv_shared_layers=2,
                        sliding_window=8,hidden_size_per_layer_input=0,max_position_embeddings=128,
                        layer_types=["sliding_attention","full_attention"]*2,attention_k_eq_v=False)
    return E2BState(cfg,64,owner=owner,assistant=assistant,full_static_cache=static)


def append(held,count):
    begin=held.position
    shared={}
    for index,(kind,width) in enumerate((("sliding_attention",4),("full_attention",8))):
        key=torch.arange(begin,begin+count).to(torch.bfloat16).view(1,1,-1,1).expand(1,2,-1,width).clone()
        value=key+32
        shared[kind]=held.cache.update(key,value,index)
    if held.assistant is not None:
        hidden=torch.arange(begin,begin+count).to(torch.bfloat16).unsqueeze(1).expand(-1,16).clone()
        held.assistant.stage(begin,hidden,shared)
    if held.attention_mask is not None:
        held.attention_mask=torch.cat((held.attention_mask,held.attention_mask.new_ones((1,count))),dim=-1)
    held.position+=count


@pytest.mark.parametrize("assistant",[False,True])
def test_static_snapshot_restore_and_reset_keep_live_slab_and_cache_addresses(assistant):
    held=state(assistant=assistant)
    append(held,9)
    held.rollback(9)
    snapshot=held.snapshot()
    cache=held.cache
    pointer=held.cache.layers[1].keys.data_ptr()
    assert snapshot.full_static_cache and snapshot.cache.position==9
    assert snapshot.cache.layers[1].keys.shape[-2]==9
    append(held,3)
    held.rollback(11)
    held.restore(snapshot)
    assert held.position==9 and not held.poisoned
    assert held.cache is cache and held.cache.layers[1].keys.data_ptr()==pointer
    if assistant:
        assert torch.equal(held.assistant.last_hidden,snapshot.assistant.last_hidden)
    held.reset()
    assert held.cache is cache and held.cache.layers[1].keys.data_ptr()==pointer
    assert held.position==0 and held.attention_mask is None and not held.poisoned
    if assistant:
        assert held.assistant.pending is held.assistant.last_hidden is None


def test_default_state_cache_still_native_dynamic_and_mode_mismatch_rejected_before_cloning(monkeypatch):
    owner=object()
    dynamic=state(static=False,owner=owner)
    static=state(static=True,owner=owner)
    assert type(dynamic.cache) is DynamicCache and type(static.cache) is FullStaticCache
    append(dynamic,3)
    dynamic.rollback(3)
    append(static,3)
    static.rollback(3)
    a,b=dynamic.snapshot(),static.snapshot()
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,"clone",lambda *_:pytest.fail("cache mode mismatch allocated before validation"))
        for live,snapshot in ((dynamic,b),(static,a)):
            with pytest.raises(ValueError,match="cache policy"):
                live.restore(snapshot)
    assert dynamic.position==static.position==3
    with pytest.raises(ValueError,match="cache policy"):
        static.restore(replace(b,full_static_cache=1))
    with pytest.raises(ValueError,match="outer state"):
        static.restore(replace(b,position=2))


def test_mask_and_assistant_are_prepared_before_cache_mutation_and_retired_on_failure(monkeypatch):
    held=state(assistant=True)
    append(held,9)
    held.rollback(9)
    held.attention_mask=torch.ones(1,9,dtype=torch.bool)
    snapshot=held.snapshot()
    append(held,3)
    held.rollback(12)
    old_helper=held.assistant
    pointers={snapshot.attention_mask.data_ptr(),snapshot.assistant.last_hidden.data_ptr(),
              snapshot.assistant.sliding_key.data_ptr(),snapshot.assistant.sliding_value.data_ptr()}
    cloned=[]
    refs=[]
    clone=torch.Tensor.clone
    failure=RuntimeError("cache restore mutation failed")

    def observe(value,*args,**kwargs):
        result=clone(value,*args,**kwargs)
        if value.data_ptr() in pointers:
            cloned.append(value.data_ptr())
            refs.append(weakref.ref(result))
        return result

    def fail(_):
        assert held.poisoned and set(cloned)==pointers
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,"clone",observe)
        patch.setattr(held.cache,"restore",fail)
        with pytest.raises(RuntimeError) as caught:
            held.restore(snapshot)
    assert caught.value is failure and failure.__traceback__ is not None
    assert held.poisoned and held.assistant is old_helper
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
    held.restore(snapshot)
    assert held.position==9 and not held.poisoned
    assert held.attention_mask.shape==(1,9)


@pytest.mark.parametrize("assistant",[False,True])
def test_static_crop_failure_poison_and_valid_snapshot_recovery(monkeypatch,assistant):
    held=state(assistant=assistant)
    append(held,9)
    held.rollback(9)
    snapshot=held.snapshot()
    append(held,3)
    failure=RuntimeError("cache crop failed")

    def fail(_):raise failure

    with monkeypatch.context() as patch:
        patch.setattr(held.cache,"crop",fail)
        with pytest.raises(RuntimeError) as caught:
            held.rollback(10)
    assert caught.value is failure and held.poisoned
    held.restore(snapshot)
    assert not held.poisoned and held.position==9


def test_static_flag_scope_and_boolean_checked_before_gpu(tmp_path,monkeypatch):
    from tensorfold.families.gemma4.cuda.e2b_engine import E2BEngine
    (tmp_path/"config.json").write_text(json.dumps(e2b_config()))
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:pytest.fail("unsupported static flag reached GPU"))
    with pytest.raises(ValueError,match="E4B"):
        E2BEngine(tmp_path,full_static_cache=True)
    with pytest.raises(ValueError,match="E4B"):
        E2BTarget(tmp_path,full_static_cache=True)
    with pytest.raises(ValueError,match="boolean"):
        E2BEngine(tmp_path,full_static_cache=1)


def test_static_admission_uses_live_recording_and_compact_prefix_restore_owner_bounds(tmp_path,monkeypatch):
    import tensorfold.cuda.capacity as capacity
    from tensorfold.families.gemma4.cuda import e2b_engine
    from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig
    from test_gemma4_e4b import e4b_config
    cfg=E2BConfig.from_dict(e4b_config())
    monkeypatch.setattr(capacity,"headers",lambda _: {"fake":{"shape":[1],"dtype":"BF16"}})
    geometry=[]
    bias_charges=[]

    def admitted(*args,**kwargs):
        geometry.append(args[4])
        bias_charges.append((kwargs["device_workspace_bytes"],kwargs["host_workspace_bytes"]))
        return {"context_window":131072}

    monkeypatch.setattr(capacity,"admit",admitted)
    normal=e2b_engine.admit(tmp_path,cfg,131072,True,torch,None,1,False)
    held=e2b_engine.admit(tmp_path,cfg,131072,True,torch,None,1,False,full_static_cache=True)
    assert normal["logit_bias_workspace"] == held["logit_bias_workspace"]
    assert bias_charges == [(normal["logit_bias_workspace"]["device_bytes"],
                             normal["logit_bias_workspace"]["host_bytes"])]*2
    assert bias_charges[0][0] == 16*cfg.vocab and bias_charges[0][1] >= 12*cfg.vocab
    dynamic,static=geometry
    # Default reserves live511+prefill512 rows for every snapshot. Static
    # snapshots hold committed511 only, plus ONE prepared511-row restore owner.
    swa_row_bytes=20*2*256*4
    expected_difference=(3*512-511)*swa_row_bytes
    assert dynamic.needed(131072)-static.needed(131072)==expected_difference


def test_static_engine_forwards_policy_and_declares_actual_native_capacity_slab(tmp_path,monkeypatch):
    import tensorfold.cuda.chat_template as template
    import tensorfold.token_metadata as tokens
    from tensorfold.families.gemma4.cuda import e2b_engine
    from test_gemma4_e4b import e4b_config
    (tmp_path/"config.json").write_text(json.dumps(e4b_config()))
    plan={"context_window":131072}
    admission=[]
    options=[]
    failure=RuntimeError("stop before any target weight allocation")

    def target(path,**kwargs):
        options.append(kwargs)
        raise failure

    def admit(*args,**kwargs):
        admission.append(kwargs)
        return plan

    monkeypatch.setattr(template,"prepare_chat_template",lambda *_,**__:object())
    monkeypatch.setattr(tokens,"load_tokenizer",lambda _:type("Tokenizer",(),{"token_to_id":lambda *_:2})())
    monkeypatch.setattr(tokens,"resolve_eos",lambda *_,**__:[1])
    monkeypatch.setattr(torch.cuda,"set_device",lambda _:None)
    monkeypatch.setattr(e2b_engine,"admit",admit)
    monkeypatch.setattr(e2b_engine,"E2BTarget",target)
    with pytest.raises(RuntimeError) as caught:
        e2b_engine.E2BEngine(tmp_path,full_static_cache=True)
    assert caught.value is failure
    assert admission==[{"full_static_cache":True}] and options[0]["full_static_cache"] is True
    assert plan["permanent_full_slab_bytes"]==2147483648

"""Unconnected fixed full-donor candidate against native HF DynamicCache."""
import copy
from dataclasses import replace
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers import DynamicCache, Gemma4TextConfig  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda import e2b_full_cache  # noqa: E402 - optional runtime admitted above


def config(window=8):
    return Gemma4TextConfig(hidden_size=16, num_hidden_layers=6, num_attention_heads=8,
                           num_key_value_heads=2, head_dim=4, global_head_dim=8,
                           layer_types=["sliding_attention", "full_attention"]*3,
                           num_kv_shared_layers=2, sliding_window=window,
                           max_position_embeddings=2048, attention_k_eq_v=False)


def cache_pair(capacity=64, window=8):
    cfg = config(window)
    reference = DynamicCache(config=cfg)
    reference.activate_past_recording()
    candidate = e2b_full_cache.FullStaticCache(cfg, capacity)
    return reference, candidate


def values(begin, end, width, offset):
    return (torch.arange(begin, end).float()/16+offset).to(torch.bfloat16).view(1, 1, -1, 1).expand(1, 2, -1, width).clone()


def append(reference, candidate, count):
    begin = reference.get_seq_length()
    for i, kind in enumerate(candidate.kinds):
        assert candidate.get_mask_sizes(count, i) == reference.get_mask_sizes(count, i)
        width = 8 if kind == "full_attention" else 4
        key, value = values(begin, begin+count, width, i), values(begin, begin+count, width, i+32)
        expected = reference.update(key, value, i)
        actual = candidate.update(key, value, i)
        assert all(torch.equal(a, b) for a, b in zip(expected, actual))
    assert candidate.get_seq_length() == reference.get_seq_length()


def equal_committed(reference, candidate):
    length = reference.get_seq_length()
    assert candidate.get_seq_length() == length
    for kind, a, b in zip(candidate.kinds, reference.layers, candidate.layers):
        key, value = (b.keys[:, :, :length], b.values[:, :, :length]) if kind == "full_attention" else (b.keys,b.values)
        assert torch.equal(a.keys, key) and torch.equal(a.values, value)


@pytest.mark.parametrize("prefix", [7, 8, 9, 511, 512, 513])
@pytest.mark.parametrize("accepted", [0, 1, 3, 7])
def test_native_full_and_sliding_updates_partial_rejection_and_snapshot_replay(prefix, accepted):
    reference, candidate = cache_pair(capacity=1024, window=512 if prefix>9 else 8)
    append(reference, candidate, prefix)
    reference.crop(0)
    candidate.crop(0)
    snapshot = candidate.snapshot()
    native_snapshot = copy.deepcopy(reference)
    pointers = [layer.keys.data_ptr() for kind,layer in zip(candidate.kinds,candidate.layers) if kind=="full_attention"]
    append(reference, candidate, 7)
    reference.crop(-(7-accepted))
    candidate.crop(-(7-accepted))
    equal_committed(reference,candidate)
    candidate.restore(snapshot)
    equal_committed(native_snapshot,candidate)
    assert pointers == [layer.keys.data_ptr() for kind,layer in zip(candidate.kinds,candidate.layers) if kind=="full_attention"]
    assert not candidate.is_compileable and candidate.is_croppable
    append(native_snapshot,candidate,1)
    native_snapshot.crop(0)
    candidate.crop(0)
    equal_committed(native_snapshot,candidate)


def test_snapshots_clone_only_logical_full_prefix_and_restore_never_clones_full_sources(monkeypatch):
    reference,candidate = cache_pair(capacity=1024)
    append(reference,candidate,9)
    candidate.crop(0)
    snapshot = candidate.snapshot()
    for kind,saved in zip(candidate.kinds,snapshot.layers):
        rows = 9 if kind=="full_attention" else 7
        assert saved.keys.shape[-2] == rows
        assert saved.keys.untyped_storage().nbytes() == saved.keys.numel()*saved.keys.element_size()
    immutable = [saved.keys.clone() for kind,saved in zip(candidate.kinds,snapshot.layers) if kind=="full_attention"]
    full_sources = {saved.keys.data_ptr() for kind,saved in zip(candidate.kinds,snapshot.layers) if kind=="full_attention"}
    full_sources |= {saved.values.data_ptr() for kind,saved in zip(candidate.kinds,snapshot.layers) if kind=="full_attention"}
    clone = torch.Tensor.clone

    def no_double_copy(value,*args,**kwargs):
        if value.data_ptr() in full_sources:
            pytest.fail("full immutable source was cloned during restore")
        return clone(value,*args,**kwargs)

    monkeypatch.setattr(torch.Tensor,"clone",no_double_copy)
    candidate.restore(snapshot)
    for kind,layer in zip(candidate.kinds,candidate.layers):
        if kind=="full_attention":
            layer.keys[:, :, :9].zero_()
    for saved,expected in zip((s for k,s in zip(candidate.kinds,snapshot.layers) if k=="full_attention"),immutable):
        assert torch.equal(saved.keys,expected)
    with pytest.raises(TypeError,match="compact snapshot"):
        copy.deepcopy(candidate)


def test_empty_reset_preserves_full_addresses_and_never_returns_stale_rows():
    reference,candidate = cache_pair()
    empty = candidate.snapshot()
    candidate.crop(0)
    append(reference,candidate,9)
    candidate.crop(0)
    pointers = [layer.keys.data_ptr() for kind,layer in zip(candidate.kinds,candidate.layers) if kind=="full_attention"]
    candidate.reset()
    assert candidate.get_seq_length()==0
    candidate.restore(empty)
    assert candidate.get_seq_length()==0
    assert pointers == [layer.keys.data_ptr() for kind,layer in zip(candidate.kinds,candidate.layers) if kind=="full_attention"]
    fresh = DynamicCache(config=config())
    fresh.activate_past_recording()
    append(fresh,candidate,1)
    fresh.crop(0)
    candidate.crop(0)
    equal_committed(fresh,candidate)


def test_snapshot_refuses_uncommitted_sliding_before_full_clone(monkeypatch):
    reference,candidate = cache_pair()
    append(reference,candidate,9)
    monkeypatch.setattr(torch.Tensor,"clone",lambda *_: pytest.fail("pending snapshot allocated before validation"))
    with pytest.raises(RuntimeError,match="committed native sliding"):
        candidate.snapshot()


def test_restore_swa_preparation_failure_leaves_live_cache_unchanged(monkeypatch):
    reference,candidate = cache_pair()
    append(reference,candidate,9)
    reference.crop(0)
    candidate.crop(0)
    snapshot = candidate.snapshot()
    append(reference,candidate,3)
    reference.crop(0)
    candidate.crop(0)
    def fail(_):
        raise MemoryError("SWA preparation failed")
    monkeypatch.setattr(e2b_full_cache,"_copy_sliding",fail)
    with pytest.raises(MemoryError,match="preparation"):
        candidate.restore(snapshot)
    assert not candidate.poisoned
    equal_committed(reference,candidate)


@pytest.mark.parametrize("failure_at", [1, 2, 3, 4])
def test_partial_full_restore_is_poisoned_retry_preserves_slab_addresses(monkeypatch, failure_at):
    reference,candidate = cache_pair()
    append(reference,candidate,9)
    reference.crop(0)
    candidate.crop(0)
    snapshot = candidate.snapshot()
    saved_reference = copy.deepcopy(reference)
    append(reference,candidate,3)
    reference.crop(0)
    candidate.crop(0)
    pointers = [layer.keys.data_ptr() for kind,layer in zip(candidate.kinds,candidate.layers) if kind=="full_attention"]
    copy_into = torch.Tensor.copy_
    calls=[]
    injected=RuntimeError("full restore copy failed")

    def fail(value,source,*args,**kwargs):
        calls.append(True)
        if len(calls)==failure_at:
            raise injected
        return copy_into(value,source,*args,**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,"copy_",fail)
        with pytest.raises(RuntimeError) as caught:
            candidate.restore(snapshot)
        assert caught.value is injected and candidate.poisoned
    with pytest.raises(RuntimeError,match="failed"):
        candidate.get_mask_sizes(1,0)
    candidate.restore(snapshot)
    assert not candidate.poisoned
    equal_committed(saved_reference,candidate)
    assert pointers == [layer.keys.data_ptr() for kind,layer in zip(candidate.kinds,candidate.layers) if kind=="full_attention"]


def test_snapshot_identity_capacity_geometry_and_input_fail_before_mutation():
    reference,candidate = cache_pair()
    append(reference,candidate,3)
    candidate.crop(0)
    snapshot=candidate.snapshot()
    with pytest.raises(ValueError,match="different cache"):
        candidate.restore(replace(snapshot,owner=object()))
    with pytest.raises(ValueError,match="different cache"):
        candidate.restore(replace(snapshot,position=65))
    with pytest.raises(ValueError):
        candidate.crop(-4)
    with pytest.raises(ValueError):
        candidate.crop(1)
    with pytest.raises(ValueError,match="geometry"):
        candidate.update(torch.zeros(1,1,1,8,dtype=torch.bfloat16),torch.zeros(1,1,1,8,dtype=torch.bfloat16),1)
    assert not candidate.poisoned and candidate.get_seq_length()==3


@pytest.mark.parametrize("failure_at", [1, 2])
def test_failed_native_full_index_update_poison_and_snapshot_recovers_device_counter(monkeypatch,failure_at):
    reference,candidate=cache_pair()
    append(reference,candidate,9)
    reference.crop(0)
    candidate.crop(0)
    snapshot=candidate.snapshot()
    original=torch.Tensor.index_copy_
    calls=[]
    failure=RuntimeError("native index copy failed")

    def index_copy(value,*args,**kwargs):
        calls.append(True)
        if len(calls)==failure_at:
            raise failure
        return original(value,*args,**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,"index_copy_",index_copy)
        key=values(9,10,8,1)
        with pytest.raises(RuntimeError) as caught:
            candidate.update(key,key+32,1)
        assert caught.value is failure and candidate.poisoned
    with pytest.raises(RuntimeError,match="failed"):
        candidate.update(values(9,10,4,0),values(9,10,4,32),0)
    candidate.restore(snapshot)
    equal_committed(reference,candidate)
    assert candidate.layers[1].cumulative_length.item()==9  # CPU fixture observes native scalar; runtime never reads it.


def test_partial_crop_failure_poison_and_known_snapshot_recovery(monkeypatch):
    reference,candidate=cache_pair()
    append(reference,candidate,9)
    reference.crop(0)
    candidate.crop(0)
    snapshot=candidate.snapshot()
    before=copy.deepcopy(reference)
    append(reference,candidate,3)

    def fail(_):
        raise RuntimeError("full crop failed")

    with monkeypatch.context() as patch:
        patch.setattr(candidate.layers[1],"crop",fail)
        with pytest.raises(RuntimeError,match="crop failed"):
            candidate.crop(-2)
        assert candidate.poisoned
    candidate.restore(snapshot)
    equal_committed(before,candidate)


def test_partial_first_slab_allocation_retires_unpublished_owner_on_reset(monkeypatch):
    import weakref
    _,candidate=cache_pair()
    original=torch.zeros
    owners=[]
    calls=[]

    def zeros(*args,**kwargs):
        if args and args[0]==(1,2,64,8):
            calls.append(True)
            if len(calls)==2:
                raise MemoryError("value slab allocation failed")
            value=original(*args,**kwargs)
            owners.append(weakref.ref(value))
            return value
        return original(*args,**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(torch,"zeros",zeros)
        key=values(0,1,8,1)
        with pytest.raises(MemoryError,match="value slab"):
            candidate.update(key,key+32,1)
        assert candidate.poisoned and owners[0]() is not None
    candidate.reset()
    assert not candidate.poisoned and owners[0]() is None
    assert candidate.get_seq_length()==0
    assert candidate.layers[1].keys is candidate.layers[1].values is None


def test_full_native_hf_model_logits_match_across_swa_and_compact_prefix_restore():
    from transformers import Gemma4ForCausalLM
    cfg=Gemma4TextConfig(hidden_size=16,intermediate_size=32,num_hidden_layers=6,num_attention_heads=8,
                        num_key_value_heads=2,head_dim=4,global_head_dim=8,num_kv_shared_layers=2,
                        sliding_window=8,hidden_size_per_layer_input=4,vocab_size_per_layer_input=64,
                        vocab_size=64,max_position_embeddings=128,enable_moe_block=False,
                        use_double_wide_mlp=False,attention_k_eq_v=False,
                        layer_types=["sliding_attention","full_attention"]*3)
    cfg._attn_implementation="sdpa"
    model=Gemma4ForCausalLM(cfg).to(torch.bfloat16).eval()
    reference=DynamicCache(config=cfg)
    reference.activate_past_recording()
    candidate=e2b_full_cache.FullStaticCache(cfg,64)
    with torch.inference_mode():
        for ids in ([2,7,11,19,4,8,12],[6,15,22,23],[9]):
            tokens=torch.tensor([ids])
            a=model(input_ids=tokens,past_key_values=reference,use_cache=True).logits
            b=model(input_ids=tokens,past_key_values=candidate,use_cache=True).logits
            assert torch.equal(a,b)
            reference.crop(0)
            candidate.crop(0)
        snapshot=candidate.snapshot()
        native_snapshot=copy.deepcopy(reference)
        tokens=torch.tensor([[18]])
        expected=model(input_ids=tokens,past_key_values=reference,use_cache=True).logits
        candidate.restore(snapshot)
        actual=model(input_ids=tokens,past_key_values=candidate,use_cache=True).logits
        assert torch.equal(actual,expected)
        candidate.crop(0)
        candidate.restore(snapshot)
        assert torch.equal(model(input_ids=tokens,past_key_values=candidate,use_cache=True).logits,
                           model(input_ids=tokens,past_key_values=native_snapshot,use_cache=True).logits)


def test_late_snapshot_failure_retires_owned_prefix_clones_while_primary_traceback_is_retained(monkeypatch):
    import gc
    import weakref
    reference,candidate=cache_pair()
    append(reference,candidate,9)
    candidate.crop(0)
    clone=torch.Tensor.clone
    refs=[]
    full_calls=[]
    failure=MemoryError("later full-prefix clone failed")

    def fail_late(value,*args,**kwargs):
        if value.ndim==4 and value.shape[-1]==8:
            full_calls.append(True)
            if len(full_calls)==3:
                raise failure
        result=clone(value,*args,**kwargs)
        if value.ndim==4:
            refs.append(weakref.ref(result))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,"clone",fail_late)
        with pytest.raises(MemoryError) as caught:
            candidate.snapshot()
    assert caught.value is failure and failure.__traceback__ is not None
    gc.collect()
    assert refs and all(ref() is None for ref in refs)
    assert not candidate.poisoned


def test_failed_restore_retires_prepared_swa_clones_without_stripping_primary(monkeypatch):
    import gc
    import weakref
    reference,candidate=cache_pair()
    append(reference,candidate,9)
    candidate.crop(0)
    snapshot=candidate.snapshot()
    copy_sliding=e2b_full_cache._copy_sliding
    refs=[]
    failure=RuntimeError("full restore failed after SWA preparation")

    def observe(layer):
        prepared=copy_sliding(layer)
        refs.extend((weakref.ref(prepared.keys),weakref.ref(prepared.values)))
        return prepared

    def fail(*_):
        raise failure

    with monkeypatch.context() as patch:
        patch.setattr(e2b_full_cache,"_copy_sliding",observe)
        patch.setattr(torch.Tensor,"copy_",fail)
        with pytest.raises(RuntimeError) as caught:
            candidate.restore(snapshot)
    assert caught.value is failure and failure.__traceback__ is not None and candidate.poisoned
    gc.collect()
    assert refs and all(ref() is None for ref in refs)


def test_failed_swa_clone_drops_owned_memo_clone_before_propagation(monkeypatch):
    import gc
    import weakref
    reference,candidate=cache_pair()
    append(reference,candidate,9)
    candidate.crop(0)
    clone=torch.Tensor.clone
    refs=[]
    failure=MemoryError("second SWA clone failed")

    def fail(value,*args,**kwargs):
        if refs:
            raise failure
        result=clone(value,*args,**kwargs)
        refs.append(weakref.ref(result))
        return result

    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor,"clone",fail)
        with pytest.raises(MemoryError) as caught:
            e2b_full_cache._copy_sliding(candidate.layers[0])
    assert caught.value is failure and failure.__traceback__ is not None
    gc.collect()
    assert refs[0]() is None

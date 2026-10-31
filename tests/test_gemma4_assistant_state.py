"""Committed assistant feature ownership across target verification/rejection."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
from tensorfold.families.gemma4.cuda.assistant_state import AssistantHistory  # noqa: E402 - optional runtime admitted above


def history():
    return AssistantHistory(SimpleNamespace(sliding_window=512, num_hidden_layers=42, num_kv_shared_layers=18,
                                           layer_types=["full_attention" if (i+1)%6 == 0 else "sliding_attention"
                                                        for i in range(42)]))


def tensor(begin, end, heads=2, width=4):
    return torch.arange(begin, end).to(torch.bfloat16).view(1, 1, -1, 1).expand(1, heads, -1, width).clone()


def stage(held, start, rows):
    begin = max(0, start-511)
    key = tensor(begin, start+rows)
    value = key+1024
    hidden = torch.arange(start, start+rows).to(torch.bfloat16).unsqueeze(1).expand(-1, 8).clone()
    held.stage(start, hidden, {"sliding_attention": (key, value)})
    return hidden, key, value


@pytest.mark.parametrize("prefix", [511, 512, 513])
@pytest.mark.parametrize("accepted", [0, 1, 3, 8])
def test_native_512_window_and_last_accepted_hidden_exclude_rejected_rows(prefix, accepted):
    held = history()
    stage(held, 0, prefix)
    held.commit(prefix)
    before = held.snapshot()
    hidden, key, value = stage(held, prefix, 8)
    held.commit(prefix+accepted)
    position = prefix+accepted
    expected = tensor(max(0, position-512), position)
    assert torch.equal(held.sliding_key, expected)
    assert torch.equal(held.sliding_value, expected+1024)
    assert torch.equal(held.last_hidden, torch.full((1, 1, 8), position-1, dtype=torch.bfloat16))
    key.zero_()
    value.zero_()
    hidden.zero_()
    assert torch.equal(held.sliding_key, expected)  # current features own storage independently of forward/cache
    held.restore(before)
    assert torch.equal(held.sliding_key, tensor(max(0,prefix-512),prefix))
    held.sliding_key.zero_()
    assert bool(before.sliding_key.any())  # prefix snapshot never aliases mutable owners


def test_view_borrows_only_committed_donors_and_preserves_owned_padding_mask():
    held = history()
    stage(held, 0, 513)
    held.commit(513)
    layer = SimpleNamespace(keys=tensor(0, 520, width=8), values=tensor(1024, 1544, width=8))
    mask = torch.ones(1, 513, dtype=torch.bool)
    mask[:, 4] = False
    state = SimpleNamespace(position=513, poisoned=False, attention_mask=mask,
                            cache=SimpleNamespace(layers=[None]*23+[layer]))
    view = held.view(state)
    assert view["position"] == 513 and view["attention_mask"] is mask
    assert view["last_hidden_state"] is held.last_hidden
    assert view["shared_kv_states"]["sliding_attention"][0] is held.sliding_key
    full_key, full_value = view["shared_kv_states"]["full_attention"]
    assert full_key.shape[-2] == full_value.shape[-2] == 513
    assert full_key.data_ptr() == layer.keys.data_ptr()
    stage(held, 513, 1)
    with pytest.raises(RuntimeError, match="committed healthy"):
        held.view(state)
    with pytest.raises(RuntimeError, match="uncommitted"):
        held.snapshot()


def test_zero_commit_preserves_previous_snapshot_and_reset_retires_all_features():
    held = history()
    stage(held, 0, 9)
    held.commit(9)
    snapshot = held.snapshot()
    stage(held, 9, 2)
    held.commit(9)
    assert torch.equal(held.last_hidden, snapshot.last_hidden)
    held.reset()
    assert held.last_hidden is held.sliding_key is held.sliding_value is held.pending is None
    with pytest.raises(RuntimeError, match="healthy"):
        held.view(SimpleNamespace(poisoned=False, position=0))


@pytest.mark.parametrize("prefix", [511, 512, 513])
def test_retained_512_inputs_match_public_native_assistant_padding_masks(prefix):
    pytest.importorskip("transformers")
    from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig
    cfg = Gemma4TextConfig(hidden_size=16, num_hidden_layers=4, num_attention_heads=4,
                          num_key_value_heads=2, head_dim=4, global_head_dim=8, intermediate_size=32,
                          hidden_size_per_layer_input=0, vocab_size_per_layer_input=0, vocab_size=64,
                          num_kv_shared_layers=4, sliding_window=512,
                          layer_types=["sliding_attention"]*3+["full_attention"])
    assistant = Gemma4AssistantForCausalLM(Gemma4AssistantConfig(text_config=cfg.to_dict(),
                                                               backbone_hidden_size=8, num_centroids=8,
                                                               centroid_intermediate_top_k=4)).eval()
    assistant.config._attn_implementation = "eager"
    held = history()
    stage(held, 0, prefix)
    held.commit(prefix)
    # Simulate a speculative8-row forward accepting3, preserving native512.
    stage(held, prefix, 8)
    held.commit(prefix+3)
    position = prefix+3
    mask = torch.ones(1, position, dtype=torch.bool)
    mask[:, max(0,position-512)] = False
    full = SimpleNamespace(keys=tensor(0, position, width=8), values=tensor(0, position, width=8))
    state = SimpleNamespace(position=position, poisoned=False, attention_mask=mask,
                            cache=SimpleNamespace(layers=[None]*23+[full]))
    actual = held.view(state)
    # Independent native-style donor output: last512ofacceptedfullprefix.
    expected_shared = {"sliding_attention": (tensor(max(0,position-512),position),
                                             tensor(max(0,position-512),position)+1024),
                       "full_attention": (full.keys, full.values)}
    embeddings = torch.zeros(1, 1, 16)
    masks = assistant.create_attention_masks(embeddings, actual["attention_mask"], actual["shared_kv_states"])
    expected = assistant.create_attention_masks(embeddings, mask, expected_shared)
    for name in masks:
        if masks[name] is None:
            assert expected[name] is None
        else:
            assert torch.equal(masks[name], expected[name])
    assert actual["shared_kv_states"]["sliding_attention"][0].shape[-2] == 512


def native_state():
    from transformers import Gemma4TextConfig
    from tensorfold.families.gemma4.cuda.e2b_target import E2BState
    cfg = Gemma4TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                          num_attention_heads=8, num_key_value_heads=2, head_dim=4,
                          global_head_dim=4, num_kv_shared_layers=2, sliding_window=512,
                          hidden_size_per_layer_input=0, max_position_embeddings=1024,
                          layer_types=["sliding_attention", "full_attention"]*2)
    return E2BState(cfg, 1024, assistant=True)


def append_native(state, count):
    begin = state.position
    key = tensor(begin, begin+count)
    shared = {}
    for i, kind in enumerate(("sliding_attention", "full_attention")):
        k, v = state.cache.update(key, key+1024, i)
        shared[kind] = k, v
    hidden = torch.arange(begin, begin+count).to(torch.bfloat16).unsqueeze(1).expand(-1, 8).clone()
    state.assistant.stage(begin, hidden, shared)
    state.position += count


@pytest.mark.parametrize("failure", ["clone", "crop"])
def test_failed_assistant_commit_poison_and_snapshot_recovery(monkeypatch, failure):
    state = native_state()
    append_native(state, 513)
    state.rollback(513)
    before = state.snapshot()
    append_native(state, 8)
    injected = RuntimeError("commit allocation/retirement failed")

    def fail(*_):
        raise injected

    with monkeypatch.context() as patch:
        if failure == "clone":
            patch.setattr(state.assistant, "commit", fail)
        else:
            patch.setattr(state.cache, "crop", fail)
        with pytest.raises(RuntimeError) as caught:
            state.rollback(516)
        assert caught.value is injected and state.poisoned
    with pytest.raises(RuntimeError, match="failed"):
        state.snapshot()
    state.restore(before)
    assert state.position == 513 and not state.poisoned
    assert torch.equal(state.assistant.sliding_key, tensor(1, 513))
    assert state.cache.layers[0].keys.shape[-2] == 511
    state.reset()
    assert state.assistant.pending is state.assistant.last_hidden is None


def test_pending_snapshot_rejected_before_large_cache_copy_and_mode_restore_checked(monkeypatch):
    from tensorfold.families.gemma4.cuda import e2b_target
    state = native_state()
    append_native(state, 3)
    with monkeypatch.context() as patch:
        patch.setattr(e2b_target.copy, "deepcopy", lambda _: pytest.fail("pending snapshot copied whole cache"))
        with pytest.raises(RuntimeError, match="uncommitted"):
            state.snapshot()
    state.rollback(3)
    snapshot = state.snapshot()
    plain = e2b_target.E2BState(state.config, 1024, owner=state.owner)
    with pytest.raises(ValueError, match="feature ownership"):
        plain.restore(snapshot)
    with pytest.raises(ValueError, match="tentative"):
        state.rollback(2)


@pytest.mark.parametrize("assistant", [False, True])
def test_target_public_hf_forward_stages_native_final_hidden_and_donor_kv(assistant):
    from transformers import Gemma4ForCausalLM
    from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy
    from tensorfold.families.gemma4.cuda.e2b_target import E2BTarget
    state = native_state()
    native = Gemma4ForCausalLM(state.config).to(torch.bfloat16).eval()
    owner = object.__new__(E2BTarget)
    owner._closed, owner.device = False, torch.device("cpu")
    owner._row_policy = E2BRowPolicy()
    owner.config = SimpleNamespace(profile="E4B", vocab=state.config.vocab_size, softcap=30.0)
    owner.tap_layers = ()
    owner.model = SimpleNamespace(config=state.config, model=native.model, lm_head=native.lm_head)
    state = owner.make_state(1024, assistant=assistant)
    ids = [2, 7, 11, 19]
    with torch.inference_mode():
        result = owner.forward(ids, state, prompt=True)
        if assistant:
            assert state.assistant.pending is not None
        else:
            assert state.assistant is None
        state.rollback(state.position)
        if assistant:
            held = owner.assistant_state(state)
            assert torch.equal(held["last_hidden_state"], result.hidden[-1:].unsqueeze(0))
            assert held["shared_kv_states"]["sliding_attention"][0].shape[-2] == 4
            assert held["shared_kv_states"]["full_attention"][0].shape[-2] == 4
        else:
            with pytest.raises(ValueError, match="feature contract"):
                owner.assistant_state(state)

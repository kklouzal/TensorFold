"""Project-owned position/rollback/snapshot adapter at native HF cache boundaries."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from tensorfold.families.gemma4.cuda.e2b_target import E2BState, E2BTarget, TAP_LAYERS  # noqa: E402 - optional runtime admitted above


def state():
    from transformers import Gemma4TextConfig
    cfg = Gemma4TextConfig(hidden_size=32, num_hidden_layers=4, num_attention_heads=2,
                          num_key_value_heads=1, head_dim=4, global_head_dim=4,
                          layer_types=["sliding_attention", "full_attention"] * 2,
                          num_kv_shared_layers=2, sliding_window=512)
    return E2BState(cfg, 1024)


def append(s, count):
    start = s.position
    data = torch.arange(start, start + count, dtype=torch.float32).reshape(1, 1, count, 1).expand(1, 1, count, 4)
    for i in range(2):
        s.cache.update(data, data + 10000, i)
    s.position += count


@pytest.mark.parametrize("length", [511, 512, 513])
@pytest.mark.parametrize("accepted", [0, 1, 7])
def test_partial_and_full_rejection_preserve_committed_history(length, accepted):
    s = state()
    append(s, length)
    s.rollback(length)
    before = s.snapshot()
    append(s, 7)
    s.rollback(length + accepted)
    assert s.position == length + accepted
    assert s.cache.get_seq_length() == length + accepted
    full = s.cache.layers[1].keys[0, 0, :, 0]
    assert torch.equal(full, torch.arange(length + accepted, dtype=torch.float32))
    assert s.cache.layers[0].keys.shape[-2] == min(length + accepted, 511)
    s.restore(before)
    assert s.position == length
    assert s.cache.get_seq_length() == length
    append(s, 1)
    assert before.cache.get_seq_length() == length  # no aliasing the immutable prefix snapshot


def test_invalid_rollback_cannot_advance_or_underflow_position():
    s = state()
    append(s, 5)
    with pytest.raises(ValueError, match="rollback"):
        s.rollback(6)
    with pytest.raises(ValueError, match="rollback"):
        s.rollback(-1)


def test_four_owned_taps_match_native_hf_hidden_state_indices():
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    cfg = Gemma4TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=35,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                          global_head_dim=8, num_kv_shared_layers=20, sliding_window=8,
                          hidden_size_per_layer_input=4, vocab_size_per_layer_input=64,
                          vocab_size=64, max_position_embeddings=128, enable_moe_block=False,
                          use_double_wide_mlp=True, attention_k_eq_v=False,
                          layer_types=["full_attention" if (i + 1) % 5 == 0 else "sliding_attention"
                                       for i in range(35)])
    model = Gemma4ForCausalLM(cfg).eval()
    owner = object.__new__(E2BTarget)
    owner._collect_taps, owner._taps = True, {}
    handles = [model.model.layers[i].register_forward_hook(owner._tap_hook(i)) for i in TAP_LAYERS]
    try:
        with torch.inference_mode():
            result = model(input_ids=torch.tensor([[2, 7, 11, 19]]), output_hidden_states=True, use_cache=False)
        for layer, hidden_index in zip(TAP_LAYERS, (3, 13, 23, 33)):
            assert torch.equal(owner._taps[layer], result.hidden_states[hidden_index].squeeze(0))
    finally:
        for handle in handles:
            handle.remove()


def test_snapshot_cannot_cross_state_owner_or_context():
    one, another = state(), state()
    append(one, 3)
    snapshot = one.snapshot()
    with pytest.raises(ValueError, match="ownership"):
        another.restore(snapshot)
    one.limit = 2
    with pytest.raises(ValueError, match="ownership"):
        one.restore(snapshot)


def test_target_state_constructor_rejects_unsupported_options_before_use():
    target = object.__new__(E2BTarget)
    with pytest.raises(TypeError, match="rows"):
        target.make_state(1024, rows=8)

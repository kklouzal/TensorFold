"""Native assistant GQA selection and independent public HF oracle scopes."""
import copy
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
pytest.importorskip("triton")
from transformers import Gemma4AssistantConfig, Gemma4AssistantForCausalLM, Gemma4TextConfig  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.assistant_attention import (  # noqa: E402 - optional runtime admitted above
    AssistantAttentionPolicy, assistant_native_reference, install_assistant_attention,
)


def fixture(position=9):
    torch.manual_seed(719)
    text = Gemma4TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=4,
                           num_attention_heads=4, num_key_value_heads=2, head_dim=4, global_head_dim=8,
                           vocab_size=64, layer_types=["sliding_attention"]*3+["full_attention"],
                           num_kv_shared_layers=4, sliding_window=512, hidden_size_per_layer_input=0,
                           vocab_size_per_layer_input=0, use_double_wide_mlp=False,
                           attention_k_eq_v=False, attention_dropout=0.0,
                           rope_parameters={"sliding_attention": {"rope_type": "default", "rope_theta": 10000.0},
                                            "full_attention": {"rope_type": "proportional", "rope_theta": 1000000.0,
                                                               "partial_rotary_factor": .5}})
    config = Gemma4AssistantConfig(text_config=text.to_dict(), backbone_hidden_size=16,
                                   num_centroids=8, centroid_intermediate_top_k=2, use_ordered_embeddings=True)
    native = Gemma4AssistantForCausalLM(config).to(torch.bfloat16).eval()
    native.masked_embedding.token_ordering.copy_(torch.randperm(64))
    shared = {kind: tuple(torch.randn(1, 2, count, width, dtype=torch.bfloat16) for _ in range(2))
              for kind, count, width in (("sliding_attention", min(position, 512), 4),
                                         ("full_attention", position, 8))}
    kwargs = {"inputs_embeds": torch.randn(1, 1, 32, dtype=torch.bfloat16),
              "position_ids": torch.tensor([[position]]),
              "attention_mask": torch.ones(1, position, dtype=torch.bool),
              "shared_kv_states": shared, "use_cache": False, "return_dict": True}
    return native, kwargs


@pytest.mark.parametrize("position", [1, 511, 512, 513])
@pytest.mark.parametrize("padding", [False, True])
def test_full_native_four_layer_assistant_gqa_matches_logits_states_and_immutable_kv(position, padding):
    native, kwargs = fixture(position)
    if padding:
        kwargs["attention_mask"][:, 0] = False
    candidate = copy.deepcopy(native)
    old_q = [layer.self_attn.q_proj.weight for layer in candidate.model.layers]
    old_state_keys = tuple(candidate.state_dict())
    policy = install_assistant_attention(candidate, enabled=True)
    assert policy.prompt and policy.prompt_sdpa_gqa
    assert tuple(candidate.state_dict()) == old_state_keys
    assert all(layer.self_attn.q_proj.weight is weight for layer, weight in zip(candidate.model.layers, old_q))
    original = {kind: tuple(value.clone() for value in values)
                for kind, values in kwargs["shared_kv_states"].items()}
    with torch.inference_mode():
        expected = native(**kwargs)
        actual = candidate(**kwargs)
    assert torch.equal(expected.logits, actual.logits)
    assert torch.equal(expected.last_hidden_state, actual.last_hidden_state)
    for kind, values in original.items():
        assert all(torch.equal(a, b) for a, b in zip(values, kwargs["shared_kv_states"][kind]))


def test_disabled_path_keeps_native_modules_and_never_reads_oracle_policy():
    native, _ = fixture()
    original = [layer.self_attn for layer in native.model.layers]
    assert install_assistant_attention(native) is None
    assert all(layer.self_attn is old for layer, old in zip(native.model.layers, original))
    with assistant_native_reference(native, None) as oracle:
        assert oracle is native


def test_head_cannot_enter_target_canonical_decoder_path():
    policy = AssistantAttentionPolicy()
    with pytest.raises(ValueError, match="target decoder"):
        policy.prompt = False
    assert policy.prompt is True


def test_native_reference_nesting_failure_forces_public_hf_without_gqa_candidate(monkeypatch):
    from tensorfold.families.gemma4.cuda import e2b_attention, e2b_prompt_sdpa
    native, kwargs = fixture(513)
    candidate = copy.deepcopy(native)
    policy = install_assistant_attention(candidate, enabled=True)
    policy.position = 42
    monkeypatch.setattr(e2b_prompt_sdpa, "prompt_sdpa_gqa", lambda *_, **__: pytest.fail("native oracle used GQA candidate"))
    monkeypatch.setattr(e2b_attention, "chronological_attention", lambda *_, **__: pytest.fail("assistant used target decoder"))
    failure = RuntimeError("reference failed")
    with torch.inference_mode():
        expected = native(**kwargs).logits
        with pytest.raises(RuntimeError) as caught:
            with assistant_native_reference(candidate, policy) as oracle:
                assert policy.reference_depth == 1 and policy.prompt
                assert torch.equal(oracle(**kwargs).logits, expected)
                with pytest.raises(ValueError):
                    with assistant_native_reference(candidate, policy):
                        assert policy.reference_depth == 2
                        assert torch.equal(oracle(**kwargs).logits, expected)
                        raise ValueError("inner reference failed")
                assert policy.reference_depth == 1
                raise failure
        assert caught.value is failure
    assert policy.reference_depth == 0 and policy.prompt and policy.position == 42
    assert policy.prompt_sdpa_gqa is True


@pytest.mark.parametrize("value", [1, None, "true"])
def test_invalid_enable_switch_rejected_without_replacing_modules(value):
    native, _ = fixture()
    original = [layer.self_attn for layer in native.model.layers]
    with pytest.raises(ValueError, match="boolean"):
        install_assistant_attention(native, enabled=value)
    assert all(layer.self_attn is old for layer, old in zip(native.model.layers, original))


def test_wrong_head_profile_rejected_before_partial_installation():
    native, _ = fixture()
    original = [layer.self_attn for layer in native.model.layers]
    native.model.layers[-1].self_attn.is_kv_shared_layer = False
    with pytest.raises(ValueError, match="exclusively shared"):
        install_assistant_attention(native, enabled=True)
    assert all(layer.self_attn is old for layer, old in zip(native.model.layers, original))

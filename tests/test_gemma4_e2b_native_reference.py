"""Explicit native whole-HF oracle policy, independent of candidate prompt routing."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_target import E2BTarget  # noqa: E402 - optional runtime admitted above


def target(policy, model=None):
    owner = object.__new__(E2BTarget)
    owner._closed = False
    owner._row_policy = policy
    owner.model = object() if model is None else model
    return owner


def test_default_prompt_fusion_remains_disabled():
    assert E2BRowPolicy().fuse_prompt_norms is False


def test_nested_reference_scopes_restore_enclosing_mode_and_failure():
    policy = E2BRowPolicy(fuse_prompt_norms=True)
    policy.prompt, policy.position = False, 42
    owner = target(policy)
    failure = RuntimeError("golden failed")
    with pytest.raises(RuntimeError) as caught:
        with owner.native_reference() as model:
            assert model is owner.model and policy.prompt and policy.reference_depth == 1
            policy.position = 11
            with pytest.raises(ValueError):
                with owner.native_reference():
                    assert policy.reference_depth == 2
                    policy.position = 23
                    raise ValueError("inner golden failed")
            assert (policy.prompt, policy.position, policy.reference_depth) == (True, 11, 1)
            raise failure
    assert caught.value is failure
    assert (policy.prompt, policy.position, policy.reference_depth) == (False, 42, 0)
    assert policy.fuse_prompt_norms is True


def test_reference_rejects_target_adapter_forward_and_closed_owner():
    policy = E2BRowPolicy(fuse_prompt_norms=True)
    owner = target(policy)
    with owner.native_reference():
        with pytest.raises(RuntimeError, match="direct whole-HF"):
            owner.forward([1], object())
        assert policy.prompt and policy.reference_depth == 1
    owner._closed = True
    with pytest.raises(RuntimeError, match="closed"):
        with owner.native_reference():
            pytest.fail("closed native reference entered")
    assert policy.reference_depth == 0


def test_public_whole_hf_golden_is_native_when_prompt_fusion_enabled(monkeypatch):
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from tensorfold.families.gemma4.cuda import e2b_attention, forward
    from tensorfold.families.gemma4.cuda.e2b_norm import install_row_norms
    cfg = Gemma4TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                          num_attention_heads=2, num_key_value_heads=1, head_dim=4,
                          global_head_dim=8, num_kv_shared_layers=0, sliding_window=8,
                          hidden_size_per_layer_input=4, vocab_size_per_layer_input=64,
                          vocab_size=64, max_position_embeddings=128, enable_moe_block=False,
                          attention_k_eq_v=False, layer_types=["sliding_attention", "full_attention"])
    model = Gemma4ForCausalLM(cfg).eval()
    ids = torch.tensor([[2, 7, 11, 19]])
    with torch.inference_mode():
        golden = model(input_ids=ids, use_cache=False).logits
    policy = E2BRowPolicy(fuse_prompt_norms=True)
    policy.prompt = False
    owner = target(policy, model)
    install_row_norms(model.model, policy)
    e2b_attention.install_attention(model.model, policy)
    monkeypatch.setattr(forward, "ct_rms_norm", lambda *_: pytest.fail("golden called fused norm"))
    monkeypatch.setattr(e2b_attention, "chronological_attention", lambda *_, **__: pytest.fail("golden called custom attention"))
    with torch.inference_mode(), owner.native_reference() as oracle:
        actual = oracle(input_ids=ids, use_cache=False).logits
    assert torch.equal(actual, golden)
    assert not policy.prompt and policy.reference_depth == 0


@pytest.mark.parametrize("value", [1, None, "true"])
def test_candidate_flag_requires_boolean(value):
    with pytest.raises(ValueError, match="boolean"):
        E2BRowPolicy(fuse_prompt_norms=value)


@pytest.mark.parametrize("value", [1, None, "true"])
def test_engine_rejects_invalid_flag_before_gpu(monkeypatch, tmp_path, value):
    from tensorfold.families.gemma4.cuda.e2b_engine import E2BEngine
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("GPU touched before option validation"))
    with pytest.raises(ValueError, match="boolean"):
        E2BEngine(tmp_path, fuse_prompt_norms=value)

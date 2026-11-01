"""Public SDPA GQA selection preserves native HF prompt masks and KV ownership."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402 - optional runtime admitted above
from tensorfold.families.gemma4.cuda.e2b_prompt_sdpa import prompt_sdpa_gqa  # noqa: E402 - optional runtime admitted above


@pytest.mark.parametrize("rows,keys,masked", [(1, 7, False), (2, 7, True), (7, 7, False), (2, 7, False)])
def test_native_hf_wrapper_agrees_for_causal_cached_and_single_query_regions(rows, keys, masked):
    q = torch.zeros(1, 8, rows, 4)
    k = torch.zeros(1, 2, keys, 4)
    v = torch.arange(keys).float().view(1, 1, keys, 1).expand(1, 2, keys, 4).clone()
    v[:, 1] += 10
    module = SimpleNamespace(is_causal=True, num_key_value_groups=4)
    mask = None
    if masked:
        mask = (torch.arange(keys)[None, :] <= torch.arange(keys-rows, keys)[:, None])[None, None]
    actual = prompt_sdpa_gqa(module, q, k, v, mask)
    native = sdpa_attention_forward(module, q, k, v, mask, dropout=0.0, scaling=1.0)[0]
    assert torch.equal(actual, native)


def test_independent_one_hot_lower_right_group_mapping_has_no_rounding_tolerance():
    rows, keys = 3, 7
    q, k = torch.zeros(1, 8, rows, 4), torch.zeros(1, 2, keys, 4)
    v = torch.arange(keys).float().view(1, 1, keys, 1).expand(1, 2, keys, 4).clone()
    v[:, 1] += 10
    mask = torch.zeros(1, 1, rows, keys, dtype=torch.bool)
    for row in range(rows):
        mask[:, :, row, keys-rows+row] = True
    actual = prompt_sdpa_gqa(SimpleNamespace(is_causal=True), q, k, v, mask)
    expected = torch.arange(keys-rows, keys).float()
    assert torch.equal(actual[0, :, :4, 0], expected[:, None].expand(rows, 4))
    assert torch.equal(actual[0, :, 4:, 0], (expected+10)[:, None].expand(rows, 4))


@pytest.mark.parametrize("mask_kind", ["none", "causal", "padding", "fully-masked"])
def test_candidate_keeps_kv_heads_mask_scale_dropout_and_explicit_causality(monkeypatch, mask_kind):
    q = torch.empty(1, 8, 3, 4)
    k, v = torch.empty(1, 2, 7, 4), torch.empty(1, 2, 7, 4)
    mask = None if mask_kind == "none" else torch.ones(1, 1, 3, 7, dtype=torch.bool)
    if mask_kind == "padding":
        mask[..., 1] = False
    if mask_kind == "fully-masked":
        mask.zero_()
    calls = []

    def sdpa(query, key, value, **kwargs):
        calls.append((query, key, value, kwargs))
        return torch.zeros_like(query)

    monkeypatch.setattr(torch.nn.functional, "scaled_dot_product_attention", sdpa)
    actual = prompt_sdpa_gqa(SimpleNamespace(is_causal=True), q, k, v, mask,
                             dropout=.125, scaling=1.0, is_causal=False)
    query, key, value, kwargs = calls[0]
    assert query is q and key is k and value is v  # no repeat_kv, copy, or upper-left crop
    assert key.shape[1] == value.shape[1] == 2
    assert kwargs == {"attn_mask": mask, "dropout_p": .125, "scale": 1.0,
                      "is_causal": False, "enable_gqa": True}
    assert actual.shape == (1, 3, 8, 4) and actual.is_contiguous()


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_prompt_gqa_selection_requires_boolean_before_engine_gpu_use(monkeypatch, tmp_path, value):
    from tensorfold.families.gemma4.cuda.e2b_engine import E2BEngine
    from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy
    with pytest.raises(ValueError, match="boolean"):
        E2BRowPolicy(prompt_sdpa_gqa=value)
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("invalid flag reached GPU"))
    with pytest.raises(ValueError, match="boolean"):
        E2BEngine(tmp_path, prompt_sdpa_gqa=value)


def test_prompt_gqa_default_is_disabled():
    from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy
    assert E2BRowPolicy().prompt_sdpa_gqa is False


def tiny_models(kv_heads):
    import copy
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from tensorfold.families.gemma4.cuda.e2b_attention import install_attention
    from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy
    cfg = Gemma4TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=42,
                          num_attention_heads=8, num_key_value_heads=kv_heads, head_dim=4,
                          global_head_dim=8, num_kv_shared_layers=18, sliding_window=8,
                          hidden_size_per_layer_input=4, vocab_size_per_layer_input=64,
                          vocab_size=64, max_position_embeddings=128, enable_moe_block=False,
                          use_double_wide_mlp=False, attention_k_eq_v=False,
                          final_logit_softcapping=30.0,
                          layer_types=["full_attention" if (i+1)%6 == 0 else "sliding_attention" for i in range(42)])
    cfg._attn_implementation = "sdpa"
    native = Gemma4ForCausalLM(cfg).to(torch.bfloat16).eval()
    candidate = copy.deepcopy(native)
    policy = E2BRowPolicy(prompt_sdpa_gqa=True)
    install_attention(candidate.model, policy)
    return native, candidate, policy


@pytest.mark.parametrize("kv_heads", [1, 2])
@pytest.mark.parametrize("padding", [False, True])
def test_tiny_42_layer_public_hf_cached_prompt_matches_native_masks_and_shared_kv(kv_heads, padding):
    native, candidate, policy = tiny_models(kv_heads)
    ids = torch.tensor([[2, 7, 11, 19, 4, 8, 12, 6, 15, 22, 23, 9]])
    mask = torch.ones_like(ids, dtype=torch.bool)
    if padding:
        mask[:, 1] = False
    a_cache = b_cache = None
    begin = 0
    with torch.inference_mode():
        for end in (7, 11, 12):
            inputs = {"input_ids": ids[:, begin:end], "attention_mask": mask[:, :end],
                      "use_cache": True, "return_dict": True, "return_shared_kv_states": True}
            a = native(**inputs, past_key_values=a_cache)
            b = candidate(**inputs, past_key_values=b_cache)
            assert policy.prompt and policy.prompt_sdpa_gqa
            assert torch.equal(a.logits, b.logits)
            a_cache, b_cache = a.past_key_values, b.past_key_values
            assert len(a_cache.layers) == len(b_cache.layers) == 24
            for ca, cb in zip(a_cache.layers, b_cache.layers):
                assert torch.equal(ca.keys, cb.keys) and torch.equal(ca.values, cb.values)
            begin = end


def test_candidate_supports_uncached_public_hf_forward():
    native, candidate, _ = tiny_models(2)
    ids = torch.tensor([[2, 7, 11, 19]])
    with torch.inference_mode():
        assert torch.equal(native(input_ids=ids, use_cache=False).logits,
                           candidate(input_ids=ids, use_cache=False).logits)


def test_default_and_nested_native_reference_never_enter_gqa_candidate(monkeypatch):
    from tensorfold.families.gemma4.cuda import e2b_prompt_sdpa
    from tensorfold.families.gemma4.cuda import forward
    from tensorfold.families.gemma4.cuda.e2b_norm import install_row_norms
    from tensorfold.families.gemma4.cuda.e2b_target import E2BTarget
    native, candidate, policy = tiny_models(2)
    install_row_norms(candidate.model, policy)
    owner = object.__new__(E2BTarget)
    owner._closed, owner.model, owner._row_policy = False, candidate, policy
    ids = torch.tensor([[2, 7, 11, 19]])
    monkeypatch.setattr(e2b_prompt_sdpa, "prompt_sdpa_gqa", lambda *_, **__: pytest.fail("native oracle entered GQA candidate"))
    monkeypatch.setattr(forward, "ct_rms_norm", lambda *_: pytest.fail("native oracle entered fused norm candidate"))
    with torch.inference_mode():
        expected = native(input_ids=ids, use_cache=False).logits
        policy.prompt_sdpa_gqa = False
        assert torch.equal(candidate(input_ids=ids, use_cache=False).logits, expected)
        policy.prompt_sdpa_gqa = True
        policy.fuse_prompt_norms = True
        policy.prompt, policy.position = False, 42
        with owner.native_reference() as oracle:
            assert policy.prompt and policy.reference_depth == 1
            assert torch.equal(oracle(input_ids=ids, use_cache=False).logits, expected)
            with pytest.raises(ValueError):
                with owner.native_reference():
                    assert torch.equal(oracle(input_ids=ids, use_cache=False).logits, expected)
                    raise ValueError("nested oracle failed")
            assert policy.reference_depth == 1
    assert not policy.prompt and policy.position == 42 and policy.reference_depth == 0
    assert policy.prompt_sdpa_gqa is True
    assert policy.fuse_prompt_norms is True


def test_decode_does_not_read_prompt_gqa_or_reference_fields(monkeypatch):
    from transformers import DynamicCache, Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention
    from tensorfold.families.gemma4.cuda import e2b_attention, e2b_prompt_sdpa
    accesses = []

    class DecodePolicy:
        @property
        def prompt(self):
            accesses.append("prompt")
            return False

        position = 0
        attention_query_rows = 2

        def __getattr__(self, name):
            pytest.fail("decode read prompt-only field " + name)

    cfg = Gemma4TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=2,
                          num_attention_heads=8, num_key_value_heads=2, head_dim=4,
                          global_head_dim=8, num_kv_shared_layers=0, sliding_window=8,
                          layer_types=["sliding_attention", "full_attention"])
    native = Gemma4TextAttention(cfg, 0).to(torch.bfloat16).eval()
    candidate = e2b_attention.E2BTextAttention(native, DecodePolicy())
    calls = []

    def canonical(query, key, value, mask, *, start, window, query_rows):
        assert query_rows == 2
        calls.append((start, window))
        return query.transpose(1, 2).contiguous()

    monkeypatch.setattr(e2b_attention, "chronological_attention", canonical)
    monkeypatch.setattr(e2b_prompt_sdpa, "prompt_sdpa_gqa", lambda *_, **__: pytest.fail("decode entered prompt provider"))
    with torch.inference_mode():
        candidate(torch.randn(1, 1, 16, dtype=torch.bfloat16),
                  (torch.ones(1, 1, 4, dtype=torch.bfloat16), torch.zeros(1, 1, 4, dtype=torch.bfloat16)),
                  None, {}, past_key_values=DynamicCache(config=cfg))
    assert accesses == ["prompt"] and calls == [(0, 8)]

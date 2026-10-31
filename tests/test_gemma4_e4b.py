"""Exact E4B architecture/dispatch and owned HF glue, without model weights."""
import copy
import json
from types import SimpleNamespace
import pytest

from tensorfold.families.gemma4.cuda.e2b_metadata import E2BConfig, is_dense_gemma, is_e2b, is_e4b
from test_gemma4_e2b_metadata import e2b_config


def e4b_config():
    raw = e2b_config()
    text = raw["text_config"]
    text.update(hidden_size=2560, num_hidden_layers=42, num_key_value_heads=2,
                intermediate_size=10240, num_kv_shared_layers=18, use_double_wide_mlp=False,
                num_global_key_value_heads=None,
                layer_types=["full_attention" if (i+1) % 6 == 0 else "sliding_attention" for i in range(42)])
    # Actual E4B checkpoint uses the official torch_dtype spelling.
    raw["torch_dtype"] = raw.pop("dtype")
    text["torch_dtype"] = "bfloat16"
    return raw


def test_exact_profiles_keep_geometry_tap_and_cache_contracts_distinct():
    raw = e4b_config()
    before = copy.deepcopy(raw)
    cfg = E2BConfig.from_dict(raw)
    assert raw == before
    assert is_e4b(raw) and is_dense_gemma(raw) and not is_e2b(raw)
    assert (cfg.profile, cfg.hidden, cfg.layers, cfg.kv_heads, cfg.shared_layers) == ("E4B", 2560, 42, 2, 18)
    assert cfg.tap_layers == ()
    assert cfg.donor_layers.count("full_attention") == 4
    assert cfg.donor_layers.count("sliding_attention") == 20
    assert cfg.cache_bytes(131072, 8) == 4*131072*2*512*4 + 20*(511+8)*2*256*4
    e2b = E2BConfig.from_dict(e2b_config())
    assert e2b.profile == "E2B" and e2b.tap_layers == (2, 12, 22, 32)


@pytest.mark.parametrize("key,value", [("num_key_value_heads", 1), ("num_global_key_value_heads", 1),
                                       ("num_global_key_value_heads", True), ("num_kv_shared_layers", 20),
                                       ("use_double_wide_mlp", True), ("intermediate_size", 6144)])
def test_e4b_cannot_be_misread_as_e2b(key, value):
    raw = e4b_config()
    raw["text_config"][key] = value
    with pytest.raises(ValueError):
        E2BConfig.from_dict(raw)


def test_family_routes_e4b_dense_with_no_auto_or_explicit_incompatible_drafter(tmp_path, monkeypatch):
    from tensorfold import families
    from tensorfold.families import gemma4
    from tensorfold.families.gemma4.cuda import e2b_engine
    raw = e4b_config()
    (tmp_path/"config.json").write_text(json.dumps(raw))
    family = families.Family("gemma4", "Gemma 4", "tensorfold.families.gemma4", True)
    families.require_readable(family, raw, "cuda")
    gemma4.check(tmp_path)
    assert gemma4.cuda_drafter(tmp_path) == ""
    with pytest.raises(ValueError):
        families.require_readable(family, raw, "mlx")
    seen = []
    monkeypatch.setattr(e2b_engine, "E2BEngine", lambda *args, **kwargs: seen.append((args, kwargs)) or "dense-engine")
    assert gemma4.cuda_engine(tmp_path) == "dense-engine"
    assert seen[0][1]["drafter"] == ""


@pytest.mark.parametrize("disabled", [False, True])
def test_explicit_e4b_drafter_is_rejected_before_any_gpu_or_weight_allocation(tmp_path, monkeypatch, disabled):
    torch = pytest.importorskip("torch")
    from tensorfold.families.gemma4.cuda import e2b_engine
    (tmp_path/"config.json").write_text(json.dumps(e4b_config()))
    monkeypatch.setattr(torch.cuda, "set_device", lambda _: pytest.fail("E4B drafter reached GPU"))
    monkeypatch.setattr(e2b_engine, "E2BTarget", lambda *_: pytest.fail("E4B drafter loaded weights"))
    with pytest.raises(ValueError, match="no admitted compatible DFlash"):
        e2b_engine.E2BEngine(tmp_path, drafter="incompatible-E2B", no_drafts=disabled)


def test_native_hf_e4b_full_kv_heads_retain_configured_two_heads():
    pytest.importorskip("torch")
    from transformers import Gemma4Config
    native = Gemma4Config.from_dict(e4b_config()).text_config
    assert [(native.per_layer_config[i].num_key_value_heads, native.per_layer_config[i].head_dim)
            for i in (0, 5, 23, 24, 41)] == [(2, 256), (2, 512), (2, 512), (2, 256), (2, 512)]


def test_cli_vision_preflight_routes_dense_e4b_clipped_towers_to_native_adapter():
    from tensorfold.vision.config import validate_vision_config
    raw = e4b_config()
    raw["vision_config"]["use_clipped_linears"] = True
    assert validate_vision_config(raw, "gemma4") is raw["vision_config"]
    raw["text_config"]["num_key_value_heads"] = 1
    with pytest.raises(ValueError):
        validate_vision_config(raw, "gemma4")


def test_e4b_target_does_not_publish_invented_speculative_taps():
    from tensorfold.families.gemma4.cuda.e2b_target import E2BTarget
    owner = object.__new__(E2BTarget)
    owner._closed = False
    owner._row_policy = SimpleNamespace(reference_depth=0)
    owner.tap_layers = ()
    with pytest.raises(ValueError, match="no admitted speculative"):
        owner.forward([1], object(), output_taps=True)


def test_e4b_42_layer_kv2_owned_glue_matches_native_hf_and_rolls_back_shared_caches(monkeypatch):
    torch = pytest.importorskip("torch")
    pytest.importorskip("triton")
    from transformers import Gemma4ForCausalLM, Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import eager_attention_forward
    from tensorfold.families.gemma4.cuda import e2b_attention
    from tensorfold.families.gemma4.cuda.e2b_rows import E2BRowPolicy
    from tensorfold.families.gemma4.cuda.e2b_target import E2BState
    cfg = Gemma4TextConfig(hidden_size=16, intermediate_size=32, num_hidden_layers=42,
                          num_attention_heads=8, num_key_value_heads=2, head_dim=4,
                          global_head_dim=8, num_kv_shared_layers=18, sliding_window=8,
                          hidden_size_per_layer_input=4, vocab_size_per_layer_input=64,
                          vocab_size=64, max_position_embeddings=128, enable_moe_block=False,
                          use_double_wide_mlp=False, attention_k_eq_v=False,
                          layer_types=["full_attention" if (i+1) % 6 == 0 else "sliding_attention" for i in range(42)])
    cfg._attn_implementation = "eager"
    native = Gemma4ForCausalLM(cfg).to(torch.bfloat16).eval()
    owned = copy.deepcopy(native)
    policy = E2BRowPolicy()
    e2b_attention.install_attention(owned.model, policy)

    def independent_core(query, key, value, mask, *, start, window):
        assert key.shape[1] == value.shape[1] == 2
        module = SimpleNamespace(head_dim=query.shape[-1], num_key_value_groups=4, training=False)
        return eager_attention_forward(module, query, key, value, mask, scaling=1.0)[0]

    monkeypatch.setattr(e2b_attention, "chronological_attention", independent_core)
    one, two = E2BState(cfg, 128), E2BState(cfg, 128)
    prompt = torch.tensor([[2, 7, 11, 19, 4, 6, 8, 9, 12]])
    with torch.inference_mode():
        native(input_ids=prompt, past_key_values=one.cache, use_cache=True)
        owned(input_ids=prompt, past_key_values=two.cache, use_cache=True)
        one.position = two.position = prompt.shape[1]
        one.rollback(one.position)
        two.rollback(two.position)
        checkpoint = two.snapshot()
        policy.prompt, policy.position = False, two.position
        teacher = torch.tensor([[13, 14, 15]])
        expected = native(input_ids=teacher, past_key_values=one.cache, use_cache=True).logits
        actual = owned(input_ids=teacher, past_key_values=two.cache, use_cache=True).logits
    assert torch.equal(actual, expected)
    assert len(two.cache.layers) == 24  # donors only;18 shared tail layers reuse donor KV
    assert all(layer.keys.shape[1] == 2 for layer in two.cache.layers)
    one.position += 3
    two.position += 3
    one.rollback(10)
    two.rollback(10)
    for a, b in zip(one.cache.layers, two.cache.layers):
        assert torch.equal(a.keys, b.keys) and torch.equal(a.values, b.values)
    two.restore(checkpoint)
    assert two.position == 9 and two.cache.get_seq_length() == 9

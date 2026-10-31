"""Owned native projection/RoPE/cache glue; GPU kernel tests run on the actual target."""
from types import SimpleNamespace
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.families.gemma4.cuda import e2b_attention  # noqa: E402 - optional runtime admitted above


def test_owned_attention_preserves_native_glue_and_loaded_parameter_identity(monkeypatch):
    from transformers import DynamicCache, Gemma4TextConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention, eager_attention_forward
    cfg = Gemma4TextConfig(hidden_size=32, num_attention_heads=2, num_key_value_heads=1,
                          head_dim=4, global_head_dim=8, num_hidden_layers=4, num_kv_shared_layers=2,
                          layer_types=["sliding_attention", "full_attention"] * 2,
                          sliding_window=8, attention_k_eq_v=False, attention_dropout=0)
    cfg._attn_implementation = "eager"
    native = Gemma4TextAttention(cfg, 0).to(torch.bfloat16).eval()
    owned = e2b_attention.E2BTextAttention(native, SimpleNamespace(prompt=False, position=0))
    assert owned.q_proj.weight is native.q_proj.weight
    assert owned.state_dict().keys() == native.state_dict().keys()
    assert owned.training is native.training
    calls = []

    def reference(query, key, value, mask, *, start, window):
        calls.append((start, window))
        return eager_attention_forward(native, query, key, value, mask, scaling=1.0)[0]

    monkeypatch.setattr(e2b_attention, "chronological_attention", reference)
    hidden = torch.randn(1, 3, 32, dtype=torch.bfloat16)
    position = (torch.ones(1, 3, 4, dtype=torch.bfloat16), torch.zeros(1, 3, 4, dtype=torch.bfloat16))
    mask = torch.triu(torch.full((1, 1, 3, 3), float("-inf")), diagonal=1)
    with torch.inference_mode():
        expected = native(hidden, position, mask, {}, past_key_values=DynamicCache(config=cfg))[0]
        actual = owned(hidden, position, mask, {}, past_key_values=DynamicCache(config=cfg))[0]
    assert torch.equal(actual, expected)
    assert calls == [(0, 8)]

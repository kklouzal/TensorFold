"""Decode preparation opts into serial geometry while prompt controls stay unchanged."""
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.torch


@pytest.mark.parametrize("prefill", [False, True])
def test_attention_forward_selects_normalization_geometry_by_phase(monkeypatch, prefill):
    from tensorfold.families.qwen4_exp.cuda import forward as module

    rows, heads, kv, dim, ih, iw = 6, 3, 1, 256, 4, 128
    cfg = SimpleNamespace(heads=heads, kv_heads=kv, head_dim=dim, index_heads=ih, index_dim=iw,
                          eps=1e-6, rope_attention_factor=1.)
    w = SimpleNamespace(cfg=cfg, inv_freq=torch.ones(32))
    attention = SimpleNamespace(proj=object(), o=object(), q_scale=torch.ones(dim), k_scale=torch.ones(dim),
                                iq_scale=torch.ones(iw), ik_scale=torch.ones(iw))
    layer = SimpleNamespace(index=0, attn=attention)
    b = SimpleNamespace(prefill=prefill, h=torch.empty(rows, 10240, dtype=torch.float16),
                        mixed=torch.empty(rows, 2560), xs_mixed=torch.empty(rows, 80), pa=torch.empty(rows, 1),
                        q=torch.empty(rows, heads, dim), iq=torch.empty(rows, ih, iw),
                        attn=SimpleNamespace(qsa=False, centroids=None, thresholds=None),
                        pos_blk=torch.zeros(1, dtype=torch.int32), attn_o=torch.empty(rows, heads, dim),
                        gated=torch.empty(rows, heads * dim), xs_gated=torch.empty(rows, heads * dim // 32), rotor_lookup=False)
    st = SimpleNamespace(image_positions=None, rope_delta=0, kv_key_bits=0, kv_key_codec=0,
                         kv_value_bits=0, kv_value_codec=0, kv_status=None)
    cache = SimpleNamespace(k=torch.empty(rows, kv, dim), v=torch.empty(rows, kv, dim), ks=None, vs=None)
    selected = []
    marker = object()
    monkeypatch.setattr(module, "_mm", lambda *a, **kw: None)
    monkeypatch.setattr(module, "_caches", lambda *a: (cache, torch.empty(rows, iw), None, torch.zeros(1, dtype=torch.int32), 0))
    monkeypatch.setattr(module.glue, "attn_prep", lambda *a, **kw: selected.append(kw["independent_rows"]))
    monkeypatch.setattr(module.glue, "attn_gate", lambda *a, **kw: None)
    monkeypatch.setattr(module.attn_mod, "attention", lambda *a, **kw: b.attn_o)
    monkeypatch.setattr(module, "_out_proj", lambda *a: marker)
    assert module.attn_block(layer, w, [(st, 0, rows)], b, rows, False) is marker
    assert selected == [not prefill]


@pytest.mark.parametrize("independent", [False, True])
def test_half_glue_forwards_explicit_geometry_to_reference_provider(monkeypatch, independent):
    from tensorfold.families.qwen4_exp.cuda import glue

    calls = []
    marker = object()
    monkeypatch.setattr(glue._half_ops, "attn_prep", lambda *a, **kw: calls.append(kw) or marker)
    assert glue.attn_prep(SimpleNamespace(dtype=torch.float16), *([None] * 11), q_heads=3, kv_heads=1,
                          head_dim=256, index_heads=4, index_dim=128, independent_rows=independent) is marker
    assert calls[0]["independent_rows"] is independent

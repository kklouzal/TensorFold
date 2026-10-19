"""Qwen3.6 static YaRN wiring without a GPU: the loaders forward the policy and the MTP layer rotates with its
amplitude (the shared forward's rope_scale), as the target's attention does."""

from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")

from tensorfold.families.qwen3_5_moe.cuda import mtp, weights  # noqa: E402


@pytest.mark.parametrize("quantized", [True, False])
def test_load_forwards_the_rope_policy_to_either_loader(monkeypatch, quantized):
    from tensorfold.families.qwen3_5.cuda import nvfp4_load as backbone
    from tensorfold.families.qwen3_5_moe.cuda import nvfp4_load

    got = []
    rope = object()
    monkeypatch.setattr(backbone, "quantized", lambda path: quantized)
    monkeypatch.setattr(nvfp4_load, "load_nvfp4", lambda *a, **k: got.append(("nvfp4", k)))
    monkeypatch.setattr(weights, "load_dense", lambda *a, **k: got.append(("dense", k)))
    weights.load("model", rope=rope, eos_token_ids=(1,))
    weights.load("model")
    (kind, first), (_, second) = got
    assert kind == ("nvfp4" if quantized else "dense")
    assert first["rope"] is rope and first["eos_token_ids"] == (1,) and "rope" not in second


def test_the_mtp_layer_rotates_with_the_configured_amplitude(monkeypatch):
    seen = {}

    class Stop(Exception):
        pass

    def attn_prep(*args, **kwargs):
        seen.update(kwargs)
        raise Stop

    rows = torch.zeros(2, 4)
    glue = NS(embed=lambda *a: rows, add_rmsnorm=lambda x, r, w, eps: (x, x, x), attn_prep=attn_prep)
    monkeypatch.setattr(mtp, "glue", glue)
    monkeypatch.setattr(mtp, "matmul", lambda x, w, xs=None: torch.zeros(x.shape[0], 8))
    head = object.__new__(mtp.Head)
    attn = NS(q=None, k=None, v=None, o=None, q_norm=None, k_norm=None)
    head.w = NS(config=NS(hidden=4, eps=1e-6, heads=1, kv_heads=1, head_dim=8, rope_scale=1.25),
                embed=NS(weight=None, scales=None, biases=None), inv_freq=None)
    head.m = NS(norm_e=None, norm_h=None, fc_e=None, fc_h=None, input_norm=None, attn=attn)
    with pytest.raises(Stop):
        head._layer(rows, torch.zeros(2, dtype=torch.int32), torch.zeros(2, dtype=torch.int64), lambda *a: None)
    assert seen["amplitude"] == 1.25

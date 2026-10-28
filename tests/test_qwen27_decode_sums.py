"""Serving-layout selection preserves affine consumers and raw forward defaults."""

from types import SimpleNamespace as NS

import pytest

from tensorfold.families.qwen3_5.cuda.forward import prepare_decode
from tensorfold.families.qwen3_5.cuda.weights import QLinear


def _layout():
    native = object()
    gdn = NS(qkv=native, z=native, b=native, a=native, zba=None, out=native)
    attn = NS(q=native, k=native, v=native, kv=None, o=native)
    dense = NS(linear=True, gdn=gdn, gate=native, up=native, down=native, moe=None)
    routed = NS(linear=False, attn=attn, gate=None, up=None, down=None, moe=object())
    return NS(quant="nvfp4", head=native, layers=[dense, routed]), gdn, attn


@pytest.mark.parametrize("role", ["head", "qkv", "z", "b", "a", "out", "q", "k", "v", "o",
                                  "gate", "up", "down", "zba", "kv"])
def test_each_active_affine_consumer_keeps_sums_and_reprepare_tracks_replacement(role):
    w, gdn, attn = _layout()
    assert not hasattr(w, "decode_sums")  # direct forwards retain their default until serving preparation
    prepare_decode(w)
    assert w.decode_sums is False
    owner = w if role == "head" else gdn if role in ("qkv", "z", "b", "a", "out", "zba") else (
        attn if role in ("q", "k", "v", "o", "kv") else w.layers[0])
    original = getattr(owner, role)
    setattr(owner, role, QLinear(None, None, None))
    prepare_decode(w)
    assert w.decode_sums is True
    setattr(owner, role, original)
    prepare_decode(w)
    assert w.decode_sums is False


def test_fused_consumers_and_routed_mlp_do_not_select_unused_affine_fields():
    w, gdn, attn = _layout()
    gdn.zba, attn.kv = object(), object()
    gdn.z = gdn.b = gdn.a = attn.k = attn.v = QLinear(None, None, None)
    w.layers[1].gate = w.layers[1].up = w.layers[1].down = QLinear(None, None, None)
    prepare_decode(w)
    assert w.decode_sums is False


@pytest.mark.parametrize("quant", ["mlx", "exl3", None])
def test_other_layouts_keep_original_default_even_after_repreparation(quant):
    w, _, _ = _layout()
    prepare_decode(w)
    w.quant = quant
    prepare_decode(w)
    assert w.decode_sums is True

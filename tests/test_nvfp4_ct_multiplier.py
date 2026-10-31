"""Original CT multiplier propagation; legacy direct scales keep their own arithmetic."""
from types import SimpleNamespace as NS

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")

from tensorfold.cuda.nvfp4 import checkpoint as ck


def test_explicit_multiplier_avoids_reciprocal_reconstruction(monkeypatch):
    calls = []
    ext = NS(quant4=lambda x, inverse, codes, scales, tb: calls.append((inverse, tb)))
    monkeypatch.setattr(ck, "_ext", lambda: ext)
    raw = float(torch.tensor(0x3E63FFFF, dtype=torch.int32).view(torch.float32))
    act = float(torch.tensor(1., dtype=torch.float32) / torch.tensor(raw))
    assert ck._inv(act) != raw
    x = torch.full((1, 64), .5, dtype=torch.bfloat16)
    ck.quant4(x, act, inverse=raw)
    ck.quant4(x, act)
    assert calls == [(raw, 0), (ck._inv(act), 0)]


@pytest.mark.parametrize("prompt", [False, True])
def test_projection_dispatch_keeps_the_original_multiplier(monkeypatch, prompt):
    calls = []
    monkeypatch.setattr(ck, "_out", lambda m, lin, out, f32: torch.empty(m, 1))
    monkeypatch.setattr(ck, "_rowsq", lambda *a, **kw: calls.append(kw["inverse"]) or ())
    monkeypatch.setattr(ck, "_lane", lambda *a: None)
    monkeypatch.setattr(ck, "_gemm", lambda *a: None)
    monkeypatch.setattr(ck, "_capability", lambda: (12, 0))
    projection = NS(act=.5, act_inverse=2.000000238418579, scale=.125)
    operator = ck.prompt if prompt else ck.matmul
    operator(ck.A4, torch.ones(1, 64, dtype=torch.bfloat16), projection)
    assert calls == [projection.act_inverse]


def test_equal_rounded_scales_cannot_share_distinct_original_multipliers():
    left = NS(words=True, act=.5, act_inverse=2.)
    right = NS(words=True, act=.5, act_inverse=2.000000238418579)
    assert ck.matmul_group(torch.ones(1, 64), [left, right]) is None
    assert ck.mlp_prompt(torch.ones(1, 64), left, right, NS(words=True, act=.5)) is None


def test_fused_down_quantizer_receives_original_multiplier(monkeypatch):
    values = []
    rows = ck.Rows4(torch.empty(1, 64, dtype=torch.uint8), torch.empty(2, 64, 4, dtype=torch.uint8))
    monkeypatch.setattr(ck, "quant4", lambda *a, **kw: values.append(("up", kw["inverse"])) or rows)
    monkeypatch.setattr(ck, "_ext", lambda: NS(gemm_gu_ck=lambda *a: values.append(("down", a[10]))))
    monkeypatch.setattr(ck, "_out", lambda *a: torch.empty(1, 128))
    monkeypatch.setattr(ck, "_gemm", lambda *a: None)
    gate = NS(words=True, bs=True, act=.5, act_inverse=2.000000238418579, scale=.125, n=128, npad=128, k=128)
    down = NS(words=True, act=.25, act_inverse=4.000000476837158, scale=.5, k=128)
    ck.mlp_prompt(torch.ones(1, 128), gate, gate, down)
    assert values == [("up", gate.act_inverse), ("down", down.act_inverse)]

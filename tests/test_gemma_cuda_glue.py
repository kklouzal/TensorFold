"""Fused Gemma4 layer glue equals the unfused norm/add/scale chain (BF16 boundaries kept; FP32 reduction order may
move a value by one BF16 ulp, rarely) and is row-count invariant."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("glue kernels run on CUDA", allow_module_level=True)

import torch.nn.functional as F  # noqa: E402

from tensorfold.families.gemma4.cuda import glue  # noqa: E402
from tensorfold.families.gemma4.cuda.forward import ct_rms_norm, rms_norm  # noqa: E402

DEV = torch.device("cuda", 0)
EPS = 1e-6


def rnd(*shape, seed, scale=1.0, shift=0.0):
    g = torch.Generator().manual_seed(seed)
    return (torch.randn(*shape, generator=g) * scale + shift).to(torch.bfloat16).to(DEV)


def ulps(a, b):
    return (a.view(torch.int16).int() - b.view(torch.int16).int()).abs()


def close(actual, expected):
    diff = ulps(actual, expected)
    assert diff.max() <= 1 and (diff > 0).float().mean() < 0.01


@pytest.mark.parametrize("mlx", [False, True])
def test_attention_tail_equals_unfused(mlx):
    norm = (lambda x, w: rms_norm(x, w, EPS)) if mlx else (lambda x, w: ct_rms_norm(x, w, EPS))
    h, o = rnd(37, 2816, seed=1, scale=4), rnd(37, 2816, seed=2, scale=9)
    wa, wm, we = (rnd(2816, seed=s, scale=0.2, shift=1) for s in (3, 4, 5))
    got = glue.attention_tail(h, o, wa, wm, we, EPS, mlx)
    h2 = h + norm(o, wa)
    for a, b in zip(got, (h2, norm(h2, wm), norm(h2, we))):
        close(a, b)
    rows = [glue.attention_tail(h[i:i + 1], o[i:i + 1], wa, wm, we, EPS, mlx) for i in range(37)]
    assert all(torch.equal(torch.cat([r[j] for r in rows]), got[j]) for j in range(3))


@pytest.mark.parametrize("mlx", [False, True])
def test_moe_tail_equals_unfused(mlx):
    norm = (lambda x, w: rms_norm(x, w, EPS)) if mlx else (lambda x, w: ct_rms_norm(x, w, EPS))
    res, dense, routed = rnd(19, 2816, seed=6, scale=5), rnd(19, 2816, seed=7, scale=3), rnd(19, 2816, seed=8)
    wd, wr, wo, wn = (rnd(2816, seed=s, scale=0.2, shift=1) for s in (9, 10, 11, 12))
    scalar = torch.tensor([0.87], dtype=torch.bfloat16, device=DEV)
    h, nxt = glue.moe_tail(res, dense, routed, wd, wr, wo, scalar, wn, EPS, mlx)
    ref = (res + norm(norm(dense, wd) + norm(routed, wr), wo)) * scalar
    close(h, ref)
    close(nxt, norm(ref, wn))
    rows = [glue.moe_tail(res[i:i + 1], dense[i:i + 1], routed[i:i + 1], wd, wr, wo, scalar, wn, EPS, mlx)
            for i in range(19)]
    assert all(torch.equal(torch.cat([r[j] for r in rows]), (h, nxt)[j]) for j in range(2))


def test_geglu_equals_torch_bit_for_bit():
    gate, up = rnd(5, 2112, seed=13, scale=3), rnd(5, 2112, seed=14, scale=2)
    assert torch.equal(glue.geglu(gate, up), F.gelu(gate, approximate="tanh") * up)

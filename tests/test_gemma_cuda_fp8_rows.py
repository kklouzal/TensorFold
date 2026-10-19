"""FP8 row-scaled proposal weights: codes/scales reconstruct the weight within E4M3 rounding; the kernel equals the
float64 product of the reconstructed weight within FP32 accumulation; head mode softcaps."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available():
    pytest.skip("FP8 proposal rows run a CUDA kernel", allow_module_level=True)

from tensorfold.families.gemma4.cuda.fp8_rows import Fp8Rows  # noqa: E402

DEV = torch.device("cuda", 0)


def data(n=1000, k=384, m=37, seed=0):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(n, k, generator=g) * torch.rand(n, 1, generator=g)).to(torch.bfloat16).to(DEV)
    x = torch.randn(m, k, generator=g).to(torch.bfloat16).to(DEV)
    return w, x


def test_reconstruction_and_product_bounds():
    w, x = data()
    rows = Fp8Rows(w)
    shadow = rows.codes.double() * rows.scale.double()[:, None]
    # E4M3 keeps 3 mantissa bits: relative rounding <= 2^-4 of each element (normal range).
    big = w.double().abs() > rows.scale.double()[:, None] * 2.0 ** -6
    assert ((shadow - w.double()).abs()[big] <= w.double().abs()[big] * 2.0 ** -4).all()
    out = rows(x)
    ref = x.double() @ shadow.t()
    mass = x.double().abs() @ shadow.abs().t()
    assert ((out.double() - ref).abs() <= ref.abs() * 2.0 ** -8 + mass * 2.0 ** -20 + 1e-30).all()
    assert torch.equal(rows.prefix(100)(x), out[:, :100])
    assert torch.equal(torch.cat([rows(x[i:i + 5]) for i in range(0, 37, 5)]), out)


def test_head_mode_softcaps_in_fp32():
    w, x = data(m=4)
    head = Fp8Rows(w, softcap=30.0)
    logits = head(x)
    plain = Fp8Rows(w)
    ref = torch.tanh((x.double() @ (plain.codes.double() * plain.scale.double()[:, None]).t()) / 30) * 30
    assert logits.dtype == torch.float32 and (logits.double() - ref).abs().max() < 1e-3

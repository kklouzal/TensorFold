"""Gemma4 CT projection dispatch: grouped quantization keeps each projection's bits; prompt rows are row invariant."""

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("triton")
if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native CT A4 projections need SM12.x CUDA", allow_module_level=True)

from tensorfold.cuda.nvfp4.linear import Fp4Linear  # noqa: E402
from tensorfold.families.gemma4.cuda.forward import Gemma4Target  # noqa: E402

DEV = torch.device("cuda", 0)


def linear(n, k, seed, act=0.02, inverse=50.0):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 256, (n, k // 2), generator=g, dtype=torch.uint8).to(DEV)
    scales = torch.randint(0x30, 0x40, (n, k // 16), generator=g, dtype=torch.uint8).to(DEV)
    return Fp4Linear.from_checkpoint(codes, scales.view(torch.float8_e4m3fn), 0.001, act=act, act_inverse=inverse)


def dispatcher():
    target = Gemma4Target.__new__(Gemma4Target)
    target._prompt_gemm = target._group_projections = True
    return target


def test_grouped_projections_keep_each_projections_bits_on_both_paths():
    q, k, v = linear(512, 256, 1), linear(256, 256, 2), linear(256, 256, 3)
    x = (torch.randn(37, 256, generator=torch.Generator().manual_seed(4)) * 3).to(torch.bfloat16).to(DEV)
    target = dispatcher()
    for prompt in (False, True):
        grouped = target._project(x, (q, k, v), prompt)
        single = [lin.prefill(x) if prompt else lin(x) for lin in (q, k, v)]
        assert all(torch.equal(a, b) for a, b in zip(grouped, single))
    # Different original input multipliers cannot share one quantization: each projection runs alone.
    other = linear(256, 256, 5, inverse=50.000004)
    out = target._project(x, (q, other), True)
    assert torch.equal(out[1], other.prefill(x))


def test_prompt_rows_do_not_depend_on_the_chunk():
    q = linear(512, 256, 7)
    x = (torch.randn(300, 256, generator=torch.Generator().manual_seed(8)) * 3).to(torch.bfloat16).to(DEV)
    target = dispatcher()
    whole = target._project(x, (q,), True)[0]
    parts = torch.cat([target._project(x[i:i + n], (q,), True)[0] for i, n in ((0, 128), (128, 44), (172, 128))])
    assert torch.equal(whole, parts)
    rows = target._project(x, (q,), False)[0]
    assert torch.equal(rows, torch.cat([target._project(x[i:i + 5], (q,), False)[0] for i in range(0, 300, 5)]))


def test_mlx_affine_numerics_keep_one_kernel():
    target = Gemma4Target.__new__(Gemma4Target)
    target._prompt_gemm = target._group_projections = False
    calls = []
    affine = type("Affine", (), {"__call__": lambda self, x: calls.append(x.shape[0]) or x})()
    assert target._project(torch.zeros(3, 4), (affine, affine), True)[0].shape == (3, 4)
    assert calls == [3, 3]

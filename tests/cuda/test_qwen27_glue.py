"""Glue kernels: rows alone equal rows in a window, and values match plain torch."""

import math

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.families.qwen3_5.cuda import glue, prefill_bf16

dev = "cuda"


def _rows_alone(fn, *args):
    full = fn(*args)
    W = args[0].shape[0]
    for r in range(W):
        one = fn(*[a[r:r + 1] if torch.is_tensor(a) and a.dim() > 0 and a.shape[0] == W else a for a in args])
        for f, o in zip(full, one):
            assert torch.equal(f[r:r + 1], o), f"row {r} differs"
    return full


def test_add_rmsnorm():
    x = torch.randn(9, 5120, device=dev).bfloat16()
    r = torch.randn(9, 5120, device=dev).bfloat16()
    w = (torch.rand(5120, device=dev) + 0.5).bfloat16()
    h, y, xs = _rows_alone(lambda x, r, w: glue.add_rmsnorm(x, r, w, 1e-6), x, r, w)
    hf = (x.float() + r.float()).bfloat16().float()
    ref = hf * torch.rsqrt(hf.pow(2).mean(-1, keepdim=True) + 1e-6) * w.float()
    assert (y.float() - ref).abs().max() < 0.05
    assert torch.allclose(xs, y.float().reshape(9, 80, 64).sum(-1), atol=1e-3)


def test_swiglu_and_gate_mul():
    g = torch.randn(5, 17408, device=dev).bfloat16()
    u = torch.randn(5, 17408, device=dev).bfloat16()
    a, xs = _rows_alone(glue.swiglu, g, u)
    ref = torch.nn.functional.silu(g.float()) * u.float()
    assert (a.float() - ref).abs().max() < 0.05
    o = torch.randn(5, 24, 256, device=dev).bfloat16()
    qg = torch.randn(5, 24 * 512, device=dev).bfloat16()
    out, _ = _rows_alone(lambda o, qg: glue.gate_mul(o, qg, heads=24, head_dim=256), o, qg)
    gate = qg.reshape(5, 24, 512)[..., 256:].float()
    assert (out.float().reshape(5, 24, 256) - o.float() * torch.sigmoid(gate)).abs().max() < 0.05


def test_gdn_pre_matches_torch():
    W, C = 6, 10240
    qkv = torch.randn(W, C, device=dev).bfloat16()
    cs = torch.randn(3, C, device=dev).bfloat16()
    cw = torch.randn(C, 4, device=dev).bfloat16()
    win = torch.tensor([[0, 1, 2, 3 + i] if i == 0 else [1, 2, 3, 3 + i] for i in range(W)], device=dev, dtype=torch.int32)
    a = torch.randn(W, 48, device=dev).bfloat16()
    b = torch.randn(W, 48, device=dev).bfloat16()
    alog = torch.randn(48, device=dev)
    dtb = torch.randn(48, device=dev)
    q, k, v, g, beta = glue.gdn_pre(qkv, cs, cw, win, a, b, alog, dtb, kh=16, vh=48, dk=128)
    src = torch.cat([cs, qkv]).float()
    conv = sum(src[win[:, j].long()] * cw.float()[:, j] for j in range(4))
    c = torch.nn.functional.silu(conv).bfloat16().float()
    qr = c[:, :2048].reshape(W, 16, 128)
    qr = qr * torch.rsqrt(qr.pow(2).mean(-1, keepdim=True) + 1e-6) / 128
    assert (q.float() - qr).abs().max() < 2e-3
    # plain torch adds the 4 taps unfused; the kernel may fuse multiply-adds, so allow one bf16 step
    assert (v.float() - c[:, 4096:].reshape(W, 48, 128)).abs().max() <= c.abs().max() * 2 ** -7
    gr = torch.exp(-torch.exp(alog) * torch.nn.functional.softplus(a.float() + dtb))
    assert torch.allclose(g, gr, rtol=1e-5, atol=1e-6)
    assert torch.allclose(beta, torch.sigmoid(b.float()), atol=1e-6)


@pytest.mark.parametrize("rows,width,dim", [(1, 64, 64), (5, 192, 64), (17, 5120, 128),
                                          (129, 6144, 256), (3, 17408, 128)])
@pytest.mark.parametrize("edges", [False, True], ids=["random", "bf16-edges"])
def test_bf16_prompt_glue_preserves_every_retained_bit(rows, width, dim, edges):
    """Removing unused sums preserves BF16 bits, including signed zero, subnormals and NaN payloads."""

    generator = torch.Generator().manual_seed(431)

    def values(shape):
        if edges:
            bits = torch.tensor([0, 32768, 1, 32769, 127, 32895, 128, 32896, 16256, 49024,
                                 32639, 65407, 32640, 65408, 32705], dtype=torch.int32).to(torch.int16)
            count = math.prod(shape)
            return bits.repeat((count + len(bits) - 1) // len(bits))[:count].view(torch.bfloat16).reshape(shape).to(dev)
        return torch.randn(shape, generator=generator).bfloat16().to(dev)

    def same(actual, expected):
        assert actual.shape == expected.shape and actual.dtype == expected.dtype and actual.device == expected.device
        assert torch.equal(actual.view(torch.int16), expected.view(torch.int16))

    x, r, weight = values((rows, width)), values((rows, width)), values((width,))
    for residual in (None, r):
        baseline = glue.add_rmsnorm(x, residual, weight, 1e-6)
        actual = prefill_bf16.add_rmsnorm(x, residual, weight, 1e-6)
        for a, b in zip(actual, baseline[:2], strict=True):
            same(a, b)
        if residual is None:
            assert actual[0] is x
        else:
            added = actual
    swiglu = prefill_bf16.swiglu(x, r)
    same(swiglu, glue.swiglu(x, r)[0])
    heads = width // dim
    y, z = x.view(rows, heads, dim), r.view(rows, heads, dim)
    norm = values((dim,))
    normalized = prefill_bf16.gated_norm(y, z, norm, 1e-6)
    same(normalized, glue.gated_norm(y, z, norm, 1e-6)[0])
    qg = values((rows, heads * 2 * dim))
    gated = prefill_bf16.gate_mul(y, qg, heads=heads, head_dim=dim)
    same(gated, glue.gate_mul(y, qg, heads=heads, head_dim=dim)[0])
    for row in {0, rows // 2, rows - 1}:
        same(prefill_bf16.swiglu(x[row:row + 1], r[row:row + 1]), swiglu[row:row + 1])
        for a, b in zip(prefill_bf16.add_rmsnorm(x[row:row + 1], r[row:row + 1], weight, 1e-6),
                        added, strict=True):
            same(a, b[row:row + 1])
        same(prefill_bf16.gated_norm(y[row:row + 1], z[row:row + 1], norm, 1e-6),
             normalized[row:row + 1])
        same(prefill_bf16.gate_mul(y[row:row + 1], qg[row:row + 1], heads=heads, head_dim=dim), gated[row:row + 1])

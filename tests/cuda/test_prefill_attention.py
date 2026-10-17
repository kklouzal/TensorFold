"""Prompt attention: the CUDA kernel gives _attend's bits on every served shape, at any depth, row offset and chunking."""

import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)

from tensorfold.cuda.kernels.prefill_attention import _ext, attention, triton_attention

# 27B, Qwen3.6-35B-A3B, Nemotron, GLM-5.3 per-head keys (TF_GLM_LATENT=0), a small group
SHAPES = [(24, 4, 256), (16, 2, 256), (32, 2, 128), (32, 32, 256), (12, 4, 128), (12, 4, 64)]
CHUNKS = [(0, 1), (0, 100), (0, 333), (17, 7), (64, 64), (1000, 129), (4133, 300)]


def _caches(gen, keys, kv_heads, dim):
    k = (torch.randn(keys, kv_heads, dim, generator=gen, device="cuda") * 1.5).bfloat16()
    v = torch.randn(keys, kv_heads, dim, generator=gen, device="cuda").bfloat16()
    return k, v


@pytest.mark.parametrize("heads,kv_heads,dim", SHAPES)
def test_cuda_prompt_attention_has_the_triton_bits(heads, kv_heads, dim):
    gen = torch.Generator(device="cuda").manual_seed(heads * dim + kv_heads)
    scale = dim ** -0.5
    for p0, w in CHUNKS + ([(70001, 257)] if heads == 24 else []):
        k, v = _caches(gen, p0 + w + 50, kv_heads, dim)
        k[p0 + w:] = float("nan")                                   # past the chunk's keys: never read
        v[p0 + w:] = float("nan")
        q = (torch.randn(w, heads, dim, generator=gen, device="cuda") * 1.5).bfloat16()
        got = attention(q, k, v, p0, scale=scale)
        want = triton_attention(q, k, v, p0, scale=scale)
        assert torch.equal(got.view(torch.int16), want.view(torch.int16)), (p0, w)


@pytest.mark.parametrize("heads,kv_heads,dim", SHAPES[:3] + [SHAPES[-1]])
def test_cuda_prompt_attention_rows_do_not_depend_on_chunking(heads, kv_heads, dim):
    gen = torch.Generator(device="cuda").manual_seed(9)
    total, p0 = 900, 2000
    k, v = _caches(gen, p0 + total, kv_heads, dim)
    q = torch.randn(total, heads, dim, generator=gen, device="cuda").bfloat16()
    whole = attention(q, k, v, p0, scale=dim ** -0.5)
    for size in (1, 15, 16, 17, 64, 100, 512):
        parts = [attention(q[a:a + size].contiguous(), k, v, p0 + a, scale=dim ** -0.5) for a in range(0, total, size)]
        assert torch.equal(whole, torch.cat(parts)), size


@pytest.mark.parametrize("dim", [64, 128, 256])
@pytest.mark.parametrize("sign", [-1, 1])
def test_zero_scores_preserve_cancellation_across_key_tiles(dim, sign):
    """Uniform attention has a directly computed result independent of either kernel."""

    q = torch.zeros((1, 1, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((66, 1, dim), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    v[0, 0, 0], v[64, 0, 0], v[65, 0, 0] = sign * 2**24, -sign * 2**24, 1
    expected = torch.zeros_like(q)
    expected[0, 0, 0] = 1 / 66
    for implementation in (attention, triton_attention):
        assert torch.equal(implementation(q, k, v, 65, scale=dim**-0.5), expected)


@pytest.mark.parametrize("invalid", ["zero_query_heads", "zero_kv_heads", "cpu_q", "cpu_k", "cpu_v",
                                     "wrong_dim", "wrong_dtype", "negative_position", "large_position"])
def test_prompt_attention_rejects_invalid_inputs(invalid):
    q = torch.zeros((1, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((1, 1, 128), device="cuda", dtype=torch.bfloat16)
    v, p0 = torch.zeros_like(k), 0
    if invalid == "zero_query_heads":
        q = q[:, :0]
    elif invalid == "zero_kv_heads":
        k, v = k[:, :0], v[:, :0]
    elif invalid == "cpu_q":
        q = q.cpu()
    elif invalid == "cpu_k":
        k = k.cpu()
    elif invalid == "cpu_v":
        v = v.cpu()
    elif invalid == "wrong_dim":
        k, v = k[:, :, :64].contiguous(), v[:, :, :64].contiguous()
    elif invalid == "wrong_dtype":
        k = k.float()
    elif invalid == "negative_position":
        p0 = -1
    else:
        p0 = 2**63 - 1
    for implementation in (attention, triton_attention):
        with pytest.raises(ValueError, match="prefill attention"):
            implementation(q, k, v, p0, scale=0.125)
    with pytest.raises(RuntimeError):
        _ext().prefill_attention(q, k, v, torch.empty_like(q), p0, 0.125, 1)


@pytest.mark.parametrize("invalid", ["cpu", "shape", "dtype", "strided"])
def test_prompt_attention_rejects_invalid_output(invalid):
    q = torch.zeros((1, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((1, 1, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    out = torch.empty_like(q)
    if invalid == "cpu":
        out = out.cpu()
    elif invalid == "shape":
        out = out[:, :1]
    elif invalid == "dtype":
        out = out.float()
    else:
        out = torch.empty((1, 2, 256), device="cuda", dtype=torch.bfloat16)[:, :, ::2]
    with pytest.raises(ValueError, match="out must"):
        triton_attention(q, k, v, 0, scale=0.125, out=out)
    with pytest.raises(RuntimeError, match="out:"):
        _ext().prefill_attention(q, k, v, out, 0, 0.125, 1)


@pytest.mark.parametrize("hpc", [0, -1, 3, 16])
def test_prompt_attention_rejects_invalid_head_tiles(hpc):
    q = torch.zeros((1, 16, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((1, 1, 128), device="cuda", dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match="heads a block"):
        _ext().prefill_attention(q, k, k, torch.empty_like(q), 0, 0.125, hpc)


@pytest.mark.parametrize("alias", ["q", "partial_q", "k", "partial_k", "v"])
def test_prompt_attention_rejects_output_overlapping_reads(alias):
    q = torch.zeros((2, 2, 128), device="cuda", dtype=torch.bfloat16)
    k = torch.zeros((6, 1, 128), device="cuda", dtype=torch.bfloat16)
    v = torch.zeros_like(k)
    if alias == "q":
        out = q
    elif alias == "partial_q":
        backing = torch.zeros(640, device="cuda", dtype=torch.bfloat16)
        q, out = backing[:512].view_as(q), backing[128:].view_as(q)
    elif alias == "k":
        out = k[:4].view_as(q)
    elif alias == "partial_k":
        out = k[1:5].view_as(q)
    else:
        out = v[:4].view_as(q)
    with pytest.raises(ValueError, match="must not overlap"):
        triton_attention(q, k, v, 0, scale=0.125, out=out)
    with pytest.raises(RuntimeError, match="must not overlap"):
        _ext().prefill_attention(q, k, v, out, 0, 0.125, 1)


def test_prompt_attention_allows_output_in_unfilled_cache_tail():
    gen = torch.Generator(device="cuda").manual_seed(29)
    q = torch.randn((2, 2, 128), generator=gen, device="cuda").bfloat16()
    k, v = _caches(gen, 6, 1, 128)
    expected = attention(q, k, v, 0, scale=0.125)
    out = k[2:6].view_as(q)
    assert torch.equal(triton_attention(q, k, v, 0, scale=0.125, out=out), expected)
    _ext().prefill_attention(q, k, v, out, 0, 0.125, 1)
    assert torch.equal(out, expected)


@pytest.mark.parametrize("dim", [64, 128, 256])
def test_prompt_attention_empty_window(dim):
    q = torch.empty((0, 2, dim), device="cuda", dtype=torch.bfloat16)
    k = torch.empty((0, 1, dim), device="cuda", dtype=torch.bfloat16)
    for implementation in (attention, triton_attention):
        assert implementation(q, k, k, 0, scale=dim**-0.5).shape == q.shape

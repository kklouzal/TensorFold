"""Requantized copies on CUDA (tensorfold.cuda.requant): products within E4M3 error, heads shared, sizes as counted."""

import pytest
import torch

from tensorfold.cuda.requant import Head, nvfp4_rows, prefix, requant_rows, requantize, resident_bytes, staging_bytes

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def _deq(w, kind):
    """What a ``kind`` copy of ``w`` holds, exactly, in FP32."""
    if kind == "nvfp4":
        from tests.test_requant import _nvfp4_values

        return _nvfp4_values(*(t.cpu() for t in nvfp4_rows(w))).to(w.device)
    codes, scales = requant_rows(w)
    return codes.float() * scales[:, None]


def _assert_product_envelope(got, x, deq):
    """Retain the existing empirical product regression envelope: (K+1)*2**-23 times the sum of absolute products,
    plus a BF16 output rounding term when applicable. qmmf/prompt use BF16 MMA: PTX leaves internal accumulation
    order, rounding and subnormal handling unspecified, so this is not a universal per-operation error bound.
    https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-matrix-instructions-mma
    """
    unit = 2.0 ** -8 if got.dtype == torch.bfloat16 else 0.0
    xd, dd = x.double(), deq.double()
    exact = xd @ dd.t()
    bound = unit * exact.abs() + (1 + unit) * (x.shape[1] + 1) * 2.0 ** -23 * (xd.abs() @ dd.abs().t())
    assert bool(((got.double() - exact).abs() <= bound).all())


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
@pytest.mark.parametrize("rows", [1, 5, 16, 200])
def test_copies_compute_the_dequantized_product(kind, rows):
    gen = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(1000, 256, generator=gen, device="cuda") * 0.05).bfloat16()
    x = torch.randn(rows, 256, generator=gen, device="cuda").bfloat16()
    lin = requantize(w, kind)
    assert lin.nbytes() == resident_bytes(kind, 1000, 256) and (lin.n, lin.k) == (1000, 256)
    deq = _deq(w, kind)                                    # what the copy holds, exactly
    got = lin(x)
    _assert_product_envelope(got, x, deq)
    view = prefix(lin, 128)                                # the first 128 output rows, sharing storage
    torch.testing.assert_close(view(x), got[:, :128], rtol=0, atol=0)
    for bad in (100, 192):                                 # 192: decode tiles, but not the prompt GEMM's 128
        with pytest.raises(ValueError):
            prefix(lin, bad)


@pytest.mark.parametrize("kind", ["fp8", "nvfp4"])
def test_prompt_rows_run_on_whole_copies_and_row_prefixes(kind):
    """A fused [k|v|q] copy: prompt chunks (up to 1024 rows) through prefill(), on the whole copy and on its [k|v]
    prefix, agree with the dequantized product (Gemma's companion: kv 2048 of 6144 rows, K 2816)."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    w = (torch.randn(6144, 2816, generator=gen, device="cuda") * 0.02).bfloat16()
    lin = requantize(w, kind)
    deq = _deq(w, kind)
    for rows in (17, 1024):
        x = torch.randn(rows, 2816, generator=gen, device="cuda").bfloat16()
        for got, held in ((lin.prefill(x), deq), (prefix(lin, 2048).prefill(x), deq[:2048])):
            assert got.shape == (rows, held.shape[0])
            _assert_product_envelope(got, x, held)


def _assert_softcap_envelope(got, x, deq, cap):
    """Retain the existing softcap regression threshold: the empirical FP32 product envelope carried through the
    1-Lipschitz ``tanh(y / cap) * cap``, plus the existing FP32 scale/tanh rounding terms. Its MMA contribution has
    the same empirical scope as ``_assert_product_envelope``, not a universal architecture-level error proof.
    """
    xd, dd = x.double(), deq.double()
    exact = xd @ dd.t()
    sums = (x.shape[1] + 1) * 2.0 ** -23 * (xd.abs() @ dd.abs().t())
    want = torch.tanh(exact / cap) * cap
    bound = sums + 2.0 ** -24 * (exact.abs() + sums) + (2 * 2.0 ** -23 + 2.0 ** -24) * (want.abs() + sums)
    assert bool(((got.double() - want).abs() <= bound).all())


def test_heads_return_fp32_softcapped_logits_and_share_one_copy():
    gen = torch.Generator(device="cuda").manual_seed(9)
    w = (torch.randn(4100, 128, generator=gen, device="cuda") * 0.3).bfloat16()

    class Plain:
        weight = w

    head = Head(Plain(), "fp8", softcap=30.0)
    x = torch.randn(3, 128, generator=gen, device="cuda").bfloat16()
    logits = head(x)
    assert logits.dtype == torch.float32 and logits.shape == (3, 4100)
    _assert_softcap_envelope(logits, x, _deq(w, "fp8"), 30.0)
    _assert_product_envelope(Head(head, "fp8")(x), x, _deq(w, "fp8"))     # uncapped: the FP32 sums themselves
    shared = Head(head, "fp8")                             # a drafter reusing the target's copy: no second copy
    assert shared.linear is head.linear and shared.softcap is None
    with pytest.raises(ValueError):
        Head(head.linear, "nvfp4")                         # another kind starts from the BF16 source
    with pytest.raises(ValueError):
        Head(head, "int4")                                 # unknown kinds, shared copy or not
    with pytest.raises(ValueError, match="floating head"):           # a head stored quantized is borrowed, not copied
        Head(type("Packed", (), {"weight": w.view(torch.uint8)})(), "fp8")
    with pytest.raises(ValueError, match="same kind"):
        Head(requantize(w, "nvfp4"), "fp8")
    for cap in (0, True, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            Head(Plain(), "fp8", softcap=cap)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_weights_are_refused_on_cuda(bad):
    w = torch.ones(300, 64, dtype=torch.bfloat16, device="cuda")
    w[200, 0], w[200, 3] = 1e4, bad                        # a later chunk, beside a larger finite value
    with pytest.raises(ValueError):
        requant_rows(w, chunk=128)


def test_nvfp4_heads_share_one_copy_and_kinds_do_not_mix():
    gen = torch.Generator(device="cuda").manual_seed(13)
    w = (torch.randn(4100, 128, generator=gen, device="cuda") * 0.3).bfloat16()

    class Plain:
        weight = w

    head = Head(Plain(), "nvfp4")
    x = torch.randn(3, 128, generator=gen, device="cuda").bfloat16()
    logits = head(x)
    assert logits.dtype == torch.float32
    _assert_product_envelope(logits, x, _deq(w, "nvfp4"))
    assert Head(head, "nvfp4").linear is head.linear and head.nbytes() == resident_bytes("nvfp4", 4100, 128)
    with pytest.raises(ValueError):
        Head(head, "fp8")
    with pytest.raises(ValueError):
        Head(Head(Plain(), "fp8"), "nvfp4")


@pytest.mark.parametrize("n,k", [(1, 64), (129, 256), (4100, 256)])
def test_fp8_build_stays_within_its_staging_estimate(n, k):
    gen = torch.Generator(device="cuda").manual_seed(23)
    w = (torch.randn(n, k, generator=gen, device="cuda") * 0.02).bfloat16()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    lin = requantize(w, "fp8")
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before == lin.nbytes() == resident_bytes("fp8", n, k)
    assert torch.cuda.max_memory_allocated() - before <= resident_bytes("fp8", n, k) + staging_bytes("fp8", n, k)


def test_nvfp4_build_stays_within_its_staging_estimate():
    gen = torch.Generator(device="cuda").manual_seed(17)
    n, k = 20000, 2048
    w = (torch.randn(n, k, generator=gen, device="cuda") * 0.02).bfloat16()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    before = torch.cuda.memory_allocated()
    lin = requantize(w, "nvfp4")
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() - before == lin.nbytes() == resident_bytes("nvfp4", n, k)
    assert torch.cuda.max_memory_allocated() - before <= resident_bytes("nvfp4", n, k) + staging_bytes("nvfp4", n, k)


def test_expert_quantize_is_the_shared_recipe_per_expert():
    """``nvfp4.experts.quantize`` (Flash Next drafts) = ``nvfp4_rows`` per expert = the previous whole-tensor recipe."""
    from tensorfold.cuda.nvfp4 import experts as nvx
    from tests.test_requant import _recipe_oracle

    gen = torch.Generator(device="cuda").manual_seed(21)
    spread = torch.tensor([0.01, 0.1, 1.0], device="cuda")[:, None, None]       # a different global scale an expert
    w = (torch.randn(3, 96, 128, generator=gen, device="cuda") * spread).bfloat16()
    words, scales, g = nvx.quantize(w)
    for e in range(3):
        want = _recipe_oracle(w[e].cpu())
        assert torch.equal(words[e].cpu(), want[0]) and torch.equal(scales[e].cpu(), want[1])
        assert torch.equal(g[e].cpu(), want[2])
    # loaders pass views: gate_up[:, :ni] (row slices) and a rank's down[:, :, lo:hi] (strided rows)
    for view in (torch.cat([w, w], 1)[:, :96], torch.cat([w, w], 2)[:, :, 128:]):
        assert all(torch.equal(a, b) for a, b in zip(nvx.quantize(view), (words, scales, g)))

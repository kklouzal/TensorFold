"""Fused MoE routing kernels versus float64 references: logits GEMV (plain and CT Gemma pre-norm) and Gemma selection."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
    pytest.skip("routing kernels need CUDA SM 8.9+", allow_module_level=True)

from tensorfold.cuda.moe import route_logits, route_select


U = 2.0 ** -24            # FP32 unit roundoff, round to nearest


def _gamma(n):
    return n * U / (1 - n * U)


def _mma_gamma(k):
    """route.cu's FP32 K-sum error relative to sum(|products|), for any m16n8k16 accumulation PTX admits.

    PTX does not specify the rounding of mma's FP32 accumulation; published tensor-core models (arXiv 2502.15999,
    2512.07004) align each block of b >= 1 exact BF16 products and the accumulator to the largest exponent with
    truncation and truncate the normalized sum, so a block errs by under (b + 1) * 2^-23 of its operands' magnitude.
    A warp's slice of ``per`` inputs (fresh accumulator; earlier products re-enter later blocks through it) then errs
    by under 2 * per * 2^-23 = 4 * per * U of the slice's sum(|products|), and the eight round-to-nearest slice
    additions add gamma_8. ``per`` is the kernel's shape-only slice width.
    """
    per = -(-(k // 32) // 8) * 32
    return _gamma(4 * per + 8)


def _logits(x, w, **kwargs):
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float32, device="cuda")
    return route_logits(x, w, out, **kwargs)


@pytest.mark.parametrize("experts,k", [(128, 2816), (257, 2048), (5, 64), (16, 96)])
def test_plain_logits_are_row_invariant_and_within_the_fp32_summation_bound(experts, k):
    generator = torch.Generator().manual_seed(experts + k)
    w = torch.randn((experts, k), generator=generator).to("cuda", torch.bfloat16)
    x = torch.randn((133, k), generator=generator).to("cuda", torch.bfloat16)
    actual = _logits(x, w)
    # BF16 products are exact in FP32; the accumulation error is bounded by _mma_gamma.
    reference = x.double() @ w.double().T
    bound = _mma_gamma(k) * (x.double().abs() @ w.double().abs().T)
    assert torch.all((actual.double() - reference).abs() <= bound)
    assert torch.equal(actual, torch.cat([_logits(row[None], w) for row in x]))
    for lo, hi in ((0, 5), (3, 20), (17, 33), (100, 133)):
        assert torch.equal(actual[lo:hi], _logits(x[lo:hi], w))
    # A strided row view (the caller's hidden buffer) gives the same bits.
    wide = torch.zeros((133, k + 64), dtype=torch.bfloat16, device="cuda")
    wide[:, :k] = x
    assert torch.equal(_logits(wide[:, :k], w), actual)


def _inv_error(k):
    """Relative error bound of route.cu's FP32 1/RMS factor: a lane's FMA chain over at most 8 * ceil(K / 256)
    nonnegative squares, the 5-level shuffle tree, the division by K, the FP32-rounded eps and its addition (one
    rounding each, gamma_n over nonnegative terms), then rsqrtf's 2 ulp (CUDA C++ Programming Guide). Doubled to cover
    the square root's and the products' second-order terms."""
    return 2 * (_gamma(8 * -(-k // 256) + 5 + 3) + 2.0 ** -22)


def _ct_input_candidates(x, eps, root, scale):
    """The CT Gemma router input from a float64 RMS, and the BF16 inputs route.cu can form instead.

    Only the 1/RMS factor differs (relative error under _inv_error, plus the FP32 product and the reference's own
    float64 error). The BF16 roundings are monotone and BF16 x BF16 products are exact in FP32, so the kernel's
    normalized element is one of the roundings of x * inv * (1 +- e) and its root/scale products round exactly as
    here. Returns (reference input, |difference| between the two candidates, larger candidate magnitude), float64.
    """
    k = x.shape[-1]
    inv = (x.double().square().mean(-1, keepdim=True) + eps).rsqrt()
    e = _inv_error(k) + 2 * U + (k + 8) * 2.0 ** -52

    def finish(normed):
        rooted = (normed.double() * root).to(torch.bfloat16)
        return (rooted.double() * scale.double()).to(torch.bfloat16).double()

    magnitude = x.double().abs() * inv
    lo = (magnitude * (1 - e)).to(torch.bfloat16)
    hi = (magnitude * (1 + e)).to(torch.bfloat16)
    steps = hi.view(torch.int16).long() - lo.view(torch.int16).long()
    assert torch.all((steps >= 0) & (steps <= 1))      # at most one BF16 rounding boundary per interval
    sign = 1 - 2 * (x < 0).double()
    first, second = finish((lo.double() * sign).to(torch.bfloat16)), finish((hi.double() * sign).to(torch.bfloat16))
    reference = finish((x.double() * inv).to(torch.bfloat16))
    assert torch.all((reference == first) | (reference == second))
    return reference, (first - second).abs(), torch.maximum(first.abs(), second.abs())


@pytest.mark.parametrize("experts,k", [(128, 2816), (4, 64)])
def test_ct_prenorm_logits_match_the_bf16_router_input_and_are_row_invariant(experts, k):
    generator = torch.Generator().manual_seed(7 * k + experts)
    w = (torch.randn((experts, k), generator=generator) * .05).to("cuda", torch.bfloat16)
    x = (torch.randn((129, k), generator=generator) * 3).to("cuda", torch.bfloat16)
    scale = (torch.rand((k,), generator=generator) + .5).to("cuda", torch.bfloat16)
    eps = 1e-6
    root = float(torch.tensor(k ** -.5, dtype=torch.bfloat16))
    actual = _logits(x, w, prenorm=(eps, root, scale))
    router_input, delta, magnitude = _ct_input_candidates(x, eps, root, scale)
    weight = w.double().abs()
    reference = router_input @ w.double().T
    # The kernel's own inputs sum within _mma_gamma; an input on a BF16 boundary may differ by delta.
    bound = _mma_gamma(k) * (magnitude @ weight.T) + delta @ weight.T
    assert torch.all((actual.double() - reference).abs() <= bound)
    assert torch.equal(actual, torch.cat([_logits(row[None], w, prenorm=(eps, root, scale)) for row in x]))
    assert torch.equal(actual[3:8], _logits(x[3:8], w, prenorm=(eps, root, scale)))


def _select(logits, scale, top_k):
    rows = logits.shape[0]
    picks = torch.empty((rows, top_k), dtype=torch.int32, device="cuda")
    weights = torch.empty((rows, top_k), dtype=torch.float32, device="cuda")
    route_select(logits, scale, picks, weights)
    return picks, weights


@pytest.mark.parametrize("experts,top_k", [(128, 8), (4, 2), (256, 8), (1000, 32)])
def test_selection_matches_stable_descending_argsort_and_renormalized_probabilities(experts, top_k):
    generator = torch.Generator().manual_seed(experts)
    logits = (torch.randn((37, experts), generator=generator) * 2).to("cuda")
    # Exact ties: equal logits must keep the lower expert id first.
    logits[0, :] = 0.5
    logits[1, 10:20] = 9.0
    logits[2, :] = -float("inf")
    logits[2, experts - 1] = 1.0
    scale = (torch.rand((experts,), generator=generator) + .5).to("cuda", torch.bfloat16)
    picks, weights = _select(logits, scale, top_k)
    order = torch.argsort(logits, dim=-1, descending=True, stable=True)[:, :top_k]
    assert torch.equal(picks, order.to(torch.int32))
    probabilities = torch.softmax(logits.double(), dim=-1)
    selected = torch.gather(probabilities, 1, order)
    reference = selected / selected.sum(-1, keepdim=True) * scale[order].double()
    finite = torch.isfinite(reference)
    assert torch.equal(finite, torch.isfinite(weights))
    # A selected weight's FP32 error is its probability's: the subtraction of the row maximum (U * gap), expf (2 ulp,
    # CUDA C++ Programming Guide) and the division by the total, which cancels in the renormalization, for the weight
    # and for the selected sum; then top_k - 1 additions, the division and the scale.
    gap = logits.double().amax(-1, keepdim=True) - torch.gather(logits.double(), 1, order)
    gap = torch.where(torch.isfinite(gap), gap, torch.zeros_like(gap)).amax(-1, keepdim=True)
    assert torch.all(gap < 80)                         # every selected nonzero probability is a normal FP32 number
    rtol = 2 * (U * gap + 2.0 ** -22 + U) + _gamma(top_k + 1)
    error = (weights.double() - reference).abs()
    # 1.01 covers products of first-order terms (each under 1e-5).
    assert torch.all((error <= 1.01 * rtol * reference.abs())[finite])
    sub_picks, sub_weights = _select(logits[5:9], scale, top_k)
    assert torch.equal(sub_picks, picks[5:9]) and torch.equal(sub_weights, weights[5:9])


def test_selection_with_nan_logits_returns_distinct_valid_ids_without_failing():
    logits = torch.randn((3, 128), device="cuda")
    logits[0, 7] = float("nan")
    logits[1, :] = float("nan")
    logits[2, ::3] = float("nan")
    scale = torch.ones((128,), dtype=torch.bfloat16, device="cuda")
    picks, _ = _select(logits, scale, 8)
    torch.cuda.synchronize()
    for row in picks.cpu().tolist():
        assert len(set(row)) == 8 and all(0 <= e < 128 for e in row)
    # NaN orders before every number, then the lower id (a stable descending argsort's order).
    assert picks[0, 0].item() == 7
    assert picks[1].cpu().tolist() == list(range(8))
    assert picks[2].cpu().tolist() == list(range(0, 24, 3))

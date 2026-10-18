"""Fused MoE routing kernels versus float64 references: logits GEMV (plain and CT Gemma pre-norm) and Gemma selection."""

import pytest
import torch

if not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9):
    pytest.skip("routing kernels need CUDA SM 8.9+", allow_module_level=True)

from tensorfold.cuda.moe import route_logits, route_select


def _gamma(k):
    u = 2.0 ** -24
    return k * u / (1 - k * u)


def _logits(x, w, **kwargs):
    out = torch.empty((x.shape[0], w.shape[0]), dtype=torch.float32, device="cuda")
    return route_logits(x, w, out, **kwargs)


@pytest.mark.parametrize("experts,k", [(128, 2816), (257, 2048), (5, 64), (16, 96)])
def test_plain_logits_are_row_invariant_and_within_the_fp32_summation_bound(experts, k):
    generator = torch.Generator().manual_seed(experts + k)
    w = torch.randn((experts, k), generator=generator).to("cuda", torch.bfloat16)
    x = torch.randn((133, k), generator=generator).to("cuda", torch.bfloat16)
    actual = _logits(x, w)
    # BF16 products are exact in FP32; every addition (inside an mma or between slices) is covered by gamma_K.
    reference = x.double() @ w.double().T
    bound = _gamma(k) * (x.double().abs() @ w.double().abs().T)
    assert torch.all((actual.double() - reference).abs() <= bound)
    assert torch.equal(actual, torch.cat([_logits(row[None], w) for row in x]))
    for lo, hi in ((0, 5), (3, 20), (17, 33), (100, 133)):
        assert torch.equal(actual[lo:hi], _logits(x[lo:hi], w))
    # A strided row view (the caller's hidden buffer) gives the same bits.
    wide = torch.zeros((133, k + 64), dtype=torch.bfloat16, device="cuda")
    wide[:, :k] = x
    assert torch.equal(_logits(wide[:, :k], w), actual)


def _ct_input_reference(x, eps, root, scale):
    """Float64 RMS, then each BF16 tensor operation's rounding, as the CT Gemma router input."""
    inv = (x.double().square().mean(-1, keepdim=True) + eps).rsqrt()
    normed = (x.double() * inv).to(torch.bfloat16)
    rooted = (normed.double() * root).to(torch.bfloat16)
    return (rooted.double() * scale.double()).to(torch.bfloat16)


@pytest.mark.parametrize("experts,k", [(128, 2816), (4, 64)])
def test_ct_prenorm_logits_match_the_bf16_router_input_and_are_row_invariant(experts, k):
    generator = torch.Generator().manual_seed(7 * k + experts)
    w = (torch.randn((experts, k), generator=generator) * .05).to("cuda", torch.bfloat16)
    x = (torch.randn((129, k), generator=generator) * 3).to("cuda", torch.bfloat16)
    scale = (torch.rand((k,), generator=generator) + .5).to("cuda", torch.bfloat16)
    eps = 1e-6
    root = float(torch.tensor(k ** -.5, dtype=torch.bfloat16))
    actual = _logits(x, w, prenorm=(eps, root, scale))
    router_input = _ct_input_reference(x, eps, root, scale).double()
    reference = router_input @ w.double().T
    products = router_input.abs() @ w.double().abs().T
    # FP32 rsqrt may move a few inputs across a BF16 rounding boundary (each of the three roundings at most one
    # BF16 ulp, 2^-8 relative): allow eight such inputs at the row's largest product.
    largest = (router_input.abs()[:, None, :] * w.double().abs()[None]).amax(-1)
    bound = _gamma(k) * products + 8 * 3 * 2.0 ** -8 * largest
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
    torch.testing.assert_close(weights.double()[finite], reference[finite], rtol=2.0 ** -18, atol=0)
    assert torch.equal(_select(logits[5:9], scale, top_k)[0], picks[5:9])
    assert torch.equal(_select(logits[5:9], scale, top_k)[1], weights[5:9])


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

"""Requantized copies on CUDA (tensorfold.cuda.requant): products within E4M3 error, heads shared, sizes as counted."""

import pytest
import torch

from tensorfold.cuda.requant import Head, prefix, requant_rows, requantize, resident_bytes

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


@pytest.mark.parametrize("kind", ["fp8", "fp8a8"])
@pytest.mark.parametrize("rows", [1, 5, 16, 200])
def test_copies_compute_the_dequantized_product(kind, rows):
    gen = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(1000, 256, generator=gen, device="cuda") * 0.05).bfloat16()
    x = torch.randn(rows, 256, generator=gen, device="cuda").bfloat16()
    lin = requantize(w, kind)
    assert lin.nbytes() == resident_bytes(kind, 1000, 256) and (lin.n, lin.k) == (1000, 256)
    codes, scales = requant_rows(w)
    deq = codes.float() * scales[:, None]                  # what the copy holds, exactly
    ref = x.float() @ deq.t()
    got = lin(x).float()
    tol = 0.02 if kind == "fp8" else 0.08                  # bf16 output (and e4m3 rows for fp8a8)
    assert float(((got - ref).abs().max() / ref.abs().max())) < tol
    view = prefix(lin, 128)                                # the first 128 output rows, sharing storage
    torch.testing.assert_close(view(x).float(), got[:, :128], rtol=0, atol=0)
    for rows in (100, 192):                                # 192: decode tiles, but not the prompt GEMM's 128
        with pytest.raises(ValueError):
            prefix(lin, rows)


@pytest.mark.parametrize("kind", ["fp8", "fp8a8"])
def test_prompt_rows_run_on_whole_copies_and_row_prefixes(kind):
    """A fused [k|v|q] copy: prompt chunks (up to 1024 rows) through prefill(), on the whole copy and on its [k|v]
    prefix, agree with the dequantized product (Gemma's companion: kv 2048 of 6144 rows, K 2816)."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    w = (torch.randn(6144, 2816, generator=gen, device="cuda") * 0.02).bfloat16()
    lin = requantize(w, kind)
    codes, scales = requant_rows(w)
    deq = codes.float() * scales[:, None]
    for rows in (17, 1024):
        x = torch.randn(rows, 2816, generator=gen, device="cuda").bfloat16()
        ref = x.float() @ deq.t()
        tol = 0.02 if kind == "fp8" else 0.08
        for got, want in ((lin.prefill(x).float(), ref), (prefix(lin, 2048).prefill(x).float(), ref[:, :2048])):
            assert got.shape == want.shape
            assert float(((got - want).abs().max() / want.abs().max())) < tol


def test_heads_return_fp32_softcapped_logits_and_share_one_copy():
    gen = torch.Generator(device="cuda").manual_seed(9)
    w = (torch.randn(4100, 128, generator=gen, device="cuda") * 0.3).bfloat16()

    class Plain:
        weight = w

    head = Head(Plain(), "fp8", softcap=30.0)
    x = torch.randn(3, 128, generator=gen, device="cuda").bfloat16()
    logits = head(x)
    codes, scales = requant_rows(w)
    ref = torch.tanh((x.float() @ (codes.float() * scales[:, None]).t()) / 30.0) * 30.0
    assert logits.dtype == torch.float32 and logits.shape == (3, 4100)
    assert float((logits - ref).abs().max()) < 0.05
    shared = Head(head, "fp8")                             # a drafter reusing the target's copy: no second copy
    assert shared.linear is head.linear and shared.softcap is None
    with pytest.raises(ValueError):
        Head(head.linear, "fp8a8")                         # another kind starts from the BF16 source
    with pytest.raises(ValueError):
        Head(head, "int4")                                 # unknown kinds, shared copy or not
    assert Head(Plain(), "fp8a8").linear.token
    for cap in (0, True, float("inf"), float("nan")):
        with pytest.raises(ValueError):
            Head(Plain(), "fp8", softcap=cap)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_weights_are_refused_on_cuda(bad):
    w = torch.ones(300, 64, dtype=torch.bfloat16, device="cuda")
    w[200, 0], w[200, 3] = 1e4, bad                        # a later chunk, beside a larger finite value
    with pytest.raises(ValueError):
        requant_rows(w, chunk=128)

"""Requantized copies on CUDA (tensorfold.cuda.requant): products within E4M3 error, heads shared, sizes as counted."""

import pytest
import torch

from tensorfold.cuda.requant import (Head, nvfp4_rows, prefix, requant_rows, requantize, resident_bytes,
                                    staging_bytes)

if not torch.cuda.is_available():
    pytest.skip("CUDA only", allow_module_level=True)


def _deq(w, kind):
    """What a ``kind`` copy of ``w`` holds, exactly, in FP32."""
    if kind == "nvfp4":
        from tests.test_requant import _nvfp4_values

        return _nvfp4_values(*(t.cpu() for t in nvfp4_rows(w))).to(w.device)
    codes, scales = requant_rows(w)
    return codes.float() * scales[:, None]


@pytest.mark.parametrize("kind", ["fp8", "fp8a8", "nvfp4"])
@pytest.mark.parametrize("rows", [1, 5, 16, 200])
def test_copies_compute_the_dequantized_product(kind, rows):
    gen = torch.Generator(device="cuda").manual_seed(5)
    w = (torch.randn(1000, 256, generator=gen, device="cuda") * 0.05).bfloat16()
    x = torch.randn(rows, 256, generator=gen, device="cuda").bfloat16()
    lin = requantize(w, kind)
    assert lin.nbytes() == resident_bytes(kind, 1000, 256) and (lin.n, lin.k) == (1000, 256)
    deq = _deq(w, kind)                                    # what the copy holds, exactly
    ref = x.float() @ deq.t()
    got = lin(x).float()
    tol = 0.08 if kind == "fp8a8" else 0.02                # bf16 output (and e4m3 rows for fp8a8)
    assert float(((got - ref).abs().max() / ref.abs().max())) < tol
    view = prefix(lin, 128)                                # the first 128 output rows, sharing storage
    torch.testing.assert_close(view(x).float(), got[:, :128], rtol=0, atol=0)
    with pytest.raises(ValueError):
        prefix(lin, 100)


@pytest.mark.parametrize("kind", ["fp8", "fp8a8", "nvfp4"])
def test_prompt_rows_run_on_whole_copies_and_row_prefixes(kind):
    """A fused [k|v|q] copy: prompt chunks (up to 1024 rows) through prefill(), on the whole copy and on its [k|v]
    prefix, agree with the dequantized product (Gemma's companion: kv 2048 of 6144 rows, K 2816)."""
    gen = torch.Generator(device="cuda").manual_seed(11)
    w = (torch.randn(6144, 2816, generator=gen, device="cuda") * 0.02).bfloat16()
    lin = requantize(w, kind)
    deq = _deq(w, kind)
    for rows in (17, 1024):
        x = torch.randn(rows, 2816, generator=gen, device="cuda").bfloat16()
        ref = x.float() @ deq.t()
        tol = 0.08 if kind == "fp8a8" else 0.02
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
    assert Head(Plain(), "fp8a8").linear.token
    with pytest.raises(ValueError):
        Head(Plain(), "fp8", softcap=0)


def test_nvfp4_heads_share_one_copy_and_kinds_do_not_mix():
    gen = torch.Generator(device="cuda").manual_seed(13)
    w = (torch.randn(4100, 128, generator=gen, device="cuda") * 0.3).bfloat16()

    class Plain:
        weight = w

    head = Head(Plain(), "nvfp4")
    x = torch.randn(3, 128, generator=gen, device="cuda").bfloat16()
    logits = head(x)
    ref = x.float() @ _deq(w, "nvfp4").t()
    assert logits.dtype == torch.float32 and float((logits - ref).abs().max() / ref.abs().max()) < 1e-3
    assert Head(head, "nvfp4").linear is head.linear and head.nbytes() == resident_bytes("nvfp4", 4100, 128)
    for kind in ("fp8", "fp8a8"):
        with pytest.raises(ValueError):
            Head(head, kind)
    with pytest.raises(ValueError):
        Head(Head(Plain(), "fp8"), "nvfp4")


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
    from tests.test_requant import _modelopt_nvfp4

    gen = torch.Generator(device="cuda").manual_seed(21)
    spread = torch.tensor([0.01, 0.1, 1.0], device="cuda")[:, None, None]       # a different global scale an expert
    w = (torch.randn(3, 96, 128, generator=gen, device="cuda") * spread).bfloat16()
    words, scales, g = nvx.quantize(w)
    for e in range(3):
        want = _modelopt_nvfp4(w[e].cpu())
        assert torch.equal(words[e].cpu(), want[0]) and torch.equal(scales[e].cpu(), want[1])
        assert torch.equal(g[e].cpu(), want[2])

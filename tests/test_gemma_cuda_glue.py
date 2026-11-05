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


def ordered_slots_reference(y, picks, weights):
    """Independent Torch BF16 storage after each expert-ID-ordered contribution."""
    rows, _, dims = y.shape
    result = torch.zeros((rows, dims), dtype=torch.bfloat16, device=y.device)
    for row in range(rows):
        order = sorted(enumerate(picks[row].cpu().tolist()), key=lambda pair: (pair[1], pair[0]))
        for slot, _ in order:
            term = (y[row, slot].float() * weights[row, slot]).to(torch.bfloat16)
            result[row] = result[row] + term
    return result


@pytest.mark.parametrize('rows', [1, 2, 3, 4, 5])
def test_fused_slot_tail_keeps_ordered_rounding_and_both_output_words(rows):
    from tensorfold.families.gemma4.cuda.nvfp4_experts import _combine
    import triton

    dims, slots = 2816, 8
    y = rnd(rows, slots, dims, seed=9901 + rows, scale=3)
    # Cancellation and shuffled IDs distinguish expert order, per-add BF16 and FP32 weights.
    y[:, :, 0] = torch.tensor([1., 256., -256., .25, 0., 0., 0., 0.], device=DEV).to(torch.bfloat16)
    y[:, :, 1] = torch.tensor([-256., 256., 1., .25, 0., 0., 0., 0.], device=DEV).to(torch.bfloat16)
    y[:, :, 2] = 0
    y[:, 3, 2] = 1.5
    picks = torch.tensor([[2, 0, 1, 3, 7, 6, 5, 4]], dtype=torch.int32, device=DEV).expand(rows, -1).contiguous()
    routing = torch.tensor([[1., 1., 1., 1.0038, .25, .5, .75, .125]], device=DEV).expand(rows, -1).contiguous()
    routed = torch.empty((rows, dims), dtype=torch.bfloat16, device=DEV)
    _combine[(rows, triton.cdiv(dims, 256))](y, picks, routing, routed, N=dims, R=slots, RS=slots,
        BLOCK=256, num_warps=4, enable_fp_fusion=False)
    independent = ordered_slots_reference(y, picks, routing)
    # Explicit mutations prove these fixture columns distinguish each forbidden shortcut.
    wrong_order = torch.zeros((rows, 3), device=DEV, dtype=torch.bfloat16)
    wrong_accum = torch.zeros((rows, 3), device=DEV, dtype=torch.float32)
    wrong_weight = torch.zeros((rows, 3), device=DEV, dtype=torch.bfloat16)
    for row in range(rows):
        for slot in range(slots):
            wrong_order[row] = wrong_order[row] + (y[row, slot, :3].float() * routing[row, slot]).to(torch.bfloat16)
        order = sorted(enumerate(picks[row].cpu().tolist()), key=lambda pair: (pair[1], pair[0]))
        for slot, _ in order:
            wrong_accum[row] += (y[row, slot, :3].float() * routing[row, slot]).to(torch.bfloat16).float()
            narrowed = routing[row, slot].to(torch.bfloat16).float()
            wrong_weight[row] = wrong_weight[row] + (y[row, slot, :3].float() * narrowed).to(torch.bfloat16)
    assert not torch.equal(wrong_order[:, 0], independent[:, 0])
    assert not torch.equal(wrong_accum.to(torch.bfloat16)[:, 1], independent[:, 1])
    assert not torch.equal(wrong_weight[:, 2], independent[:, 2])
    assert torch.equal(routed.view(torch.uint8), independent.view(torch.uint8))
    res, dense = rnd(rows, dims, seed=9908, scale=5), rnd(rows, dims, seed=9909, scale=3)
    wd, wr, wo, wn = (rnd(dims, seed=seed, scale=.2, shift=1) for seed in (9910, 9911, 9912, 9913))
    scalar = torch.tensor([.87], dtype=torch.bfloat16, device=DEV)
    expected = glue.moe_tail(res, dense, independent, wd, wr, wo, scalar, wn, EPS, False)
    retained = glue.moe_tail(res, dense, routed, wd, wr, wo, scalar, wn, EPS, False)
    actual = glue.moe_tail_slots(res, dense, y, picks, routing, wd, wr, wo, scalar, wn, EPS)
    assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) and torch.equal(b.view(torch.uint8), c.view(torch.uint8)) for a, b, c in zip(actual, expected, retained))
    assert all(value.dtype == torch.bfloat16 and tuple(value.shape) == (rows, dims) for value in actual)
    assert actual[0].data_ptr() != y.data_ptr() and actual[1].data_ptr() != y.data_ptr()
    serial = [glue.moe_tail_slots(res[i:i + 1], dense[i:i + 1], y[i:i + 1], picks[i:i + 1], routing[i:i + 1],
        wd, wr, wo, scalar, wn, EPS) for i in range(rows)]
    assert all(torch.equal(torch.cat([pair[j] for pair in serial]).view(torch.uint8), actual[j].view(torch.uint8)) for j in (0, 1))

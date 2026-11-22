"""CPU (TRITON_INTERPRET=1) coverage of the vendored V4.1 attention, RoPE/QDQ, norm, packing, top-k and SwiGLU
kernels, against the oracles and tolerances of the ported upstream tests (shapes reduced for the interpreter).

attend_paged_latents / attend_gathered_keys / attend_draft_latents are GPU-only here: their FP32-output
FP16 bmm (aten::bmm.dtype_out) has no CPU kernel; test_deepseek_v41_tp8_heads.py and the ported
test_deepseek_v41_attention_ops.py cover them on CUDA.

Run: TRITON_INTERPRET=1 CUDA_VISIBLE_DEVICES= pytest tests/cuda/deepseek_v41/test_deepseek_v41_attention_interpret.py
"""

import os

import pytest
import torch

pytestmark = pytest.mark.skipif(
    os.environ.get("TRITON_INTERPRET", "0") != "1", reason="Triton interpreter (TRITON_INTERPRET=1) required"
)

from test_deepseek_v41_quant_rope import _reference

from tensorfold.families.deepseek_v4.cuda.kernels import attention_ops as ops
from tensorfold.families.deepseek_v4.cuda.kernels.activations import swiglu
from tensorfold.families.deepseek_v4.cuda.kernels.index_topk import index_topk, index_topk_parallel
from tensorfold.families.deepseek_v4.cuda.kernels.kv_pack import pack_rows, unpack_rows
from tensorfold.families.deepseek_v4.cuda.kernels.norm import rms_norm
from tensorfold.families.deepseek_v4.cuda.kernels.quant_rope import apply_rope, attention_qdq, rope_table


@pytest.mark.parametrize("fp4,group,e4m3", [(False, 128, False), (True, 16, True), (True, 32, False)])
@pytest.mark.parametrize("magnitude", [0.0, 2**-20, 0.03125, 1.37, 100.0])
def test_attention_qdq_exact(fp4, group, e4m3, magnitude):
    x = (torch.randn(6, 512, generator=torch.Generator().manual_seed(41)) * magnitude).half()
    actual = attention_qdq(x, fp4=fp4, group=group, e4m3_scale=e4m3)
    torch.testing.assert_close(actual, _reference(x, fp4, group, e4m3), rtol=0, atol=0)


@pytest.mark.parametrize("dim", [128, 512])
@pytest.mark.parametrize("inverse", [False, True])
def test_rope_including_beyond_table(dim, inverse):
    table = rope_table(max_length=64, compressed=True, device="cpu")
    positions = torch.tensor([0, 1, 33, 63, 64, 200])
    x = torch.randn(6, 2, dim, generator=torch.Generator().manual_seed(41)).half()
    output = apply_rope(x, positions, table, inverse=inverse)
    full = rope_table(max_length=256, compressed=True, device="cpu")
    cosine, sine = full[positions].split(32, dim=-1)
    phase = torch.complex(cosine, -sine if inverse else sine)[:, None]
    expected = x.clone()
    tail = torch.view_as_complex(expected[..., -64:].float().unflatten(-1, (32, 2)))
    expected[..., -64:] = torch.view_as_real(tail * phase).flatten(-2).half()
    torch.testing.assert_close(output, expected, rtol=0.001, atol=2e-6)
    torch.testing.assert_close(output[..., :-64], expected[..., :-64], rtol=0, atol=0)


@pytest.mark.parametrize("hidden", [128, 5120])
def test_short_rms(hidden):
    g = torch.Generator().manual_seed(4190)
    x = (torch.randn(6, hidden, generator=g) * 1.37).half()
    weight = torch.rand(hidden, generator=g) + 0.5
    output = rms_norm(x, weight, 1e-20)
    values = x.float()
    expected = (values * torch.rsqrt(values.square().mean(-1, keepdim=True) + 1e-20) * weight).half()
    torch.testing.assert_close(output, expected, rtol=0.001, atol=2e-6)
    for rows in range(1, 7):
        torch.testing.assert_close(rms_norm(x[:rows], weight, 1e-20), output[:rows], rtol=0, atol=0)


@pytest.mark.parametrize("compressed", [True, False])
@pytest.mark.parametrize("scale", [1e-6, 0.05, 30.0, 3000.0])
def test_kv_pack_bit_exact(compressed, scale):
    dim = 512 if compressed else 128
    x = (torch.randn(16, dim, generator=torch.Generator().manual_seed(0)) * scale).half()
    x[0].zero_()
    x[1, :17] = 0
    ref = attention_qdq(x, fp4=True, group=16 if compressed else 32, e4m3_scale=compressed)
    got = unpack_rows(pack_rows(x, compressed), compressed)
    assert torch.equal(ref.view(torch.int16), got.view(torch.int16))


def test_index_topk_single_and_parallel():
    g = torch.Generator().manual_seed(0)
    rows, width = 3, 20000
    sc = torch.randn(rows, width, generator=g).half()
    sc[:, ::7] = 0.5  # ties
    pos = torch.tensor([19990, 300, 100])
    valid = torch.tensor([True, True, False])
    a = torch.empty(rows, 512, dtype=torch.int32)
    b = torch.empty_like(a)
    index_topk(a, sc, pos, valid, 1, 512)
    index_topk_parallel(b, sc, pos, valid, 1, 512)
    assert torch.equal(a, b)
    # Contract: ascending top-512 of scores[:n]; ties broken toward the lowest index; -1 padding.
    n = 19991
    order = sorted(range(n), key=lambda i: (-float(sc[0, i]), i))[:512]
    assert a[0].tolist() == sorted(order)
    assert a[1].tolist() == list(range(301)) + [-1] * 211
    assert a[2].tolist() == [-1] * 512


@pytest.mark.parametrize("heads", [8, 4])  # TP4 / TP8 indexer heads per rank
def test_top512_masks_future(heads):
    # Upstream test_top512_is_retained_above_short_bypass_threshold_and_masks_future, ratio 2.
    q = torch.zeros(6, heads, 128, dtype=torch.half)
    q[..., 0] = 1
    weights = torch.ones(6, heads, dtype=torch.half) / heads
    keys = torch.zeros(1024, 128, dtype=torch.half)
    keys[:, 0] = torch.arange(1024) / 1024
    positions = torch.tensor([0, 2, 510, 700, 1023, 1400])
    valid = torch.tensor([True, True, True, True, True, False])
    selected = ops.select_compressed_positions(q, weights, keys, positions, valid, 2, lambda x: x)
    for row, position in enumerate(positions.tolist()):
        n = min((position + 1) // 2, 1024) if valid[row] else 0
        actual = selected[row][selected[row] >= 0]
        torch.testing.assert_close(actual, torch.arange(max(n - 512, 0), n, dtype=torch.int32), rtol=0, atol=0)


def test_paged_index_cache_group_boundaries():
    for ratio in (1, 2):
        rows = 16 // ratio
        backing = torch.full((64, rows * 128 + 32), float("nan"), dtype=torch.half)
        cache = backing[:, : rows * 128].view(64, rows, 128)
        table = torch.randperm(64, generator=torch.Generator().manual_seed(1)).int()[None]
        positions = torch.arange(63, 69)
        values = torch.arange(6 * 128).view(6, 128).half()
        valid = torch.ones(6, dtype=torch.bool)
        ops.insert_latents(cache, values, positions, valid, table, block_size=16, ratio=ratio)
        result = ops.gather_index_keys(cache, table, positions, valid, block_size=16, ratio=ratio, max_context=1024)
        for i in range(6):
            p = int(positions[i])
            if (p + 1) % ratio == 0:
                torch.testing.assert_close(result[p // ratio], values[i], rtol=0, atol=0)
        assert not result[69 // ratio :].any()
        assert torch.isnan(backing[:, -32:]).all()


def test_packed_index_rows_unpack_to_qdq_keys():
    raw = (torch.randn(96, 128, generator=torch.Generator().manual_seed(3)) * 0.5).half()
    packed = pack_rows(raw, False)
    positions = torch.arange(90, 96)
    out = ops.unpack_index_keys(packed, positions)
    ref = attention_qdq(raw, fp4=True, group=32)
    assert torch.equal(out[:96].view(torch.int16), ref.view(torch.int16))


def test_swiglu_matches_upstream_formula_in_interpreter():
    # The interpreter's exp is NumPy FP32 exp; the same formula in NumPy is the reference here. GPU bit identity
    # against the upstream CUDA kernel is test_deepseek_v41_activations.py.
    import numpy as np

    gate = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.float16).view(256, 256)
    up = torch.randn(256, 256, generator=torch.Generator().manual_seed(7)).mul(6).half()
    actual = swiglu(gate, up)
    g = gate.float().numpy()
    u = up.float().numpy()
    g = np.where(g > 10, np.float32(10), g)
    u = np.where(u > 10, np.float32(10), u)
    u = np.where(u < -10, np.float32(-10), u)
    with np.errstate(all="ignore"):
        expected = torch.from_numpy((g / (np.float32(1) + np.exp(-g))) * u).half()
    nan = torch.isnan(expected)
    assert torch.equal(torch.isnan(actual), nan)
    assert torch.equal(actual.view(torch.int16)[~nan], expected.view(torch.int16)[~nan])


@pytest.mark.parametrize("heads", [8, 4])
def test_packed_index_scores_decode_heads(heads):
    # select_packed_positions decode (Triton _index_scores_decode) vs the FP16 gather + mm path on the same keys:
    # both apply the official FP16 dot / ReLU*weight / head-sum boundaries; compare the selected sets' scores.
    g = torch.Generator().manual_seed(9)
    rows, T = 1024, 2
    raw = (torch.randn(rows, 128, generator=g) * 0.5).half()
    packed = pack_rows(raw, False)
    keys = attention_qdq(raw, fp4=True, group=32)
    q = attention_qdq((torch.randn(T * heads, 128, generator=g) * 0.5).half(), fp4=True, group=32).view(T, heads, 128)
    w = (torch.randn(T, heads, generator=g) * 0.2).half()
    pos = torch.tensor([900, 901])
    valid = torch.ones(T, dtype=torch.bool)
    new = ops.select_packed_positions(q, w, packed, pos, valid, 1, lambda x: x)
    old = ops.select_compressed_positions(q, w, keys, pos, valid, 1, lambda x: x)
    assert new.shape == old.shape == (T, 512)
    for r in range(T):
        live = new[r][new[r] >= 0]
        assert live.numel() == 512 and bool((live < pos[r] + 1).all()) and bool((live[1:] > live[:-1]).all())


def test_markov_greedy_and_bias():
    # No upstream test exists for markov_fused.py; reference: torch argmax(logits + W_fp32 @ embed(prev)) (lowest
    # index on ties), chained over the steps; and the FP32 bias rows. Sums are FP32 over K in both, so only the
    # reduction order differs: bias is compared at FP32 tolerance and logits are spaced so the argmax is decisive.
    from tensorfold.families.deepseek_v4.cuda.kernels.markov_fused import MarkovGreedy, markov_bias

    g = torch.Generator().manual_seed(11)
    vocab, rank, steps = 300, 32, 5
    weight = (torch.randn(vocab, rank, generator=g) * 0.05).bfloat16()
    embed = torch.randn(vocab, rank, generator=g).half()
    logits = torch.randperm(vocab * steps, generator=g).float().view(steps, vocab)
    greedy = MarkovGreedy(weight, embed, steps=steps)
    tokens = greedy(logits, torch.tensor([7]))
    prev, expected = 7, []
    for step in range(steps):
        bias = weight.float() @ embed[prev].float()
        prev = int(torch.argmax(logits[step] + bias))
        expected.append(prev)
    assert tokens.tolist() == expected
    rows = embed[:3]
    torch.testing.assert_close(markov_bias(greedy, rows), rows.float() @ weight.float().T, rtol=1e-5, atol=1e-5)

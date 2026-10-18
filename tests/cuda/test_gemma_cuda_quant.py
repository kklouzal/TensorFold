"""Original 4/6-bit Gemma payloads, numerical bounds and row-count identity on CUDA."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from gemma_cuda_quant_reference import pack_codes
from tensorfold.families.gemma4.cuda.experts import ExpertLinear, Experts
from tensorfold.families.gemma4.cuda.linear import Linear


def _weights(shape, bits, seed):
    rng = np.random.default_rng(seed)
    codes = rng.integers(0, 1 << bits, shape, dtype=np.int32)
    packed = torch.from_numpy(pack_codes(codes, bits)).cuda()
    scales = torch.from_numpy(rng.uniform(0.001, 0.02, (*shape[:-1], shape[-1] // 64)).astype(np.float32))
    scales = scales.cuda().to(torch.bfloat16)
    biases = torch.from_numpy(rng.normal(-0.03, 0.02, scales.shape).astype(np.float32)).cuda().to(torch.bfloat16)
    return packed, scales, biases, torch.from_numpy(codes).cuda()


def _reference(x, codes, scales, biases):
    w = codes.double() * scales.double().repeat_interleave(64, dim=-1)
    w += biases.double().repeat_interleave(64, dim=-1)
    return x.double() @ w.transpose(-1, -2)


def _assert_fp32_accum_bf16(got, truth, x, codes, scales, biases):
    # A conservative FP32 gamma bound for the fixed group accumulation, plus
    # one BF16 rounding. This derives from operation count and magnitudes,
    # rather than using an empirical tolerance fitted to this implementation.
    k = x.shape[-1]
    u = 2.0 ** -24
    operations = 4 * k + 8 * (k // 64)
    gamma = operations * u / (1.0 - operations * u)
    wabs = codes.double() * scales.double().abs().repeat_interleave(64, dim=-1)
    wabs += biases.double().abs().repeat_interleave(64, dim=-1)
    accumulation = x.double().abs() @ wabs.transpose(-1, -2)
    bound = gamma * accumulation + truth.abs() * (2.0 ** -8) + torch.finfo(torch.bfloat16).tiny
    assert torch.all((got.double() - truth).abs() <= bound)


@pytest.mark.parametrize('bits', [4, 6])
@pytest.mark.parametrize('n,k', [(19, 64), (128, 2816), (704, 2816), (2816, 704)])
def test_linear_preserves_native_codes_and_fp32_affine_contract(bits, n, k):
    words, scales, biases, codes = _weights((n, k), bits, bits + n + k)
    saved = [v.clone() for v in (words, scales, biases)]
    linear = Linear(words, scales, biases, bits=bits)
    x = torch.randn((7, k), generator=torch.Generator(device='cuda').manual_seed(71), device='cuda')
    x = x.to(torch.bfloat16)
    got = linear(x)
    truth = _reference(x, codes, scales, biases)
    _assert_fp32_accum_bf16(got, truth, x, codes, scales, biases)
    assert all(torch.equal(old, new) for old, new in zip(saved, (words, scales, biases)))
    if bits == 6:
        assert linear.weight.data_ptr() == words.data_ptr()
    else:
        from tensorfold.cuda.kernels.qmm import unpack

        back = unpack(linear._q4)
        assert all(torch.equal(a.view(torch.int32) if i == 0 else a, b)
                   for i, (a, b) in enumerate(zip(saved, back)))


@pytest.mark.parametrize('bits', [4, 6])
def test_linear_rows_are_identical_alone_batched_permuted_and_strided(bits):
    words, scales, biases, _ = _weights((137, 2816), bits, 18)
    linear = Linear(words, scales, biases, bits)
    x = torch.randn((65, 2816), generator=torch.Generator(device='cuda').manual_seed(1), device='cuda')
    x = x.to(torch.bfloat16)
    serial = torch.cat([linear(row[None]) for row in x])
    for count in (1, 2, 3, 7, 15, 16, 17, 31, 32, 33, 64, 65):
        assert torch.equal(linear(x[:count]), serial[:count])
    order = torch.randperm(len(x), generator=torch.Generator().manual_seed(29)).cuda()
    assert torch.equal(linear(x[order]), serial[order])
    storage = torch.empty((65, 5632), device='cuda', dtype=torch.bfloat16)
    storage[:, ::2] = x
    assert torch.equal(linear(storage[:, ::2]), serial)


def test_tied_six_bit_embedding_reads_exact_original_codes_across_word_boundaries():
    words, scales, biases, codes = _weights((31, 2816), 6, 91)
    linear = Linear(words, scales, biases, 6)
    ids = torch.tensor([[30, 0, 7], [1, 19, 11]], device='cuda', dtype=torch.int64)
    got = linear.embedding(ids)
    ref = (codes.double() * scales.double().repeat_interleave(64, dim=-1)
           + biases.double().repeat_interleave(64, dim=-1)).to(torch.bfloat16)
    assert torch.equal(got, ref[ids])
    assert linear.nbytes() == sum(v.numel() * v.element_size() for v in (words, scales, biases))


@pytest.mark.parametrize('bits', [4, 6])
def test_gathered_expert_projection_has_independent_row_and_slot_semantics(bits):
    words, scales, biases, codes = _weights((9, 19, 704), bits, 43)
    bank = ExpertLinear(words, scales, biases, bits)
    x = torch.randn((4, 3, 704), device='cuda').to(torch.bfloat16)
    ids = torch.tensor([[8, 0, 2], [3, 8, 1], [4, 0, 4], [2, 5, 7]], device='cuda', dtype=torch.int32)
    got = bank(x, ids)
    for row in range(4):
        for slot in range(3):
            expert = int(ids[row, slot])
            xx = x[row, slot:slot + 1]
            ref = _reference(xx, codes[expert], scales[expert], biases[expert])
            _assert_fp32_accum_bf16(got[row, slot:slot + 1], ref, xx,
                                  codes[expert], scales[expert], biases[expert])
    serial = torch.cat([bank(x[r:r + 1], ids[r:r + 1]) for r in range(4)])
    assert torch.equal(got, serial)
    shared = bank(x[:, 0], ids)
    expanded = bank(x[:, :1].expand(-1, 3, -1), ids)
    assert torch.equal(shared, expanded)


@pytest.mark.parametrize('down_bits,expert_count,width,slots', [(4, 9, 128, 3), (6, 9, 128, 3), (6, 128, 704, 8)])
def test_geglu_and_weighted_slots_match_separate_contract_and_serial_rows(down_bits, expert_count, width, slots):
    gate = ExpertLinear(*_weights((expert_count, width, 192), 4, 81)[:3], bits=4)
    up = ExpertLinear(*_weights((expert_count, width, 192), 4, 82)[:3], bits=4)
    down = ExpertLinear(*_weights((expert_count, 192, width), down_bits, 83)[:3], bits=down_bits)
    experts = Experts(gate, up, down)
    x = torch.randn((5, 192), device='cuda').to(torch.bfloat16)
    picks = torch.tensor([[8, 0, 3], [0, 1, 2], [3, 5, 8], [1, 8, 3], [0, 3, 2]], device='cuda')
    if slots == 8:
        picks = torch.tensor([[127, 0, 3, 14, 91, 44, 111, 2]], device='cuda').expand(5, -1)
    weights = torch.linspace(1.0, 0.125, slots, device='cuda', dtype=torch.bfloat16)[None].expand(5, -1)
    g, u = gate(x, picks).float(), up(x, picks).float()
    inner = 0.7978845608028654 * (g + 0.044715 * g * g * g)
    act = (0.5 * g * (1.0 + torch.tanh(inner)) * u).to(torch.bfloat16)
    projected = down(act, picks)
    terms = (projected.float() * weights[:, :, None].float()).to(torch.bfloat16).float()
    expected = torch.zeros((5, 192), device='cuda', dtype=torch.float32)
    for slot in range(slots):
        expected = expected + terms[:, slot]
    got = experts.forward(x, picks, weights)
    assert torch.equal(got, expected.to(torch.bfloat16))
    assert torch.equal(got, torch.cat([experts.forward(x[i:i + 1], picks[i:i + 1], weights[i:i + 1])
                                      for i in range(5)]))


def test_six_bit_full_vocabulary_head_uses_valid_cuda_launch_geometry():
    # The real262144-output tied head has65536 four-output tiles. CUDA gridY
    # is limited to65535; the vocabulary tiles must occupy gridX.
    n, k = 262144, 64
    words, scales, biases, codes = _weights((n, k), 6, 97)
    head = Linear(words, scales, biases, bits=6)
    x = torch.randn((2, k), generator=torch.Generator(device='cuda').manual_seed(37), device='cuda')
    x = x.to(torch.bfloat16)
    got = head(x)
    truth = _reference(x, codes, scales, biases)
    _assert_fp32_accum_bf16(got, truth, x, codes, scales, biases)
    assert torch.equal(got, torch.cat([head(row[None]) for row in x]))

"""Gemma CT native A4 versus independent prepared operators and original bytes."""

from __future__ import annotations

import pytest
import torch

from nvfp4_a4_inputs import Context, PREFIX, ct_quant_reference, linears
from test_nvfp4_a4_host_control import oracle_blob

if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip('native Gemma activation FP4 requires SM12', allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.experts_a4 import pack_projection_a4
from tensorfold.cuda.nvfp4.grouped_a4 import _ext
from tensorfold.families.gemma4.cuda.nvfp4_experts import _combine, geglu, load_experts


@pytest.mark.parametrize('n,k', [(64, 64), (704, 2816), (2816, 704)])
def test_shared_gpu_pack_preserves_every_original_byte_against_scalar_cpu_oracle(n, k):
    generator = torch.Generator(device='cpu').manual_seed(98031 + n + k)
    raw = torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, generator=generator)
    scales = torch.randint(0, 256, (n, k // 16), dtype=torch.uint8, generator=generator)
    # Exercise every byte encoding once without periodic rows that could hide
    # incorrect row permutations at the actual704/2816 projection widths.
    raw.view(-1)[:256] = torch.arange(256, dtype=torch.uint8)
    scales.view(-1)[:256] = torch.arange(256, dtype=torch.uint8)
    expected = oracle_blob(raw, scales.view(torch.float8_e4m3fn))
    gpu = pack_projection_a4(raw.cuda(), scales.cuda().view(torch.float8_e4m3fn))
    assert torch.equal(gpu.cpu(), expected)
    assert torch.equal(pack_projection_a4(raw, scales.view(torch.float8_e4m3fn)), expected)


def _prepared(ctx):
    # The real target owner preloads and validates all original globals once.
    # Derive these test tuples independently from original CPU payload values.
    result = {}
    for expert in range(ctx.cfg.experts):
        for projection in ('gate_proj', 'up_proj', 'down_proj'):
            base = f'{PREFIX}.experts.{expert}.{projection}'
            weight = float(ctx.payload[base + '.weight_global_scale'][0])
            inverse = float(ctx.payload[base + '.input_global_scale'][0])
            result[base] = 1.0 / weight, 1.0 / inverse, inverse
    return result


def _reference(ctx, x, picks, weights):
    banks = [linears(ctx, expert) for expert in range(ctx.cfg.experts)]
    out = torch.zeros_like(x)
    for row in range(len(x)):
        order = sorted(enumerate(picks[row].cpu().tolist()), key=lambda pair: pair[1])
        for slot, expert in order:
            g, u, d = banks[expert]
            gate = checkpoint.prompt(checkpoint.A4, x[row:row + 1], g)
            up = checkpoint.prompt(checkpoint.A4, x[row:row + 1], u)
            # Independent eager BF16 GELU and product, matching HF storage.
            activated = torch.nn.functional.gelu(gate, approximate='tanh')
            hidden = activated * up
            down = checkpoint.prompt(checkpoint.A4, hidden, d)
            term = (down.float() * weights[row, slot]).to(torch.bfloat16)
            out[row:row + 1] = out[row:row + 1] + term
    return out


@pytest.mark.parametrize('count,width,dims,slots', [(3, 64, 128, 3), (3, 704, 2816, 3), (128, 64, 128, 8)])
def test_original_ct_calibrations_geglu_and_hf_combine_are_exact_and_row_invariant(count, width, dims, slots):
    source = Context(count=count, width=width, dims=dims)
    bank = load_experts(source.read, PREFIX + '.experts', count=count, width=width, dims=dims,
                        device='cuda:0', calibrated=_prepared(source))
    assert len(source.reads) == count * 6
    assert not any(name.endswith("global_scale") for name in source.reads)
    fields = bank.bank.fields
    for expert in range(count):
        for p, projection in enumerate(('gate_proj', 'up_proj', 'down_proj')):
            base = f'{PREFIX}.experts.{expert}.{projection}'
            expect = source.payload[base + '.input_global_scale']
            actual = fields[4][expert, p:p + 1] if p < 2 else fields[5][expert]
            assert torch.equal(actual.cpu().view(torch.int32), expect.view(torch.int32))
    x = (torch.randn((3, dims), generator=torch.Generator().manual_seed(391)) * .3).cuda().to(torch.bfloat16)
    x[0] = .5  # original CT multiplier0x3e63ffff exercises the quantizer boundary
    picks = torch.tensor([list(range(count - 1, count - 1 - slots, -1))], device='cuda', dtype=torch.int32)
    picks = picks.expand(len(x), -1).contiguous()
    weights = torch.linspace(.03125, .812345, slots, dtype=torch.float32, device='cuda')[None].expand(len(x), -1)
    want = _reference(Context(count=count, width=width, dims=dims), x, picks, weights)
    got = bank.forward(x, picks, weights)
    assert torch.equal(got, want)
    serial = torch.cat([bank.forward(x[r:r + 1], picks[r:r + 1], weights[r:r + 1]) for r in range(len(x))])
    assert torch.equal(got, serial)
    reverse = torch.arange(len(x) - 1, -1, -1, device='cuda')
    assert torch.equal(bank.forward(x[reverse], picks[reverse], weights[reverse]), got[reverse])
    assert bank.nbytes() == sum(t.numel() * t.element_size() for t in fields)
    assert bank.bank.swiglu is geglu


def test_original_ct_input_multiplier_matches_independent_boundary_quantizer():
    source = Context()
    bank = load_experts(source.read, PREFIX + '.experts', count=3, width=64, dims=128,
                        device='cuda:0', calibrated=_prepared(source))
    x = torch.full((1, 128), .5, dtype=torch.bfloat16, device='cuda')
    plan = torch.tensor([[0, 0, 0]], device='cuda', dtype=torch.int32)
    codes = torch.empty((2, 1, 64), dtype=torch.uint8, device='cuda')
    scales = torch.empty((2, 1, 8), dtype=torch.uint8, device='cuda')
    _ext().quantize(x, plan, bank.bank.fields[4], codes, scales, False, False)
    raw = float(source.payload[PREFIX + '.experts.0.gate_proj.input_global_scale'][0])
    expected_codes, expected_scales = ct_quant_reference(x.cpu(), raw)
    assert torch.equal(codes[0].cpu(), expected_codes)
    assert torch.equal(scales[0].cpu(), expected_scales)
    assert torch.equal(scales[0], torch.full_like(scales[0], 0x0a))


def test_geglu_has_bf16_activation_before_product_and_differs_from_one_round_product():
    gate = torch.linspace(-4., 4., 2048, device='cuda').to(torch.bfloat16)
    up = torch.linspace(.125, 3., 2048, device='cuda').to(torch.bfloat16)
    actual = geglu(gate, up)
    expected = torch.nn.functional.gelu(gate, approximate='tanh') * up
    one_round = (torch.nn.functional.gelu(gate.float(), approximate='tanh') * up.float()).to(torch.bfloat16)
    assert torch.equal(actual, expected)
    assert not torch.equal(actual, one_round)


def test_routing_keeps_fp32_weights_and_rounds_each_ascending_expert_addition():
    import triton

    # Deliberately shuffled IDs and cancellation distinguish expert order from
    # selected-slot order, and BF16 per-add storage from an FP32 slot reduction.
    y = torch.tensor([[[1.], [256.], [-256.], [.25]]], dtype=torch.bfloat16, device='cuda')
    picks = torch.tensor([[2, 0, 1, 3]], dtype=torch.int32, device='cuda')
    weights = torch.tensor([[1., 1., 1., 1.0038]], device='cuda', dtype=torch.float32)
    got = torch.empty((1, 1), device='cuda', dtype=torch.bfloat16)
    _combine[(1, 1)](y, picks, weights, got, N=1, R=4, RS=triton.next_power_of_2(4), BLOCK=256,
                     num_warps=4, enable_fp_fusion=False)
    expected = torch.zeros_like(got)
    for slot in (1, 2, 0, 3):
        expected += (y[:, slot].float() * weights[:, slot, None]).to(torch.bfloat16)
    wrong_slot = torch.zeros_like(got)
    for slot in range(4):
        wrong_slot += (y[:, slot].float() * weights[:, slot, None]).to(torch.bfloat16)
    assert torch.equal(got, expected)
    assert not torch.equal(got, wrong_slot)


def test_loader_rejects_invalid_original_scale_bytes_before_bank_publication():
    source = Context()
    source.payload[PREFIX + '.experts.2.down_proj.weight_scale'].view(torch.uint8)[-1, -1] = 0x80
    with pytest.raises(ValueError, match='finite unsigned E4M3'):
        load_experts(source.read, PREFIX + '.experts', count=3, width=64, dims=128,
                     device='cuda:0', calibrated=_prepared(source))


def test_routing_weight_is_not_narrowed_to_bf16_before_multiplication():
    y = torch.tensor([[[3.]]], device='cuda', dtype=torch.bfloat16)
    ids = torch.zeros((1, 1), device='cuda', dtype=torch.int32)
    weights = torch.tensor([[1.0038]], device='cuda', dtype=torch.float32)
    got = torch.empty((1, 1), device='cuda', dtype=torch.bfloat16)
    _combine[(1, 1)](y, ids, weights, got, N=1, R=1, RS=1, BLOCK=256,
                     num_warps=4, enable_fp_fusion=False)
    expected = (y[:, 0].float() * weights).to(torch.bfloat16)
    wrong = (y[:, 0].float() * weights.to(torch.bfloat16).float()).to(torch.bfloat16)
    assert torch.equal(got, expected)
    assert not torch.equal(got, wrong)

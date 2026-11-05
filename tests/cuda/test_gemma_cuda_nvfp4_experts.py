"""Gemma CT native A4 versus independent prepared operators and original bytes."""

from __future__ import annotations

import pytest
import torch

from nvfp4_a4_inputs import Context, PREFIX, ct_quant_reference, linears, plan_order_positions
from test_nvfp4_a4_host_control import oracle_blob

if torch.cuda.get_device_capability()[0] != 12:
    pytest.skip('native Gemma activation FP4 requires SM12', allow_module_level=True)

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.experts_a4 import PlanBuffers, pack_projection_a4
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


@pytest.mark.parametrize('uniform', [False, True])
@pytest.mark.parametrize('count,width,dims,slots', [(3, 64, 128, 3), (3, 704, 2816, 3), (128, 64, 128, 8)])
def test_original_ct_calibrations_geglu_and_hf_combine_are_exact_and_row_invariant(count, width, dims, slots, uniform):
    source = Context(count=count, width=width, dims=dims, uniform=uniform)
    bank = load_experts(source.read, PREFIX + '.experts', count=count, width=width, dims=dims,
                        device='cuda:0', calibrated=_prepared(source))
    # The loader's CPU bit proof selects one activation quantization per row exactly for a shared gate/up global.
    assert bank.bank.shared_input == uniform
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
    want = _reference(Context(count=count, width=width, dims=dims, uniform=uniform), x, picks, weights)
    got = bank.forward(x, picks, weights)
    assert torch.equal(got, want)
    # Picks routed straight into the workspace's rows take the same plan without a copy.
    in_place = bank.picks_for(len(x), slots)
    in_place.copy_(picks)
    assert torch.equal(bank.forward(x, in_place, weights), want)
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


def test_retained_workspace_never_exceeds_the_admitted_expert_workspace():
    from types import SimpleNamespace

    from tensorfold.families.gemma4.cuda.capacity import expert_workspace

    count, width, dims, slots = 128, 64, 128, 8
    # The real checkpoint shares one gate/up input global per layer: the shared-input row plan is retained too.
    source = Context(count=count, width=width, dims=dims, uniform=True)
    bank = load_experts(source.read, PREFIX + '.experts', count=count, width=width, dims=dims,
                        device='cuda:0', calibrated=_prepared(source))
    assert bank.bank.shared_input
    generator = torch.Generator().manual_seed(17)
    for rows in (1, 5, 17, 129):
        x = (torch.randn((rows, dims), generator=generator) * .3).cuda().to(torch.bfloat16)
        picks = torch.stack([torch.randperm(count, generator=generator)[:slots] for _ in range(rows)]).cuda().int()
        weights = torch.full((rows, slots), .125, device='cuda')
        bank.forward(x, picks, weights)
    retained = 0
    for buf in bank.workspace._buffers.values():
        tensors = [buf.pick, buf.y, buf.plan.members, buf.plan.items, buf.plan.counts, buf.plan.rank, buf.plan.hist,
                   *(t for scratch in buf.a4_scratch.values() for t in scratch)]
        tensors += [buf.a4_row_plan] if getattr(buf, 'a4_row_plan', None) is not None else []
        retained += sum(t.numel() * t.element_size() for t in tensors)
    cfg = SimpleNamespace(hidden=dims, moe_width=width, experts=count, top_k=slots)

    def bucket(size):
        return expert_workspace(cfg, size) - (expert_workspace(cfg, size // 2) if size > 16 else 0)

    # Rows 1 and 5 share bucket 16, 17 takes 32 and 129 takes 256: the admitted formula mirrors each exactly.
    assert sorted(key[0] for key in bank.workspace._buffers) == [16, 32, 256]
    assert retained == bucket(16) + bucket(32) + bucket(256)
    assert retained <= expert_workspace(cfg, 129)


@pytest.mark.parametrize('rows', [1, 2, 3, 4, 5])
@pytest.mark.parametrize('width,dims', [(64, 128), (704, 2816)])
def test_grouped_down_encoding_matches_independent_linears_and_gemma_combine(rows, width, dims):
    source = Context(count=3, width=width, dims=dims, uniform=True)
    adapter = load_experts(source.read, PREFIX + '.experts', count=3, width=width, dims=dims,
                           device='cuda:0', calibrated=_prepared(source))
    bank = adapter.bank
    x = (torch.randn((rows, dims), generator=torch.Generator().manual_seed(199 + rows)) * .3).cuda().bfloat16()
    x[0] = .5
    picks = torch.tensor([[2, 0, 1]], dtype=torch.int32, device='cuda').expand(rows, -1).contiguous()
    weights = torch.tensor([[.03125, .812345, 1.0038]], device='cuda').expand(rows, -1)
    buf = PlanBuffers(rows, 3, 3, dims, 'cuda', prefill=True)
    buf.pick.copy_(picks)
    buf.route(rows)
    bank.run_plan(x, buf, prefill=True)
    held = buf.a4_scratch[(dims, width)]
    # Separate checkpoint linears + CPU quantizer independently check every
    # down-code/scale byte across both halves of every original N64 tile.
    original = Context(count=3, width=width, dims=dims, uniform=True)
    prepared = [linears(original, expert) for expert in range(3)]
    # Fused down encodings are stored in grouped plan order; only down_plan
    # scatters them through members back to original row/slot output positions.
    plan_positions = plan_order_positions(buf.plan.members[:rows * 3].cpu().tolist())
    for row in range(rows):
        for slot, expert in enumerate((2, 0, 1)):
            gate, up, down = prepared[expert]
            g = checkpoint.prompt(checkpoint.A4, x[row:row + 1], gate)
            u = checkpoint.prompt(checkpoint.A4, x[row:row + 1], up)
            hidden = torch.nn.functional.gelu(g, approximate='tanh') * u
            expected_slot = checkpoint.prompt(checkpoint.A4, hidden, down)
            assert torch.equal(buf.y[row, slot].view(torch.uint8), expected_slot[0].view(torch.uint8))
            raw = float(original.payload[f'{PREFIX}.experts.{expert}.down_proj.input_global_scale'][0])
            codes, scales = ct_quant_reference(hidden.cpu(), raw)
            at = plan_positions[row * 3 + slot]
            assert torch.equal(held[2][0, at].cpu(), codes[0])
            assert torch.equal(held[3][0, at].cpu(), scales[0])
    want = _reference(original, x, picks, weights)
    got = adapter.forward(x, picks, weights)
    assert torch.equal(got.view(torch.uint8), want.view(torch.uint8))
    serial = torch.cat([adapter.forward(x[r:r + 1], picks[r:r + 1], weights[r:r + 1]) for r in range(rows)])
    assert torch.equal(serial.view(torch.uint8), got.view(torch.uint8))


def test_gemma_actual_shape_graph_replays_new_routes_with_stable_workspace():
    source = Context(count=4, width=704, dims=2816, uniform=True)
    adapter = load_experts(source.read, PREFIX + '.experts', count=4, width=704, dims=2816,
                           device='cuda:0', calibrated=_prepared(source))
    bank = adapter.bank
    rows, slots = 5, 3
    x = torch.zeros((rows, 2816), device='cuda', dtype=torch.bfloat16)
    buf = PlanBuffers(rows, slots, 4, 2816, 'cuda', prefill=True)
    buf.pick.copy_(torch.tensor([[2, 0, 1]], device='cuda').expand(rows, -1))
    buf.route(rows)
    bank.run_plan(x, buf, prefill=True)
    retained = tuple(t.data_ptr() for t in buf.a4_scratch[(2816, 704)])
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        buf.route(rows)
        bank.run_plan(x, buf, prefill=True)
    for seed in (11, 12, 13):
        generator = torch.Generator().manual_seed(seed)
        x.copy_((torch.randn(rows, 2816, generator=generator) * .4).to('cuda', torch.bfloat16))
        picks = torch.stack([torch.randperm(4, generator=generator)[:slots] for _ in range(rows)]).cuda().int()
        buf.pick.copy_(picks)
        graph.replay()
        replayed = buf.y.clone()
        codes, scales = (t.clone() for t in buf.a4_scratch[(2816, 704)][2:])
        bank.run_plan(x, buf, prefill=True)  # default reference on same immutable source bank
        assert torch.equal(replayed.view(torch.uint8), buf.y.view(torch.uint8))
        assert torch.equal(codes, buf.a4_scratch[(2816, 704)][2])
        assert torch.equal(scales, buf.a4_scratch[(2816, 704)][3])
        assert retained == tuple(t.data_ptr() for t in buf.a4_scratch[(2816, 704)])
    torch.cuda.synchronize()


def test_fused_tail_consumes_real_borrowed_slots_and_replays_changed_routes():
    from tensorfold.families.gemma4.cuda import glue

    # Actual admitted bank/hidden/top8 geometry; underlying native math has retained independent oracles above.
    source = Context(count=128, width=704, dims=2816, uniform=True)
    adapter = load_experts(source.read, PREFIX + '.experts', count=128, width=704, dims=2816,
                           device='cuda:0', calibrated=_prepared(source))
    generator = torch.Generator().manual_seed(8123)
    norm = [(.8 + torch.rand(2816, generator=generator) * .4).cuda().bfloat16() for _ in range(4)]
    scalar = torch.tensor([.87], device='cuda', dtype=torch.bfloat16)
    for rows in range(1, 6):
        x = (torch.randn(rows, 2816, generator=generator) * .3).cuda().bfloat16()
        picks = torch.stack([torch.randperm(128, generator=generator)[:8] for _ in range(rows)]).cuda().int()
        weights = torch.linspace(.03125, .812345, 8, device='cuda')[None].expand(rows, -1).contiguous()
        residual = (torch.randn(rows, 2816, generator=generator) * 3).cuda().bfloat16()
        dense = (torch.randn(rows, 2816, generator=generator) * 2).cuda().bfloat16()
        borrowed = adapter.forward_slots(x, picks, weights)
        buf = borrowed.owner
        pointers = (buf.y.data_ptr(), buf.pick.data_ptr(), *(t.data_ptr() for t in buf.a4_scratch[(2816, 704)]))
        assert borrowed.y.data_ptr() == buf.y.data_ptr() and borrowed.picks.data_ptr() == buf.pick.data_ptr()
        actual = glue.moe_tail_slots(residual, dense, borrowed.y, borrowed.picks, borrowed.routing,
            *norm[:3], scalar, norm[3], 1e-6)
        expected = glue.moe_tail(residual, dense, adapter.forward(x, picks, weights), *norm[:3], scalar, norm[3], 1e-6, False)
        assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(actual, expected))
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            held = adapter.forward_slots(x, picks, weights)
            replayed = glue.moe_tail_slots(residual, dense, held.y, held.picks, held.routing,
                *norm[:3], scalar, norm[3], 1e-6)
        for seed in (8124, 8125):
            changed = torch.Generator().manual_seed(seed)
            x.copy_((torch.randn(rows, 2816, generator=changed) * .3).cuda().bfloat16())
            picks.copy_(torch.stack([torch.randperm(128, generator=changed)[:8] for _ in range(rows)]).cuda().int())
            weights.copy_(torch.rand(rows, 8, generator=changed).cuda())
            graph.replay()
            # This reference reuses/overwrites the same slot workspace AFTER the fused consumer was enqueued.
            expected = glue.moe_tail(residual, dense, adapter.forward(x, picks, weights),
                *norm[:3], scalar, norm[3], 1e-6, False)
            assert all(torch.equal(a.view(torch.uint8), b.view(torch.uint8)) for a, b in zip(replayed, expected))
            assert pointers == (buf.y.data_ptr(), buf.pick.data_ptr(), *(t.data_ptr() for t in buf.a4_scratch[(2816, 704)]))
        assert adapter.workspace.buffers(rows, 8, 128, 2816, 'cuda:0') is buf

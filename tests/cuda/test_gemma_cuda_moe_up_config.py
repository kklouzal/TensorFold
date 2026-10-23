"""Private Gemma-up configurations versus independent operators and replay."""
from itertools import product

import pytest
import torch
from nvfp4_a4_inputs import PREFIX, Context, ct_quant_reference, linears

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip('native Gemma-up configuration checks require SM12', allow_module_level=True)

from test_gemma_cuda_moe_up_layout import reference_pair_members
from test_gemma_cuda_nvfp4_experts import _prepared, _reference
from test_nvfp4_a4_device_plan import _route

from tensorfold.cuda.nvfp4 import checkpoint
from tensorfold.cuda.nvfp4.experts_a4 import PlanBuffers
from tensorfold.cuda.nvfp4.grouped_a4 import _ext
from tensorfold.families.gemma4.cuda.nvfp4_experts import _combine, load_experts

CONFIGS = tuple(product((2, 4), (2, 3, 4)))


def _case(width, dims, rows, uniform):
    source = Context(count=3, width=width, dims=dims, uniform=uniform)
    wrapped = load_experts(source.read, PREFIX + '.experts', count=3, width=width, dims=dims,
                           device='cuda:0', calibrated=_prepared(source))
    bank = wrapped.bank
    buf = PlanBuffers(rows + 3, 3, 3, dims, 'cuda:0', prefill=True)
    generator = torch.Generator().manual_seed(1137 + rows + width)
    x = (torch.randn((rows, dims), generator=generator) * .3).cuda().to(torch.bfloat16)
    x[0] = .5  # Independent original-CT multiplier rounding boundary.
    picks = torch.stack([torch.tensor([(row + slot) % 3 for slot in (2, 0, 1)]) for row in range(rows)]).cuda().int()
    weights = torch.tensor([.03125, .312345, .812345], device='cuda', dtype=torch.float32)[None].expand(rows, -1).contiguous()
    buf.pick[:rows].copy_(picks)
    _route(buf, rows, 16)
    bank.run_plan(x, buf, prefill=True)  # Ordinary eager initialization/quantization, never a measured candidate call.
    return source, bank, buf, x, picks, weights


def _arguments(bank, buf, rows):
    pairs = rows * buf.slots
    scratch = buf.a4_scratch[(bank.dims, bank.width)]
    common = bank.shared_input and rows <= 65535
    projections, count = (1, rows) if common else (2, pairs)
    codes = scratch[0].view(-1)[:projections * count * (bank.dims // 2)].view(projections, count, bank.dims // 2)
    scales = scratch[1].view(-1)[:projections * count * (bank.dims // 16)].view(projections, count, bank.dims // 16)
    dc, ds = (value[:, :pairs] for value in scratch[2:])
    items = buf.plan.items[:min(pairs, bank.count + 1 + pairs // 16)]
    members = buf.plan.members[:pairs]
    fields = bank.fields
    return (codes, scales, fields[0], fields[2], fields[4], fields[5], items, buf.plan.counts,
            members, dc, ds, bank.width, bank.dims, buf.slots, common)


def _launch(bank, buf, rows, warps, stages):
    args = _arguments(bank, buf, rows)
    ext = _ext()
    ext.gemma_fused_up(*args, warps, stages)
    fields = bank.fields
    ext.down_plan(args[9], args[10], bank.native_down, fields[3], fields[5], args[6], args[7], args[8],
                  buf.y[:rows], bank.dims, bank.width, 1, buf.slots, False, 16)


def _quant_oracle(source, x, picks):
    # Separate prepared GEMMs and eager BF16 GeGLU, followed by the existing
    # independent CPU per16 CT quantizer. Scratch is in grouped-plan pair
    # coordinates; derive those from original picks, never the native plan.
    banks = [linears(source, expert) for expert in range(3)]
    selected = picks.cpu().tolist()
    members = reference_pair_members(selected)
    slots = len(selected[0])
    codes, scales = [], []
    for member in members:
        row, slot = divmod(member, slots)
        expert = selected[row][slot]
        gate, up, _down = banks[expert]
        g = checkpoint.prompt(checkpoint.A4, x[row:row + 1], gate)
        u = checkpoint.prompt(checkpoint.A4, x[row:row + 1], up)
        hidden = torch.nn.functional.gelu(g, approximate='tanh') * u
        multiplier = float(source.payload[f'{PREFIX}.experts.{expert}.down_proj.input_global_scale'][0])
        c, s = ct_quant_reference(hidden.cpu(), multiplier)
        codes.append(c)
        scales.append(s)
    return torch.cat(codes)[None].cuda(), torch.cat(scales)[None].cuda(), members


@pytest.mark.parametrize('uniform', (False, True))
@pytest.mark.parametrize('width,dims,rows', ((64, 64, 1), (64, 128, 1), (704, 2816, 5), (64, 128, 17)))
def test_all_six_configs_match_every_code_scale_and_independent_final_output(width, dims, rows, uniform):
    source, bank, buf, x, picks, weights = _case(width, dims, rows, uniform)
    args = _arguments(bank, buf, rows)
    baseline_codes, baseline_scales, baseline_y = args[9].clone(), args[10].clone(), buf.y[:rows].clone()
    expected_codes, expected_scales, expected_members = _quant_oracle(source, x, picks)
    expected = _reference(Context(count=3, width=width, dims=dims, uniform=uniform), x, picks, weights)
    assert args[8].cpu().tolist() == expected_members
    assert torch.equal(baseline_codes, expected_codes)
    assert torch.equal(baseline_scales, expected_scales)
    for warps, stages in CONFIGS:
        args[9].fill_(0xa5)
        args[10].fill_(0x5a)
        _launch(bank, buf, rows, warps, stages)
        assert torch.equal(args[9], expected_codes), (warps, stages)
        assert torch.equal(args[10], expected_scales), (warps, stages)
        assert torch.equal(buf.y[:rows], baseline_y), (warps, stages)
        out = torch.empty_like(x)
        _combine[(rows, (dims + 255) // 256)](buf.y, buf.pick, weights, out,
            N=dims, R=3, RS=4, BLOCK=256, num_warps=4, enable_fp_fusion=False)
        assert torch.equal(out, expected), (warps, stages)


@pytest.mark.parametrize('warps,stages', CONFIGS)
@pytest.mark.parametrize('uniform', (False, True))
def test_graph_retains_config_and_replays_changed_routes(warps, stages, uniform):
    _source, bank, buf, x, picks, _weights = _case(64, 128, 17, uniform)
    _launch(bank, buf, 17, warps, stages)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch(bank, buf, 17, warps, stages)
    # The all-zero route also changes the active item count and creates more
    # than16 pairs for one expert; this is valid native-plan input.
    for routes in ((picks + 1) % 3, torch.zeros_like(picks), (picks + 2) % 3):
        x.add_(.03125)
        buf.pick[:17].copy_(routes)
        _route(buf, 17, 16)
        bank.run_plan(x, buf, prefill=True)
        expected = buf.y[:17].clone()
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(buf.y[:17], expected)


@pytest.mark.parametrize('warps,stages', ((0, 2), (1, 2), (8, 2), (2, 1), (2, 5), (4, -1)))
def test_invalid_configuration_stops_before_any_launch(warps, stages):
    # CPU tensors cannot be launched. Config admission runs before tensor or
    # CUDA-guard work, and must report its specific failure without a launch.
    empty = torch.empty(0, dtype=torch.uint8)
    with pytest.raises(RuntimeError, match='private Gemma-up requires'):
        _ext().gemma_fused_up(*([empty] * 11), 64, 128, 3, True, warps, stages)


@pytest.mark.parametrize('damage', ('codes-dtype', 'down-scale-shape', 'partial-n', 'zero-slots'))
def test_private_entry_retains_existing_tensor_plan_boundaries(damage):
    _source, bank, buf, _x, _picks, _weights = _case(64, 128, 1, True)
    args = list(_arguments(bank, buf, 1))
    if damage == 'codes-dtype':
        args[0] = args[0].float()
    elif damage == 'down-scale-shape':
        args[10] = args[10][..., :-1].contiguous()
    elif damage == 'partial-n':
        args[11] = 65
    else:
        args[13] = 0
    before = (args[9].clone(), _arguments(bank, buf, 1)[10].clone())
    with pytest.raises(RuntimeError):
        _ext().gemma_fused_up(*args, 2, 3)
    assert torch.equal(_arguments(bank, buf, 1)[9], before[0])
    assert torch.equal(_arguments(bank, buf, 1)[10], before[1])

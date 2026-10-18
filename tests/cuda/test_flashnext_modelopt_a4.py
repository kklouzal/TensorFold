"""Real native ModelOpt A4: independent linears, direct scales and live GPU routes."""

import pytest
import torch
from modelopt_a4_inputs import PREFIX, Reader, individual_reference, quant_reference

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native activation FP4 requires SM12", allow_module_level=True)

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.moe import MoEBuffers
from tensorfold.cuda.nvfp4.grouped_a4 import _ext
from tensorfold.cuda.nvfp4.modelopt_experts import load_modelopt_a4


class Shape:
    def __init__(self, cfg):
        self.num_experts, self.num_experts_per_tok = cfg.experts, cfg.top_k
        self.moe_intermediate_size, self.hidden_size = cfg.moe_width, cfg.hidden


@pytest.mark.parametrize("prefill,rows", [(False, 1), (False, 17), (True, 129)])
def test_actual_projection_shape_device_plan_matches_individual_original_linears(prefill, rows):
    reader = Reader(width=640, dims=2560)
    bank = load_modelopt_a4(reader, PREFIX, reader.cfg, device="cuda:0")
    x = (torch.randn(rows, 2560, generator=torch.Generator().manual_seed(388)) * .25).to("cuda", torch.bfloat16)
    buf = MoEBuffers(rows + 1, Shape(reader.cfg), "cuda", prefill=prefill)
    picks = (torch.arange(rows, device="cuda", dtype=torch.int32)[:, None]
             + torch.arange(buf.slots, device="cuda", dtype=torch.int32)) % reader.cfg.experts
    picks[:, -1] = reader.cfg.experts  # shared expert is intentionally outside the routed consumer
    buf.pick[:rows].copy_(picks)
    grouped.route(buf.pick[:rows], buf.plan, 16)
    buf.y.fill_(37.)
    bank.run_plan(x, buf, prefill=prefill)
    expected = individual_reference(Reader(width=640, dims=2560), x, picks, prefill=prefill)
    expected[:, -1].fill_(37.)
    assert torch.equal(buf.y[:rows], expected)
    assert torch.equal(buf.y[rows], torch.full_like(buf.y[rows], 37.))
    # Both projections and every expert use their own original direct scale.
    plan = torch.tensor([[0, 0, e] for e in range(reader.cfg.experts)], dtype=torch.int32, device="cuda")
    codes = torch.empty((2, 3, 1280), dtype=torch.uint8, device="cuda")
    scales = torch.empty((2, 3, 160), dtype=torch.uint8, device="cuda")
    _ext().quantize(x, plan, bank.fields[4], codes, scales, False, True)
    for expert in range(3):
        for projection, name in enumerate(("gate_proj", "up_proj")):
            direct = float(reader.payload[f"{PREFIX}.experts.{expert}.{name}.input_scale"])
            c, s = quant_reference(x[:1].cpu(), direct)
            assert torch.equal(codes[projection, expert].cpu(), c[0])
            assert torch.equal(scales[projection, expert].cpu(), s[0])
    torch.cuda.synchronize()


@pytest.mark.parametrize("common", [False, True])
def test_captured_device_plan_replays_different_picks_and_values_without_cpu_route_read(monkeypatch, common):
    reader = Reader()
    if common:
        value = reader.payload[f"{PREFIX}.experts.0.gate_proj.input_scale"].item()
        for name, tensor in reader.payload.items():
            if name.endswith(("gate_proj.input_scale", "up_proj.input_scale")):
                tensor.fill_(value)
    bank = load_modelopt_a4(reader, PREFIX, reader.cfg, device="cuda:0")
    assert bank.shared_input is common
    x = torch.ones((3, reader.cfg.hidden), dtype=torch.bfloat16, device="cuda")
    buf = MoEBuffers(3, Shape(reader.cfg), "cuda", prefill=True)
    picks = torch.tensor([[0, 1, 3], [1, 2, 3], [2, 0, 3]], dtype=torch.int32, device="cuda")
    buf.pick.copy_(picks)
    buf.y.zero_()

    def run():
        grouped.route(buf.pick, buf.plan, 16)
        bank.run_plan(x, buf, prefill=True)

    run()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", lambda *_a, **_k: pytest.fail("CPU route read during capture"))
        with torch.cuda.graph(graph):
            run()
    for step in range(3):
        changed = picks.clone()
        changed[:, :2] = (changed[:, :2] + step + 1) % reader.cfg.experts
        buf.pick.copy_(changed)
        x.fill_(.125 * (step + 1))
        graph.replay()
        expected = individual_reference(reader, x, changed, prefill=True)
        assert torch.equal(buf.y, expected)
    torch.cuda.synchronize()


@pytest.mark.parametrize("prefill", [False, True])
def test_mixed_cache_a4_then_w4a16_mtp_then_a4_preserves_complete_fields_and_slots(prefill):
    from tensorfold.cuda.nvfp4 import experts as nvx
    from tensorfold.cuda.nvfp4.modelopt_cache import ModelOptExpertCache
    from tensorfold.families.qwen4_exp.cuda.nvfp4_moe import load_moe_host

    reader = Reader(width=640, dims=2560)
    cfg = reader.cfg
    rng = torch.Generator().manual_seed(422)
    mtp = "mtp.layers.0.mlp"
    reader.payload[mtp + ".experts.gate_up_proj"] = (torch.randn(3, 1280, 2560, generator=rng) * .02).to(torch.bfloat16)
    reader.payload[mtp + ".experts.down_proj"] = (torch.randn(3, 2560, 640, generator=rng) * .02).to(torch.bfloat16)
    reader.where = dict.fromkeys(reader.payload, "fixture.safetensors")
    entry = 3 * 921600 + 64
    cache = ModelOptExpertCache(entry, entry, "cuda:0", width=640, dims=2560)
    try:
        native = load_modelopt_a4(reader, PREFIX, cfg, device="cuda:0", cache=cache, layer_id=0)
        draft = load_moe_host(reader, mtp, cfg, "cuda:0", cache=cache, include_shared=False, layer_id=1)
        assert len(cache._layers[0].source) == 6 and len(cache._layers[1].source) == 4
        draft_source = cache._layers[1].source
        reference = nvx.Experts4(*(t.cuda() for t in draft_source), cfg.moe_width, cfg.hidden)
        cache.configure_before_use(2 * entry)
        x = (torch.randn(17, cfg.hidden, generator=rng) * .3).to("cuda", torch.bfloat16)
        picks = (torch.arange(17, dtype=torch.int32, device="cuda")[:, None]
                 + torch.arange(3, dtype=torch.int32, device="cuda")) % 3
        expected = individual_reference(Reader(width=640, dims=2560), x, picks, prefill=prefill)
        output = torch.empty_like(expected)
        plan = grouped.Plan(17, 3, 3, "cuda", prefill=prefill)
        grouped.route(picks, plan, 16)
        activation = torch.empty((51, cfg.moe_width), dtype=torch.bfloat16, device="cuda")
        mtp_expected = torch.empty_like(expected)
        nvx.gate_up(x, reference, plan, activation, 17)
        nvx.down(activation, reference, plan, mtp_expected.view(-1, cfg.hidden), 17)
        for _ in range(2):
            native.run(x, picks, output, prefill=prefill)
            assert torch.equal(output, expected)
            draft.run_picks(x, picks, output, prefill=prefill)
            assert torch.equal(output, mtp_expected)
            native.run(x, picks, output, prefill=prefill)
            assert torch.equal(output, expected)
        assert cache.evictions > 0
        assert len(cache._layers[0].views) == 6 and len(cache._layers[1].views) == 4
        torch.cuda.synchronize()
    finally:
        cache.close()


def test_direct_alpha_uses_original_scale_without_double_reciprocal():
    import struct

    from tensorfold.cuda.nvfp4.experts_a4 import pack_projection_a4

    direct = struct.unpack("f", struct.pack("I", 0x3b350913))[0]
    codes = torch.full((1, 1, 32), 0x22, dtype=torch.uint8, device="cuda")
    scales = torch.full((1, 1, 4), 0x38, dtype=torch.uint8, device="cuda")
    blob = pack_projection_a4(torch.full((64, 32), 0x22, dtype=torch.uint8),
                              torch.full((64, 4), 0x38, dtype=torch.uint8).view(torch.float8_e4m3fn)).cuda().view(1, 1, -1)
    weight = torch.ones((1, 1), dtype=torch.float32, device="cuda")
    inputs = torch.full((1, 1), direct, dtype=torch.float32, device="cuda")
    items = torch.tensor([[0, 0, 1]], dtype=torch.int32, device="cuda")
    plan = torch.tensor([[0, 0, 0]], dtype=torch.int32, device="cuda")
    output = torch.empty((1, 1, 64), dtype=torch.float32, device="cuda")
    _ext().gemm(codes, scales, blob, weight, inputs, items, plan, output, 64, 64, 1, True, True)
    # Every product is exactly1, their exact sum64, and multiplication by64
    # only shifts the exponent. The original direct FP32 bits remain visible.
    assert torch.equal(output, torch.full_like(output, 64 * direct))
    rounded = float(1. / (1. / torch.tensor(direct, dtype=torch.float32)))
    assert not torch.equal(output, torch.full_like(output, 64 * rounded))


def test_direct_activation_quantizer_extremes_ties_underflow_and_signed_zero():
    from tensorfold.cuda.nvfp4.modelopt_experts import ModelOptExpertsA4

    reader = Reader(count=1, width=64, dims=128)
    bank = load_modelopt_a4(reader, PREFIX, reader.cfg, device="cuda:0")
    assert isinstance(bank, ModelOptExpertsA4)
    values = [0., -0., .25, -.25, .75, -.75, 1.25, -1.25, 1.75, -1.75, 2.5, -2.5, 3.5, -3.5, 6., -6.]
    maximum = torch.finfo(torch.bfloat16).max
    x = torch.tensor([values * 8, [maximum, -maximum] * 64, [2.**-20] * 128,
                      [float("nan")] * 128], dtype=torch.bfloat16, device="cuda")
    plan = torch.tensor([[r, 0, 0] for r in range(4)], dtype=torch.int32, device="cuda")
    calibration = torch.ones((1, 2), dtype=torch.float32, device="cuda")
    codes = torch.empty((2, 4, 64), dtype=torch.uint8, device="cuda")
    scales = torch.empty((2, 4, 8), dtype=torch.uint8, device="cuda")
    _ext().quantize(x, plan, calibration, codes, scales, False, True)
    for projection in range(2):
        expected_codes, expected_scales = quant_reference(x.cpu(), 1.)
        assert torch.equal(codes[projection].cpu(), expected_codes)
        assert torch.equal(scales[projection].cpu(), expected_scales)
    assert torch.count_nonzero(codes[0, 2]) == 0 and torch.count_nonzero(scales[0, 2]) == 0
    calibration.fill_(.001964750699698925)
    _ext().quantize(x, plan, calibration, codes, scales, False, True)
    expected_codes, expected_scales = quant_reference(x.cpu(), float(calibration[0, 0]))
    assert torch.equal(codes[0].cpu(), expected_codes)
    assert torch.equal(scales[0].cpu(), expected_scales)
    assert torch.equal(codes[0, 1], torch.full_like(codes[0, 1], 0xf7))
    torch.cuda.synchronize()


@pytest.mark.parametrize("prefill,rows", [(False, 1), (False, 17), (True, 129)])
def test_proven_row_shared_input_matches_general_plan_and_individual_reference(prefill, rows):
    from tensorfold.cuda.nvfp4.modelopt_experts import ModelOptExpertsA4

    reader = Reader(width=640, dims=2560)
    direct = reader.payload[f"{PREFIX}.experts.0.gate_proj.input_scale"].item()
    for name, value in reader.payload.items():
        if name.endswith(("gate_proj.input_scale", "up_proj.input_scale")):
            value.fill_(direct)
    selected = load_modelopt_a4(reader, PREFIX, reader.cfg, device="cuda:0")
    assert selected.shared_input
    general = ModelOptExpertsA4(selected.fields, selected.width, selected.dims)
    x = (torch.randn(rows, 2560, generator=torch.Generator().manual_seed(677)) * .3).to("cuda", torch.bfloat16)
    buf = MoEBuffers(rows + 1, Shape(reader.cfg), "cuda", prefill=prefill)
    picks = (torch.arange(rows, dtype=torch.int32, device="cuda")[:, None]
             + torch.arange(buf.slots, dtype=torch.int32, device="cuda")) % 3
    picks[:, -1] = 3
    buf.pick[:rows].copy_(picks)
    grouped.route(buf.pick[:rows], buf.plan, 16)
    buf.y.zero_()
    expected = individual_reference(reader, x, picks, prefill=prefill)
    # Both policies consume the same reusable backing in alternating order.
    # No stale projection plane/member order can survive a policy transition.
    for bank in (selected, general, selected, general):
        bank.run_plan(x, buf, prefill=prefill)
        assert torch.equal(buf.y[:rows], expected)
    codes = buf.modelopt_a4[0].view(-1)[:rows * 1280].view(rows, 1280)
    scales = buf.modelopt_a4[1].view(-1)[:rows * 160].view(rows, 160)
    selected.run_plan(x, buf, prefill=prefill)
    c, s = quant_reference(x.cpu(), direct)
    assert torch.equal(codes.cpu(), c)
    assert torch.equal(scales.cpu(), s)
    torch.cuda.synchronize()

"""Native E512/short-row plans consume bounded items, including live replay."""

from types import SimpleNamespace

import pytest
import torch
from modelopt_a4_inputs import PREFIX, Reader, individual_reference

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native activation FP4 requires SM12", allow_module_level=True)

from tensorfold.cuda import experts as grouped
from tensorfold.cuda.moe import MoEBuffers
from tensorfold.cuda.nvfp4.modelopt_experts import ModelOptExpertsA4, load_modelopt_a4


@pytest.mark.parametrize("prefill,rows", [(False, 2), (True, 2), (True, 17)])
def test_large_bank_short_live_plan_matches_independent_linears_and_changing_graph(monkeypatch, prefill, rows):
    # Three independently calibrated source experts are repeated into a real
    # E512 bank, keeping this plan regression small. Picks exercise distant
    # bank indices; the separate three-expert oracle maps each original row.
    reader = Reader(width=64, dims=128)
    original = load_modelopt_a4(reader, PREFIX, reader.cfg, device="cuda:0")
    index = torch.arange(512, dtype=torch.int64, device="cuda") % 3
    bank = ModelOptExpertsA4(tuple(field.index_select(0, index) for field in original.fields), 64, 128)
    cfg = SimpleNamespace(num_experts=512, num_experts_per_tok=10, moe_intermediate_size=64, hidden_size=128)
    buf = MoEBuffers(rows + 1, cfg, "cuda", prefill=prefill)
    x = (torch.randn(rows, 128, generator=torch.Generator().manual_seed(981)) * .25).to("cuda", torch.bfloat16)
    row = torch.arange(rows, dtype=torch.int32, device="cuda")[:, None]
    slot = torch.arange(11, dtype=torch.int32, device="cuda")[None]
    picks = (row * 97 + slot * 113) % 512
    picks[:, -1] = 512
    buf.pick[:rows].copy_(picks)
    buf.y.fill_(37)
    assert buf.plan.items.shape[0] > rows * 11  # retained backing is deliberately oversized

    def run():
        grouped.route(buf.pick[:rows], buf.plan, 16)
        bank.run_plan(x, buf, prefill=prefill)

    def expected():
        logical = torch.where(buf.pick[:rows] == 512, 3, buf.pick[:rows] % 3)
        result = individual_reference(reader, x, logical, prefill=prefill)
        result[:, -1].fill_(37)
        return result

    run()
    assert torch.equal(buf.y[:rows], expected())
    assert 0 < int(buf.plan.counts[0]) <= rows * 11
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with monkeypatch.context() as patch:
        patch.setattr(torch.Tensor, "cpu", lambda *_args, **_kwargs: pytest.fail("CPU routing during capture"))
        with torch.cuda.graph(graph):
            run()
    for step in range(3):
        changed = (picks + 19 * (step + 1)) % 512
        if step == 1:
            changed.zero_()  # a few long items after many one-pair items
        changed[0, 0] = 511
        changed[:, -1] = 512
        buf.pick[:rows].copy_(changed)
        x.fill_(.125 * (step + 1))
        graph.replay()
        assert torch.equal(buf.y[:rows], expected())
        assert torch.equal(buf.y[rows], torch.full_like(buf.y[rows], 37))
    torch.cuda.synchronize()

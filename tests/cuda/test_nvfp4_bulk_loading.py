"""Actual owned-file bulk loading into resident and leased native A4 consumers."""

import pytest
import torch
from nvfp4_a4_inputs import PREFIX, Context, individual_reference, swiglu
from test_nvfp4_a4_bulk import FileContext

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native activation FP4 requires SM12", allow_module_level=True)

from tensorfold.cuda.nvfp4.cache_a4 import A4ExpertCache
from tensorfold.cuda.nvfp4.experts_a4 import PlanBuffers
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4
from tensorfold.cuda.nvfp4.host_layout import entry_layout_a4


@pytest.mark.parametrize("width,dims", [(64, 128), (512, 2048)])
@pytest.mark.parametrize("prefill", [False, True])
def test_file_bulk_resident_authority_and_real_device_plan_are_exact(tmp_path, width, dims, prefill):
    context = FileContext(count=2, width=width, dims=dims).materialize(tmp_path)
    ordinary = Context(count=2, width=width, dims=dims)
    try:
        bank = load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", swiglu=swiglu)
        reference = load_host_experts_a4(ordinary, PREFIX, ordinary.cfg, "cuda:0", swiglu=swiglu)
        assert len(context.run_reads) == 2
        assert all(torch.equal(a, b) for a, b in zip(bank.fields, reference.fields, strict=True))
        rows, slots = 5, 2
        x = (torch.randn(rows, dims, generator=torch.Generator().manual_seed(1701)) * .2).to("cuda", torch.bfloat16)
        picks = torch.tensor([[r % 2, (r + 1) % 2] for r in range(rows)], dtype=torch.int32, device="cuda")
        expected = individual_reference(Context(count=2, width=width, dims=dims), x, picks, prefill=prefill)
        buf = PlanBuffers(rows + 2, slots, 2, dims, "cuda", prefill=prefill)
        buf.y.fill_(17.)
        buf.pick[:rows].copy_(picks)
        buf.route(rows)
        bank.run_plan(x, buf, prefill=prefill)
        assert torch.equal(buf.y[:rows], expected)
        assert torch.equal(buf.y[rows:], torch.full_like(buf.y[rows:], 17.))
    finally:
        context.files.close()


@pytest.mark.parametrize("prefill", [False, True])
def test_file_bulk_strided_host_authority_runs_through_real_single_cell_eviction(tmp_path, prefill):
    width, dims = 512, 2048
    context = FileContext(count=2, width=width, dims=dims).materialize(tmp_path)
    entry = entry_layout_a4(width, dims).entry_bytes
    cache = A4ExpertCache(entry, entry, "cuda:0", width=width, dims=dims)
    try:
        bank = load_host_experts_a4(context, PREFIX, context.cfg, "cuda:0", cache=cache, swiglu=swiglu)
        assert len(context.run_reads) == 2
        rows = 5
        x = (torch.randn(rows, dims, generator=torch.Generator().manual_seed(2802)) * .2).to("cuda", torch.bfloat16)
        picks = torch.tensor([[r % 2, (r + 1) % 2] for r in range(rows)], dtype=torch.int32, device="cuda")
        expected = individual_reference(Context(count=2, width=width, dims=dims), x, picks, prefill=prefill)
        output = torch.empty_like(expected)
        bank.run(x, picks, output, prefill=prefill)
        assert torch.equal(output, expected) and cache.evictions > 0
    finally:
        context.files.close()
        cache.close()

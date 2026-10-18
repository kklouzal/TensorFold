"""CT (reciprocal) native A4 on device routing plans versus host windows, independent operators and graph replay."""

import pytest
import torch
from nvfp4_a4_inputs import PREFIX, Context, individual_reference, swiglu

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native activation FP4 requires SM12", allow_module_level=True)

from tensorfold.cuda.nvfp4.experts_a4 import PlanBuffers
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4


def _context(count, width, dims, *, uniform):
    ctx = Context(count=count, width=width, dims=dims)
    if uniform:
        # Real CT MoE checkpoints share one gate/up input global over the bank
        # (Gemma4 and Qwen3.6 layers 0/1/15 verified); down globals differ.
        value = ctx.payload[f"{PREFIX}.experts.0.gate_proj.input_global_scale"].clone()
        for name, tensor in ctx.payload.items():
            if name.endswith(("gate_proj.input_global_scale", "up_proj.input_global_scale")):
                tensor.copy_(value)
    return ctx


def _picks(rows, slots, count, seed):
    """Distinct experts per row, deterministic and unordered (like top-k)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(count, generator=generator)[:slots] for _ in range(rows)]).to(
        "cuda", torch.int32)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("prefill,rows", [(False, 1), (False, 5), (True, 1), (True, 17), (True, 129)])
def test_device_plan_matches_host_windows_and_independent_reference(uniform, prefill, rows):
    count, width, dims, slots = 9, 128, 256, 8
    bank = load_host_experts_a4(_context(count, width, dims, uniform=uniform), PREFIX,
                                _context(count, width, dims, uniform=uniform).cfg, "cuda:0", swiglu=swiglu)
    assert bank.capturable and bank.shared_input == uniform and not bank.direct_input
    x = (torch.randn(rows, dims, generator=torch.Generator().manual_seed(77 + rows)) * .4).to("cuda", torch.bfloat16)
    picks = _picks(rows, slots, count, 5 + rows)
    host = torch.empty((rows, slots, dims), dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")
    bank.run(x, picks, host, prefill=prefill)
    buf = PlanBuffers(rows + 3, slots, count, dims, "cuda", prefill=prefill)
    buf.y.fill_(17.)
    buf.pick[:rows].copy_(picks)
    buf.route(rows)
    bank.run_plan(x, buf, prefill=prefill)
    assert torch.equal(buf.y[:rows], host)
    # Rows beyond the live count are never written.
    assert torch.equal(buf.y[rows:], torch.full_like(buf.y[rows:], 17.))
    expected = individual_reference(_context(count, width, dims, uniform=uniform), x, picks, prefill=prefill)
    assert torch.equal(buf.y[:rows], expected)


def test_device_plan_shared_cell_is_skipped_and_slot_left_to_the_caller():
    count, width, dims, slots = 4, 64, 128, 3
    ctx = _context(count, width, dims, uniform=False)
    bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", swiglu=swiglu)
    rows = 6
    x = (torch.randn(rows, dims, generator=torch.Generator().manual_seed(3)) * .4).to("cuda", torch.bfloat16)
    buf = PlanBuffers(rows, slots, count + 1, dims, "cuda", prefill=False)
    picks = torch.tensor([[r % count, (r + 1) % count, count] for r in range(rows)], dtype=torch.int32, device="cuda")
    buf.pick.copy_(picks)
    buf.route(rows)
    buf.y.fill_(-3.)
    bank.run_plan(x, buf, prefill=False)
    assert torch.equal(buf.y[:, 2], torch.full_like(buf.y[:, 2], -3.))
    expected = individual_reference(Context(count=count, width=width, dims=dims), x, picks[:, :2], prefill=False)
    assert torch.equal(buf.y[:, :2], expected)


@pytest.mark.parametrize("prefill", [False, True])
def test_captured_device_plan_replays_new_routes_without_host_reads(prefill):
    count, width, dims, slots, rows = 16, 128, 256, 8, 3
    ctx = _context(count, width, dims, uniform=True)
    bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", swiglu=swiglu)
    buf = PlanBuffers(16, slots, count, dims, "cuda", prefill=prefill)
    x = torch.zeros((rows, dims), dtype=torch.bfloat16, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        buf.pick[:rows].copy_(_picks(rows, slots, count, 1))
        buf.route(rows)
        bank.run_plan(x, buf, prefill=prefill)              # warm: compiles and retains scratch
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        buf.route(rows)
        bank.run_plan(x, buf, prefill=prefill)
    for seed in (11, 12, 13):
        source = (torch.randn(rows, dims, generator=torch.Generator().manual_seed(seed)) * .4).to(
            "cuda", torch.bfloat16)
        picks = _picks(rows, slots, count, seed)
        x.copy_(source)
        buf.pick[:rows].copy_(picks)
        graph.replay()
        replayed = buf.y[:rows].clone()
        eager = torch.empty_like(replayed)
        bank.run(source, picks, eager, prefill=prefill)
        assert torch.equal(replayed, eager)
    torch.cuda.synchronize()


def test_plan_refuses_out_of_range_expert_ids_on_the_device():
    import subprocess
    import sys
    import textwrap

    # A trap poisons the CUDA context, so prove it in a child process.
    code = textwrap.dedent("""
        import torch
        from tensorfold.cuda.experts import Plan, route
        plan = Plan(1, 2, 4, "cuda")
        route(torch.tensor([[0, 4]], dtype=torch.int32, device="cuda"), plan, 16)
        torch.cuda.synchronize()
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert "unspecified launch failure" in result.stderr or "device-side" in result.stderr or "trap" in result.stderr \
        or "illegal instruction" in result.stderr

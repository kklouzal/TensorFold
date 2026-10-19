"""CT (reciprocal) native A4 on device routing plans versus host windows, independent operators and graph replay."""

import pytest
import torch
from nvfp4_a4_inputs import PREFIX, Context, individual_reference, swiglu

if not torch.cuda.is_available() or torch.cuda.get_device_capability()[0] != 12:
    pytest.skip("native activation FP4 requires SM12", allow_module_level=True)

from tensorfold.cuda.nvfp4.experts_a4 import PlanBuffers
from tensorfold.cuda.nvfp4.host_experts import load_host_experts_a4


def _picks(rows, slots, count, seed):
    """Distinct experts per row, deterministic and unordered (like top-k)."""
    generator = torch.Generator().manual_seed(seed)
    return torch.stack([torch.randperm(count, generator=generator)[:slots] for _ in range(rows)]).to(
        "cuda", torch.int32)


@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("prefill,rows", [(False, 1), (False, 5), (True, 1), (True, 17), (True, 129)])
def test_device_plan_matches_host_windows_and_independent_reference(uniform, prefill, rows):
    count, width, dims, slots = 9, 128, 256, 8
    ctx = Context(count=count, width=width, dims=dims, uniform=uniform)
    bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", swiglu=swiglu)
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
    expected = individual_reference(Context(count=count, width=width, dims=dims, uniform=uniform), x, picks,
                                    prefill=prefill)
    assert torch.equal(buf.y[:rows], expected)


def test_device_plan_shared_cell_is_skipped_and_slot_left_to_the_caller():
    count, width, dims, slots = 4, 64, 128, 3
    ctx = Context(count=count, width=width, dims=dims, uniform=False)
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
    ctx = Context(count=count, width=width, dims=dims, uniform=True)
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


@pytest.mark.parametrize("rows,bad", [(1, 4), (129, -1)])
def test_plan_refuses_out_of_range_expert_ids_on_the_device(rows, bad):
    import subprocess
    import sys
    import textwrap

    # A trap poisons the CUDA context, so prove it in a child process: the one-block plan (8 pairs) and the wide
    # ranking pass (1032 pairs, over the 1024-pair one-block limit) each refuse an id outside [0, 4).
    code = textwrap.dedent(f"""
        import torch
        from tensorfold.cuda.experts import Plan, route
        plan = Plan({rows}, 8, 4, "cuda")
        picks = torch.zeros(({rows}, 8), dtype=torch.int32, device="cuda")
        picks[-1, -1] = {bad}
        route(picks, plan, 16)
        torch.cuda.synchronize()
    """)
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
    assert result.returncode != 0
    assert "unspecified launch failure" in result.stderr or "device-side" in result.stderr or "trap" in result.stderr \
        or "illegal instruction" in result.stderr


@pytest.mark.parametrize("prefill,rows", [(False, 5), (True, 129)])
def test_retained_moe_buffer_and_plan_scratch_match_qwen_admission(prefill, rows):
    from types import SimpleNamespace

    from tensorfold.cuda.experts import TILE, route
    from tensorfold.cuda.moe import MoEBuffers
    from tensorfold.families.qwen3_5_moe.ram_experts import _moe_buffers

    count, width, dims, top_k = 9, 128, 256, 3
    ctx = Context(count=count, width=width, dims=dims, uniform=True)
    bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", swiglu=swiglu)
    size = 1 << max(4, (rows - 1).bit_length())
    shape = SimpleNamespace(num_experts_per_tok=top_k, num_experts=count, moe_intermediate_size=width,
                            hidden_size=dims)
    buf = MoEBuffers(size, shape, "cuda", prefill=prefill)
    x = (torch.randn(rows, dims, generator=torch.Generator().manual_seed(9)) * .4).to("cuda", torch.bfloat16)
    picks = _picks(rows, top_k, count, 4)
    buf.pick[:rows, :top_k].copy_(picks)
    buf.pick[:rows, top_k] = count
    route(buf.pick[:rows], buf.plan, TILE)
    bank.run_plan(x, buf, prefill=prefill)
    tensors = [buf.logits, buf.pick, buf.wts, buf.act, buf.y, buf.plan.members, buf.plan.items, buf.plan.counts,
               buf.plan.rank, buf.plan.hist, *(t for s in buf.a4_scratch.values() for t in s), buf.a4_row_plan]
    retained = sum(t.numel() * t.element_size() for t in tensors)
    smaller = _moe_buffers(size // 2, count + 1, top_k + 1, width, dims, prefill, planned=True) if size > 16 else 0
    assert retained == _moe_buffers(size, count + 1, top_k + 1, width, dims, prefill, planned=True) - smaller


@pytest.mark.parametrize("prefill,rows", [(False, 1), (False, 6), (True, 33)])
def test_shared_expert_cell_matches_the_separate_shared_projections(prefill, rows):
    from functools import partial
    from types import SimpleNamespace

    from nvfp4_a4_inputs import linears

    from tensorfold.cuda.experts import TILE, route
    from tensorfold.cuda.moe import MoEBuffers
    from tensorfold.cuda.nvfp4 import checkpoint

    count, width, dims, top_k = 6, 128, 256, 3

    def context():
        return Context(count=count + 1, width=width, dims=dims, uniform=True).alias_shared_expert(count)

    full = SimpleNamespace(experts=count, moe_width=width, hidden=dims, top_k=top_k)
    source = context()
    fused = load_host_experts_a4(source, PREFIX, full, "cuda:0", swiglu=swiglu, shared=True)
    assert fused.count == count + 1 and fused.shared_input
    assert not any(".experts." + str(count) + "." in name for name in source.reads)
    separate = load_host_experts_a4(context(), PREFIX, full, "cuda:0", swiglu=swiglu)
    gate, up, down = linears(context(), count)
    shape = SimpleNamespace(num_experts_per_tok=top_k, num_experts=count, moe_intermediate_size=width,
                            hidden_size=dims)
    x = (torch.randn(rows, dims, generator=torch.Generator().manual_seed(21)) * .4).to("cuda", torch.bfloat16)
    picks = _picks(rows, top_k, count, 8)
    outputs = []
    for bank in (fused, separate):
        buf = MoEBuffers(rows + 2, shape, "cuda", prefill=prefill)
        buf.y.fill_(5.)
        buf.pick[:rows, :top_k].copy_(picks)
        buf.pick[:rows, top_k] = count
        route(buf.pick[:rows], buf.plan, TILE)
        bank.run_plan(x, buf, prefill=prefill)
        outputs.append(buf.y[:rows].clone())
    assert torch.equal(outputs[0][:, :top_k], outputs[1][:, :top_k])
    assert torch.equal(outputs[1][:, top_k], torch.full_like(outputs[1][:, top_k], 5.))
    if prefill:
        shared = down.prefill(swiglu(gate.prefill(x), up.prefill(x)))
    else:
        shared = torch.empty((rows, dims), dtype=torch.float32, device="cuda")
        partial(checkpoint.matmul, checkpoint.A4, lin=down, f32=True)(swiglu(gate(x), up(x)), out=shared)
    assert torch.equal(outputs[0][:, top_k], shared)


def _route(buf, rows, tile):
    from tensorfold.cuda import pdl
    from tensorfold.cuda.experts import _ext as plan_ext

    plan = buf.plan
    plan.tile = tile
    plan_ext().plan(buf.pick[:rows], rows * buf.slots, plan.experts, tile, plan.members, plan.items, plan.counts,
                    plan.rank, plan.hist, pdl.ON)


@pytest.mark.parametrize("activation", ["silu", "geglu"])
@pytest.mark.parametrize("uniform", [False, True])
@pytest.mark.parametrize("prefill,rows", [(False, 1), (False, 7), (True, 3), (True, 40), (True, 300)])
def test_16_64_and_128_pair_items_equal_the_host_window_path(activation, uniform, prefill, rows):
    from tensorfold.families.gemma4.cuda.nvfp4_experts import geglu

    count, width, dims, slots = 12, 192, 1024, 8          # decode gate/up: two K slices (split_k)
    ctx = Context(count=count, width=width, dims=dims, uniform=uniform)
    bank = load_host_experts_a4(ctx, PREFIX, ctx.cfg, "cuda:0", swiglu=swiglu if activation == "silu" else geglu)
    x = (torch.randn(rows, dims, generator=torch.Generator().manual_seed(31 + rows)) * .4).to("cuda", torch.bfloat16)
    picks = _picks(rows, slots, count, 41 + rows)
    # Independent arithmetic route: host windows, per-warp GEMM, eager PyTorch activation, separate quantizer.
    expected = torch.empty((rows, slots, dims), dtype=torch.bfloat16 if prefill else torch.float32, device="cuda")
    for lo in range(0, rows, 16):
        bank.run(x[lo:lo + 16], picks[lo:lo + 16], expected[lo:lo + 16], prefill=prefill)
    for tile in (16, 64, 128):
        buf = PlanBuffers(rows + 5, slots, count, dims, "cuda", prefill=prefill)
        buf.y.fill_(3.)
        buf.pick[:rows].copy_(picks)
        _route(buf, rows, tile)
        bank.run_plan(x, buf, prefill=prefill)
        assert torch.equal(buf.y[:rows], expected), tile

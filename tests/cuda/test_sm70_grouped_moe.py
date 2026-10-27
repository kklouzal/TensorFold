"""Exact-oriented grouped reuse against unchanged packed and independent BMM oracles."""
import gc
from types import SimpleNamespace as NS
import weakref

import pytest

torch = pytest.importorskip("torch")

from gguf_fixture import blocks  # noqa: E402
from tensorfold.cuda.moe import MoEBuffers  # noqa: E402
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFExperts, GGUFPacked  # noqa: E402
from tensorfold.families.qwen4_exp.gguf import FORMATS  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                               reason="actual SM70 grouped expert qualification")


@pytest.fixture(scope="module", autouse=True)
def prepared_kernel_residency():
    from tensorfold.cuda.sm70 import gguf_ops

    loaded = gguf_ops.extension().prepare_grouped()
    assert loaded == 19


def bits(actual, expected):
    torch.testing.assert_close(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8),
                               rtol=0, atol=0)


@pytest.fixture(scope="module", params=[(12, 13, 20, 0), (13, 12, 7, 16)])
def bank(request):
    gate_kind, up_kind, down_kind, offset = request.param
    parts = []
    for kind, rows, columns, seed in ((gate_kind, 512 * 80, 2560, 2921),
                                      (up_kind, 512 * 80, 2560, 2922),
                                      (down_kind, 512 * 2560, 80, 2923)):
        unit = FORMATS[kind][1]
        physical = -(-(columns + offset) // unit) * unit
        raw = blocks(kind, rows, physical, seed=seed)
        parts.append(GGUFPacked(torch.tensor(raw, device="cuda"), kind, rows, columns, offset))
    torch.manual_seed(2924)
    shared = [GGUFDense(torch.randn(shape, device="cuda", dtype=torch.float16) * .002)
              for shape in ((80, 2560), (80, 2560), (2560, 80))]
    value = GGUFExperts(*parts, shared, count=512, width=80, dims=2560)
    yield value
    del value, parts, shared
    gc.collect()
    torch.cuda.empty_cache()


def owner(topk):
    return MoEBuffers(32, NS(num_experts_per_tok=topk, num_experts=512, moe_intermediate_size=80,
                            hidden_size=2560, execution_dtype="float16"), "cuda")


def routes(workspace, rows, topk, *, concentrated):
    picks = workspace.pick[:rows]
    for slot in range(topk):
        if concentrated:
            picks[:, slot].fill_(511 if slot == 0 else slot - 1)
        else:
            picks[:, slot].copy_((torch.arange(rows, device="cuda", dtype=torch.int32) * 37 + slot * 11) % 512)
    picks[:, topk].fill_(512)
    return picks


@pytest.mark.parametrize("rows,topk", [(4, 10), (32, 10), (32, 32)])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_original_bmm_and_packed_bits_actual512_experts_shared_and_wide_plan(bank, rows, topk, dtype):
    torch.manual_seed(2925)
    x = torch.randn((rows, 2560), device="cuda", dtype=torch.float16) * .01
    workspace = owner(topk)
    output = torch.empty((rows, topk + 1, 2560), device="cuda", dtype=dtype)
    for concentrated in (True, False):
        picks = routes(workspace, rows, topk, concentrated=concentrated)
        reference = torch.empty_like(output)
        bank.forward_reference(x, picks, reference)
        packed = torch.empty_like(output)
        bank.forward_exact(x, picks, packed)
        bits(packed, reference)  # Existing independent mechanisms must agree before comparing the candidate.
        assert bank.forward_grouped(x, picks, workspace, output) is output
        bits(output, reference)
        bits(output, packed)


def test_duplicate_slots_and_malformed_ids_keep_original_nan_and_valid_bits(bank):
    workspace = owner(3)
    x = torch.full((4, 2560), .01, dtype=torch.float16, device="cuda")
    picks = workspace.pick[:4]
    picks.copy_(torch.tensor([[7, 7, -1, 512], [511, 511, 513, 512],
                              [0, 0, 2**31 - 1, 512], [1, 1, 1, 512]], device="cuda", dtype=torch.int32))
    old = bank.forward_exact(x, picks)
    actual = bank.forward_grouped(x, picks, workspace, workspace.y[:4])
    assert torch.equal(torch.isnan(actual), torch.isnan(old))
    valid = ~torch.isnan(old)
    bits(actual[valid], old[valid])
    bits(actual[:, 0], actual[:, 1])


def test_owned_buffers_graph_replay_no_retained_bank_owner_or_native_core_allocation(bank):
    from tensorfold.cuda.sm70 import gguf_ops

    workspace = owner(10)
    borrowed = weakref.ref(workspace)
    x = torch.ones((32, 2560), device="cuda", dtype=torch.float16) * .01
    picks = routes(workspace, 32, 10, concentrated=True)
    output = workspace.y
    before = tuple(t.data_ptr() for t in (workspace.act, workspace.plan.members, workspace.plan.items,
                                         workspace.plan.counts, workspace.plan.rank, workspace.plan.hist))
    gguf_ops.experts_grouped(x, picks, bank.gate, bank.up, bank.down, bank.shared, output, workspace)
    torch.cuda.synchronize()
    allocated = torch.cuda.memory_allocated()
    gguf_ops.experts_grouped(x, picks, bank.gate, bank.up, bank.down, bank.shared, output, workspace)
    torch.cuda.synchronize()
    assert torch.cuda.memory_allocated() == allocated
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        bank.forward_grouped(x, picks, workspace, output)
    for concentrated in (False, True):
        routes(workspace, 32, 10, concentrated=concentrated)
        x.mul_(.75)
        expected = bank.forward_exact(x, picks)
        graph.replay()
        torch.cuda.synchronize()
        bits(output, expected)
        assert before == tuple(t.data_ptr() for t in (workspace.act, workspace.plan.members, workspace.plan.items,
                                                      workspace.plan.counts, workspace.plan.rank, workspace.plan.hist))
    graph.reset()
    torch.cuda.synchronize()
    del graph, workspace
    gc.collect()
    assert borrowed() is None


def test_workspace_output_overlap_and_unsupported_layout_refuse_before_writes(bank):
    from tensorfold.cuda.sm70 import gguf_ops

    workspace = owner(10)
    x = torch.ones((4, 2560), dtype=torch.float16, device="cuda")
    picks = routes(workspace, 4, 10, concentrated=True)
    output = workspace.y[:4]
    output.fill_(-739)
    original = workspace.act
    workspace.act = output.view(torch.float16).reshape(-1)[:workspace.act.numel()].view_as(workspace.act)
    with pytest.raises(RuntimeError, match="overlap"):
        gguf_ops.experts_grouped(x, picks, bank.gate, bank.up, bank.down, bank.shared, output, workspace)
    assert torch.all(output == -739)
    workspace.act = original
    bad = torch.empty((4, 11, 5120), dtype=torch.float32, device="cuda")[:, :, ::2]
    with pytest.raises(RuntimeError, match="contiguous"):
        gguf_ops.experts_grouped(x, picks, bank.gate, bank.up, bank.down, bank.shared, bad, workspace)
    assert torch.all(output == -739)

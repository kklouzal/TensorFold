"""Unselected owned ETP4 candidate controls; requires actual SM70 execution.

Exact comparisons use the existing packed operator as an arithmetic control.
These component controls do not qualify model quality or distributed collectives.
"""
import pytest
import torch

from gguf_fixture import blocks
from tensorfold.cuda.sm70 import gguf_ops
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFExperts, GGUFPacked
from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition
from tensorfold.gguf_file import FORMATS

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="requires actual SM70 owned experts")


def make_bank(*, group=0, dims=256, kinds=(13, 12, 7)):
    part = ExpertPartition(8, group * 4, 512, 640, 4)
    width, local = part.width, part.count
    physical = -(-(16 + width) // FORMATS[kinds[2]][1]) * FORMATS[kinds[2]][1]
    packed = [GGUFPacked(torch.tensor(blocks(kind, n, physical_k, seed=731 + i), device="cuda"),
                         kind, n, k, offset)
              for i, (kind, n, physical_k, k, offset) in enumerate([
                  (kinds[0], local * width, dims, dims, 0),
                  (kinds[1], local * width, dims, dims, 0),
                  (kinds[2], local * dims, physical, width, 16)])]
    torch.manual_seed(734)
    shared = [GGUFDense(torch.randn(shape, device="cuda", dtype=torch.float16) * .02)
              for shape in [(80, dims), (80, dims), (dims, 80)]]
    return GGUFExperts(*packed, shared, count=512, width=width, dims=dims, partition=part)


def inputs(bank, rows, dtype=torch.int64):
    torch.manual_seed(735)
    x = torch.randn((rows, bank.dims), device="cuda", dtype=torch.float16) * .1
    ids = torch.tensor([0, 255, 256, 511, 512], device="cuda", dtype=dtype)
    picks = ids.repeat(rows, 1)
    local = torch.empty_like(picks, dtype=torch.int64)
    activation = torch.empty((picks.numel(), bank.width), device="cuda", dtype=torch.float16)
    out = torch.full((*picks.shape, bank.dims), float("nan"), device="cuda")
    return x, picks, local, activation, out


def run(bank, data):
    x, picks, local, activation, out = data
    return bank.forward_partitioned_native_candidate(x, picks, local=local, activation=activation, out=out)


def arithmetic_control(bank, x, picks):
    """Reuse original native math with valid local IDs, mask foreign slots by assignment."""
    p = bank.partition
    mapped = torch.where((picks >= p.first) & (picks < p.stop), picks - p.first, -1)
    # No shared local sentinel is passed. Original native validates matching
    # shared width, so this independent control supplies unused zero matrices.
    dummy = [GGUFDense(torch.zeros(shape, dtype=torch.float16, device="cuda"))
             for shape in [(160, bank.dims), (160, bank.dims), (bank.dims, 160)]]
    control = GGUFExperts(bank.gate, bank.up, bank.down, dummy,
                          count=p.count, width=160, dims=bank.dims)
    expected = control.forward_exact(x, mapped) if bank.dims == 2560 else control.forward_blocks2w(x, mapped)
    expected.masked_fill_((mapped < 0)[..., None], 0)
    bank._write_shared(x, picks, expected)
    return expected


@pytest.mark.parametrize("group", [0, 1])
@pytest.mark.parametrize("rows", [1, 3, 17, 33])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_global_bounds_foreign_zero_shared_and_tail_reuse_existing_arithmetic(group, rows, dtype):
    bank = make_bank(group=group)
    data = inputs(bank, rows, dtype)
    actual = run(bank, data)
    assert actual is data[-1]
    expected = arithmetic_control(bank, data[0], data[1])
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.isfinite(actual).all()
    data[1].fill_(256 if group == 0 else 255)
    actual.fill_(float("nan"))
    assert torch.count_nonzero(run(bank, data)) == 0


@pytest.mark.parametrize("kinds", [(8, 20, 7), (12, 13, 20), (13, 12, 7)])
def test_capture_replays_changed_routes_and_publishes_invalid_without_bank_reads(kinds):
    bank = make_bank(kinds=kinds)
    data = inputs(bank, 3)
    run(bank, data)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run(bank, data)
    data[1].copy_(torch.tensor([[512, 511, 255, 0, 256], [0, 255, 512, 511, 256],
                              [256, 511, 0, 512, 255]], device="cuda"))
    data[0].mul_(.5)
    data[-1].fill_(float("nan"))
    data[3].fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    expected = arithmetic_control(bank, data[0], data[1])
    assert torch.equal(data[-1].view(torch.uint8), expected.view(torch.uint8))
    data[1][0] = torch.tensor([-1, 513, 2**40, 512, 255], device="cuda")
    graph.replay()
    torch.cuda.synchronize()
    assert torch.isnan(data[-1][0, :3]).all()
    assert torch.isfinite(data[-1][0, 3:]).all()


def test_actual_target_down_geometry_and_alias_scratch_refusal():
    bank = make_bank(dims=2560, kinds=(12, 13, 7))
    data = inputs(bank, 1)
    assert torch.equal(run(bank, data).view(torch.uint8), arithmetic_control(bank, data[0], data[1]).view(torch.uint8))
    with pytest.raises(RuntimeError, match="overlap"):
        bank.forward_partitioned_native_candidate(data[0], data[1], local=data[1], activation=data[3], out=data[4])
    with pytest.raises(RuntimeError, match="exact admitted shapes"):
        bank.forward_partitioned_native_candidate(data[0], data[1], local=data[2], activation=data[3][:-1], out=data[4])


def test_two_method_roles_keep_independent_native_scratch_and_outputs():
    """Target/MTP bank-method contract, component fixtures rather than model import."""
    target = make_bank(group=0)
    companion = make_bank(group=1)
    target_rows = inputs(target, 3)
    companion_rows = inputs(companion, 1)
    assert target_rows[2].data_ptr() != companion_rows[2].data_ptr()
    assert target_rows[3].data_ptr() != companion_rows[3].data_ptr()
    target_result = run(target, target_rows).clone()
    run(companion, companion_rows)
    assert torch.equal(target_rows[-1], target_result)
    assert torch.equal(companion_rows[-1].view(torch.uint8),
                       arithmetic_control(companion, companion_rows[0], companion_rows[1]).view(torch.uint8))
    assert gguf_ops.owned_workspace_bytes(3, 5) + gguf_ops.owned_workspace_bytes(1, 5) == sum(
        value.numel() * value.element_size() for data in (target_rows, companion_rows) for value in data[2:4])

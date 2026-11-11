"""Actual active GPU frame→GU→current shared/down, no retained plane cache."""
import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import gguf_ops, onecat_moe_gu, onecat_gpu_packing
from tensorfold.families.qwen4_exp.expert_partition import ExpertPartition
from tensorfold.families.qwen4_exp.cuda.gguf_weights import GGUFDense, GGUFExperts, GGUFPacked
from tensorfold.families.qwen4_exp.cuda.onecat_experts import OneCatGU, _Bank, _Lease
from tensorfold.families.qwen4_exp.cuda.onecat_frame_experts import _View

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM70 1Cat runtime composition")


@pytest.fixture(scope="module")
def prepared():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 1Cat runtime composition")
    device = torch.device("cuda", torch.cuda.current_device())
    gu = onecat_moe_gu.prepare(device)
    down = gguf_ops.prepare_owned_from_activation(device)
    packer = onecat_gpu_packing.prepare(device)
    frame = torch.empty(117_964_800, dtype=torch.uint8, device=device)
    facts = onecat_gpu_packing.layout(23)
    gate_view, up_view = _View(23), _View(23)
    offset = 0
    for view in (gate_view, up_view):
        count = facts["codes_shape"][0] * facts["codes_shape"][1]
        view.codes = frame[offset:offset + count].view(facts["codes_shape"])
        offset += count
        count = facts["scales_shape"][0] * facts["scales_shape"][1]
        view.scales = frame[offset:offset + count].view(facts["scales_shape"])
        offset += count

    def gate(scale):
        raw = np.full((256 * 160, 10, 136), 0x88, np.uint8)
        raw[:, :, :2] = np.array([scale], dtype="<f2").view(np.uint8)
        raw[:, :, 2:4] = 0xAA
        raw[:, :, 4:8] = 0x11
        cpu = torch.from_numpy(raw.reshape(40960, 1360))
        return GGUFPacked(cpu.to(device), 23, 40960, 2560)

    gate_raw = gate(.5)
    up_raw = gate(.25)
    down_raw = torch.full((256 * 2560, 10, 18), 0xAA, dtype=torch.uint8, device=device)
    down_raw[:, :, :2] = torch.tensor([1.], dtype=torch.float16, device=device).view(torch.uint8)
    packed_down = GGUFPacked(down_raw.reshape(256 * 2560, 180), 42, 256 * 2560, 160, 160)
    shared_gate = torch.zeros((80, 2560), dtype=torch.float16, device=device)
    shared_gate[:, 0] = .5
    shared_up = torch.zeros_like(shared_gate)
    shared_up[:, 0] = .25
    shared_down = torch.full((2560, 80), .5, dtype=torch.float16, device=device)
    shared = tuple(GGUFDense(value) for value in (shared_gate, shared_up, shared_down))
    table = torch.empty(0, dtype=torch.uint8, device=device)
    try:
        yield gu, down, packer, gate_raw, up_raw, packed_down, gate_view, up_view, shared, table, frame
    finally:
        torch.cuda.synchronize(device)


@pytest.mark.parametrize("method", onecat_moe_gu.METHODS)
@pytest.mark.parametrize("rows", [1, 24, 32])
@pytest.mark.parametrize("first", [0, 256])
def test_real_composition_preserves_original_down_shared_foreign_and_arena(prepared, method, rows, first):
    gu, down, packer, gate_raw, up_raw, packed_down, gate_sink, up_sink, shared, table, frame = prepared
    partition = ExpertPartition(8, 0 if first == 0 else 4, 512, 640, 4)
    bank = GGUFExperts(gate_raw, up_raw, packed_down, shared, count=512, width=160, dims=2560, partition=partition)
    x = torch.zeros((rows, 2560), dtype=torch.float16, device="cuda")
    x[:, 0] = 1.
    picks = torch.tensor([first, first + 63, first + 127, first + 191, first + 255,
                          256 - first, 257 - first, 383 - first, 447 - first, 511 - first, 512],
                         dtype=torch.int32, device="cuda").repeat(rows, 1)
    local = torch.empty((rows, 11), dtype=torch.int64, device="cuda")
    activation = torch.empty((rows * 11, 160), dtype=torch.float16, device="cuda")
    out = torch.empty((rows, 11, 2560), dtype=torch.float32, device="cuda")
    grouping = torch.empty(gguf_ops.grouped_workspace_words(rows * 11, 256), dtype=torch.int32, device="cuda")
    # Actual current incumbent is independent; it consumes original raw banks.
    bank.forward_partitioned_native_candidate(x, picks, local=local, activation=activation, out=out,
                                              grouping=grouping, _gate_up=gguf_ops._GroupedGateUp(4))
    expected = out.clone()
    lease = _Lease()
    lease.enabled = True  # Lower component fixture, not a forged whole-model admission.
    spec = OneCatGU(method, 117_976_064, 1 << 20)
    provider = _Bank(spec, lease, gu, down, gate_sink, up_sink, table, partition, bank, packer=packer)
    assert all(t.untyped_storage().data_ptr() == frame.untyped_storage().data_ptr()
               for t in (gate_sink.codes, gate_sink.scales, up_sink.codes, up_sink.scales))
    assert provider.run_slots(bank, x, picks, local=local, activation=activation, out=out, grouping=grouping) is out
    torch.testing.assert_close(out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)
    assert torch.count_nonzero(out[:, 5:10]) == 0
    assert torch.count_nonzero(out[:, 10]) == rows * 2560
    torch.cuda.synchronize()

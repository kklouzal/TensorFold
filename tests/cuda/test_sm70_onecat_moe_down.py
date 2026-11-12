"""Active Q8_1 down candidates versus original wire and actual incumbent.

Exact onehot/power-of-two cases isolate the representation and fold contracts.
General output drift is recorded, not accepted by a fitted tolerance. Shared
is independent; no live b.y/shared tensor may be used for component scratch.
"""

from types import SimpleNamespace as NS
import json

import numpy as np
import pytest
import torch

from tensorfold.cuda.sm70 import onecat_moe_down as down, onecat_moe_down_packing as packing, gguf_ops, family_ops
from tests.onecat_down_oracle import q8_block, route_bits, quantization_bound

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM70 Q8_1 down component")


@pytest.fixture(scope="module")
def prepared():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 down component")
    return down.prepare(torch.device("cuda", torch.cuda.current_device()))


@pytest.fixture(scope="module")
def incumbent_gu_banks():
    # Original public grouped GU/down, independent of the new split bridge.
    def bank(scale):
        blocks = torch.full((256 * 160, 10, 136), 0x88, dtype=torch.uint8, device="cuda")
        blocks[:, :, :2] = torch.tensor([scale], dtype=torch.float16, device="cuda").view(torch.uint8)
        blocks[:, :, 2:4] = 0xAA
        blocks[:, :, 4:8] = 0x11
        return NS(data=blocks.reshape(256 * 160, 1360), ggml_type=23, column_offset=0)

    return bank(32.0), bank(127.0 / 32.0)


@pytest.fixture(scope="module")
def raw_crop():
    # Independent original GGUF18-byte blocks, NOT manually invented DMV planes.
    blocks = np.empty((2560, 20, 18), np.uint8)
    blocks[:, :, :2] = np.array([2**-7], dtype="<f2").view(np.uint8)
    row = np.arange(2560)[:, None, None]
    group = np.arange(20)[None, :, None]
    column = np.arange(16)[None, None, :]
    blocks[:, :, 2:] = ((row + column + group) % 16 | (((row + 3 * column + 2 * group) % 16) << 4)).astype(np.uint8)
    bank = np.empty((512, 2560, 360), np.uint8)
    for expert in range(512):
        bank[expert] = blocks.reshape(2560, 360)
        scales = bank[expert].reshape(2560, 20, 18)[:, :, :2]
        scales[:] = np.array([2 ** -(7 + expert % 4 + expert // 256)], dtype="<f2").view(np.uint8)
    result = {}
    for first in (0, 256):
        for lane in (0, 3):
            output = np.empty(packing.requirements(ggml_type=20)["output_shape"], np.uint8)
            receipt = packing.pack_owned_into(bank, output, first=first, lane=lane, ggml_type=20)
            result[first, lane] = (torch.from_numpy(output).cuda(), receipt)
    return result


def operands(rows, first, method, raw):
    activation = torch.zeros((rows * 11, 160), dtype=torch.float16, device="cuda")
    indices = torch.arange(rows * 11, device="cuda") % 160
    activation[torch.arange(rows * 11, device="cuda"), indices] = 127.0
    activation.view(rows, 11, 160)[:, 10].zero_()
    picks = torch.arange(first, first + 11, dtype=torch.int64, device="cuda").repeat(rows, 1)
    picks[:, 10] = 512
    weights = (torch.arange(1, 12, device="cuda", dtype=torch.float32) / 16).repeat(rows, 1)
    q8 = torch.empty((rows * 11, 5, 36), dtype=torch.uint8, device="cuda")
    slots = torch.empty((rows * 11, 2560), dtype=torch.float16, device="cuda")
    status = torch.empty(rows, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, 2560), dtype=torch.float16 if method == down.METHODS[0] else torch.float32, device="cuda")
    return activation, picks, weights, raw, q8, slots, status, out


@pytest.mark.parametrize("rows", [1, 2, 3, 8, 9, 32])
@pytest.mark.parametrize("first", [0, 256])
@pytest.mark.parametrize("lane", [0, 3])
def test_original_iq4nl_crop_to_both_methods_and_actual_incumbent(
    prepared, incumbent_gu_banks, raw_crop, rows, first, lane, record_property
):
    raw, receipt = raw_crop[first, lane]
    args = operands(rows, first, down.METHODS[1], raw)
    activation, picks, weights, _, q8, slots, status, out = args
    picks[:, :10] = torch.tensor([first + i for i in (0, 1, 2, 3, 63, 127, 191, 253, 254, 255)], device="cuda")
    gate, up = incumbent_gu_banks
    x = torch.zeros((rows, 2560), device="cuda", dtype=torch.float16)
    x[:, 0] = 1.0
    shared = torch.zeros_like(x, dtype=torch.float32)
    local = torch.empty_like(picks, dtype=torch.int64)
    grouping = torch.empty(gguf_ops.grouped_workspace_words(rows * 11, 256), device="cuda", dtype=torch.int32)
    incumbent_slots = torch.empty((rows, 11, 2560), device="cuda", dtype=torch.float16)
    gguf_ops.experts_owned(
        x,
        picks,
        gate,
        up,
        NS(data=raw, ggml_type=20, column_offset=0),
        first=first,
        experts=512,
        width=160,
        shared=shared,
        local=local,
        activation=activation,
        out=incumbent_slots,
        grouping=grouping,
        _gate_up=gguf_ops._GroupedGateUp(4),
    )
    assert torch.all(activation.view(rows, 11, 160)[:, :10] == 127.0)
    # Old GU has no producer-status ABI; its unused shared row is reset only
    # after the independent incumbent has consumed the original activation.
    activation.view(rows, 11, 160)[:, 10].zero_()
    incumbent_fold = torch.empty((rows, 2560), device="cuda")
    family_ops._reference["moe_partial"](incumbent_slots, weights, incumbent_fold, rows)
    prepared.down(*args, first=first, ggml_type=20, column_offset=0, method=down.METHODS[1])
    assert torch.count_nonzero(status) == 0
    assert torch.equal(slots.view(torch.int16), incumbent_slots.reshape(rows * 11, 2560).view(torch.int16))
    assert torch.equal(out.view(torch.int32), incumbent_fold.view(torch.int32))
    original = torch.empty((rows, 2560), device="cuda", dtype=torch.float16)
    prepared.down(*args[:-1], original, first=first, ggml_type=20, column_offset=0, method=down.METHODS[0])
    # These bounded dyadic terms sum exactly inF32 in every permitted fold order.
    assert torch.equal(original.view(torch.int16), incumbent_fold.half().view(torch.int16))
    record_property("original_raw_source_receipt", json.dumps(receipt, sort_keys=True))


@pytest.mark.parametrize("rows", [1, 8, 24, 32])
@pytest.mark.parametrize("scale", [2**-24, 0.01, 1.0, 1000.0])
def test_quantization_bytes_and_general_original_dot_differences(prepared, raw_crop, rows, scale, record_property):
    args = operands(rows, 0, down.METHODS[1], raw_crop[0, 0][0])
    generator = torch.Generator(device="cuda").manual_seed(8171 + rows)
    args[0][:] = torch.randn(args[0].shape, generator=generator, device="cuda").half() * scale
    args[0].view(rows, 11, 160)[:, 10].zero_()
    prepared.down(*args, first=0, ggml_type=20, column_offset=0, method=down.METHODS[1])
    assert torch.count_nonzero(args[6]) == 0
    observations = []
    for route, group in [(0, 0), ((rows - 1) * 11 + 9, 4)]:
        bits = [int(v) & 65535 for v in args[0][route, group * 32 : (group + 1) * 32].view(torch.int16).cpu().tolist()]
        expected = q8_block(bits)
        actual = bytes(args[4][route, group].cpu().tolist())
        assert actual == expected
        observations.append(
            {"route": route, "group": group, "activation_error_bound": str(quantization_bound(bits, actual))}
        )
    # Independent exact integer/FP32 scale arithmetic, not candidate decoder reuse.
    rawrow = bytes(args[3][0].cpu().tolist())
    wire = bytes(args[4][0].cpu().reshape(-1).tolist())
    expected = route_bits(rawrow, wire)
    assert int(args[5][0, 0].view(torch.int16).item()) & 65535 == expected
    record_property(
        "Q8_1_quantization_observations",
        json.dumps(
            {
                "samples": observations,
                "quality_acceptance": False,
                "P3_valid": False,
                "original_activation_equivalence": False,
            }
        ),
    )


@pytest.mark.parametrize(
    "fault", ["foreign", "invalid", "duplicate", "shared", "nonfinite-activation", "producer-status"]
)
@pytest.mark.parametrize("method", down.METHODS)
def test_safe_foreign_shared_and_poison_before_invalid_conversion(prepared, raw_crop, fault, method):
    args = operands(2, 256, method, raw_crop[0, 0][0])
    if fault == "foreign":
        args[1][0, :10] -= 256
    elif fault == "invalid":
        args[1][0, 0] = 2**40
    elif fault == "duplicate":
        args[1][0, 1] = args[1][0, 0]
    elif fault == "shared":
        args[1][0, 10] = 511
    elif fault == "producer-status":
        args[1][0, :10] -= 256
        args[0].view(2, 11, 160)[0, 10, 0] = float("nan")
    else:
        args[0][0, 0] = float("nan")
    args[5].fill_(-17.0)
    slots_before = args[5].clone()
    slot_version = args[5]._version
    prepared.down(*args, first=256, ggml_type=20, column_offset=0, method=method)
    if method == down.METHODS[0]:
        assert torch.equal(args[5].view(torch.int16), slots_before.view(torch.int16))
        assert args[5]._version == slot_version
    if fault == "foreign":
        assert args[6][0] == 0 and torch.count_nonzero(args[-1][0]) == 0
        if method == down.METHODS[1]:
            assert torch.count_nonzero(args[5][:11]) == 0
    else:
        assert args[6][0] != 0 and torch.isnan(args[-1][0]).all()
        if method == down.METHODS[1]:
            assert torch.isnan(args[5][:11]).all()
    assert args[6][1] == 0 and torch.isfinite(args[-1][1]).all()
    if method == down.METHODS[1]:
        assert torch.count_nonzero(args[5].view(2, 11, 2560)[1, 10]) == 0


@pytest.mark.parametrize("method", down.METHODS)
def test_nondefault_stream_capture_replay_versions_and_alias_refusal(prepared, raw_crop, method):
    args = operands(9, 0, method, raw_crop[0, 0][0])
    args[5].fill_(-17.0)
    slots_before = args[5].clone()
    versions = [t._version for t in args[4:]]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared.down(*args, first=0, ggml_type=20, column_offset=0, method=method)
    torch.cuda.current_stream().wait_stream(stream)
    assert [t._version for t in args[4:]] == [v + (0 if method == down.METHODS[0] and i == 1 else 1) for i, v in enumerate(versions)]
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.down(*args, first=0, ggml_type=20, column_offset=0, method=method)
    before = args[-1].clone()
    args[0].mul_(0.5)
    graph.replay()
    torch.cuda.synchronize()
    bits = torch.int16 if method == down.METHODS[0] else torch.int32
    assert torch.equal(args[-1].view(bits), (before * 0.5).view(bits))
    if method == down.METHODS[0]:
        assert torch.equal(args[5].view(torch.int16), slots_before.view(torch.int16))
    bad = list(args)
    bad[-1] = args[5].view(args[-1].dtype).reshape(-1)[: 9 * 2560].view(9, 2560)
    with pytest.raises(RuntimeError, match="overlap"):
        prepared.down(*bad, first=0, ggml_type=20, column_offset=0, method=method)


def test_original_inference_slots_unchanged_success_foreign_and_poison(prepared, raw_crop):
    with torch.inference_mode():
        args = operands(3, 256, down.METHODS[0], raw_crop[256, 0][0])
        args[5].fill_(-17.0)
        before = args[5].clone()
        args[1][1, :10] -= 256
        args[1][2, 0] = -1
        prepared.down(*args, first=256, ggml_type=20, column_offset=0, method=down.METHODS[0])
        assert args[5].is_inference() and torch.equal(args[5].view(torch.int16), before.view(torch.int16))
        assert args[6][0] == 0 and torch.isfinite(args[-1][0]).all()
        assert args[6][1] == 0 and torch.count_nonzero(args[-1][1]) == 0
        assert args[6][2] != 0 and torch.isnan(args[-1][2]).all()


@pytest.mark.parametrize('dtype,bits', [(torch.float16, 16), (torch.float32, 32)])
@pytest.mark.parametrize('rows,first,lane', [(1, 0, 0), (9, 256, 3), (32, 0, 3)])
def test_unweighted_slots_independent_q8_oracle_shared_and_foreign(prepared, raw_crop, dtype, bits, rows, first, lane):
    raw = raw_crop[first, lane][0]
    activation, picks, _, _, q8, _, status, _ = operands(rows, first, down.METHODS[1], raw)
    generator = torch.Generator(device='cuda').manual_seed(93971 + rows)
    activation[:] = torch.randn(activation.shape, generator=generator, device='cuda').half()
    activation.view(rows, 11, 160)[:, 10].zero_()
    picks[:, 5:10] = torch.arange(256 - first, 261 - first, device='cuda')
    shared = torch.full((rows, 2560), .12345679, dtype=torch.float32, device='cuda')
    out = torch.full((rows, 11, 2560), float('nan'), dtype=dtype, device='cuda')
    prepared.slots(activation, picks, raw, shared, q8, status, out,
                   first=first, ggml_type=20, column_offset=0)
    assert torch.count_nonzero(status) == 0
    assert torch.count_nonzero(out[:, 5:10]) == 0
    assert torch.equal(out[:, 10], shared.to(dtype))
    for row, slot, column in [(0, 0, 0), (rows - 1, 4, 2559)]:
        route = row * 11 + slot
        for group in (0, 4):
            source = [int(v) & 65535 for v in activation[route, group*32:(group+1)*32].view(torch.int16).cpu().tolist()]
            assert bytes(q8[route, group].cpu().tolist()) == q8_block(source)
        expert = int(picks[row, slot].item()) - first
        rawrow = bytes(raw[expert*2560+column].cpu().tolist())
        wire = bytes(q8[route].cpu().reshape(-1).tolist())
        expected = route_bits(rawrow, wire, output_bits=bits)
        actual = out[row, slot, column].view(torch.int16 if bits == 16 else torch.int32).item()
        assert actual & ((1 << bits) - 1) == expected


@pytest.mark.parametrize('fault', ['duplicate', 'invalid', 'shared', 'producer', 'nonfinite'])
def test_unweighted_slots_poison_all_outputs_and_preserve_readonly_shared(prepared, raw_crop, fault):
    raw = raw_crop[0, 0][0]
    activation, picks, _, _, q8, _, status, _ = operands(2, 0, down.METHODS[1], raw)
    if fault == 'duplicate':
        picks[0, 1] = picks[0, 0]
    elif fault == 'invalid':
        picks[0, 0] = 2**40
    elif fault == 'shared':
        picks[0, 10] = 511
    elif fault == 'producer':
        activation[10, 0] = -0.
    else:
        activation[0, 0] = float('inf')
    shared = torch.full((2, 2560), .12345679, dtype=torch.float32, device='cuda')
    before = shared.clone()
    out = torch.empty((2, 11, 2560), dtype=torch.float32, device='cuda')
    prepared.slots(activation, picks, raw, shared, q8, status, out,
                   first=0, ggml_type=20, column_offset=0)
    assert status[0].item() != 0 and status[1].item() == 0
    assert torch.isnan(out[0]).all() and torch.isfinite(out[1]).all()
    assert torch.equal(shared, before)


def test_unweighted_slots_capture_replay_consumes_changed_route_ids(prepared, raw_crop):
    raw = raw_crop[0, 0][0]
    activation, picks, _, _, q8, _, status, _ = operands(9, 0, down.METHODS[1], raw)
    shared = torch.full((9, 2560), .12345679, dtype=torch.float32, device='cuda')
    output = torch.empty((9, 11, 2560), dtype=torch.float32, device='cuda')
    def execute(target):
        prepared.slots(activation, picks, raw, shared, q8, status, target,
                       first=0, ggml_type=20, column_offset=0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        execute(output)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        execute(output)
    picks[:, 0] = 255
    picks[:, 5:10] = torch.arange(256, 261, device='cuda')
    graph.replay()
    actual = output.clone()
    execute(output)
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.int32), output.view(torch.int32))

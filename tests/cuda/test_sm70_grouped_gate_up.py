"""Private original-format SIMT grouping against a separately frozen incumbent ELF."""
import torch
import pytest

from tensorfold.cuda.sm70 import gguf_ops
from tests.cuda import test_sm70_grouped_down160 as incumbent

native_pair = incumbent.native_pair

pytestmark = pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
                                reason="requires actual SM70 and frozen incumbent ELF")


@pytest.fixture(scope="module", params=[(18, 18, 20), (22, 22, 42)])
def resident_banks(request):
    kinds = request.param
    return kinds, (incumbent.packed(kinds[0], 256 * 160, 2560), incumbent.packed(kinds[1], 256 * 160, 2560),
                   incumbent.packed(kinds[2], 256 * 2560, 640))


def invoke(module, inputs, kinds, output, offset, rows=None, first=0):
    x, picks, gate, up, down, shared = inputs
    local = torch.empty_like(picks, dtype=torch.int64)
    activation = torch.empty((picks.numel(), 160), device=x.device, dtype=torch.float16)
    grouping = torch.empty(gguf_ops.grouped_workspace_words(picks.numel(), gate.shape[0] // 160),
                           device=x.device, dtype=torch.int32)
    args = (x, picks, gate, kinds[0], 0, up, kinds[1], 0, down, kinds[2], offset,
            160, first, 512, shared, local, activation, output, grouping)
    def call():
        if rows is None:
            module.experts_owned_grouped(*args)
        else:
            module._experts_owned_grouped_gate_up(*args, rows)
    return call, local, activation


@pytest.mark.parametrize("tile", [2, 4])
def test_actual_formats_ragged_rows_original_slots_and_casts(native_pair, resident_banks, tile):
    old, new = native_pair
    kinds, banks = resident_banks
    for case, rows in enumerate((1, 3, 5, 17, 40, 160, 256)):
        dtype = torch.float32 if case % 2 else torch.float16
        index = torch.int32 if case % 2 else torch.int64
        offset = (0, 160, 320, 480, 16)[case % 5]
        first = 0 if case % 2 else 256
        foreign = 256 if first == 0 else 0
        inputs, actual = incumbent.operands(kinds, rows, dtype, index, offset=offset, first=first,
                                           resident_experts=256, banks=banks)
        expected = torch.empty_like(actual)
        control, old_local, old_act = invoke(old, inputs, kinds, expected, offset, first=first)
        candidate, new_local, new_act = invoke(new, inputs, kinds, actual, offset, tile, first=first)
        # Unique router-like rows, deep >256 bins, sparse/foreign/shared/invalid.
        patterns = [[[(first + (i * 19 + row * 13) % 256) for i in range(10)] + [512] for row in range(rows)],
                    [[first + 255] * 7 + [first, foreign, 512, 513] for _ in range(rows)],
                    [[foreign, 512, -1, 513, first + 255, first, first + 127, 2**31 - 1, foreign, 512, first + 255] for _ in range(rows)]]
        if rows in (5, 256):
            patterns += [[[foreign] * 11 for _ in range(rows)], [[512] * 11 for _ in range(rows)]]
        for pattern in patterns:
            inputs[1].copy_(torch.tensor(pattern, device="cuda", dtype=index))
            control()
            candidate()
            incumbent.bits(new_local, old_local)
            incumbent.bits(new_act, old_act)  # includes every foreign/shared/invalid half NaN
            incumbent.bits(actual, expected)


@pytest.mark.parametrize("tile", [2, 4])
def test_replayed_routes_reset_counts_and_preserve_sensitive_half_inputs(native_pair, resident_banks, tile):
    old, new = native_pair
    kinds, banks = resident_banks
    inputs, actual = incumbent.operands(kinds, 17, torch.float32, torch.int64, offset=480,
                              resident_experts=256, banks=banks)
    values = torch.tensor([0., -0., 2**-24, -2**-24, 2**-14, -2**-14, .25, -.25,
                           65504., -65504., 32752., -32752., 1., -1., 16., -16.],
                          device="cuda", dtype=torch.float16)
    inputs[0].copy_(values.repeat(17 * 2560 // 16).reshape(17, 2560))
    expected = torch.empty_like(actual)
    control, old_local, old_act = invoke(old, inputs, kinds, expected, 480)
    candidate, new_local, new_act = invoke(new, inputs, kinds, actual, 480, tile)
    candidate()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        candidate()
    for ids in ([255] * 11, [256] * 11, [0, 255, 512, -1, 513, 0, 255, 256, 0, 512, 255]):
        inputs[1].copy_(torch.tensor([ids] * 17, device="cuda", dtype=torch.int64))
        control()
        graph.replay()
        incumbent.bits(new_local, old_local)
        incumbent.bits(new_act, old_act)
        incumbent.bits(actual, expected)


def test_native_private_entry_rejects_selector_before_touching_inputs(native_pair):
    _, new = native_pair
    # Valid tensors are not needed: bad selector is rejected before CUDA reads.
    empty = torch.empty(0)
    args = (empty, empty, empty, 18, 0, empty, 18, 0, empty, 20, 0,
            160, 0, 512, empty, empty, empty, empty, empty)
    for rows in (0, 1, 3, 8):
        with pytest.raises(RuntimeError, match="exactly 2 or 4"):
            new._experts_owned_grouped_gate_up(*args, rows)


@pytest.mark.parametrize("tile", [2, 4])
def test_default_grouped_entry_and_empty_native_extent_are_preserved(native_pair, tile):
    old, new = native_pair
    kinds = (18, 18, 20)
    inputs, actual = incumbent.operands(kinds, 3, torch.float16, torch.int64, offset=16)
    expected = torch.empty_like(actual)
    control, old_local, old_act = invoke(old, inputs, kinds, expected, 16)
    default, new_local, new_act = invoke(new, inputs, kinds, actual, 16)
    control()
    default()
    incumbent.bits(actual, expected)
    incumbent.bits(new_local, old_local)
    incumbent.bits(new_act, old_act)
    x, _, gate, up, down, shared = inputs
    empty_inputs = (x[:0], torch.empty((0, 11), dtype=torch.int64, device="cuda"),
                    gate, up, down, shared[:0])
    empty = actual[:0]
    call, local, activation = invoke(new, empty_inputs, kinds, empty, 16, tile)
    call()
    assert local.numel() == activation.numel() == empty.numel() == 0


@pytest.mark.parametrize("tile", [2, 4])
@pytest.mark.parametrize("kinds", incumbent.KINDS)
def test_admitted_original_format_readers_preserve_their_paired_arithmetic(native_pair, tile, kinds):
    old, new = native_pair
    dtype = torch.float16 if kinds[0] % 2 else torch.float32
    index = torch.int32 if kinds[0] % 2 else torch.int64
    inputs, actual = incumbent.operands(kinds, 3, dtype, index, offset=16)
    expected = torch.empty_like(actual)
    control, old_local, old_act = invoke(old, inputs, kinds, expected, 16)
    candidate, new_local, new_act = invoke(new, inputs, kinds, actual, 16, tile)
    control()
    candidate()
    incumbent.bits(new_local, old_local)
    incumbent.bits(new_act, old_act)
    incumbent.bits(actual, expected)

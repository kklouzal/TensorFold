"""Component qualification only; neither method is selected by serving dispatch."""

import pytest
import torch

from tensorfold.cuda.sm70 import onecat_moe_gu as gu

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="SM70 component")


@pytest.fixture(scope="module")
def prepared():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip("SM70 component")
    return gu.prepare(torch.device("cuda", torch.cuda.current_device()))


def operands(rows, first):
    facts = gu.layout(rows, first, "IQ4_XS")
    # Every nibble8 decodes to IQ4NL value+1. Four repeated scale halves
    # describe each128-column group in the independent pinned LUT4 layout.
    gc = torch.full(facts["codes_shape"], 0x88, dtype=torch.uint8, device="cuda")
    gs = torch.full((256, facts["scales_shape"][1] // 2), 256., dtype=torch.float16, device="cuda").view(torch.uint8)
    uc = gc.clone()
    us = torch.full_like(gs.view(torch.float16), 2. ** -24).view(torch.uint8)
    x = torch.zeros((rows, 2560), dtype=torch.float16, device="cuda")
    x[:, 0] = 256
    picks = torch.arange(first, first + 11, dtype=torch.int64, device="cuda").repeat(rows, 1)
    picks[:, 10] = 512
    out = torch.empty(facts["activation_shape"], dtype=torch.float16, device="cuda")
    return x, picks, gc, gs, uc, us, torch.empty(0, dtype=torch.uint8, device="cuda"), out


@pytest.mark.parametrize("rows", [1, 8, 9, 24, 32])
@pytest.mark.parametrize("first", [0, 256])
@pytest.mark.parametrize("method", gu.METHODS)
def test_exact_finite_operand_overflow_boundary_and_complete_panels(prepared, rows, first, method):
    args = operands(rows, first)
    version = args[-1]._version
    prepared.gate_up(*args, first=first, format_name="IQ4_XS", method=method)
    actual = args[-1].reshape(rows, 11, 160)
    # Independent dyadic dot: g=256*256=65536, u=256*2^-24=2^-16.
    # Original epilogue yields1. Upstream earlyHalf g overflows to+Inf.
    expected = torch.full_like(actual[:, :10], 1. if method == gu.METHODS[0] else float("inf"))
    torch.testing.assert_close(actual[:, :10], expected, rtol=0, atol=0)
    assert torch.count_nonzero(actual[:, 10]) == 0
    assert args[-1]._version == version + 1


@pytest.mark.parametrize("corruption", ["foreign", "invalid", "duplicate", "shared"])
def test_foreign_shared_and_poison_rows_never_read_invalid_bank(prepared, corruption):
    args = operands(2, 256)
    if corruption == "foreign":
        args[1][0, :10] -= 256
    elif corruption == "invalid":
        args[1][0, 0] = 2 ** 40
    elif corruption == "duplicate":
        args[1][0, 1] = args[1][0, 0]
    else:
        args[1][0, 10] = 511
    prepared.gate_up(*args, first=256, format_name="IQ4_XS", method=gu.METHODS[0])
    result = args[-1].reshape(2, 11, 160)
    if corruption == "foreign":
        assert torch.count_nonzero(result[0]) == 0
    else:
        assert torch.isnan(result[0]).all()
    torch.testing.assert_close(result[1, :10], torch.ones_like(result[1, :10]), rtol=0, atol=0)


def test_nondefault_stream_graph_replay_and_alias_refusal(prepared):
    args = operands(1, 0)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        prepared.gate_up(*args, first=0, format_name="IQ4_XS", method=gu.METHODS[0])
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        prepared.gate_up(*args, first=0, format_name="IQ4_XS", method=gu.METHODS[0])
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(args[-1][:10], torch.ones_like(args[-1][:10]), rtol=0, atol=0)
    bad = list(args)
    bad[-1] = args[0].view(-1)[:1760].view(11, 160)
    with pytest.raises(RuntimeError, match="overlap"):
        prepared.gate_up(*bad, first=0, format_name="IQ4_XS", method=gu.METHODS[0])


@pytest.mark.parametrize("rows", [24, 32])
@pytest.mark.parametrize("grouped_rows", [None, 4])
def test_actual_incumbent_packed_gu_remains_independent_third_arm(prepared, rows, grouped_rows):
    from types import SimpleNamespace
    from tensorfold.cuda.sm70 import gguf_ops

    args = operands(rows, 256)
    # Original IQ4_XS: scale bits33 encode scale+1 (33-32); code8=+1.
    # Construct original GGUF blocks independently of the new plane decoder.
    def bank(scale):
        blocks = torch.zeros((10 * 160, 10, 136), dtype=torch.uint8, device="cuda")
        scale_bytes = torch.tensor([scale], dtype=torch.float16, device="cuda").view(torch.uint8)
        blocks[:, :, :2] = scale_bytes
        blocks[:, :, 2:4] = 0xAA
        blocks[:, :, 4:8] = 0x11
        blocks[:, :, 8:] = 0x88
        return SimpleNamespace(data=blocks.reshape(10 * 160, 1360), ggml_type=23, column_offset=0)

    gate, up = bank(256.), bank(2. ** -24)
    down = SimpleNamespace(data=torch.zeros((10 * 2560, 160), dtype=torch.float16, device="cuda").view(torch.uint8),
                           ggml_type=1, column_offset=0)
    local = torch.empty_like(args[1])
    activation = torch.empty_like(args[-1])
    shared = torch.zeros_like(args[0], dtype=torch.float32)
    output = torch.empty((rows, 11, 2560), dtype=torch.float32, device="cuda")
    grouping, selector = None, None
    if grouped_rows is not None:
        grouping = torch.empty(gguf_ops.grouped_workspace_words(rows * 11, 10),
                               dtype=torch.int32, device="cuda")
        selector = gguf_ops._GroupedGateUp(grouped_rows)
    gguf_ops.experts_owned(args[0], args[1], gate, up, down, first=256, experts=512,
                          width=160, shared=shared, local=local, activation=activation, out=output,
                          grouping=grouping, _gate_up=selector)
    old = activation.reshape(rows, 11, 160)[:, :10].clone()
    torch.testing.assert_close(old, torch.ones_like(old), rtol=0, atol=0)
    prepared.gate_up(*args, first=256, format_name="IQ4_XS", method=gu.METHODS[0])
    torch.testing.assert_close(args[-1].reshape(rows, 11, 160)[:, :10], old, rtol=0, atol=0)
    prepared.gate_up(*args, first=256, format_name="IQ4_XS", method=gu.METHODS[1])
    assert torch.isposinf(args[-1].reshape(rows, 11, 160)[:, :10]).all()


@pytest.mark.parametrize("field", ["output", "codes", "scales", "table", "dtype", "alignment", "grad", "negative-input", "negative-output"])
def test_native_refusals_before_output_mutation(prepared, field):
    args = list(operands(1, 0))
    args[-1].fill_(19.)
    if field == "output":
        args[-1] = args[-1][:-1]
    elif field == "codes":
        args[2] = args[2][:-1]
    elif field == "scales":
        args[3] = args[3][:-1]
    elif field == "table":
        args[6] = torch.zeros(16, dtype=torch.uint8, device="cuda")
    elif field == "dtype":
        args[1] = args[1].to(torch.float32)
    elif field == "alignment":
        shape = args[2].shape
        args[2] = torch.empty(args[2].numel() + 1, dtype=torch.uint8, device="cuda")[1:].view(shape)
    elif field == "negative-input":
        args[0] = torch._neg_view(args[0])
    elif field == "negative-output":
        args[-1] = torch._neg_view(args[-1])
    else:
        args[0].requires_grad_(True)
    before = args[-1].clone()
    with pytest.raises(RuntimeError):
        prepared.gate_up(*args, first=0, format_name="IQ4_XS", method=gu.METHODS[0])
    torch.testing.assert_close(args[-1], before, rtol=0, atol=0)


def test_inference_output_outside_inference_mode_refused(prepared):
    args = list(operands(1, 0))
    with torch.inference_mode():
        args[-1] = torch.empty_like(args[-1])
    with pytest.raises(RuntimeError, match="inference output"):
        prepared.gate_up(*args, first=0, format_name="IQ4_XS", method=gu.METHODS[0])

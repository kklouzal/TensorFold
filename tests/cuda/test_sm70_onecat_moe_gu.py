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


@pytest.fixture(scope='module')
def raw_iq2_prepared():
    if torch.cuda.get_device_capability() != (7, 0):
        pytest.skip('SM70 raw IQ2 component')
    return gu.prepare_raw_iq2(torch.device('cuda', torch.cuda.current_device()))


@pytest.fixture(scope='module')
def raw_iq2_banks():
    import numpy as np
    from tensorfold.cuda.sm70 import onecat_gpu_packing as packing, onecat_moe_packing as cpu
    from tests.cuda.test_sm70_onecat_gpu_packing import raw_input, outputs
    gate, expected = raw_input(22, offset=256, strided=True, distinct=True)
    up, expected_up = raw_input(22, offset=256, strided=True, distinct=False)
    slab, planes = outputs(22, gate.device)
    producer = packing.prepare(gate.device)
    table = torch.from_numpy(np.asarray(cpu.table(22))).to(gate.device)
    return gate, up, expected, expected_up, slab, planes, producer, table


@pytest.mark.parametrize('expert,tile,group', [(0, 0, 0), (127, 2, 9), (255, 4, 19)])
def test_raw_iq2_defined_register_words_match_independent_CPU_wire(raw_iq2_prepared, raw_iq2_banks, expert, tile, group):
    import numpy as np
    from tensorfold.cuda.sm70 import onecat_moe_packing as cpu
    gate, _, expected, _, _, _, _, _ = raw_iq2_banks
    _, codes, meta = cpu._pack_expert(expected(expert), 22)
    c = codes.view('<u4').reshape(5, 20, 2, 32, 4)
    m = meta.view('<u4').reshape(5, 20, 32, 4)
    wanted = np.concatenate([c[tile, group, 0], c[tile, group, 1], m[tile, group]], axis=1)
    out = torch.empty((32, 12), dtype=torch.int32, device=gate.device)
    version = out._version
    raw_iq2_prepared.register_words(gate, out, expert=expert, tile=tile, group=group, column_offset=256)
    torch.cuda.synchronize()
    np.testing.assert_array_equal(out.cpu().numpy().view(np.uint32), wanted)
    assert out._version == version+1


@pytest.mark.parametrize('rows,first,dtype', [(1, 0, torch.int32), (8, 256, torch.int64),
                                            (9, 0, torch.int64), (24, 256, torch.int32), (32, 0, torch.int64)])
@pytest.mark.parametrize('method', gu.METHODS)
@torch.no_grad()
def test_raw_iq2_full_activation_bits_equal_activepack_then_incumbentGU(
        prepared, raw_iq2_prepared, raw_iq2_banks, rows, first, dtype, method):
    gate, up, _, _, _, planes, producer, table = raw_iq2_banks
    generator = torch.Generator(device=gate.device).manual_seed(1729)
    x = (torch.randn((rows, 2560), dtype=torch.float16, device=gate.device, generator=generator)*.01).contiguous()
    picks = torch.arange(first, first+11, dtype=dtype, device=gate.device).repeat(rows, 1)
    picks[:, 10] = 512
    if rows > 1:
        picks[1, :10] = torch.tensor([first+255, first+127, *range(first+12, first+20)], dtype=dtype, device=gate.device)
    if rows > 3:
        picks[2, :10] = torch.arange(256-first, 266-first, dtype=dtype, device=gate.device)
        picks[3, 1] = picks[3, 0]
    if rows > 5:
        picks[4, 10] = 511
        picks[5, 0] = -1
    expected = torch.full((rows*11, 160), -27, dtype=torch.float16, device=gate.device)
    actual = torch.full_like(expected, 23)
    producer.pack_active(gate, up, picks, *planes, first=first, ggml_type=22,
                          gate_column_offset=256, up_column_offset=256)
    prepared.gate_up(x, picks, *planes, table, expected, first=first, format_name='IQ2_S', method=method)
    version = actual._version
    assert raw_iq2_prepared.gate_up_raw_iq2(x, picks, gate, up, table, actual, first=first, method=method,
                          gate_column_offset=256, up_column_offset=256) is actual
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert actual._version == version+1


@pytest.mark.parametrize('bits', [0, 0x8000, 1, 0x8001, 0x7bff, 0xfbff, 0x7c00, 0xfc00, 0x7e11, 0xfe11])
@torch.no_grad()
def test_raw_iq2_Half_scale_classes_preserve_register_bits_and_both_epilogues(raw_iq2_prepared, prepared, bits):
    import numpy as np
    from tensorfold.cuda.sm70 import onecat_gpu_packing as packing, onecat_moe_packing as cpu
    from tests.cuda.test_sm70_onecat_gpu_packing import raw_input, outputs
    raw, _ = raw_input(22, offset=0, strided=False)
    original = raw[0].cpu().numpy().copy().reshape(160, 10, 82)
    original[:, :, :2] = np.array([bits], dtype='<u2').view(np.uint8)
    original[:, :, 34:66] = np.arange(32, dtype=np.uint8)  # nontrivial signs
    raw[0].copy_(torch.from_numpy(original.reshape(160, 820)))
    _, planes = outputs(22, raw.device)
    producer = packing.prepare(raw.device)
    table = torch.from_numpy(cpu.table(22)).to(raw.device)
    x = torch.full((1, 2560), 2.**-12, dtype=torch.float16, device=raw.device)
    picks = torch.arange(11, dtype=torch.int64, device=raw.device)[None]
    picks[:, 10] = 512
    producer.pack_active(raw, raw, picks, *planes, first=0, ggml_type=22)
    words = torch.empty((32, 12), dtype=torch.int32, device=raw.device)
    raw_iq2_prepared.register_words(raw, words, expert=0, tile=4, group=19)
    for method in gu.METHODS:
        expected = torch.empty((11, 160), dtype=torch.float16, device=raw.device)
        actual = torch.empty_like(expected)
        prepared.gate_up(x, picks, *planes, table, expected, first=0, format_name='IQ2_S', method=method)
        raw_iq2_prepared.gate_up_raw_iq2(x, picks, raw, raw, table, actual, first=0, method=method)
        torch.cuda.synchronize()
        assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    assert torch.all((words[:, 9] & 65535) == bits)


@torch.no_grad()
def test_raw_iq2_nondefault_stream_capture_replay_changed_routes_matches_control(prepared, raw_iq2_prepared, raw_iq2_banks):
    gate, up, _, _, _, planes, producer, table = raw_iq2_banks
    rows = 9
    x = torch.full((rows, 2560), 2.**-12, dtype=torch.float16, device=gate.device)
    picks = torch.arange(256, 267, dtype=torch.int32, device=gate.device).repeat(rows, 1)
    picks[:, 10] = 512
    actual = torch.empty((rows*11, 160), dtype=torch.float16, device=gate.device)
    expected = torch.empty_like(actual)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        raw_iq2_prepared.gate_up_raw_iq2(x, picks, gate, up, table, actual, first=256, method=gu.METHODS[0],
                                      gate_column_offset=256, up_column_offset=256)
    stream.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        raw_iq2_prepared.gate_up_raw_iq2(x, picks, gate, up, table, actual, first=256, method=gu.METHODS[0],
                                      gate_column_offset=256, up_column_offset=256)
    stream.synchronize()
    picks[:, :10] += 31
    x.mul_(2)
    graph.replay()
    producer.pack_active(gate, up, picks, *planes, first=256, ggml_type=22, gate_column_offset=256, up_column_offset=256)
    prepared.gate_up(x, picks, *planes, table, expected, first=256, format_name='IQ2_S', method=gu.METHODS[0])
    torch.cuda.synchronize()
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    del graph


@pytest.mark.parametrize('fault', ['raw-shape', 'raw-negative', 'raw-offset', 'output-alias', 'dtype'])
@torch.no_grad()
def test_raw_iq2_refuses_invalid_borrow_before_activation_publication(raw_iq2_prepared, raw_iq2_banks, fault):
    gate, up, _, _, _, _, _, table = raw_iq2_banks
    x = torch.zeros((1, 2560), dtype=torch.float16, device=gate.device)
    picks = torch.arange(11, dtype=torch.int64, device=gate.device)[None]
    picks[:, 10] = 512
    out = torch.full((11, 160), 19, dtype=torch.float16, device=gate.device)
    offset = 256
    if fault == 'raw-shape':
        gate = gate[:-1]
    elif fault == 'raw-negative':
        gate = torch._neg_view(gate)
    elif fault == 'raw-offset':
        offset = 512
    elif fault == 'output-alias':
        out = x.view(-1)[:1760].view(11, 160)
    else:
        picks = picks.float()
    before = out.clone()
    with pytest.raises((ValueError, RuntimeError)):
        raw_iq2_prepared.gate_up_raw_iq2(x, picks, gate, up, table, out, first=0, method=gu.METHODS[0],
                                      gate_column_offset=offset, up_column_offset=256)
    torch.cuda.synchronize()
    assert torch.equal(out, before)


@pytest.mark.parametrize('rows,distribution', [(1, 'reused'), (1, 'diverse'), (24, 'reused'), (24, 'diverse')])
@pytest.mark.parametrize('method', gu.METHODS)
@torch.no_grad()
def test_raw_iq2_preallocated_combined_event_timing_diagnostic(
        prepared, raw_iq2_prepared, raw_iq2_banks, rows, distribution, method, record_property):
    """Matched existing CUDA-event procedure; no speed threshold or P3 claim."""
    import json
    import math
    import time
    gate, up, _, _, _, planes, producer, table = raw_iq2_banks
    x = torch.full((rows, 2560), 2.**-12, dtype=torch.float16, device=gate.device)
    picks = torch.arange(11, dtype=torch.int32, device=gate.device).repeat(rows, 1)
    if distribution == 'diverse':
        picks[:, :10] = (torch.arange(rows, device=gate.device)[:, None]*10 + torch.arange(10, device=gate.device))%256
    picks[:, 10] = 512
    old = torch.empty((rows*11, 160), dtype=torch.float16, device=gate.device)
    new = torch.empty_like(old)
    def baseline():
        producer.pack_active(gate, up, picks, *planes, first=0, ggml_type=22,
                              gate_column_offset=256, up_column_offset=256)
        prepared.gate_up(x, picks, *planes, table, old, first=0, format_name='IQ2_S', method=method)
    def direct():
        raw_iq2_prepared.gate_up_raw_iq2(x, picks, gate, up, table, new, first=0, method=method,
                                        gate_column_offset=256, up_column_offset=256)
    for _ in range(3):
        baseline()
        direct()
    torch.cuda.synchronize()
    assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
    events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for _ in range(4)]
    samples = []
    for (start, stop), (name, function) in zip(events, [('pack+GU', baseline), ('rawGU', direct),
                                                     ('rawGU', direct), ('pack+GU', baseline)]):
        before = time.perf_counter()
        start.record()
        for _ in range(5):
            function()
        stop.record()
        stop.synchronize()
        elapsed = start.elapsed_time(stop)/5
        host = (time.perf_counter()-before)*1000/5
        assert math.isfinite(elapsed) and elapsed > 0 and math.isfinite(host) and host > 0
        samples.append({'path': name, 'stream_envelope_ms_per_call': elapsed, 'inclusive_host_ms_per_call': host})
    assert torch.equal(old.view(torch.uint8), new.view(torch.uint8))
    record_property('raw_iq2_combined_timing', json.dumps({'rows': rows, 'distribution': distribution,
        'method': method, 'order': 'ABBA', 'iterations_per_sample': 5, 'warmup_pairs': 3,
        'samples': samples, 'scope': 'single-device preallocated operator diagnostic, no model/P3 gain claim'}))

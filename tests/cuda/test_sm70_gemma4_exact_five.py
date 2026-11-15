"""Exact-five tiles against unchanged row-count controls and router lifecycle.

The existing Gemma kernel suite supplies independent FP64 numerical oracles;
these checks target the declared row-count invariance and newly selected layouts.
"""
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (7, 0),
    reason="needs an SM70 GPU",
)


def _exact(actual, expected):
    assert torch.equal(torch.isnan(actual), torch.isnan(expected))
    valid = ~torch.isnan(expected)
    assert torch.equal(actual.view(torch.int32)[valid], expected.view(torch.int32)[valid])


def _inputs(width):
    storage = torch.randn(6, width + 8, device="cuda")
    x = storage[:, :width]
    x[0, 0::2], x[0, 1::2] = 0.0, -0.0
    x[1] *= 1e10
    x[2] *= torch.finfo(torch.float32).tiny / 2
    return storage, x


def _check_rows(call, x, outputs, stride):
    reference = torch.empty(6, outputs, device="cuda")
    call(x, 6, reference, outputs)  # unchanged MAXM8 path
    before = x.clone()
    for rows in (4, 5, 6):
        guarded = torch.full((rows + 2, stride), 123.0, device="cuda")
        out = guarded[1:rows + 1, :outputs]
        call(x[:rows], rows, out, stride)
        _exact(out, reference[:rows])
        assert torch.all(guarded[0] == 123) and torch.all(guarded[-1] == 123)
        assert torch.all(guarded[1:rows + 1, outputs:] == 123)
    _exact(x, before)


@pytest.mark.parametrize("outputs,inputs,geglu", [
    (1024, 2816, False), (1536, 2816, False),
    (2816, 512, False), (2816, 1024, False),
    (2816, 272, False), (2816, 256, False),
    (256, 2816, True), (272, 2816, True),
    (1025, 512, False), (17, 256, False), (257, 2816, True),
    # Valid layouts deliberately left on their original MAXM8 dispatch at M5.
    (513, 512, False), (1024, 512, True), (17, 256, True),
])
def test_w4_exact_five_row_count_and_stride(outputs, inputs, geglu):
    from tensorfold.cuda.sm70.gemma4_native import extension

    torch.manual_seed(5001)
    physical = outputs * (2 if geglu else 1)
    codes = torch.randint(0, 256, (physical, inputs // 2), dtype=torch.uint8, device="cuda")
    scales = torch.randint(0, 127, (physical, inputs // 16), dtype=torch.uint8, device="cuda")
    rowmul = torch.rand(physical, device="cuda") * 1e-3 + 1e-4
    storage, x = _inputs(inputs)
    before = (codes.clone(), scales.clone(), rowmul.clone(), storage.clone())
    ext = extension()

    def call(values, rows, out, stride):
        ext.w4_rows(values, rows, codes, scales, rowmul, outputs, inputs, out, stride, geglu)

    _check_rows(call, x, outputs, 2 * outputs)
    assert torch.equal(codes, before[0]) and torch.equal(scales, before[1])
    _exact(rowmul, before[2])
    _exact(storage, before[3])


@pytest.mark.parametrize("outputs,inputs,softcap", [
    (2816, 512, 0.0), (1408, 2816, 0.0), (2816, 704, 0.0),
    (1280, 2816, 0.0), (768, 2816, 0.0), (352, 16896, 0.0),
    (32768, 2816, 30.0),  # Five proposal-head rows are possible with block6.
    (1025, 32, 30.0), (769, 32, 0.0),
])
def test_i8_exact_five_row_count_and_stride(outputs, inputs, softcap):
    from tensorfold.cuda.sm70.gemma4_native import extension

    torch.manual_seed(5002)
    weights = torch.randint(-127, 128, (outputs, inputs), dtype=torch.int8, device="cuda")
    scale = torch.rand(outputs, device="cuda") * 1e-2 + 1e-4
    storage, x = _inputs(inputs)
    before = (weights.clone(), scale.clone(), storage.clone())
    ext = extension()

    def call(values, rows, out, stride):
        ext.i8_rows(values, rows, weights, scale, outputs, inputs, out, stride, softcap)

    _check_rows(call, x, outputs, outputs + 8)
    assert torch.equal(weights, before[0])
    _exact(scale, before[1])
    _exact(storage, before[2])


@pytest.mark.parametrize("outputs,inputs", [(768, 2816), (352, 16896), (769, 32)])
def test_bf16_narrow_exact_five_row_count_and_stride(outputs, inputs):
    from tensorfold.cuda.sm70.gemma4_native import extension

    torch.manual_seed(5003)
    weights = (torch.randn(outputs, inputs, device="cuda") * 0.05).bfloat16()
    storage, x = _inputs(inputs)
    before = (weights.clone(), storage.clone())
    ext = extension()

    def call(values, rows, out, stride):
        ext.bf16_rows(values, rows, weights, outputs, inputs, out, stride, 0.0)

    _check_rows(call, x, outputs, outputs + 8)
    assert torch.equal(weights.view(torch.int16), before[0].view(torch.int16))
    _exact(storage, before[1])


@pytest.mark.parametrize("experts,topk,ties", [(128, 8, False), (128, 8, True),
                                             (127, 3, False), (127, 3, True), (256, 8, False)])
def test_router_exact_five_publication_and_replay(experts, topk, ties):
    from tensorfold.cuda.sm70.gemma4_native import extension

    torch.manual_seed(5004)
    inputs, first, count = 2816, 64, 64
    storage = torch.randn(6, inputs + 8, device="cuda")
    x = storage[:, :inputs]
    weights = (torch.randn(experts, inputs, device="cuda") * 0.05).bfloat16()
    if ties:
        weights.zero_()
    scales = (torch.rand(experts, device="cuda") + 0.5).bfloat16()
    counter = torch.zeros(1, dtype=torch.int32, device="cuda")
    ext = extension()
    before = (storage.clone(), weights.clone(), scales.clone())

    def buffers(rows):
        return (torch.full((rows + 2, experts), 123.0, device="cuda"),
                torch.full((rows + 2, topk), -123, dtype=torch.int32, device="cuda"),
                torch.full((rows + 2, topk), 123.0, device="cuda"),
                torch.full((rows + 2, topk), -123, dtype=torch.int32, device="cuda"))

    def call(rows, guarded):
        logits, ids, probability, local = (value[1:-1] for value in guarded)
        ext.router(x[:rows], rows, weights, experts, inputs, logits, counter, topk,
                   scales, first, count, ids, probability, local)

    reference = buffers(6)
    call(6, reference)
    assert int(counter.item()) == 0

    def check(rows, guarded):
        assert int(counter.item()) == 0
        for actual, expected in zip(guarded, reference):
            if actual.is_floating_point():
                _exact(actual[1:-1], expected[1:rows + 1])
                sentinel = 123
            else:
                assert torch.equal(actual[1:-1], expected[1:rows + 1])
                sentinel = -123
            assert torch.all(actual[0] == sentinel) and torch.all(actual[-1] == sentinel)
        if ties:
            assert torch.equal(guarded[1][1:-1], torch.arange(topk, device="cuda").expand(rows, -1))

    for rows in (4, 5, 6):
        guarded = buffers(rows)
        for _ in range(2):
            call(rows, guarded)
            check(rows, guarded)
    guarded = buffers(5)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        call(5, guarded)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        call(5, guarded)
    torch.cuda.current_stream().wait_stream(stream)
    for _ in range(7):
        graph.replay()
        check(5, guarded)
    _exact(storage, before[0])
    assert torch.equal(weights.view(torch.int16), before[1].view(torch.int16))
    assert torch.equal(scales.view(torch.int16), before[2].view(torch.int16))
